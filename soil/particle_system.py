"""Warp-based particle system for soil simulation.

Wraps NVIDIA Warp to provide a GPU-accelerated particle simulation that
integrates with Isaac Lab's RL environment loop. Handles particle creation,
physics stepping, and state queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

try:
    import warp as wp
except ImportError:
    wp = None  # Allow import without Warp for testing/documentation

from .soil_properties import SoilProperties
from .soil_terrain import SoilTerrain, HeapConfig


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------

if wp is not None:

    @wp.kernel
    def _integrate_particles(
        positions: wp.array(dtype=wp.vec3),
        velocities: wp.array(dtype=wp.vec3),
        forces: wp.array(dtype=wp.vec3),
        inv_mass: float,
        damping: float,
        gravity: wp.vec3,
        dt: float,
        ground_height: float,
        restitution: float,
        friction: float,
    ):
        """Semi-implicit Euler integration with ground collision."""
        tid = wp.tid()

        pos = positions[tid]
        vel = velocities[tid]
        f = forces[tid]

        # Apply forces: gravity + external + damping
        accel = gravity + f * inv_mass - vel * damping
        vel = vel + accel * dt
        pos = pos + vel * dt

        # Ground collision (simple plane at z = ground_height + radius)
        if pos[2] < ground_height:
            pos = wp.vec3(pos[0], pos[1], ground_height)
            # Reflect and damp vertical velocity
            if vel[2] < 0.0:
                vel = wp.vec3(
                    vel[0] * (1.0 - friction),
                    vel[1] * (1.0 - friction),
                    -vel[2] * restitution,
                )

        positions[tid] = pos
        velocities[tid] = vel
        # Reset forces for next step
        forces[tid] = wp.vec3(0.0, 0.0, 0.0)

    @wp.kernel
    def _apply_bucket_force(
        positions: wp.array(dtype=wp.vec3),
        velocities: wp.array(dtype=wp.vec3),
        forces: wp.array(dtype=wp.vec3),
        bucket_pos: wp.vec3,
        bucket_vel: wp.vec3,
        bucket_half_extent: wp.vec3,
        bucket_normal: wp.vec3,
        push_strength: float,
        friction_coeff: float,
    ):
        """Apply forces from bucket collision on particles."""
        tid = wp.tid()

        pos = positions[tid]
        diff = pos - bucket_pos

        # Check if particle is within bucket AABB influence zone
        in_x = wp.abs(diff[0]) < bucket_half_extent[0]
        in_y = wp.abs(diff[1]) < bucket_half_extent[1]
        in_z = wp.abs(diff[2]) < bucket_half_extent[2]

        if in_x and in_y and in_z:
            # Push particle along bucket normal (simplified contact model)
            push_force = bucket_normal * push_strength

            # Friction: transfer some bucket velocity to particle
            friction_force = (bucket_vel - velocities[tid]) * friction_coeff

            forces[tid] = forces[tid] + push_force + friction_force

    @wp.kernel
    def _count_particles_in_region(
        positions: wp.array(dtype=wp.vec3),
        region_lower: wp.vec3,
        region_upper: wp.vec3,
        count: wp.array(dtype=wp.int32),
    ):
        """Count particles within an axis-aligned bounding box."""
        tid = wp.tid()
        pos = positions[tid]

        if (
            pos[0] >= region_lower[0]
            and pos[0] <= region_upper[0]
            and pos[1] >= region_lower[1]
            and pos[1] <= region_upper[1]
            and pos[2] >= region_lower[2]
            and pos[2] <= region_upper[2]
        ):
            wp.atomic_add(count, 0, 1)


@dataclass
class ParticleSystemConfig:
    """Configuration for the Warp particle system."""

    num_particles: int = 500
    particle_radius: float = 0.005  # m
    dt: float = 1.0 / 120.0  # physics substep dt
    substeps: int = 4  # substeps per env step
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    ground_height: float = 0.0  # z coordinate of ground plane
    device: str = "cuda:0"


class ParticleSystem:
    """GPU-accelerated particle system using NVIDIA Warp.

    Manages particle state (positions, velocities, forces) and provides
    methods for stepping physics, applying bucket interactions, and
    querying particle distributions.

    Usage:
        ps = ParticleSystem(config, soil_props)
        ps.initialize(initial_positions)  # (N, 3) numpy array
        for step in range(num_steps):
            ps.apply_bucket_interaction(bucket_state)
            ps.step()
            height_map = ps.get_height_map(...)
    """

    def __init__(
        self,
        config: ParticleSystemConfig,
        soil_properties: SoilProperties,
    ):
        self.config = config
        self.soil = soil_properties
        self._initialized = False

        if wp is None:
            raise RuntimeError(
                "NVIDIA Warp is required but not installed. "
                "Install via: pip install warp-lang"
            )

        wp.init()
        self.device = config.device

    def initialize(self, positions: np.ndarray) -> None:
        """Initialize particle state from numpy positions.

        Args:
            positions: (N, 3) array of initial particle positions.
        """
        n = positions.shape[0]
        assert positions.shape == (n, 3), f"Expected (N, 3), got {positions.shape}"

        self.num_particles = n

        # Convert to Warp arrays on GPU
        self.positions = wp.array(
            positions.astype(np.float32), dtype=wp.vec3, device=self.device
        )
        self.velocities = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.forces = wp.zeros(n, dtype=wp.vec3, device=self.device)

        self._initialized = True

    def step(self) -> None:
        """Advance particle simulation by one environment step (multiple substeps)."""
        assert self._initialized, "Call initialize() before step()"

        cfg = self.config
        inv_mass = 1.0 / self.soil.particle_mass
        gravity = wp.vec3(*cfg.gravity)

        for _ in range(cfg.substeps):
            wp.launch(
                _integrate_particles,
                dim=self.num_particles,
                inputs=[
                    self.positions,
                    self.velocities,
                    self.forces,
                    inv_mass,
                    self.soil.viscous_damping,
                    gravity,
                    cfg.dt,
                    cfg.ground_height + self.soil.particle_radius,
                    self.soil.restitution,
                    self.soil.friction_dynamic,
                ],
                device=self.device,
            )

    def apply_bucket_interaction(
        self,
        bucket_pos: np.ndarray,
        bucket_vel: np.ndarray,
        bucket_half_extent: np.ndarray,
        bucket_normal: np.ndarray,
    ) -> None:
        """Apply bucket-particle interaction forces.

        Args:
            bucket_pos: (3,) bucket center position in world frame.
            bucket_vel: (3,) bucket linear velocity.
            bucket_half_extent: (3,) half-extents of bucket AABB.
            bucket_normal: (3,) bucket face normal (push direction).
        """
        assert self._initialized

        wp.launch(
            _apply_bucket_force,
            dim=self.num_particles,
            inputs=[
                self.positions,
                self.velocities,
                self.forces,
                wp.vec3(*bucket_pos.tolist()),
                wp.vec3(*bucket_vel.tolist()),
                wp.vec3(*bucket_half_extent.tolist()),
                wp.vec3(*bucket_normal.tolist()),
                self.soil.bucket_push_force,
                self.soil.friction_static,
            ],
            device=self.device,
        )

    def get_positions_numpy(self) -> np.ndarray:
        """Get current particle positions as numpy array (N, 3)."""
        return self.positions.numpy().reshape(-1, 3)

    def get_positions_torch(self) -> torch.Tensor:
        """Get current particle positions as a PyTorch tensor (N, 3)."""
        return wp.to_torch(self.positions).reshape(-1, 3)

    def count_particles_in_region(
        self, lower: np.ndarray, upper: np.ndarray
    ) -> int:
        """Count particles within an axis-aligned bounding box.

        Args:
            lower: (3,) lower corner of the region.
            upper: (3,) upper corner of the region.

        Returns:
            Number of particles inside the region.
        """
        count = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.launch(
            _count_particles_in_region,
            dim=self.num_particles,
            inputs=[
                self.positions,
                wp.vec3(*lower.tolist()),
                wp.vec3(*upper.tolist()),
                count,
            ],
            device=self.device,
        )
        return int(count.numpy()[0])

    def get_soil_center_of_mass(self) -> np.ndarray:
        """Compute center of mass of all particles. Returns (3,)."""
        positions = self.get_positions_numpy()
        return positions.mean(axis=0)

    def get_soil_spread(self) -> np.ndarray:
        """Compute spatial variance of particle distribution. Returns (3,)."""
        positions = self.get_positions_numpy()
        return positions.var(axis=0)

    def get_height_map(
        self,
        grid_center: tuple[float, float] = (0.0, 0.0),
        grid_size: float = 1.0,
        resolution: int = 5,
    ) -> np.ndarray:
        """Compute 2D height map from particle positions.

        Args:
            grid_center: (x, y) center of the grid in world frame.
            grid_size: Side length of the square grid.
            resolution: Number of cells per side.

        Returns:
            Height map of shape (resolution, resolution).
        """
        positions = self.get_positions_numpy()
        terrain = SoilTerrain()
        return terrain.compute_height_map(positions, grid_center, grid_size, resolution)

    def get_bucket_load(
        self,
        bucket_pos: np.ndarray,
        bucket_lower: np.ndarray,
        bucket_upper: np.ndarray,
    ) -> float:
        """Estimate soil mass currently in the bucket.

        Args:
            bucket_pos: (3,) bucket center in world frame.
            bucket_lower: (3,) lower bound of bucket interior in world frame.
            bucket_upper: (3,) upper bound of bucket interior in world frame.

        Returns:
            Estimated mass in kg.
        """
        count = self.count_particles_in_region(bucket_lower, bucket_upper)
        return count * self.soil.particle_mass

    def reset(self, positions: np.ndarray) -> None:
        """Reset particle state with new positions."""
        self.initialize(positions)

    @property
    def total_mass(self) -> float:
        """Total mass of all particles in kg."""
        return self.num_particles * self.soil.particle_mass
