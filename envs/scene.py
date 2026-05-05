"""Scene construction for the excavation environment.

Builds the Isaac Lab scene with robot arm, bucket end-effector, soil
particles (or rigid body proxy), ground plane, and target zone markers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .excavation_env_cfg import ExcavationEnvCfg

from robot.arm_cfg import get_arm_cfg, ArmCfg
from robot.end_effector import BucketEndEffectorCfg
from soil.soil_terrain import SoilTerrain, HeapConfig, HeapShape
from soil.soil_properties import SoilProperties, SoilType, SOIL_PRESETS
from soil.particle_system import ParticleSystem, ParticleSystemConfig


class ExcavationScene:
    """Manages the physical scene for excavation.

    In Isaac Lab integration mode, this class wraps the scene creation
    using Isaac Lab's InteractiveScene API. For standalone testing, it
    maintains the scene state as numpy/torch arrays.

    Attributes:
        arm_cfg: Robot arm configuration.
        bucket_cfg: Bucket end-effector configuration.
        particle_system: Warp particle system (None if using rigid body proxy).
        soil_terrain: Soil heap geometry generator.
    """

    def __init__(self, env_cfg: ExcavationEnvCfg):
        self.env_cfg = env_cfg
        self.scene_cfg = env_cfg.scene
        self.device = env_cfg.device

        # Robot
        self.arm_cfg: ArmCfg = get_arm_cfg(self.scene_cfg.arm_type)
        self.bucket_cfg = BucketEndEffectorCfg()

        # Soil terrain
        heap_shape_map = {
            "cone": HeapShape.CONE,
            "hemisphere": HeapShape.HEMISPHERE,
            "cylinder": HeapShape.CYLINDER,
            "flat": HeapShape.FLAT,
        }
        self.soil_terrain = SoilTerrain(
            config=HeapConfig(
                shape=heap_shape_map.get(self.scene_cfg.soil_heap_shape, HeapShape.CONE),
                center=self.scene_cfg.soil_heap_center,
                radius=self.scene_cfg.soil_heap_radius,
                height=self.scene_cfg.soil_heap_height,
                num_particles=self.scene_cfg.soil_num_particles,
            ),
            target_center=self.scene_cfg.target_center,
            target_half_extent=self.scene_cfg.target_half_extent,
        )

        # Soil properties
        self.soil_properties = SOIL_PRESETS[SoilType.DRY_SAND]

        # Particle system (initialized on first reset)
        self.particle_system: ParticleSystem | None = None
        self._use_particles = not self.scene_cfg.use_rigid_body_proxy

        # Rigid body proxy state (Stage 0)
        self.rigid_body_pos: np.ndarray | None = None
        self.rigid_body_vel: np.ndarray | None = None
        self._rigid_body_size = self.scene_cfg.rigid_body_size
        self._rigid_body_mass = 1.0       # kg
        self._rigid_body_friction = 0.5
        self._rigid_body_damping = 5.0    # velocity damping

    def build(self) -> None:
        """Build the scene. Call once at environment creation.

        In a full Isaac Lab integration, this would create USD prims and
        configure the physics scene. Here we initialize data structures.
        """
        if self._use_particles:
            ps_cfg = ParticleSystemConfig(
                num_particles=self.scene_cfg.soil_num_particles,
                particle_radius=self.soil_properties.particle_radius,
                dt=self.env_cfg.sim_dt / 4,  # substep dt
                substeps=4,
                device=self.device,
            )
            self.particle_system = ParticleSystem(ps_cfg, self.soil_properties)

    def reset(self, env_ids: np.ndarray | None = None, rng: np.random.Generator | None = None) -> None:
        """Reset scene to initial state.

        Args:
            env_ids: Indices of environments to reset. None = reset all.
            rng: Random number generator for stochastic resets.
        """
        if rng is None:
            rng = np.random.default_rng(self.env_cfg.seed)

        if self._use_particles:
            initial_positions = self.soil_terrain.generate_particle_positions(rng)
            if self.particle_system is not None:
                self.particle_system.reset(initial_positions)
            # Clear any carried-particle state from a previous episode
            if hasattr(self, "_carried_particle_offsets"):
                self._carried_particle_offsets = {}
        else:
            # Stage 0: rigid body proxy
            self.rigid_body_pos = np.array(self.scene_cfg.soil_heap_center, dtype=np.float32)
            self.rigid_body_vel = np.zeros(3, dtype=np.float32)

    def get_target_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get target zone lower and upper corners in world frame."""
        tc = np.array(self.scene_cfg.target_center)
        th = np.array(self.scene_cfg.target_half_extent)
        return tc - th, tc + th

    def get_soil_in_target_ratio(self) -> float:
        """Compute fraction of soil mass currently in the target zone."""
        lower, upper = self.get_target_bounds()

        if self._use_particles and self.particle_system is not None:
            count_in = self.particle_system.count_particles_in_region(lower, upper)
            return count_in / max(self.particle_system.num_particles, 1)
        elif self.rigid_body_pos is not None:
            # Rigid body proxy: check if cube center is in target
            in_target = np.all(
                (self.rigid_body_pos >= lower) & (self.rigid_body_pos <= upper)
            )
            return 1.0 if in_target else 0.0
        return 0.0

    def get_soil_height_map(self, resolution: int = 5, grid_size: float = 1.0) -> np.ndarray:
        """Get soil height map observation."""
        if self._use_particles and self.particle_system is not None:
            return self.particle_system.get_height_map(
                grid_center=(0.0, 0.0),
                grid_size=grid_size,
                resolution=resolution,
            )
        elif self.rigid_body_pos is not None:
            # Place the rigid body as a single "bump" in the height map
            h_map = np.zeros((resolution, resolution), dtype=np.float32)
            cell_size = grid_size / resolution
            origin = -grid_size / 2
            ix = int((self.rigid_body_pos[0] - origin) / cell_size)
            iy = int((self.rigid_body_pos[1] - origin) / cell_size)
            ix = np.clip(ix, 0, resolution - 1)
            iy = np.clip(iy, 0, resolution - 1)
            h_map[ix, iy] = max(self.rigid_body_pos[2], 0.0) + self._rigid_body_size
            return h_map
        return np.zeros((resolution, resolution), dtype=np.float32)

    def get_soil_centroid(self) -> np.ndarray:
        """Get soil center of mass (3,)."""
        if self._use_particles and self.particle_system is not None:
            return self.particle_system.get_soil_center_of_mass()
        elif self.rigid_body_pos is not None:
            return self.rigid_body_pos.copy()
        return np.zeros(3, dtype=np.float32)

    def get_soil_spread(self) -> np.ndarray:
        """Get soil spatial variance (3,)."""
        if self._use_particles and self.particle_system is not None:
            return self.particle_system.get_soil_spread()
        return np.zeros(3, dtype=np.float32)

    def get_bucket_load(self, bucket_pos: np.ndarray) -> float:
        """Estimate soil mass in bucket."""
        if self._use_particles and self.particle_system is not None:
            lower_off, upper_off = self.bucket_cfg.get_load_detection_bounds()
            bucket_lower = bucket_pos + np.array(lower_off)
            bucket_upper = bucket_pos + np.array(upper_off)
            return self.particle_system.get_bucket_load(bucket_pos, bucket_lower, bucket_upper)
        elif self.rigid_body_pos is not None:
            # Rigid body proxy: check if cube is close to and below the bucket
            diff = self.rigid_body_pos - bucket_pos
            dist = np.linalg.norm(diff)
            contact_radius = self._rigid_body_size + 0.05
            if dist < contact_radius and self.rigid_body_pos[2] < bucket_pos[2]:
                return self._rigid_body_mass * max(0, 1.0 - dist / contact_radius)
        return 0.0

    def step_physics(
        self,
        bucket_pos: np.ndarray | None = None,
        bucket_vel: np.ndarray | None = None,
        dt: float = 1.0 / 60.0,
    ) -> None:
        """Advance physics by one environment step.

        Handles both particle mode (Warp) and rigid body proxy mode
        (simplified contact dynamics).

        Args:
            bucket_pos: (3,) current bucket position in world frame.
            bucket_vel: (3,) current bucket linear velocity.
            dt: Simulation timestep.
        """
        if self._use_particles and self.particle_system is not None:
            if bucket_pos is not None and bucket_vel is not None:
                bucket_half_extent = np.array([
                    self.bucket_cfg.geometry.depth / 2,
                    self.bucket_cfg.geometry.width / 2,
                    self.bucket_cfg.geometry.height / 2,
                ])
                bucket_normal = np.array([0.0, 0.0, -1.0])
                self.particle_system.apply_bucket_interaction(
                    bucket_pos, bucket_vel, bucket_half_extent, bucket_normal
                )
                # Containment heuristic: the AABB bucket geometry has no
                # walls, so the kernel alone cannot hold particles through a
                # lift. Approximate "bucket as cup" by post-step pinning
                # any particle inside the load-detection box to the bucket
                # frame: particle's offset from bucket stays constant if it
                # was inside the box at start of step. This produces the
                # qualitative "scoop carries soil" behaviour without
                # rewriting the contact kernel.
                self._update_carried_particles(bucket_pos, bucket_vel * dt)
            self.particle_system.step()

        elif self.rigid_body_pos is not None and bucket_pos is not None:
            # Simplified rigid body proxy physics:
            # If the bucket is close to the cube, push the cube
            diff = self.rigid_body_pos - bucket_pos
            dist = np.linalg.norm(diff)
            # Contact radius = bucket extent + cube half-size
            contact_radius = 0.12 + self._rigid_body_size / 2

            if dist < contact_radius and dist > 1e-6:
                normal = diff / dist
                penetration = contact_radius - dist

                # Spring contact + velocity transfer
                push_force = normal * penetration * 80.0
                if bucket_vel is not None:
                    push_force += bucket_vel * 3.0

                accel = push_force / self._rigid_body_mass
                self.rigid_body_vel += accel * dt

            # Gravity
            self.rigid_body_vel[2] -= 9.81 * dt

            # Damping
            self.rigid_body_vel *= max(0, 1.0 - self._rigid_body_damping * dt)

            # Integrate
            self.rigid_body_pos += self.rigid_body_vel * dt

            # Ground constraint
            ground_z = self._rigid_body_size / 2
            if self.rigid_body_pos[2] < ground_z:
                self.rigid_body_pos[2] = ground_z
                self.rigid_body_vel[2] = max(0, self.rigid_body_vel[2])
                self.rigid_body_vel[:2] *= 0.95

    def _update_carried_particles(
        self,
        bucket_pos: np.ndarray,
        bucket_disp: np.ndarray,
    ) -> None:
        """Sticky-grab containment: particles that enter the bucket interior
        get tagged and ride with the bucket until explicitly released.

        Auto-release: when bucket xy enters the target zone, all carried
        particles are released (representing the agent arriving over target
        and tipping the bucket). This makes the containment mechanism usable
        by a learned policy that has no explicit "release" action.
        """
        if self.particle_system is None:
            return
        if not hasattr(self, "_carried_particle_offsets"):
            self._carried_particle_offsets: dict[int, np.ndarray] = {}

        # Auto-release: if bucket xy is over target zone, drop everything
        # (release with current bucket velocity -- empirically gave higher
        # transfer than zero-vel release because particles spread out across
        # the target zone instead of piling up at the bucket xy point).
        target_lower, target_upper = self.get_target_bounds()
        if (target_lower[0] <= bucket_pos[0] <= target_upper[0]
                and target_lower[1] <= bucket_pos[1] <= target_upper[1]
                and self._carried_particle_offsets):
            self._carried_particle_offsets = {}
            return

        positions = self.particle_system.get_positions_numpy()
        bg = self.bucket_cfg.geometry
        wall = bg.wall_thickness
        m = self.bucket_cfg.load_detection_margin

        # Detect new captures (currently in box, not yet carried)
        rel = positions - bucket_pos
        in_box = (
            (rel[:, 0] >= -bg.depth / 2 + wall + m) &
            (rel[:, 0] <= bg.depth / 2 - m) &
            (rel[:, 1] >= -bg.width / 2 + wall + m) &
            (rel[:, 1] <= bg.width / 2 - wall - m) &
            (rel[:, 2] >= -bg.height / 2 + wall + m) &
            (rel[:, 2] <= bg.height / 2)
        )
        for idx in np.where(in_box)[0]:
            i = int(idx)
            if i not in self._carried_particle_offsets:
                self._carried_particle_offsets[i] = rel[i].copy()

        if not self._carried_particle_offsets:
            return

        # Pin all carried particles
        new_positions = positions.copy()
        idxs = list(self._carried_particle_offsets.keys())
        for i in idxs:
            new_positions[i] = bucket_pos + self._carried_particle_offsets[i]
        # Don't pin below ground
        floor = self.particle_system.config.ground_height + self.particle_system.config.particle_radius
        new_positions[idxs, 2] = np.maximum(new_positions[idxs, 2], floor)

        import warp as wp
        self.particle_system.positions = wp.array(
            new_positions.astype(np.float32),
            dtype=wp.vec3,
            device=self.particle_system.device,
        )
        # Carried particles get bucket velocity so they don't accelerate down
        velocities = self.particle_system.velocities.numpy().reshape(-1, 3).copy()
        # Use bucket displacement / dt as velocity proxy
        bucket_vel_proxy = bucket_disp / max(self.env_cfg.sim_dt, 1e-6)
        velocities[idxs] = bucket_vel_proxy
        self.particle_system.velocities = wp.array(
            velocities.astype(np.float32),
            dtype=wp.vec3,
            device=self.particle_system.device,
        )

    def release_carried_particles(self) -> int:
        """Release all carried particles (e.g., when bucket tilts/dumps).
        Returns the number released."""
        if not hasattr(self, "_carried_particle_offsets"):
            return 0
        n = len(self._carried_particle_offsets)
        self._carried_particle_offsets = {}
        return n

    # Keep old name as alias for backward compatibility
    def step_particles(
        self,
        bucket_pos: np.ndarray | None = None,
        bucket_vel: np.ndarray | None = None,
    ) -> None:
        self.step_physics(bucket_pos, bucket_vel)
