"""Soil terrain generation — initial particle heap shapes.

Generates initial particle positions for different soil heap configurations.
All positions are in world frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class HeapShape(Enum):
    """Available soil heap initial shapes."""

    CONE = "cone"
    HEMISPHERE = "hemisphere"
    CYLINDER = "cylinder"
    FLAT = "flat"


@dataclass
class HeapConfig:
    """Configuration for a soil heap."""

    shape: HeapShape = HeapShape.CONE
    center: tuple[float, float, float] = (0.4, 0.0, 0.0)  # world frame
    radius: float = 0.2  # m
    height: float = 0.25  # m
    num_particles: int = 500


@dataclass
class SoilTerrain:
    """Generates initial particle positions for the soil heap.

    Supports multiple heap shapes and provides utilities for position
    queries used by the observation and reward systems.
    """

    config: HeapConfig = field(default_factory=HeapConfig)

    # Target zone specification
    target_center: tuple[float, float, float] = (-0.4, 0.0, 0.0)
    target_half_extent: tuple[float, float, float] = (0.15, 0.15, 0.3)

    def generate_particle_positions(
        self,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Generate initial particle positions based on heap configuration.

        Args:
            rng: NumPy random generator. Created if None.

        Returns:
            Array of shape (N, 3) with particle positions in world frame.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        cfg = self.config
        n = cfg.num_particles
        cx, cy, cz = cfg.center

        if cfg.shape == HeapShape.CONE:
            positions = self._generate_cone(n, cfg.radius, cfg.height, rng)
        elif cfg.shape == HeapShape.HEMISPHERE:
            positions = self._generate_hemisphere(n, cfg.radius, rng)
        elif cfg.shape == HeapShape.CYLINDER:
            positions = self._generate_cylinder(n, cfg.radius, cfg.height, rng)
        elif cfg.shape == HeapShape.FLAT:
            positions = self._generate_flat(n, cfg.radius, rng)
        else:
            raise ValueError(f"Unknown heap shape: {cfg.shape}")

        # Translate to world position
        positions[:, 0] += cx
        positions[:, 1] += cy
        positions[:, 2] += cz

        return positions.astype(np.float32)

    def _generate_cone(
        self, n: int, radius: float, height: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate particles in a cone shape using rejection sampling."""
        positions = []
        while len(positions) < n:
            batch = max(n * 3, 1000)
            r = rng.uniform(0, radius, batch)
            theta = rng.uniform(0, 2 * np.pi, batch)
            z = rng.uniform(0, height, batch)

            # Cone constraint: at height z, max radius is radius * (1 - z/height)
            max_r_at_z = radius * (1.0 - z / height)
            mask = r <= max_r_at_z

            x = r[mask] * np.cos(theta[mask])
            y = r[mask] * np.sin(theta[mask])
            pts = np.stack([x, y, z[mask]], axis=-1)
            positions.append(pts)

        positions = np.concatenate(positions, axis=0)[:n]
        return positions

    def _generate_hemisphere(
        self, n: int, radius: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate particles in a hemisphere (upper half of sphere)."""
        positions = []
        while len(positions) < n:
            batch = max(n * 3, 1000)
            # Uniform in sphere via rejection
            pts = rng.uniform(-radius, radius, (batch, 3))
            dist = np.linalg.norm(pts, axis=-1)
            mask = (dist <= radius) & (pts[:, 2] >= 0)
            positions.append(pts[mask])

        positions = np.concatenate(positions, axis=0)[:n]
        return positions

    def _generate_cylinder(
        self, n: int, radius: float, height: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate particles in a cylinder."""
        r = np.sqrt(rng.uniform(0, 1, n)) * radius
        theta = rng.uniform(0, 2 * np.pi, n)
        z = rng.uniform(0, height, n)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack([x, y, z], axis=-1)

    def _generate_flat(
        self, n: int, radius: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate particles in a flat disk (1 layer)."""
        particle_radius = 0.005
        r = np.sqrt(rng.uniform(0, 1, n)) * radius
        theta = rng.uniform(0, 2 * np.pi, n)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.full(n, particle_radius)  # one particle radius above ground
        return np.stack([x, y, z], axis=-1)

    def is_in_target(self, positions: np.ndarray) -> np.ndarray:
        """Check which particles are inside the target zone.

        Args:
            positions: (N, 3) array of particle positions.

        Returns:
            Boolean array of shape (N,).
        """
        tc = np.array(self.target_center)
        th = np.array(self.target_half_extent)
        lower = tc - th
        upper = tc + th
        inside = np.all((positions >= lower) & (positions <= upper), axis=-1)
        return inside

    def compute_height_map(
        self,
        positions: np.ndarray,
        grid_center: tuple[float, float] = (0.0, 0.0),
        grid_size: float = 1.0,
        grid_resolution: int = 5,
    ) -> np.ndarray:
        """Compute a 2D height map from particle positions.

        Divides the horizontal plane into a grid and records the maximum
        particle height in each cell.

        Args:
            positions: (N, 3) particle positions.
            grid_center: (x, y) center of the height map grid.
            grid_size: Total side length of the grid.
            grid_resolution: Number of cells per side (H = W).

        Returns:
            Height map array of shape (grid_resolution, grid_resolution).
        """
        h_map = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
        cell_size = grid_size / grid_resolution
        origin_x = grid_center[0] - grid_size / 2
        origin_y = grid_center[1] - grid_size / 2

        if len(positions) == 0:
            return h_map

        # Compute cell indices for each particle
        ix = ((positions[:, 0] - origin_x) / cell_size).astype(int)
        iy = ((positions[:, 1] - origin_y) / cell_size).astype(int)

        # Clip to valid range
        ix = np.clip(ix, 0, grid_resolution - 1)
        iy = np.clip(iy, 0, grid_resolution - 1)

        # Record max height per cell
        for i in range(len(positions)):
            h_map[ix[i], iy[i]] = max(h_map[ix[i], iy[i]], positions[i, 2])

        return h_map
