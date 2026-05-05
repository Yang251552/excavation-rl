"""Tests for the soil particle system.

Validates soil terrain generation, particle conservation, and
height map computation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soil.soil_terrain import SoilTerrain, HeapConfig, HeapShape
from soil.soil_properties import SoilProperties, SoilType, SOIL_PRESETS, randomize_soil_properties


class TestSoilTerrain:
    """Test soil heap generation."""

    @pytest.fixture
    def terrain(self):
        return SoilTerrain(
            config=HeapConfig(
                shape=HeapShape.CONE,
                center=(0.4, 0.0, 0.0),
                radius=0.2,
                height=0.25,
                num_particles=500,
            ),
            target_center=(-0.4, 0.0, 0.0),
            target_half_extent=(0.15, 0.15, 0.3),
        )

    def test_particle_count(self, terrain):
        """Generated positions should have correct count."""
        positions = terrain.generate_particle_positions()
        assert positions.shape == (500, 3)

    def test_cone_shape_bounds(self, terrain):
        """Cone particles should be within radius and height bounds."""
        positions = terrain.generate_particle_positions()

        # Translate back to local frame
        local = positions - np.array(terrain.config.center)

        # All z >= 0 (above ground)
        assert np.all(local[:, 2] >= 0), "Particles below ground"

        # All z <= height
        assert np.all(local[:, 2] <= terrain.config.height + 1e-6), "Particles above heap"

        # Horizontal distance within radius
        r = np.sqrt(local[:, 0] ** 2 + local[:, 1] ** 2)
        assert np.all(r <= terrain.config.radius + 1e-6), "Particles outside radius"

    @pytest.mark.parametrize("shape", [HeapShape.CONE, HeapShape.HEMISPHERE, HeapShape.CYLINDER, HeapShape.FLAT])
    def test_all_shapes_generate(self, shape):
        """All heap shapes should generate valid positions."""
        config = HeapConfig(shape=shape, num_particles=100)
        terrain = SoilTerrain(config=config)
        positions = terrain.generate_particle_positions()
        assert positions.shape == (100, 3)
        assert not np.any(np.isnan(positions))

    def test_target_zone_detection(self, terrain):
        """Particles at target center should be detected as in-target."""
        # Create particles at target center
        target = np.array(terrain.target_center)
        positions = np.tile(target, (10, 1)).astype(np.float32)

        result = terrain.is_in_target(positions)
        assert np.all(result), "Particles at target center not detected"

    def test_target_zone_outside(self, terrain):
        """Particles far from target should not be in-target."""
        positions = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
        result = terrain.is_in_target(positions)
        assert not np.any(result), "Distant particles incorrectly in target"

    def test_height_map_shape(self, terrain):
        """Height map should have correct shape."""
        positions = terrain.generate_particle_positions()
        h_map = terrain.compute_height_map(positions, grid_resolution=5)
        assert h_map.shape == (5, 5)

    def test_height_map_non_negative(self, terrain):
        """Height map values should be non-negative."""
        positions = terrain.generate_particle_positions()
        h_map = terrain.compute_height_map(positions, grid_resolution=5)
        assert np.all(h_map >= 0)

    def test_reproducibility(self, terrain):
        """Same RNG should produce identical positions."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        pos1 = terrain.generate_particle_positions(rng=rng1)
        pos2 = terrain.generate_particle_positions(rng=rng2)
        np.testing.assert_array_equal(pos1, pos2)


class TestSoilProperties:
    """Test soil material properties."""

    def test_preset_types(self):
        """All predefined soil types should have valid properties."""
        for soil_type in SoilType:
            props = SOIL_PRESETS[soil_type]
            assert props.density > 0
            assert 0 <= props.friction_static <= 1.5
            assert props.particle_mass > 0

    def test_particle_mass_auto_compute(self):
        """Particle mass should be computed from density if not set."""
        props = SoilProperties(density=1600.0, particle_radius=0.005)
        assert props.particle_mass > 0
        expected_volume = (4 / 3) * np.pi * 0.005 ** 3
        expected_mass = 1600.0 * expected_volume
        np.testing.assert_almost_equal(props.particle_mass, expected_mass, decimal=10)

    def test_randomize_produces_different_values(self):
        """Randomization should produce varying parameters."""
        base = SOIL_PRESETS[SoilType.DRY_SAND]
        rng = np.random.default_rng(42)

        results = [randomize_soil_properties(base, rng) for _ in range(10)]
        densities = [r.density for r in results]

        # Should have variation
        assert np.std(densities) > 10, "Randomization not producing variation"

    def test_randomize_within_range(self):
        """Randomized values should be within specified ranges."""
        base = SOIL_PRESETS[SoilType.DRY_SAND]
        rng = np.random.default_rng(42)

        density_range = (1400.0, 2200.0)
        for _ in range(100):
            props = randomize_soil_properties(base, rng, density_range=density_range)
            assert density_range[0] <= props.density <= density_range[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
