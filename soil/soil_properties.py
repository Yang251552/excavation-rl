"""Soil material properties and predefined soil types.

Defines physical parameters used by the Warp particle simulation to model
different soil materials (dry sand, wet clay, gravel, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SoilType(Enum):
    """Predefined soil types with typical physical parameters."""

    DRY_SAND = "dry_sand"
    WET_SAND = "wet_sand"
    CLAY = "clay"
    GRAVEL = "gravel"
    LOAM = "loam"


@dataclass
class SoilProperties:
    """Physical properties for soil particle simulation.

    All values in SI units. These map directly to Warp particle material
    parameters and domain randomization ranges.
    """

    # Density
    density: float = 1600.0  # kg/m^3

    # Contact properties
    friction_static: float = 0.6
    friction_dynamic: float = 0.4
    restitution: float = 0.05  # near-zero for soil

    # Cohesion (inter-particle attractive force)
    cohesion: float = 100.0  # Pa

    # Particle geometry
    particle_radius: float = 0.005  # m
    particle_mass: Optional[float] = None  # computed from density if None

    # Damping
    contact_damping: float = 1000.0  # N·s/m (currently unused; kept for future
                                     # contact-model upgrade)
    viscous_damping: float = 0.1

    # Force magnitude applied per particle when the bucket sweeps over it,
    # along the bucket's push direction. With ~6 mg particles and gravity
    # contributing ~0.06 N per particle, ~1 N is the regime where particles
    # are carried with a moving bucket without ballistic ejection (verified
    # via standalone bucket-particle sweep, push_strength ablation 0.01-1000 N).
    bucket_push_force: float = 1.0  # N

    # Internal friction angle (Mohr-Coulomb)
    friction_angle_deg: float = 30.0  # degrees

    def __post_init__(self):
        if self.particle_mass is None:
            import math
            volume = (4 / 3) * math.pi * self.particle_radius ** 3
            self.particle_mass = self.density * volume

    @property
    def friction_angle_rad(self) -> float:
        import math
        return math.radians(self.friction_angle_deg)


# Predefined soil property sets
SOIL_PRESETS: dict[SoilType, SoilProperties] = {
    SoilType.DRY_SAND: SoilProperties(
        density=1500.0,
        friction_static=0.5,
        friction_dynamic=0.35,
        restitution=0.05,
        cohesion=0.0,
        friction_angle_deg=33.0,
    ),
    SoilType.WET_SAND: SoilProperties(
        density=1900.0,
        friction_static=0.7,
        friction_dynamic=0.5,
        restitution=0.02,
        cohesion=200.0,
        friction_angle_deg=35.0,
    ),
    SoilType.CLAY: SoilProperties(
        density=2000.0,
        friction_static=0.8,
        friction_dynamic=0.6,
        restitution=0.01,
        cohesion=500.0,
        friction_angle_deg=20.0,
    ),
    SoilType.GRAVEL: SoilProperties(
        density=1800.0,
        friction_static=0.6,
        friction_dynamic=0.4,
        restitution=0.15,
        cohesion=0.0,
        particle_radius=0.008,
        friction_angle_deg=40.0,
    ),
    SoilType.LOAM: SoilProperties(
        density=1400.0,
        friction_static=0.65,
        friction_dynamic=0.45,
        restitution=0.03,
        cohesion=150.0,
        friction_angle_deg=28.0,
    ),
}


def get_soil_properties(soil_type: SoilType) -> SoilProperties:
    """Get predefined soil properties by type."""
    return SOIL_PRESETS[soil_type]


def randomize_soil_properties(
    base: SoilProperties,
    rng,
    density_range: tuple[float, float] = (1400.0, 2200.0),
    friction_range: tuple[float, float] = (0.3, 0.9),
    restitution_range: tuple[float, float] = (0.0, 0.2),
    cohesion_range: tuple[float, float] = (0.0, 500.0),
) -> SoilProperties:
    """Create a randomized copy of soil properties for domain randomization.

    Args:
        base: Base soil properties to randomize around.
        rng: NumPy random generator instance.
        *_range: (min, max) uniform sampling ranges for each parameter.

    Returns:
        New SoilProperties with randomized values.
    """
    return SoilProperties(
        density=rng.uniform(*density_range),
        friction_static=rng.uniform(*friction_range),
        friction_dynamic=rng.uniform(friction_range[0] * 0.7, friction_range[1] * 0.7),
        restitution=rng.uniform(*restitution_range),
        cohesion=rng.uniform(*cohesion_range),
        particle_radius=base.particle_radius,
        contact_damping=base.contact_damping,
        viscous_damping=base.viscous_damping,
        friction_angle_deg=base.friction_angle_deg,
    )
