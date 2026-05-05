"""Bucket end-effector configuration for excavation.

The bucket replaces the standard gripper on the robot arm and acts as a
rigid-body collider that interacts with soil particles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class BucketGeometry:
    """Geometric parameters for a simplified excavation bucket.

    The bucket is modeled as an open-top box (5 faces) attached to the
    robot's end-effector flange.

    Coordinate frame (bucket local):
        - x: forward (scoop direction)
        - y: lateral
        - z: up (opening faces +z)
    """

    width: float = 0.15       # m — lateral extent
    depth: float = 0.12       # m — scoop depth (x direction)
    height: float = 0.08      # m — wall height (z direction)
    wall_thickness: float = 0.005  # m

    @property
    def volume(self) -> float:
        """Internal volume in m^3."""
        inner_w = self.width - 2 * self.wall_thickness
        inner_d = self.depth - self.wall_thickness  # open front
        inner_h = self.height - self.wall_thickness
        return inner_w * inner_d * inner_h

    @property
    def opening_area(self) -> float:
        """Top opening area in m^2."""
        return (self.width - 2 * self.wall_thickness) * (self.depth - self.wall_thickness)


@dataclass
class BucketEndEffectorCfg:
    """Configuration for the bucket end-effector.

    The bucket is rigidly attached to the robot's last link (end-effector
    flange) with a fixed offset. In Stage 0, a simple rigid cube is used
    as a placeholder; in Stage 1+, this becomes the actual bucket collider
    for Warp particle interaction.
    """

    # Geometry
    geometry: BucketGeometry = field(default_factory=BucketGeometry)

    # Attachment offset from EE flange frame
    offset_pos: tuple[float, float, float] = (0.0, 0.0, -0.04)
    offset_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # Physics properties
    mass: float = 0.5  # kg
    friction: float = 0.8
    restitution: float = 0.0

    # Collision mesh — either "box" for simplified or path to USD mesh
    collision_shape: str = "box"
    collision_mesh_path: str = ""

    # For particle interaction: bucket interior detection
    # Particles within this bounding region (in bucket frame) count as "loaded"
    load_detection_margin: float = 0.01  # m — margin inside bucket walls

    def get_load_detection_bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Get axis-aligned bounding box for particle load detection in bucket frame.

        Returns:
            (lower_bound, upper_bound) each as (x, y, z) in bucket local frame.
        """
        g = self.geometry
        m = self.load_detection_margin
        lower = (
            -g.depth / 2 + g.wall_thickness + m,
            -g.width / 2 + g.wall_thickness + m,
            -g.height / 2 + g.wall_thickness + m,
        )
        upper = (
            g.depth / 2 - m,
            g.width / 2 - g.wall_thickness - m,
            g.height / 2,  # open top — no upper z constraint
        )
        return lower, upper

    @property
    def max_load_particles(self) -> int:
        """Estimated maximum number of particles that fit in the bucket.

        Assumes particle radius of 0.005 m and random packing fraction ~0.6.
        """
        particle_radius = 0.005
        particle_volume = (4 / 3) * math.pi * particle_radius ** 3
        packing_fraction = 0.6
        return int(self.geometry.volume * packing_fraction / particle_volume)
