"""Robot arm configurations for Franka Panda and UR10.

Defines articulation configurations compatible with Isaac Lab's
ArticulationCfg for use in the excavation environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import math


@dataclass
class JointLimits:
    """Joint position limits in radians."""

    lower: list[float]
    upper: list[float]

    def __post_init__(self):
        assert len(self.lower) == len(self.upper)

    @property
    def num_joints(self) -> int:
        return len(self.lower)


@dataclass
class ArmCfg:
    """Base configuration for a robot arm in the excavation task."""

    # Identity
    name: str = "robot_arm"
    usd_path: str = ""

    # Kinematics
    num_joints: int = 7
    ee_frame_name: str = "ee_link"

    # Joint limits (rad)
    joint_limits: JointLimits = field(default_factory=lambda: JointLimits([], []))

    # Default joint positions (rad) — a safe home configuration
    default_joint_pos: list[float] = field(default_factory=list)

    # PD controller gains for joint position control
    stiffness: float = 400.0
    damping: float = 80.0

    # Action scaling: action in [-1, 1] maps to joint delta in [-max_delta, max_delta]
    max_joint_delta: float = 0.1  # rad per step

    # Maximum joint velocity (rad/s) — used for safety clipping
    max_joint_vel: float = 2.0

    # Base pose in world frame
    base_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass
class FrankaArmCfg(ArmCfg):
    """Franka Emika Panda 7-DOF arm configuration.

    Uses Isaac Lab's bundled Franka USD asset. Joint limits and default
    positions follow the official Franka datasheet.
    """

    name: str = "franka_panda"
    usd_path: str = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Franka/franka_instanceable.usd"

    num_joints: int = 7
    ee_frame_name: str = "panda_hand"

    joint_limits: JointLimits = field(
        default_factory=lambda: JointLimits(
            lower=[
                -2.8973,  # panda_joint1
                -1.7628,  # panda_joint2
                -2.8973,  # panda_joint3
                -3.0718,  # panda_joint4
                -2.8973,  # panda_joint5
                -0.0175,  # panda_joint6
                -2.8973,  # panda_joint7
            ],
            upper=[
                2.8973,
                1.7628,
                2.8973,
                -0.0698,
                2.8973,
                3.7525,
                2.8973,
            ],
        )
    )

    default_joint_pos: list[float] = field(
        default_factory=lambda: [
            0.0,       # joint1: base rotation
            0.4,       # joint2: shoulder forward
            0.0,       # joint3
            -1.0,      # joint4: elbow — arm extended forward
            0.0,       # joint5
            1.4,       # joint6: wrist
            0.0,       # joint7
        ]
    )

    stiffness: float = 400.0
    damping: float = 80.0
    max_joint_delta: float = 0.1
    max_joint_vel: float = 2.175


@dataclass
class UR10ArmCfg(ArmCfg):
    """Universal Robots UR10 6-DOF arm configuration.

    UR10 has larger workspace than Franka, suitable for excavation tasks
    requiring greater reach.
    """

    name: str = "ur10"
    usd_path: str = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/UniversalRobots/ur10/ur10_instanceable.usd"

    num_joints: int = 6
    ee_frame_name: str = "ee_link"

    joint_limits: JointLimits = field(
        default_factory=lambda: JointLimits(
            lower=[-2 * math.pi] * 6,
            upper=[2 * math.pi] * 6,
        )
    )

    default_joint_pos: list[float] = field(
        default_factory=lambda: [
            0.0,
            -math.pi / 2,
            math.pi / 2,
            -math.pi / 2,
            -math.pi / 2,
            0.0,
        ]
    )

    stiffness: float = 800.0
    damping: float = 40.0
    max_joint_delta: float = 0.1
    max_joint_vel: float = 3.14


def get_arm_cfg(arm_type: Literal["franka", "ur10"] = "franka") -> ArmCfg:
    """Factory function to get arm configuration by name."""
    configs = {
        "franka": FrankaArmCfg,
        "ur10": UR10ArmCfg,
    }
    if arm_type not in configs:
        raise ValueError(f"Unknown arm type: {arm_type}. Choose from {list(configs.keys())}")
    return configs[arm_type]()
