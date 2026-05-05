"""Robot arm and end-effector configuration package."""

from .arm_cfg import FrankaArmCfg, UR10ArmCfg
from .end_effector import BucketEndEffectorCfg

__all__ = ["FrankaArmCfg", "UR10ArmCfg", "BucketEndEffectorCfg"]
