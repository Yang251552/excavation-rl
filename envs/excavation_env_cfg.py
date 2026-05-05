"""Environment configuration for the excavation RL task.

Defines all configurable parameters for the excavation environment using
Python dataclasses. This serves as the single source of truth for
environment, scene, reward, and curriculum settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------

@dataclass
class SceneCfg:
    """Scene layout configuration."""

    num_envs: int = 4096
    env_spacing: float = 2.5  # m — spacing between parallel envs

    # Ground plane
    ground_plane: bool = True

    # Robot
    arm_type: Literal["franka", "ur10"] = "franka"
    robot_base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Soil heap — placed within Franka reach (~0.5-0.7 m forward)
    soil_num_particles: int = 500
    soil_heap_center: tuple[float, float, float] = (0.5, 0.0, 0.05)
    soil_heap_radius: float = 0.12
    soil_heap_height: float = 0.15
    soil_heap_shape: str = "cone"

    # Target zone (axis-aligned box) — also within reach, offset laterally
    target_center: tuple[float, float, float] = (0.4, 0.3, 0.05)
    target_half_extent: tuple[float, float, float] = (0.12, 0.12, 0.2)

    # Stage 0: rigid body mode (use a cube instead of particles)
    use_rigid_body_proxy: bool = False
    rigid_body_size: float = 0.05  # m — side length of cube


# ---------------------------------------------------------------------------
# Observation configuration
# ---------------------------------------------------------------------------

@dataclass
class ObservationCfg:
    """Observation space configuration.

    Total dimension depends on the soil representation chosen.
    """

    # Proprioception (always included)
    include_joint_pos: bool = True        # (num_joints,)
    include_joint_vel: bool = True        # (num_joints,)
    include_ee_pos: bool = True           # (3,)
    include_ee_orient: bool = True        # (4,) quaternion
    include_ee_lin_vel: bool = True       # (3,)
    include_ee_ang_vel: bool = True       # (3,)

    # Bucket state
    include_bucket_load: bool = True      # (1,)
    include_bucket_contact_force: bool = True  # (3,)

    # Soil representation
    soil_obs_mode: Literal["height_map", "centroid", "none"] = "height_map"
    height_map_resolution: int = 5        # 5x5 = 25D
    height_map_size: float = 1.0          # m

    # Target
    include_target_pos: bool = True       # (3,)

    # Previous action
    include_previous_action: bool = True  # (num_joints,)

    # Noise
    observation_noise_std: float = 0.0    # 0 = no noise

    # Normalization
    normalize_obs: bool = True
    clip_obs: float = 10.0


# ---------------------------------------------------------------------------
# Action configuration
# ---------------------------------------------------------------------------

@dataclass
class ActionCfg:
    """Action space configuration."""

    mode: Literal["joint_position", "joint_velocity", "ee_position"] = "joint_position"

    # For joint_position mode: action in [-1,1] * scale → joint delta
    action_scale: float = 0.1  # rad

    # Action rate limiting
    max_action_rate: float = 1.0  # max change per step in normalized space

    # Action noise (domain randomization)
    action_noise_std: float = 0.0

    # Action delay (domain randomization)
    action_delay_steps: int = 0


# ---------------------------------------------------------------------------
# Reward configuration
# ---------------------------------------------------------------------------

@dataclass
class RewardWeights:
    """Reward component weights.

    Positive = reward, negative = penalty.
    """

    # Core rewards
    soil_transfer: float = 10.0       # w1: delta soil mass in target zone
    approach: float = 1.0             # w2: EE proximity to soil centroid
    bucket_load: float = 2.0          # w3: soil mass in bucket
    transport: float = 3.0            # w4: loaded bucket near target

    # Penalties
    action_smoothness: float = -0.01  # w5: ||a_t - a_{t-1}||^2 (was -0.05; lowered so it doesn't dominate stage-0 exploration)
    joint_limit: float = -1.0         # w6: joint limit violation
    time_penalty: float = -0.01       # w7: per-step penalty
    collision: float = -15.0          # w8: ground/self collision (was -5; raised
                                      # to make "park bucket inside the ground"
                                      # hack from prior 4500-iter run net-negative)

    # Approach reward decay.
    # Lowered 5.0 → 2.0: at alpha=5, exp(-5*dist) drops to 0.007 at 1m, so
    # an agent more than ~30cm from the soil sees essentially zero approach
    # gradient. alpha=2.0 keeps a meaningful gradient out to ~1.5m.
    approach_alpha: float = 2.0

    # Dense dig-depth shaping (added stage 1 v13). Encourages EE to descend
    # below heap top so bucket can interact with particles. Without this, the
    # agent learns "hover above heap centroid" because approach is xy-dominant.
    dig: float = 1.0
    dig_alpha: float = 8.0    # exp(-8 * dz): 12cm above target -> reward ~0.38


@dataclass
class RewardCfg:
    """Full reward configuration."""

    weights: RewardWeights = field(default_factory=RewardWeights)

    # Success bonus (applied once when task is complete).
    # Lowered from 50 → 10: at 50, the rare success event produced advantage
    # outliers ~50× the typical step reward, triggering catastrophic PPO
    # updates (run "stage0_seed42_smooth01_noise05" collapsed at iter 520).
    success_bonus: float = 10.0
    # Lowered 0.8 → 0.3: at 0.8 the agent needed to transfer 80% of soil to
    # ever see the success bonus, which never happened in 1500 iter. Starting
    # easier so the bonus actually signals; can ramp up in stage 1/2.
    success_threshold: float = 0.3

    # Target z below which the dig shaping reward (R9) saturates. Heap top is
    # at heap_center_z + heap_height = 0.05 + 0.15 = 0.20m by default; aiming
    # 0.07 puts the bucket roughly at mid-heap depth.
    dig_target_z: float = 0.07


# ---------------------------------------------------------------------------
# Termination configuration
# ---------------------------------------------------------------------------

@dataclass
class TerminationCfg:
    """Episode termination conditions."""

    max_episode_length: int = 500  # steps

    # Success
    success_soil_ratio: float = 0.8  # 80% soil in target

    # Failure
    check_joint_limits: bool = True
    check_bucket_ground_penetration: bool = True
    ground_penetration_threshold: float = -0.02  # m below ground
    check_base_displacement: bool = True
    base_displacement_threshold: float = 0.01  # m


# ---------------------------------------------------------------------------
# Domain randomization configuration
# ---------------------------------------------------------------------------

@dataclass
class DomainRandomizationCfg:
    """Domain randomization parameters."""

    enabled: bool = False

    # Soil physics
    soil_density_range: tuple[float, float] = (1400.0, 2200.0)
    soil_friction_range: tuple[float, float] = (0.3, 0.9)
    soil_restitution_range: tuple[float, float] = (0.0, 0.2)
    soil_cohesion_range: tuple[float, float] = (0.0, 500.0)

    # Soil geometry
    heap_height_range: tuple[float, float] = (0.15, 0.35)
    heap_radius_range: tuple[float, float] = (0.2, 0.4)
    heap_position_offset_range: tuple[float, float] = (-0.05, 0.05)

    # Robot
    joint_friction_scale_range: tuple[float, float] = (0.8, 1.2)
    payload_mass_range: tuple[float, float] = (0.0, 0.5)
    action_delay_range: tuple[int, int] = (0, 2)
    action_noise_std_range: tuple[float, float] = (0.0, 0.02)

    # Sensors
    observation_noise_std: float = 0.01


# ---------------------------------------------------------------------------
# Curriculum configuration
# ---------------------------------------------------------------------------

@dataclass
class CurriculumStageCfg:
    """Configuration for a single curriculum stage."""

    name: str = ""
    entry_step: int = 0              # training step to enter this stage

    # Particle count
    num_particles: int = 200

    # Heap placement (distance from robot)
    max_heap_distance: float = 0.3   # m

    # Domain randomization level
    dr_enabled: bool = False

    # Reward weight overrides (None = use default)
    approach_weight: float | None = None
    transfer_weight: float | None = None


@dataclass
class CurriculumCfg:
    """Multi-stage curriculum configuration."""

    enabled: bool = True

    stages: list[CurriculumStageCfg] = field(
        default_factory=lambda: [
            CurriculumStageCfg(
                name="stage_1_easy",
                entry_step=0,
                num_particles=200,
                max_heap_distance=0.3,
                dr_enabled=False,
                approach_weight=2.0,
                transfer_weight=10.0,
            ),
            CurriculumStageCfg(
                name="stage_2_medium",
                entry_step=200_000,
                num_particles=500,
                max_heap_distance=0.5,
                dr_enabled=True,
                approach_weight=0.5,
                transfer_weight=10.0,
            ),
            CurriculumStageCfg(
                name="stage_3_hard",
                entry_step=600_000,
                num_particles=1000,
                max_heap_distance=0.8,
                dr_enabled=True,
                approach_weight=0.1,
                transfer_weight=15.0,
            ),
        ]
    )

    def get_stage(self, total_steps: int) -> CurriculumStageCfg:
        """Get the active curriculum stage based on total training steps."""
        active = self.stages[0]
        for stage in self.stages:
            if total_steps >= stage.entry_step:
                active = stage
        return active


# ---------------------------------------------------------------------------
# Top-level environment configuration
# ---------------------------------------------------------------------------

@dataclass
class ExcavationEnvCfg:
    """Complete environment configuration.

    This is the single top-level config that aggregates all sub-configs.
    Instantiate and modify this to control all aspects of the environment.
    """

    seed: int = 42
    sim_dt: float = 1.0 / 60.0          # simulation timestep
    decimation: int = 2                   # env steps per sim step
    device: str = "cuda:0"

    scene: SceneCfg = field(default_factory=SceneCfg)
    observation: ObservationCfg = field(default_factory=ObservationCfg)
    action: ActionCfg = field(default_factory=ActionCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    termination: TerminationCfg = field(default_factory=TerminationCfg)
    domain_randomization: DomainRandomizationCfg = field(default_factory=DomainRandomizationCfg)
    curriculum: CurriculumCfg = field(default_factory=CurriculumCfg)
