"""Reward computation for the excavation task.

Implements the multi-component reward function from Section 2.3 of the
implementation plan. Each component is computed separately for logging
and ablation analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .excavation_env_cfg import RewardCfg, RewardWeights


@dataclass
class RewardComponents:
    """Individual reward components for logging and analysis.

    Each field stores the scalar reward value for one component.
    The total reward is the sum of all components.
    """

    transfer: float = 0.0       # R1: soil transfer to target
    approach: float = 0.0       # R2: EE proximity to soil
    load: float = 0.0           # R3: bucket load
    transport: float = 0.0      # R4: loaded bucket near target
    smoothness: float = 0.0     # R5: action smoothness penalty
    joint_limit: float = 0.0    # R6: joint limit penalty
    time: float = 0.0           # R7: time penalty
    collision: float = 0.0      # R8: collision penalty
    success_bonus: float = 0.0  # one-time success bonus
    dig: float = 0.0            # R9: dense shaping — EE z below heap top

    @property
    def total(self) -> float:
        return (
            self.transfer
            + self.approach
            + self.load
            + self.transport
            + self.smoothness
            + self.joint_limit
            + self.time
            + self.collision
            + self.success_bonus
            + self.dig
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "reward/_total": self.total,
            "reward/transfer": self.transfer,
            "reward/approach": self.approach,
            "reward/load": self.load,
            "reward/transport": self.transport,
            "reward/dig": self.dig,
            "reward/smooth_penalty": self.smoothness,
            "reward/joint_limit_penalty": self.joint_limit,
            "reward/time_penalty": self.time,
            "reward/collision_penalty": self.collision,
            "reward/success_bonus": self.success_bonus,
        }


@dataclass
class RewardState:
    """State needed for reward computation at each step."""

    # Soil
    soil_in_target_ratio: float = 0.0
    prev_soil_in_target_ratio: float = 0.0
    soil_centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # End-effector
    ee_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Bucket
    bucket_load: float = 0.0  # normalized [0, 1]

    # Target
    target_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Actions
    current_action: np.ndarray = field(default_factory=lambda: np.zeros(7))
    previous_action: np.ndarray = field(default_factory=lambda: np.zeros(7))

    # Joint state
    joint_pos: np.ndarray = field(default_factory=lambda: np.zeros(7))
    joint_limits_lower: np.ndarray = field(default_factory=lambda: np.zeros(7))
    joint_limits_upper: np.ndarray = field(default_factory=lambda: np.zeros(7))

    # Collision
    has_collision: bool = False

    # Success tracking
    already_succeeded: bool = False


def compute_reward(
    state: RewardState,
    cfg: RewardCfg,
    weight_overrides: dict[str, float] | None = None,
) -> RewardComponents:
    """Compute all reward components for a single environment step.

    Args:
        state: Current reward-relevant state.
        cfg: Reward configuration with weights.
        weight_overrides: Optional dict to override specific weights
            (used by curriculum learning).

    Returns:
        RewardComponents with all individual and total rewards.
    """
    w = cfg.weights

    # Apply curriculum overrides if provided
    if weight_overrides:
        # Create a copy of weights with overrides
        w = RewardWeights(
            soil_transfer=weight_overrides.get("soil_transfer", w.soil_transfer),
            approach=weight_overrides.get("approach", w.approach),
            bucket_load=weight_overrides.get("bucket_load", w.bucket_load),
            transport=weight_overrides.get("transport", w.transport),
            action_smoothness=weight_overrides.get("action_smoothness", w.action_smoothness),
            joint_limit=weight_overrides.get("joint_limit", w.joint_limit),
            time_penalty=weight_overrides.get("time_penalty", w.time_penalty),
            collision=weight_overrides.get("collision", w.collision),
            approach_alpha=weight_overrides.get("approach_alpha", w.approach_alpha),
            dig=weight_overrides.get("dig", w.dig),
            dig_alpha=weight_overrides.get("dig_alpha", w.dig_alpha),
        )

    components = RewardComponents()

    # R1: Soil transfer reward — delta of soil mass in target zone
    delta_transfer = state.soil_in_target_ratio - state.prev_soil_in_target_ratio
    components.transfer = w.soil_transfer * delta_transfer

    # R2: Approach reward — exponential decay with distance to soil centroid
    dist_to_soil = np.linalg.norm(state.ee_pos - state.soil_centroid)
    components.approach = w.approach * np.exp(-w.approach_alpha * dist_to_soil)

    # R3: Bucket load reward — normalized bucket load.
    # No "lifted" gate because the FK clamp pins ee_z to 0.005, making any
    # gate at >5cm structurally unreachable while EE is near soil. We rely
    # on the higher collision penalty (-15) to discourage actively penetrating
    # the ground; the agent's natural equilibrium becomes "hover at ground
    # level" (z ≈ 0.005, no collision) rather than "park inside the soil".
    components.load = w.bucket_load * state.bucket_load

    # R4: Transport reward — loaded bucket near target
    dist_to_target = np.linalg.norm(state.ee_pos - state.target_pos)
    components.transport = (
        w.transport * state.bucket_load * np.exp(-w.approach_alpha * dist_to_target)
    )

    # R5: Action smoothness penalty
    action_diff = state.current_action - state.previous_action
    components.smoothness = w.action_smoothness * float(np.sum(action_diff ** 2))

    # R6: Joint limit penalty
    lower_violation = np.maximum(0, state.joint_limits_lower - state.joint_pos)
    upper_violation = np.maximum(0, state.joint_pos - state.joint_limits_upper)
    total_violation = float(np.sum(lower_violation + upper_violation))
    components.joint_limit = w.joint_limit * total_violation

    # R7: Time penalty (constant per step)
    components.time = w.time_penalty

    # R8: Collision penalty
    if state.has_collision:
        components.collision = w.collision

    # R9: Dig depth dense shaping. EE starts at z~0.24, heap top at ~0.20.
    # Without this, approach reward is roughly flat in z near the heap (it
    # only depends on |EE - centroid|, which is small horizontally), so the
    # agent has no strong gradient to descend into the heap. This component
    # gives a clean, monotone dz signal that maxes out at dig_target_z.
    dive_distance = max(0.0, float(state.ee_pos[2]) - cfg.dig_target_z)
    components.dig = w.dig * np.exp(-w.dig_alpha * dive_distance)

    # Success bonus (one-time)
    if (
        state.soil_in_target_ratio >= cfg.success_threshold
        and not state.already_succeeded
    ):
        components.success_bonus = cfg.success_bonus

    return components


def compute_reward_batch(
    states: list[RewardState],
    cfg: RewardCfg,
    weight_overrides: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[RewardComponents]]:
    """Compute rewards for a batch of environments.

    Args:
        states: List of reward states, one per environment.
        cfg: Reward configuration.
        weight_overrides: Optional weight overrides.

    Returns:
        Tuple of (rewards array of shape (N,), list of RewardComponents).
    """
    all_components = []
    rewards = np.zeros(len(states), dtype=np.float32)

    for i, state in enumerate(states):
        comp = compute_reward(state, cfg, weight_overrides)
        all_components.append(comp)
        rewards[i] = comp.total

    return rewards, all_components
