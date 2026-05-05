"""Tests for the reward function.

Validates reward computation correctness, component isolation,
and response to known state transitions.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.rewards import RewardState, RewardComponents, compute_reward
from envs.excavation_env_cfg import RewardCfg, RewardWeights


@pytest.fixture
def default_cfg():
    return RewardCfg()


@pytest.fixture
def base_state():
    """Baseline reward state with neutral values."""
    return RewardState(
        soil_in_target_ratio=0.0,
        prev_soil_in_target_ratio=0.0,
        soil_centroid=np.array([0.4, 0.0, 0.0]),
        ee_pos=np.array([0.0, 0.0, 0.3]),
        bucket_load=0.0,
        target_pos=np.array([-0.4, 0.0, 0.0]),
        current_action=np.zeros(7),
        previous_action=np.zeros(7),
        joint_pos=np.zeros(7),
        joint_limits_lower=np.full(7, -3.0),
        joint_limits_upper=np.full(7, 3.0),
        has_collision=False,
        already_succeeded=False,
    )


class TestRewardComponents:
    """Test individual reward components."""

    def test_transfer_reward_positive_on_progress(self, default_cfg, base_state):
        """Transfer reward should be positive when soil moves to target."""
        base_state.soil_in_target_ratio = 0.1
        base_state.prev_soil_in_target_ratio = 0.0

        components = compute_reward(base_state, default_cfg)
        assert components.transfer > 0, "Transfer reward should be positive for progress"

    def test_transfer_reward_zero_no_change(self, default_cfg, base_state):
        """Transfer reward should be zero with no soil movement."""
        base_state.soil_in_target_ratio = 0.5
        base_state.prev_soil_in_target_ratio = 0.5

        components = compute_reward(base_state, default_cfg)
        assert components.transfer == 0.0

    def test_approach_reward_increases_when_closer(self, default_cfg, base_state):
        """Approach reward should be higher when EE is closer to soil."""
        base_state.ee_pos = np.array([0.4, 0.0, 0.0])  # at soil
        close_reward = compute_reward(base_state, default_cfg).approach

        base_state.ee_pos = np.array([0.0, 0.0, 0.3])  # far from soil
        far_reward = compute_reward(base_state, default_cfg).approach

        assert close_reward > far_reward, "Approach reward should increase with proximity"

    def test_load_reward_proportional_to_load(self, default_cfg, base_state):
        """Bucket load reward should scale with load amount."""
        base_state.bucket_load = 0.0
        empty_reward = compute_reward(base_state, default_cfg).load

        base_state.bucket_load = 1.0
        full_reward = compute_reward(base_state, default_cfg).load

        assert full_reward > empty_reward

    def test_transport_reward_requires_load(self, default_cfg, base_state):
        """Transport reward should be zero with empty bucket."""
        base_state.bucket_load = 0.0
        base_state.ee_pos = np.array([-0.4, 0.0, 0.0])  # at target

        components = compute_reward(base_state, default_cfg)
        assert components.transport == 0.0

    def test_transport_reward_with_loaded_bucket(self, default_cfg, base_state):
        """Transport reward should be positive with loaded bucket near target."""
        base_state.bucket_load = 1.0
        base_state.ee_pos = np.array([-0.4, 0.0, 0.0])  # at target

        components = compute_reward(base_state, default_cfg)
        assert components.transport > 0

    def test_smoothness_penalty_on_jerky_action(self, default_cfg, base_state):
        """Smoothness penalty should be negative for large action changes."""
        base_state.current_action = np.ones(7)
        base_state.previous_action = -np.ones(7)

        components = compute_reward(base_state, default_cfg)
        assert components.smoothness < 0, "Smoothness should penalize jerky actions"

    def test_smoothness_zero_for_constant_action(self, default_cfg, base_state):
        """Smoothness penalty should be zero for constant actions."""
        base_state.current_action = np.ones(7) * 0.5
        base_state.previous_action = np.ones(7) * 0.5

        components = compute_reward(base_state, default_cfg)
        assert components.smoothness == 0.0

    def test_joint_limit_penalty_when_violated(self, default_cfg, base_state):
        """Joint limit penalty should be negative when limits exceeded."""
        base_state.joint_pos = np.full(7, 5.0)  # way above limits

        components = compute_reward(base_state, default_cfg)
        assert components.joint_limit < 0

    def test_joint_limit_zero_when_safe(self, default_cfg, base_state):
        """Joint limit penalty should be zero within limits."""
        base_state.joint_pos = np.zeros(7)

        components = compute_reward(base_state, default_cfg)
        assert components.joint_limit == 0.0

    def test_collision_penalty(self, default_cfg, base_state):
        """Collision penalty should be negative on collision."""
        base_state.has_collision = True
        components = compute_reward(base_state, default_cfg)
        assert components.collision < 0

    def test_no_collision_no_penalty(self, default_cfg, base_state):
        """No collision penalty when no collision."""
        base_state.has_collision = False
        components = compute_reward(base_state, default_cfg)
        assert components.collision == 0.0

    def test_time_penalty_always_negative(self, default_cfg, base_state):
        """Time penalty should always be negative."""
        components = compute_reward(base_state, default_cfg)
        assert components.time < 0

    def test_success_bonus_on_threshold(self, default_cfg, base_state):
        """Success bonus should trigger at threshold."""
        base_state.soil_in_target_ratio = 0.85
        base_state.already_succeeded = False

        components = compute_reward(base_state, default_cfg)
        assert components.success_bonus > 0

    def test_success_bonus_only_once(self, default_cfg, base_state):
        """Success bonus should not repeat."""
        base_state.soil_in_target_ratio = 0.85
        base_state.already_succeeded = True

        components = compute_reward(base_state, default_cfg)
        assert components.success_bonus == 0.0


class TestRewardTotal:
    """Test total reward computation."""

    def test_total_is_sum(self, default_cfg, base_state):
        """Total should equal sum of all components."""
        components = compute_reward(base_state, default_cfg)
        expected = (
            components.transfer + components.approach + components.load
            + components.transport + components.smoothness + components.joint_limit
            + components.time + components.collision + components.success_bonus
        )
        np.testing.assert_almost_equal(components.total, expected)

    def test_to_dict_keys(self, default_cfg, base_state):
        """to_dict should include all expected keys."""
        components = compute_reward(base_state, default_cfg)
        d = components.to_dict()
        assert "reward/_total" in d
        assert "reward/transfer" in d
        assert "reward/approach" in d

    def test_weight_overrides(self, default_cfg, base_state):
        """Weight overrides should modify reward computation."""
        base_state.ee_pos = np.array([0.4, 0.0, 0.0])  # near soil

        components_default = compute_reward(base_state, default_cfg)
        components_override = compute_reward(
            base_state, default_cfg, weight_overrides={"approach": 100.0}
        )

        assert components_override.approach > components_default.approach


class TestRewardScale:
    """Test that reward components have comparable magnitudes."""

    def test_component_magnitudes_comparable(self, default_cfg):
        """Under typical conditions, no component should dominate by >100x."""
        state = RewardState(
            soil_in_target_ratio=0.05,
            prev_soil_in_target_ratio=0.0,
            soil_centroid=np.array([0.4, 0.0, 0.1]),
            ee_pos=np.array([0.3, 0.0, 0.2]),
            bucket_load=0.3,
            target_pos=np.array([-0.4, 0.0, 0.0]),
            current_action=np.random.uniform(-0.5, 0.5, 7),
            previous_action=np.random.uniform(-0.5, 0.5, 7),
            joint_pos=np.zeros(7),
            joint_limits_lower=np.full(7, -3.0),
            joint_limits_upper=np.full(7, 3.0),
            has_collision=False,
            already_succeeded=False,
        )

        components = compute_reward(state, default_cfg)
        values = [
            abs(components.transfer), abs(components.approach),
            abs(components.load), abs(components.transport),
        ]
        nonzero = [v for v in values if v > 1e-8]
        if len(nonzero) >= 2:
            ratio = max(nonzero) / min(nonzero)
            assert ratio < 100, f"Reward component ratio too large: {ratio}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
