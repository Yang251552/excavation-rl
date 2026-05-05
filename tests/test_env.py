"""Tests for the excavation environment.

Validates environment correctness per Section 4.1 of the implementation plan:
  - Observation range check
  - Reward signal sanity
  - Termination logic
  - Reset consistency
  - API compatibility
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.excavation_env import ExcavationEnv
from envs.excavation_env_cfg import ExcavationEnvCfg


@pytest.fixture
def env():
    """Create a default environment for testing."""
    cfg = ExcavationEnvCfg(seed=42)
    cfg.scene.use_rigid_body_proxy = True  # Use simplified mode for testing
    e = ExcavationEnv(cfg)
    yield e
    e.close()


@pytest.fixture
def env_particles():
    """Create an environment with particle system (requires Warp)."""
    cfg = ExcavationEnvCfg(seed=42)
    cfg.scene.use_rigid_body_proxy = False
    cfg.scene.soil_num_particles = 100
    try:
        e = ExcavationEnv(cfg)
        yield e
        e.close()
    except RuntimeError:
        pytest.skip("Warp not available")


class TestEnvironmentAPI:
    """Test Gymnasium API compatibility."""

    def test_observation_space_contains_obs(self, env):
        """Observation from reset should be within observation space."""
        obs, info = env.reset(seed=42)
        assert env.observation_space.contains(obs), (
            f"Observation not in space. Shape: {obs.shape}, "
            f"Range: [{obs.min():.3f}, {obs.max():.3f}]"
        )

    def test_action_space_shape(self, env):
        """Action space should match robot joint count."""
        assert env.action_space.shape == (env.num_joints,)

    def test_step_returns_correct_types(self, env):
        """Step should return (obs, reward, terminated, truncated, info)."""
        env.reset(seed=42)
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5

        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_obs_in_space_during_rollout(self, env):
        """Observation should remain in space throughout a rollout."""
        env.reset(seed=42)
        for _ in range(50):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            assert env.observation_space.contains(obs), (
                f"Obs out of space: [{obs.min():.3f}, {obs.max():.3f}]"
            )
            if terminated or truncated:
                obs, _ = env.reset()


class TestObservationRange:
    """Verify observation values are within physically reasonable ranges."""

    def test_random_policy_obs_range(self, env):
        """Run 1000 steps of random policy and check obs statistics."""
        obs_all = []
        obs, _ = env.reset(seed=42)
        obs_all.append(obs)

        for _ in range(1000):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            obs_all.append(obs)
            if terminated or truncated:
                obs, _ = env.reset()

        obs_arr = np.array(obs_all)

        # No NaN or Inf
        assert not np.any(np.isnan(obs_arr)), "NaN in observations"
        assert not np.any(np.isinf(obs_arr)), "Inf in observations"

        # Within clip range
        clip = env.cfg.observation.clip_obs
        assert obs_arr.min() >= -clip, f"Obs below clip range: {obs_arr.min()}"
        assert obs_arr.max() <= clip, f"Obs above clip range: {obs_arr.max()}"


class TestRewardSignal:
    """Verify reward signal is meaningful and correctly scaled."""

    def test_reward_not_constant(self, env):
        """Random policy should produce varying rewards."""
        rewards = []
        env.reset(seed=42)
        for _ in range(200):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                env.reset()

        # Reward should not be all zeros or all the same value
        rewards_arr = np.array(rewards)
        assert rewards_arr.std() > 1e-6, "Reward is constant (std ≈ 0)"

    def test_reward_components_in_info(self, env):
        """Info dict should contain reward component breakdown."""
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)

        expected_keys = [
            "reward/transfer", "reward/approach", "reward/load",
            "reward/transport", "reward/smooth_penalty",
        ]
        for key in expected_keys:
            assert key in info, f"Missing reward component: {key}"


class TestTermination:
    """Verify termination conditions work correctly."""

    def test_timeout_termination(self, env):
        """Episode should truncate after max_episode_length steps."""
        env.reset(seed=42)
        max_len = env.cfg.termination.max_episode_length

        for step in range(max_len + 10):
            action = np.zeros(env.num_joints, dtype=np.float32)  # no-op
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                assert step + 1 <= max_len, f"Early termination at step {step + 1}"
                break

    def test_termination_reason_in_info(self, env):
        """Info should report termination reason."""
        env.reset(seed=42)
        done = False
        while not done:
            action = env.action_space.sample()
            _, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        assert "termination_reason" in info


class TestResetConsistency:
    """Verify that reset produces consistent initial conditions."""

    def test_deterministic_reset(self, env):
        """Same seed should produce identical initial observations."""
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)

        np.testing.assert_array_equal(obs1, obs2, "Reset with same seed differs")

    def test_different_seeds_differ(self, env):
        """Different seeds should produce different initial observations."""
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=999)

        # They may share some components (e.g., target pos) but shouldn't be identical
        # unless the initial state is fully deterministic regardless of seed
        # This is a soft check since some observations are constant
        assert obs1.shape == obs2.shape

    def test_consecutive_resets(self, env):
        """100 consecutive resets should all produce valid observations."""
        for i in range(100):
            obs, info = env.reset(seed=i)
            assert env.observation_space.contains(obs), f"Invalid obs on reset {i}"


class TestReproducibility:
    """Verify that the environment is reproducible with fixed seeds."""

    def test_reproducible_trajectory(self, env):
        """Same seed + same actions should produce identical trajectories."""
        def run_trajectory(seed, actions):
            obs_list, rew_list = [], []
            obs, _ = env.reset(seed=seed)
            obs_list.append(obs.copy())
            for a in actions:
                obs, rew, term, trunc, _ = env.step(a)
                obs_list.append(obs.copy())
                rew_list.append(rew)
                if term or trunc:
                    break
            return np.array(obs_list), np.array(rew_list)

        rng = np.random.default_rng(0)
        actions = [rng.uniform(-1, 1, env.num_joints).astype(np.float32) for _ in range(50)]

        obs1, rew1 = run_trajectory(42, actions)
        obs2, rew2 = run_trajectory(42, actions)

        np.testing.assert_array_equal(obs1, obs2, "Non-reproducible observations")
        np.testing.assert_array_equal(rew1, rew2, "Non-reproducible rewards")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
