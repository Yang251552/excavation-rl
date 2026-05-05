"""PPO training configuration for rsl_rl.

Defines hyperparameters for Proximal Policy Optimization using the
rsl_rl library conventions. Includes both the algorithm config and
the network architecture config.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NetworkCfg:
    """MLP policy and value network architecture."""

    # Shared or separate networks
    shared_backbone: bool = False

    # Policy network
    policy_hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 128])
    policy_activation: str = "elu"

    # Value network
    value_hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 128])
    value_activation: str = "elu"

    # Initialization
    init_noise_std: float = 1.0


@dataclass
class PPOCfg:
    """PPO algorithm hyperparameters.

    These follow the rsl_rl PPO implementation conventions and the
    tuning guidelines from "37 Implementation Details of PPO".
    """

    # Training
    seed: int = 42
    num_envs: int = 4096
    num_steps_per_env: int = 24        # steps per env per update
    max_iterations: int = 1500         # total PPO updates

    # PPO core
    clip_param: float = 0.2            # epsilon for clipping
    gamma: float = 0.99                # discount factor
    lam: float = 0.95                  # GAE lambda
    num_epochs: int = 5                # epochs per PPO update
    num_mini_batches: int = 4          # mini-batches per epoch
    learning_rate: float = 3e-4
    schedule: str = "adaptive"         # "fixed" or "adaptive"
    desired_kl: float = 0.01           # target KL for adaptive lr

    # Value function
    value_loss_coeff: float = 1.0
    use_clipped_value_loss: bool = True

    # Entropy
    entropy_coeff: float = 0.01

    # Gradient
    max_grad_norm: float = 1.0

    # Network
    network: NetworkCfg = field(default_factory=NetworkCfg)

    # Logging
    log_interval: int = 10             # log every N iterations
    save_interval: int = 100           # save checkpoint every N iterations

    # Paths
    log_dir: str = "results/training_curves"
    checkpoint_dir: str = "results/checkpoints"

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps_per_env

    @property
    def mini_batch_size(self) -> int:
        return self.batch_size // self.num_mini_batches

    @property
    def total_timesteps(self) -> int:
        return self.max_iterations * self.batch_size


# Predefined configurations for different training stages

def get_stage0_ppo_cfg() -> PPOCfg:
    """PPO config for Stage 0 (rigid body proxy — fast iteration).

    Uses a fixed LR schedule because adaptive KL-based LR collapsed the
    learning rate to ~5e-7 in the first ~20 iterations on this reward scale,
    leaving the policy frozen.
    """
    return PPOCfg(
        num_envs=4096,
        num_steps_per_env=24,
        max_iterations=1000,       # simplified reward converges fast; 1000 enough
        learning_rate=3e-4,
        schedule="fixed",          # adaptive immediately bottomed out at 1e-5 in
                                   # multi-seed runs; the simplified positional
                                   # reward is well-shaped enough that we don't
                                   # need KL-based LR control.
        clip_param=0.2,            # back to PPO default — no longer fighting
                                   # value-function blowups now that the reward
                                   # is single-component (approach only).
        num_epochs=5,              # back to PPO default
        entropy_coeff=0.01,
        network=NetworkCfg(
            policy_hidden_dims=[128, 128],
            value_hidden_dims=[128, 128],
            init_noise_std=0.5,
        ),
    )


def get_stage1_ppo_cfg() -> PPOCfg:
    """PPO config for Stage 1 (particle system — moderate training).

    v10 throttle (after v9 showed reward collapsing 12 -> 2 over 220 iter
    with KL=10 sustained):
      - num_epochs 5 -> 2:  fewer Adam updates per rollout, less drift
      - num_mini_batches 4 -> 8: smaller per-update batches, less variance
      - learning_rate 1e-4 -> 5e-5: another 2x throttle on top of fewer epochs
      - desired_kl 0.01 -> 0.02: KL early-stop threshold = 4*desired = 0.08,
        loose enough that we don't stop on iter 0 but tight enough to limit
        per-iter drift. Combined with fewer epochs/smaller batches, total
        per-iter policy movement should stay under 1.0 KL.
    """
    return PPOCfg(
        num_envs=2048,
        num_steps_per_env=24,
        max_iterations=1000,
        learning_rate=5e-5,
        schedule="fixed",
        clip_param=0.2,
        num_epochs=2,
        num_mini_batches=8,
        desired_kl=0.02,
        entropy_coeff=0.01,
        network=NetworkCfg(
            policy_hidden_dims=[128, 128],
            value_hidden_dims=[128, 128],
            init_noise_std=0.5,
        ),
    )


def get_stage2_ppo_cfg() -> PPOCfg:
    """PPO config for Stage 2 (full training with curriculum)."""
    return PPOCfg(
        num_envs=4096,
        num_steps_per_env=24,
        max_iterations=2000,
        learning_rate=3e-4,
        schedule="adaptive",
        desired_kl=0.01,
        entropy_coeff=0.005,
        network=NetworkCfg(
            policy_hidden_dims=[256, 256, 128],
            value_hidden_dims=[256, 256, 128],
            init_noise_std=0.8,
        ),
    )
