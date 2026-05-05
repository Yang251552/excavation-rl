"""Training callbacks for custom logging and monitoring.

Provides callback hooks that integrate with the rsl_rl training loop
to log reward components, curriculum state, and custom metrics to
TensorBoard and (optionally) Weights & Biases.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    import wandb
except ImportError:
    wandb = None


class TrainingLogger:
    """Custom training logger for excavation RL.

    Logs to TensorBoard and optionally to Weights & Biases. Tracks all
    metrics specified in Section 4.2 of the implementation plan.
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str = "excavation",
        wandb_cfg: dict | None = None,
    ):
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.writer: SummaryWriter | None = None
        if SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        # Episode statistics buffer
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []
        self._episode_soil_ratios: list[float] = []
        self._episode_successes: list[bool] = []
        self._reward_components: dict[str, list[float]] = {}

        self._global_step = 0

        # Optional wandb. Buffer scalars per-step so we emit one wandb.log
        # call per iteration (wandb requires monotonically increasing step).
        self._use_wandb = False
        self._wandb_buffer: dict[str, float] = {}
        if wandb_cfg is not None and wandb is not None:
            init_kwargs = dict(wandb_cfg.get("init_kwargs", {}))
            init_kwargs.setdefault("dir", self.log_dir)
            init_kwargs.setdefault("name", experiment_name)
            init_kwargs.setdefault("config", {})
            wandb.init(**init_kwargs)
            self._use_wandb = True
        elif wandb_cfg is not None and wandb is None:
            print("[TrainingLogger] wandb requested but not installed — skipping.")

    def log_scalar(self, tag: str, value: float, step: int | None = None) -> None:
        """Log a scalar value to TensorBoard (and buffer for wandb)."""
        step = step if step is not None else self._global_step
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)
        if self._use_wandb:
            self._wandb_buffer[tag] = float(value)

    def log_episode(self, info: dict[str, Any]) -> None:
        """Log episode-level statistics.

        Args:
            info: Episode info dict from environment step.
        """
        if "episode" not in info:
            return

        ep = info["episode"]
        self._episode_rewards.append(ep.get("r", 0))
        self._episode_lengths.append(ep.get("l", 0))
        self._episode_soil_ratios.append(ep.get("soil_transfer_ratio", 0))
        self._episode_successes.append(ep.get("success", False))

        # Log all per-episode cumulative reward components (any "reward/*" key).
        for key, val in ep.items():
            if not key.startswith("reward/"):
                continue
            if key not in self._reward_components:
                self._reward_components[key] = []
            self._reward_components[key].append(val)

    def log_iteration(self, iteration: int, ppo_metrics: dict | None = None) -> None:
        """Log metrics at the end of a PPO iteration.

        Called every N iterations (controlled by log_interval).

        Args:
            iteration: Current PPO iteration number.
            ppo_metrics: Dict with PPO algorithm metrics (losses, KL, etc).
        """
        self._global_step = iteration

        # Episode performance
        if self._episode_rewards:
            self.log_scalar("Performance/episodic_return", np.mean(self._episode_rewards), iteration)
            self.log_scalar("Performance/episodic_return_std", np.std(self._episode_rewards), iteration)
            self.log_scalar("Performance/episodic_length", np.mean(self._episode_lengths), iteration)
            self.log_scalar("Performance/soil_transfer_ratio", np.mean(self._episode_soil_ratios), iteration)
            self.log_scalar("Performance/success_rate", np.mean(self._episode_successes), iteration)

        # Reward decomposition
        for key, values in self._reward_components.items():
            if values:
                self.log_scalar(key, np.mean(values), iteration)

        # PPO health metrics
        if ppo_metrics:
            for key, val in ppo_metrics.items():
                self.log_scalar(f"Policy/{key}", val, iteration)

        # Clear buffers
        self._episode_rewards.clear()
        self._episode_lengths.clear()
        self._episode_soil_ratios.clear()
        self._episode_successes.clear()
        self._reward_components.clear()

        # Flush this iteration's metrics to wandb in a single call
        self._flush_wandb(iteration)

    def log_curriculum(self, curriculum_info: dict, step: int) -> None:
        """Log curriculum state."""
        for key, val in curriculum_info.items():
            if isinstance(val, (int, float)):
                self.log_scalar(key, val, step)

    def _flush_wandb(self, step: int) -> None:
        """Emit buffered scalars to wandb as one log call at this step."""
        if not self._use_wandb or not self._wandb_buffer:
            return
        wandb.log(self._wandb_buffer, step=step)
        self._wandb_buffer.clear()

    def close(self) -> None:
        """Flush and close the TensorBoard writer (and wandb)."""
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        if self._use_wandb:
            self._flush_wandb(self._global_step)
            wandb.finish()


class CheckpointCallback:
    """Manages model checkpoint saving during training."""

    def __init__(self, checkpoint_dir: str, save_interval: int = 100):
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        os.makedirs(checkpoint_dir, exist_ok=True)

        self._best_reward = -float("inf")

    def should_save(self, iteration: int) -> bool:
        return iteration % self.save_interval == 0

    def update_best(self, mean_reward: float) -> bool:
        """Check if current performance is best so far.

        Returns True if this is the new best.
        """
        if mean_reward > self._best_reward:
            self._best_reward = mean_reward
            return True
        return False

    def get_save_path(self, iteration: int, is_best: bool = False) -> str:
        if is_best:
            return os.path.join(self.checkpoint_dir, "best_model.pt")
        return os.path.join(self.checkpoint_dir, f"model_{iteration:06d}.pt")
