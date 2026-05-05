"""Quantitative evaluation metrics for the excavation task.

Implements all metrics from Section 4.5 of the implementation plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics collected from a single evaluation episode."""

    total_reward: float = 0.0
    episode_length: int = 0
    soil_transfer_ratio: float = 0.0
    success: bool = False
    termination_reason: str = ""

    # Action smoothness: mean ||a_t - a_{t-1}||_2
    action_smoothness: float = 0.0

    # Energy: sum of |torque * angular_velocity| * dt
    energy_consumption: float = 0.0

    # Reward component totals
    reward_components: dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Aggregated evaluation metrics across multiple episodes."""

    num_episodes: int = 0
    num_seeds: int = 1

    # Core metrics
    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_episode_length: float = 0.0
    std_episode_length: float = 0.0
    mean_soil_transfer: float = 0.0
    std_soil_transfer: float = 0.0
    success_rate: float = 0.0

    # Action quality
    mean_action_smoothness: float = 0.0
    mean_energy: float = 0.0

    # Convergence info
    converge_step: int | None = None  # step at which training converged

    # Per-episode data (for plotting)
    episode_rewards: list[float] = field(default_factory=list)
    episode_soil_ratios: list[float] = field(default_factory=list)

    def summary_table(self) -> str:
        """Format metrics as a readable table."""
        lines = [
            "=" * 55,
            f"  Evaluation Report ({self.num_episodes} episodes, {self.num_seeds} seeds)",
            "=" * 55,
            f"  Mean Reward:          {self.mean_reward:8.2f} +/- {self.std_reward:.2f}",
            f"  Mean Episode Length:   {self.mean_episode_length:8.1f} +/- {self.std_episode_length:.1f}",
            f"  Soil Transfer Ratio:  {self.mean_soil_transfer:8.3f} +/- {self.std_soil_transfer:.3f}",
            f"  Success Rate:         {self.success_rate:8.1%}",
            f"  Action Smoothness:    {self.mean_action_smoothness:8.4f}",
            f"  Energy Consumption:   {self.mean_energy:8.2f}",
            "=" * 55,
        ]
        return "\n".join(lines)


def compute_episode_metrics(
    rewards: list[float],
    actions: np.ndarray,
    soil_ratios: list[float],
    success: bool,
    termination_reason: str,
    reward_components: dict[str, list[float]] | None = None,
    joint_torques: np.ndarray | None = None,
    joint_velocities: np.ndarray | None = None,
    dt: float = 1 / 60.0,
) -> EpisodeMetrics:
    """Compute metrics for a single episode.

    Args:
        rewards: List of per-step rewards.
        actions: (T, num_joints) array of actions.
        soil_ratios: List of per-step soil-in-target ratios.
        success: Whether the episode was successful.
        termination_reason: Reason for episode termination.
        reward_components: Optional dict of component name → list of values.
        joint_torques: Optional (T, num_joints) torque array.
        joint_velocities: Optional (T, num_joints) velocity array.
        dt: Simulation timestep.

    Returns:
        EpisodeMetrics for this episode.
    """
    metrics = EpisodeMetrics(
        total_reward=sum(rewards),
        episode_length=len(rewards),
        soil_transfer_ratio=soil_ratios[-1] if soil_ratios else 0.0,
        success=success,
        termination_reason=termination_reason,
    )

    # Action smoothness
    if len(actions) > 1:
        diffs = np.diff(actions, axis=0)
        metrics.action_smoothness = float(np.mean(np.linalg.norm(diffs, axis=-1)))

    # Energy consumption
    if joint_torques is not None and joint_velocities is not None:
        power = np.abs(joint_torques * joint_velocities)
        metrics.energy_consumption = float(np.sum(power) * dt)

    # Reward components
    if reward_components:
        for key, values in reward_components.items():
            metrics.reward_components[key] = sum(values)

    return metrics


def aggregate_metrics(episode_metrics: list[EpisodeMetrics], num_seeds: int = 1) -> EvaluationReport:
    """Aggregate metrics across multiple episodes into a report.

    Args:
        episode_metrics: List of per-episode metrics.
        num_seeds: Number of random seeds used.

    Returns:
        EvaluationReport with aggregated statistics.
    """
    n = len(episode_metrics)
    if n == 0:
        return EvaluationReport()

    rewards = [m.total_reward for m in episode_metrics]
    lengths = [m.episode_length for m in episode_metrics]
    transfers = [m.soil_transfer_ratio for m in episode_metrics]
    successes = [m.success for m in episode_metrics]
    smoothness = [m.action_smoothness for m in episode_metrics]
    energy = [m.energy_consumption for m in episode_metrics]

    return EvaluationReport(
        num_episodes=n,
        num_seeds=num_seeds,
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_episode_length=float(np.mean(lengths)),
        std_episode_length=float(np.std(lengths)),
        mean_soil_transfer=float(np.mean(transfers)),
        std_soil_transfer=float(np.std(transfers)),
        success_rate=float(np.mean(successes)),
        mean_action_smoothness=float(np.mean(smoothness)),
        mean_energy=float(np.mean(energy)),
        episode_rewards=rewards,
        episode_soil_ratios=transfers,
    )
