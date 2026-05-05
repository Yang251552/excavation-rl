"""Termination conditions for the excavation environment.

Defines success and failure conditions that end an episode. Each
condition is evaluated independently and returns a boolean + metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .excavation_env_cfg import TerminationCfg


class TerminationReason(Enum):
    """Possible reasons for episode termination."""

    NOT_TERMINATED = "not_terminated"
    SUCCESS = "success"
    TIMEOUT = "timeout"
    JOINT_LIMIT = "joint_limit"
    GROUND_PENETRATION = "ground_penetration"
    BASE_DISPLACEMENT = "base_displacement"


@dataclass
class TerminationResult:
    """Result of termination check."""

    terminated: bool = False
    truncated: bool = False  # True for timeout (not a failure)
    reason: TerminationReason = TerminationReason.NOT_TERMINATED

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass
class TerminationState:
    """State needed for termination evaluation."""

    current_step: int = 0
    soil_in_target_ratio: float = 0.0
    joint_pos: np.ndarray = None
    joint_limits_lower: np.ndarray = None
    joint_limits_upper: np.ndarray = None
    bucket_z: float = 0.0
    base_pos: np.ndarray = None
    initial_base_pos: np.ndarray = None


def check_termination(
    state: TerminationState,
    cfg: TerminationCfg,
) -> TerminationResult:
    """Evaluate all termination conditions.

    Conditions are checked in priority order: success first, then
    failure conditions, then timeout.

    Args:
        state: Current termination-relevant state.
        cfg: Termination configuration with thresholds.

    Returns:
        TerminationResult indicating if and why the episode ended.
    """
    # Success: enough soil transferred to target
    if state.soil_in_target_ratio >= cfg.success_soil_ratio:
        return TerminationResult(
            terminated=True,
            truncated=False,
            reason=TerminationReason.SUCCESS,
        )

    # Joint limit violation
    if cfg.check_joint_limits and state.joint_pos is not None:
        if state.joint_limits_lower is not None and state.joint_limits_upper is not None:
            below = np.any(state.joint_pos < state.joint_limits_lower - 0.01)
            above = np.any(state.joint_pos > state.joint_limits_upper + 0.01)
            if below or above:
                return TerminationResult(
                    terminated=True,
                    truncated=False,
                    reason=TerminationReason.JOINT_LIMIT,
                )

    # Bucket ground penetration
    if cfg.check_bucket_ground_penetration:
        if state.bucket_z < cfg.ground_penetration_threshold:
            return TerminationResult(
                terminated=True,
                truncated=False,
                reason=TerminationReason.GROUND_PENETRATION,
            )

    # Robot base displacement (simulation instability check)
    if (
        cfg.check_base_displacement
        and state.base_pos is not None
        and state.initial_base_pos is not None
    ):
        displacement = np.linalg.norm(state.base_pos - state.initial_base_pos)
        if displacement > cfg.base_displacement_threshold:
            return TerminationResult(
                terminated=True,
                truncated=False,
                reason=TerminationReason.BASE_DISPLACEMENT,
            )

    # Timeout (truncation, not failure)
    if state.current_step >= cfg.max_episode_length:
        return TerminationResult(
            terminated=False,
            truncated=True,
            reason=TerminationReason.TIMEOUT,
        )

    return TerminationResult()


def check_termination_batch(
    states: list[TerminationState],
    cfg: TerminationCfg,
) -> tuple[np.ndarray, np.ndarray, list[TerminationReason]]:
    """Check termination for a batch of environments.

    Args:
        states: List of termination states.
        cfg: Termination configuration.

    Returns:
        Tuple of (terminated_mask, truncated_mask, reasons).
    """
    n = len(states)
    terminated = np.zeros(n, dtype=bool)
    truncated = np.zeros(n, dtype=bool)
    reasons = []

    for i, state in enumerate(states):
        result = check_termination(state, cfg)
        terminated[i] = result.terminated
        truncated[i] = result.truncated
        reasons.append(result.reason)

    return terminated, truncated, reasons
