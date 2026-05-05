"""Observation computation for the excavation environment.

Computes observation vectors from robot state, soil state, and task state.
Supports multiple soil observation modes (height map, centroid, none).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .excavation_env_cfg import ObservationCfg


@dataclass
class RobotState:
    """Raw robot state from simulation."""

    joint_pos: np.ndarray        # (num_joints,)
    joint_vel: np.ndarray        # (num_joints,)
    ee_pos: np.ndarray           # (3,)
    ee_orient: np.ndarray        # (4,) quaternion wxyz
    ee_lin_vel: np.ndarray       # (3,)
    ee_ang_vel: np.ndarray       # (3,)


@dataclass
class BucketState:
    """Bucket sensor readings."""

    load: float = 0.0                          # kg
    contact_force: np.ndarray = None           # (3,)

    def __post_init__(self):
        if self.contact_force is None:
            self.contact_force = np.zeros(3, dtype=np.float32)


@dataclass
class SoilObservation:
    """Processed soil state for observation."""

    height_map: np.ndarray | None = None   # (H, W) flattened to (H*W,)
    centroid: np.ndarray | None = None     # (3,)
    spread: np.ndarray | None = None       # (3,)
    in_target_ratio: float = 0.0           # scalar


def compute_observation_dim(cfg: ObservationCfg, num_joints: int) -> int:
    """Compute the total observation vector dimensionality.

    Args:
        cfg: Observation configuration.
        num_joints: Number of robot joints.

    Returns:
        Total observation dimension.
    """
    dim = 0

    # Proprioception
    if cfg.include_joint_pos:
        dim += num_joints
    if cfg.include_joint_vel:
        dim += num_joints
    if cfg.include_ee_pos:
        dim += 3
    if cfg.include_ee_orient:
        dim += 4
    if cfg.include_ee_lin_vel:
        dim += 3
    if cfg.include_ee_ang_vel:
        dim += 3

    # Bucket
    if cfg.include_bucket_load:
        dim += 1
    if cfg.include_bucket_contact_force:
        dim += 3

    # Soil
    if cfg.soil_obs_mode == "height_map":
        dim += cfg.height_map_resolution ** 2
    elif cfg.soil_obs_mode == "centroid":
        dim += 3 + 3 + 1  # centroid(3) + spread(3) + in_target(1)

    # Target
    if cfg.include_target_pos:
        dim += 3

    # Previous action
    if cfg.include_previous_action:
        dim += num_joints

    return dim


def build_observation(
    cfg: ObservationCfg,
    robot: RobotState,
    bucket: BucketState,
    soil: SoilObservation,
    target_pos: np.ndarray,
    previous_action: np.ndarray,
) -> np.ndarray:
    """Assemble the full observation vector from components.

    Args:
        cfg: Observation configuration.
        robot: Current robot state.
        bucket: Bucket sensor state.
        soil: Processed soil observation.
        target_pos: (3,) target zone center.
        previous_action: (num_joints,) last action applied.

    Returns:
        1-D numpy observation vector.
    """
    parts = []

    # Proprioception
    if cfg.include_joint_pos:
        parts.append(robot.joint_pos.astype(np.float32))
    if cfg.include_joint_vel:
        parts.append(robot.joint_vel.astype(np.float32))
    if cfg.include_ee_pos:
        parts.append(robot.ee_pos.astype(np.float32))
    if cfg.include_ee_orient:
        parts.append(robot.ee_orient.astype(np.float32))
    if cfg.include_ee_lin_vel:
        parts.append(robot.ee_lin_vel.astype(np.float32))
    if cfg.include_ee_ang_vel:
        parts.append(robot.ee_ang_vel.astype(np.float32))

    # Bucket state
    if cfg.include_bucket_load:
        parts.append(np.array([bucket.load], dtype=np.float32))
    if cfg.include_bucket_contact_force:
        parts.append(bucket.contact_force.astype(np.float32))

    # Soil state
    if cfg.soil_obs_mode == "height_map" and soil.height_map is not None:
        parts.append(soil.height_map.flatten().astype(np.float32))
    elif cfg.soil_obs_mode == "centroid":
        centroid = soil.centroid if soil.centroid is not None else np.zeros(3, dtype=np.float32)
        spread = soil.spread if soil.spread is not None else np.zeros(3, dtype=np.float32)
        parts.append(centroid.astype(np.float32))
        parts.append(spread.astype(np.float32))
        parts.append(np.array([soil.in_target_ratio], dtype=np.float32))

    # Target
    if cfg.include_target_pos:
        parts.append(target_pos.astype(np.float32))

    # Previous action
    if cfg.include_previous_action:
        parts.append(previous_action.astype(np.float32))

    obs = np.concatenate(parts)

    # Add observation noise
    if cfg.observation_noise_std > 0:
        noise = np.random.normal(0, cfg.observation_noise_std, obs.shape).astype(np.float32)
        obs = obs + noise

    # Clip
    if cfg.clip_obs > 0:
        obs = np.clip(obs, -cfg.clip_obs, cfg.clip_obs)

    return obs


def build_observation_batch(
    cfg: ObservationCfg,
    robot_states: list[RobotState],
    bucket_states: list[BucketState],
    soil_observations: list[SoilObservation],
    target_pos: np.ndarray,
    previous_actions: np.ndarray,
) -> torch.Tensor:
    """Build batched observation tensor for vectorized environments.

    Args:
        cfg: Observation configuration.
        robot_states: List of robot states, one per environment.
        bucket_states: List of bucket states.
        soil_observations: List of soil observations.
        target_pos: (3,) target center (shared across envs).
        previous_actions: (num_envs, num_joints) previous actions.

    Returns:
        Tensor of shape (num_envs, obs_dim).
    """
    obs_list = []
    for i in range(len(robot_states)):
        obs = build_observation(
            cfg,
            robot_states[i],
            bucket_states[i],
            soil_observations[i],
            target_pos,
            previous_actions[i],
        )
        obs_list.append(obs)

    return torch.tensor(np.stack(obs_list), dtype=torch.float32)
