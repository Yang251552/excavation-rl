"""Main excavation RL environment.

Implements a Gymnasium-compatible RL environment for the excavation task.
Supports two modes:
  1. Standalone mode: runs without Isaac Lab for unit testing and prototyping
  2. Isaac Lab mode: integrates with Isaac Lab's ManagerBasedRLEnv

This file implements the standalone mode. For Isaac Lab integration, see
the Isaac Lab extension registration pattern in __init__.py.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch

from .excavation_env_cfg import ExcavationEnvCfg
from .observations import (
    ObservationCfg,
    RobotState,
    BucketState,
    SoilObservation,
    build_observation,
    compute_observation_dim,
)
from .rewards import RewardState, RewardComponents, compute_reward
from .terminations import TerminationState, TerminationReason, check_termination
from .curriculum import CurriculumManager
from .events import EventManager, RandomizedParams
from .scene import ExcavationScene

from robot.arm_cfg import get_arm_cfg


class ExcavationEnv(gym.Env):
    """Gymnasium environment for particle-based excavation.

    The robot arm must use its bucket end-effector to scoop soil particles
    from a heap and transfer them into a target zone. The environment
    provides dense reward signals to guide the learning of the full
    excavation behavior: approach → scoop → transport → dump.

    Observation space:
        Concatenated vector of robot proprioception, bucket state,
        soil state (height map or centroid), and target position.

    Action space:
        Continuous joint position deltas in [-1, 1], scaled to
        [-max_delta, +max_delta] radians.

    Reward:
        Multi-component dense reward (see rewards.py for details).

    Termination:
        Success (80% soil transferred), timeout, or failure conditions.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        cfg: ExcavationEnvCfg | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.cfg = cfg or ExcavationEnvCfg()
        self.render_mode = render_mode

        # Robot configuration
        self.arm_cfg = get_arm_cfg(self.cfg.scene.arm_type)
        self.num_joints = self.arm_cfg.num_joints

        # Observation and action spaces
        obs_dim = compute_observation_dim(self.cfg.observation, self.num_joints)
        self.observation_space = gym.spaces.Box(
            low=-self.cfg.observation.clip_obs,
            high=self.cfg.observation.clip_obs,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_joints,),
            dtype=np.float32,
        )

        # Scene
        self.scene = ExcavationScene(self.cfg)
        # Build allocates the Warp particle system in particle mode. Without
        # this, scene.particle_system stays None and stage 1 silently runs
        # with zero particles -> bucket_load=0 / soil_centroid=[0,0,0] /
        # transfer_ratio=0 forever, regardless of policy. Was missing in v1-v7.
        self.scene.build()

        # Curriculum and domain randomization
        self.curriculum = CurriculumManager(self.cfg.curriculum)
        self.event_manager = EventManager(self.cfg.domain_randomization, self.cfg.seed)

        # Internal state
        self._step_count = 0
        self._total_steps = 0
        self._episode_count = 0
        self._rng = np.random.default_rng(self.cfg.seed)

        # Robot state (simulated in standalone mode)
        self._joint_pos = np.array(self.arm_cfg.default_joint_pos, dtype=np.float32)
        self._joint_vel = np.zeros(self.num_joints, dtype=np.float32)
        self._ee_pos = np.zeros(3, dtype=np.float32)
        self._ee_orient = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._ee_lin_vel = np.zeros(3, dtype=np.float32)
        self._ee_ang_vel = np.zeros(3, dtype=np.float32)
        self._bucket_contact_force = np.zeros(3, dtype=np.float32)
        self._base_pos = np.array(self.cfg.scene.robot_base_pos, dtype=np.float32)

        # Action history
        self._prev_action = np.zeros(self.num_joints, dtype=np.float32)

        # Reward tracking
        self._prev_soil_in_target = 0.0
        self._already_succeeded = False
        self._episode_reward = 0.0
        self._reward_components_accum: dict[str, float] = {}

        # Domain randomization params for current episode
        self._dr_params: RandomizedParams = RandomizedParams()

        # Action delay buffer
        self._action_buffer: list[np.ndarray] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            Tuple of (observation, info_dict).
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Update curriculum
        stage_changed = self.curriculum.update(self._total_steps)

        # Sample domain randomization parameters
        dr_enabled = self.curriculum.is_dr_enabled() if self.curriculum.is_enabled else self.cfg.domain_randomization.enabled
        self._dr_params = self.event_manager.sample(override_enabled=dr_enabled)

        # Reset robot state. If a reverse-curriculum demo trajectory is
        # registered (via env.set_demo_trajectory), with prob 0.5 sample a
        # random t0 in [10, 180] and start the agent at that joint config.
        # This bypasses the early-trajectory exploration that pure RL gets
        # stuck on -- agent practises completing the scoop from any prefix.
        if hasattr(self, "_demo_trajectory") and self._demo_trajectory is not None and self._rng.random() < self._demo_reset_prob:
            t0 = int(self._rng.integers(10, len(self._demo_trajectory) - 20))
            self._joint_pos = self._demo_trajectory[t0].astype(np.float32).copy()
            self._reset_curriculum_t0 = t0
            # Note: scene.reset() runs below, then we patch its particle state
            # using the snapshot (if available). This way the env reset still
            # initialises a fresh heap, then we overwrite particle pos/vel +
            # carried dict to match the snapshot at t0.
            self._pending_snapshot_t0 = t0
        else:
            self._joint_pos = np.array(self.arm_cfg.default_joint_pos, dtype=np.float32)
            self._reset_curriculum_t0 = 0
            self._pending_snapshot_t0 = None
        self._joint_vel = np.zeros(self.num_joints, dtype=np.float32)
        self._ee_lin_vel = np.zeros(3, dtype=np.float32)
        self._ee_ang_vel = np.zeros(3, dtype=np.float32)
        self._update_ee_from_joints()
        self._ee_lin_vel = np.zeros(3, dtype=np.float32)  # zero velocity after FK init
        self._prev_action = np.zeros(self.num_joints, dtype=np.float32)
        self._base_pos = np.array(self.cfg.scene.robot_base_pos, dtype=np.float32)
        self._bucket_contact_force = np.zeros(3, dtype=np.float32)

        # Reset scene (soil particles)
        self.scene.reset(rng=self._rng)

        # Reverse-curriculum particle-state replay: if we sampled a t0 above
        # and a demo snapshot is registered, overwrite the freshly-initialised
        # particle state with the snapshot at t0. This makes the agent face
        # mid-trajectory soil (already partially scooped, partially in bucket)
        # instead of a fresh heap that hasn't been touched.
        if (getattr(self, "_pending_snapshot_t0", None) is not None
                and getattr(self, "_demo_snapshot", None) is not None
                and self.scene.particle_system is not None):
            t0 = self._pending_snapshot_t0
            snap = self._demo_snapshot
            t0 = min(t0, snap["particles"].shape[0] - 1)
            import warp as wp
            ps = self.scene.particle_system
            n = min(snap["particles"].shape[1], ps.num_particles)
            new_pos = ps.get_positions_numpy().copy()
            new_vel = ps.velocities.numpy().reshape(-1, 3).copy()
            new_pos[:n] = snap["particles"][t0, :n].astype(np.float32)
            new_vel[:n] = snap["particle_vel"][t0, :n].astype(np.float32)
            ps.positions = wp.array(new_pos.astype(np.float32), dtype=wp.vec3, device=ps.device)
            ps.velocities = wp.array(new_vel.astype(np.float32), dtype=wp.vec3, device=ps.device)
            # Restore carried-particle dict so containment continues correctly
            carried_idxs = snap["carried"][t0]
            carried_idxs = carried_idxs[carried_idxs >= 0]
            self.scene._carried_particle_offsets = {}
            if len(carried_idxs) > 0:
                # Reconstruct offsets from current bucket pos
                self._update_ee_from_joints()
                positions = ps.get_positions_numpy()
                for i in carried_idxs:
                    self.scene._carried_particle_offsets[int(i)] = positions[int(i)] - self._ee_pos

        # Reset episode tracking
        self._step_count = 0
        self._prev_soil_in_target = 0.0
        self._already_succeeded = False
        self._episode_reward = 0.0
        self._reward_components_accum = {}
        self._action_buffer = []
        self._episode_count += 1

        # Build observation
        obs = self._get_observation()

        info = {
            "episode_count": self._episode_count,
            "total_steps": self._total_steps,
        }
        info.update(self.curriculum.get_info())

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one environment step.

        Args:
            action: (num_joints,) array in [-1, 1].

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Apply action noise (domain randomization)
        action = self.event_manager.apply_action_noise(action, self._dr_params)

        # Action delay buffer
        if self._dr_params.action_delay_steps > 0:
            self._action_buffer.append(action.copy())
            if len(self._action_buffer) > self._dr_params.action_delay_steps:
                action = self._action_buffer.pop(0)
            else:
                action = np.zeros_like(action)

        # Apply action: scale and add to joint positions
        joint_delta = action * self.cfg.action.action_scale
        target_joint_pos = self._joint_pos + joint_delta

        # Clip to joint limits
        target_joint_pos = np.clip(
            target_joint_pos,
            np.array(self.arm_cfg.joint_limits.lower, dtype=np.float32),
            np.array(self.arm_cfg.joint_limits.upper, dtype=np.float32),
        )

        # Simulate PD control (simplified: instant position tracking)
        self._joint_vel = (target_joint_pos - self._joint_pos) / self.cfg.sim_dt
        self._joint_pos = target_joint_pos
        self._update_ee_from_joints()

        # Step particle physics
        self.scene.step_particles(
            bucket_pos=self._ee_pos,
            bucket_vel=self._ee_lin_vel,
        )

        self._step_count += 1
        self._total_steps += 1

        # Compute soil state
        soil_in_target = self.scene.get_soil_in_target_ratio()
        bucket_load = self.scene.get_bucket_load(self._ee_pos)

        # Approach reward target: STATIC heap center (not dynamic centroid).
        # In v14 the dynamic centroid moved away from EE as bucket displaced
        # particles, creating a perverse "scoop -> approach reward drops"
        # dynamic. Heap center is fixed by config and gives a stable target.
        approach_target = np.array(self.cfg.scene.soil_heap_center, dtype=np.float32)

        # Compute reward
        reward_state = RewardState(
            soil_in_target_ratio=soil_in_target,
            prev_soil_in_target_ratio=self._prev_soil_in_target,
            soil_centroid=approach_target,
            ee_pos=self._ee_pos.copy(),
            bucket_load=min(bucket_load / 0.5, 1.0),  # normalize
            target_pos=np.array(self.cfg.scene.target_center, dtype=np.float32),
            current_action=action,
            previous_action=self._prev_action,
            joint_pos=self._joint_pos.copy(),
            joint_limits_lower=np.array(self.arm_cfg.joint_limits.lower, dtype=np.float32),
            joint_limits_upper=np.array(self.arm_cfg.joint_limits.upper, dtype=np.float32),
            has_collision=getattr(self, "_ee_pos_z_raw", self._ee_pos[2]) < 0.0,
            already_succeeded=self._already_succeeded,
        )

        weight_overrides = self.curriculum.get_reward_weight_overrides()
        reward_components = compute_reward(
            reward_state, self.cfg.reward, weight_overrides or None
        )
        reward = reward_components.total

        # Track success
        if reward_components.success_bonus > 0:
            self._already_succeeded = True

        # Update state for next step
        self._prev_soil_in_target = soil_in_target
        self._prev_action = action.copy()
        self._episode_reward += reward

        # Accumulate reward components for logging
        for key, val in reward_components.to_dict().items():
            self._reward_components_accum[key] = (
                self._reward_components_accum.get(key, 0.0) + val
            )

        # Check termination
        term_state = TerminationState(
            current_step=self._step_count,
            soil_in_target_ratio=soil_in_target,
            joint_pos=self._joint_pos,
            joint_limits_lower=np.array(self.arm_cfg.joint_limits.lower, dtype=np.float32),
            joint_limits_upper=np.array(self.arm_cfg.joint_limits.upper, dtype=np.float32),
            bucket_z=self._ee_pos[2],
            base_pos=self._base_pos,
            initial_base_pos=np.array(self.cfg.scene.robot_base_pos, dtype=np.float32),
        )
        term_result = check_termination(term_state, self.cfg.termination)

        # Build observation
        obs = self._get_observation()

        # Info dict
        info = {
            "step": self._step_count,
            "soil_in_target_ratio": soil_in_target,
            "bucket_load": bucket_load,
            "episode_reward": self._episode_reward,
            "termination_reason": term_result.reason.value,
        }
        info.update(reward_components.to_dict())

        # On episode end, include cumulative metrics
        if term_result.done:
            info["episode"] = {
                "r": self._episode_reward,
                "l": self._step_count,
                "soil_transfer_ratio": soil_in_target,
                "success": term_result.reason == TerminationReason.SUCCESS,
                "termination_reason": term_result.reason.value,
                # Embed cumulative reward components so the vec wrapper
                # (which only forwards info["episode"]) carries them through.
                **self._reward_components_accum,
            }

        return obs, float(reward), term_result.terminated, term_result.truncated, info

    def _get_observation(self) -> np.ndarray:
        """Build current observation vector."""
        # Robot state
        robot_state = RobotState(
            joint_pos=self._joint_pos,
            joint_vel=self._joint_vel,
            ee_pos=self._ee_pos,
            ee_orient=self._ee_orient,
            ee_lin_vel=self._ee_lin_vel,
            ee_ang_vel=self._ee_ang_vel,
        )

        # Bucket state
        bucket_load = self.scene.get_bucket_load(self._ee_pos)
        bucket_state = BucketState(
            load=min(bucket_load / 0.5, 1.0),
            contact_force=self._bucket_contact_force,
        )

        # Soil observation
        obs_cfg = self.cfg.observation
        soil_obs = SoilObservation()
        if obs_cfg.soil_obs_mode == "height_map":
            soil_obs.height_map = self.scene.get_soil_height_map(
                resolution=obs_cfg.height_map_resolution,
                grid_size=obs_cfg.height_map_size,
            )
        elif obs_cfg.soil_obs_mode == "centroid":
            soil_obs.centroid = self.scene.get_soil_centroid()
            soil_obs.spread = self.scene.get_soil_spread()
            soil_obs.in_target_ratio = self.scene.get_soil_in_target_ratio()

        target_pos = np.array(self.cfg.scene.target_center, dtype=np.float32)

        obs = build_observation(
            obs_cfg,
            robot_state,
            bucket_state,
            soil_obs,
            target_pos,
            self._prev_action,
        )

        # Apply DR observation noise
        obs = self.event_manager.apply_observation_noise(obs, self._dr_params)

        return obs

    def _update_ee_from_joints(self) -> None:
        """Update end-effector pose from joint positions.

        Uses a simplified forward kinematics approximation for standalone
        mode. In Isaac Lab mode, this comes from the simulation.
        """
        prev_ee_pos = self._ee_pos.copy()
        q = self._joint_pos

        if self.num_joints == 7:
            # Franka Panda approximate FK
            # DH-inspired simplified model: base height + 2-link planar arm
            base_height = 0.333
            l2 = 0.316  # upper arm
            l3 = 0.384  # forearm

            # Shoulder rotation (joint 0) controls azimuth
            # Shoulder lift (joint 1) + elbow (joint 3) control reach/height
            shoulder_angle = q[1]
            elbow_angle = q[3]

            # Horizontal reach and vertical offset from shoulder
            h_reach = l2 * np.cos(shoulder_angle) + l3 * np.cos(shoulder_angle + elbow_angle)
            v_offset = l2 * np.sin(shoulder_angle) + l3 * np.sin(shoulder_angle + elbow_angle)

            self._ee_pos[0] = h_reach * np.cos(q[0])
            self._ee_pos[1] = h_reach * np.sin(q[0])
            self._ee_pos[2] = base_height + v_offset

            # Wrist joints (5, 6) add small offsets
            self._ee_pos[0] += 0.05 * np.sin(q[4]) * np.cos(q[0])
            self._ee_pos[1] += 0.05 * np.sin(q[4]) * np.sin(q[0])

        elif self.num_joints == 6:
            # UR10 approximate FK
            l1, l2, l3 = 0.1273, 0.612, 0.5723
            h_reach = l2 * np.cos(q[1]) + l3 * np.cos(q[1] + q[2])
            v_offset = l2 * np.sin(q[1]) + l3 * np.sin(q[1] + q[2])
            self._ee_pos[0] = h_reach * np.cos(q[0])
            self._ee_pos[1] = h_reach * np.sin(q[0])
            self._ee_pos[2] = l1 + v_offset

        # Preserve the unclamped z so collision detection sees the true
        # below-ground depth; clamp only what gets exposed to the rest of
        # the env (observations, rewards relying on positive height, etc).
        self._ee_pos_z_raw = float(self._ee_pos[2])
        self._ee_pos[2] = max(self._ee_pos[2], 0.005)

        # Compute EE velocity from position change
        self._ee_lin_vel = (self._ee_pos - prev_ee_pos) / self.cfg.sim_dt

    def render(self):
        """Render the environment (placeholder for Isaac Lab integration)."""
        if self.render_mode == "human":
            pass  # Isaac Lab handles rendering
        elif self.render_mode == "rgb_array":
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def close(self):
        """Clean up resources."""
        pass


# ---------------------------------------------------------------------------
# Vectorized wrapper for parallel environments
# ---------------------------------------------------------------------------

def set_demo_trajectory_for_env(env, traj: np.ndarray, reset_prob: float = 0.5,
                                  snapshot: dict | None = None) -> None:
    """Register a demo trajectory for reverse-curriculum reset on a single env.

    If `snapshot` is provided (with keys 'joint', 'particles', 'particle_vel',
    'carried' from generate_bc_demos.py), reset additionally restores particle
    positions/velocities and the carried-particle dict at the sampled t0. This
    means an agent reset at e.g. t0=120 starts mid-lift with soil already in
    the bucket -- it only needs to learn "complete the carry+release" rather
    than the full descend+plunge+lift+swing+release sequence.
    """
    env._demo_trajectory = traj
    env._demo_reset_prob = reset_prob
    env._demo_snapshot = snapshot


class VecExcavationEnv:
    """Vectorized excavation environment for parallel training.

    In Isaac Lab mode, parallelism is handled by the simulator's GPU
    parallelism. This class provides a compatible interface for
    standalone mode using multiple env instances.
    """

    def __init__(self, cfg: ExcavationEnvCfg, num_envs: int | None = None):
        self.cfg = cfg
        self.num_envs = num_envs or cfg.scene.num_envs

        # For standalone mode: create individual environments
        self.envs = [ExcavationEnv(cfg) for _ in range(min(self.num_envs, 32))]

        self.num_obs = self.envs[0].observation_space.shape[0]
        self.num_actions = self.envs[0].action_space.shape[0]

    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reset all environments."""
        obs_list = []
        infos = {}
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
        obs_tensor = torch.tensor(np.stack(obs_list), dtype=torch.float32)
        return obs_tensor, infos

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step all environments.

        Args:
            actions: (num_envs, num_actions) tensor.

        Returns:
            Tuple of (obs, rewards, terminated, truncated, infos).
        """
        actions_np = actions.cpu().numpy()
        obs_list, rew_list, term_list, trunc_list = [], [], [], []
        infos = {"episode": []}

        for i, env in enumerate(self.envs):
            obs, rew, term, trunc, info = env.step(actions_np[i])
            if term or trunc:
                obs, _ = env.reset()
            obs_list.append(obs)
            rew_list.append(rew)
            term_list.append(term)
            trunc_list.append(trunc)
            if "episode" in info:
                infos["episode"].append(info["episode"])

        return (
            torch.tensor(np.stack(obs_list), dtype=torch.float32),
            torch.tensor(rew_list, dtype=torch.float32),
            torch.tensor(term_list, dtype=torch.bool),
            torch.tensor(trunc_list, dtype=torch.bool),
            infos,
        )

    def close(self):
        for env in self.envs:
            env.close()
