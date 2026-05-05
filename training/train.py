"""Training entry point for excavation RL.

Supports two modes:
  1. Standalone mode: trains with simplified environment (for prototyping)
  2. Isaac Lab mode: trains with full physics simulation

Usage:
    # Standalone mode (default)
    python -m training.train --stage 0

    # Isaac Lab mode (requires Isaac Sim)
    python -m training.train --stage 1 --isaac-lab

    # Resume training
    python -m training.train --stage 2 --resume results/checkpoints/model_001000.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.excavation_env import ExcavationEnv, VecExcavationEnv
from envs.excavation_env_cfg import ExcavationEnvCfg
from training.ppo_cfg import PPOCfg, get_stage0_ppo_cfg, get_stage1_ppo_cfg, get_stage2_ppo_cfg
from training.callbacks import TrainingLogger, CheckpointCallback


# ---------------------------------------------------------------------------
# Observation Normalizer (RunningMeanStd)
# ---------------------------------------------------------------------------

class RunningMeanStd(nn.Module):
    """Tracks running mean & variance of observations across rollouts.

    Standard PPO obs normalization: (obs - mean) / sqrt(var + eps). Without
    this, dim-mixed obs (here: meters for ee_pos, radians for joints, m^2
    for soil_spread, dimensionless for in_target_ratio) feed network at
    wildly different scales, producing gradients that violate KL trust
    region. Stage-1 v1-v10 saw KL=7-13 with clip_fraction>0.9 directly
    because of this.

    State (mean/var/count) is saved/loaded with checkpoint via state_dict.
    """

    def __init__(self, num_obs: int, eps: float = 1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(num_obs))
        self.register_buffer("var", torch.ones(num_obs))
        self.register_buffer("count", torch.tensor(eps, dtype=torch.float32))
        self.register_buffer("frozen", torch.tensor(False))
        self.eps = eps

    def freeze(self) -> None:
        """Stop accumulating new observations into the running stats.

        Why we freeze: in v17 (3-seed × 500 iter) KL drifted 2.79 -> 6.54
        because obs_rms kept updating across iterations, slowly shifting the
        normalised obs distribution out from under the trained policy. Once
        ~10k samples are collected, additional updates contribute << 1% to
        the running mean/var but enough to destabilise PPO's trust region.
        """
        self.frozen.fill_(True)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if bool(self.frozen.item()):
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(tot_count)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)


# ---------------------------------------------------------------------------
# MLP Actor-Critic Network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """MLP actor-critic network for PPO.

    Policy outputs a Gaussian distribution over actions.
    Value function outputs a scalar state value estimate.
    """

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        policy_hidden_dims: list[int],
        value_hidden_dims: list[int],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        normalize_obs: bool = True,
    ):
        super().__init__()

        self.obs_rms = RunningMeanStd(num_obs) if normalize_obs else None

        activation_fn = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]

        # Policy network
        policy_layers = []
        in_dim = num_obs
        for hidden_dim in policy_hidden_dims:
            policy_layers.append(nn.Linear(in_dim, hidden_dim))
            policy_layers.append(activation_fn())
            in_dim = hidden_dim
        policy_layers.append(nn.Linear(in_dim, num_actions))
        self.policy_net = nn.Sequential(*policy_layers)

        # Value network
        value_layers = []
        in_dim = num_obs
        for hidden_dim in value_hidden_dims:
            value_layers.append(nn.Linear(in_dim, hidden_dim))
            value_layers.append(activation_fn())
            in_dim = hidden_dim
        value_layers.append(nn.Linear(in_dim, 1))
        self.value_net = nn.Sequential(*value_layers)

        # Action noise (learnable log std)
        self.log_std = nn.Parameter(
            torch.ones(num_actions) * np.log(init_noise_std)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor):
        """Forward pass — returns action mean and value.

        Obs is normalized via running mean/var (if enabled). Caller must call
        obs_rms.update() on raw rollout obs separately to keep stats current.
        """
        if self.obs_rms is not None:
            obs = self.obs_rms.normalize(obs)
        action_mean = self.policy_net(obs)
        value = self.value_net(obs)
        return action_mean, value.squeeze(-1)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample action from policy.

        Returns:
            Tuple of (action, log_prob, value, action_mean).
        """
        action_mean, value = self.forward(obs)
        std = torch.exp(self.log_std)

        if deterministic:
            action = action_mean
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            dist = torch.distributions.Normal(action_mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value, action_mean

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate log probability and value for given obs-action pairs.

        Returns:
            Tuple of (log_prob, value, entropy).
        """
        action_mean, value = self.forward(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, std)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, value, entropy


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Buffer to store rollout data for PPO updates."""

    def __init__(self, num_envs: int, num_steps: int, num_obs: int, num_actions: int, device: str):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device

        self.observations = torch.zeros(num_steps, num_envs, num_obs, device=device)
        self.actions = torch.zeros(num_steps, num_envs, num_actions, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.advantages = torch.zeros(num_steps, num_envs, device=device)
        self.returns = torch.zeros(num_steps, num_envs, device=device)

        self.step = 0

    def add(self, obs, actions, rewards, dones, values, log_probs):
        self.observations[self.step] = obs
        self.actions[self.step] = actions
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.values[self.step] = values
        self.log_probs[self.step] = log_probs
        self.step += 1

    def compute_returns_and_advantages(self, last_values: torch.Tensor, gamma: float, lam: float):
        """Compute GAE advantages and discounted returns."""
        last_gae = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae = delta + gamma * lam * next_non_terminal * last_gae

        self.returns = self.advantages + self.values

    def get_batches(self, num_mini_batches: int):
        """Yield mini-batches for PPO update."""
        batch_size = self.num_envs * self.num_steps
        indices = torch.randperm(batch_size, device=self.device)
        mini_batch_size = batch_size // num_mini_batches

        obs_flat = self.observations.reshape(-1, self.observations.shape[-1])
        actions_flat = self.actions.reshape(-1, self.actions.shape[-1])
        log_probs_flat = self.log_probs.reshape(-1)
        advantages_flat = self.advantages.reshape(-1)
        returns_flat = self.returns.reshape(-1)
        values_flat = self.values.reshape(-1)

        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            batch_indices = indices[start:end]

            yield {
                "obs": obs_flat[batch_indices],
                "actions": actions_flat[batch_indices],
                "old_log_probs": log_probs_flat[batch_indices],
                "advantages": advantages_flat[batch_indices],
                "returns": returns_flat[batch_indices],
                "old_values": values_flat[batch_indices],
            }

    def reset(self):
        self.step = 0


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """PPO training loop for the excavation environment.

    Implements the complete training pipeline:
    1. Collect rollouts from vectorized environments
    2. Compute advantages using GAE
    3. Update policy/value networks with clipped PPO objective
    4. Log metrics and save checkpoints
    """

    def __init__(
        self,
        env: VecExcavationEnv,
        ppo_cfg: PPOCfg,
        device: str = "cuda:0",
        experiment_name: str = "excavation",
        wandb_cfg: dict | None = None,
    ):
        self.env = env
        self.cfg = ppo_cfg
        self.device = device

        # Actor-Critic network
        self.actor_critic = ActorCritic(
            num_obs=env.num_obs,
            num_actions=env.num_actions,
            policy_hidden_dims=ppo_cfg.network.policy_hidden_dims,
            value_hidden_dims=ppo_cfg.network.value_hidden_dims,
            activation=ppo_cfg.network.policy_activation,
            init_noise_std=ppo_cfg.network.init_noise_std,
        ).to(device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=ppo_cfg.learning_rate
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(
            num_envs=len(env.envs),
            num_steps=ppo_cfg.num_steps_per_env,
            num_obs=env.num_obs,
            num_actions=env.num_actions,
            device=device,
        )

        # Logging
        self.logger = TrainingLogger(
            ppo_cfg.log_dir, experiment_name, wandb_cfg=wandb_cfg
        )
        self.checkpoint_cb = CheckpointCallback(
            ppo_cfg.checkpoint_dir, ppo_cfg.save_interval
        )

        # Training state
        self.current_lr = ppo_cfg.learning_rate
        self.iteration = 0

        # V34: Running stats over GAE returns. Used to scale the value loss into
        # O(1) units so the value head can fit large-magnitude returns without
        # the value-loss gradient dominating clip_grad_norm and starving the
        # policy head. Episodic returns logged by callbacks stay in raw units;
        # only the inside-PPO value-loss target is normalised. See V34 entry in
        # docs/stage1_5_tuning_log.md.
        self.return_rms = RunningMeanStd(num_obs=1).to(device)

    def train(self, resume_path: str | None = None) -> None:
        """Run the full training loop.

        Args:
            resume_path: Optional path to checkpoint to resume from.
        """
        if resume_path:
            self._load_checkpoint(resume_path)

        # Initial reset
        obs, _ = self.env.reset()
        obs = obs.to(self.device)

        print(f"Starting training: {self.cfg.max_iterations} iterations")
        print(f"  Envs: {len(self.env.envs)}, Steps/env: {self.cfg.num_steps_per_env}")
        print(f"  Batch size: {self.cfg.batch_size}, Mini-batch: {self.cfg.mini_batch_size}")
        print(f"  Obs dim: {self.env.num_obs}, Action dim: {self.env.num_actions}")
        print()

        for iteration in range(self.iteration, self.cfg.max_iterations):
            self.iteration = iteration
            start_time = time.time()

            # Collect rollouts
            self.buffer.reset()
            for step in range(self.cfg.num_steps_per_env):
                with torch.no_grad():
                    if self.actor_critic.obs_rms is not None:
                        self.actor_critic.obs_rms.update(obs)
                    actions, log_probs, values, _ = self.actor_critic.get_action(obs)

                # Clip actions to [-1, 1]
                actions_clipped = torch.clamp(actions, -1.0, 1.0)

                # Environment step
                next_obs, rewards, terminated, truncated, infos = self.env.step(actions_clipped)
                dones = terminated | truncated

                # Store transition
                self.buffer.add(
                    obs,
                    actions_clipped,
                    rewards.to(self.device),
                    dones.float().to(self.device),
                    values,
                    log_probs,
                )

                obs = next_obs.to(self.device)

                # Log episodes
                if infos.get("episode"):
                    for ep_info in infos["episode"]:
                        self.logger.log_episode({"episode": ep_info})

            # Compute returns and advantages
            with torch.no_grad():
                _, last_values = self.actor_critic.forward(obs)
            self.buffer.compute_returns_and_advantages(
                last_values, self.cfg.gamma, self.cfg.lam
            )
            # V34: maintain running stats of GAE returns for value-loss scaling.
            self.return_rms.update(self.buffer.returns.reshape(-1, 1))

            # NOTE on obs_rms freezing: v18 tried freezing at iter 50 to stop
            # the v17 KL drift (2.79->6.54). Result: zero benefit on seeds
            # 7/42 (KL still drifted to ~7), and seed 123 KL exploded to
            # 22820 in early iters (before freeze kicked in) — the drift
            # wasn't caused by obs_rms updates after all. Freeze removed.

            # PPO update
            ppo_metrics = self._ppo_update()

            # Adaptive learning rate
            if self.cfg.schedule == "adaptive" and "approx_kl" in ppo_metrics:
                self._adapt_lr(ppo_metrics["approx_kl"])

            # Logging
            elapsed = time.time() - start_time
            if iteration % self.cfg.log_interval == 0:
                ppo_metrics["learning_rate"] = self.current_lr
                ppo_metrics["fps"] = (
                    self.cfg.num_steps_per_env * len(self.env.envs) / elapsed
                )
                self.logger.log_iteration(iteration, ppo_metrics)

                mean_reward = ppo_metrics.get("mean_reward", 0)
                print(
                    f"Iter {iteration:5d} | "
                    f"Reward: {mean_reward:8.2f} | "
                    f"KL: {ppo_metrics.get('approx_kl', 0):.4f} | "
                    f"Entropy: {ppo_metrics.get('entropy', 0):.4f} | "
                    f"FPS: {ppo_metrics.get('fps', 0):.0f}"
                )

            # Checkpoint
            if self.checkpoint_cb.should_save(iteration):
                is_best = self.checkpoint_cb.update_best(
                    ppo_metrics.get("mean_reward", -float("inf"))
                )
                self._save_checkpoint(iteration, is_best)

        # Final save
        self._save_checkpoint(self.cfg.max_iterations, is_best=False)
        self.logger.close()
        print("Training complete.")

    def _ppo_update(self) -> dict[str, float]:
        """Perform PPO policy and value network update.

        Returns:
            Dict of training metrics.
        """
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        total_clip_frac = 0
        num_updates = 0

        # Normalize advantages
        advantages = self.buffer.advantages.reshape(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages.reshape(
            self.buffer.num_steps, self.buffer.num_envs
        )

        # KL-based early stopping: break out when policy has drifted too far
        # from the rollout policy. Without this, all 5 epochs × num_mini_batches
        # gradient steps run regardless of KL — observed in stage 1 v3/v4 to
        # produce KL=8-9 with clip_fraction=0.97 (97% of updates wasted), and
        # in v1 with full reward to produce KL=14 with destructive updates.
        # Tightened 4x -> 2x desired_kl after v17 KL drift analysis: with
        # 4x at desired=0.02, threshold=0.08 was too loose -- KL drifted
        # 2.79 -> 6.54 over 500 iter. With 2x (=0.04) we stop the very
        # first mini-batch that overshoots, preventing accumulated drift.
        kl_stop_threshold = self.cfg.desired_kl * 2.0
        early_stopped = False

        for epoch in range(self.cfg.num_epochs):
            if early_stopped:
                break
            for batch in self.buffer.get_batches(self.cfg.num_mini_batches):
                log_probs, values, entropy = self.actor_critic.evaluate_actions(
                    batch["obs"], batch["actions"]
                )

                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(log_probs - batch["old_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = (
                    torch.clamp(ratio, 1 - self.cfg.clip_param, 1 + self.cfg.clip_param)
                    * batch["advantages"]
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (optionally clipped)
                # V34: scale predictions, targets, and old values by the running
                # std of returns so the value loss is O(1) regardless of the
                # reward scale (Andrychowicz 2020; "37 Implementation Details
                # of PPO" #5). The clip_param then bounds the *relative*
                # value-prediction change rather than an absolute step in raw
                # return units (which is meaningless for O(100) returns).
                return_std = torch.sqrt(self.return_rms.var + 1e-8)
                values_n = values / return_std
                returns_n = batch["returns"] / return_std
                old_values_n = batch["old_values"] / return_std
                if self.cfg.use_clipped_value_loss:
                    value_pred_clipped_n = old_values_n + torch.clamp(
                        values_n - old_values_n,
                        -self.cfg.clip_param,
                        self.cfg.clip_param,
                    )
                    value_loss1 = (values_n - returns_n) ** 2
                    value_loss2 = (value_pred_clipped_n - returns_n) ** 2
                    value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                else:
                    value_loss = 0.5 * ((values_n - returns_n) ** 2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Optional BC anchor: KL(current || frozen_BC) on this mini-batch
                # observations. Computed analytically for diagonal Gaussians:
                #   KL = sum_i [ log(s_BC/s_curr) + (s_curr^2 + (mu_curr - mu_BC)^2)/(2*s_BC^2) - 0.5 ]
                bc_anchor_loss = torch.tensor(0.0, device=self.device)
                if getattr(self, "_bc_anchor_coef", 0.0) > 0 and getattr(self, "_bc_actor_critic", None) is not None:
                    with torch.no_grad():
                        bc_mean, _ = self._bc_actor_critic.forward(batch["obs"])
                        bc_log_std = self._bc_actor_critic.log_std
                    cur_mean, _ = self.actor_critic.forward(batch["obs"])
                    cur_log_std = self.actor_critic.log_std
                    var_cur = torch.exp(2 * cur_log_std)
                    var_bc = torch.exp(2 * bc_log_std)
                    kl_per_dim = (bc_log_std - cur_log_std
                                   + (var_cur + (cur_mean - bc_mean) ** 2) / (2 * var_bc) - 0.5)
                    bc_anchor_loss = self._bc_anchor_coef * kl_per_dim.sum(dim=-1).mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.cfg.value_loss_coeff * value_loss
                    + self.cfg.entropy_coeff * entropy_loss
                    + bc_anchor_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

                # Metrics
                with torch.no_grad():
                    approx_kl = (batch["old_log_probs"] - log_probs).mean().item()
                    clip_frac = (
                        (torch.abs(ratio - 1.0) > self.cfg.clip_param).float().mean().item()
                    )

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_kl += approx_kl
                total_clip_frac += clip_frac
                num_updates += 1

                if approx_kl > kl_stop_threshold:
                    early_stopped = True
                    break

        n = max(num_updates, 1)
        mean_reward = self.buffer.rewards.sum(dim=0).mean().item()

        # Explained variance
        with torch.no_grad():
            returns_flat = self.buffer.returns.reshape(-1)
            values_flat = self.buffer.values.reshape(-1)
            var_returns = returns_flat.var()
            explained_var = (
                1.0 - (returns_flat - values_flat).var() / (var_returns + 1e-8)
            ).item()

        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "approx_kl": total_kl / n,
            "clip_fraction": total_clip_frac / n,
            "explained_variance": explained_var,
            "mean_reward": mean_reward,
        }

    def _adapt_lr(self, kl: float) -> None:
        """Adapt learning rate based on KL divergence.

        Floor 1e-5: low enough that adaptive can react meaningfully to early
        KL spikes (5e-5 floor was hit on iter 1 and never recovered), but
        well above the original 1e-6 that froze training entirely.
        """
        if kl > self.cfg.desired_kl * 2.0:
            self.current_lr = max(self.current_lr / 1.5, 1e-5)
        elif kl < self.cfg.desired_kl / 2.0:
            self.current_lr = min(self.current_lr * 1.5, 1e-2)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.current_lr

    def _save_checkpoint(self, iteration: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "iteration": iteration,
            "model_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "learning_rate": self.current_lr,
            "num_obs": self.env.num_obs,
            "num_actions": self.env.num_actions,
            "policy_hidden_dims": self.cfg.network.policy_hidden_dims,
            "value_hidden_dims": self.cfg.network.value_hidden_dims,
            "activation": self.cfg.network.policy_activation,
        }

        path = self.checkpoint_cb.get_save_path(iteration, is_best=False)
        torch.save(checkpoint, path)

        if is_best:
            best_path = self.checkpoint_cb.get_save_path(iteration, is_best=True)
            torch.save(checkpoint, best_path)
            print(f"  New best model saved: reward = {self.checkpoint_cb._best_reward:.2f}")

    def _load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_lr = checkpoint.get("learning_rate", self.cfg.learning_rate)
        self.iteration = checkpoint.get("iteration", 0)
        print(f"Resumed from checkpoint: {path} (iteration {self.iteration})")

    def bc_pretrain(self, data_path: str, n_steps: int, batch_size: int = 256,
                    lr: float = 3e-4) -> None:
        """Behavior-cloning pretraining: MSE loss between policy mean and
        demo actions. Also warms up obs_rms with demo observations.

        Bypasses the random-policy plateau (~2% transfer) by initialising the
        policy near a hand-coded scoop trajectory (~22% demo transfer ratio).
        PPO then refines from this starting point.
        """
        import numpy as np
        data = np.load(data_path)
        demo_obs = torch.tensor(data["obs"], dtype=torch.float32, device=self.device)
        demo_act = torch.tensor(data["actions"], dtype=torch.float32, device=self.device)
        N = demo_obs.shape[0]
        print(f"\n=== BC pretrain ===")
        print(f"Loaded {N} (obs, action) pairs from {data_path}")
        print(f"Demo transfer ratio mean: {data['transfer_ratios'].mean():.3f}")
        print(f"Running {n_steps} MSE updates @ batch_size={batch_size} lr={lr}")

        # Warm up obs_rms with all demo obs
        if self.actor_critic.obs_rms is not None:
            for chunk in torch.split(demo_obs, 1024):
                self.actor_critic.obs_rms.update(chunk)
            print(f"obs_rms warmed up: mean range [{self.actor_critic.obs_rms.mean.min():.3f}, "
                  f"{self.actor_critic.obs_rms.mean.max():.3f}]")

        bc_optim = optim.Adam(self.actor_critic.parameters(), lr=lr)
        log_interval = max(1, n_steps // 10)
        for step in range(n_steps):
            idx = torch.randint(0, N, (batch_size,), device=self.device)
            obs_batch = demo_obs[idx]
            act_batch = demo_act[idx]
            # Forward through policy net (action mean)
            if self.actor_critic.obs_rms is not None:
                obs_norm = self.actor_critic.obs_rms.normalize(obs_batch)
            else:
                obs_norm = obs_batch
            pred_mean = self.actor_critic.policy_net(obs_norm)
            loss = ((pred_mean - act_batch) ** 2).mean()
            bc_optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 1.0)
            bc_optim.step()
            if step % log_interval == 0:
                print(f"  bc step {step:>5d}: mse_loss={loss.item():.4f}")
        print(f"BC pretrain complete.\n")

        # Snapshot BC policy weights for the optional KL-anchor in PPO.
        # We freeze a deep copy so the anchor target is fixed throughout PPO.
        import copy
        self._bc_actor_critic = copy.deepcopy(self.actor_critic)
        for p in self._bc_actor_critic.parameters():
            p.requires_grad = False
        self._bc_actor_critic.eval()
        print(f"Snapshotted frozen BC policy for KL-anchor (use --bc-anchor > 0 to activate)")

        # NOTE: do NOT freeze obs_rms here. The v28b ablation tested
        # post-BC freeze and produced catastrophic KL=193k (PPO inevitably
        # visits states the BC obs distribution did not cover; the frozen
        # normalizer maps those into ±10σ activations and the value head
        # explodes). The V28b entry's Decision section mandates leaving
        # obs_rms.update() running throughout PPO. See
        # docs/stage1_5_tuning_log.md V28b and V33 entries.

        # Evaluate BC-only policy (deterministic, no PPO yet) to know the
        # warm-start ceiling. If PPO degrades transfer below this, use a
        # KL-anchor or shorter PPO horizon.
        self._eval_deterministic("BC-only (no PPO)", n_episodes=4, verbose=True)

    def _eval_deterministic(self, label: str, n_episodes: int = 4, verbose: bool = False) -> dict[str, float]:
        """Run deterministic policy across env episodes, report transfer_ratio."""
        import numpy as np
        transfers = []
        loads_max = []
        for ep in range(n_episodes):
            reset_out = self.env.reset()
            obs_arr = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            ep_load = 0.0
            for t in range(self.env.envs[0].cfg.termination.max_episode_length):
                with torch.no_grad():
                    obs_t = obs_arr.to(self.device) if hasattr(obs_arr, "to") else torch.tensor(np.asarray(obs_arr), dtype=torch.float32, device=self.device)
                    action_mean, _ = self.actor_critic.forward(obs_t)
                    actions = torch.clamp(action_mean, -1.0, 1.0)
                step_out = self.env.step(actions)
                obs_arr = step_out[0]
                terminated = step_out[2] if len(step_out) >= 5 else step_out[2]
                truncated = step_out[3] if len(step_out) >= 5 else torch.zeros_like(terminated)
                if verbose and ep == 0 and t in [0, 30, 50, 70, 100, 150, 199]:
                    e0 = self.env.envs[0]
                    bp = e0._ee_pos
                    bl = e0.scene.get_bucket_load(bp)
                    a = actions[0].cpu().numpy()
                    print(f"    t={t:3d}: EE=[{bp[0]:.3f},{bp[1]:.3f},{bp[2]:.3f}] load={bl:.4f} act_norm={np.linalg.norm(a):.3f} act_mean_abs={np.abs(a).mean():.3f}")
                # Track per-env max bucket load
                for env in self.env.envs:
                    bp = env._ee_pos
                    ep_load = max(ep_load, env.scene.get_bucket_load(bp))
                if terminated.any() or truncated.any():
                    pass  # episodes can vary in length; just use max steps
            # Average transfer across vec envs
            tr = np.mean([env.scene.get_soil_in_target_ratio() for env in self.env.envs])
            transfers.append(tr)
            loads_max.append(ep_load)
        result = {
            "transfer_ratio_mean": float(np.mean(transfers)),
            "transfer_ratio_max": float(np.max(transfers)),
            "max_load_seen": float(np.max(loads_max)),
        }
        print(f"  [{label}] transfer_ratio across {n_episodes} eps: "
              f"mean={result['transfer_ratio_mean']:.4f}, max={result['transfer_ratio_max']:.4f}, "
              f"max_load={result['max_load_seen']:.4f}")
        return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train excavation RL agent")
    parser.add_argument(
        "--stage", type=int, default=0, choices=[0, 1, 2],
        help="Training stage (0=rigid body, 1=particles, 2=full)",
    )
    parser.add_argument("--num-envs", type=int, default=None, help="Number of parallel environments")
    parser.add_argument("--max-iterations", type=int, default=None, help="Maximum training iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--bc-data", type=str, default=None, help="BC demo .npz path; if set, BC pretrain before PPO")
    parser.add_argument("--bc-steps", type=int, default=2000, help="BC gradient steps")
    parser.add_argument("--bc-lr", type=float, default=3e-4, help="BC learning rate")
    parser.add_argument("--reverse-curriculum", action="store_true",
                        help="Reset envs at random points along the demo trajectory (reverse curriculum)")
    parser.add_argument("--reverse-curriculum-prob", type=float, default=0.5,
                        help="Per-episode probability of starting from a demo state (vs. default reset)")
    parser.add_argument("--snapshot-data", type=str, default=None,
                        help="Path to scoop_snapshot.npz; if set with --reverse-curriculum, particle state at t0 is also restored")
    parser.add_argument("--bc-anchor", type=float, default=0.0,
                        help="If > 0, add bc_anchor * KL(policy || frozen_BC_policy) to PPO loss to prevent drift away from BC")
    parser.add_argument("--experiment-name", type=str, default=None, help="Experiment name for logging")
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="excavation-rl", help="wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity (team/user)")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="wandb run name (defaults to experiment-name)")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"],
                        help="wandb mode")
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None, help="wandb tags")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Per-run subdir for checkpoints/logs (avoids clobber when running multiple jobs in parallel)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Get stage-specific configs
    ppo_cfgs = {0: get_stage0_ppo_cfg, 1: get_stage1_ppo_cfg, 2: get_stage2_ppo_cfg}
    ppo_cfg = ppo_cfgs[args.stage]()

    if args.num_envs is not None:
        ppo_cfg.num_envs = args.num_envs
    if args.max_iterations is not None:
        ppo_cfg.max_iterations = args.max_iterations
    ppo_cfg.seed = args.seed

    # Isolate per-run output paths so parallel jobs don't clobber each other.
    run_id = args.run_id or args.wandb_run_name or f"stage{args.stage}_seed{args.seed}"
    ppo_cfg.checkpoint_dir = os.path.join(ppo_cfg.checkpoint_dir, run_id)
    ppo_cfg.log_dir = os.path.join(ppo_cfg.log_dir, run_id)

    # Environment config
    env_cfg = ExcavationEnvCfg(
        seed=args.seed,
        device=args.device,
    )
    env_cfg.scene.use_rigid_body_proxy = (args.stage == 0)
    env_cfg.scene.num_envs = ppo_cfg.num_envs

    # Stage 0 uses a SIMPLIFIED reward to demonstrate that the training pipeline
    # (PPO + custom env + wandb + checkpointing) converges. The proxy scene's
    # heuristic soil/load/collision signals are too coarse to support the full
    # multi-component reward without exploits, so Stage 0 keeps only a dense
    # approach reward + small regularization. Stage 1+ enables the full reward
    # once particle physics constrains the action space.
    if args.stage == 0:
        # CRITICAL: disable curriculum first. CurriculumManager.get_reward_weight_overrides()
        # silently overrides every weight in compute_reward() if a stage is active,
        # which previously made our stage-0 reward overrides no-ops (we saw stray
        # transfer=10 spikes whenever the proxy scene's soil_in_target heuristic
        # fired, despite soil_transfer being "set" to 0).
        env_cfg.curriculum.enabled = False

        rw = env_cfg.reward.weights
        rw.soil_transfer = 0.0
        rw.bucket_load = 0.0
        rw.transport = 0.0
        rw.collision = 0.0
        rw.joint_limit = 0.0
        rw.time_penalty = -0.01
        rw.action_smoothness = -0.01
        rw.approach = 1.0
        rw.approach_alpha = 2.0
        rw.dig = 0.0  # stage 0 uses proxy scene; no heap to dig into
        env_cfg.reward.success_bonus = 0.0
        # Also raise success_threshold so the proxy scene's noisy
        # get_soil_in_target_ratio() doesn't accidentally trigger SUCCESS
        # termination (which truncates episodes and adds variance to value loss).
        env_cfg.reward.success_threshold = 1.01  # unreachable

    if args.stage == 1:
        # Curriculum trap (same as stage 0): disable.
        env_cfg.curriculum.enabled = False

        # Stage 1 reward (v6): POSITIVES ONLY. v5 showed that even the softened
        # collision (-2) penalty fires per-step on a fraction of episodes and
        # produces bimodal episode returns (-335 ± 379). The bimodality breaks
        # advantage normalization (the std no longer represents the spread of a
        # unimodal distribution), so PPO's policy gradient sees opposite signals
        # each mini-batch -> KL locks at ~11, no learning. Strip the binary
        # penalties to get a unimodal reward; re-introduce later as graduated
        # signals (penetration depth, joint-limit margin) once base task works.
        rw = env_cfg.reward.weights
        rw.collision = 0.0
        rw.joint_limit = 0.0
        # action_smoothness lowered 0.01 -> 0.001 after v17 reward decomposition
        # showed smooth_penalty mean -10/iter, comparable in magnitude to the
        # positive components. The aggressive smoothness penalty was suppressing
        # the rapid joint motions a real scoop maneuver requires.
        rw.action_smoothness = -0.001
        rw.time_penalty = -0.01
        rw.approach = 1.0
        rw.approach_alpha = 2.0
        rw.bucket_load = 10.0  # boosted 2 -> 10: dominant signal that the
                               # agent has scooped something. v14 had dig=2
                               # creating a "hover above heap" attractor with
                               # bucket_load=2 too weak to break out of it.
        rw.transport = 3.0
        rw.soil_transfer = 10.0
        rw.dig = 0.5           # reduced 2 -> 0.5: kept as initial scaffolding
                               # but no longer dominates. once bucket_load
                               # fires, that becomes the much larger signal.
        rw.dig_alpha = 8.0
        env_cfg.reward.success_bonus = 5.0
        env_cfg.reward.success_threshold = 0.3

    # For standalone mode, limit parallel envs
    num_standalone_envs = min(ppo_cfg.num_envs, 32)

    # Create environment
    env = VecExcavationEnv(env_cfg, num_envs=num_standalone_envs)

    # Experiment name
    experiment_name = args.experiment_name or f"stage{args.stage}_seed{args.seed}"

    # Build wandb config (None disables wandb)
    wandb_cfg = None
    if args.wandb:
        wandb_config = {
            "stage": args.stage,
            "seed": args.seed,
            "num_envs": ppo_cfg.num_envs,
            "num_steps_per_env": ppo_cfg.num_steps_per_env,
            "max_iterations": ppo_cfg.max_iterations,
            "learning_rate": ppo_cfg.learning_rate,
            "gamma": ppo_cfg.gamma,
            "lam": ppo_cfg.lam,
            "clip_param": ppo_cfg.clip_param,
            "entropy_coeff": ppo_cfg.entropy_coeff,
            "value_loss_coeff": ppo_cfg.value_loss_coeff,
            "num_epochs": ppo_cfg.num_epochs,
            "num_mini_batches": ppo_cfg.num_mini_batches,
            "schedule": ppo_cfg.schedule,
            "desired_kl": ppo_cfg.desired_kl,
            "policy_hidden_dims": ppo_cfg.network.policy_hidden_dims,
            "value_hidden_dims": ppo_cfg.network.value_hidden_dims,
            "init_noise_std": ppo_cfg.network.init_noise_std,
        }
        wandb_cfg = {
            "init_kwargs": {
                "project": args.wandb_project,
                "entity": args.wandb_entity,
                "name": args.wandb_run_name or experiment_name,
                "mode": args.wandb_mode,
                "tags": args.wandb_tags,
                "config": wandb_config,
                "sync_tensorboard": False,
            }
        }

    # Create trainer
    trainer = PPOTrainer(
        env=env,
        ppo_cfg=ppo_cfg,
        device=args.device if torch.cuda.is_available() else "cpu",
        experiment_name=experiment_name,
        wandb_cfg=wandb_cfg,
    )

    # Optional reverse-curriculum: register demo trajectory on each env
    if args.reverse_curriculum:
        from scripts.scoop_demo import build_scoop_trajectory
        from envs.excavation_env import set_demo_trajectory_for_env
        default_q = np.array(env.envs[0].arm_cfg.default_joint_pos, dtype=np.float32)
        traj = build_scoop_trajectory(default_q,
                                       heap_center=env_cfg.scene.soil_heap_center,
                                       target_center=env_cfg.scene.target_center,
                                       n_steps=200)
        snapshot = None
        if args.snapshot_data:
            snap_npz = np.load(args.snapshot_data)
            snapshot = {k: snap_npz[k] for k in snap_npz.files}
            print(f"Loaded scoop snapshot from {args.snapshot_data}: "
                  f"steps={snapshot['joint'].shape[0]}, particles={snapshot['particles'].shape[1]}, "
                  f"transfer={snapshot.get('transfer_ratio', [-1.0])[0]:.3f}")
        for e in env.envs:
            set_demo_trajectory_for_env(e, traj, reset_prob=args.reverse_curriculum_prob, snapshot=snapshot)
        print(f"Reverse curriculum enabled: {len(env.envs)} envs, prob={args.reverse_curriculum_prob}, traj_len={len(traj)}, particle_replay={snapshot is not None}")

    # BC pretraining (optional)
    if args.bc_data:
        trainer.bc_pretrain(args.bc_data, n_steps=args.bc_steps, lr=args.bc_lr)
        if args.bc_anchor > 0:
            trainer._bc_anchor_coef = args.bc_anchor
            print(f"BC KL-anchor active: coef={args.bc_anchor}")

    # Train
    trainer.train(resume_path=args.resume)

    env.close()


if __name__ == "__main__":
    main()
