"""Evaluation script for trained excavation RL agents.

Loads a trained checkpoint and evaluates it across multiple episodes,
computing all quantitative metrics from Section 4.5.

Usage:
    python -m evaluation.evaluate --checkpoint results/checkpoints/best_model.pt
    python -m evaluation.evaluate --checkpoint results/checkpoints/best_model.pt --num-episodes 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.excavation_env import ExcavationEnv
from envs.excavation_env_cfg import ExcavationEnvCfg
from training.train import ActorCritic
from training.ppo_cfg import PPOCfg, NetworkCfg
from evaluation.metrics import (
    EpisodeMetrics,
    EvaluationReport,
    compute_episode_metrics,
    aggregate_metrics,
)


def evaluate_checkpoint(
    checkpoint_path: str,
    env_cfg: ExcavationEnvCfg | None = None,
    network_cfg: NetworkCfg | None = None,
    num_episodes: int = 100,
    seeds: list[int] | None = None,
    deterministic: bool = True,
    device: str = "cpu",
) -> EvaluationReport:
    """Evaluate a trained model checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint.
        env_cfg: Environment configuration (uses default if None).
        network_cfg: Network architecture config.
        num_episodes: Number of evaluation episodes.
        seeds: List of random seeds (one evaluation run per seed).
        deterministic: Whether to use deterministic actions.
        device: PyTorch device.

    Returns:
        EvaluationReport with aggregated metrics.
    """
    if seeds is None:
        seeds = [42]

    if env_cfg is None:
        env_cfg = ExcavationEnvCfg()

    # Load checkpoint once to infer network architecture
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Infer network config from checkpoint (saved by trainer) or fall back to argument
    if network_cfg is None:
        network_cfg = NetworkCfg(
            policy_hidden_dims=checkpoint.get("policy_hidden_dims", [256, 256, 128]),
            value_hidden_dims=checkpoint.get("value_hidden_dims", [256, 256, 128]),
            policy_activation=checkpoint.get("activation", "elu"),
        )

    all_episode_metrics = []

    for seed in seeds:
        env = ExcavationEnv(env_cfg)
        env_cfg_copy = env_cfg
        env_cfg_copy.seed = seed

        # Build model with correct architecture
        num_obs = checkpoint.get("num_obs", env.observation_space.shape[0])
        num_actions = checkpoint.get("num_actions", env.action_space.shape[0])
        model = ActorCritic(
            num_obs=num_obs,
            num_actions=num_actions,
            policy_hidden_dims=network_cfg.policy_hidden_dims,
            value_hidden_dims=network_cfg.value_hidden_dims,
            activation=network_cfg.policy_activation,
        ).to(device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        episodes_per_seed = num_episodes // len(seeds)

        for ep in range(episodes_per_seed):
            ep_rewards = []
            ep_actions = []
            ep_soil_ratios = []
            ep_reward_components: dict[str, list[float]] = {}

            obs, info = env.reset(seed=seed + ep)

            done = False
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action, _, _, _ = model.get_action(obs_tensor, deterministic=deterministic)
                action_np = action.squeeze(0).cpu().numpy()

                obs, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated

                ep_rewards.append(reward)
                ep_actions.append(action_np.copy())
                ep_soil_ratios.append(info.get("soil_in_target_ratio", 0.0))

                # Collect reward components
                for key in [
                    "reward/transfer", "reward/approach", "reward/load",
                    "reward/transport", "reward/smooth_penalty",
                ]:
                    if key in info:
                        if key not in ep_reward_components:
                            ep_reward_components[key] = []
                        ep_reward_components[key].append(info[key])

            # Compute episode metrics
            actions_arr = np.array(ep_actions)
            ep_info = info.get("episode", {})
            metrics = compute_episode_metrics(
                rewards=ep_rewards,
                actions=actions_arr,
                soil_ratios=ep_soil_ratios,
                success=ep_info.get("success", False),
                termination_reason=ep_info.get("termination_reason", "unknown"),
                reward_components=ep_reward_components,
            )
            all_episode_metrics.append(metrics)

        env.close()

    report = aggregate_metrics(all_episode_metrics, num_seeds=len(seeds))
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained excavation RL agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456], help="Random seeds")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic policy")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for metrics")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()

    print(f"Evaluating checkpoint: {args.checkpoint}")
    print(f"  Episodes: {args.num_episodes}, Seeds: {args.seeds}")

    report = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        num_episodes=args.num_episodes,
        seeds=args.seeds,
        deterministic=args.deterministic,
        device=args.device,
    )

    print()
    print(report.summary_table())

    # Save results
    if args.output:
        output_data = {
            "checkpoint": args.checkpoint,
            "num_episodes": report.num_episodes,
            "num_seeds": report.num_seeds,
            "mean_reward": report.mean_reward,
            "std_reward": report.std_reward,
            "mean_soil_transfer": report.mean_soil_transfer,
            "std_soil_transfer": report.std_soil_transfer,
            "success_rate": report.success_rate,
            "mean_action_smoothness": report.mean_action_smoothness,
            "mean_energy": report.mean_energy,
            "episode_rewards": report.episode_rewards,
            "episode_soil_ratios": report.episode_soil_ratios,
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
