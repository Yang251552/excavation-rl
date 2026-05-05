"""Video recording utility for trained excavation agents.

Records evaluation episodes as video files for qualitative analysis
and presentation. Supports side-by-side comparison videos.

Usage:
    python -m evaluation.record_video --checkpoint results/checkpoints/best_model.pt
    python -m evaluation.record_video --checkpoint results/checkpoints/best_model.pt --episodes 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import imageio

    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

from envs.excavation_env import ExcavationEnv
from envs.excavation_env_cfg import ExcavationEnvCfg
from training.train import ActorCritic
from training.ppo_cfg import NetworkCfg


def record_episode(
    env: ExcavationEnv,
    model: ActorCritic,
    device: str = "cpu",
    deterministic: bool = True,
    max_steps: int = 500,
) -> tuple[list[np.ndarray], dict]:
    """Record a single episode.

    Args:
        env: Excavation environment (must support render_mode="rgb_array").
        model: Trained actor-critic model.
        device: PyTorch device.
        deterministic: Whether to use deterministic actions.
        max_steps: Maximum episode length.

    Returns:
        Tuple of (list of frames, episode info dict).
    """
    frames = []
    obs, info = env.reset()

    total_reward = 0.0
    step = 0
    done = False

    while not done and step < max_steps:
        # Render frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        else:
            # Generate a simple status frame if no renderer available
            frames.append(_generate_status_frame(obs, step, total_reward, info))

        # Get action
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _, _, _ = model.get_action(obs_tensor, deterministic=deterministic)
        action_np = action.squeeze(0).cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action_np)
        total_reward += reward
        done = terminated or truncated
        step += 1

    episode_info = {
        "total_reward": total_reward,
        "length": step,
        "soil_transfer_ratio": info.get("soil_in_target_ratio", 0),
        "termination_reason": info.get("termination_reason", "unknown"),
        "success": info.get("episode", {}).get("success", False),
    }

    return frames, episode_info


def _generate_status_frame(
    obs: np.ndarray, step: int, reward: float, info: dict
) -> np.ndarray:
    """Generate a simple text-based status frame when no renderer is available."""
    frame = np.ones((240, 320, 3), dtype=np.uint8) * 30  # dark background

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(3.2, 2.4), dpi=100)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        soil_ratio = info.get("soil_in_target_ratio", 0)
        text = (
            f"Step: {step}\n"
            f"Reward: {reward:.2f}\n"
            f"Soil in target: {soil_ratio:.1%}\n"
            f"EE pos: [{obs[14]:.2f}, {obs[15]:.2f}, {obs[16]:.2f}]"
        )
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", color="white", fontfamily="monospace")

        fig.patch.set_facecolor("#1e1e1e")
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)[:, :, :3].copy()
        plt.close(fig)
    except Exception:
        pass  # Return default dark frame

    return frame


def save_video(
    frames: list[np.ndarray],
    output_path: str,
    fps: int = 30,
) -> None:
    """Save frames as an MP4 video file.

    Args:
        frames: List of (H, W, 3) uint8 frames.
        output_path: Output file path.
        fps: Frames per second.
    """
    if not HAS_IMAGEIO:
        print("imageio not installed, cannot save video. Install with: pip install imageio imageio-ffmpeg")
        return

    if not frames:
        print("No frames to save")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    print(f"Video saved: {output_path} ({len(frames)} frames at {fps} fps)")


def record_best_median_worst(
    checkpoint_path: str,
    output_dir: str = "results/videos",
    num_candidate_episodes: int = 20,
    device: str = "cpu",
) -> None:
    """Record best, median, and worst episodes for qualitative analysis.

    Runs multiple episodes and selects the best, median, and worst
    by total reward for video recording.

    Args:
        checkpoint_path: Path to model checkpoint.
        output_dir: Directory to save videos.
        num_candidate_episodes: Number of episodes to evaluate for selection.
        device: PyTorch device.
    """
    cfg = ExcavationEnvCfg()
    env = ExcavationEnv(cfg, render_mode="rgb_array")

    # Load checkpoint and infer network architecture
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy_dims = checkpoint.get("policy_hidden_dims", [256, 256, 128])
    value_dims = checkpoint.get("value_hidden_dims", [256, 256, 128])
    activation = checkpoint.get("activation", "elu")
    num_obs = checkpoint.get("num_obs", env.observation_space.shape[0])
    num_actions = checkpoint.get("num_actions", env.action_space.shape[0])

    model = ActorCritic(
        num_obs=num_obs,
        num_actions=num_actions,
        policy_hidden_dims=policy_dims,
        value_hidden_dims=value_dims,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Run candidate episodes
    episodes = []
    for i in range(num_candidate_episodes):
        frames, info = record_episode(env, model, device=device, deterministic=True)
        episodes.append((frames, info, info["total_reward"]))
        print(f"  Episode {i + 1}/{num_candidate_episodes}: reward={info['total_reward']:.2f}")

    # Sort by reward
    episodes.sort(key=lambda x: x[2])

    selections = {
        "worst": episodes[0],
        "median": episodes[len(episodes) // 2],
        "best": episodes[-1],
    }

    os.makedirs(output_dir, exist_ok=True)
    for label, (frames, info, reward) in selections.items():
        path = os.path.join(output_dir, f"{label}_episode.mp4")
        save_video(frames, path)
        print(f"  {label}: reward={reward:.2f}, length={info['length']}, "
              f"soil={info['soil_transfer_ratio']:.1%}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Record evaluation videos")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--output-dir", type=str, default="results/videos")
    parser.add_argument("--episodes", type=int, default=20, help="Candidate episodes for selection")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print(f"Recording videos for: {args.checkpoint}")
    record_best_median_worst(
        args.checkpoint,
        output_dir=args.output_dir,
        num_candidate_episodes=args.episodes,
        device=args.device,
    )


if __name__ == "__main__":
    main()
