"""Record a single deterministic rollout from a trained PPO policy and save
a 3-panel matplotlib animation (top XY / side XZ / 3D iso) as a GIF.

Usage:
    python scripts/record_demo.py \
        --ckpt results/checkpoints/stage1_v31_500iter_repro_seed500/best_model.pt \
        --output results/figures/demos/v31_seed500.gif \
        --seed 500

Designed for portfolio-quality visualization. Subsamples particles for
plotting speed, captures bucket pos + soil-in-target ratio + bucket load
per step, renders ~120 frames at 12 fps for a ~10s GIF.
"""
from __future__ import annotations

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from envs.excavation_env_cfg import ExcavationEnvCfg
from envs.excavation_env import ExcavationEnv
from training.train import ActorCritic


def infer_hidden_dims(state_dict: dict, prefix: str) -> list[int]:
    """Read MLP hidden dim list from a saved state_dict by walking
    Linear weight shapes under e.g. 'policy_net' / 'value_net' prefix.
    The final layer's out_features is the action / value dim and is dropped.
    """
    weight_keys = sorted(
        k for k in state_dict if k.startswith(prefix + ".") and k.endswith(".weight")
    )
    out_dims = [state_dict[k].shape[0] for k in weight_keys]
    return out_dims[:-1]  # drop final (action_dim or 1)


def build_actor_critic(ckpt_path: str, device: str) -> tuple[ActorCritic, int, int]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    num_obs = int(ckpt["num_obs"])
    num_actions = int(ckpt["num_actions"])

    policy_hidden = infer_hidden_dims(sd, "policy_net")
    value_hidden = infer_hidden_dims(sd, "value_net")

    ac = ActorCritic(
        num_obs=num_obs,
        num_actions=num_actions,
        policy_hidden_dims=policy_hidden,
        value_hidden_dims=value_hidden,
        activation="elu",
        init_noise_std=0.5,
        normalize_obs=("obs_rms.mean" in sd),
    ).to(device)
    ac.load_state_dict(sd)
    ac.eval()
    return ac, num_obs, num_actions


def rollout(env: ExcavationEnv, ac: ActorCritic, device: str,
            max_steps: int, seed: int) -> dict:
    """Run a single deterministic episode, recording per-step state.

    Returns a dict of arrays for downstream rendering.
    """
    obs, _ = env.reset(seed=seed)

    # Subsample particles for plotting (full ~500 is OK but trimming helps GIF size).
    pos0 = env.scene.particle_system.get_positions_numpy()
    N = pos0.shape[0]
    plot_idx = np.random.default_rng(0).choice(N, size=min(N, 200), replace=False)

    ee_traj = []
    particles_traj = []
    transfer_traj = []
    bucket_load_traj = []
    reward_traj = []

    for t in range(max_steps):
        with torch.no_grad():
            obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device)
            action_mean, _ = ac.forward(obs_t)
            action = action_mean.squeeze(0).cpu().numpy()

        obs, reward, terminated, truncated, _info = env.step(action)

        ee_traj.append(env._ee_pos.copy())
        particles_traj.append(
            env.scene.particle_system.get_positions_numpy()[plot_idx].copy()
        )
        transfer_traj.append(float(env.scene.get_soil_in_target_ratio()))
        bucket_load_traj.append(float(env.scene.get_bucket_load(env._ee_pos)))
        reward_traj.append(float(reward))

        if terminated or truncated:
            break

    target_lower, target_upper = env.scene.get_target_bounds()
    heap_center = np.asarray(env.cfg.scene.soil_heap_center, dtype=np.float32)

    return {
        "ee": np.array(ee_traj),
        "particles": np.array(particles_traj),
        "transfer": np.array(transfer_traj),
        "bucket_load": np.array(bucket_load_traj),
        "reward": np.array(reward_traj),
        "target_lower": np.asarray(target_lower, dtype=np.float32),
        "target_upper": np.asarray(target_upper, dtype=np.float32),
        "heap_center": heap_center,
    }


def render_gif(traj: dict, output: str, fps: int = 12, frame_stride: int = 4):
    """Render a 3-panel GIF: top XY, side XZ, 3D iso, plus a transfer-ratio bar."""
    ee = traj["ee"]
    particles = traj["particles"]
    transfer = traj["transfer"]
    load = traj["bucket_load"]

    T = ee.shape[0]
    frames = list(range(0, T, frame_stride))

    # Plot bounds
    pp_all = particles.reshape(-1, 3)
    x_lo, x_hi = pp_all[:, 0].min() - 0.1, pp_all[:, 0].max() + 0.1
    y_lo, y_hi = pp_all[:, 1].min() - 0.1, pp_all[:, 1].max() + 0.1
    z_lo, z_hi = -0.05, max(pp_all[:, 2].max(), ee[:, 2].max()) + 0.1

    tl, tu = traj["target_lower"], traj["target_upper"]
    hc = traj["heap_center"]

    fig = plt.figure(figsize=(12, 4.5), dpi=90)
    ax_top = fig.add_subplot(1, 3, 1)
    ax_side = fig.add_subplot(1, 3, 2)
    ax_iso = fig.add_subplot(1, 3, 3, projection="3d")

    def setup_top(ax):
        ax.set_aspect("equal")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_title("Top view (xy)")
        # target zone
        ax.add_patch(plt.Rectangle((tl[0], tl[1]), tu[0] - tl[0], tu[1] - tl[1],
                                    fill=True, alpha=0.15, color="tab:green",
                                    label="target"))
        # heap center marker
        ax.plot(hc[0], hc[1], "x", color="saddlebrown", markersize=8, label="heap")
        ax.legend(loc="upper right", fontsize=7)

    def setup_side(ax):
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(z_lo, z_hi)
        ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
        ax.set_title("Side view (xz)")
        ax.axhline(0.0, color="k", lw=0.5, alpha=0.5)
        ax.axvspan(tl[0], tu[0], alpha=0.15, color="tab:green")

    def setup_iso(ax):
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi); ax.set_zlim(z_lo, z_hi)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title("3D")
        ax.view_init(elev=22, azim=-55)

    setup_top(ax_top); setup_side(ax_side); setup_iso(ax_iso)

    # initial scatter handles
    pts0 = particles[0]
    sc_top = ax_top.scatter(pts0[:, 0], pts0[:, 1], s=4, c="saddlebrown", alpha=0.7)
    sc_side = ax_side.scatter(pts0[:, 0], pts0[:, 2], s=4, c="saddlebrown", alpha=0.7)
    sc_iso = ax_iso.scatter(pts0[:, 0], pts0[:, 1], pts0[:, 2],
                            s=3, c="saddlebrown", alpha=0.5)

    bk_top, = ax_top.plot([ee[0, 0]], [ee[0, 1]], "s", color="tab:blue",
                          markersize=10, label="bucket")
    bk_side, = ax_side.plot([ee[0, 0]], [ee[0, 2]], "s", color="tab:blue",
                            markersize=10)
    bk_iso, = ax_iso.plot([ee[0, 0]], [ee[0, 1]], [ee[0, 2]], "s",
                          color="tab:blue", markersize=8)

    title = fig.suptitle("", fontsize=11)

    def update(t):
        pts = particles[t]
        sc_top.set_offsets(pts[:, [0, 1]])
        sc_side.set_offsets(pts[:, [0, 2]])
        sc_iso._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])

        bk_top.set_data([ee[t, 0]], [ee[t, 1]])
        bk_side.set_data([ee[t, 0]], [ee[t, 2]])
        bk_iso.set_data_3d([ee[t, 0]], [ee[t, 1]], [ee[t, 2]])

        title.set_text(
            f"step {t:3d} / {T-1}    transfer={transfer[t]*100:5.1f}%    "
            f"bucket_load={load[t]:.0f} particles"
        )
        return sc_top, sc_side, sc_iso, bk_top, bk_side, bk_iso, title

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps,
                         blit=False)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    anim.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--output", required=True, help="Output GIF path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--frame-stride", type=int, default=4,
                        help="Subsample factor for animation frames (every Nth env step)")
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    print(f"[record_demo] device={args.device} ckpt={args.ckpt}")
    ac, num_obs, num_actions = build_actor_critic(args.ckpt, args.device)
    print(f"[record_demo] loaded model: num_obs={num_obs} num_actions={num_actions}")

    cfg = ExcavationEnvCfg()
    cfg.scene.use_rigid_body_proxy = False
    env = ExcavationEnv(cfg)
    print(f"[record_demo] env obs_dim={env.observation_space.shape[0]} "
          f"action_dim={env.action_space.shape[0]}")

    traj = rollout(env, ac, args.device, args.max_steps, args.seed)
    final_transfer = float(traj["transfer"][-1]) if len(traj["transfer"]) else 0.0
    peak_transfer = float(traj["transfer"].max()) if len(traj["transfer"]) else 0.0
    peak_load = float(traj["bucket_load"].max()) if len(traj["bucket_load"]) else 0.0
    total_reward = float(traj["reward"].sum()) if len(traj["reward"]) else 0.0
    print(f"[record_demo] episode steps={traj['ee'].shape[0]} "
          f"final_transfer={final_transfer*100:.1f}% peak_transfer={peak_transfer*100:.1f}% "
          f"peak_bucket_load={peak_load:.0f} total_reward={total_reward:.1f}")

    render_gif(traj, args.output, fps=args.fps, frame_stride=args.frame_stride)
    print(f"[record_demo] saved {args.output}")


if __name__ == "__main__":
    main()
