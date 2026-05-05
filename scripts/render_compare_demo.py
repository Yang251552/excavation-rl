"""Render a 2-row comparison GIF: BC reference trajectory (top row) vs.
trained PPO policy with stochastic actions (bottom row). Each row shows
top-down (xy) + 3D iso views of bucket and particles.

Usage:
    python scripts/render_compare_demo.py \
        --ckpt results/checkpoints/stage1_v31_500iter_repro_seed500/best_model.pt \
        --output results/figures/demos/v31_vs_bc_seed500.gif \
        --seed 500
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
from scripts.scoop_demo import build_scoop_trajectory, trajectory_to_action_deltas
from scripts.record_demo import build_actor_critic


N_BC_STEPS = 200


def _capture(env, plot_idx):
    return {
        "ee": env._ee_pos.copy(),
        "particles": env.scene.particle_system.get_positions_numpy()[plot_idx].copy(),
        "transfer": float(env.scene.get_soil_in_target_ratio()),
        "bucket_load": float(env.scene.get_bucket_load(env._ee_pos)),
    }


def rollout_bc(env: ExcavationEnv, seed: int) -> dict:
    """Replay the hand-coded scoop trajectory."""
    env.reset(seed=seed)

    pos0 = env.scene.particle_system.get_positions_numpy()
    N = pos0.shape[0]
    plot_idx = np.random.default_rng(0).choice(N, size=min(N, 200), replace=False)

    default_q = np.array(env.arm_cfg.default_joint_pos, dtype=np.float32)
    joint_traj = build_scoop_trajectory(
        default_q,
        heap_center=env.cfg.scene.soil_heap_center,
        target_center=env.cfg.scene.target_center,
        n_steps=N_BC_STEPS,
    )
    actions = trajectory_to_action_deltas(joint_traj, default_q,
                                          env.cfg.action.action_scale)

    frames = []
    frames.append(_capture(env, plot_idx))
    for t in range(N_BC_STEPS):
        _, _, terminated, truncated, _ = env.step(actions[t])
        frames.append(_capture(env, plot_idx))
        if terminated or truncated:
            break

    return _stack_frames(frames, plot_idx)


def rollout_policy(env: ExcavationEnv, ac: ActorCritic, device: str,
                    seed: int, max_steps: int, stochastic: bool) -> dict:
    obs, _ = env.reset(seed=seed)
    pos0 = env.scene.particle_system.get_positions_numpy()
    N = pos0.shape[0]
    plot_idx = np.random.default_rng(0).choice(N, size=min(N, 200), replace=False)

    frames = [_capture(env, plot_idx)]
    for _ in range(max_steps):
        with torch.no_grad():
            obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device)
            if stochastic:
                action, _, _, _ = ac.get_action(obs_t, deterministic=False)
            else:
                action_mean, _ = ac.forward(obs_t)
                action = action_mean
            action_np = action.squeeze(0).cpu().numpy()

        obs, _r, terminated, truncated, _info = env.step(action_np)
        frames.append(_capture(env, plot_idx))
        if terminated or truncated:
            break

    return _stack_frames(frames, plot_idx)


def _stack_frames(frames: list[dict], plot_idx: np.ndarray) -> dict:
    return {
        "ee": np.array([f["ee"] for f in frames]),
        "particles": np.array([f["particles"] for f in frames]),
        "transfer": np.array([f["transfer"] for f in frames]),
        "bucket_load": np.array([f["bucket_load"] for f in frames]),
    }


def _pad_to(traj: dict, T: int) -> dict:
    """Pad a trajectory's last frame to length T (so two trajectories align)."""
    cur = traj["ee"].shape[0]
    if cur >= T:
        return {k: v[:T] for k, v in traj.items()}
    pad = T - cur
    out = {}
    for k, v in traj.items():
        last = v[-1:]
        repeats = np.repeat(last, pad, axis=0)
        out[k] = np.concatenate([v, repeats], axis=0)
    return out


def render_compare(bc: dict, ppo: dict, target_lower, target_upper, heap_center,
                    output: str, fps: int = 12, frame_stride: int = 4):
    T = max(bc["ee"].shape[0], ppo["ee"].shape[0])
    bc = _pad_to(bc, T)
    ppo = _pad_to(ppo, T)

    pp_all = np.concatenate([bc["particles"].reshape(-1, 3),
                             ppo["particles"].reshape(-1, 3)], axis=0)
    x_lo, x_hi = pp_all[:, 0].min() - 0.1, pp_all[:, 0].max() + 0.1
    y_lo, y_hi = pp_all[:, 1].min() - 0.1, pp_all[:, 1].max() + 0.1
    ee_all = np.concatenate([bc["ee"], ppo["ee"]], axis=0)
    z_lo, z_hi = -0.05, max(pp_all[:, 2].max(), ee_all[:, 2].max()) + 0.1

    fig = plt.figure(figsize=(10, 8), dpi=90)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2], hspace=0.35, wspace=0.25)
    ax_bc_top = fig.add_subplot(gs[0, 0])
    ax_bc_iso = fig.add_subplot(gs[0, 1], projection="3d")
    ax_pp_top = fig.add_subplot(gs[1, 0])
    ax_pp_iso = fig.add_subplot(gs[1, 1], projection="3d")

    def setup_top(ax, label):
        ax.set_aspect("equal")
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_title(f"{label} — top view", fontsize=10)
        ax.add_patch(plt.Rectangle(
            (target_lower[0], target_lower[1]),
            target_upper[0] - target_lower[0],
            target_upper[1] - target_lower[1],
            fill=True, alpha=0.18, color="tab:green"))
        ax.plot(heap_center[0], heap_center[1], "x", color="saddlebrown",
                markersize=9, mew=2)

    def setup_iso(ax, label):
        ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi); ax.set_zlim(z_lo, z_hi)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"{label} — 3D", fontsize=10)
        ax.view_init(elev=22, azim=-55)

    setup_top(ax_bc_top, "BC reference")
    setup_iso(ax_bc_iso, "BC reference")
    setup_top(ax_pp_top, "PPO (V31, stochastic)")
    setup_iso(ax_pp_iso, "PPO (V31, stochastic)")

    sc_bc_top = ax_bc_top.scatter(bc["particles"][0][:, 0], bc["particles"][0][:, 1],
                                   s=4, c="saddlebrown", alpha=0.7)
    sc_pp_top = ax_pp_top.scatter(ppo["particles"][0][:, 0], ppo["particles"][0][:, 1],
                                   s=4, c="saddlebrown", alpha=0.7)
    sc_bc_iso = ax_bc_iso.scatter(bc["particles"][0][:, 0], bc["particles"][0][:, 1],
                                   bc["particles"][0][:, 2], s=3, c="saddlebrown", alpha=0.5)
    sc_pp_iso = ax_pp_iso.scatter(ppo["particles"][0][:, 0], ppo["particles"][0][:, 1],
                                   ppo["particles"][0][:, 2], s=3, c="saddlebrown", alpha=0.5)

    bk_bc_top, = ax_bc_top.plot([bc["ee"][0, 0]], [bc["ee"][0, 1]], "s",
                                 color="tab:blue", markersize=10)
    bk_pp_top, = ax_pp_top.plot([ppo["ee"][0, 0]], [ppo["ee"][0, 1]], "s",
                                 color="tab:orange", markersize=10)
    bk_bc_iso, = ax_bc_iso.plot([bc["ee"][0, 0]], [bc["ee"][0, 1]],
                                 [bc["ee"][0, 2]], "s", color="tab:blue", markersize=8)
    bk_pp_iso, = ax_pp_iso.plot([ppo["ee"][0, 0]], [ppo["ee"][0, 1]],
                                 [ppo["ee"][0, 2]], "s", color="tab:orange", markersize=8)

    suptitle = fig.suptitle("", fontsize=11)

    def update(t):
        sc_bc_top.set_offsets(bc["particles"][t][:, [0, 1]])
        sc_pp_top.set_offsets(ppo["particles"][t][:, [0, 1]])
        sc_bc_iso._offsets3d = (bc["particles"][t][:, 0], bc["particles"][t][:, 1],
                                 bc["particles"][t][:, 2])
        sc_pp_iso._offsets3d = (ppo["particles"][t][:, 0], ppo["particles"][t][:, 1],
                                 ppo["particles"][t][:, 2])
        bk_bc_top.set_data([bc["ee"][t, 0]], [bc["ee"][t, 1]])
        bk_pp_top.set_data([ppo["ee"][t, 0]], [ppo["ee"][t, 1]])
        bk_bc_iso.set_data_3d([bc["ee"][t, 0]], [bc["ee"][t, 1]], [bc["ee"][t, 2]])
        bk_pp_iso.set_data_3d([ppo["ee"][t, 0]], [ppo["ee"][t, 1]], [ppo["ee"][t, 2]])

        suptitle.set_text(
            f"step {t:3d}/{T-1}    "
            f"BC transfer={bc['transfer'][t]*100:5.1f}% load={bc['bucket_load'][t]:.0f}    "
            f"PPO transfer={ppo['transfer'][t]*100:5.1f}% load={ppo['bucket_load'][t]:.0f}"
        )
        return (sc_bc_top, sc_pp_top, sc_bc_iso, sc_pp_iso,
                bk_bc_top, bk_pp_top, bk_bc_iso, bk_pp_iso, suptitle)

    frames = list(range(0, T, frame_stride))
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    anim.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--frame-stride", type=int, default=4)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--deterministic", action="store_true",
                   help="Use deterministic policy (default: stochastic)")
    args = p.parse_args()

    print(f"[compare] device={args.device} ckpt={args.ckpt}")
    ac, _, _ = build_actor_critic(args.ckpt, args.device)

    cfg = ExcavationEnvCfg()
    cfg.scene.use_rigid_body_proxy = False
    env = ExcavationEnv(cfg)

    print(f"[compare] BC replay (n_steps={N_BC_STEPS})...")
    bc = rollout_bc(env, args.seed)
    print(f"[compare]   BC: steps={bc['ee'].shape[0]} "
          f"final_transfer={bc['transfer'][-1]*100:.1f}% "
          f"peak_transfer={bc['transfer'].max()*100:.1f}% "
          f"peak_load={bc['bucket_load'].max():.0f}")

    print(f"[compare] PPO rollout (stochastic={not args.deterministic}, "
          f"max_steps={args.max_steps})...")
    ppo = rollout_policy(env, ac, args.device, args.seed,
                          args.max_steps, stochastic=not args.deterministic)
    print(f"[compare]   PPO: steps={ppo['ee'].shape[0]} "
          f"final_transfer={ppo['transfer'][-1]*100:.1f}% "
          f"peak_transfer={ppo['transfer'].max()*100:.1f}% "
          f"peak_load={ppo['bucket_load'].max():.0f}")

    target_lower, target_upper = env.scene.get_target_bounds()
    heap_center = np.asarray(env.cfg.scene.soil_heap_center, dtype=np.float32)

    print(f"[compare] rendering {args.output}...")
    render_compare(bc, ppo, target_lower, target_upper, heap_center,
                    output=args.output, fps=args.fps, frame_stride=args.frame_stride)
    print(f"[compare] saved {args.output}")


if __name__ == "__main__":
    main()
