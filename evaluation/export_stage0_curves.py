"""Export Stage 0 multi-seed wandb runs to PNG figures for the README.

Usage:
    pip install wandb pandas matplotlib
    wandb login
    python evaluation/export_stage0_curves.py \
        --entity yangchenghan2515-eth-z-rich \
        --project excavation-rl \
        --tag multi-seed-v3 \
        --out assets/stage0
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import wandb


METRICS = [
    ("reward/_total", "Episode Total Reward"),
    ("reward/approach", "Approach Reward (dense)"),
    ("reward/smooth_penalty", "Action Smoothness Penalty"),
    ("Performance/episodic_return", "Episodic Return"),
    ("Performance/episodic_length", "Episodic Length"),
    ("Policy/value_loss", "Value Loss"),
    ("Policy/policy_loss", "Policy Loss"),
    ("Policy/explained_variance", "Explained Variance"),
    ("Policy/learning_rate", "Learning Rate"),
    ("Policy/fps", "Steps / Second"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--entity", required=True, help="wandb entity (team/user)")
    p.add_argument("--project", required=True, help="wandb project")
    p.add_argument("--tag", required=True, help="wandb tag to filter runs")
    p.add_argument("--out", default="assets/stage0", help="output directory for PNGs")
    p.add_argument("--aggregate", action="store_true",
                   help="Plot mean ± std across seeds instead of individual lines")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    api = wandb.Api()
    runs = list(api.runs(f"{args.entity}/{args.project}",
                         filters={"tags": {"$in": [args.tag]}}))
    if not runs:
        raise SystemExit(f"No runs found with tag={args.tag}")
    print(f"Found {len(runs)} runs: {[r.name for r in runs]}")

    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    for metric_key, title in METRICS:
        fig, ax = plt.subplots()

        # Pull metric history per run
        series_per_run = []
        for r in runs:
            h = r.history(keys=[metric_key, "_step"], pandas=True)
            if h.empty or metric_key not in h:
                continue
            h = h.dropna(subset=[metric_key])
            series_per_run.append((r.name, h["_step"].to_numpy(),
                                   h[metric_key].to_numpy()))

        if not series_per_run:
            print(f"  skip {metric_key} — no data")
            plt.close(fig)
            continue

        if args.aggregate and len(series_per_run) >= 2:
            # Resample onto common step grid, plot mean ± std
            all_steps = np.unique(np.concatenate([s for _, s, _ in series_per_run]))
            stacked = np.full((len(series_per_run), len(all_steps)), np.nan)
            for i, (_, s, v) in enumerate(series_per_run):
                stacked[i] = np.interp(all_steps, s, v, left=np.nan, right=np.nan)
            mean = np.nanmean(stacked, axis=0)
            std = np.nanstd(stacked, axis=0)
            ax.plot(all_steps, mean, color="C0", linewidth=2, label=f"mean (n={len(series_per_run)})")
            ax.fill_between(all_steps, mean - std, mean + std, color="C0", alpha=0.2,
                            label="± 1 std")
        else:
            for i, (name, s, v) in enumerate(series_per_run):
                ax.plot(s, v, color=f"C{i}", linewidth=1.5, label=name)

        ax.set_xlabel("PPO iteration")
        ax.set_ylabel(metric_key)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()

        # Filename: replace / with _ for filesystem safety
        fname = metric_key.replace("/", "_") + ".png"
        path = os.path.join(args.out, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path}")

    print("\nDone. Embed in README like:")
    for metric_key, _ in METRICS:
        fname = metric_key.replace("/", "_") + ".png"
        print(f"![{metric_key}]({args.out}/{fname})")


if __name__ == "__main__":
    main()
