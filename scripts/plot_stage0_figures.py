"""Generate the three stage-0 README figures from wandb data.

Same visual style as scripts/plot_stage1_comparison.py: 2x2 panel grid, mean +/- std
envelope across the 3 seeds {7, 42, 123} of the multi-seed-v3 campaign.

Outputs (all to results/figures/stage0/):
  - stage0_performance_metrics.png   (README headline figure)
  - stage0_reward_components.png     (referenced in stage0 appendix)
  - stage0_ppo_diagnostics.png       (referenced in stage0 appendix)
"""
import os
import numpy as np
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/home/ubuntu/excavation-rl/results/figures/stage0"
ITERATIONS = 1000  # stage 0 ran 1000 PPO iterations per seed
COLOR = "#4477cc"
SEEDS = {
    7:   "yangchenghan2515-eth-z-rich/excavation-rl/xvlprb9g",
    42:  "yangchenghan2515-eth-z-rich/excavation-rl/j1ri0uwt",
    123: "yangchenghan2515-eth-z-rich/excavation-rl/rmib0x5z",
}

FIGURES = {
    "stage0_performance_metrics.png": {
        "suptitle": "Stage 0: performance metrics  (mean ± std across seeds {7, 42, 123})",
        "panels": [
            ("Performance/episodic_return", "Episodic return\n(rises across all seeds → policy is learning)", "linear"),
            ("Performance/episodic_length", "Episodic length\n(constant 500 → no early termination)", "linear"),
            ("Performance/success_rate", "Success rate\n(0 by design — disabled in stage 0)", "linear"),
            ("Performance/soil_transfer_ratio", "Soil transfer ratio\n(0 by design — proxy scene)", "linear"),
        ],
    },
    "stage0_reward_components.png": {
        "suptitle": "Stage 0: reward decomposition  (mean ± std across seeds {7, 42, 123})",
        "panels": [
            ("reward/approach", "reward / approach\n(positional shaping, primary signal)", "linear"),
            ("reward/smooth_penalty", "reward / smooth_penalty\n(−50 → −15: trajectories smooth out)", "linear"),
            ("reward/time_penalty", "reward / time_penalty\n(constant −5 = −0.01 × 500 steps)", "linear"),
            ("reward/transfer", "reward / transfer\n(exactly 0 → stage-0 weight override works)", "linear"),
        ],
    },
    "stage0_ppo_diagnostics.png": {
        "suptitle": "Stage 0: PPO diagnostics  (mean ± std across seeds {7, 42, 123})",
        "panels": [
            ("Policy/explained_variance", "Explained variance\n(value fn fits dense reward)", "linear"),
            ("Policy/value_loss", "Value loss\n(isolated spikes auto-recover)", "symlog"),
            ("Policy/learning_rate", "Learning rate\n(fixed schedule, 3e−4)", "log"),
            ("Policy/fps", "Throughput (steps / sec)\n(~1200 with 3 parallel processes)", "linear"),
        ],
    },
}


def all_keys():
    keys = set()
    for fig in FIGURES.values():
        for k, _, _ in fig["panels"]:
            keys.add(k)
    return sorted(keys)


def load_runs(keys):
    api = wandb.Api()
    out = {}
    for seed, path in SEEDS.items():
        print(f"  loading seed {seed} ({path})...")
        run = api.run(path)
        rows = list(run.scan_history(keys=keys + ["_step"]))
        out[seed] = {
            k: np.array([r.get(k) for r in rows if r.get(k) is not None], dtype=float)
            for k in keys
        }
    return out


def aligned_stack(data, key):
    series = [data[s][key] for s in data if len(data[s].get(key, [])) > 0]
    if not series:
        return None
    L = min(len(s) for s in series)
    return np.stack([s[:L] for s in series])


def render(data, panels, suptitle, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (key, title, yscale) in zip(axes.flat, panels):
        a = aligned_stack(data, key)
        if a is None:
            ax.set_visible(False)
            continue
        x = np.arange(a.shape[1]) * (ITERATIONS / a.shape[1])
        m, s = a.mean(axis=0), a.std(axis=0)
        ax.plot(x, m, color=COLOR, linewidth=1.8, label="multi-seed-v3 (3 seeds)")
        ax.fill_between(x, m - s, m + s, color=COLOR, alpha=0.18)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("PPO iteration")
        ax.grid(True, alpha=0.3)
        if yscale == "log":
            ax.set_yscale("log")
        elif yscale == "symlog":
            ax.set_yscale("symlog", linthresh=0.1)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=1,
               bbox_to_anchor=(0.5, 1.05), frameon=False, fontsize=11)
    fig.suptitle(suptitle, fontsize=12, y=1.10)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    keys = all_keys()
    print(f"Loading {len(keys)} keys from {len(SEEDS)} stage-0 runs...")
    data = load_runs(keys)

    for filename, spec in FIGURES.items():
        print(f"Rendering {filename}...")
        out_path = os.path.join(OUT_DIR, filename)
        render(data, spec["panels"], spec["suptitle"], out_path)


if __name__ == "__main__":
    main()
