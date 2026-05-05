"""Generate the 2x2 v17 vs v22 comparison figure for the stage 1 README.

Panels: Policy/approx_kl, Policy/clip_fraction, reward/load, reward/transport.
Each panel overlays mean +/- std across the 3 seeds, v17 (red) vs v22 (blue).
"""
import os
import numpy as np
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_PATH = "/home/ubuntu/excavation-rl/results/figures/stage1/v17_vs_v22_vs_v28a_comparison.png"

api = wandb.Api()
v17 = {
    7:   "yangchenghan2515-eth-z-rich/excavation-rl/wnmqx350",
    42:  "yangchenghan2515-eth-z-rich/excavation-rl/x1nvjpgq",
    123: "yangchenghan2515-eth-z-rich/excavation-rl/mtbrd8u4",
}
v22 = {
    7:   "yangchenghan2515-eth-z-rich/excavation-rl/auek7u4r",
    42:  "yangchenghan2515-eth-z-rich/excavation-rl/cc68bjg9",
    123: "yangchenghan2515-eth-z-rich/excavation-rl/bc64lv3c",
}
v28a = {
    7:   "yangchenghan2515-eth-z-rich/excavation-rl/yu9hdybj",
    42:  "yangchenghan2515-eth-z-rich/excavation-rl/qzl7ev8t",
    123: "yangchenghan2515-eth-z-rich/excavation-rl/44e5gk2z",
}

PANELS = [
    ("Policy/approx_kl", "PPO trust-region drift\n(approx KL per update)", "log"),
    ("Policy/clip_fraction", "Fraction of PPO updates clipped", "linear"),
    ("reward/transfer", "reward / transfer\n(per-iter transfer signal)", "symlog"),
    ("Performance/soil_transfer_ratio", "Performance / soil_transfer_ratio\n(end-of-window heap-to-target fraction)", "linear"),
]

KEYS = [k for k, _, _ in PANELS]


def load(run_dict):
    out = {}
    for seed, path in run_dict.items():
        run = api.run(path)
        rows = list(run.scan_history(keys=KEYS + ["_step"]))
        out[seed] = {
            k: np.array([r.get(k) for r in rows if r.get(k) is not None], dtype=float)
            for k in KEYS
        }
    return out


def aligned_stack(d, key):
    """Stack across seeds; truncate to shortest length so the array is rectangular."""
    series = [d[s][key] for s in d if len(d[s].get(key, [])) > 0]
    if not series:
        return None
    L = min(len(s) for s in series)
    return np.stack([s[:L] for s in series])


def main():
    print("Loading v17...")
    d17 = load(v17)
    print("Loading v22...")
    d22 = load(v22)
    print("Loading v28a...")
    d28a = load(v28a)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)

    series_specs = [
        ("v17 (pure PPO, 8 bugs fixed)", "#cc4444", d17),
        ("v22 (+ BC + containment)", "#4477cc", d22),
        ("v28a (+ KL anchor + reverse curriculum)", "#22aa55", d28a),
    ]

    for ax, (key, title, yscale) in zip(axes.flat, PANELS):
        for label, color, d in series_specs:
            arr = aligned_stack(d, key)
            if arr is None:
                continue
            # Map log points to PPO iter assuming each run = 500 iter
            x = np.arange(arr.shape[1]) * (500 / max(arr.shape[1], 1))
            m, s = arr.mean(axis=0), arr.std(axis=0)
            ax.plot(x, m, color=color, label=label, linewidth=1.8)
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.16)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("PPO iteration")
        ax.grid(True, alpha=0.3)
        if yscale == "log":
            ax.set_yscale("symlog", linthresh=0.1)
        elif yscale == "symlog":
            ax.set_yscale("symlog", linthresh=0.01)

    # Reference line: random baseline transfer = 0.15 on transfer_ratio panel
    for ax, (key, _, _) in zip(axes.flat, PANELS):
        if key == "Performance/soil_transfer_ratio":
            ax.axhline(0.15, color="black", linestyle="--", linewidth=1, alpha=0.6,
                       label="random baseline (0.15)")
            ax.axhline(0.224, color="gray", linestyle=":", linewidth=1, alpha=0.6,
                       label="hand-coded demo (0.224)")
            ax.legend(fontsize=8, loc="upper left")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05),
               frameon=False, fontsize=10)

    fig.suptitle("Stage 1 → Stage 1.5 progression  (3 seeds {7, 42, 123} mean ± std, 500 iter each)",
                 fontsize=12, y=1.10)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
