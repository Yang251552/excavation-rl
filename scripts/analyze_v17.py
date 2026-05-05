"""Pull v17 multi-seed reward decomposition + behavior signal analysis."""
import wandb
import numpy as np

api = wandb.Api()
runs = {
    7:   "yangchenghan2515-eth-z-rich/excavation-rl/wnmqx350",
    42:  "yangchenghan2515-eth-z-rich/excavation-rl/x1nvjpgq",
    123: "yangchenghan2515-eth-z-rich/excavation-rl/mtbrd8u4",
}

keys = [
    "reward/_total", "reward/approach", "reward/dig", "reward/load",
    "reward/transport", "reward/transfer", "reward/success_bonus",
    "reward/smooth_penalty", "reward/time_penalty",
    "reward/collision_penalty", "reward/joint_limit_penalty",
    "Performance/episodic_return", "Performance/soil_transfer_ratio",
    "Performance/episodic_return_std", "Policy/approx_kl",
    "Policy/clip_fraction", "Policy/explained_variance", "Policy/entropy",
]

# samples=500 means up to 500 evenly-spaced points
# But the run only has ~25 logged points (every 20 iters), so we get all of them
data = {}
for seed, path in runs.items():
    run = api.run(path)
    rows = list(run.scan_history(keys=keys))  # full unsampled history
    series = {k: np.array([r.get(k) for r in rows if r.get(k) is not None], dtype=float) for k in keys}
    data[seed] = series
    print(f"seed {seed}: {len(rows)} log points; reward/_total len={len(series['reward/_total'])}")

print()
print(f"{'metric':<35} | {'seed7':>10} {'seed42':>10} {'seed123':>10} | {'mean':>8} | nz%")
print("-" * 100)
for k in keys:
    last10s = []
    nz_pcts = []
    for seed in runs:
        v = data[seed].get(k, np.array([]))
        if len(v) == 0:
            last10s.append(np.nan)
            nz_pcts.append(np.nan)
            continue
        last10s.append(v[-10:].mean() if len(v) >= 10 else v.mean())
        nz_pcts.append((np.abs(v) > 1e-3).mean() * 100)
    mean_last = np.nanmean(last10s)
    mean_nz = np.nanmean(nz_pcts)
    line = "  ".join(f"{x:>8.3f}" if not np.isnan(x) else "      nan" for x in last10s)
    print(f"{k:<35} | {line} | {mean_last:>8.3f} | {mean_nz:5.1f}%")

print()
print("=== success_bonus events (correlation with other rewards) ===")
for seed in runs:
    sb = data[seed].get("reward/success_bonus", np.array([]))
    if len(sb) == 0:
        continue
    fire_idx = np.where(sb > 0.05)[0]
    print(f"seed {seed}: success_bonus > 0.05 at {len(fire_idx)}/{len(sb)} log points "
          f"({len(fire_idx)/len(sb)*100:.1f}%); peak={sb.max():.3f}")
    if len(fire_idx) > 0:
        peak = int(np.argmax(sb))
        for k in ["reward/transfer", "reward/load", "reward/transport", "Performance/soil_transfer_ratio"]:
            v = data[seed].get(k, np.array([]))
            if peak < len(v):
                print(f"   at peak iter (idx {peak}): {k} = {v[peak]:.4f}")

print()
print("=== final-window means (last 5 log points = ~last 100 iter) ===")
for seed in runs:
    print(f"\nseed {seed}:")
    for k in ["reward/approach", "reward/dig", "reward/load", "reward/transport",
             "reward/transfer", "reward/success_bonus", "reward/smooth_penalty"]:
        v = data[seed].get(k, np.array([]))
        if len(v) == 0:
            continue
        tail = v[-5:]
        print(f"  {k:<28} mean={tail.mean():>9.3f}  max_ever={v.max():>9.3f}  "
              f"fire%={(np.abs(v)>0.001).mean()*100:5.1f}")

print()
print("=== KL trajectory (do we see drift?) ===")
for seed in runs:
    kl = data[seed].get("Policy/approx_kl", np.array([]))
    if len(kl) >= 5:
        print(f"seed {seed}: kl[0:5]={kl[:5].round(2).tolist()}  kl[-5:]={kl[-5:].round(2).tolist()}  "
              f"first10mean={kl[:10].mean():.2f}  last10mean={kl[-10:].mean():.2f}")
