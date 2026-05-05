"""Pull v18 multi-seed full data and compare against v17."""
import wandb
import numpy as np

api = wandb.Api()
v17 = {
    7:   "yangchenghan2515-eth-z-rich/excavation-rl/wnmqx350",
    42:  "yangchenghan2515-eth-z-rich/excavation-rl/x1nvjpgq",
    123: "yangchenghan2515-eth-z-rich/excavation-rl/mtbrd8u4",
}
v18 = {
    7:   "yangchenghan2515-eth-z-rich/excavation-rl/x775cluw",
    42:  "yangchenghan2515-eth-z-rich/excavation-rl/juy7w31w",
    123: "yangchenghan2515-eth-z-rich/excavation-rl/ygg4tkk1",
}
v19 = {
    7:   "yangchenghan2515-eth-z-rich/excavation-rl/7bmiruid",
    42:  "yangchenghan2515-eth-z-rich/excavation-rl/q1414nr5",
    123: "yangchenghan2515-eth-z-rich/excavation-rl/t6ybiy9r",
}
v22 = {
    7:   "yangchenghan2515-eth-z-rich/excavation-rl/auek7u4r",
    42:  "yangchenghan2515-eth-z-rich/excavation-rl/cc68bjg9",
    123: "yangchenghan2515-eth-z-rich/excavation-rl/bc64lv3c",
}
v26 = {
    7:   "yangchenghan2515-eth-z-rich/excavation-rl/u7lajnuc",
    42:  "yangchenghan2515-eth-z-rich/excavation-rl/ce7vxeo4",
    123: "yangchenghan2515-eth-z-rich/excavation-rl/6cf3hfdl",
}
v27 = {
    7:   "yangchenghan2515-eth-z-rich/excavation-rl/qi76fx68",
    42:  "yangchenghan2515-eth-z-rich/excavation-rl/20i7ab3t",
    123: "yangchenghan2515-eth-z-rich/excavation-rl/13fptlp2",
}
v28a = {
    7:   "yangchenghan2515-eth-z-rich/excavation-rl/yu9hdybj",
    42:  "yangchenghan2515-eth-z-rich/excavation-rl/qzl7ev8t",
    123: "yangchenghan2515-eth-z-rich/excavation-rl/44e5gk2z",
}
v28b = {
    7:   "yangchenghan2515-eth-z-rich/excavation-rl/cwks4rhw",
    42:  "yangchenghan2515-eth-z-rich/excavation-rl/s9tv3dax",
    123: "yangchenghan2515-eth-z-rich/excavation-rl/vsc0u0k3",
}
v31 = {
    500: "yangchenghan2515-eth-z-rich/excavation-rl/4ezad4vf",
    555: "yangchenghan2515-eth-z-rich/excavation-rl/b637zru6",
    999: "yangchenghan2515-eth-z-rich/excavation-rl/vi7np6d8",
}
v33 = {
    500: "yangchenghan2515-eth-z-rich/excavation-rl/l01moel9",
    555: "yangchenghan2515-eth-z-rich/excavation-rl/um9a8te5",
    999: "yangchenghan2515-eth-z-rich/excavation-rl/ykqfc9za",
}
v34 = {
    500: "yangchenghan2515-eth-z-rich/excavation-rl/voaf8iza",
    555: "yangchenghan2515-eth-z-rich/excavation-rl/oqc8mxs7",
    999: "yangchenghan2515-eth-z-rich/excavation-rl/r9sqjj67",
}
v35 = {
    500: "yangchenghan2515-eth-z-rich/excavation-rl/7utp05yg",
    555: "yangchenghan2515-eth-z-rich/excavation-rl/4g64ujvs",
    999: "yangchenghan2515-eth-z-rich/excavation-rl/gsul9iw1",
}
v36 = {
    500: "yangchenghan2515-eth-z-rich/excavation-rl/vgiuq0zt",
    555: "yangchenghan2515-eth-z-rich/excavation-rl/vgsu0xkx",
    999: "yangchenghan2515-eth-z-rich/excavation-rl/d0kckokl",
}

keys = [
    "reward/_total", "reward/approach", "reward/dig", "reward/load",
    "reward/transport", "reward/transfer", "reward/success_bonus",
    "reward/smooth_penalty",
    "Performance/episodic_return", "Performance/soil_transfer_ratio",
    "Performance/episodic_return_std", "Policy/approx_kl",
    "Policy/clip_fraction", "Policy/explained_variance",
    "Policy/entropy", "Policy/value_loss", "Policy/policy_loss",
    "Policy/learning_rate", "Policy/mean_reward",
]

def load(run_dict):
    out = {}
    for seed, path in run_dict.items():
        run = api.run(path)
        rows = list(run.scan_history(keys=keys))
        out[seed] = {k: np.array([r.get(k) for r in rows if r.get(k) is not None], dtype=float) for k in keys}
    return out

print("Loading v17...")
d17 = load(v17)
print("Loading v18...")
d18 = load(v18)
print("Loading v19...")
d19 = load(v19)
print("Loading v22 (BC + PPO)...")
d22 = load(v22)
print("Loading v26 (BC + KL anchor 0.5 + particle replay)...")
d26 = load(v26)
print("Loading v27 (BC + KL anchor 2.0 + prob 0.7)...")
d27 = load(v27)
print("Loading v28a (v27 long: 500 iter)...")
d28a = load(v28a)
print("Loading v28b (freeze obs_rms after BC)...")
d28b = load(v28b)
print("Loading v31 (v28a config, fresh seeds 500/555/999)...")
d31 = load(v31)
print("Loading v33 (freeze removed, same seeds as v31)...")
d33 = load(v33)
print("Loading v34 (v33 + return normalization)...")
d34 = load(v34)
print("Loading v35 (anchor 1.0)...")
d35 = load(v35)
print("Loading v36 (anchor 0.5)...")
d36 = load(v36)

def stat(d, k, fn):
    vals = [fn(d[s].get(k, np.array([]))) for s in d if len(d[s].get(k, np.array([]))) > 0]
    return np.mean(vals) if vals else float('nan')

print(f"\n{'metric':<35} | {'v22':>9} | {'v27':>9} | {'v28a':>9} | {'v28b':>9} | v28b vs v28a")
print("-" * 100)
for k in keys:
    v22_v = stat(d22, k, lambda v: v[-10:].mean() if len(v) >= 10 else v.mean())
    v27_v = stat(d27, k, lambda v: v[-10:].mean() if len(v) >= 10 else v.mean())
    v28a_v = stat(d28a, k, lambda v: v[-10:].mean() if len(v) >= 10 else v.mean())
    v28b_v = stat(d28b, k, lambda v: v[-10:].mean() if len(v) >= 10 else v.mean())
    delta = v28b_v - v28a_v
    pct = (delta / abs(v28a_v) * 100) if abs(v28a_v) > 1e-6 else 0
    arrow = "↑" if delta > 0 else "↓"
    print(f"{k:<35} | {v22_v:>9.3f} | {v27_v:>9.3f} | {v28a_v:>9.3f} | {v28b_v:>9.3f} | {arrow} {delta:>+7.3f} ({pct:+5.1f}%)")

print(f"\n=== KL drift comparison (per seed) ===")
for s in [7, 42, 123]:
    print(f"seed {s}:")
    for label, d in [("v22", d22), ("v27(300)", d27), ("v28a(500)", d28a)]:
        kl = d[s].get("Policy/approx_kl", np.array([]))
        if len(kl) >= 10:
            print(f"  {label:>10} :  first10={kl[:10].mean():.2f}, last10={kl[-10:].mean():.2f}, max={kl.max():.2f}")

print(f"\n=== success_bonus + transfer_ratio peaks ===")
for label, d in [("v22(BC)", d22), ("v27(300,a=2)", d27), ("v28a(500,a=2)", d28a)]:
    print(f"\n{label}:")
    for s in [7, 42, 123]:
        sb = d[s].get("reward/success_bonus", np.array([]))
        tr = d[s].get("Performance/soil_transfer_ratio", np.array([]))
        if len(sb) > 0:
            print(f"  seed {s}: success_bonus fired {(sb>0.05).sum()}/{len(sb)} ({(sb>0.05).mean()*100:.0f}%); "
                  f"max success_bonus={sb.max():.3f}; transfer_ratio max={tr.max() if len(tr) else 0:.3f}, last5_mean={tr[-5:].mean() if len(tr)>=5 else np.nan:.4f}")

print(f"\n=== final-window transfer_ratio (last 5 log points = ~last 100 iter) ===")
for label, d in [("v17", d17), ("v22", d22), ("v26(a=0.5)", d26), ("v27(a=2,300iter)", d27), ("v28a(a=2,500iter)", d28a), ("v31(repro,fresh)", d31), ("v33(freeze-revert)", d33)]:
    seeds = sorted(d.keys())
    means = []
    for s in seeds:
        tr = d[s].get("Performance/soil_transfer_ratio", np.array([]))
        if len(tr) >= 5:
            means.append(tr[-5:].mean())
    if means:
        print(f"  {label}: per-seed last5 = {[f'{m:.4f}' for m in means]}, mean = {np.mean(means):.4f}")

print(f"\n=== v31 §5 PASS check (3-seed mean over last 50 iter ≈ last ~3 log points) ===")
def last_n_mean(arr, n):
    return arr[-n:].mean() if len(arr) >= n else (arr.mean() if len(arr) else float('nan'))

v31_seeds = sorted(d31.keys())
print(f"\nPer-seed v31 (last-5-window for transfer, last-10 for KL/clip, full for EV final):")
for s in v31_seeds:
    tr = d31[s].get("Performance/soil_transfer_ratio", np.array([]))
    kl = d31[s].get("Policy/approx_kl", np.array([]))
    cf = d31[s].get("Policy/clip_fraction", np.array([]))
    ev = d31[s].get("Policy/explained_variance", np.array([]))
    er = d31[s].get("Performance/episodic_return", np.array([]))
    er_std = d31[s].get("Performance/episodic_return_std", np.array([]))
    sb = d31[s].get("reward/success_bonus", np.array([]))
    rl = d31[s].get("reward/load", np.array([]))
    rt = d31[s].get("reward/transfer", np.array([]))
    cov = (er_std[-10:].mean() / abs(er[-10:].mean())) if len(er_std) >= 10 and len(er) >= 10 and abs(er[-10:].mean()) > 1e-6 else float('nan')
    print(f"  seed {s}: tr_last5={last_n_mean(tr,5):.4f} (max={tr.max() if len(tr) else 0:.4f}), "
          f"KL_last10={last_n_mean(kl,10):.2f}, clip_last10={last_n_mean(cf,10):.3f}, "
          f"EV_final={ev[-1] if len(ev) else float('nan'):.3f}, CoV={cov:.3f}, "
          f"SB_fire={(sb>0.05).mean()*100 if len(sb) else 0:.0f}%, "
          f"r_load_final={last_n_mean(rl,10):.1f}, r_transfer_final={last_n_mean(rt,10):.3f}")

# 3-seed means
tr_means = [last_n_mean(d31[s].get("Performance/soil_transfer_ratio", np.array([])), 5) for s in v31_seeds]
kl_means = [last_n_mean(d31[s].get("Policy/approx_kl", np.array([])), 10) for s in v31_seeds]
cf_means = [last_n_mean(d31[s].get("Policy/clip_fraction", np.array([])), 10) for s in v31_seeds]
ev_finals = [d31[s].get("Policy/explained_variance", np.array([float('nan')]))[-1] for s in v31_seeds]
covs = []
for s in v31_seeds:
    er = d31[s].get("Performance/episodic_return", np.array([]))
    er_std = d31[s].get("Performance/episodic_return_std", np.array([]))
    if len(er) >= 10 and len(er_std) >= 10 and abs(er[-10:].mean()) > 1e-6:
        covs.append(er_std[-10:].mean() / abs(er[-10:].mean()))

tr_3mean = np.nanmean(tr_means)
kl_3mean = np.nanmean(kl_means)
cf_3mean = np.nanmean(cf_means)
ev_3mean = np.nanmean(ev_finals)
cov_3mean = np.mean(covs) if covs else float('nan')
print(f"\nv31 3-seed means: transfer_ratio={tr_3mean:.4f}, KL={kl_3mean:.2f}, clip={cf_3mean:.3f}, EV={ev_3mean:.3f}, CoV={cov_3mean:.3f}")
print(f"§5 PASS: tr>0.15 [{'PASS' if tr_3mean>0.15 else 'FAIL'}] | KL<1 [{'PASS' if kl_3mean<1 else 'FAIL'}] | clip<0.5 [{'PASS' if cf_3mean<0.5 else 'FAIL'}] | CoV<0.5 [{'PASS' if cov_3mean<0.5 else 'FAIL'}] | EV>0.5 [{'PASS' if ev_3mean>0.5 else 'FAIL'}]")

# 6-seed mean (v28a + v31) for decision tree
v28a_seeds = sorted(d28a.keys())
v28a_tr = [last_n_mean(d28a[s].get("Performance/soil_transfer_ratio", np.array([])), 5) for s in v28a_seeds]
combined = v28a_tr + tr_means
print(f"\nv28a per-seed last5: {[f'{m:.4f}' for m in v28a_tr]} | v31 per-seed last5: {[f'{m:.4f}' for m in tr_means]}")
print(f"6-seed combined transfer_ratio mean (v28a+v31): {np.mean(combined):.4f}")
print(f"v31 seeds clearing 0.20: {sum(1 for m in tr_means if m>=0.20)}/3")
print(f"v31 seeds clearing 0.15: {sum(1 for m in tr_means if m>=0.15)}/3")

print(f"\n=== v33 §5 PASS check (3-seed mean) ===")
v33_seeds = sorted(d33.keys())
for s in v33_seeds:
    tr = d33[s].get("Performance/soil_transfer_ratio", np.array([]))
    kl = d33[s].get("Policy/approx_kl", np.array([]))
    cf = d33[s].get("Policy/clip_fraction", np.array([]))
    ev = d33[s].get("Policy/explained_variance", np.array([]))
    er = d33[s].get("Performance/episodic_return", np.array([]))
    er_std = d33[s].get("Performance/episodic_return_std", np.array([]))
    sb = d33[s].get("reward/success_bonus", np.array([]))
    rl = d33[s].get("reward/load", np.array([]))
    rt = d33[s].get("reward/transfer", np.array([]))
    cov = (er_std[-10:].mean() / abs(er[-10:].mean())) if len(er_std) >= 10 and len(er) >= 10 and abs(er[-10:].mean()) > 1e-6 else float('nan')
    print(f"  seed {s}: tr_last5={last_n_mean(tr,5):.4f} (max={tr.max() if len(tr) else 0:.4f}), KL_last10={last_n_mean(kl,10):.2f}, clip_last10={last_n_mean(cf,10):.3f}, EV_final={ev[-1] if len(ev) else float('nan'):.3f}, CoV={cov:.3f}, SB_fire={(sb>0.05).mean()*100 if len(sb) else 0:.0f}%, r_load_final={last_n_mean(rl,10):.1f}, r_transfer_final={last_n_mean(rt,10):.3f}, ER_l10={last_n_mean(er,10):.1f}")

tr_means_33 = [last_n_mean(d33[s].get("Performance/soil_transfer_ratio", np.array([])), 5) for s in v33_seeds]
kl_means_33 = [last_n_mean(d33[s].get("Policy/approx_kl", np.array([])), 10) for s in v33_seeds]
cf_means_33 = [last_n_mean(d33[s].get("Policy/clip_fraction", np.array([])), 10) for s in v33_seeds]
ev_finals_33 = [d33[s].get("Policy/explained_variance", np.array([float('nan')]))[-1] for s in v33_seeds]
covs_33 = []
for s in v33_seeds:
    er = d33[s].get("Performance/episodic_return", np.array([]))
    er_std = d33[s].get("Performance/episodic_return_std", np.array([]))
    if len(er) >= 10 and len(er_std) >= 10 and abs(er[-10:].mean()) > 1e-6:
        covs_33.append(er_std[-10:].mean() / abs(er[-10:].mean()))

tr_3m = np.nanmean(tr_means_33)
kl_3m = np.nanmean(kl_means_33)
cf_3m = np.nanmean(cf_means_33)
ev_3m = np.nanmean(ev_finals_33)
cov_3m = np.mean(covs_33) if covs_33 else float('nan')
print(f"\nv33 3-seed means: transfer_ratio={tr_3m:.4f}, KL={kl_3m:.2f}, clip={cf_3m:.3f}, EV={ev_3m:.3f}, CoV={cov_3m:.3f}")
print(f"§5 PASS: tr>0.15 [{'PASS' if tr_3m>0.15 else 'FAIL'}] | KL<1 [{'PASS' if kl_3m<1 else 'FAIL'}] | clip<0.5 [{'PASS' if cf_3m<0.5 else 'FAIL'}] | CoV<0.5 [{'PASS' if cov_3m<0.5 else 'FAIL'}] | EV>0.5 [{'PASS' if ev_3m>0.5 else 'FAIL'}]")
n_pass = sum([tr_3m>0.15, kl_3m<1, cf_3m<0.5, cov_3m<0.5, ev_3m>0.5])
print(f"v33 §5 PASS count: {n_pass}/5")

print(f"\n=== v34 §5 PASS check (3-seed mean) ===")
v34_seeds = sorted(d34.keys())
for s in v34_seeds:
    tr = d34[s].get("Performance/soil_transfer_ratio", np.array([]))
    kl = d34[s].get("Policy/approx_kl", np.array([]))
    cf = d34[s].get("Policy/clip_fraction", np.array([]))
    ev = d34[s].get("Policy/explained_variance", np.array([]))
    er = d34[s].get("Performance/episodic_return", np.array([]))
    er_std = d34[s].get("Performance/episodic_return_std", np.array([]))
    sb = d34[s].get("reward/success_bonus", np.array([]))
    rl = d34[s].get("reward/load", np.array([]))
    rt = d34[s].get("reward/transfer", np.array([]))
    vl = d34[s].get("Policy/value_loss", np.array([]))
    cov = (er_std[-10:].mean() / abs(er[-10:].mean())) if len(er_std) >= 10 and len(er) >= 10 and abs(er[-10:].mean()) > 1e-6 else float('nan')
    print(f"  seed {s}: tr_l5={last_n_mean(tr,5):.4f} (max={tr.max() if len(tr) else 0:.4f}), KL_l10={last_n_mean(kl,10):.2f}, clip_l10={last_n_mean(cf,10):.3f}, EV_f={ev[-1] if len(ev) else float('nan'):.3f}, CoV={cov:.3f}, vl_l10={last_n_mean(vl,10):.4f}, vl_max={vl.max() if len(vl) else float('nan'):.2f}, ER_l10={last_n_mean(er,10):.1f}, r_load_f={last_n_mean(rl,10):.1f}, r_transfer_f={last_n_mean(rt,10):.3f}")

tr_means_34 = [last_n_mean(d34[s].get("Performance/soil_transfer_ratio", np.array([])), 5) for s in v34_seeds]
kl_means_34 = [last_n_mean(d34[s].get("Policy/approx_kl", np.array([])), 10) for s in v34_seeds]
cf_means_34 = [last_n_mean(d34[s].get("Policy/clip_fraction", np.array([])), 10) for s in v34_seeds]
ev_finals_34 = [d34[s].get("Policy/explained_variance", np.array([float('nan')]))[-1] for s in v34_seeds]
covs_34 = []
for s in v34_seeds:
    er = d34[s].get("Performance/episodic_return", np.array([]))
    er_std = d34[s].get("Performance/episodic_return_std", np.array([]))
    if len(er) >= 10 and len(er_std) >= 10 and abs(er[-10:].mean()) > 1e-6:
        covs_34.append(er_std[-10:].mean() / abs(er[-10:].mean()))

tr_3m4 = np.nanmean(tr_means_34)
kl_3m4 = np.nanmean(kl_means_34)
cf_3m4 = np.nanmean(cf_means_34)
ev_3m4 = np.nanmean(ev_finals_34)
cov_3m4 = np.mean(covs_34) if covs_34 else float('nan')
print(f"\nv34 3-seed means: transfer_ratio={tr_3m4:.4f}, KL={kl_3m4:.2f}, clip={cf_3m4:.3f}, EV={ev_3m4:.3f}, CoV={cov_3m4:.3f}")
print(f"§5 PASS: tr>0.15 [{'PASS' if tr_3m4>0.15 else 'FAIL'}] | KL<1 [{'PASS' if kl_3m4<1 else 'FAIL'}] | clip<0.5 [{'PASS' if cf_3m4<0.5 else 'FAIL'}] | CoV<0.5 [{'PASS' if cov_3m4<0.5 else 'FAIL'}] | EV>0.5 [{'PASS' if ev_3m4>0.5 else 'FAIL'}]")
n_pass4 = sum([tr_3m4>0.15, kl_3m4<1, cf_3m4<0.5, cov_3m4<0.5, ev_3m4>0.5])
print(f"v34 §5 PASS count: {n_pass4}/5")

print(f"\n=== v35 §5 PASS check (3-seed mean) ===")
v35_seeds = sorted(d35.keys())
for s in v35_seeds:
    tr = d35[s].get("Performance/soil_transfer_ratio", np.array([]))
    kl = d35[s].get("Policy/approx_kl", np.array([]))
    cf = d35[s].get("Policy/clip_fraction", np.array([]))
    ev = d35[s].get("Policy/explained_variance", np.array([]))
    er = d35[s].get("Performance/episodic_return", np.array([]))
    er_std = d35[s].get("Performance/episodic_return_std", np.array([]))
    sb = d35[s].get("reward/success_bonus", np.array([]))
    rl = d35[s].get("reward/load", np.array([]))
    rt = d35[s].get("reward/transfer", np.array([]))
    vl = d35[s].get("Policy/value_loss", np.array([]))
    cov = (er_std[-10:].mean() / abs(er[-10:].mean())) if len(er_std) >= 10 and len(er) >= 10 and abs(er[-10:].mean()) > 1e-6 else float('nan')
    print(f"  seed {s}: tr_l5={last_n_mean(tr,5):.4f} (max={tr.max() if len(tr) else 0:.4f}), KL_l10={last_n_mean(kl,10):.2f}, clip_l10={last_n_mean(cf,10):.3f}, EV_f={ev[-1] if len(ev) else float('nan'):.3f}, CoV={cov:.3f}, vl_l10={last_n_mean(vl,10):.4f}, ER_l10={last_n_mean(er,10):.1f}, r_load_f={last_n_mean(rl,10):.1f}")

tr_means_35 = [last_n_mean(d35[s].get("Performance/soil_transfer_ratio", np.array([])), 5) for s in v35_seeds]
kl_means_35 = [last_n_mean(d35[s].get("Policy/approx_kl", np.array([])), 10) for s in v35_seeds]
cf_means_35 = [last_n_mean(d35[s].get("Policy/clip_fraction", np.array([])), 10) for s in v35_seeds]
ev_finals_35 = [d35[s].get("Policy/explained_variance", np.array([float('nan')]))[-1] for s in v35_seeds]
covs_35 = []
for s in v35_seeds:
    er = d35[s].get("Performance/episodic_return", np.array([]))
    er_std = d35[s].get("Performance/episodic_return_std", np.array([]))
    if len(er) >= 10 and len(er_std) >= 10 and abs(er[-10:].mean()) > 1e-6:
        covs_35.append(er_std[-10:].mean() / abs(er[-10:].mean()))

tr_3m5 = np.nanmean(tr_means_35)
kl_3m5 = np.nanmean(kl_means_35)
cf_3m5 = np.nanmean(cf_means_35)
ev_3m5 = np.nanmean(ev_finals_35)
cov_3m5 = np.mean(covs_35) if covs_35 else float('nan')
print(f"\nv35 3-seed means: transfer_ratio={tr_3m5:.4f}, KL={kl_3m5:.2f}, clip={cf_3m5:.3f}, EV={ev_3m5:.3f}, CoV={cov_3m5:.3f}")
print(f"§5 PASS: tr>0.15 [{'PASS' if tr_3m5>0.15 else 'FAIL'}] | KL<1 [{'PASS' if kl_3m5<1 else 'FAIL'}] | clip<0.5 [{'PASS' if cf_3m5<0.5 else 'FAIL'}] | CoV<0.5 [{'PASS' if cov_3m5<0.5 else 'FAIL'}] | EV>0.5 [{'PASS' if ev_3m5>0.5 else 'FAIL'}]")
n_pass5 = sum([tr_3m5>0.15, kl_3m5<1, cf_3m5<0.5, cov_3m5<0.5, ev_3m5>0.5])
print(f"v35 §5 PASS count: {n_pass5}/5")

print(f"\n=== v36 §5 PASS check (3-seed mean) ===")
v36_seeds = sorted(d36.keys())
for s in v36_seeds:
    tr = d36[s].get("Performance/soil_transfer_ratio", np.array([]))
    kl = d36[s].get("Policy/approx_kl", np.array([]))
    cf = d36[s].get("Policy/clip_fraction", np.array([]))
    ev = d36[s].get("Policy/explained_variance", np.array([]))
    er = d36[s].get("Performance/episodic_return", np.array([]))
    er_std = d36[s].get("Performance/episodic_return_std", np.array([]))
    sb = d36[s].get("reward/success_bonus", np.array([]))
    rl = d36[s].get("reward/load", np.array([]))
    rt = d36[s].get("reward/transfer", np.array([]))
    vl = d36[s].get("Policy/value_loss", np.array([]))
    cov = (er_std[-10:].mean() / abs(er[-10:].mean())) if len(er_std) >= 10 and len(er) >= 10 and abs(er[-10:].mean()) > 1e-6 else float('nan')
    print(f"  seed {s}: tr_l5={last_n_mean(tr,5):.4f} (max={tr.max() if len(tr) else 0:.4f}), KL_l10={last_n_mean(kl,10):.2f}, clip_l10={last_n_mean(cf,10):.3f}, EV_f={ev[-1] if len(ev) else float('nan'):.3f}, CoV={cov:.3f}, vl_l10={last_n_mean(vl,10):.4f}, ER_l10={last_n_mean(er,10):.1f}")

tr_means_36 = [last_n_mean(d36[s].get("Performance/soil_transfer_ratio", np.array([])), 5) for s in v36_seeds]
kl_means_36 = [last_n_mean(d36[s].get("Policy/approx_kl", np.array([])), 10) for s in v36_seeds]
cf_means_36 = [last_n_mean(d36[s].get("Policy/clip_fraction", np.array([])), 10) for s in v36_seeds]
ev_finals_36 = [d36[s].get("Policy/explained_variance", np.array([float('nan')]))[-1] for s in v36_seeds]
covs_36 = []
for s in v36_seeds:
    er = d36[s].get("Performance/episodic_return", np.array([]))
    er_std = d36[s].get("Performance/episodic_return_std", np.array([]))
    if len(er) >= 10 and len(er_std) >= 10 and abs(er[-10:].mean()) > 1e-6:
        covs_36.append(er_std[-10:].mean() / abs(er[-10:].mean()))

tr_3m6 = np.nanmean(tr_means_36)
kl_3m6 = np.nanmean(kl_means_36)
cf_3m6 = np.nanmean(cf_means_36)
ev_3m6 = np.nanmean(ev_finals_36)
cov_3m6 = np.mean(covs_36) if covs_36 else float('nan')
print(f"\nv36 3-seed means: transfer_ratio={tr_3m6:.4f}, KL={kl_3m6:.2f}, clip={cf_3m6:.3f}, EV={ev_3m6:.3f}, CoV={cov_3m6:.3f}")
print(f"§5 PASS: tr>0.15 [{'PASS' if tr_3m6>0.15 else 'FAIL'}] | KL<1 [{'PASS' if kl_3m6<1 else 'FAIL'}] | clip<0.5 [{'PASS' if cf_3m6<0.5 else 'FAIL'}] | CoV<0.5 [{'PASS' if cov_3m6<0.5 else 'FAIL'}] | EV>0.5 [{'PASS' if ev_3m6>0.5 else 'FAIL'}]")
n_pass6 = sum([tr_3m6>0.15, kl_3m6<1, cf_3m6<0.5, cov_3m6<0.5, ev_3m6>0.5])
print(f"v36 §5 PASS count: {n_pass6}/5")

print(f"\n=== V32 entropy-collapse diagnostic ===")
print(f"Action dim n=7. Diagonal-Gaussian entropy = sum_i [0.5*log(2*pi*e) + log(sigma_i)] = 7*1.4189 + sum_i log(sigma_i).")
print(f"At sigma=1.0 entropy=9.93. At sigma=0.1 entropy=-6.18. At sigma=0.01 entropy=-22.30. At sigma=0.001 entropy=-38.42.")
print(f"Implied per-dim sigma from entropy E: sigma ~= exp((E - 7*1.4189) / 7).\n")
import math
def implied_std(entropy):
    return math.exp((entropy - 7*1.4189) / 7) if entropy is not None and not math.isnan(entropy) else float('nan')

for label, d in [("v22 (no anchor)", d22), ("v28a (anchor=2)", d28a), ("v31 (anchor=2, fresh seeds, freeze)", d31), ("v33 (anchor=2, fresh seeds, NO freeze)", d33)]:
    print(f"\n{label}:")
    for s in sorted(d.keys()):
        ent = d[s].get("Policy/entropy", np.array([]))
        vl = d[s].get("Policy/value_loss", np.array([]))
        pl = d[s].get("Policy/policy_loss", np.array([]))
        lr = d[s].get("Policy/learning_rate", np.array([]))
        kl = d[s].get("Policy/approx_kl", np.array([]))
        ev = d[s].get("Policy/explained_variance", np.array([]))
        if len(ent) == 0:
            print(f"  seed {s}: no entropy data")
            continue
        ent_l10 = ent[-10:].mean() if len(ent) >= 10 else ent.mean()
        vl_l10 = vl[-10:].mean() if len(vl) >= 10 else (vl.mean() if len(vl) else float('nan'))
        vl_max = vl.max() if len(vl) else float('nan')
        pl_l10 = pl[-10:].mean() if len(pl) >= 10 else (pl.mean() if len(pl) else float('nan'))
        lr_l10 = lr[-10:].mean() if len(lr) >= 10 else (lr.mean() if len(lr) else float('nan'))
        lr_max = lr.max() if len(lr) else float('nan')
        kl_l10 = kl[-10:].mean() if len(kl) >= 10 else (kl.mean() if len(kl) else float('nan'))
        ev_final = ev[-1] if len(ev) else float('nan')
        print(f"  seed {s}: ent_l10={ent_l10:+6.2f} (sig~{implied_std(ent_l10):.3f}), KL_l10={kl_l10:>10.2f}, EV_f={ev_final:+.2f}, value_loss_l10={vl_l10:>10.2f} (max={vl_max:>10.1f}), policy_loss_l10={pl_l10:+8.4f}, lr_l10={lr_l10:.2e} (max={lr_max:.2e})")
