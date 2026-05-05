# Stage 1 Results — Appendix

Detailed tables, bug catalogue, wandb run URLs, and reproducibility commands for the Stage 1 v17 / v22 / v28a campaigns and the v28b ablation. The README's Stage 1 Results section keeps only the narrative + headline metrics; everything below is for readers who want to dig in or reproduce.

Per-iteration tuning history:
- v17 → v18 → … → v22 in `stage1_tuning_log.md`
- v25 → v26 → v27 → v28a → v28b → v31+ in `stage1_5_tuning_log.md`

**Stage status (per CLAUDE.md §5.1 / §5.2 — PASS is binary AND across 3 seeds)**: v22 is FAIL, v28a is FAIL (4/5 criteria unmet despite seed-123 reaching `transfer_ratio = 0.285`), v28b is FAIL (catastrophic KL=193 000), v31 is FAIL (transfer ✅ at 3-seed mean 0.267 but KL=585 408 / EV=−0.20 / clip=0.998 fail). Stage 1.5 continues; Stage 2 has not been entered. v31 settles the cross-seed reproducibility question (mechanism is reproducible — fresh seeds {500, 555, 999} cluster at 0.23–0.30) and reframes the bottleneck from cross-seed variance to PPO-health collapse during BC pretrain.

---

## Reward Used (Both Campaigns)

```
R_t =  10.00 · ΔP_target           (soil transfer to target zone)
     +  1.00 · exp(−2·d_heap)      (approach to static heap centre)
     +  0.50 · exp(−8·max(0, ee_z − 0.07))   (dig depth shaping)
     + 10.00 · L_bucket            (bucket load, normalised)
     +  3.00 · L_bucket·exp(−2·d_target)     (loaded transport)
     −  0.01 · ‖a_t − a_{t−1}‖²    (action smoothness)
     −  0.01                        (per-step time penalty)
```

Penalty components (`collision`, `joint_limit`) are zeroed in Stage 1 after diagnostic ablation showed the binary collision penalty produced bimodal episode returns (−335 ± 379 in v5) which broke advantage normalisation. They are scheduled to be reintroduced in Stage 2 as graduated penetration-depth signals once the base task is learned.

---

## Bugs Identified and Fixed During Stage 1 Bring-Up

The Stage 1 bring-up surfaced ten non-obvious bugs. The first three are fully blocking — with bug #1 alone, no version of stage 1 prior to v8 was actually simulating particles, regardless of the bucket-force or PPO tuning effort spent on those runs.

| # | Bug | File:Line | Effect | Resolution |
|---|---|---|---|---|
| 1 | `scene.build()` never called from env constructor | `envs/excavation_env.py:93` | `particle_system` stays `None`; `bucket_load`, `soil_centroid`, `transfer_ratio` are 0 every step regardless of policy | Added `self.scene.build()` after scene construction |
| 2 | `bucket_push_force` defaulted to `contact_damping = 1000 N` (unit confusion: damping coefficient vs force) | `soil/particle_system.py:248`, `soil/soil_properties.py:48` | One bucket–particle contact accelerates a 6 mg particle at ~160 000 m/s²; particles eject ~90 m per 20 steps | Added explicit `bucket_push_force: float = 1.0` field on `SoilProperties`; standalone sweep ablation (0.01–1000 N) verified 1 N produces the bucket-carries-particle regime |
| 3 | Missing observation normalisation on a 66-dim mixed-unit obs | `training/train.py` (added `RunningMeanStd` class) | `KL = 7–13` and `clip_fraction > 0.95` across versions 1–10, regardless of LR / entropy / penalty tuning | Added Welford-style running mean/var normaliser; KL dropped from ~10 to ~0.2 on smoke test |
| 4 | `CurriculumManager.get_reward_weight_overrides()` silently overrides explicit reward weights when curriculum is enabled (default) | `envs/curriculum.py`, observed in `envs/rewards.py:118-131` | Any user-set weight on a curriculum-managed component (`approach`, `transfer`) is ignored | Disabled curriculum in stage-1 bring-up; documented for stage 2 |
| 5 | Default `init_noise_std = 1.0` in `NetworkCfg` produces extremely wide action distribution at start | `training/ppo_cfg.py:29` | Stage 1 v1 entropy = 9.98 with rising trend; effectively random actions throughout training | Lowered to `0.5` for stage 1 |
| 6 | Default LR schedule `"adaptive"` collapses LR to floor on noisy reward signal | `training/ppo_cfg.py:53` | LR drops to ~5×10⁻⁷ in early iterations, freezing the policy | Set `schedule="fixed"` for stage 1 |
| 7 | Binary collision penalty (`-15`, then `-2`) produces bimodal episodic returns | `envs/excavation_env_cfg.py:134`, observed in v5 | `episodic_return = −335 ± 379` (std ≈ mean): the per-step PPO advantage normalisation by σ is meaningless on a bimodal distribution; gradient sign flips between mini-batches and policy oscillates | Set `collision = 0` for the bring-up; Stage 2 will reintroduce as `-k·max(0, −ee_z)` |
| 8 | Approach reward computed against **dynamic** soil centroid, which moves *away* from the bucket as the bucket displaces particles | `envs/excavation_env.py:246` | Successful scoop attempts decrease `approach` reward; PPO learns "stay back from heap" as the reward-maximising policy | Switched to **static** `heap_center` from scene config |
| 9 | Bucket geometry is a bare AABB with no walls; the contact kernel cannot retain particles through a lift — they fall out the bottom every step | `envs/scene.py:_update_carried_particles`, added | Even when the bucket reaches the heap and the kernel pushes particles into the AABB, the next step's gravity drops them back to the ground; `bucket_load` averaged 0.14 in v17 | Sticky-grab containment heuristic: particles inside the load-detection box are pinned to the bucket frame until the bucket reaches the target zone (auto-release). Boosted v22 `reward/load` to 390 (vs 0.14 in v17) and `reward/transport` to 43 (vs 0.02) |
| 10 | Aggressive `action_smoothness = -0.01` penalty was comparable in magnitude to the positive components (`smooth_penalty` mean = -10.9 in v17), suppressing the rapid joint motions a real scoop manoeuvre requires | `training/train.py:701`, stage 1 override | Agent converged to a "minimum-action" policy that approached the heap but did not commit to the high-acceleration descent + push needed to scoop | Lowered to -0.001; v18-v19 saw a +71 % uplift in `reward/transfer` from this one change |

---

## Full Multi-Seed Performance Table

The four campaigns differ in: (a) whether bug #9 (containment) and bug #10 (smoothness) are fixed (v22+ yes; v17 no); (b) whether a 2 000-step BC pretraining warm-start runs before PPO (v22+ yes); (c) whether the KL anchor on PPO updates is active (v28a/v28b yes); (d) whether reverse-curriculum reset with particle-state replay is active (v28a/v28b yes); (e) whether `RunningMeanStd` is frozen post-BC (only v28b — failed).

| Metric | v17 (pure PPO) | v22 (BC + containment) | **v28a (Stage 1.5)** | v28b (freeze ablation, FAIL) |
|---|---|---|---|---|
| `Performance/episodic_return` (final 10 mean) | 209 | **837** | 553–443 | 532 |
| `Performance/soil_transfer_ratio` (3-seed last-5-window mean) | 0.014 | 0.024 | **0.107** | 0.092 |
| `Performance/soil_transfer_ratio` (best single-seed final) | 2.9 % (seed 123) | 4.2 % (seed 42) | **28.5 %** (seed 123) | similar to v28a per seed |
| `Performance/soil_transfer_ratio` (single-window peak across run) | 9.2 % (seed 123) | 80.0 % (seed 42, transient) | **68.8 % across all 3 seeds** (transient) | also reaches 68.8 % |
| `Policy/approx_kl` (final 10 mean) | 6.67 | **0.44** | 14.27 | **193 076** ⚠️ |
| `Policy/clip_fraction` (final 10) | 0.96 | **0.33** | 0.90 | 0.995 |
| `Policy/explained_variance` (final) | 0.45 | 0.45 | 0.26 | **−0.22** ⚠️ |
| `reward/load` (per-iter, final) | 0.14 | **390** | 118 | 235 |
| `reward/transport` (per-iter, final) | 0.02 | **43** | 15 | 27 |
| `reward/transfer` (per-iter, final) | 0.10 | 0.19 | **1.50** (v27 short) / 0.62 (v28a long) | 0.92 |
| `episodic_return_std / mean` | 0.41 | 0.37 | **0.36** ✅ | 0.49 |

**Per-seed `transfer_ratio` (last-5-window mean), all campaigns**:

| Campaign | Seed 7 | Seed 42 | Seed 123 | 3-seed mean | Spread (max/min) |
|---|---|---|---|---|---|
| v17 | 0.0086 | 0.0047 | 0.0288 | **0.0140** | 6× |
| v22 | 0.0208 | 0.0420 | 0.0097 | **0.0242** | 4.3× |
| v26 (anchor=0.5) | 0.0046 | 0.0108 | 0.0197 | 0.0117 | 4.3× |
| v27 (anchor=2.0, 300 iter) | 0.0191 | 0.0076 | 0.2669 | 0.0979 | 35× |
| **v28a (anchor=2.0, 500 iter)** | **0.0342** | **0.0002** | **0.2854** | **0.1066** | **1426×** |

v31 (anchor=2.0, 500 iter, **fresh** seeds — same config as v28a, only seeds differ):

| Campaign | Seed 500 | Seed 555 | Seed 999 | 3-seed mean | Spread |
|---|---|---|---|---|---|
| **v31 (anchor=2.0, 500 iter, repro)** | **0.3012** | **0.2717** | **0.2275** | **0.2668** | **1.3×** |

v31 settles the v28a spread question: seed-123's 0.285 was **not a tail event** — it was the representative basin. v28a's seeds 7 and 42 were the cross-seed outliers. The transfer signal is mechanistically reproducible (3/3 fresh seeds clear 0.20). 6-seed combined mean across (v28a + v31) = **0.1867**, which clears the 0.15 random baseline.

But v31 also surfaces a new failure: **PPO health degrades catastrophically across seed groups under identical config**. KL drifted from 14 (v28a) to 585 408 (v31), clip_fraction = 0.998, EV = −0.20. Same pathology as v28b. Likely mechanism: action_std collapses to ~0 during the 2 000-step BC pretrain, so analytic-KL between near-deterministic Gaussians explodes on every PPO update, while the policy mean stays pinned near BC's faithful scoop replication (demo replay = 0.224, v31 = 0.267 → only +0.04 above pure replay). To be diagnosed in v32 (entropy / action_std trajectory pull) before v33 (entropy floor + softer anchor) is run.

**§5 PASS criteria check** (CLAUDE.md §5 — PASS is binary, AND across all 5 criteria, computed on 3-seed mean):

| Criterion | Threshold | v17 | v22 | v28a | **v31** |
|---|---|---|---|---|---|
| `Performance/soil_transfer_ratio` (3-seed mean) | > 0.15 | ❌ 0.014 | ❌ 0.024 | ❌ 0.107 | ✅ **0.267** |
| `Policy/approx_kl` (final 50 mean) | < 1.0 | ❌ 6.67 | ✅ 0.44 | ❌ 14.27 | ❌ **585 408** |
| `Policy/clip_fraction` (final 50 mean) | < 0.5 | ❌ 0.96 | ✅ 0.33 | ❌ 0.90 | ❌ 0.998 |
| `episodic_return_std / mean` | < 0.5 | ✅ 0.41 | ✅ 0.37 | ✅ 0.36 | ✅ 0.468 |
| `Policy/explained_variance` (final) | > 0.5 | ❌ 0.45 | ❌ 0.45 | ❌ 0.26 | ❌ −0.20 |
| **Result** | AND of all 5 | **FAIL (1/5)** | **FAIL (3/5)** | **FAIL (1/5)** | **FAIL (2/5)** |

No campaign has cleared §5 PASS. Per CLAUDE.md §5.1 there is no "partial PASS" — Stage 1.5 work continues until all 5 criteria clear on a 3-seed mean. v31 is the first campaign to clear the transfer threshold but trades it for a 5-OOM KL regression vs v28a — the next iteration must restore PPO health while preserving the transfer signal.

---

## Wandb Run URLs

| Run | Wandb URL |
|---|---|
| v17 seed 7 | https://wandb.ai/yangchenghan2515-eth-z-rich/excavation-rl/runs/wnmqx350 |
| v17 seed 42 | https://wandb.ai/yangchenghan2515-eth-z-rich/excavation-rl/runs/x1nvjpgq |
| v17 seed 123 | https://wandb.ai/yangchenghan2515-eth-z-rich/excavation-rl/runs/mtbrd8u4 |
| v22 seed 7 | https://wandb.ai/yangchenghan2515-eth-z-rich/excavation-rl/runs/auek7u4r |
| v22 seed 42 | https://wandb.ai/yangchenghan2515-eth-z-rich/excavation-rl/runs/cc68bjg9 |
| v22 seed 123 | https://wandb.ai/yangchenghan2515-eth-z-rich/excavation-rl/runs/bc64lv3c |
| v26 (anchor=0.5) seed 7 | https://wandb.ai/yangchenghan2515-eth-z-rich/excavation-rl/runs/u7lajnuc |
| v26 seed 42 | https://wandb.ai/yangchenghan2515-eth-z-rich/excavation-rl/runs/ce7vxeo4 |
| v26 seed 123 | https://wandb.ai/yangchenghan2515-eth-z-rich/excavation-rl/runs/6cf3hfdl |
| v27 (anchor=2.0, 300 iter) seed 7 | https://wandb.ai/yangchenghan2515-eth-z-rich/excavation-rl/runs/qi76fx68 |
| v27 seed 42 | https://wandb.ai/yangchenghan2515-eth-z-rich/excavation-rl/runs/20i7ab3t |
| v27 seed 123 | https://wandb.ai/yangchenghan2515-eth-z-rich/excavation-rl/runs/13fptlp2 |
| **v28a (anchor=2.0, 500 iter) seed 7** | https://wandb.ai/yangchenghan2515-eth-z-rich/excavation-rl/runs/yu9hdybj |
| **v28a seed 42** | https://wandb.ai/yangchenghan2515-eth-z-rich/excavation-rl/runs/qzl7ev8t |
| **v28a seed 123** | https://wandb.ai/yangchenghan2515-eth-z-rich/excavation-rl/runs/44e5gk2z |
| v28b (freeze ablation, FAIL) seed 7 | https://wandb.ai/yangchenghan2515-eth-z-rich/excavation-rl/runs/cwks4rhw |
| v28b seed 42 | https://wandb.ai/yangchenghan2515-eth-z-rich/excavation-rl/runs/s9tv3dax |
| v28b seed 123 | https://wandb.ai/yangchenghan2515-eth-z-rich/excavation-rl/runs/vsc0u0k3 |
| **v31 (anchor=2.0, 500 iter, fresh seeds — repro check) seed 500** | https://wandb.ai/yangchenghan2515-eth-z-rich/excavation-rl/runs/4ezad4vf |
| **v31 seed 555** | https://wandb.ai/yangchenghan2515-eth-z-rich/excavation-rl/runs/b637zru6 |
| **v31 seed 999** | https://wandb.ai/yangchenghan2515-eth-z-rich/excavation-rl/runs/vi7np6d8 |

---

## What Stage 1 Validates — and What It Doesn't

| ✓ Validated | ✗ Not Validated by Stage 1 |
|---|---|
| Warp GPU particle simulation runs end-to-end with the env (FPS ≈ 700 with 500 particles × 32 envs on A10G); 3 seeds run in parallel on a single A10G at ~80 % utilisation | Whether the system reaches §5 PASS on a 3-seed mean basis. v31 (3-seed mean 0.267) is the first campaign to clear the 0.15 transfer threshold, but fails the other 3/5 PPO-health criteria (KL=585 408, clip=0.998, EV=−0.20). The 22 % hand-coded demo replay proves the task is structurally solvable in this physics; the new gap is PPO-health collapse during BC pretrain |
| Bucket–particle force kernel produces correct horizontal displacement (`+373 mm` mean over a 20-step sweep at `bucket_push_force = 1 N`, `tests/standalone_bucket_sweep.md`) | A robust closed-loop policy: BC pretraining converges to `mse_loss = 0.002` in 2 000 steps but the deterministic BC policy scores 0 % transfer on fresh seeds — state errors compound off-trajectory after ~20 steps |
| Observation normalisation (`RunningMeanStd`) keeps PPO's trust region intact when active (v22 KL = 0.4 vs v17 KL = 6.7); the v28b ablation confirms freezing it post-BC explodes KL to 193 000 (PPO inevitably visits states the BC obs distribution did not cover) | Cross-seed reproducibility was the v28a-era open question; v31 settles it — fresh seeds {500, 555, 999} all cluster at 0.23–0.30 (1.3× spread), so v28a's seed-123 was the representative basin and seeds 7/42 were the cross-seed outliers. The remaining un-validated piece is now PPO-health stability: v31 produces transfer via faithful BC replication (only +0.04 above demo replay = 0.224) while KL/clip/EV degenerate, suggesting action_std collapses during BC pretrain. v32 (entropy / action_std diagnostic) and v33 (entropy floor + softer anchor) are queued |
| Per-component reward decomposition logged in wandb (10 components after adding `dig` and the BC-enabled `reward/load` rising 2 600× from 0.14 to 390 in v22; `reward/transfer` rising 8× to 1.50 in v27) | Whether the bucket-as-cup containment heuristic generalises beyond this scene (currently a hardcoded AABB pin in `scene._update_carried_particles`; auto-release fires on bucket-xy entering target zone). It is intentional scaffolding for the Isaac Lab mesh-particle PhysX integration |
| BC warm-start (`--bc-data`, `--bc-steps`), KL anchor (`--bc-anchor`), reverse-curriculum reset with particle-state replay (`--reverse-curriculum --snapshot-data`) all wired into the training loop and CLI; full snapshot of joint state + particle pos/vel + carried-particle dict at every demo step recorded in `results/scoop_snapshot.npz` | Whether an entropy / action_std floor (clamped during PPO updates) combined with a softer KL anchor can keep PPO's trust region intact while preserving v31's transfer signal. v32 (action_std diagnostic) and v33 (entropy floor + softer anchor) are queued in `stage1_5_tuning_log.md` |

---

## Reproducibility

V17 (pure PPO, no BC, no containment — replicate the bug-fix-only baseline):

```bash
# Make sure scripts/scoop_demo.py and BC are not invoked, and revert
# scene._update_carried_particles to a no-op for true v17 reproduction
for SEED in 7 42 123; do
    PYTHONUNBUFFERED=1 nohup python -m training.train --stage 1 --wandb \
        --wandb-project excavation-rl \
        --wandb-run-name stage1_v17_500iter_seed${SEED} --seed ${SEED} \
        --max-iterations 500 --wandb-tags multi-seed-stage1 \
        > logs/stage1_v17_seed${SEED}.log 2>&1 &
done
```

V22 (BC pretraining + PPO + containment — note: the v22 demo set was generated with `N_EPISODES = 32` and `ACTION_NOISE_STD = 0.02`; before reproducing, edit `scripts/generate_bc_demos.py:21,23` back to those values, otherwise the freshly-generated `bc_demos.npz` will be the v25+ 64-episode 0.05-noise variant):

```bash
# 1. Generate 32 demo episodes (~6.4k (obs, action) pairs) -- v22 era
python scripts/generate_bc_demos.py

# 2. Train 3 seeds in parallel: 2000 BC steps + 500 PPO iter each
for SEED in 7 42 123; do
    PYTHONUNBUFFERED=1 nohup python -m training.train --stage 1 --wandb \
        --wandb-project excavation-rl \
        --wandb-run-name stage1_v22_500iter_bc_seed${SEED} --seed ${SEED} \
        --max-iterations 500 --bc-data results/bc_demos.npz --bc-steps 2000 \
        --wandb-tags multi-seed-v22 bc \
        > logs/stage1_v22_seed${SEED}.log 2>&1 &
done
```

V28a (Stage 1.5: BC + KL anchor + reverse-curriculum reset with particle-state replay — current best campaign, FAIL on 4/5 §5 criteria):

```bash
# 1. Generate v28a demos (64 episodes, 0.05 action noise, with snapshot)
#    Currently the default in scripts/generate_bc_demos.py:21,23
python scripts/generate_bc_demos.py
#    -> writes results/bc_demos.npz AND results/scoop_snapshot.npz

# 2. Train 3 seeds in parallel: BC pretrain + 500 PPO iter, KL anchor + particle replay
for SEED in 7 42 123; do
    PYTHONUNBUFFERED=1 nohup python -m training.train --stage 1 --wandb \
        --wandb-project excavation-rl \
        --wandb-run-name stage1_v28a_500iter_anchor2_seed${SEED} --seed ${SEED} \
        --max-iterations 500 \
        --bc-data results/bc_demos.npz --bc-steps 2000 --bc-anchor 2.0 \
        --reverse-curriculum --reverse-curriculum-prob 0.7 \
        --snapshot-data results/scoop_snapshot.npz \
        --wandb-tags multi-seed-v28a stage1.5 long anchor2 \
        > logs/stage1_v28a_seed${SEED}.log 2>&1 &
done
```

V28b (FAIL ablation: same as v28a + freeze obs_rms after BC). Reproduces the catastrophic KL=193 000 result documented in `stage1_5_tuning_log.md`. **Do not use as a baseline** — included for ablation completeness.

```bash
# Requires editing training/train.py:bc_pretrain to call
# self.actor_critic.obs_rms.freeze() after the deepcopy of BC policy.
# See the v28b entry in docs/stage1_5_tuning_log.md for the exact diff.
```

The 3 runs complete in approximately 12–15 minutes wall-clock on a `g5.xlarge` (4 vCPU, 16 GB, NVIDIA A10G) instance; each consumes ~1 GB of GPU memory so the three seeds run in parallel on the single GPU at ~80 % utilisation. The BC pretraining step adds ~1 minute per seed.

---

## Pulling Reward Decomposition From Wandb

Aggregating the 10-component `reward/*` decomposition from the wandb summary is truncated past the first 9 keys (`+15 ...`). Use the wandb API directly:

```bash
python scripts/analyze_v18.py
```

The script in `scripts/analyze_v18.py` is the template that surfaced the v17 → v22 hidden gains in `reward/load` (0.14 → 390) and `reward/transport` (0.02 → 43), along with the per-seed transfer-ratio peak that revealed transient 80 % scoops in v22.
