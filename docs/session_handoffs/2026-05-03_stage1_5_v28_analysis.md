# Session Handoff — 2026-05-03 — Stage 1.5 v28 analysis & doc cleanup

This file is auto-injected at the start of every new Claude Code session via `.claude/settings.json` `SessionStart` hook (see CLAUDE.md §0). Read it once and acknowledge in your first user-facing reply, e.g. *"Picking up from 2026-05-03 handoff: queued action is …"*.

---

## 1. What this session completed (facts)

- **README slim refactor**: Stage 0 / Stage 1 Results sections rewritten to narrative + headline figure + appendix link. Detailed bug catalogue, metric tables, wandb URLs, reproducibility commands moved to `docs/stage0_results_appendix.md` and `docs/stage1_results_appendix.md`.
- **Two architecture figures inserted**: `results/figures/architecture/system_architecture.png` (replaces the ASCII diagram in §System Architecture) and `results/figures/architecture/excavation_env_structure.png` (opens §Environment Design). Both have caveat notes about Isaac Lab being "target architecture" not current.
- **Stage 0 figures regenerated** in matplotlib via wandb API (style-matched to Stage 1's `v17_vs_v22_comparison.png`); generator at `scripts/plot_stage0_figures.py`. Wandb runs: `xvlprb9g` (seed 7), `j1ri0uwt` (seed 42), `rmib0x5z` (seed 123).
- **Cross-doc consistency fixed (A/B/C audit)**: 56→66 dim obs, 8/9/10→10 component reward, "eight"→"ten" bugs, broken `#todo` anchor, soften "built on top of Isaac Lab" claim, Stage 1 max_iter table annotated, reward-table version notes moved to tuning log.
- **CLAUDE.md additions**:
  - §3 — replaced 9-row standalone bug table with 10-row cheat sheet pointing to `docs/stage1_results_appendix.md` as single source of truth.
  - §7.5 — v22 artefacts and Stage 1.5 pickup points (BC demos, containment heuristic, reverse-curriculum status, Stage 1.5 priority list).
  - §9 — README/CLAUDE.md consistency audit protocol (3-step audit, 5-bullet output cap, ownership map). Mandatory after structural edits, skipped for typos.
  - M1/M2/M3 fixes: success threshold disambiguation, "11 reward components" → "10 + _total aggregator", "80% headroom" → "80% utilisation", v18 template wording.
- **v28a / v28b wandb analysis** completed via `scripts/analyze_v18.py` (already updated with v28 run-IDs).

## 2. Queued next actions (priority order)

**Do these first unless the user explicitly redirects.**

1. **Backfill `docs/stage1_5_tuning_log.md` entries V26 / V27 / V28a / V28b.** The log currently stops at V26 stub. CLAUDE.md §4.5 explicitly forbids starting V29 before logging V28. Use the analysis numbers below.
2. **Update CLAUDE.md §3 bug #9 with the BC-warm-start caveat** discovered in v28b: freezing `obs_rms` is incompatible with BC + PPO when PPO exploration shifts the obs distribution far from BC's training distribution. v28b last-10 KL = 193 076 confirms this. The §3 fix description should add: "Caveat: do NOT freeze obs_rms when training resumes from a BC-pretrained policy — PPO will normalise novel obs against stale BC stats and KL explodes."

## 3. Decisions that should not be re-litigated

- **v28b's "freeze obs_rms after BC" approach is dead.** Don't propose it again. KL = 193 076 + EV = −0.22 + clip_fraction = 0.995. The mechanism is fundamentally incompatible with BC distribution shift.
- **Stage 2 entry is gated on**: 3-seed `transfer_ratio` mean ≥ 15 % AND CoV < 0.5. v28a is at 10.7 % mean / 1400× cross-seed spread — does not pass.
- **Reverse-curriculum reset MUST replay particle state**, not just joint state. Adding particle replay (v25+) is what made the mechanism actually work; do not regress.
- **Anchor coef sweep is not the next move**. v28a (coef=2.0) shows coef alone doesn't fix per-seed instability — seed 123 with KL=23 has best transfer; seed 42 with KL=6 has near-zero transfer. The bottleneck is direction of drift, not magnitude.

## 4. v28 quantitative summary (for tuning log backfill)

Final-window (last 10 log points) means across 3 seeds:

| Metric | v22 (BC + PPO) | v27 (300 iter, anchor=2) | **v28a (500 iter, anchor=2)** | v28b (500 iter, anchor=2 + freeze obs_rms) |
|---|---|---|---|---|
| `Performance/soil_transfer_ratio` (last 5 windows) | 2.42 % | 9.79 % | **10.66 %** | 9.21 % |
| `Performance/soil_transfer_ratio` per-seed last 5 | 2.1 / 4.2 / 1.0 % | 1.9 / 0.8 / 26.7 % | **3.4 / 0.02 / 28.5 %** | per-seed not tabulated |
| `Performance/soil_transfer_ratio` single-seed peak | 80.0 % (seed 42, transient) | 68.8 % | 68.8 % (sustained → seed 123 last5 = 28.5 %) | 68.8 % |
| `Policy/approx_kl` (last 10) | 0.44 | 17.04 | 14.27 | **193 076** |
| `Policy/approx_kl` per-seed last 10 | < 1 all seeds | s7=18.4, s42=7.7, s123=25.1 | s7=12.9, s42=6.3, s123=23.7 | catastrophic |
| `Policy/clip_fraction` (last 10) | 0.33 | 0.84 | 0.90 | 0.995 |
| `Policy/explained_variance` (final) | 0.45 | 0.16 | 0.26 | **−0.22** |
| `reward/load` (final) | 390.4 | 245.9 | 117.8 | 234.9 |
| `reward/transport` (final) | 42.8 | 32.3 | 15.5 | 26.8 |
| success_bonus fired (count / log points) per seed | 19/23, 25/41, 21/23 | 10/16, 9/13, 10/13 | 14/27, 9/23, 21/24 | not tabulated |

Key qualitative findings to record in tuning log:
- v28a is the **best result the project has produced**: 4.4× v22 mean transfer, single-seed sustained 28.5 % (vs v22's 80 % being transient). Mechanism works.
- BUT cross-seed variance went up, not down — seed 42 collapsed to 0.02 % even with the same hyperparameters that gave seed 123 28.5 %.
- v28a fails 4 of 5 PASS gates (CLAUDE.md §5): only return CoV < 0.5 passes; transfer mean < 15 %, KL > 1.0, clip_fraction > 0.5, explained_variance < 0.5 all fail.
- v28b fails all 5; the freeze-obs_rms hypothesis is disproved.

## 5. v29 candidate directions (in priority order, not yet started)

1. **Diagnose seed 42 collapse**: pull iter-0-to-100 trajectory of seed 42 vs seed 123 — what diverges? Is it reverse-curriculum reset hit-rate? early success_bonus frequency? action variance? This tells us whether v29 should target init or training dynamics.
2. **Adaptive KL anchor**: anchor coef adjusts based on running KL (e.g. coef *= 1.5 if KL > target for N iter, /= 1.5 if KL < target/2). Better than static coef sweep because v28a shows static coef can't satisfy all seeds.
3. **Seed 123 reproducibility**: rerun seed 123's full config 3× with different RNG seeds (e.g. 123_a, 123_b, 123_c) — is 28.5 % a property of the (config × env-state) interaction, or pure init luck? Drives whether to invest in init engineering.
4. **Broaden BC demo distribution**: more `ACTION_NOISE_STD` variants + multiple heap geometries in `scripts/generate_bc_demos.py`. Wider demo manifold should reduce per-seed sensitivity.

## 6. Currently running / waiting

- No background runs. v28a/v28b finished 2026-05-03 ~13:40 UTC.
- No `ScheduleWakeup` pending.

## 7. References (read these first if context is thin)

- `docs/stage1_5_tuning_log.md` — V25 / V26 entries (V27 / V28 to be backfilled per §2 above).
- `docs/stage1_results_appendix.md` — Authoritative Stage 1 bug catalogue (10 entries) + v17 / v22 metric tables + wandb URLs.
- `scripts/analyze_v18.py` — Includes v17 / v18 / v19 / v22 / v26 / v27 / v28a / v28b run-ID dicts; rerun on EC2 for fresh numbers.
- `scripts/scoop_demo.py` + `scripts/generate_bc_demos.py` — Hand-coded scoop trajectory (22.4 % per replay) and BC demo generator (12 800 pairs after v25 expansion).
- `CLAUDE.md` §3 (bug cheat sheet), §4.5 (tuning log discipline), §5 (PASS/FAIL gate criteria), §7.5 (v22 artefacts), §9 (consistency audit protocol).

## 8. Unresolved threads (non-blocking but should be flagged)

- README §Reward Function table still shows R6 / R8 = 0 (collision / joint_limit); Stage 2 plan to re-enable as graduated penetration depth has not been scoped concretely. Worth a tuning-log brainstorm before Stage 2 kicks off.
- `docs/stage0_tuning_log.md` does not exist; if Stage 0 is revisited (e.g. for Isaac Lab port in Phase 1), create per CLAUDE.md §4.5.
- The architecture diagram captions in README assert Isaac Lab is "target". When Phase 1 lands, update the captions and consider regenerating both PNGs without the aspirational block.
