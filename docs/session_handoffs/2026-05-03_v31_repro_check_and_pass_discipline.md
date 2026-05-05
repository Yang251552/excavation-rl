# Session Handoff — 2026-05-03 — v31 reproducibility check + PASS-discipline cleanup

This file is auto-injected at the start of every new Claude Code session via `.claude/settings.json` `SessionStart` hook (see CLAUDE.md §0). Read it once and acknowledge in your first user-facing reply, e.g. *"Picking up from 2026-05-03 v31-repro handoff: queued action is `<X>`."*

---

## 1. What this session completed (facts)

- **Backfilled all four queued tuning-log entries** from the previous handoff: V26, V27, V28a, V28b are now full entries in `docs/stage1_5_tuning_log.md` with Hypothesis / Change / Test / Result table / Decision / Next sections per CLAUDE.md §4.5. V27 entry has a process-violation note (was written *after* launching the run; flagged for future avoidance).
- **CLAUDE.md §5.1 + §5.2 added** (No "Partial PASS" / No "Best Seed" Reframing). Two new anti-patterns appended to §6 ("inventing partial-PASS to advance prematurely", "citing single-seed when 3-seed mean fails"). These rules came directly from the user catching me declaring v28a "Stage 1.5 partial-PASS" while only 1/5 §5 criteria held.
- **Reverted the wrong "v28a partial-PASS / advance to Stage 2" framing** in three docs:
  - `README.md` top status, Stage 1 Results section, TODO Stage 1.5 / Stage 2 entries — all rewritten to "v28a is FAIL per §5, Stage 2 is blocked on 3-seed mean clearing 0.15".
  - `docs/stage1_5_tuning_log.md` v28b decision section + new authoritative "Stage 1.5 Status (NOT concluded — v28a is FAIL per §5)" section. Original "Stage 1.5 Conclusion" preserved as historical record (per §4.5 entries-immutable rule), with strikethrough header and prefix note pointing to the new authoritative section.
- **Generated `results/figures/stage1/v17_vs_v22_vs_v28a_comparison.png`** via `scripts/plot_stage1_comparison.py` (which now loads v28a in addition to v17/v22). Bottom-right panel shows `Performance/soil_transfer_ratio` with random baseline (0.15) and demo-replay (0.224) reference lines. README's Stage 1 Results section embeds this figure.
- **Created `docs/stage0_tuning_log.md`** (149 lines) by code-archaeology backfill — explicit notice that early entries are reconstructed not contemporaneous. Documents v0 → v3 (the v3 PASS run that's also in `docs/stage0_results_appendix.md`). Mentions Stage-0-specific PASS criteria (different from Stage 1+ because no transfer signal exists in proxy mode).
- **Updated `docs/stage1_results_appendix.md`** to cover v17 / v22 / v26 / v27 / v28a / v28b: 4-campaign performance table, per-seed transfer table across all versions including the 1426× v28a spread, §5 PASS criteria check table (all FAIL), 18 wandb run URLs, v28a reproducibility command, v28b ablation note (marked "do not use as baseline").
- **v28b wandb analysis run** via updated `scripts/analyze_v18.py`. Confirmed KL=193 076, clip_fraction=0.995, EV=−0.22, transfer_mean=0.092 (bizarrely close to v28a despite catastrophic PPO health metrics — agent finds local optima through chaos).
- **v31 launched**: 3-seed × 500 iter reproducibility check on fresh seeds {500, 555, 999}, identical config to v28a (`--bc-anchor 2.0 --reverse-curriculum-prob 0.7 --snapshot-data results/scoop_snapshot.npz --bc-data results/bc_demos.npz --bc-steps 2000 --max-iterations 500`). v31 entry written in tuning log *before* launch (per §4.5).

## 2. Queued next actions (priority order)

**Do these first unless the user explicitly redirects.**

1. **Pull v31 results, write v31 result table in `docs/stage1_5_tuning_log.md`, then act on the v31 decision tree.** Wandb run IDs: `logs/stage1_v31_seed{500,555,999}.log` contain `View run` lines. Update `scripts/analyze_v18.py` to add a `v31` dict and rerun on EC2. The decision tree (already in the v31 entry):
   - ≥1/3 fresh seeds clear `transfer_ratio = 0.20` AND 6-seed mean (v28a + v31) ≥ 0.15 → §5 PASS check; if all 5 criteria pass, Stage 2.
   - 1-2/3 reach 0.20 but 6-seed mean < 0.15 → seed 123 was reproducible, proceed to **v32** (broader BC distribution: multiple heap geometries / EE start poses in `scripts/generate_bc_demos.py`).
   - 0/3 reach 0.20 → v28a's seed 123 was a tail event. Switch to **architecture investigation** (e.g. KL anchor on advantage *before* clip vs *after*); v32/v33 hyperparameter tuning is unlikely to help.
2. **If v31 lands in case (b) or (c) above**, write the v32 entry first, then run. The next versions are queued in `stage1_5_tuning_log.md` v31 entry's Next section.
3. **(Bookkeeping)** Stage 1.5 state across all versions is now consolidated in `docs/stage1_5_tuning_log.md`. If a future session finds a v25/v26/v27/v28a/v28b table inconsistency between this file, the appendix, and the README, the tuning log is the source of truth.

## 3. Decisions that should not be re-litigated

- **CLAUDE.md §5.1 / §5.2 are now in force.** PASS is binary AND across 5 criteria on 3-seed mean. There is no "partial PASS / near PASS / PASS with caveats" — those terms must not appear in any new tuning log entry, README section, or appendix. Single-seed numbers are not partial credit; the 3-seed mean is the criterion.
- **v28b's "freeze obs_rms after BC" is dead** (carried from previous handoff; re-confirmed). KL 193 076 / EV −0.22 / clip 0.995. Mechanism is fundamentally incompatible with PPO's exploration distribution drift past BC's training distribution.
- **v28a is FAIL** (1/5 §5 criteria satisfied: only `return_std/mean = 0.36 < 0.5`). Mean transfer 0.107 < 0.15 threshold. Citing "seed 123 reaches 28.5 %" to argue otherwise is the §5.2 anti-pattern.
- **Stage 2 is BLOCKED on Stage 1.5 PASS.** Do not re-enable the 3-stage curriculum, do not start DR work, do not edit `excavation_env_cfg.py:243-275` until 3-seed mean transfer clears 0.15.
- **Reverse-curriculum reset MUST replay particle state** (carried from previous handoff). Adding particle replay (v25+) is what made the mechanism produce any seed 28.5 % at all; do not regress.
- **Anchor coef sweep alone is not the next move** (carried from previous handoff). v26 (coef=0.5) and v28a (coef=2.0) bracket; v28a shows that strong anchor doesn't fix per-seed instability — seed 42 worsens with stronger anchor. v33 (coef sweep in {1.0, 3.0}) is queued but lower priority than v31 (reproducibility) and v32 (broader demos).

## 4. v17 / v22 / v26 / v27 / v28a / v28b summary (3-seed final-window means, for reference)

Pull via `scripts/analyze_v18.py` for fresh numbers; static snapshot below.

| Metric | v17 | v22 | v26 (a=0.5) | v27 (300iter, a=2) | **v28a (500iter, a=2)** | v28b (freeze) |
|---|---|---|---|---|---|---|
| `transfer_ratio` (last-5-window 3-seed mean) | 0.014 | 0.024 | 0.012 | 0.098 | **0.107** | 0.092 |
| `transfer_ratio` (best seed final) | 0.029 (s123) | 0.042 (s42) | 0.020 (s123) | 0.267 (s123) | **0.285 (s123)** | similar |
| `approx_kl` (final 10 mean) | 6.67 | **0.44** | 34.82 | 17.04 | 14.27 | **193 076** |
| `clip_fraction` (final 10) | 0.96 | **0.33** | 0.94 | 0.84 | 0.90 | 0.995 |
| `explained_variance` (final) | 0.45 | 0.45 | 0.29 | 0.16 | 0.26 | **−0.22** |
| `episodic_return_std / mean` | 0.41 | 0.37 | — | 0.51 | **0.36** ✅ | 0.49 |
| `reward/transfer` (per-iter final) | 0.10 | 0.19 | 0.36 | **1.50** | 0.62 | 0.92 |
| `reward/load` (per-iter final) | 0.14 | **390** | 329 | 246 | 118 | 235 |
| **§5 PASS** (1=mean, 2=KL, 3=clip, 4=CoV, 5=EV) | 0/5 | 2/5 (KL+clip) | 0/5 | 1/5 | **1/5 (CoV)** | 1/5 |

V28a per-seed `transfer_ratio` last-5-window mean: seed 7 = 0.034, seed 42 = 0.0002, seed 123 = 0.285 → **1426× spread** (the headline cross-seed-variance failure). Confirms v28a mechanism produces a working policy on at least one seed but does not robustly attract all seeds — v31's job is to determine whether seed 123 is reproducible-across-fresh-seeds (mechanism reliable in some basin) or a tail outlier (mechanism unreliable, redesign needed).

## 5. v32+ candidate directions (not yet started, all gated on v31 outcome)

1. **v32 — Broader BC demo distribution.** `scripts/generate_bc_demos.py` currently uses one heap geometry (cone, fixed center) and one EE start pose (the default joint config). Add randomization: heap shape ∈ {cone, hemisphere, cylinder}, heap center within ±0.05 m, EE initial joint perturbation. Wider demo manifold → BC sees more of the state distribution PPO will visit → less seed-dependent collapse. Direct attack on cross-seed variance root cause. **Run only if v31 case (b)** (1-2/3 fresh seeds reach 0.20).
2. **v33 — Adaptive KL anchor.** Anchor coef adjusts based on running KL (e.g. coef *= 1.5 if KL > target for N iter, /= 1.5 if KL < target/2). Better than static coef sweep because v28a shows static coef can't satisfy all seeds. Lower priority — v32 attacks the root cause more directly.
3. **(architecture) KL anchor placement**. Currently anchor is added to PPO's loss alongside the clipped surrogate. Alternative: move anchor inside the surrogate by replacing `policy_loss + bc_anchor * KL` with a Lagrangian where the anchor adjusts the surrogate's effective clip range. **Run only if v31 case (c)** (0/3 fresh seeds reach 0.20 — architecture problem confirmed).
4. **(diagnostic) Seed 42 collapse trajectory analysis** (carried from previous handoff). Compare seed 42's iter-0-to-100 trajectory against seed 123's. What diverges first — bucket-load fire rate? early KL spike? action variance? — tells which knob to turn.

## 6. Currently running / waiting

- **v31 in flight**: PIDs 25167 (seed 500), 25168 (seed 555), 25169 (seed 999) on EC2. Launched ~16:15 UTC, ETA ~13 min wall clock. **`ScheduleWakeup` is set for 16:30** with prompt to: pull run IDs from logs, update `analyze_v18.py` with v31 dict, write v31 result table in `stage1_5_tuning_log.md` BEFORE acting on the decision tree, then execute decision tree (§2 above).
- No other background runs.
- A10G is at ~80 % utilisation; do not launch v32 in parallel — wait for v31 to complete first.

## 7. References (read these first if context is thin)

- `docs/stage1_5_tuning_log.md` — Authoritative log for v25 → v31. Entries V26 / V27 / V28a / V28b are now full (backfilled from previous handoff queue). V31 entry has Hypothesis + Test + decision tree but no Result table yet (waiting on the run).
- `docs/stage1_results_appendix.md` — Updated this session to cover v17 / v22 / v26 / v27 / v28a / v28b. 4-campaign performance table, all wandb URLs, §5 PASS check (all FAIL).
- `docs/stage0_tuning_log.md` — Created this session (backfill notice at top). Documents v0 → v3 PASS for Stage 0.
- `docs/session_handoffs/2026-05-03_stage1_5_v28_analysis.md` — Previous handoff. §2 queued actions are all done in this session; §3 decisions still hold.
- `CLAUDE.md` §0 (resume protocol), §3 (bug cheat sheet, 10 entries), §4.5 (tuning log discipline — mandatory before next iteration), §5 + **§5.1 + §5.2** (PASS criteria, no partial PASS, no best-seed reframing), §7.5 (v22 artefacts and Stage 1.5 pickup points), §9 (consistency audit), §10 (this file's format).
- `scripts/analyze_v18.py` — Has v17 / v18 / v19 / v22 / v26 / v27 / v28a / v28b run-ID dicts; **add v31 dict** before next analysis.
- `scripts/plot_stage1_comparison.py` — Generates the README's headline figure (`results/figures/stage1/v17_vs_v22_vs_v28a_comparison.png`). Add v31 series after the v31 result table is filled, regenerate, link from README only if v31 changes the story (i.e. if PASS).

## 8. Unresolved threads (non-blocking but should be flagged)

- **Process-violation note**: V27 entry in `stage1_5_tuning_log.md` was written *after* launching the run (the v26 → v27 transition didn't pause to log). Not repeated for v28a/v28b/v31 — those entries were written before launch. Future sessions: the §4.5 discipline is enforceable only if you literally write the entry first, then `scp + ssh + nohup`. Don't reorder.
- **`scripts/plot_stage1_comparison.py` will not include v26/v27 series** even though they exist on wandb — the figure intentionally shows only the cleanest 3-version arc (v17 → v22 → v28a). If a future review asks for "all six campaigns overlaid", regenerate from a copy of the script with all six dicts; do not edit the canonical `plot_stage1_comparison.py` to include them by default (clutter trade-off).
- **Stage 2 work items in README §TODO** still reference "graduated penetration-depth `collision` penalty (replacing the binary `-15` that produced bimodal returns in stage 1 v5)" — this is the right Stage 2 plan but currently un-scoped. Worth a tuning-log brainstorm entry before Stage 2 actually starts (which is gated on Stage 1.5 PASS).
- **Architecture diagram captions** in README still say Isaac Lab is "target architecture, not current". When Phase 1 (Isaac Lab port) starts, update the captions and consider regenerating the PNG without the aspirational dashed-block. (Carried from previous handoff.)
