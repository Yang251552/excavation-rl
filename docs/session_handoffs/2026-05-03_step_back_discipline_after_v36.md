# Session Handoff — 2026-05-03 — Step-back discipline added after V32-V36 dead-end

## 1. What this session completed (facts)

- **V32 diagnostic** — pulled `Policy/entropy`, `Policy/value_loss`, `Policy/policy_loss`, `Policy/learning_rate` for v22/v28a/v31 via `scripts/analyze_v18.py`. Falsified the action-std-collapse hypothesis (σ ≈ 0.50 stable across all 9 seeds × 3 campaigns; LR pinned at 5e-5 floor). **Found actual root cause** by reading `training/train.py:702-706`: `bc_pretrain()` had an unconditional `obs_rms.freeze()` block that the V28b "REVERT freeze" Decision had mandated removing but never did. v31 silently ran on v28b's code path.
- **V33** — removed the unconditional freeze block from `training/train.py:bc_pretrain` (lines 693-706 replaced with explanatory comment). KL: 585 408 → 56 (4 OOM). value_loss max: 47 515 → 626. Wandb runs `l01moel9` / `um9a8te5` / `ykqfc9za`. §5 PASS 2/5.
- **V34** — added `return_rms = RunningMeanStd(num_obs=1)` to `Trainer.__init__`; updated after `compute_returns_and_advantages`; scales `values` / `old_values` / `returns` by `1/sqrt(return_rms.var + 1e-8)` inside the value loss. value_loss numerics now O(1) (max 8.15) but §5 PASS regressed to 1/5 (CoV broke 0.5). Wandb `voaf8iza` / `oqc8mxs7` / `r9sqjj67`.
- **V35** — `--bc-anchor 2.0 → 1.0`. KL 60 → 39 (−35 %), transfer 0.275 → 0.266 (held), CoV back to PASS. §5 PASS 2/5. Wandb `7utp05yg` / `4g64ujvs` / `gsul9iw1`.
- **V36** — `--bc-anchor 1.0 → 0.5`. KL 39 → 51 (went BACK UP, non-monotonic), transfer 0.266 → 0.239 (slight drop), EV 0.076 → −0.013. §5 PASS 2/5. Wandb `vgiuq0zt` / `vgsu0xkx` / `d0kckokl`. **Confirmed**: anchor coef ∈ {0.5, 1.0, 2.0} all give KL in 40-60 band; coef magnitude is not a §5-PASS lever.
- **V32 / V33 / V34 / V35 / V36 entries** appended to `docs/stage1_5_tuning_log.md` per CLAUDE.md §4.5. "Stage 1.5 Status" section (line ~377) rewritten to consolidate V32-V36 trajectory.
- **`training/train.py`** has two persistent code changes kept active: (a) `bc_pretrain()` no longer freezes `obs_rms`; (b) `return_rms` added in `__init__` line 355, updated line 419, used in value loss lines 514-528.
- **`scripts/analyze_v18.py`** updated with `v33` / `v34` / `v35` / `v36` dicts and per-version §5 PASS-check blocks. `Policy/entropy` + value_loss + policy_loss + learning_rate added to keys.
- **CLAUDE.md workflow update** to encode the V32-V36 lesson: new **§4.6 Step-Back Discipline** (sweep-stop rule + big-direction sanity check + death-spiral detector); new **step 0** in §4 Standard Iteration Loop pointing at §4.6; **4 new anti-patterns** in §6 (iterating on same knob without movement, hypothesis without external grounding, treating smoke as long-run predictor, auto-iterating across multi-FAIL); §8 split into **§8.A single-version stuck** vs **§8.B multi-version stuck** (B mandates literature pass + gap analysis + threshold sanity *before* any further training run).
- **§10 rewritten by user externally** during session. Handoff is now user-triggered only; auto-trigger conditions removed; format moved to `docs/handoff_template.md`.
- **Memory file** `feedback_readme_decision_judgment.md` saved — 3-question test for promoting a version to README (user-endorsed after v31 promotion).
- **Deleted** `docs/session_handoffs/2026-05-03_v32_v36_freeze_revert_anchor_sweep.md` — auto-generated under old §10 trigger; new §10 disallows auto-write.

## 2. Queued next actions (priority order)

1. **§4.6.2 Big-direction sanity check before V37** — do NOT launch V37 directly. Per the new CLAUDE.md §4.6.2 + §8.B (which fired because V32-V36 was a death spiral), the next action is read-only investigation:
   - **(2.1) Literature pass**: WebSearch + WebFetch on BC + PPO fine-tuning for robot manipulation. Reference papers: DAPG (Rajeswaran 2017), AWAC (Nair 2020), RLPD (Ball 2023), ROT (Haldar 2022), robosuite-imitation. Goal: report typical KL / EV / clip_fraction ranges they show on similar benchmarks; identify the dominant mechanism (demo mixing? KL anchor? entropy schedule?).
   - **(2.2) Codebase-vs-standard gap analysis** (Explore subagent): enumerate every "self-invented" mechanism in our codebase (BC anchor, particle replay, containment heuristic, return-rms, obs-rms freeze/no-freeze, reverse-curriculum reset). For each row label `match` / `differ-on-purpose` / `differ-without-justification`. The third column is the redesign surface.
   - **(2.3) §5 threshold sanity check**: KL < 1.0 was inherited from `rsl_rl` default. Confirm whether BC-anchored PPO papers routinely report KL > 1.0 on similar tasks. If yes, propose threshold edit *before* further runs (log the threshold change as its own decision in `stage1_5_tuning_log.md`).
2. **Decide V37 form** — only after (2.1)–(2.3) complete:
   - If literature shows BC + reverse-curriculum-replay alone (no anchor) is viable → V37 = `--bc-anchor 0` as originally queued.
   - If literature favors DAPG-style demo mixing → propose redesign rather than V37.
   - If §5 thresholds were inherited and miscalibrated for this task family → edit thresholds first; some prior version may already PASS under corrected thresholds.
3. **README + appendix lag** (bookkeeping) — both lag the tuning log by V33-V36. Update once V37 (or its replacement) reaches 5/5 PASS, in the same edit pass per CLAUDE.md §9 audit. Don't update for FAIL versions per the user-endorsed 3-question test in memory.

## 3. Decisions that should not be re-litigated

- **The v28b "REVERT freeze" decision is now actually executed in code.** `training/train.py:bc_pretrain` does NOT freeze `obs_rms`. Re-adding the freeze is forbidden unless behind an explicit `--freeze-obs-rms-after-bc` CLI flag. Evidence: V32 diagnostic + V33 result (KL 585k → 56 with the freeze removed and nothing else changed).
- **Return normalization (V34) stays active.** Standard PPO best practice (Andrychowicz 2020, "37 Implementation Details" #5). Numerically clean (value_loss O(1)), net-neutral on §5. Reverting forbidden absent specific load-bearing-conflict evidence.
- **Anchor coef sweep ∈ [0.5, 2.0] is exhausted.** KL trajectory 60 → 39 → 51 across V34/V35/V36 is non-monotonic. clip_grad_norm = 1.0 renormalizes total gradient regardless of anchor coef magnitude, so coef is not a §5-PASS lever in this regime. Do NOT propose anchor 0.7 / 1.5 / 0.25 / 0.1 etc — V37 (anchor 0) is the only un-tested point in this design space.
- **CLAUDE.md §4.6 Step-Back Discipline is in force.** Future iterations must apply §4.6 step 0 before each new tuning version. The death-spiral detector (§4.6.3) is a passive guard that fires after 3 consecutive FAILs with §5 PASS count flat — when it fires, switch to §4.6.2 (literature + gap analysis), not another local tweak. V32-V36 is the cautionary example.
- **§5.1 binary PASS / §5.2 no best-seed reframing still in force** (carried from prior handoff).
- **Stage 2 BLOCKED on §5 5/5 PASS.**

## 4. Quantitative summary

3-seed final-window means. Pull via `scripts/analyze_v18.py` for fresh numbers; static snapshot below.

| Metric | v22 | v28a | v31 (broken freeze) | V33 (freeze removed) | V34 (+return-rms) | V35 (anchor 1.0) | V36 (anchor 0.5) |
|---|---|---|---|---|---|---|---|
| `transfer_ratio` 3-seed mean | 0.024 | 0.107 | 0.267 | 0.248 | 0.275 | **0.266** | 0.239 |
| `Policy/approx_kl` last 10 | 0.4 ✅ | 14.3 | 585 408 | 55.6 | 60.2 | 39.4 | 50.7 |
| `Policy/clip_fraction` last 10 | 0.33 ✅ | 0.90 | 0.998 | 0.987 | 0.980 | 0.970 | 0.964 |
| `Policy/explained_variance` final | 0.42 | 0.26 | −0.20 | 0.039 | 0.037 | **0.076** | −0.013 |
| `Policy/value_loss` last 10 (raw / V34+ normalized) | 165 | 64 | 7811 | 83 | 0.69 | 0.81 | 0.68 |
| `episodic_return_std / mean` | 0.37 ✅ | 0.36 ✅ | 0.47 ✅ | 0.36 ✅ | 0.51 ❌ | 0.44 ✅ | 0.48 ✅ |
| **§5 PASS count** | 3/5 (transfer ❌) | 1/5 | 2/5 | 2/5 | 1/5 | **2/5** | 2/5 |

Wandb run IDs:
- V33: `l01moel9` (500), `um9a8te5` (555), `ykqfc9za` (999)
- V34: `voaf8iza` (500), `oqc8mxs7` (555), `r9sqjj67` (999)
- V35: `7utp05yg` (500), `4g64ujvs` (555), `gsul9iw1` (999)
- V36: `vgiuq0zt` (500), `vgsu0xkx` (555), `d0kckokl` (999)
- V32 entropy diagnostic ran against existing v22 / v28a / v31 wandb data (no new training).

## 5. Candidate next directions (if open)

1. **V37 = `--bc-anchor 0`** — gated on §2 (1.) literature pass. Tests whether reverse-curriculum-particle-replay alone (the v25+ mechanism) holds transfer ≥ 0.15 without any KL anchor.
2. **V38 = anchor decay schedule** (start 1.0 at iter 0, linearly to 0 over iter 0-300, stay 0 thereafter) — only if V37 transfer collapses. Tests "anchor as warm-start regularizer, not steady-state regularizer".
3. **Tighten `clip_grad_norm 1.0 → 0.5`** — moderates the direction-dominance of any single loss term. Could be combined with V37 if V37 KL is still > 1.
4. **Increase Stage 1 `num_epochs` floor** (currently 2). Free-PPO regime (anchor off) might tolerate 5 epochs, recovering effective learning rate.
5. **DAPG-style demo mixing** — add demo `(obs, action, reward)` tuples directly into PPO rollout buffer with controllable mixing ratio. Larger redesign; defer until V37 / V38 settle.
6. **Threshold edit on §5** — only if (2.3) shows the literature consensus differs from our PASS thresholds. Logged as a separate decision, not bundled with a tuning version.

## 6. Currently running / waiting

- **No background runs.** All V32-V36 long runs completed in this session. EC2 idle, A10G ~0 % utilization.
- **No `ScheduleWakeup` set.** The auto-iteration loop was stopped by the user after V36; no pending wakeups.
- V37 not launched in this session — gated on §2 literature pass per the new §4.6.

## 7. References

- `docs/stage1_5_tuning_log.md` — V25 → V36 full entries. The "Stage 1.5 Status" section (around line 377) is the consolidated current state. **Read this first** before proposing the next iteration.
- `training/train.py` — V33 freeze revert at `bc_pretrain` (lines 693-700 replaced with comment); V34 return-rms (`__init__` line 355, update line 419, value-loss scaling lines 514-528).
- `scripts/analyze_v18.py` — has v33 / v34 / v35 / v36 dicts; `v37 = {...}` not yet added.
- **`CLAUDE.md` §4.6 (newly added)** — must read before V37 work. §6 has new V32-V36 anti-patterns. §8.B is the multi-version-stuck escalation order.
- `docs/handoff_template.md` — canonical handoff format (added externally by user during this session).
- `~/.claude-account2/projects/.../memory/feedback_readme_decision_judgment.md` — 3-question test for promoting result to README.
- Prior handoffs that still apply: `docs/session_handoffs/2026-05-03_v31_repro_check_and_pass_discipline.md` (CLAUDE.md §5.1/§5.2 PASS-binary discipline still in force), `docs/session_handoffs/2026-05-03_stage1_5_v28_analysis.md` (older).

## 8. Unresolved threads

- **README + appendix lag the tuning log** by V33-V36. Updating now would clutter README with FAIL versions; defer to next §5 PASS event per the 3-question test in memory.
- **The "smoke test predicts the long run" assumption is a confirmed false friend** on this codebase. V34 smoke ended iter 50 with KL=0.07; long run reached KL=60 by iter 500 (857× higher). Now codified as §6 anti-pattern. Future iterations: smoke tests confirm "process launches without crashing", nothing more.
- **§10 was rewritten externally during this session.** Auto-handoff triggers removed — handoffs are now user-triggered only. The auto-generated handoff this session originally produced (`2026-05-03_v32_v36_freeze_revert_anchor_sweep.md`) was deleted as inconsistent with new §10. If a future session finds an auto-generated handoff floating around, it likely violated the new §10 — verify trigger before keeping.
- **Anchor mechanism's mechanistic role is still ambiguous.** v22 (no anchor) had transfer=0.024 but PPO health PASS-near. V35 (anchor 1.0) had transfer=0.266 but PPO health FAIL. Anchor seems to *cause* transfer + *break* PPO health simultaneously. The §2 literature pass should specifically check whether published BC-anchored PPO papers show this same trade-off, or whether their anchor mechanism is structurally different from ours (e.g. only applied on BC-distribution states, or decayed schedule, or analytic-vs-empirical KL).
