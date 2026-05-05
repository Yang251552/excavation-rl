# Claude Code Operating Notes — excavation-rl

This file is auto-loaded by Claude Code when working in this repo. It exists so future sessions can resume the bring-up loop without re-discovering setup, repeated bugs, or anti-patterns.

If something in this file is wrong or stale, fix it. It is the single source of truth for "how do we work on this project."

---

## 1. Environment & Access

- **Local working dir**: `/Users/yangchenghan/Downloads/semester project2 挖掘/excavation-rl/` (Mac, zsh, no GPU)
- **Training EC2**: `ubuntu@16.171.144.168` (g5.xlarge, NVIDIA A10G 22 GB, CUDA 12.2)
  - Project mirror: `/home/ubuntu/excavation-rl/`
  - venv: `source /home/ubuntu/venv/bin/activate`
  - `~/.netrc` already has wandb credentials — no `wandb login` needed
- **SSH key**: `~/.ssh/excavation-key.pem`
- **IP changes** when EC2 stops/starts. If `ssh ubuntu@...` fails, ask user for the current IP and update this file.
- **Permissions** for ssh/scp/wandb-cli are pre-allowed in `.claude/settings.local.json`. If revoked, re-add:
  ```bash
  Bash(ssh -i ~/.ssh/excavation-key.pem ubuntu@<ip> *)
  Bash(scp -i ~/.ssh/excavation-key.pem * ubuntu@<ip>:*)
  ```

---

## 2. Reading Training Data — DO and DON'T

**DO**: Pull the full per-component reward history via `wandb.Api` from a script that runs on EC2 (where credentials are already loaded).

```bash
# Latest template: scripts/analyze_v18.py. For a new campaign, copy it to
# scripts/analyze_vXX.py, swap the run-ID dict at the top, and run on EC2:
ssh ubuntu@<ip> "cd ~/excavation-rl && source ~/venv/bin/activate && python scripts/analyze_vXX.py"
```

The template uses `run.scan_history(keys=...)` (not `.history()`) because EC2 has no pandas. Reads all `reward/*` keys (the 10 reward components — 9 weighted + success bonus — plus the `reward/_total` aggregator), `Performance/*`, and `Policy/*`.

**DON'T**: `tail logs/...log | grep ...` for the wandb summary block. The summary truncates with `+15 ...` after the first 9 keys, and **the hidden 15 keys are exactly the per-component reward decomposition**. This is how stage-1 v17 was misdiagnosed as "no learning" when in fact `success_bonus` was firing 21–65% of the time and seed-123 hit `transfer_ratio = 9.2%` at one peak — invisible in the truncated summary.

**Spark sequences in the summary are useful** but their resolution is 8 chars across the whole run; check actual numeric values via API for anything load-bearing.

---

## 3. Known Bugs Already Found and Fixed (Stage 1 Bring-Up)

If a metric looks wrong, check the bug catalogue before assuming the bug is new.

**Single source of truth**: `docs/stage1_results_appendix.md` (currently **10 entries**, full file:line + Effect + Resolution columns). Update **that** file when adding bugs 11+; do NOT maintain a parallel list here.

Quick-reference cheat sheet (file:line → root cause), so you don't have to open the appendix mid-debug:

| # | File:Line | Root cause |
|---|---|---|
| 1 | `envs/excavation_env.py:93` | `scene.build()` never called — particle_system stays None, all soil metrics report 0 |
| 2 | `soil/soil_properties.py:54`, `soil/particle_system.py:248` | `bucket_push_force` defaulted to `contact_damping = 1000 N` — particles eject at 90 m / 20 steps |
| 3 | `training/train.py` (`RunningMeanStd`) | No obs normalisation — KL locks at 7–13, clip_fraction > 0.95 across all reward / LR variants |
| 4 | `envs/curriculum.py`, `envs/rewards.py:118-131` | Curriculum override silently masks explicit reward weights when enabled (default) |
| 5 | `training/ppo_cfg.py:29` | `init_noise_std = 1.0` default — entropy starts at 9.98, effectively random throughout |
| 6 | `training/ppo_cfg.py:53` | `schedule="adaptive"` default collapses LR to floor on noisy reward |
| 7 | `envs/excavation_env_cfg.py:134` | Binary collision penalty (-15) — `episodic_return = -335 ± 379` bimodal breaks advantage normalisation |
| 8 | `envs/excavation_env.py:246` | Approach reward used **dynamic** soil centroid that moves away as bucket scoops |
| 9 | `training/train.py` (`RunningMeanStd.freeze()`) | `obs_rms.update()` every rollout → stats drift → KL drifts 2.79 → 6.54 over 500 iter |
| 10 | `envs/scene.py:_update_carried_particles`, `training/train.py:701` | Bucket AABB has no walls (particles fall out every step) + `action_smoothness = -0.01` dominates positive components — agent converges to "minimum-action / hover" policy |

For the full Effect / Resolution columns, see [`docs/stage1_results_appendix.md`](docs/stage1_results_appendix.md). When you fix bug N+1, **add it to that appendix and append one row to this cheat sheet** — never the other way around.

---

## 4. Standard Iteration Loop

0. **Before writing the next version's Hypothesis**, check four stop conditions:
   (a) last 2+ versions changed the same knob within an order of magnitude AND the targeted §5 criterion did not move monotonically in the predicted direction (§4.6.1).
   (b) last 3 versions all FAIL with §5 PASS count flat or oscillating ≤±1 (§4.6.3).
   (c) the next hypothesis would have no literature anchor or codebase-vs-standard gap (§4.6.2 trigger).
   (d) **5 consecutive FAILs in the current stage** — hard cap (§4.6.4).
   If (a) / (b) / (c) fires: run §4.6.2 (literature + gap + threshold) before iterating; resume rules in §4.6.5.
   If (d) fires: §4.6.4 applies *regardless* of (a) / (b) / (c) — STOP autonomous iteration, surface to user, the user picks the next branch.
1. **Edit locally** (Mac) — edit → review.
2. **Push to EC2**: `scp -i ~/.ssh/excavation-key.pem <files> ubuntu@<ip>:/home/ubuntu/excavation-rl/<dest>/`
3. **Verify scp landed** before training: `ssh ... grep <new-symbol> <file>` — files don't always land if path is wrong; failing silently wastes a 10 min run.
4. **Clear pycache** before each run: `find . -name __pycache__ -type d -exec rm -rf {} +`. Otherwise Python caches old `.pyc` files and the run uses pre-fix code.
5. **Smoke test (50 iter, foreground)** before any long run. Use `| tee logs/...log | tail -25` to capture+display. **Smoke confirms launch, not metric trajectory** — KL especially grows over training as policy drifts from BC; do not extrapolate iter-50 numbers to long-run §5 PASS (§6).
6. **Long runs (≥ 200 iter)**: `nohup ... &` + capture PID + use `ScheduleWakeup` (10–15 min for 500-iter runs at ~700 FPS).
7. **Multi-seed**: 3 seeds (`{7, 42, 123}`) in parallel — A10G runs at ~80 % utilisation with 3 concurrent stage-1 runs (each ~1 GB GPU memory, ~700 FPS combined).
8. **After run**: pull data via wandb API (see §2), do not grep the log.
9. **Log the iteration to `docs/<stage>_tuning_log.md`** — see §4.5 below. **Mandatory for every parameter change, before moving on.**

---

## 4.5 Tuning Log Discipline (Mandatory)

Every parameter change — even a smoke test that fails or is reverted — gets a **single new entry** appended to the tuning log file for the current stage:

| Stage | Log file |
|---|---|
| Stage 0 | `docs/stage0_tuning_log.md` (create if missing, mirror the structure of stage 1's) |
| Stage 1 | `docs/stage1_tuning_log.md` (already exists, follow its format) |
| Stage 1.5 | `docs/stage1_5_tuning_log.md` (create when starting BC + KL anchor work) |
| Stage 2 | `docs/stage2_tuning_log.md` (create when curriculum + DR work begins) |
| Cross-stage tooling / infra | `docs/infra_changelog.md` (create lazily) |

**Per-entry template** (drop in at the bottom of the relevant file under a `### Vxx (one-line title)` heading):

```markdown
### V<n> (one-line title — what changed)

**Hypothesis**: which earlier wandb signal triggered this change. Cite the metric name + value.
Example: "v17 reward/load mean = 0.14 across all seeds → bucket containment likely missing".

**Change** (one bullet per file edited):
- `path/to/file.py:LINE` — `old_value → new_value`. Why this magnitude / sign / name.

**Test**: smoke (50 iter, single seed) or long (500 iter × 3 seeds). Wandb run IDs.

**Result table** (pull via `scripts/analyze_v18.py` template, never via grep):

| Metric | v<n-1> | v<n> | delta | comment |
| --- | --- | --- | --- | --- |
| `Performance/soil_transfer_ratio` (final 5 windows) | ... | ... | ... | ... |
| `Policy/approx_kl` (final 10 iter mean) | ... | ... | ... | ... |
| `reward/load` (final mean) | ... | ... | ... | ... |
| ... at minimum cover transfer_ratio + KL + the reward components affected ... |

**Decision**: keep / revert / partial-keep. If reverted, explicitly say which earlier version's value the parameter is restored to.

**Next**: which signal in this version's data motivates the next iteration. **Do not start the next iteration before writing this**.
```

**Why mandatory**: stage 1 went 22 versions deep (v1 → v22). Without the log, future sessions (or this one tomorrow) cannot tell which knobs have been tried, which were validated, and which were just "smoke tested then reverted because of an unrelated bug". The log is the project's debugging memory; skipping entries strands future work in the same dead-ends we've already explored.

**Anti-patterns**:
- Skipping a log entry because "this one is just a typo fix" → if the fix touches a hyperparameter or code path the agent runs through, log it. The bar is "could a future session benefit from knowing this was tried?", not "is this a real experiment?".
- Logging only successful versions → failed attempts are higher signal because they rule out hypotheses. v18 (freeze obs_rms — exploded seed 123 KL to 22 820) is the most valuable entry in the stage-1 log.
- Editing earlier entries to "make them consistent" → entries are immutable. If a later run revealed an earlier conclusion was wrong, write a new entry pointing back, don't rewrite history.
- Free-form prose without the table → numbers are the actual signal; cross-version comparison only works if every entry has the same structured table.

---

## 4.6 Step-Back Discipline (Avoiding Parameter-Tuning Death Spirals)

Tuning logs are mandatory (§4.5) but not sufficient. Logging every iteration prevents *forgetting*; it doesn't prevent *iterating in a known-not-working direction*. This section codifies when to stop tuning the current knob and step back to a higher level: literature, mechanism redesign, or scope re-examination.

**Background**: V32-V36 spent 5 iterations in a death spiral — V32 falsified hypothesis without literature reference, V33 fixed a real bug (the only iteration that moved a §5 criterion meaningfully), V34-V36 then sank into "anchor coef sweep" that produced KL 60 → 39 → 51 across coefs 2.0 → 1.0 → 0.5 (non-monotonic; clip_grad_norm renormalization made coef magnitude not the lever it appeared to be). The §5 PASS count oscillated 2 → 1 → 2 → 2 with no trend. None of these versions cited a single paper or external benchmark. The whole sweep should have been called off after V35 — instead, "diminishing returns toward the right direction" rationalisation kept it going. This section exists so future sessions don't repeat that pattern.

### 4.6.1 Sweep-stop rule

A *sweep* is two or more consecutive versions targeting the same single criterion via the same knob (e.g. anchor coef across V34/V35/V36 all targeting KL).

**Stop the sweep** when ALL of the following hold:

1. **2+ versions in a row** changed the targeted knob within an order of magnitude.
2. The targeted criterion's 3-seed mean did NOT move in the predicted direction monotonically (a single-version dip followed by a regression counts as non-monotonic — the trend is noise).
3. No other §5 criterion improved as a side effect.

When stopping, write a one-paragraph "sweep verdict" into the tuning log of the last sweep version's Decision section, stating: (a) which knob was tested, (b) the range tested, (c) the metric trajectory, (d) the mechanistic reason the knob failed (e.g. "clip_grad_norm renormalizes total gradient regardless of this coef"). Then jump to §4.6.2 *before* writing the next version's Hypothesis.

### 4.6.2 Big-direction sanity check (before the next version)

When the sweep-stop rule fires, OR when 3+ consecutive versions failed to advance the §5 PASS count, OR when a hypothesis is being formed without external grounding, run this check before launching anything new:

1. **Literature pass** (use WebSearch + WebFetch via Agent tool). For the current task family (BC + PPO fine-tuning on robot manipulation in our case): what are the reference papers? what KL / EV / clip_fraction ranges do they report on similar benchmarks? what mechanism do they use (DAPG demo mixing? AWR? KL anchor? entropy bonus schedule?)? Cite at least one paper in the next Hypothesis section. If the codebase's mechanism is not in the literature, that's information — either it's a known dead end or it's a novel choice that needs explicit defence.
2. **Codebase-vs-standard gap analysis** (use the Explore subagent). Make a 2-column list: "ours" vs "what the dominant published implementation does". For each row, label `match` / `differ-on-purpose` / `differ-without-justification`. The third category is the bug surface for redesign.
3. **Threshold sanity check** (re-read §5). Are our PASS thresholds calibrated for *this* task family? If the literature reports KL=5-15 as routine for BC-anchored PPO and we set < 1.0 because RSL_RL defaulted there, the threshold may be wrong, not the run. If the threshold is wrong, propose an explicit threshold edit *before* tuning further — log it as a separate decision in the tuning log.

These three checks are read-only (no training launches, no code writes during the check itself). They produce a written assessment that becomes the Hypothesis section of the next version. **The next version's hypothesis must cite at least one external reference (paper title or benchmark URL) or one specific gap from the gap analysis.** "I noticed metric X is high → tweak Y" is not a sufficient hypothesis after the sweep-stop rule has fired.

### 4.6.3 Death-spiral detector (passive guard)

At the start of every new tuning iteration (before writing the §4.5 entry), check the last 3 entries' Decision sections in the tuning log. If 3 consecutive Decisions read "FAIL" with the §5 PASS count flat or oscillating ≤ ±1, you are in a death spiral. Stop and run §4.6.2 before writing the new entry. Future sessions reading a handoff (per §10) should also apply this check on their first action — if the handoff queues another local-tweak version after a 3-FAIL streak, push back and ask the user whether to step back instead.

### 4.6.4 Hard cap on autonomous iteration (mandatory user surface)

Regardless of whether §4.6.1 / §4.6.2 / §4.6.3 have already fired in the current campaign, **after 5 consecutive FAILs in the current stage**, stop autonomous iteration entirely:

- Do **NOT** draft v(n+1)'s Hypothesis.
- Do **NOT** launch any new training run.
- Surface to the user with: (a) the v(n-4) → v(n) chain summary (one line each: knob touched + §5 PASS count); (b) the literature-pass output from the most recent §4.6.2 invocation (or run a fresh one if none has been done); (c) two or three concrete branch options — *adopt-standard-mechanism* / *re-scope-task* / *continue-with-explicit-user-approval*.

The user picks the branch. **This rule overrides any prior "auto-iterate until X" instruction** — the agent's job under such an instruction includes recognising when X has stopped being plausible and surfacing the question.

Why 5: it lets §4.6.1 (2-version sweep-stop) and §4.6.3 (3-version death-spiral) trigger and resolve at least once before forcing user involvement. If those mechanisms work, the hard cap rarely fires. If the LLM ignores them, the hard cap is the backstop. V32-V36 was 5 deep before being externally stopped; under this rule it surfaces automatically after V36 (and ideally even after V35 via §4.6.1).

User override is allowed: if the user explicitly authorises continuation, the next version's tuning-log entry **must cite the user's authorisation quote** in its Hypothesis section, and the FAIL counter does NOT reset until a PASS occurs.

### 4.6.5 Resuming autonomous iteration after §4.6.2

The lit-search output is **not** automatically a green light to launch v(n+1). Different findings call for different next moves. Follow these 5 steps in order.

**Step 1 — Classify the lit-search output into ONE of 4 categories** (this is the most-skipped step; do it explicitly):

| Category | What it means | Recognition cue |
|---|---|---|
| **Threshold issue** | Literature routinely reports the failing criterion *outside* our §5 PASS threshold | Lit: "KL=5–15 routine for BC-anchored PPO"; ours: §5 says < 1.0. Our gate is wrong, not the run |
| **Mechanism issue** | Literature solves this task family with a *fundamentally different* mechanism than ours | Lit: DAPG demo mixing inside PPO loss; ours: KL anchor against frozen BC. Different family, not a parameter tweak |
| **Tactical gap** | Same mechanism family as literature, but a specific implementation detail differs | Lit: KL anchor *paired with* entropy bonus floor; ours: KL anchor alone. Same mechanism, missing one piece |
| **No clear answer** | Literature is divided, contradictory, or has no precedent for this specific scenario | 3 papers, 3 different recommendations, no consensus emerges |

**Step 2 — Branch by category:**

- **Threshold issue → DO NOT silently continue.** Threshold edits are governance, not tuning.
  - Write a separate tuning-log entry titled `Threshold proposal` citing the paper(s).
  - **Surface to the user** with the proposed §5 edit (e.g. "relax KL gate from < 1.0 to < 5.0?"). Wait for approval.
  - After approval: edit §5 in CLAUDE.md, then resume from v(n+1) per Step 3.
- **Mechanism issue → DO NOT unilaterally swap mechanism.** This is a scope decision.
  - Surface to user with three things: (a) what we're using, (b) what literature uses, (c) tradeoffs (sample efficiency / stability / implementation cost).
  - User picks between: *adopt-standard-mechanism* / *defend-novel-with-rigorous-experiment* / *re-scope-task*. Wait for choice.
  - After user's branch decision, that's the new direction (may bypass version numbering for a redesign).
- **Tactical gap → Apply directly in v(n+1).** No user surface needed for this category.
  - Write the v(n+1) Hypothesis citing the paper title/URL or the specific gap row. Then proceed to Step 3.
- **No clear answer → Surface to user.** Continuing autonomously with a low-confidence guess is high-risk.
  - State the situation: "Literature reviewed (cite 2-3 papers), no consensus emerged on X. Options: (a) try Y informed-guess as a cheap diagnostic, (b) wait for user to provide additional reference, (c) accept current FAIL and re-scope".
  - Wait for user direction. Hard cap (§4.6.4) still counts forward.

**Step 3 — Only if Step 2 routed to "apply directly" or the user approved Resume:**

- v(n+1)'s Hypothesis section MUST cite at least one paper title/URL or one specific gap row from the lit-search output. "I noticed metric X is high → tweak Y" is insufficient.
- Run v(n+1) per §4 standard iteration loop.

**Step 4 — Counter reset on §4.6.2 completion:**

| Counter | Reset? | Why |
|---|---|---|
| §4.6.1 sweep-stop (same-knob non-monotonic) | **YES** → 0 | The lit-grounded hypothesis is not "same knob" |
| §4.6.3 death-spiral (3 consecutive FAIL) | **YES** → 0 | New external grounding earns a fresh 3-attempt window |
| §4.6.4 hard cap (5 consecutive FAIL in stage) | **NO** | Total iteration budget per stage is bounded regardless |
| §5.1 / §5.2 framing rules | **NEVER** | Independent of iteration count |

**Step 5 — Post v(n+1):**

- Each subsequent iteration re-runs §4 step 0. Counters tick forward; §4.6.x may re-trigger and the entire 5-step flow re-applies.
- **One §4.6.2 per FAIL chain.** A *second* lit search in the same campaign must pull *different* references; the v(n+2) Hypothesis must explicitly note "this is the second §4.6.2 of the current campaign — references differ from the first because <reason>".
- **Hard cap is non-negotiable.** Even if every post-lit-search version cites fresh papers, after 5 cumulative consecutive FAILs in the stage, §4.6.4 fires and user surface is mandatory. Lit search is a tool; it does not extend the cap.

**Concrete walkthrough — V32-V36 under this flow:**
- V32 FAIL (count=1) → V33 same knob FAIL (count=2) → §4.6.1 fires → §4.6.2 lit search
- Step 1 classifies output as "Tactical gap" → Step 2 says "apply directly" → V34 lit-grounded, run.
- V34 FAIL (count=3) → §4.6.3 fires → §4.6.2 second lit search (must use *different* references than first)
- IF Step 1 now classifies as "Mechanism issue" or "Threshold issue" → **surface to user; V35 not autonomous**.
- IF "Tactical gap" again → V35 lit-grounded → V35 FAIL (4) → V36 FAIL (5) → **§4.6.4 fires → STOP, user surface**.
- v(n) → v(n+1) automation **never reaches V37 without explicit user approval**.

---

## 5. PASS / FAIL Criteria (Stage 1)

A version "PASSes" the bring-up only when ALL of these hold across 3 seeds:

| Criterion | Threshold | Why |
|---|---|---|
| `Performance/soil_transfer_ratio` (mean of 3 seeds) | **> 0.15** (15 %, the random baseline) | If we can't beat random there is no learned skill |
| `Policy/approx_kl` (mean of last 50 iter) | **< 1.0** | KL > 1 means PPO trust region is broken; updates won't accumulate productively |
| `Policy/clip_fraction` (mean of last 50 iter) | **< 0.5** | > 0.5 means most updates are wasted |
| `Performance/episodic_return_std / mean` | **< 0.5** | High CoV means "success" is noise, not skill |
| `Policy/explained_variance` (final) | **> 0.5** | Value function should fit the dense reward |

Random baseline (uniform `[-0.3, 0.3]` joint deltas, 500 steps): `transfer_ratio ≈ 0.15` (single rollout, seed 42) — see `tests/random_rollout.md`. Trained policy should clearly exceed this.

**Anti-pattern**: declaring "PASS" because `episodic_return` is positive. Stage 1 v6's `episodic_return = +236` came entirely from the dense `approach + dig` shaping the agent learned to maximise without ever scooping. Always cross-check against `transfer_ratio` and `bucket_load`.

### 5.1 No "Partial PASS"

PASS is binary. The 5 criteria above are AND, not majority vote. If any one fails, the version is **FAIL**, period. There is no "partial PASS", "near PASS", "PASS with caveats", or "ready for next stage with X to fix later". When tempted to use such language, write **FAIL** and continue iterating on the current stage.

This rule exists because v28a was misdeclared "Stage 1.5 partial PASS" with mean `transfer_ratio = 0.107` (below the 0.15 threshold) and KL = 14 (above the 1.0 threshold) — only 1 of 5 criteria satisfied. The "partial PASS" framing made it feel like the natural next step was Stage 2, when in fact it was "FAIL, continue Stage 1.5". The label hid that 4/5 criteria were unmet.

### 5.2 No "Best Seed" Reframing

A single seed exceeding a criterion is not partial credit. The 3-seed mean is the criterion. If seed-to-seed variance produces one outlier that clears a threshold, that's evidence the mechanism *can* work, not that the system *does* work. Continue iterating until the 3-seed mean clears.

This rule exists because v28a's seed 123 reached `transfer_ratio = 0.285` while seeds 7 and 42 stayed at 0.034 and 0.0002. The headline "first seed above random baseline" anchored decision-making and hid that the cross-seed mean was 0.107. Future sessions: if you find yourself citing one seed's number to justify the version, you are violating §5.2 — go back and look at the 3-seed mean.

---

## 6. Anti-Patterns (Mistakes Made That Wasted Real Time)

#### Data-trust failures
- **Trusting truncated wandb summary** instead of API → misdiagnosed v17 as "no learning" when seeds 42/123 had transient transfer_ratio = 3 % and 9 %.
- **Tuning reward weights with `print(reward[-1])`** instead of per-component decomposition → spent v9–v12 chasing "PPO instability" when the real bug was missing obs normalisation.

#### Process / infra failures
- **Trusting that "code in the repo" = "code on EC2"**: scp can fail silently (wrong path, permission); always grep on EC2 to verify a fix landed before launching a 10-min run.
- **Killing a run, then realising you needed its checkpoint**. If unsure, `tail logs/...log` first, kill second.
- **Running long jobs with no `ScheduleWakeup` set**: we lose track and the agent sits idle while the run completes.

#### Design-iteration failures
- **Sweep persistence past 2+ versions on the same knob** with no monotonic movement — see §4.6.1.
- **Hypothesis without literature anchor or codebase-vs-standard gap** — see §4.6.2. If a hypothesis has no citation and no gap to point at, it's a guess; budget it as a cheap diagnostic, not a long run.
- **Treating smoke test (50 iter) metrics as a long-run predictor**. V34 smoke ended at iter 50 with KL = 0.07; long run reached KL = 60 by iter 500 (857× higher). Smoke confirms launch, not metric trajectory — KL especially grows over training as the policy drifts from BC.
- **Auto-iterating across multiple FAILs without death-spiral check** — see §4.6.3.
- **Continuing past 5 consecutive FAILs without surfacing to user** — see §4.6.4. The hard cap exists because V32-V36 ran 5 iterations deep and would have continued if not externally stopped. Lit search does not exempt you from the cap (§4.6.5).

#### Scope-framing failures
- **Inventing "partial PASS" / "near PASS" framing** when §5 criteria fail — see §5.1.
- **Best-seed reframing** (citing one seed's number when 3-seed mean fails) — see §5.2.
- **Concluding "reward design is exhausted" after 3 seed × 500 iter** without checking the per-component decomposition for transient successes (which v17 had).

---

## 7. Existing Long-Running Decisions (Don't Re-Litigate)

- Stage 0 uses **simplified single-component (approach only) reward** intentionally; that is not a bug.
- Stage 0 uses **rigid body proxy**, not particles; particles only exist in stage 1+.
- Curriculum is **disabled** in stage 0 and stage 1 bring-up because of bug #4. Re-enabling for stage 2 must be paired with verifying which weights it overrides.
- **Two distinct thresholds — don't conflate them:**
  - **Bonus reward threshold** = `0.3` (stage 1 / 2): `reward/success_bonus = +5` fires the first step `transfer_ratio ≥ 0.3`. Lowered from 0.8 in early stage-1 versions so the bonus actually fires during training (otherwise PPO never sees the bonus signal).
  - **Episode termination threshold** = `0.8` (stage 1 / 2): `soil_in_target_ratio ≥ 0.8` terminates the episode as SUCCESS. This is the real task definition; the bonus threshold is just a shaping aid.
  - Stage 0 sets the **termination** threshold to `1.01` (unreachable, i.e. SUCCESS termination disabled) so `episodic_length` stays constant at 500 — see [README §Termination Conditions](README.md#termination-conditions).
  - Don't change either without flagging — they are tuned together.

---

## 7.5 Stage 1.5 Artefact Pointers

Where Stage 1.5 mechanisms live. Don't re-derive — find / extend.

- **BC demo data**: `results/bc_demos.npz` (12 800 obs/action pairs after v25 expansion, `ACTION_NOISE_STD = 0.05`). Generator: `scripts/generate_bc_demos.py` wraps `scripts/scoop_demo.py`'s 5-phase parametric scoop. Replay-only baseline reaches `transfer_ratio = 22.4 %` per episode.
- **KL anchor (BC drift bound)**: `--bc-anchor <coef>` CLI flag. Hook: `training/train.py:_ppo_update`. After `bc_pretrain`, a `deepcopy(actor_critic)` is frozen as `_bc_actor_critic`; PPO loss adds `coef · KL(curr ‖ frozen_BC)` (analytic Gaussian KL).
- **Containment heuristic (sticky-grab)**: `envs/scene.py:_update_carried_particles`. Particles inside the bucket's interior detection box are pinned to the bucket frame; auto-released when bucket xy enters the target zone. Scaffolding for Isaac Lab mesh-particle collision (Phase 1 in README TODO).
- **Reverse-curriculum reset**: `--reverse-curriculum [--reverse-curriculum-prob p]`. Resets both joint state AND particle state from a sampled point along the demo trajectory (v25+).
- **Particle snapshot replay**: `--snapshot-data results/scoop_snapshot.npz`. Recorded by `generate_bc_demos.py` per step (joint pos + particle pos/vel + carried indices); used by `set_demo_trajectory_for_env(snapshot=)` at reset.

Per-iteration history of these mechanisms: `docs/stage1_5_tuning_log.md`.

---

## 8. When You're Stuck — Escalation Order

"Stuck" has two flavours; the escalation differs.

### 8.A — Stuck on a single version (one run looks broken or contradictory)

1. Pull the full reward decomposition + KL trajectory via wandb API. Often the answer is in data already collected.
2. Run a **standalone test** of the suspect component (see `tests/standalone_bucket_sweep.md` for the format). 100 lines of Python beats 30 min of guess-tuning.
3. Compare against **random baseline** (`tests/random_rollout.md`) — if trained < random, the issue is policy collapse, not reward magnitude.
4. **Re-read the most recently changed code path** (e.g. `bc_pretrain`, `_ppo_update`) end-to-end. The V32 finding — that `obs_rms.freeze()` was unconditionally called inside `bc_pretrain` because a prior session's "REVERT" decision was never executed in code — would have surfaced 4 versions earlier if any iteration had read `train.py:bc_pretrain` end-to-end before launching. Trust documented decisions less than the actual code.
5. Only then propose a code change. Tag with version number (`v18`, `v19`...) so we can grep the tuning log for "v18" later.

### 8.B — Stuck across multiple versions (3+ FAILs in a row, §5 PASS count flat or oscillating)

Single-version diagnostics are not the right escalation — the problem is at the design / mechanism / threshold level. Apply this order:

1. **Death-spiral check** (§4.6.3). Confirm the pattern: list last 3 versions' Decision + §5 PASS counts. If FAIL / FAIL / FAIL with PASS count flat ≤±1, you are in a death spiral; the next step is *not* another tuning iteration.
2. **Apply §4.6.2** (literature pass + codebase-vs-standard gap analysis + threshold sanity check). The output of these three reads becomes either the next version's Hypothesis section or a redesign proposal that bypasses version numbering.
3. **Surface to the user** if the gap analysis or literature pass reveals our Stage 1.5 design is fundamentally different from the literature consensus. Don't unilaterally redesign — that's a scope decision; the user picks between (a) adopting the standard mechanism, (b) defending our novel choice with a more rigorous experiment, or (c) re-scoping the task.

Steps 1-2 are read-only. **Do NOT launch a new training run while in 8.B until §4.6.2 is completed.**

---

## 9. README / CLAUDE.md Consistency Audit (Mandatory After Structural Edits)

After **structural** edits to `README.md`, `CLAUDE.md`, or `docs/stage*_results_appendix.md` — defined as any of:

- adding / removing a section or table
- changing numbers (metric values, dimensions, counts, thresholds, run IDs)
- changing file paths, anchors, or wandb URLs
- moving content between files (README ↔ docs/, CLAUDE.md ↔ appendix)

run this 3-step audit before declaring the task done. **Skip for**: typo fixes, formatting tweaks, link-text rewording, comment changes — anything that does not move facts.

**Step 1 — Cross-file fact check.** For each numeric claim or named entity touched (obs dim, reward component count, bug count, threshold values, run IDs, file paths), grep the other docs to confirm they agree. Example sweep:

```bash
grep -rn "<changed-value>\|<old-value>" README.md CLAUDE.md docs/
```

**Step 2 — Anchor check.** For every markdown link `[...](#anchor)` touched or pasted, confirm the target heading exists in the same file. GitHub slug rule: lowercase, spaces → hyphens, punctuation stripped (e.g. `## Current Progress & TODO` → `#current-progress--todo`).

**Step 3 — Single-source-of-truth check.** If the edit duplicates a list or table that already lives elsewhere (bug catalogue, run URLs, reward-weight history, reproducibility commands), replace the duplicate with a one-line pointer to the authoritative file. The current ownership map:

| Content | Authoritative file |
|---|---|
| Stage 1 bug catalogue (full Effect / Resolution columns) | `docs/stage1_results_appendix.md` |
| Stage 1 + Stage 1.5 quantitative results (multi-seed tables, wandb URLs, per-seed transfer/KL/clip) | `docs/stage1_results_appendix.md` |
| Per-iteration tuning history — Stage 1 (v17 → v22) | `docs/stage1_tuning_log.md` |
| Per-iteration tuning history — Stage 1.5 (v25+) | `docs/stage1_5_tuning_log.md` |
| Reproducibility commands | `docs/stage1_results_appendix.md` / `docs/stage0_results_appendix.md` |
| Stage 1 / 1.5 PASS/FAIL gate criteria | `CLAUDE.md` §5 |
| Bug cheat sheet (file:line → root cause) | `CLAUDE.md` §3 (10 rows, mirrors appendix) |

**Scoping convention** (do not split unilaterally — read this before proposing a new appendix file):
- **Results appendices group by env/scene boundary, not stage label.** All campaigns running on the same Stage 1 env (Warp particles, 10-component reward, current scene layout) — including Stage 1.5 mechanism extensions like BC anchor / reverse curriculum — go into `docs/stage1_results_appendix.md`. Reward formula, bug catalogue, scene description are shared and would otherwise drift if duplicated.
- **Tuning logs split by stage.** Each stage has its own iteration cadence and own set of hypotheses; mixing them blurs the chronology that future debugging relies on.
- A new `docs/stageN_results_appendix.md` is justified **only** when the env, reward formula, or bug surface changes substantially (e.g. Stage 2 will add curriculum + DR + graduated penetration penalty — a new env state space, new bug surface, warrants `stage2_results_appendix.md`).

**Output format**: max **5 bullets**, each one is `<severity> | <file:line> | <issue> | <suggested fix>`. Surface findings only — **do not modify anything in the audit pass itself**. Wait for the user to confirm which items to fix.

**When to skip the audit entirely**: if the only change in this turn was content the user explicitly drafted and pasted (no synthesis on your part), the audit is optional — they presumably already know what they wrote.

---

## 10. Session Handoff (Pointer)

Files live in `docs/session_handoffs/`, named `<YYYY-MM-DD>_<topic-slug>.md` (filename sort = chronological).

**To read the latest** (e.g. user says "读最新 handoff" / "pick up where we left off" / similar):
```bash
ls -t docs/session_handoffs/*.md | head -1
```
Read the file that returns. Acknowledge in one sentence including the date and §2 queued action — *"Picking up from `<date>` handoff: queued action is `<X>`."* If the user's first message clearly redirects to a different topic, note the queued items exist but address the user's actual ask. Always honour §3 (Don't re-litigate).

**To write**: follow [`docs/handoff_template.md`](docs/handoff_template.md). The `<topic-slug>` is auto-generated by you from the session's primary subject — do not ask the user. Only write when the user explicitly triggers (e.g. "write a handoff").
