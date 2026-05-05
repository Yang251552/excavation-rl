# Retrospective — excavation-rl (Part 2 of research portfolio)

> **Status**: skeleton — section guidance below, content to be written by the author in their own voice. Each `<!-- WRITE: ... -->` block describes what the reader should learn from that section.

This document is the load-bearing complement to README.md. The README states the headline finding in two paragraphs; this file is where I (the author) lay out the full reasoning, what I tried, why each branch failed, and what the diagnosis says about what *would* work. It is intentionally written in first person — claims and judgement calls are mine to defend.

If you only have 2 minutes, read §3 (where it stuck) and §4 (what conditions would unblock it). Everything else is supporting evidence.

---

## 1. What I attempted (the five attack lines)

<!-- WRITE: A chronological narrative of the five major campaign families, NOT a per-version log. The per-version log already exists in `docs/stage1_tuning_log.md` (v17 → v22) and `docs/stage1_5_tuning_log.md` (v25+). This section consolidates 30+ versions into 5 named storylines, each ~3-5 sentences. The reader should leave this section knowing what conceptual moves I made and why each was the natural next step.

Structure I'm imagining:

(a) **v1–v17 — Stage 1 bring-up under pure PPO.**
    Multi-component reward, full Warp particle physics, 3 seeds. Mostly bug-hunting:
    of the 10 bugs catalogued in `docs/stage1_results_appendix.md`, the first 7
    silently broke the env (no particles in scene, 1000× wrong bucket force, etc.).
    After all bugs fixed, v17 plateaus at 1.8% mean transfer < 15% random baseline.
    Conclusion: the env is finally sound, but pure PPO can't escape the
    exploration plateau in this physics.

(b) **v18–v22 — Adding BC warm-start + sticky-grab containment.**
    Hand-coded a parametric scoop trajectory (`scripts/scoop_demo.py`) that
    achieves 22% transfer on replay alone. Trained PPO with 2000 steps of BC
    pretraining on 6400 (obs, action) pairs from this trajectory. Added a
    "sticky-grab" containment heuristic to compensate for the bucket being a
    bare AABB without walls. v22 reaches 2.4% mean with single-seed transient
    peaks of 80% — but the peak doesn't persist, the policy drifts off the
    BC manifold.

(c) **v23–v28 — KL anchor + reverse-curriculum reset with particle-state replay.**
    To stop the BC drift: deepcopy the post-BC actor-critic as a frozen
    "anchor" and add a KL-divergence term to the PPO loss. Independently,
    fix the reverse-curriculum reset to actually replay the particle field
    at each demo step (previously only joint state was reset). v28a reaches
    10.7% mean transfer; seed 123 alone reaches 28.5%.

(d) **v28b — Sanity check: freeze obs_rms after BC.** A diagnostic, not a
    serious attempt: confirms that freezing the observation normalizer
    *during PPO* breaks training catastrophically (KL → 193 076). Rules out
    a candidate "fix" that other PPO codebases use post-BC; in our setup
    the obs distribution shifts during PPO exploration so a frozen rms
    misnormalizes new states.

(e) **v29–v31 — Reproducibility check on fresh seeds.** v28a's headline
    relied on seed 123 being the strong seed. Reran the v28a config on
    fresh seeds {500, 555, 999}: v31 mean transfer **26.7 %** with all three
    seeds at 23–30% — which **settles the cross-seed reproducibility
    question** and confirms the v28a mechanism generalises beyond a single
    lucky seed. **But v31 still fails the §5 PASS gate** because PPO health
    metrics (KL, clip_fraction, explained_variance) all collapse during
    BC pretrain. The blocker is no longer reproducibility; it's PPO health.
-->

---

## 2. What failed and why — three specific lessons

<!-- WRITE: Pick the 3-4 most instructive bug / wrong-hypothesis stories from the 30+ versions. Each is ~half a page in the format "I assumed X → the actual cause was Y → here's the diagnostic that proved it". The reader should leave knowing how I think when something looks wrong.

Candidates (pick the strongest 3-4 of these):

(i) **"No learning" misdiagnosis (v17).** I read the truncated wandb summary
    and concluded "training is dead, transfer ≈ 0". The full per-component
    decomposition (15 hidden keys behind the `+15 ...` truncation) showed
    success_bonus firing 21–65% of the time and seed-123 hitting 9.2 %
    transfer at one peak. I had been chasing the wrong problem (PPO
    instability) when the actual problem was that I was reading 9 of the
    24 logged metrics. **Lesson**: never grep stdout for telemetry; always
    pull via wandb.Api. Codified in `CLAUDE.md` §2.

(ii) **Approach-reward perverse incentive (bug #8).** The approach reward
     was computed against the dynamic soil centroid, which moves *away*
     from the bucket as the bucket displaces particles. So scoop attempts
     decreased the reward — the agent learned "stay back from the heap"
     as the reward-maximizing policy. **Lesson**: shaping rewards must be
     verified against an adversarial scoop, not just inspected. Switching
     to a static `heap_center` from scene config fixed it.

(iii) **Action-smoothness penalty dominating positive reward (bug #10).**
      `smooth_penalty` averaged −10.9 in v17 — comparable in magnitude to
      the entire positive reward signal. The agent converged to a
      "minimum-action" policy that gestured toward the heap but never
      committed to the high-acceleration push needed to scoop. Lowering
      from -0.01 to -0.001 lifted `reward/transfer` 71% in v18-19.
      **Lesson**: a "small" hyperparameter is small relative to other
      reward components, not absolute. Always print component magnitudes
      side-by-side.

(iv) **"Best seed" framing trap (v28a → v31).** I was anchored by
     v28a's seed-123 reaching 28.5% — felt like the mechanism was working.
     Then seeds 7 and 42 came in at 3.4% and 0.02%. The 1400× cross-seed
     spread is invisible if you cite the best seed. **Lesson**: 3-seed
     mean is the criterion, period. Codified in `CLAUDE.md` §5.2.

(v) **PPO-health collapse vs transfer-rate (v31).** v31 hit 26.7% mean
    transfer (1.8× random baseline, 11× v17, 2.5× v28a) — but KL=585 408,
    clip_fraction=0.998, explained_variance=−0.20. These numbers say PPO
    is essentially not optimising; v31's transfer holds because the policy
    mean is pinned near the BC-faithful scoop. So "transfer up + PPO health
    catastrophic" is internally consistent: BC does the work, PPO just
    fights itself in the background. The mechanism stack works at one
    level (imitation) but not the other (refinement).
-->

---

## 3. Where it stuck (the load-bearing diagnosis)

<!-- WRITE: This is the most important section. The reader should leave knowing exactly which mechanism failed, with what evidence, and why the obvious next-tweak doesn't help. ~half page.

Core findings to convey:
- v31's 26.7% comes from BC mimicry, not PPO learning
- The PPO update path during BC pretrain drives action_std toward 0 (suspected; v32 was queued to confirm via `Policy/entropy` + `Policy/action_std` trajectories)
- A near-zero action_std turns the analytic Gaussian KL into a 1/σ² explosion — this is why anchor coef sweeps (V34/V35/V36 across {2.0, 1.0, 0.5}) didn't move KL: the variance term in the denominator dominates the coef
- This is also why "just lower the anchor coef more" doesn't work — the bottleneck isn't the coef, it's that PPO has no exploratory signal during BC pretrain because std has collapsed

Tie this back to the broader claim: "the diagnosis is at the level of how PPO and BC interact in this codebase, not at the level of any single hyperparameter."
-->

---

## 4. What conditions would unblock it

<!-- WRITE: Conditional findings, not commitments. Reframe what was previously "TODO Phase 1-5" in the README as "if I had X, here's what I'd test." The reader should leave with a concrete list of "this is solvable, here's how" — turning the project from "unfinished" to "diagnosed".

Three conditions, each with the rationale:

(a) **Entropy floor + softer anchor (v33-style).** Clamp action_std ≥ ε (e.g. 0.1)
    inside PPO updates so the analytic KL doesn't explode. Combined with a
    softer anchor coef (≤ 1.0) so PPO can actually deviate from BC when the
    advantage is large enough. Predicted: KL falls below 5, EV recovers,
    transfer holds or rises.

(b) **Mesh-particle collision via Isaac Lab (replaces sticky-grab).** The
    sticky-grab containment heuristic (`envs/scene.py:_update_carried_particles`)
    inflates `reward/load` by ~2600× (0.14 → 390) but is a hardcoded AABB pin.
    Real PhysX bucket geometry would let the policy learn containment from the
    physics, not the heuristic. Predicted: less spurious credit assignment,
    more honest learning curve. Requires Isaac Lab integration (Phase 1 in
    the original TODO; out of scope for this semester's compute / time budget).

(c) **PASS-gate threshold recalibration.** §5 sets KL < 1.0, but the
    BC-anchored PPO literature routinely reports KL = 5–15 as a healthy
    operating range. If the gate is wrong, no amount of tuning passes it.
    Predicted: with entropy floor (a) + softer anchor + relaxed KL gate to
    < 5, v33 becomes a credible PASS candidate.

Make explicit: I did not run any of these to completion in this campaign because
of compute and time budget. The codified §4.6.4 "5 consecutive FAILs → mandatory
user surface" rule (added to CLAUDE.md after V32-V36 death spiral) is what
stopped me from continuing in autonomous mode. Surfacing here, in the
retrospective, is the canonical place to do this.
-->

---

## 5. What I'd do differently

<!-- WRITE: This is the research-taste section. Talk about your own judgement calls — places where you stayed too long on a wrong hypothesis, places where you should have run a literature pass earlier, places where you over-trusted local diagnostics. ~half page.

Things to consider including:
- Should have run a literature pass on BC + PPO instability *before* spending V18–V22 tuning bug fixes. The "missing obs normalisation" bug is famous in PPO papers — I rediscovered it locally over 10 versions.
- Should have set §5 gate thresholds with literature reference, not RSL_RL defaults. KL < 1.0 was defaulted; in a BC-anchored setup it's miscalibrated.
- Should have made cross-seed reproducibility (v31) a routine habit not a v28a-era afterthought. Each headline metric should have been verified on fresh seeds before the next iteration.
- The v32-v36 anchor-coef sweep was 5 iterations of "diminishing returns toward the right direction" rationalization. Should have stopped at v34 and run lit search instead. CLAUDE.md §4.6 codifies the stop rule going forward.
-->

---

## 6. What this taught me for the next project

<!-- WRITE: This is the bridge to Part 3 (manipulation extension). Talk about what excavation taught you that you'll bring back to the standard framework. The Part 3 plan (per the email exchange that prompted this doc): take a small, sharp question on Isaac Lab Lift-Cube or similar, with a falsifiable hypothesis, and produce data — possibly even a negative result that's more rigorous than novelty.

Things to consider including:
- The instinct to design `CLAUDE.md` §5 (PASS gate as binary AND of 5 criteria) came from realizing how many ways "partial PASS" framing can deceive you. That instinct comes back to Part 3 directly.
- The §4.6 step-back discipline (literature pass + gap analysis + threshold sanity check before iterating) is now a habit, not a procedure I have to remember.
- Excavation taught me that the task itself (granular media manipulation) was not the right size for one semester — too many bugs to surface before the actual research question even came into view. Part 3 should pick a task where the pipeline is already validated, so the experiment is purely about the research question.
-->

---

## Reference

The evidence behind these claims:
- Per-version tuning history: [`stage1_tuning_log.md`](stage1_tuning_log.md), [`stage1_5_tuning_log.md`](stage1_5_tuning_log.md)
- Multi-seed quantitative results + bug catalogue: [`stage1_results_appendix.md`](stage1_results_appendix.md), [`stage0_results_appendix.md`](stage0_results_appendix.md)
- Project operating notes (PASS gate definition, anti-patterns, step-back discipline): [`../CLAUDE.md`](../CLAUDE.md)
- Session decision records: [`session_handoffs/`](session_handoffs/)
