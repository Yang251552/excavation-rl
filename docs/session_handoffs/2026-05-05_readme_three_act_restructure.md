# Session Handoff — 2026-05-05 — README narrative restructure (three-act portfolio framing)

To resume this work in a new session: tell the next session "read latest handoff" — CLAUDE.md §10 will route it here automatically. Acknowledge in one sentence and execute §2 only if continuing this thread; otherwise honour §3 (don't re-litigate) and proceed with the user's actual ask.

---

## 1. What this session completed (facts)

- **Reframed the project as Part 2 of a three-repository research portfolio** instead of "an RL framework working toward a complete excavation policy". Story line: Part 1 = `isaac-lab-manipulation` (production-framework competency), Part 2 = this repo (diagnostic case study), Part 3 = granular pick on Isaac Lab (research-budget reset, retains granular-media theme).
- **Rewrote `README.md` lines 1–17**: replaced old project description + "Current status" with three-act portfolio block + headline finding + "How to read this repo". File now opens with title `# excavation-rl — Diagnostic Case Study in RL for Granular Manipulation`.
- **Rewrote `README.md` Motivation section (~lines 51–61)**: from "this project investigates whether RL can learn excavation, with the long-term goal of sim-to-real transfer" to "I chose excavation for Part 2 of the portfolio precisely because it surfaces failure modes that polished benchmarks hide. The point was not to ship an excavation policy in one semester."
- **Audited and updated `README.md` Quick Start**: Stage 2 commands now have explicit "(not reached)" header and disclaimer "This command runs but is not validated; not part of the diagnostic record."
- **Added Part 3 bridge section to `README.md` Stage 1 Results (line 442)**: "Bridge to Part 3 (granular pick)" — explains why two of the three "unblock" conditions require leaving the Part 2 codebase, and identifies the single most valuable Part 3 question (sticky-grab necessity test under real PhysX collision).
- **Replaced `README.md` "Current Progress & TODO" section (~70 lines, including 5-phase TODO)** with shorter `## What This Repo Contains` (~28 lines): pure inventory framing, no future-work commitments. Final paragraph adds the three-act portfolio role summary.
- **Refined Part 3 description in `README.md` line 10** to be specific to *granular pick on Isaac Lab*, naming the engineering-budget-reset rationale and the sticky-grab necessity test as the example diagnostic question.
- **Created `docs/retrospective.md` (192 lines, skeleton only)**: 6 numbered sections (What I attempted / What failed and why / Where it stuck / What conditions would unblock it / What I'd do differently / What this taught me for the next project), each with `<!-- WRITE: ... -->` blocks describing what the section should accomplish + candidate bullet points. **The user writes the actual prose, not me.**
- **Fixed broken anchor `#phase-3-ablation-experiments--evaluation` (line 687, Design Decisions table)** — pointed to a deleted TODO Phase 3 section. Replaced with a description of "ablation validation never reached" + pointer to retrospective §4.
- **Created `docs/handoff_template.md` (102 lines)**: convention + 8-section format + storage convention (auto-generated topic slug, single flat folder, immutable). Template covers writing handoffs only; reading flow is in CLAUDE.md §10.
- **CLAUDE.md additions / refinements during the session** (kept from earlier turns of this session):
  - §4.6.4 — Hard cap on autonomous iteration (5 consecutive FAILs → mandatory user surface)
  - §4.6.5 — Resuming autonomous iteration after §4.6.2 (5-step decision tree: classify lit-search output → branch by category → counter reset rules → post-iteration checks)
  - §10 — Session Handoff pointer (3 inline lines covering read flow + write trigger; details in `docs/handoff_template.md`)
  - §6 anti-patterns — added cross-ref to §4.6.4 hard cap; pre-existing dedup of v28 / V32-V36 stories kept
  - §3 bug catalogue — preserved as 10-row cheat sheet pointing to `docs/stage1_results_appendix.md` as single source of truth
  - §7 — Two-threshold disambiguation (bonus 0.3 vs termination 0.8)
  - §7.5 — Stage 1.5 artefact pointers (BC demo paths, KL anchor flag, containment hook, reverse-curriculum, particle snapshot replay)
- **§9 consistency audit run on each structural edit**: no broken anchors, no stale framing in editable docs (README / CLAUDE.md / handoff_template.md / retrospective.md). Tuning logs and results appendices retain "Stage 1.5 continues" historical references — these are immutable snapshots per CLAUDE.md §4.5 and were not edited.

## 2. Queued next actions (priority order)

**Do these first unless the user explicitly redirects.**

1. **Replace `<TODO: Part 1 GitHub URL>` in README.md line 8** with the real `isaac-lab-manipulation` GitHub URL. Cosmetic but blocks the cold-landing reader from following the Part 1 link. **User responsibility** — only the user knows the URL.
2. **Author writes `docs/retrospective.md` content**. The skeleton has `<!-- WRITE: ... -->` blocks in 6 sections; user fills them with their own prose, then deletes the WRITE comments. The retrospective is load-bearing for the three-act story — without it, the README's "deliverable is the diagnosis" claim has no destination. **User must write personally** — first-person voice and judgement calls cannot be outsourced.
3. **(Optional, after Part 3 work begins)** revisit README line 10 + line 442 to swap the placeholder example "sticky-grab necessity test" for the actual Part 3 diagnostic question chosen, if it differs.
4. **(Optional, after Part 3 repo is published)** add the Part 3 GitHub URL to README line 10 (currently "Released separately when ready").

## 3. Decisions that should not be re-litigated

- **Part 3 will be *granular pick on Isaac Lab's standard manipulation stack*, not a continuation of excavation.** User explicitly chose this in this session ("act3我决定做一个有 excavation 味道的 manipulation 扩展" + selecting option A: Cube → 颗粒物 granular pick). The "engineering budget reset" framing — keeping the granular-media theme but moving to Isaac Lab + rsl_rl — is the deliberate research-judgement move. Don't propose Part 3 as "continue tuning excavation".
- **The retrospective is a separate file (`docs/retrospective.md`), not embedded in README.** Decision rationale: README is for cold-landing readers (5–30 second skim), retrospective is for deep readers in the user's own voice. README line 16 ("Read the retrospective if you have only a few minutes") + 8 inline links across the README route deep readers there. Don't propose merging retrospective into README.
- **Storage convention for handoffs is `docs/session_handoffs/<YYYY-MM-DD>_<topic-slug>.md`, single flat folder, no INDEX.md, topic slug auto-generated by LLM.** Codified in `docs/handoff_template.md`. Don't propose subfolders, INDEX, or asking the user for a topic name.
- **Three-act framing is the canonical project narrative.** It lives at the top of README (lines 1–17) plus 6 reinforcement anchors. Don't reframe to "Stage 0 / 1 / 2" or "completion-track" framing — those are dead.
- **Stage 2 (curriculum + DR) was never run and won't be run in this repo.** Stage 2 commands are scaffolded in code and remain visible in README Quick Start, but with explicit "(not reached)" disclaimer. Don't propose running Stage 2; per the retrospective, Stage 2 results would not be informative without first fixing the BC-PPO interaction.
- **The TODO Phase 1–5 list is permanently deleted from README.** Conditional findings ("if I had Isaac Lab integration..." etc.) live in `docs/retrospective.md` §4 as "what would unblock", framed as diagnostic conclusions not commitments. Don't propose re-adding a TODO list.
- **Tuning logs are immutable** (CLAUDE.md §4.5). The "Stage 1.5 continues" historical references in `docs/stage1_5_tuning_log.md` and `docs/stage1_results_appendix.md` are correct as point-in-time snapshots and must not be edited. Don't propose retroactively editing them to match the new framing.

## 4. Quantitative summary

This session was a documentation/narrative restructure — no training runs, no new wandb data, no metric changes. The previous v17 → v22 → v28a → v31 quantitative results are unchanged (see `docs/stage1_results_appendix.md` and `docs/stage1_5_tuning_log.md`).

Latest training-side state remains:
- v31 mean transfer = 26.7 % across fresh seeds {500, 555, 999} (per-seed 30.1 / 27.2 / 22.8 %)
- v31 §5 PASS gate: FAIL (transfer ✓, CoV ✓; KL = 585 408, clip_fraction = 0.998, EV = −0.20 all fail)
- Diagnosis: BC pretrain drives action_std → 0; analytic Gaussian KL explodes; transfer holds because policy mean stays pinned near BC's faithful scoop replication

## 5. Candidate next directions (if open)

For the **README** (this repo):
- None major. Pending the two §2 user actions, README is publication-ready.

For **Part 3 (granular pick)**, when it's started in a separate repo:
- Build on Isaac Lab's particle support directly; do NOT port `soil/particle_system.py` from this repo
- Use `rsl_rl.runners.OnPolicyRunner` as the training loop; do NOT port the custom `PPOTrainer`
- The single most valuable diagnostic question (per retrospective §4): with real PhysX bucket-mesh containment, does the policy still need a containment heuristic? Result either way is informative
- Keep CLAUDE.md operating notes (§5 PASS gate, §4.6 step-back discipline, §10 handoff protocol) as templates — they generalise to any RL training campaign

For **excavation-rl future revisits** (if user wants to come back after Part 3):
- The v33 idea (entropy floor + softer KL anchor) is logged in retrospective §4 but explicitly not committed; revisiting requires a new tuning-log entry under CLAUDE.md §4.5 discipline
- v32 action_std diagnostic was queued in the older 2026-05-03 handoff and remains unrun — consider whether running it adds value vs investing the same time in Part 3

## 6. Currently running / waiting

- No training runs active. v31 finished in the previous session; no new launches in this session.
- No `ScheduleWakeup` pending.
- Plan file at `/Users/yangchenghan/.claude-account2/plans/1-isaac-lab-manipulation-cheerful-bentley.md` was created during this session for plan-mode use; the plan has been fully executed and the file can be deleted (it's outside the project repo).

## 7. References

For cold pickup (read in this order if context is thin):
- `README.md` lines 1–17 (top three-act framing block + headline finding) — establishes the project's role in the portfolio
- `docs/retrospective.md` (skeleton; **needs author's prose**) — the diagnostic narrative
- `docs/stage1_results_appendix.md` — 10-bug catalogue (single source of truth) + multi-seed v17/v22/v28a/v31 metric tables + wandb URLs + reproducibility commands
- `docs/stage1_5_tuning_log.md` — per-version Stage 1.5 history (v25 → v31), immutable
- `CLAUDE.md` — project operating notes (§4.5 tuning log discipline, §4.6 step-back rules, §5 PASS gate, §7.5 Stage 1.5 artefacts, §10 handoff pointer)
- Older session handoffs in `docs/session_handoffs/` — `2026-05-03_*.md` files cover the v28 / v31 / step-back-discipline sessions that preceded this narrative-restructure work

## 8. Unresolved threads

- **README Motivation paragraph references "(see [Stage 1 Results](#stage-1-results), bug #8)" without explaining bug #8 inline.** A reader who jumps to Motivation cold won't know what bug #8 is until they follow the link. Acceptable since the link works, but a Motivation-first reader gets a forward reference. Not worth fixing unless the user wants to.
- **Part 3 repo URL is intentionally not a dead link** (line 10 says "Released separately when ready" instead of placeholder URL). When the Part 3 repo is created, return here to update both line 10 (URL) and line 726 (What This Repo Contains portfolio summary mentions Part 3).
- **Plan file outside repo** at `/Users/yangchenghan/.claude-account2/plans/1-isaac-lab-manipulation-cheerful-bentley.md` is now a stale artefact (plan executed). User can delete; not in this repo so doesn't affect README/docs integrity.
- **Older 2026-05-03 handoffs in `docs/session_handoffs/` still contain "queued action: backfill V26-V28 tuning log entries" and "v32 action_std diagnostic"** — these were valid at time of writing but are now superseded by this restructure (Part 3 takes priority over continuing v32 in this repo). New session reading the *latest* handoff (this file) sees the current state; older handoffs are correct as point-in-time snapshots and not edited.
- **isaac-lab-manipulation README and manipulation extension README will need to know about *this* repo's framing** (the three-act story is bidirectional — Part 1 and Part 3 both need to mention they are part of a three-repo portfolio). Out of scope for this session, but flag when working on Part 1 or Part 3 README.
