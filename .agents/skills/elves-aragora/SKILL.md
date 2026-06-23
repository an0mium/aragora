---
name: elves-aragora
description: Autonomous multi-batch development for the aragora repo, with every batch's validation gate bound to aragora's own governance — adversarial debate, a verifiable DecisionReceipt, and tier-appropriate settlement — instead of a bare test run. Use when the user says "run overnight", "I'm going offline", "implement this plan", "keep going without me", "do not stop", "I'll be back in the morning", or "run this end-to-end" while working inside aragora. Takes a plan, breaks it into sprint-sized batches, implements with tests, then gates each batch on a receipt-backed model quorum and the operating contract's auto-halt rules. Tier 0-2 batches can settle autonomously; Tier 3-4 always stop for human risk acceptance.
license: MIT
compatibility: Works with Codex (.agents/skills) and Claude Code (.claude/skills). Requires git, gh CLI, and the aragora CLI on PATH (`aragora --help`).
metadata:
  author: Synaptent (aragora) — forked from aigorahub/elves (MIT)
  upstream: https://github.com/aigorahub/elves
  version: "0.2.1-aragora"
  argument-hint: Path to plan file, or plan text directly.
---

# Elves (aragora-native)

This is an aragora-bound fork of the Elves autonomous-development skill. The unattended Ralph
Loop (try → check → feed back → repeat across sprint-sized batches, surviving context compaction)
is unchanged. **What changes is the gate.** Upstream Elves closes a batch on `npm test` plus a
GitHub PR comment. In aragora, "the tests are the watch" is necessary but not sufficient: a batch
is only complete when it has passed aragora's own evidence-first governance.

> **Positioning.** aragora's README states it "does not sell lights-out autonomy as the default
> story" and "does not advance work without evidence, review, and clear terminal states." This
> fork honors that. It keeps Elves' overnight leverage but subordinates it to receipts, model
> quorum, and human settlement. The elves work the night shift; aragora decides what is allowed
> to land.

## When to use

Trigger inside `~/development/aragora` (or any aragora checkout) when the user wants a long,
unattended run: "run overnight", "I'm going offline until 8", "implement this plan end-to-end",
"keep going without me", "do not stop". For one-off single edits, do not use this skill.

## The two-call handoff (unchanged from Elves)

1. **Stage** the run — clean the plan, generate the survival guide / learnings / execution log
   from the templates in `references/`, claim a dedicated branch + worktree, run preflight, and
   stop. See `references/kickoff-prompt-template.md`.
2. **Launch** in a fresh call with a short, behavior-heavy prompt.

## The aragora validation gate (the core difference)

Every batch runs through `references/validation-gate-aragora.md`. In short, a batch is **not**
complete until all of the following hold, in order:

1. **Local truth.** Project validation passes against current HEAD:
   `pre-commit run --all-files`, `mypy` (no new errors above `.mypy-baseline`), and the relevant
   `pytest` slice. Total test count must never decrease.
2. **Adversarial review = a debate, not a comment.** Run a heterogeneous-model adversarial review
   of the batch diff via aragora itself (e.g. `aragora ask` framed as a review of
   `git diff <default-branch>...HEAD`, or the repo's `aragora-review-gate` path). Capture the
   exact head SHA reviewed, the model/provider families, independence from the authoring lane,
   the recommendation, and any dissent. Unresolved dissent blocks the batch.
3. **Receipt.** Produce a `DecisionReceipt` for the batch decision and verify it:
   `aragora verify <receipt.json>` (and/or `aragora receipt verify`). The receipt binds the
   decision to the reviewed SHA. No receipt → batch not closed.
4. **Tier classification + settlement.** Classify the batch against
   `docs/REVIEW_AUTHORITY_PRINCIPLES.md` (Tier 0-4):
   - **Tier 0-2** (docs/tests, additive internal, live automation/CLI/observability): a green,
     receipt-backed model quorum with no unresolved dissent is sufficient to record settlement
     autonomously and continue. **Still never merge by default** — settlement records the
     authorization packet; the human or a recorded merge-on-green preference plus helper
     authorization does the merge. §Conductor never grants merge authority by itself.
   - **Tier 3-4** (semantic correctness, persistence, security/RBAC/auth, public API/SDK,
     migrations; or secrets/deployment/workflow policy/destructive ops/merge-authority
     self-modification): **HARD STOP.** The model quorum prepares the packet, but the run must
     pause and require explicit human risk acceptance (the `aragora/human-settlement` signal)
     before that batch is considered landed. Continue with *other* non-blocked batches if any
     exist; otherwise checkpoint and wait.

The Tier 3-4 hard stop is the aragora-native expression of Elves' "you never merge by default"
non-negotiable: autonomy is granted exactly where the governance model grants it, and revoked
exactly where it requires a human.

## Conductor protocol (evidence-last, anti-molasses) — read every cycle

For any long-running PR-advancing loop, the **single source of truth** is
`docs/AGENT_OPERATING_CONTRACT.md` **§Conductor**. Re-read it each cycle and obey it verbatim;
do not re-embed its rules into recursive prompts — carry only the current exact-head target +
one next action (see §Conductor's thin-prompt template). Do **not** maintain a second summary
of its rules here. "Evidence-first" in this skill means governance must be evidence-grounded;
§Conductor defines the timing rule for countable evidence.

## Operating-contract auto-halts (always active)

Beyond the per-batch gate, the run obeys `docs/AGENT_OPERATING_CONTRACT.md` at all times. Halt
immediately and surface to the user when any of these fire:

- **MAIN RED** — any required check on `origin/main` red >30 min: stop roadmap work, bisect/fix first.
- Two consecutive same-wave PRs fail CI for distinct reasons.
- A dep bump introduces >5 transitive changes, or a consolidation diff exceeds 800 LOC.
- The work touches an **approval-required** item: GitHub Actions workflows, runner/CI matrix,
  secrets/auth, pre-commit/pre-push hooks, release workflows, major-version dep bumps, public
  API/SDK removals, schema drops/renames, branch deletion with unmerged commits, `git push
  --force`, or edits to `CLAUDE.md` / `AGENTS.md` / `docs/AGENT_OPERATING_CONTRACT.md` /
  `docs/REVIEW_AUTHORITY_PRINCIPLES.md` / `scripts/nomic_loop.py` / `.env` / `secrets/`.
  These are never autonomous — pause and ask.

## Non-negotiables (aragora additions to the Elves base)

- Never merge by default; never approve a merge. Tier 3-4 always requires human settlement first.
  §Conductor's autonomy rail means "keep advancing safe units"; it does not override this merge
  rule.
- Every closing commit carries a `Co-authored-by: codex[bot]` (or `claude[bot]`) trailer.
- Never modify a test to make it pass; fix the code. Total test count never decreases.
- One **coordinator** owns the run. Serial batches share the run's branch + worktree.
  **Independent batches may fan out in parallel**: each parallel lane gets its own subagent,
  branch, and worktree (`python3 scripts/codex_worktree_autopilot.py ensure --agent claude
  --base main --force-new --print-path`); lanes must touch file-disjoint surfaces; only the
  coordinator writes the execution log / survival guide and records settlements. Within any
  one worktree, a surprise tip move = collision, not diverge → stop that lane.
- **Lanes finish synchronously — never via background watchers.** A lane's background
  monitors/waiters die with the lane; "standing by for the watcher to re-invoke me" is lane
  death (observed 4× in run-20260610: lanes ended mid-settlement and required manual
  resumption). Wait for external state with bounded *foreground* until-loops
  (`until <check>; do sleep 20; done`, hard-capped), and register every lane in the run's
  lane ledger (`.aragora/run-<id>/lanes/<lane>.json`: lane, agent_id, branch, brief,
  launched_at, status) at launch so the sentinel's liveness check can detect silent death.
- No destructive git (`reset --hard`, `checkout .`, `clean -fd`, `push --force`, rebase on shared).
- Receipts are mandatory per batch. A batch with no verified receipt is not complete. Store
  receipts under the run-scoped dir `.aragora/run-<run-id>/receipts/` (legacy
  `.aragora/receipts/` accepted) and link the path in the execution log and PR body.

## Budget, liveness, and pacing guards (defaults — a plan's Run Control may tighten them)

- **Gate attempt cap.** Two full gate cycles per batch: implement → gate; on failure revise
  once → re-gate. Still blocked after that → **park** the batch. Do not keep iterating.
- **Parking is a legal disposition.** Park = push the branch, log the dissent/evidence and
  receipt-so-far, queue the batch for human settlement, move to the next unblocked batch. A
  parked batch is never counted as done and never merges — but parking does not violate
  "unresolved dissent blocks the batch"; it *is* the documented disposition.
- **Identical-error breaker.** Three identical tool/infra errors in a row → stop retrying that
  tool. Switch to the documented fallback or park the affected batch. Never burn the night on
  one error signature.
- **Gate-tooling failure** (e.g. `aragora ask`/`aragora gauntlet` exits "no agents could be
  created"): spend ≤15 min diagnosing (`python scripts/claude_pool_verify.py`, key hydration via
  Secrets Manager — never raw keys in env), then retry once with an alternate provider set. If
  the debate still cannot run, the gate **fails closed**: Tier 0-1 batches may close on local
  truth + a single-model adversarial review explicitly logged as `degraded-evidence`; Tier ≥2
  batches park. If no gate can run for *any* batch, checkpoint and halt the run.
- **No-progress halt.** No batch closed or parked in 2 hours of wall-clock → checkpoint, write a
  postmortem entry in the execution log, re-read the plan, then either advance to the next phase
  or halt.
- **Waiting on external state** (PR CI, merge quorum): never busy-poll and never sit idle. Log
  `waiting-on:<thing>`, start another unblocked batch, and re-check at intervals matched to the
  state's real cadence (PR CI ≈ every 4–5 min; quorum evidence ≈ every 15 min). On the FIRST
  check, verify the required workflows actually started (this repo has a known permanent-pending
  failure mode) and re-trigger with `gh run rerun` if they never ran.

## Plan Run-Control precedence

If the plan file carries its own **Autonomy Contract / Run Control** section, that section
governs *scheduling policy* — attempt caps, budgets, parallel fan-out, pacing, receipt paths —
and may tighten anything in this skill. It can never weaken the governance gate: receipts,
dissent handling, tier hard-stops, approval-required surfaces, and never-merge-by-default are
floor constraints. Where a plan says "abandon batch and continue", always read it as **park**
(push + log + queue for settlement), never as counting the batch done.

## References

- `references/validation-gate-aragora.md` — the batch gate wired to aragora primitives (read first).
- `references/survival-guide-template.md` — compaction-survival brief, tool config = aragora commands.
- `references/execution-log-template.md` — per-batch log with SHA / receipt / tier / settlement fields.
- `references/kickoff-prompt-template.md` — stage + launch templates for an aragora run.

## Attribution

Forked from [aigorahub/elves](https://github.com/aigorahub/elves) (MIT). Upstream license and
credit recorded in `THIRD_PARTY_LICENSES.md` at the repo root. This fork rewrites the validation
and review layer; the autonomy/compaction-survival framework is upstream's.
