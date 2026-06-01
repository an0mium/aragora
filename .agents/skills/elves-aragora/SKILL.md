---
name: elves-aragora
description: Autonomous multi-batch development for the aragora repo, with every batch's validation gate bound to aragora's own governance — adversarial debate, a verifiable DecisionReceipt, and tier-appropriate settlement — instead of a bare test run. Use when the user says "run overnight", "I'm going offline", "implement this plan", "keep going without me", "do not stop", "I'll be back in the morning", or "run this end-to-end" while working inside aragora. Takes a plan, breaks it into sprint-sized batches, implements with tests, then gates each batch on a receipt-backed model quorum and the operating contract's auto-halt rules. Tier 0-2 batches can settle autonomously; Tier 3-4 always stop for human risk acceptance.
license: MIT
compatibility: Works with Codex (.agents/skills) and Claude Code (.claude/skills). Requires git, gh CLI, and the aragora CLI on PATH (`aragora --help`).
metadata:
  author: Synaptent (aragora) — forked from aigorahub/elves (MIT)
  upstream: https://github.com/aigorahub/elves
  version: "0.1.0-aragora"
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
     authorization packet; the human or a recorded merge-on-green preference does the merge.
   - **Tier 3-4** (semantic correctness, persistence, security/RBAC/auth, public API/SDK,
     migrations; or secrets/deployment/workflow policy/destructive ops/merge-authority
     self-modification): **HARD STOP.** The model quorum prepares the packet, but the run must
     pause and require explicit human risk acceptance (the `aragora/human-settlement` signal)
     before that batch is considered landed. Continue with *other* non-blocked batches if any
     exist; otherwise checkpoint and wait.

The Tier 3-4 hard stop is the aragora-native expression of Elves' "you never merge by default"
non-negotiable: autonomy is granted exactly where the governance model grants it, and revoked
exactly where it requires a human.

## Operating-contract auto-halts (always active)

Beyond the per-batch gate, the run obeys `docs/AGENT_OPERATING_CONTRACT.md` at all times. Halt
immediately and surface to the user when any of these fire:

- **MAIN RED** — any required check on `origin/main` red >30 min: stop roadmap work, bisect/fix first.
- Two consecutive same-wave PRs fail CI for distinct reasons.
- A dep bump introduces >5 transitive changes, or a consolidation diff exceeds 800 LOC.
- The work touches an **approval-required** item: GitHub Actions workflows, runner/CI matrix,
  secrets/auth, pre-commit/pre-push hooks, release workflows, major-version dep bumps, public
  API/SDK removals, schema drops/renames, branch deletion with unmerged commits, `git push
  --force`, or edits to `CLAUDE.md` / `AGENTS.md` / `scripts/nomic_loop.py` / `.env` / `secrets/`.
  These are never autonomous — pause and ask.

## Non-negotiables (aragora additions to the Elves base)

- Never merge by default; never approve a merge. Tier 3-4 always requires human settlement first.
- Every closing commit carries a `Co-authored-by: codex[bot]` (or `claude[bot]`) trailer.
- Never modify a test to make it pass; fix the code. Total test count never decreases.
- One run owns one branch + one worktree. A surprise tip move = collision, not diverge → stop.
- No destructive git (`reset --hard`, `checkout .`, `clean -fd`, `push --force`, rebase on shared).
- Receipts are mandatory per batch. A batch with no verified receipt is not complete.

## References

- `references/validation-gate-aragora.md` — the batch gate wired to aragora primitives (read first).
- `references/survival-guide-template.md` — compaction-survival brief, tool config = aragora commands.
- `references/execution-log-template.md` — per-batch log with SHA / receipt / tier / settlement fields.
- `references/kickoff-prompt-template.md` — stage + launch templates for an aragora run.

## Attribution

Forked from [aigorahub/elves](https://github.com/aigorahub/elves) (MIT). Upstream license and
credit recorded in `THIRD_PARTY_LICENSES.md` at the repo root. This fork rewrites the validation
and review layer; the autonomy/compaction-survival framework is upstream's.
