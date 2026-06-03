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

Every batch runs through `references/validation-gate-aragora.md`, which encodes aragora's **real
proof-first land loop** — the same gates the repo enforces on `main`, not a weaker stand-in. A
batch is PR-grounded: evidence and checks bind to a PR head SHA, not a local diff. A batch is
**not** complete until all of the following hold, in order:

0. **Tier 4 first, before code.** If the batch is Tier 4 or touches an approval-required surface
   (secrets/auth, workflows, runner/CI matrix, branch protection, release, destructive/irreversible
   ops, public API/SDK removal, schema drop/rename, **merge-authority self-modification** to
   `aragora/cli/commands/review_queue.py`, or protected files): **HARD STOP for human pre-approval
   *before* implementing anything.** Do not write the change and then pause. Move to another
   unblocked batch meanwhile.
1. **Local truth (necessary, not sufficient).** `pre-commit run --all-files`, `mypy` (no new errors
   above `.mypy-baseline`), the relevant `pytest` slice. Total test count must never decrease.
   Local green does **not** close a batch.
2. **Draft PR + required checks green.** Open the batch's PR as a **draft** and confirm all
   required CI checks pass at the exact head: `gh pr checks <pr> --required` — `lint`, `typecheck`,
   `sdk-parity`, `Generate & Validate`, `TypeScript SDK Type Check`, and the enforcing
   `aragora-merge-quorum` check (`.github/workflows/aragora-merge-quorum.yml`). Exact-head evidence
   expires on head drift; re-collect after any repair push.
3. **Independent model-quorum evidence at the exact head.** Run `aragora review-pr <pr> --reviewer
   claude` and `aragora review-pr <pr> --reviewer codex` (two distinct provider families) against
   the live PR head, plus a focused adversarial dogfood note. If a review is `changes_requested`,
   **repair first** — the findings are usually real; never silence a reviewer. Lint the evidence
   comment for countability before posting: `aragora review-queue evidence-lint --pr <pr>
   --head-sha <sha> --body-file <file> --json` (expect `would_count: true`), then post it.
4. **Receipt + authorization surfaces agree.** Produce a `DecisionReceipt` and verify it:
   `aragora verify <receipt.json>` (binds the decision to the reviewed SHA; no receipt → not
   closed). Then `aragora review-queue merge-packet --pr <pr> --json` shows the entry satisfied
   (not blocked except draft) **and** `python3 scripts/settle_one_pr.py --pr <pr> --json` returns
   `blockers == ['PR is draft']` — i.e. the only thing left is that the PR is still a draft. Both
   are read-only; neither merges.
5. **Tier classification + settlement.** Classify against `docs/REVIEW_AUTHORITY_PRINCIPLES.md`:
   - **Tier 0-2** (docs/tests; additive internal; live automation/CLI/observability): when steps
     1-4 hold — required checks green, ≥2 distinct-family signals (Tier 2 also needs focused
     dogfood + no unresolved dissent), verified receipt, and a clean `settle_one_pr` dry-run —
     mark the draft ready (`gh pr ready <pr>`) and settle via the repo's protected **squash**
     path. **Never `--admin`, never a bypass**; the `aragora-merge-quorum` required check still
     gates the non-draft PR. Record settlement with `aragora review-queue record-settlement` /
     `act`.
   - **Tier 3-4** (semantic correctness, persistence, security/RBAC/auth, public API/SDK,
     migrations; or secrets/deployment/workflow policy/destructive ops/merge-authority
     self-modification): **HARD STOP.** The model quorum prepares the packet, but the run must
     pause and require explicit human risk acceptance (the `aragora/human-settlement` commit
     status; Tier 4 also needs pre-approval *before implementation* per step 0) before that batch
     counts as landed. Keep the PR draft until then. Continue with *other* unblocked batches if any
     exist; otherwise checkpoint and wait.

The Tier 3-4 hard stop — and the Tier 4 pre-implementation stop — are the aragora-native
expression of Elves' "you never merge by default" non-negotiable: autonomy is granted exactly
where the governance model grants it, and revoked exactly where it requires a human. The agent
never admin-merges and never bypasses a gate.

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

- **Never admin-merge; never bypass a gate.** Tier 0-2 settle by marking the draft ready and
  letting the repo's protected squash + `aragora-merge-quorum` required check land it — never
  `--admin`. Tier 3-4 always require human settlement first; Tier 4 also requires human pre-approval
  *before* implementation.
- The batch gate is PR-grounded and uses the repo's real surfaces:
  `gh pr checks <pr> --required`, `aragora review-pr`, `aragora review-queue evidence-lint` /
  `merge-packet`, `scripts/settle_one_pr.py`, `scripts/settle_tier4_pr.py`. Local tests alone never
  close a batch.
- Every closing commit carries a `Co-authored-by: codex[bot]` (or `claude[bot]`) trailer.
- Never modify a test to make it pass; fix the code. Total test count never decreases.
- One run owns one branch + one worktree. A surprise tip move = collision, not diverge → stop.
- No destructive git (`reset --hard`, `checkout .`, `clean -fd`, `push --force`, rebase on shared).
- Rollback tags are namespaced by branch (`elves/<branch>/pre-batch-N`) — never bare
  `elves/pre-batch-N`, which collides globally across runs.
- Receipts are mandatory per batch. A batch with no verified receipt (`aragora verify`) is not
  complete.

## References

- `references/validation-gate-aragora.md` — the batch gate wired to aragora primitives (read first).
- `references/survival-guide-template.md` — compaction-survival brief, tool config = aragora commands.
- `references/execution-log-template.md` — per-batch log with SHA / receipt / tier / settlement fields.
- `references/kickoff-prompt-template.md` — stage + launch templates for an aragora run.

## Attribution

Forked from [aigorahub/elves](https://github.com/aigorahub/elves) (MIT). Upstream license and
credit recorded in `THIRD_PARTY_LICENSES.md` at the repo root. This fork rewrites the validation
and review layer; the autonomy/compaction-survival framework is upstream's.
