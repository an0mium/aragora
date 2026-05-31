---
name: elves-aragora
description: Crash-resumable, plan-driven overnight execution bound to aragora's proof-first gates. Use when the user says "run overnight", "I'm going offline", "implement this plan", "keep going without me", "don't stop", "I'll be back in the morning", or otherwise asks for long unattended autonomous work in this repo. Unlike a generic overnight runner, every batch lands through aragora's real gates — model-quorum evidence, decision receipts, draft-PR only, never admin merge — and the run survives a crash or context compaction via an on-disk execution log.
license: MIT
metadata:
  author: Aragora Contributors
  version: "0.1.0"
  derived-from: aigorahub/elves (MIT) — survival-guide / execution-log scaffolding pattern
  argument-hint: Path to a plan file (docs/plans/*.md), or plan text directly.
---

# Elves-Aragora — governed overnight runs

You are the night shift. The user is the day manager who handed you a written plan
before going offline. Execute it batch by batch — with testing, model-quorum review,
receipts, and documentation — until the plan is done or you hit a genuine blocker.

**You never admin-merge. You never bypass a gate. The user (or the quorum) merges.**
**The run must survive a crash:** every batch's state lives in an on-disk execution
log so a fresh session can resume from live truth, not transcript memory.

This skill does **not** reinvent aragora's autonomy loop — that already exists
(`AGENTS.md`, `docs/AGENT_OPERATING_CONTRACT.md`, `scripts/codex_worktree_autopilot.py`,
the merge-quorum gate, decision receipts). It adds the missing piece: a **resumable,
plan-driven wrapper** so those gates run unattended overnight and recover from interruption.

## Install / location

This skill lives at `.agents/skills/elves-aragora/` (the Codex per-repo skills path —
read by the Codex CLI, IDE extension, and desktop app). Claude Code reads
`~/.claude/skills/`; because this repo `.gitignore`s `.claude/`, Claude Code users copy
or symlink this directory into `~/.claude/skills/elves-aragora/`. Restart the agent to re-index.

## Three documents (the scaffolding)

Borrowed from elves (credit: `aigorahub/elves`, MIT), adapted to aragora gates:

1. **Plan** — `docs/plans/<name>.md`. The user authors this: goal, milestones broken
   into sprint-sized batches, each batch independently shippable and testable.
2. **Survival Guide** — `docs/plans/<name>.survival.md`. Per-project gates and invariants
   the night shift must obey. For aragora the gate is NOT `npm test` — see "Batch gate" below.
3. **Execution Log** — `docs/plans/<name>.log.md`. Append-only. After every batch step,
   write: batch id, branch, head SHA, what changed, gate results, PR number, current
   blocker. **This is the crash/compaction recovery anchor** — a resumed session reads
   this first, re-derives live truth, and continues.

Bootstrap missing docs from `~/.claude/skills/elves/references/` (plan-template.md,
survival-guide-template.md, execution-log-template.md) then rewrite the gate sections per below.

## Loop (per batch)

0. **Recover state first.** Read the execution log. Run the gate-discovery commands
   (below). Trust live truth over the transcript — a crash may have happened mid-batch.
1. **Isolate.** Work in a fresh git worktree (never the shared root if dirty/owned).
2. **Implement** the next batch with TDD. Keep batches small and independently shippable.
3. **Batch gate (aragora-specific — this replaces a plain test run):**
   - Required CI checks green (`gh pr checks <pr> --required`).
   - Genuine model-quorum evidence at exact head: run `review-pr <pr> --reviewer claude`
     and `--reviewer codex`; if a review returns `changes_requested`, **repair first** —
     the findings are usually real. Post head-grounded countable evidence
     (`review-queue evidence-lint` → `would_count` → `gh pr comment`).
   - `settle_one_pr.py --pr <pr>` blockers reduce to `['PR is draft']` only.
4. **Open a DRAFT PR** (`Closes #...`). Never mark ready unless the merge-packet and
   settle agree the only blocker is draft status, and tier ≤ 2 with no human-risk settlement.
5. **Log it** (append to the execution log) and notify (`scripts/notify.sh` if configured).
6. **Stop conditions** (hand back to the human): publisher not ready, active-owner conflict,
   Tier 3+/security/policy settlement, workflow/branch-protection change, normal merge rejected,
   or no unowned high-value batch remains.

## Batch gate — exact commands

Gate discovery (run at start of every batch and after any resume):
```
git status --short --branch
python3 scripts/publisher_freshness_check.py
python3 scripts/agent_bridge.py --json health || true
python3 -m aragora.cli.main work robot --json
python3 scripts/identify_lane_owner.py --pr <PR> --json   # skip owned lanes
python3 -m aragora.cli.main review-queue merge-packet --pr <PR> --json
python3 scripts/settle_one_pr.py --pr <PR> --json
```

Hard constraints (from `docs/AGENT_OPERATING_CONTRACT.md`):
- Never admin-merge; never edit branch protection, workflows, labels, publisher/outbox,
  launchd, or protected files (`CLAUDE.md`, `aragora/__init__.py`, `.env`, `scripts/nomic_loop.py`).
- Tier 0 = 1 model signal; Tier 2 = 2 distinct-model signals + adversarial dogfood;
  Tier 3 (semantic/security) = + human risk settlement → STOP.
- Exact-head evidence expires on head drift. Re-review after any repair push.

## Crash / compaction recovery

If you wake with no memory of an in-flight run:
1. Find the plan + `.log.md` under `docs/plans/`.
2. Read the log's last entry → branch, head SHA, last gate result, open PR.
3. Re-run gate discovery to confirm live state (the PR may have merged or drifted while you were gone).
4. Resume at the first incomplete batch step. Do not re-do completed, logged work.

## Notes
- Reference implementation of the underlying gate mechanics: the proven no-admin
  proof-first land loop, and `docs/prompts/MASTER_FANOUT_PROMPT.md` (idempotent Phase-0 discovery).
- The generic upstream skill is installed at `~/.claude/skills/elves/` for its templates; this
  aragora-native skill is the one to invoke for repo work because its gates are real, not placeholders.
