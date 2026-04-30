# Round discipline — autonomous Claude rounds

This document codifies the round-pattern that has converged across 4 consecutive
autonomous Claude rounds (2026-04-29, 2026-04-30b, 2026-04-30c, 2026-04-30d).
It is descriptive, not prescriptive: the pattern emerged from operational
observation and is documented here so future rounds (and other contributors
following the same shape) inherit the discipline rather than re-deriving it.

## When to invoke

A "round" is a bounded autonomous work session, typically 6-12 hours, that
makes thesis-aligned progress through a structured 9-phase sequence. Each
round produces ~1-6 PRs plus dogfood receipts under
`.aragora/evolve-round/<round-id>/`.

Triggers:
- Operator approves a planned round explicitly.
- Round goal is bounded enough to fit one continuous session.
- Queue pressure (open-PR depth) is below ~10. Above that, prefer review-only
  rounds.

## Phase plan template

Every round follows phases A-I:

| Phase | Purpose |
|---|---|
| A | Baseline + queue gate snapshot |
| B | Foreign-PR review + reviewer-signal contribution |
| C | Live dogfood of one specific surface (often a CLI verb or hot path) |
| D | Benchmark / measurement / quantitative validation of a recent landing |
| E | One substantive code PR — chosen reactively from B/C/D findings |
| F | Findings synthesis (RCA + improvement notes) |
| G | Multi-harness tmux dispatch — almost always plan-only due to gate |
| H | Live e2e verification of round deliverables |
| I | Round briefing PR (docs-only) |

Phases B-F are the "substance"; A, G, H, I are scaffolding.

## Round invariants

- **Disposable detached worktrees per phase** under
  `~/.claude-worktrees/aragora/round-<id>/phase-{b..i}-*/`. Worktrees are
  destroyed after the round; no shared state leaks across rounds.
- **Author-side `## Claude review` comment on every PR I open.** This is
  not the merge-quorum signal (an author cannot be the heterogeneous
  reviewer for their own PR), but it documents the author's verification
  state for receipts.
- **Standing rule held: no author-merges, no draft-flips, no GitHub
  review-state changes.** Every round PR awaits independent reviewer
  signal + operator settlement.
- **Per-phase JSON receipts** at
  `.aragora/evolve-round/<round-id>/dogfood/phase-{a..i}-receipt.json`.
  Each receipt is self-contained: phase title, status, deliverables,
  halt-class triggered, links to PRs/comments.
- **Pre-push hooks must pass**: ruff, ruff format, mypy
  (baseline-filtered), gitleaks, RBAC decorator audit, env-mutation
  audit. Hook failures are halt-class events.
- **Dogfood findings: fix inline if <30 LOC + high-confidence; else file
  follow-up.** This is the "uncover issues that can be fixed
  autonomously" pattern.

## Halt criteria

A round halts (cleanly, with partial-receipts) when any of:

- A test, lint, mypy, or pre-existing CI check regresses beyond what was
  inherited from main.
- A phase exceeds 2× its budget without producing output.
- The queue-empty-60min gate (Phase G) fails — Phase G stays plan-only.
- A foreign critical-path PR opens that touches the same files as a
  current phase. Defer to avoid collision.
- Codex A or another operator directive pauses the round.
- Pre-push hooks fail and the issue is not resolvable in <30 LOC.

When a halt fires, the receipt for the halting phase records `halt_class`,
the round briefing notes it, and the next round may resume (or not) based
on operator direction.

## PR-cap discipline

The optimal PR-output count per round depends on queue depth at round
start:

| Queue depth at start | New-code PR cap |
|---|---|
| <= 5 | 5-6 |
| 6-10 | 3-4 |
| 11-15 | 2-3 |
| 16+ | 1-2 |

Above 16 open PRs, the round should pivot to **review-only** —
contributing reviewer-signal to foreign PRs without adding to the queue.
This is the lesson from Round 30d.

## Worktree discipline

- Worktrees live under `~/.claude-worktrees/aragora/round-<id>/phase-*`,
  outside the main repo's `.worktrees/` to avoid colliding with the
  active automation lane.
- The main repo's working tree is **never modified during a round**.
  Round work happens entirely in disposable worktrees. Operators can
  inspect main without seeing round-in-progress dirt.
- A "rebase worktree" is created on demand when a round PR goes DIRTY
  due to a foreign merge; the rebase happens in the disposable worktree
  and is force-pushed back to the round branch.

## Receipt schema

Every phase emits a JSON receipt with at least:

```json
{
  "phase": "B",
  "title": "<short phase summary>",
  "status": "complete | partial | plan-only | subsumed-by-other-pr",
  "halt_class_tripped": false,
  "tier": 1
}
```

Substantive phases also emit:
- `pr_number`, `pr_url`, `head_sha_at_open` for code PRs
- `tests_pass`, `ruff`, `ruff_format`, `mypy`, `preflight`, `gitleaks`
  for verification
- `discovery_path` for dogfood-driven findings
- Free-form findings keys for investigation phases

## Round briefing PR

Phase I emits a briefing markdown to
`docs/plans/<round-id>-claude-<theme>-round-briefing.md`. The briefing
is the single closer artifact; per-phase receipts are the audit trail.

The briefing includes:
- Round goal + scope
- PRs opened (table with tier + status)
- Foreign reviews posted (table with verdict + bug-found flag)
- Round outcomes (3-5 bullets)
- Halt criteria + which gates fired/didn't
- Round invariants honored (checklist)
- Round metrics (test count, bugs found, file overlap with foreign work)
- Next-round handoff candidates

## Dogfood-driven discovery pattern

The most consistent value-producing pattern across the 4 rounds has been:

1. **Live invocation** of one CLI verb / surface per round (Phase C)
2. **Adversarial scenarios** beyond the in-PR test bank (Phase D)
3. **Inline fix** when bugs are <30 LOC + high-confidence
4. **Findings doc** consolidating non-blocking observations (Phase F)

Examples:
- Round 30c Phase B found launchd plist drift (`375 EX_CONFIG=78`
  invisible) → shipped #6842 to surface launchd exit-code in the
  freshness diagnostic.
- Round 30c Phase F found `aragora crux ImportError on get_default_agents`
  → shipped #6856 with a regression test that catches the inner-import
  path.
- Round 30d Phase C found `aragora crux --agents demo` silent fallback to
  majority consensus → shipped #6881 with operator-actionable error
  diagnostic.
- Round 30d Phase B found `#6874 test_unknown_code_raises` constructs
  invalid enum at the wrong layer → posted Claude review with 3 fix
  options.

The pattern is durable: **running the actual invocation surfaces UX bugs
that pure unit tests miss**.

## Anti-patterns

- **Author-merging**. The merge-quorum gate exists for a reason.
- **Touching the active automation-substrate lane** during a round.
  Codex/Factory has been merging substrate-repair PRs at high cadence;
  rounds avoid `aragora/cli/commands/review_queue.py`,
  `scripts/run_codex_automation_publisher*.sh`,
  `scripts/publish_automation_handoffs.py`, and any file mentioned in
  open foreign PRs.
- **Speculative work on dependents of unmerged PRs**. Wait for the
  parent to merge before opening the dependent.
- **Burning the queue with too many PRs at once**. The PR-cap
  discipline above is the explicit rule.
- **Re-deriving the round-pattern**. This document exists so future
  rounds inherit the discipline.

## Round series so far

| Round | Date | PRs opened | Theme |
|---|---|---|---|
| 2026-04-29 | initial | 4 | merge-authority self-modification + B0 freshness diagnostic |
| 2026-04-30b | morning | 5 | reliability lane (debate, freshness, multi-hop) |
| 2026-04-30c | midday | 6 | crux lineage + B0 substrate |
| 2026-04-30d | afternoon | 2 | dogfood-driven discovery (queue-pressure-aware) |

The round-pattern itself has produced the discipline that enabled the
queue-pressure-aware Round 30d.
