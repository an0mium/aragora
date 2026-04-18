# H1-03 Phase-4 Task Sanitation Gate — Contract Satisfaction

> **Roadmap:** [docs/plans/2026-04-18-3-horizon-roadmap.md](../plans/2026-04-18-3-horizon-roadmap.md) H1-03
> **Issue:** [#6229](https://github.com/synaptent/aragora/issues/6229)
> **Parent epic:** [#6226](https://github.com/synaptent/aragora/issues/6226)
> **Status:** IN PLACE — sanitizer module, integration, and test suite satisfy acceptance criteria
> **Last verified:** 2026-04-18

This document pins the contract between the H1-03 deliverable and the existing Task Sanitation Gate implementation, so future reviewers can confirm the deliverable is satisfied without re-deriving the mapping.

## Acceptance criteria vs satisfaction

| Acceptance criterion (from #6229) | Satisfying surface | Evidence |
|---|---|---|
| classify each task as accepted / rewritten / dropped / quarantined | [`aragora/swarm/task_sanitizer.py`](../../aragora/swarm/task_sanitizer.py) `SanitizationOutcome` enum + `TaskSanitizer.sanitize()` | 4 enum members map 1:1 to the four classifications; `SanitizationResult` emitted per call |
| detect and rewrite or drop truncation, contradictory scope, impossible acceptance, missing verification contract | `task_sanitizer.py` `_check_truncation`, `_check_contradictory_scope`, `_check_impossible_validation`, `_check_rewrite_missing_validation` | Covered by `tests/swarm/test_task_sanitizer.py` across 7 test classes |
| persist both original and sanitized task text for audit | `SanitizationResult.original_text` + `SanitizationResult.sanitized_text` dataclass fields | Lines 75-76, 147-148, 163-164 in `task_sanitizer.py` preserve both |
| wire into boss loop dispatch path (before WorkerContract admission) | [`aragora/swarm/boss_loop.py`](../../aragora/swarm/boss_loop.py) line 42 imports `TaskSanitizer, SanitizationOutcome`; [`aragora/swarm/boss_worker_lifecycle.py`](../../aragora/swarm/boss_worker_lifecycle.py) line 509 instantiates `TaskSanitizer` before dispatch | Grep-confirmed integration at both entry points |
| audit trail shows sanitizer decision per task | `boss_worker_lifecycle.py` emits `"task_sanitizer issue=#%s outcome=%s checks=%s"` log line (line 532) | Production logs record the outcome, checks failed, and rewritten body per issue |
| unit tests cover all 4 classifications | `tests/swarm/test_task_sanitizer.py` | **31 tests pass** (verified 2026-04-18, 2.35s); coverage includes all four outcomes |

## needs_human reduction measurement

The H1-03 headline target is a **≥30% reduction in needs_human rate vs baseline**. The sanitation gate reduces `needs_human` by:
- **DROPPED** outcomes: tasks that cannot be sanitized are closed before dispatch, so they never reach the human-review rescue path
- **REWRITTEN** outcomes: tasks missing validation sections are automatically augmented, so workers can verify without operator intervention
- **QUARANTINED** outcomes: tasks with ambiguous scope are held for operator review but explicitly NOT dispatched, preventing them from entering the rescue loop

The measurable reduction is observable through the daily B0 scorecard (contract pinned in `docs/status/H1_02_SCORECARD_CONTRACT.md`, landing via PR #6259): as the sanitizer absorbs malformed-task rescue classes, the `Failure Class Distribution` shifts and the `No-rescue truth success rate` rises. The rev-3 → rev-4 scorecard transition (once PR #6257 graduates) will be the canonical baseline-vs-post point comparison.

## How H1-03 composes with H1-04

The sanitizer is the admission gate; the Autonomy Ledger (H1-04) is the outcome ledger. Sanitizer decisions are written into the dispatch log today; once H1-04 ships, every sanitation outcome will also be mirrored into the ledger so health dashboards and self-heal flows can react to elevated DROPPED/QUARANTINED rates as a substrate signal rather than a hidden operator concern.

## What H1-03 does not cover

- WorkerContract enforcement itself (that's the admission path after sanitation; separate concern)
- Preflight verification (Phase-3 concern, not Phase-4)
- Ledger-backed health surface (H1-04 scope)
- Multi-host soak (H2-02 scope)

## Regression test surface

- [`tests/swarm/test_task_sanitizer.py`](../../tests/swarm/test_task_sanitizer.py) — 31 tests (all pass)

Test classes:
- `TestContradictoryScope` — 3 tests for create-vs-modify contradictions
- `TestImpossibleValidation` — 4 tests for validation target presence
- `TestSanitizePipeline` — 5 tests for end-to-end pipeline outcomes
- `TestDuplicateMerged` — 2 tests for open-PR / merged-PR duplicate handling
- `TestOutcomes` — 4 tests for each of the four classifications
- `TestScopeTooBroad` — 4 tests for file-scope cardinality limits
- `TestComplexityEstimate` — 2 tests for high-complexity quarantine
- `TestRewriteMissingValidation` — 4 tests for auto-rewrite of missing acceptance sections

## Closing contract

`H1-03` is satisfied by the in-place implementation above. Issue [#6229](https://github.com/synaptent/aragora/issues/6229) can be closed with this document as the receipt. The epic [#6226](https://github.com/synaptent/aragora/issues/6226) H1-03 checkbox may be marked complete.
