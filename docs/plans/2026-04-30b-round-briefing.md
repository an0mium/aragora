# Round 2026-04-30b — H1-01 Promotion + Reliability Wedge Push

**Author**: Droid (`spec-mode` plan approved by founder via `ExitSpecMode`)
**Window**: 2026-04-29T20:30Z → 2026-04-29T22:55Z
**Phases**: A → J (10 phases)
**PRs opened**: 4 (#6828, #6829, #6831, #6832)
**Round artifacts**: `.aragora/evolve-round/2026-04-30b/` (gitignored)

This briefing aggregates a 10-phase round whose binding constraint was
pushing the H1-01 reliability wedge toward promotion of the rev-4
staging corpus, while exercising AGT-04 / AGT-03 / gauntlet on the
round's own deliverables and patching the boss_metrics empty-prompt
anomaly that was hiding a closed-issue retry loop.

## Summary table

| Phase | Name | PR | Tier | Status |
|---|---|---:|:---:|---|
| A | round seed + queue gate baseline | — | — | complete |
| B | H1-01 dry-run dispatch protocol | [#6828](https://github.com/synaptent/aragora/pull/6828) | 2 | complete |
| C | H1-02 advisory rev-4 staging scorecard | [#6829](https://github.com/synaptent/aragora/pull/6829) | 2 | complete |
| D | tmux multi-agent dispatch dialog | — | — | plan-only-with-runnable-harness |
| E | AGT-04/AGT-03 self-dogfood | — | — | live-shadow |
| F | gauntlet self-verification | — | — | live-local |
| G | boss_metrics empty-prompt anomaly fix | [#6831](https://github.com/synaptent/aragora/pull/6831) | 2 | complete |
| H | debate rounds_completed=0 floor | — | — | already-on-main |
| I | AGT-05 stale-policy seed | [#6832](https://github.com/synaptent/aragora/pull/6832) | 2 | shadow-only |
| J | round briefing PR | _this PR_ | docs | complete |

## What landed

### PR #6828 — H1-01 dry-run dispatch protocol

`scripts/h1_01_dry_run_dispatch.py` reads
`tests/benchmarks/corpus_rev4.json`, fetches each issue body via `gh
issue view`, runs the production sanitizer
(`aragora.swarm.task_sanitizer.TaskSanitizer.sanitize`), and writes
namespaced `dry_run:*` rows to
`.aragora/overnight/boss_metrics_h1_01_dry_run.jsonl`. Production
ledger untouched.

**Live evidence** against all 33 staging entries:

```
fetch_ok=33  sanitizer_outcome=accepted=33  promotion_floor=15 (CLEARED 2.2x)
```

Tests: 13/13.

### PR #6829 — H1-02 advisory rev-4 staging scorecard

`scripts/h1_02_rev4_staging_scorecard.py` consumes the dry-run ledger
and emits a JSON+Markdown scorecard with `dispatched_count`,
`accepted_rate`, `promotion_floor_met`, and a per-execution-class
breakdown. Read-only; never touches the canonical rev-3 corpus.

**Live verification**:

```
rev=4 status=staging dispatched=33/33 accepted=33
accepted_rate=100.0% promotion_floor=15 (MET)
```

Tests: 15/15.

### PR #6831 — boss_metrics dispatch_skip_reason

Survey of the production ledger on origin/main today found
**289/406 rows (71%)** with `prompt_chars=0` and **35 of those rows on
issue #1**, a closed/merged dependabot PR. The boss-loop was
retry-looping on a closed issue but each row was indistinguishable
from a real dispatch attempt that produced a 0-char prompt.

The fix is purely additive: a new `dispatch_skip_reason` field with
values `needs_human_no_prompt`, `dispatch_dropped_no_prompt`,
`no_work_orders`, or `null`. Existing readers (the
`prompt_chars<=0` filter in `issue_scanner._load_metrics_rows`) keep
working unchanged.

Tests: 19/19, including 4 new regressions in `TestDispatchSkipReason`.

### PR #6832 — AGT-05 stale-policy shadow

`aragora/reputation/stale_policy.py` adds a named, reviewable
predicate (`is_stale`, `StalePolicy`, `StaleDecision`) for the
stale-claim question that several call sites currently roll their own
age math for. Conservative defaults (7 / 30 / 180 days) mirror the
existing 30-day half-life. Shadow-only: no production code path calls
into it yet.

Tests: 17/17.

## Cross-cutting verification

### Phase E — AGT-04/AGT-03 self-dogfood

Opened **two synthetic markets** at
`.aragora/evolve-round/2026-04-30b/markets/` (round-local store, never
touches `.aragora_markets/`):

| PR | Market ID | Kind |
|---|---|---|
| #6828 | `mkt_pr_merge_b5de94d0f606d71f` | pr_merge |
| #6829 | `mkt_pr_merge_c54e4a3c97d72c3a` | pr_merge |

Took **6 positions** with three synthetic agents (oracle-droid 0.90,
skeptic-codex 0.65, bear-claude 0.40). Markets remain open; a future
round can resolve them and re-run the calibration leaderboard against
the resulting Brier deltas.

### Phase F — gauntlet self-verification

Wrote `phase-f-gauntlet-input.md` listing this round's three
load-bearing claims and ran:

```
aragora gauntlet --agents demo,demo --profile quick \
  --no-redteam --no-audit --no-probing
```

**Verdict: APPROVED, 95% confidence**, 0 critical/high findings.
Limitation: demo agents are offline echo agents; without
`ANTHROPIC_API_KEY`/`OPENAI_API_KEY` the LLM-driven gauntlet was not
exercised. Pipeline integration verified end-to-end.

### Phase D — tmux harness (gated)

Queue-empty-60min gate evaluated to **CLOSED** (most recent foreign
merge #6827 was 8.4 minutes prior). Per the standing rule, did not
launch tmux harness; persisted plan-only receipt and a runnable
gate-aware bash harness at
`.aragora/evolve-round/2026-04-30b/dogfood/phase-d-tmux-harness.sh`
(`gate` / `launch` / `status` / `teardown` subcommands). Future rounds
can `bash phase-d-tmux-harness.sh launch` once the gate opens.

### Phase H — debate rounds_completed=0

Verified Option I floor is **already implemented on origin/main** at
`aragora/debate/debate_state.py:432`:

```python
rounds_used = self.current_round or self.result.rounds_used or self.partial_rounds
```

with regression test
`tests/debate/test_debate_state.py::test_finalize_falls_back_to_partial_rounds`
that even cites the original 2026-04-28 dogfood discovery. No PR
needed; phase complete on verification.

## Standing rule

I do not author-merge. PRs #6828, #6829, #6831, #6832 await Codex
signal + CI green.

## Receipts

Round-local artifacts:

- `state.json` — phase ledger
- `dogfood/phase-{a..j}-receipt.json` — per-phase machine-readable receipts
- `dogfood/h1-01-dry-run-summary.{json,md}` — Phase B output
- `dogfood/h1-02-rev4-staging-scorecard.{json,md}` — Phase C output
- `dogfood/phase-d-tmux-harness.sh` — runnable harness
- `dogfood/phase-e-summary.md` — markets and positions table
- `dogfood/phase-f-gauntlet-receipt.json` — gauntlet decision receipt
- `markets/` — round-local synthetic-market store

All artifacts under `.aragora/` are gitignored; nothing in this PR
touches the production `.aragora/overnight/boss_metrics.jsonl` or the
canonical `.aragora_markets/` store.
