# H1-01 Rev-4 Promotion Readiness

Last updated: 2026-05-13T03:27:10Z

This is the operator-facing readiness surface for promoting the staged rev-4 benchmark corpus into the canonical B0 truth loop.

## Verdict

- Status: `promotion_ready`
- Decision: Ready to promote the first canonical rev-4 slice.
- Staging corpus: `tests/benchmarks/corpus_rev4.json`
- Metrics source: `.aragora/overnight/boss_metrics.jsonl`
- Promotion target: `docs/benchmarks/corpus.json`

## Corpus

- Corpus id: `tw-01-bounded-execution-v1`
- Revision: `4`
- Manifest status: `staging`
- Recorded on: `2026-04-18`
- Total staged issues: `33`

## Dispatch Evidence

| Metric | Value |
| --- | --- |
| Dispatch floor for first canonical slice | 15 |
| Metrics-backed staged issues eligible for canonical promotion | 15 |
| Staged issues still missing metrics-backed evidence | 18 |
| Additional metrics-backed dispatches needed | 0 |
| Advisory dispatch evidence from any source | 15 |
| ...via metrics ledger only | 15 |
| ...via merged/open boss-harvest PR only (advisory) | 0 |

## Next Metrics Evidence Gaps

`#5128`, `#5130`, `#5188`, `#5788`, `#5791`, `#5792`, `#5793`, `#5794`, `#5801`, `#5808`

These are the earliest staged issues missing metrics-backed `worker_outcome` rows; verify live issue state before dispatching new work.

## Execution-Class Coverage

| Execution class | Dispatched | Total | Missing |
| --- | ---: | ---: | ---: |
| `docs_reconciliation` | 1 | 2 | 1 |
| `exception_narrowing` | 2 | 8 | 6 |
| `missing_test_coverage` | 7 | 10 | 3 |
| `silent_exception_replacement` | 0 | 8 | 8 |
| `small_refactor` | 3 | 3 | 0 |
| `validation_tightening` | 2 | 2 | 0 |

## Dispatched Issues

`#5176`, `#5180`, `#5185`, `#5187`, `#5197`, `#5198`, `#5200`, `#5426`, `#5427`, `#5428`, `#5764`, `#5765`, `#5789`, `#5790`, `#5844`

## Missing Evidence

`#5128`, `#5130`, `#5188`, `#5788`, `#5791`, `#5792`, `#5793`, `#5794`, `#5801`, `#5808`, `#5809`, `#5810`, `#5811`, `#5812`, `#5821`, `#5823`, `#5825`, `#5883`

## Promotion Rule

Promote only a first canonical rev-4 slice after at least 15 staged entries have metrics-backed dispatch evidence: at least one `boss_metrics.jsonl` row for the issue with a recorded `worker_outcome`. Merged or open boss-harvest PRs are useful advisory evidence, but they are not sufficient for canonical corpus promotion because `tests/benchmarks/test_corpus_freshness.py` requires metrics-backed dispatch history for every `in_progress` entry.
