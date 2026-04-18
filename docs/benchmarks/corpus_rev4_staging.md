# Benchmark Corpus rev-4 (Staging Manifest)

> **Status:** staging — not yet the canonical corpus on current `main`.
> **Canonical rev-3 corpus:** `docs/benchmarks/corpus.json`.
> **Staging rev-4 corpus:** `tests/benchmarks/corpus_rev4.json`.
> **Parent issue:** [H1-01 #6227](https://github.com/synaptent/aragora/issues/6227).
> **Parent epic:** [#6226](https://github.com/synaptent/aragora/issues/6226).

## Why a staging manifest

H1-01 requires a frozen rev-4 benchmark corpus with ≥30 bounded tasks spanning
canonical safe task classes. The rev-3 corpus freshness invariant in
`tests/benchmarks/test_corpus_freshness.py` additionally demands that every
`in_progress` entry has at least one recorded `worker_outcome` in
`.aragora/overnight/boss_metrics.jsonl`. Those two requirements cannot both
be satisfied in a single PR before the Mac Studio boss loop has dispatched
the new entries, so rev-4 lands first as a **staging** manifest and is
promoted to the canonical corpus only once dispatch evidence accumulates.

This preserves two invariants the rev-3 honesty audit established:

1. Benchmark membership never silently includes aspirational-but-never-run
   issues (the exact failure mode the audit flagged for rev-2).
2. Canonical benchmark truth publication (`docs/status/B0_BENCHMARK_TRUTH_STATUS.md`)
   stays fresh and free of un-dispatched entries.

## What the rev-4 manifest contains

- 33 open bounded tasks sourced from real GitHub issues labelled `autonomous`
  and/or `boss-stuck`, or with an obvious single-PR-resolution shape.
- Coverage of six canonical safe task classes:
  - `missing_test_coverage` (target 10, observed 10)
  - `exception_narrowing` (target 8, observed 8)
  - `silent_exception_replacement` (target 8, observed 8)
  - `validation_tightening` (target 2, observed 2)
  - `small_refactor` (target 3, observed 3)
  - `docs_reconciliation` (target 2, observed 2)
- A self-signed SHA-256 digest over the canonical JSON body (minus the
  signature field itself, issues sorted by `issue_id`). Edits to the
  manifest that are not accompanied by a regenerated signature fail
  `tests/benchmarks/test_corpus_rev4_manifest.py::test_rev4_signature_matches_body`.

## Signature regeneration

```bash
python3 -c "
import json, hashlib, pathlib
p = pathlib.Path('tests/benchmarks/corpus_rev4.json')
d = json.loads(p.read_text())
body = {k: v for k, v in d.items() if k != 'sha256_signature'}
body['issues'] = sorted((dict(i) for i in body['issues']), key=lambda x: int(x['issue_id']))
d['sha256_signature'] = hashlib.sha256(
    json.dumps(body, sort_keys=True, separators=(',', ':')).encode()
).hexdigest()
p.write_text(json.dumps(d, indent=2) + chr(10))
"
```

## Promotion path to canonical rev-4

1. Mac Studio boss loop dispatches each staging entry at least once; each
   attempt writes a `worker_outcome` row to
   `.aragora/overnight/boss_metrics.jsonl`.
2. Once ≥15 entries have dispatch evidence (first bounded slice worth
   scoring), open a follow-up PR that:
   - Moves the dispatched entries (with `expected_status: "in_progress"`)
     into `docs/benchmarks/corpus.json` as a rev-4 revision block.
   - Adds a `revision_log` entry explaining the promotion and linking back
     to this staging manifest.
   - Leaves un-dispatched staging entries in
     `tests/benchmarks/corpus_rev4.json` until they also accumulate
     dispatch evidence.
3. `scripts/build_benchmark_truth_artifact.py` runs against the promoted
   canonical corpus; `docs/status/generated/benchmark_scorecards/` is
   regenerated as part of the promotion PR.

## Why this unblocks H1-02

H1-02 (daily no-rescue scorecard) depends on a stable, signed corpus with
enough membership to produce a meaningful zero-rescue percentage. Even
before promotion, the staging manifest:

- gives H1-02 a 33-entry target to score against independently of rev-3;
- lets reviewers verify that the rev-4 membership has not silently drifted
  (signature mismatch fails CI);
- defines the promotion contract so H1-04 self-heal work can rely on a
  predictable growth path for the canonical corpus.

## What is explicitly out of scope for this PR

- Changing the canonical rev-3 corpus at `docs/benchmarks/corpus.json`.
- Editing the rev-3 freshness invariant.
- Regenerating `docs/status/generated/benchmark_scorecards/`.
- Wiring `scripts/build_benchmark_truth_artifact.py` to read the staging
  manifest — that is the promotion step above.
- Producing baseline zero-rescue % — that is the H1-02 work that depends on
  this manifest.
