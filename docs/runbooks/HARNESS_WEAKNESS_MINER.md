# Harness Weakness Miner

`scripts/harness_weakness_miner.py` is advisory self-harness tooling for issue
#8973. It reads gate receipts, exported PR/issue comments, and conductor ledger
records, then produces a ranked weakness report for the harness-edit proposer.
It does not mutate receipts, GitHub comments, branches, evidence, settlement
state, or merge gates.

Default usage keeps the report local under `.aragora/`:

```bash
python3 scripts/harness_weakness_miner.py \
  --output .aragora/harness-weakness-reports/latest.md \
  --json
```

The production path uses the bounded Claude consult helper to classify examples
by causal mechanism and harness surface. The classifier output feeds two passes:

- `taxonomy_seeded`: groups examples against
  `docs/artifacts/2026-07-reviewer-failure-taxonomy.md`.
- `emergent_bottom_up`: groups model-written evidence summaries by explicit
  emergent cluster keys or, when those are absent, a local density-style
  similarity pass over the summaries.

The `--since-days` window applies to every input source. Examples with missing,
malformed, stale, or future timestamps are excluded so untrusted metadata cannot
inflate recent weakness counts.

Tests and offline repros should avoid live model calls by supplying fixture
classifications:

```bash
python3 scripts/harness_weakness_miner.py \
  --input-json tests/fixtures/harness-weakness/examples.json \
  --classification-json tests/fixtures/harness-weakness/classifications.json \
  --output /tmp/harness-weakness-report.md \
  --json
```

Committed scheduling, required-check wiring, branch-protection changes, and
automatic harness edits are intentionally out of scope. A report is reviewable
input only; follow-up changes need their own bounded PRs and normal gates.
