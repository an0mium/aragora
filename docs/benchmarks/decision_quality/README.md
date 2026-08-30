# Outcome-Backed Decision Quality Corpus

This directory holds the frozen public-case contract for the outcome-backed
decision-quality benchmark. It extends the existing
[`DECISION_QUALITY_DELTA_BENCHMARK_SPEC.md`](../../plans/DECISION_QUALITY_DELTA_BENCHMARK_SPEC.md)
without claiming that a benchmark result exists yet.

## Contract

- 24 already-resolved public cases.
- Six cases in each domain: software engineering, business/operations,
  nonpartisan policy/compliance, and science/forecasting.
- Four development and two holdout cases per domain, for a 16/8 split.
- Exactly two explicit options per case and one named option whose probability
  is forecast.
- Model-visible evidence was public at or before the case information cutoff.
- Resolution evidence and three to five preregistered cruxes live only in the
  outcome sidecar.
- The sidecar names the canonical SHA-256 of the model-visible corpus.
- Any correction produces a new corpus revision and invalidates affected runs.

The JSON Schema at `corpus.schema.json` describes both documents. The CLI adds
cross-document checks, timestamp ordering, unique identifiers, exact domain
balance, and hash binding:

```bash
python3 scripts/debate_quality_benchmark.py validate-corpus \
  --corpus docs/benchmarks/decision_quality/corpus.json \
  --outcomes docs/benchmarks/decision_quality/outcomes.json
```

During corpus construction, `--allow-partial` skips only the final 24-case
domain/split count gate. It does not relax source cutoffs, outcome separation,
hash binding, or per-case semantics. A partial corpus must never be used for a
counted benchmark run.

The future runner must load the model-visible corpus independently from the
outcome sidecar. Outcomes, correct options, resolution summaries, and
preregistered cruxes must never enter model prompts.
