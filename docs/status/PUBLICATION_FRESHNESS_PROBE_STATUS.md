# Publication Freshness Probe Status

Generated at: 2026-05-17T20:48:42Z
Verdict: **drift**  (total drift: 4)
Stale threshold: 48.0h

## Canonical Metrics

pass=7 warn=1 fail=2 skip=0; drift_count=3
- [fail] canonical.km_adapters.count: observed=46, claimed=41
- [warn] canonical.test_definitions.count: observed=159378, claimed=216016+
- [fail] security.model_pins.frontier_aligned: observed=missing: OPUS_4_7, GPT_5_4, GEMINI_3_1_PRO, claimed=OPUS_4_7, GPT_5_4, GEMINI_3_1_PRO exported

## Status-doc Reconciliation

critical=0 warning=1 info=4 total=5; drift_count=1
- [warning] connectors/STATUS.md: Document is 40 days old (threshold: 30d)

## Benchmark Truth Artifacts

stale_threshold=48.0h; drift_count=0
- [ok   ] tw-01-bounded-execution-v1: age=6.2h, coverage=complete
