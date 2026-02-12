# SICA + GraphRAG Rollout Runbook

Date: February 12, 2026
Scope: staged production-like rollout for Nomic SICA self-improvement and Knowledge Mound GraphRAG retrieval.

## 1) Current Baseline

### Test baseline (targeted hardening lanes)

```bash
python -m pytest tests/scripts/test_nomic_sica.py \
  tests/nomic/test_handlers.py \
  tests/nomic/test_sica_settings.py \
  tests/reasoning/test_claim_check.py \
  tests/verification/test_hilbert.py \
  tests/debate/test_stability_detector.py \
  tests/knowledge/test_lara_router.py \
  tests/nomic/phases/test_commit_phase.py \
  tests/nomic/phases/test_scope_limiter.py \
  tests/nomic/test_meta_planner.py \
  tests/debate/test_convergence_cache.py \
  -q -p no:randomly --tb=short
```

Result: `189 passed, 0 failed`.

### Benchmark baseline

```bash
python scripts/benchmarks/bench_lara_routing.py
python scripts/benchmarks/bench_adaptive_stopping.py
```

Observed baseline:
- LaRA routing benchmark (200 iterations):
  - mean latency: `0.0073 ms`
  - median latency: `0.0038 ms`
  - route distribution: `graph=66, rlm=73, semantic=60, keyword=1`
- Adaptive stability benchmark (200 iterations):
  - mean stability score: `0.3367`
  - mean latency: `0.0016 ms`

## 2) Feature Defaults and Guardrails

### GraphRAG defaults
- `enable_graph_rag = False`
- `graph_rag_max_hops = 2`
- `graph_rag_vector_top_k = 10`
- `graph_rag_vector_threshold = 0.5`
- `graph_rag_final_top_k = 20`

### SICA defaults
- `NOMIC_SICA_ENABLED = 0`
- `NOMIC_SICA_REQUIRE_APPROVAL = 1`
- `NOMIC_SICA_RUN_TESTS = 1`
- `NOMIC_SICA_RUN_TYPECHECK = 1`
- `NOMIC_SICA_RUN_LINT = 1`
- `NOMIC_SICA_MAX_OPPORTUNITIES = 5`
- `NOMIC_SICA_MAX_ROLLBACKS = 3`

## 3) Rollout Sequence (Best Order)

### Stage A: GraphRAG Canary (Read Path Only)

Enable GraphRAG for one canary workspace/process first:

```python
# MoundConfig overrides
enable_graph_rag = True
graph_rag_max_hops = 1
graph_rag_max_neighbors_per_hop = 3
graph_rag_vector_top_k = 8
graph_rag_vector_threshold = 0.55
graph_rag_graph_weight = 0.25
graph_rag_final_top_k = 10
graph_rag_enable_community_detection = False
```

Acceptance checks:
- No increase in query error rate.
- No material increase in query p95 latency (>20% vs baseline).
- Relevant retrieval remains stable for core smoke queries.

### Stage B: SICA Canary (Write Path, Strict Gate)

Enable SICA with hard safety defaults:

```bash
export NOMIC_SICA_ENABLED=1
export NOMIC_SICA_REQUIRE_APPROVAL=1
export NOMIC_SICA_RUN_TESTS=1
export NOMIC_SICA_RUN_TYPECHECK=1
export NOMIC_SICA_RUN_LINT=1
export NOMIC_SICA_MAX_OPPORTUNITIES=2
export NOMIC_SICA_MAX_ROLLBACKS=1
export NOMIC_SICA_GENERATOR_MODEL=codex
```

Acceptance checks:
- No unapproved patches are applied.
- Validation gates run for every applied patch.
- No increase in failed verify->recovery loops for canary pipelines.

### Stage C: Controlled Expansion

- Expand GraphRAG canary from 1 workspace to 10-25% traffic.
- Expand SICA to additional pipelines only after 48h clean canary.
- Keep human approval required until two consecutive clean windows.

## 4) Operational Checks

Run before and after each rollout stage:

```bash
python -m pytest tests/scripts/test_nomic_sica.py tests/nomic/test_handlers.py tests/nomic/test_sica_settings.py -q -p no:randomly
python -m pytest tests/knowledge/test_lara_router.py tests/debate/test_stability_detector.py -q -p no:randomly
python scripts/benchmarks/bench_lara_routing.py
python scripts/benchmarks/bench_adaptive_stopping.py
```

## 5) Metrics to Watch

Knowledge Mound metrics (`aragora/observability/metrics/km.py`):
- `aragora_km_lara_routing_total{route=*}`
- `aragora_km_operation_latency_seconds{operation=*}`
- `aragora_km_federated_queries_total{status=*}`

Debate metrics (`aragora/observability/metrics/debate.py`):
- `aragora_debate_early_termination_total{reason=*}`
- `aragora_debate_stability_score`
- `aragora_debate_round_latency_seconds`

SICA operational checks (logs/receipts):
- Count cycles with `status=success` vs `no_changes`.
- Count rollbacks per cycle.
- Track verify reruns after SICA and pass rate.

## 6) Rollback Plan

### GraphRAG rollback
- Set `enable_graph_rag=False`.
- Redeploy/restart the affected process.
- Confirm routing returns to semantic/keyword/long_context without GraphRAG branch.

### SICA rollback
- Set `NOMIC_SICA_ENABLED=0`.
- Keep `NOMIC_SICA_REQUIRE_APPROVAL=1` in all environments.
- Confirm verify handler no longer attempts SICA cycle.

## 7) Exit Criteria for General Availability

- Canary period complete with no sev-1/sev-2 incidents.
- Query/error/latency regressions within accepted thresholds.
- SICA patch validation pass rate stable and rollback rate low.
- Targeted hardening test lanes remain green.
