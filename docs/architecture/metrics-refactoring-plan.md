# Metrics Refactoring Plan

This document outlines the plan to refactor the metrics infrastructure in Aragora, specifically focusing on reducing the size of the main metrics files while maintaining backward compatibility.

## 1. Current State Analysis

### File Sizes (as of 2026-01-29)

| File | Lines of Code | Purpose |
|------|---------------|---------|
| `aragora/server/metrics.py` | 1,253 LOC | Prometheus-style metrics with custom Counter/Gauge/Histogram types, billing, API, security, business metrics, Knowledge Mound metrics |
| `aragora/observability/metrics.py` | 1,559 LOC | Prometheus client facade with lazy initialization, cross-functional metrics, Phase 9 metrics, feature metrics |

### Existing Submodules in `aragora/observability/metrics/`

The observability metrics package has already been partially refactored into submodules:

| Submodule | LOC | Purpose |
|-----------|-----|---------|
| `agent.py` | ~7.7k | Agent-specific metrics |
| `base.py` | ~3.6k | NoOpMetric and base utilities |
| `bridge.py` | ~12k | Cross-pollination bridge metrics |
| `checkpoint.py` | ~5.2k | Checkpoint store metrics |
| `consensus.py` | ~8.3k | Consensus ingestion metrics |
| `control_plane.py` | ~14.7k | Control plane metrics |
| `debate.py` | ~8.3k | Debate-specific metrics |
| `explainability.py` | ~5.7k | Batch explainability metrics |
| `fabric.py` | ~12.6k | Fabric metrics |
| `gauntlet.py` | ~10.9k | Gauntlet metrics |
| `governance.py` | ~6.3k | Governance store metrics |
| `km.py` | ~19.2k | Knowledge Mound metrics |
| `marketplace.py` | ~6.6k | Marketplace metrics |
| `notification.py` | ~5.4k | Notification metrics |
| `platform.py` | ~12.8k | Platform metrics |
| `request.py` | ~5k | Request metrics |
| `security.py` | ~21.5k | Security, encryption, RBAC metrics |
| `slo.py` | ~26.9k | SLO alerting metrics |
| `stores.py` | ~13.9k | Store operation metrics |
| `task_queue.py` | ~5.7k | Task queue metrics |
| `user_mapping.py` | ~4.4k | User ID mapping metrics |
| `webhook.py` | ~8.7k | Webhook delivery metrics |

### Metrics Groups Still in Main Files

**In `aragora/server/metrics.py`:**
- Core metric types (Counter, Gauge, Histogram, LabeledCounter, etc.)
- Billing metrics (SUBSCRIPTION_*, USAGE_*, BILLING_*, PAYMENT_*)
- API metrics (API_REQUESTS, API_LATENCY, ACTIVE_DEBATES, WEBSOCKET_CONNECTIONS)
- Security metrics (AUTH_FAILURES, RATE_LIMIT_HITS, SECURITY_VIOLATIONS)
- Business/Debate outcome metrics (DEBATES_TOTAL, CONSENSUS_*, DEBATE_*, AGENT_*)
- Vector store metrics (VECTOR_*)
- Knowledge Mound metrics (KNOWLEDGE_*)
- Helper functions (track_*, classify_*, generate_metrics)

**In `aragora/observability/metrics.py`:**
- Internal initialization functions (_init_debate_metrics_internal, _init_cache_metrics_internal, etc.)
- Cross-functional metrics (KNOWLEDGE_CACHE_*, MEMORY_COORDINATOR_*, etc.)
- Phase 9 metrics (RLM_CACHE_*, CALIBRATION_*, LEARNING_*, etc.)
- Slow debate metrics (SLOW_DEBATES_*, DEBATE_ROUND_LATENCY)
- Feature metrics (TTS_*, CONVERGENCE_*, EVIDENCE_CITATION_*, etc.)
- Gauntlet and workflow template metrics (GAUNTLET_*, WORKFLOW_TEMPLATES_*)
- Context managers (track_debate, track_phase, track_bridge_sync, etc.)
- Decorator utilities (measure_latency, measure_async_latency)

## 2. Proposed New Submodules

### 2.1 `aragora/observability/metrics/workflow.py`

**Metrics to move from `observability/metrics.py`:**
- `WORKFLOW_TRIGGERS`
- `WORKFLOW_TEMPLATES_CREATED`
- `WORKFLOW_TEMPLATE_EXECUTIONS`
- `WORKFLOW_TEMPLATE_EXECUTION_LATENCY`

**Functions to move:**
- `record_workflow_trigger()`
- `record_workflow_template_created()`
- `record_workflow_template_execution()`
- `track_workflow_template_execution()` context manager

**Estimated size:** ~150-200 LOC

### 2.2 `aragora/observability/metrics/tts.py`

**Metrics to move from `observability/metrics.py`:**
- `TTS_SYNTHESIS_TOTAL`
- `TTS_SYNTHESIS_LATENCY`

**Functions to move:**
- `record_tts_synthesis()`
- `record_tts_latency()`

**Estimated size:** ~80-100 LOC

### 2.3 `aragora/observability/metrics/cache.py`

**Metrics to move from `observability/metrics.py`:**
- `CACHE_HITS`
- `CACHE_MISSES`
- `KNOWLEDGE_CACHE_HITS`
- `KNOWLEDGE_CACHE_MISSES`
- `RLM_CACHE_HITS`
- `RLM_CACHE_MISSES`

**Functions to move:**
- `record_cache_hit()`
- `record_cache_miss()`
- `record_knowledge_cache_hit()`
- `record_knowledge_cache_miss()`
- `record_rlm_cache_hit()`
- `record_rlm_cache_miss()`

**Estimated size:** ~150-180 LOC

### 2.4 `aragora/observability/metrics/memory.py`

**Metrics to move from `observability/metrics.py`:**
- `MEMORY_OPERATIONS`
- `MEMORY_COORDINATOR_WRITES`
- `ADAPTIVE_ROUND_CHANGES`

**Functions to move:**
- `record_memory_operation()`
- `record_memory_coordinator_write()`
- `record_adaptive_round_change()`

**Estimated size:** ~100-120 LOC

### 2.5 `aragora/observability/metrics/ranking.py`

**Metrics to move from `observability/metrics.py`:**
- `CALIBRATION_ADJUSTMENTS`
- `LEARNING_BONUSES`
- `VOTING_ACCURACY_UPDATES`
- `SELECTION_FEEDBACK_ADJUSTMENTS`
- `NOVELTY_SCORE_CALCULATIONS`
- `NOVELTY_PENALTIES`
- `ECHO_CHAMBER_DETECTIONS`
- `RELATIONSHIP_BIAS_ADJUSTMENTS`
- `RLM_SELECTION_RECOMMENDATIONS`
- `CALIBRATION_COST_CALCULATIONS`
- `BUDGET_FILTERING_EVENTS`
- `PERFORMANCE_ROUTING_DECISIONS`
- `PERFORMANCE_ROUTING_LATENCY`
- `OUTCOME_COMPLEXITY_ADJUSTMENTS`
- `ANALYTICS_SELECTION_RECOMMENDATIONS`

**Functions to move:**
- `record_calibration_adjustment()`
- `record_learning_bonus()`
- `record_voting_accuracy_update()`
- `record_selection_feedback_adjustment()`

**Estimated size:** ~300-350 LOC

### 2.6 `aragora/observability/metrics/evidence.py`

**Metrics to move from `observability/metrics.py`:**
- `EVIDENCE_STORED`
- `EVIDENCE_CITATION_BONUSES`
- `CULTURE_PATTERNS`

**Functions to move:**
- `record_evidence_stored()`
- `record_evidence_citation_bonus()`
- `record_culture_patterns()`

**Estimated size:** ~100-120 LOC

### 2.7 `aragora/observability/metrics/convergence.py`

**Metrics to move from `observability/metrics.py`:**
- `CONVERGENCE_CHECKS_TOTAL`
- `PROCESS_EVALUATION_BONUSES`
- `RLM_READY_QUORUM_EVENTS`

**Functions to move:**
- `record_convergence_check()`
- `record_process_evaluation_bonus()`
- `record_rlm_ready_quorum()`

**Estimated size:** ~100-120 LOC

### 2.8 `aragora/server/metrics/types.py` (New package structure)

**Classes to extract from `server/metrics.py`:**
- `Counter`
- `LabeledCounter`
- `Gauge`
- `LabeledGauge`
- `Histogram`
- `LabeledHistogram`
- `get_percentile()`
- `get_percentiles()`

**Estimated size:** ~300 LOC

### 2.9 `aragora/server/metrics/vector.py`

**Metrics to move from `server/metrics.py`:**
- `VECTOR_OPERATIONS`
- `VECTOR_LATENCY`
- `VECTOR_RESULTS`
- `VECTOR_INDEX_BATCH_SIZE`

**Functions to move:**
- `track_vector_operation()` context manager
- `track_vector_search_results()`
- `track_vector_index_batch()`

**Estimated size:** ~150 LOC

### 2.10 `aragora/server/metrics/billing.py`

**Metrics to move from `server/metrics.py`:**
- `SUBSCRIPTION_EVENTS`
- `SUBSCRIPTION_ACTIVE`
- `USAGE_DEBATES`
- `USAGE_TOKENS`
- `BILLING_REVENUE`
- `PAYMENT_FAILURES`

**Functions to move:**
- `track_subscription_event()`
- `track_debate()`
- `track_tokens()`

**Estimated size:** ~100-120 LOC

## 3. Backward Compatibility Strategy

### 3.1 Re-export from Parent Modules

All metrics and functions moved to submodules must be re-exported from the parent module to maintain backward compatibility:

```python
# aragora/observability/metrics.py
from aragora.observability.metrics.workflow import (
    WORKFLOW_TRIGGERS,
    WORKFLOW_TEMPLATES_CREATED,
    record_workflow_trigger,
    record_workflow_template_created,
    track_workflow_template_execution,
)
```

### 3.2 Deprecation Warnings (Optional Phase)

For consumers who import directly from the parent module, consider adding deprecation warnings in a future phase:

```python
import warnings

def record_workflow_trigger(success: bool) -> None:
    warnings.warn(
        "Import from aragora.observability.metrics.workflow instead",
        DeprecationWarning,
        stacklevel=2,
    )
    from aragora.observability.metrics.workflow import record_workflow_trigger as _impl
    return _impl(success)
```

### 3.3 `__all__` Maintenance

Each submodule must define its own `__all__` list, and the parent module must aggregate these.

## 4. Implementation Sequence

### Phase 1: Low-Risk Extraction (Week 1)

1. **Create `aragora/observability/metrics/tts.py`** - Isolated metrics, no dependencies
2. **Create `aragora/observability/metrics/cache.py`** - Simple counter-only module
3. **Create `aragora/observability/metrics/convergence.py`** - Small, well-defined scope

**Testing:** Run full test suite after each extraction to catch import errors early.

### Phase 2: Medium Complexity (Week 2)

4. **Create `aragora/observability/metrics/workflow.py`** - Includes context manager
5. **Create `aragora/observability/metrics/memory.py`** - Cross-functional but stable
6. **Create `aragora/observability/metrics/evidence.py`** - Knowledge integration

### Phase 3: Complex Extraction (Week 3)

7. **Create `aragora/observability/metrics/ranking.py`** - Many related metrics
8. **Convert `aragora/server/metrics.py` to package** - Create `__init__.py`
9. **Create `aragora/server/metrics/types.py`** - Extract core metric types
10. **Create `aragora/server/metrics/vector.py`** - Vector store metrics
11. **Create `aragora/server/metrics/billing.py`** - Billing and subscription metrics

### Phase 4: Cleanup and Documentation (Week 4)

12. Update import statements in consuming modules (if direct submodule imports are preferred)
13. Update `__all__` exports in all modules
14. Add module-level docstrings with usage examples
15. Update `docs/OBSERVABILITY.md` with new module structure

## 5. Final Package Structure

```
aragora/
├── observability/
│   ├── metrics.py              # Facade (~800 LOC after refactoring)
│   └── metrics/
│       ├── __init__.py         # Re-exports and aggregation (~500 LOC)
│       ├── base.py             # NoOpMetric (existing)
│       ├── agent.py            # (existing)
│       ├── bridge.py           # (existing)
│       ├── cache.py            # NEW - cache hit/miss metrics
│       ├── checkpoint.py       # (existing)
│       ├── consensus.py        # (existing)
│       ├── control_plane.py    # (existing)
│       ├── convergence.py      # NEW - convergence detection metrics
│       ├── debate.py           # (existing)
│       ├── evidence.py         # NEW - evidence and culture metrics
│       ├── explainability.py   # (existing)
│       ├── fabric.py           # (existing)
│       ├── gauntlet.py         # (existing)
│       ├── governance.py       # (existing)
│       ├── km.py               # (existing)
│       ├── marketplace.py      # (existing)
│       ├── memory.py           # NEW - memory operation metrics
│       ├── notification.py     # (existing)
│       ├── platform.py         # (existing)
│       ├── ranking.py          # NEW - ELO, calibration, selection metrics
│       ├── request.py          # (existing)
│       ├── security.py         # (existing)
│       ├── slo.py              # (existing)
│       ├── stores.py           # (existing)
│       ├── task_queue.py       # (existing)
│       ├── tts.py              # NEW - TTS synthesis metrics
│       ├── user_mapping.py     # (existing)
│       ├── webhook.py          # (existing)
│       └── workflow.py         # NEW - workflow template metrics
└── server/
    ├── metrics/                # NEW package structure
    │   ├── __init__.py         # Re-exports from submodules + generate_metrics
    │   ├── types.py            # Counter, Gauge, Histogram classes
    │   ├── api.py              # API_REQUESTS, API_LATENCY, track_request
    │   ├── billing.py          # Subscription, usage, revenue metrics
    │   ├── security.py         # Auth failures, rate limits, violations
    │   ├── debate.py           # Debate outcome and consensus metrics
    │   ├── agent.py            # Agent requests, latency, tokens
    │   ├── vector.py           # Vector store operation metrics
    │   └── knowledge.py        # Knowledge Mound metrics from server
    └── metrics.py              # DEPRECATED - import redirect to metrics/
```

## 6. Success Criteria

| Metric | Target |
|--------|--------|
| `aragora/observability/metrics.py` | < 800 LOC |
| `aragora/server/metrics.py` (or `__init__.py`) | < 400 LOC |
| Test suite passes | 100% |
| No import errors in existing code | 0 errors |
| Each submodule has docstring | 100% |

## 7. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Circular imports | Medium | High | Careful dependency ordering; use TYPE_CHECKING |
| Missed re-export | Medium | Medium | Comprehensive `__all__` review; grep for imports |
| Test coverage gaps | Low | Medium | Run test suite after each extraction |
| Performance regression | Low | Low | Lazy imports already in place |

## 8. Open Questions

1. **Should we deprecate direct imports from parent module?**
   - Pro: Cleaner imports, better IDE support
   - Con: Breaking change for external consumers (if any)

2. **Should `aragora/server/metrics.py` be converted to a package?**
   - Pro: Consistent structure with observability
   - Con: More invasive change; less isolated metrics may not need it

3. **Should we unify `server/metrics` and `observability/metrics`?**
   - Consider consolidating into a single `aragora/metrics/` package
   - Requires careful analysis of current consumers

---

*Document created: 2026-01-29*
*Last updated: 2026-01-29*
