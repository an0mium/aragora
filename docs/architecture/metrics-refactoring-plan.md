# Metrics Refactoring Plan

This document outlines the plan to refactor the metrics infrastructure in Aragora, specifically focusing on reducing the size of the main metrics files while maintaining backward compatibility.

## 1. Current State Analysis

### File Sizes (as of 2026-01-29)

| File | Lines of Code | Purpose |
|------|---------------|---------|
| `aragora/server/metrics.py` | 1,253 LOC | Prometheus-style metrics with custom Counter/Gauge/Histogram types, billing, API, security, business metrics, Knowledge Mound metrics |
| `aragora/observability/metrics.py` | 1,558 LOC | Prometheus client facade with lazy initialization, cross-functional metrics, Phase 9 metrics, feature metrics |

### Existing Submodules in `aragora/observability/metrics/`

The observability metrics package has already been partially refactored into submodules:

| Submodule | LOC | Purpose |
|-----------|-----|---------|
| `__init__.py` | 285 | Re-exports and aggregation |
| `agent.py` | 286 | Agent-specific metrics |
| `base.py` | 133 | NoOpMetric and base utilities |
| `bridge.py` | 397 | Cross-pollination bridge metrics |
| `checkpoint.py` | 180 | Checkpoint store metrics |
| `consensus.py` | 267 | Consensus ingestion metrics |
| `control_plane.py` | 455 | Control plane metrics |
| `debate.py` | 319 | Debate-specific metrics |
| `explainability.py` | 187 | Batch explainability metrics |
| `fabric.py` | 435 | Fabric metrics |
| `gauntlet.py` | 317 | Gauntlet metrics |
| `governance.py` | 205 | Governance store metrics |
| `km.py` | 583 | Knowledge Mound metrics |
| `marketplace.py` | 227 | Marketplace metrics |
| `notification.py` | 190 | Notification metrics |
| `platform.py` | 402 | Platform metrics |
| `request.py` | 195 | Request metrics |
| `security.py` | 673 | Security, encryption, RBAC metrics |
| `slo.py` | 853 | SLO alerting metrics |
| `stores.py` | 406 | Store operation metrics |
| `task_queue.py` | 200 | Task queue metrics |
| `user_mapping.py` | 155 | User ID mapping metrics |
| `webhook.py` | 300 | Webhook delivery metrics |

**Total submodule LOC:** 7,650

### Metrics Groups Still in Main Files

**In `aragora/server/metrics.py` (1,253 LOC):**
- Core metric types (Counter, Gauge, Histogram, LabeledCounter, LabeledGauge, LabeledHistogram)
- Percentile helpers (get_percentile, get_percentiles)
- Billing metrics (SUBSCRIPTION_*, USAGE_*, BILLING_*, PAYMENT_*)
- API metrics (API_REQUESTS, API_LATENCY, ACTIVE_DEBATES, WEBSOCKET_CONNECTIONS)
- Security metrics (AUTH_FAILURES, RATE_LIMIT_HITS, SECURITY_VIOLATIONS)
- Business/Debate outcome metrics (DEBATES_TOTAL, CONSENSUS_*, DEBATE_*, AGENT_*)
- Vector store metrics (VECTOR_OPERATIONS, VECTOR_LATENCY, VECTOR_RESULTS, VECTOR_INDEX_BATCH_SIZE)
- Knowledge Mound metrics (KNOWLEDGE_VISIBILITY_*, KNOWLEDGE_SHARES, KNOWLEDGE_FEDERATION_*)
- Helper functions (track_*, classify_*, generate_metrics)

**In `aragora/observability/metrics.py` (1,558 LOC):**
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

## 3. Metrics and Functions to Move from metrics.py

### From `aragora/observability/metrics.py`

| Metric/Function | Target Submodule | Type |
|-----------------|------------------|------|
| `WORKFLOW_TRIGGERS` | `workflow.py` | Counter |
| `WORKFLOW_TEMPLATES_CREATED` | `workflow.py` | Counter |
| `WORKFLOW_TEMPLATE_EXECUTIONS` | `workflow.py` | Counter |
| `WORKFLOW_TEMPLATE_EXECUTION_LATENCY` | `workflow.py` | Histogram |
| `record_workflow_trigger()` | `workflow.py` | Function |
| `record_workflow_template_created()` | `workflow.py` | Function |
| `record_workflow_template_execution()` | `workflow.py` | Function |
| `track_workflow_template_execution()` | `workflow.py` | Context Manager |
| `TTS_SYNTHESIS_TOTAL` | `tts.py` | Counter |
| `TTS_SYNTHESIS_LATENCY` | `tts.py` | Histogram |
| `record_tts_synthesis()` | `tts.py` | Function |
| `record_tts_latency()` | `tts.py` | Function |
| `CACHE_HITS` | `cache.py` | Counter |
| `CACHE_MISSES` | `cache.py` | Counter |
| `KNOWLEDGE_CACHE_HITS` | `cache.py` | Counter |
| `KNOWLEDGE_CACHE_MISSES` | `cache.py` | Counter |
| `RLM_CACHE_HITS` | `cache.py` | Counter |
| `RLM_CACHE_MISSES` | `cache.py` | Counter |
| `record_cache_hit()` | `cache.py` | Function |
| `record_cache_miss()` | `cache.py` | Function |
| `record_knowledge_cache_hit()` | `cache.py` | Function |
| `record_knowledge_cache_miss()` | `cache.py` | Function |
| `record_rlm_cache_hit()` | `cache.py` | Function |
| `record_rlm_cache_miss()` | `cache.py` | Function |
| `MEMORY_OPERATIONS` | `memory.py` | Counter |
| `MEMORY_COORDINATOR_WRITES` | `memory.py` | Counter |
| `ADAPTIVE_ROUND_CHANGES` | `memory.py` | Counter |
| `record_memory_operation()` | `memory.py` | Function |
| `record_memory_coordinator_write()` | `memory.py` | Function |
| `record_adaptive_round_change()` | `memory.py` | Function |
| `CALIBRATION_ADJUSTMENTS` | `ranking.py` | Counter |
| `LEARNING_BONUSES` | `ranking.py` | Counter |
| `VOTING_ACCURACY_UPDATES` | `ranking.py` | Counter |
| `SELECTION_FEEDBACK_ADJUSTMENTS` | `ranking.py` | Counter |
| `record_calibration_adjustment()` | `ranking.py` | Function |
| `record_learning_bonus()` | `ranking.py` | Function |
| `record_voting_accuracy_update()` | `ranking.py` | Function |
| `record_selection_feedback_adjustment()` | `ranking.py` | Function |
| `EVIDENCE_STORED` | `evidence.py` | Counter |
| `EVIDENCE_CITATION_BONUSES` | `evidence.py` | Counter |
| `CULTURE_PATTERNS` | `evidence.py` | Counter |
| `record_evidence_stored()` | `evidence.py` | Function |
| `record_evidence_citation_bonus()` | `evidence.py` | Function |
| `record_culture_patterns()` | `evidence.py` | Function |
| `CONVERGENCE_CHECKS_TOTAL` | `convergence.py` | Counter |
| `PROCESS_EVALUATION_BONUSES` | `convergence.py` | Counter |
| `RLM_READY_QUORUM_EVENTS` | `convergence.py` | Counter |
| `record_convergence_check()` | `convergence.py` | Function |
| `record_process_evaluation_bonus()` | `convergence.py` | Function |
| `record_rlm_ready_quorum()` | `convergence.py` | Function |

### From `aragora/server/metrics.py`

| Metric/Function | Target Submodule | Type |
|-----------------|------------------|------|
| `Counter` class | `types.py` | Class |
| `LabeledCounter` class | `types.py` | Class |
| `Gauge` class | `types.py` | Class |
| `LabeledGauge` class | `types.py` | Class |
| `Histogram` class | `types.py` | Class |
| `LabeledHistogram` class | `types.py` | Class |
| `get_percentile()` | `types.py` | Function |
| `get_percentiles()` | `types.py` | Function |
| `VECTOR_OPERATIONS` | `vector.py` | Counter |
| `VECTOR_LATENCY` | `vector.py` | Histogram |
| `VECTOR_RESULTS` | `vector.py` | Histogram |
| `VECTOR_INDEX_BATCH_SIZE` | `vector.py` | Histogram |
| `track_vector_operation()` | `vector.py` | Context Manager |
| `track_vector_search_results()` | `vector.py` | Function |
| `track_vector_index_batch()` | `vector.py` | Function |
| `SUBSCRIPTION_EVENTS` | `billing.py` | Counter |
| `SUBSCRIPTION_ACTIVE` | `billing.py` | Gauge |
| `USAGE_DEBATES` | `billing.py` | Counter |
| `USAGE_TOKENS` | `billing.py` | Counter |
| `BILLING_REVENUE` | `billing.py` | Counter |
| `PAYMENT_FAILURES` | `billing.py` | Counter |
| `track_subscription_event()` | `billing.py` | Function |
| `track_debate()` | `billing.py` | Function |
| `track_tokens()` | `billing.py` | Function |

## 4. Backward Compatibility Strategy

### 4.1 Re-export from Parent Modules

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

### 4.2 Deprecation Warnings (Optional Phase)

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

### 4.3 `__all__` Maintenance

Each submodule must define its own `__all__` list, and the parent module must aggregate these:

```python
# aragora/observability/metrics/workflow.py
__all__ = [
    "WORKFLOW_TRIGGERS",
    "WORKFLOW_TEMPLATES_CREATED",
    "WORKFLOW_TEMPLATE_EXECUTIONS",
    "WORKFLOW_TEMPLATE_EXECUTION_LATENCY",
    "record_workflow_trigger",
    "record_workflow_template_created",
    "record_workflow_template_execution",
    "track_workflow_template_execution",
]

# aragora/observability/metrics/__init__.py
from aragora.observability.metrics.workflow import *
# ... other submodules
```

## 5. Implementation Sequence

### Phase 1: Low-Risk Extraction (Week 1)

1. **Create `aragora/observability/metrics/tts.py`** - Isolated metrics, no dependencies
2. **Create `aragora/observability/metrics/cache.py`** - Simple counter-only module
3. **Create `aragora/observability/metrics/convergence.py`** - Small, well-defined scope

**Testing:** Run full test suite after each extraction to catch import errors early.

```bash
# After each extraction:
pytest tests/observability/ -v
pytest tests/ -k "metrics" -v
```

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

## 6. Final Package Structure

```
aragora/
├── observability/
│   ├── metrics.py              # Facade (~800 LOC after refactoring)
│   └── metrics/
│       ├── __init__.py         # Re-exports and aggregation (~500 LOC)
│       ├── base.py             # NoOpMetric (existing, 133 LOC)
│       ├── agent.py            # (existing, 286 LOC)
│       ├── bridge.py           # (existing, 397 LOC)
│       ├── cache.py            # NEW - cache hit/miss metrics (~150 LOC)
│       ├── checkpoint.py       # (existing, 180 LOC)
│       ├── consensus.py        # (existing, 267 LOC)
│       ├── control_plane.py    # (existing, 455 LOC)
│       ├── convergence.py      # NEW - convergence detection metrics (~100 LOC)
│       ├── debate.py           # (existing, 319 LOC)
│       ├── evidence.py         # NEW - evidence and culture metrics (~100 LOC)
│       ├── explainability.py   # (existing, 187 LOC)
│       ├── fabric.py           # (existing, 435 LOC)
│       ├── gauntlet.py         # (existing, 317 LOC)
│       ├── governance.py       # (existing, 205 LOC)
│       ├── km.py               # (existing, 583 LOC)
│       ├── marketplace.py      # (existing, 227 LOC)
│       ├── memory.py           # NEW - memory operation metrics (~100 LOC)
│       ├── notification.py     # (existing, 190 LOC)
│       ├── platform.py         # (existing, 402 LOC)
│       ├── ranking.py          # NEW - ELO, calibration, selection metrics (~350 LOC)
│       ├── request.py          # (existing, 195 LOC)
│       ├── security.py         # (existing, 673 LOC)
│       ├── slo.py              # (existing, 853 LOC)
│       ├── stores.py           # (existing, 406 LOC)
│       ├── task_queue.py       # (existing, 200 LOC)
│       ├── tts.py              # NEW - TTS synthesis metrics (~80 LOC)
│       ├── user_mapping.py     # (existing, 155 LOC)
│       ├── webhook.py          # (existing, 300 LOC)
│       └── workflow.py         # NEW - workflow template metrics (~150 LOC)
└── server/
    ├── metrics/                # NEW package structure
    │   ├── __init__.py         # Re-exports from submodules + generate_metrics (~400 LOC)
    │   ├── types.py            # Counter, Gauge, Histogram classes (~300 LOC)
    │   ├── api.py              # API_REQUESTS, API_LATENCY, track_request (~150 LOC)
    │   ├── billing.py          # Subscription, usage, revenue metrics (~120 LOC)
    │   ├── security.py         # Auth failures, rate limits, violations (~100 LOC)
    │   ├── debate.py           # Debate outcome and consensus metrics (~200 LOC)
    │   ├── agent.py            # Agent requests, latency, tokens (~100 LOC)
    │   ├── vector.py           # Vector store operation metrics (~150 LOC)
    │   └── knowledge.py        # Knowledge Mound metrics from server (~200 LOC)
    └── metrics.py              # DEPRECATED - import redirect to metrics/
```

## 7. Success Criteria

| Metric | Target |
|--------|--------|
| `aragora/observability/metrics.py` | < 800 LOC |
| `aragora/server/metrics.py` (or `__init__.py`) | < 400 LOC |
| Test suite passes | 100% |
| No import errors in existing code | 0 errors |
| Each submodule has docstring | 100% |
| Each submodule has `__all__` | 100% |

## 8. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Circular imports | Medium | High | Careful dependency ordering; use TYPE_CHECKING |
| Missed re-export | Medium | Medium | Comprehensive `__all__` review; grep for imports |
| Test coverage gaps | Low | Medium | Run test suite after each extraction |
| Performance regression | Low | Low | Lazy imports already in place |
| External consumers break | Low | High | Keep re-exports in parent module; version bump |

## 9. Open Questions

1. **Should we deprecate direct imports from parent module?**
   - Pro: Cleaner imports, better IDE support
   - Con: Breaking change for external consumers (if any)
   - Recommendation: Keep re-exports indefinitely, add optional deprecation warnings later

2. **Should `aragora/server/metrics.py` be converted to a package?**
   - Pro: Consistent structure with observability
   - Con: More invasive change; less isolated metrics may not need it
   - Recommendation: Yes, for consistency and maintainability

3. **Should we unify `server/metrics` and `observability/metrics`?**
   - Consider consolidating into a single `aragora/metrics/` package
   - Requires careful analysis of current consumers
   - Recommendation: Defer to future refactoring; keep separate for now

4. **Should we add type hints to all metric functions?**
   - Pro: Better IDE support, catches errors early
   - Con: Some additional effort
   - Recommendation: Yes, add during extraction

---

*Document created: 2026-01-29*
*Last updated: 2026-01-29*
