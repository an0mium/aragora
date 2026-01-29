"""
Prometheus metrics for Aragora.

Provides metrics for monitoring request rates, latencies, agent performance,
and debate statistics.

This module serves as the facade for the metrics subsystem. Most functionality
is delegated to specialized submodules in aragora/observability/metrics/.

Usage:
    from aragora.observability.metrics import record_request, record_agent_call

    # Record a request
    record_request("GET", "/api/debates", 200, 0.05)

    # Record an agent call
    record_agent_call("claude", success=True, latency=1.2)

Requirements:
    pip install prometheus-client

Environment Variables:
    METRICS_ENABLED: Set to "true" to enable metrics (default: true)
    METRICS_PORT: Port for /metrics endpoint (default: 9090)

See docs/OBSERVABILITY.md for configuration guide.
"""

from __future__ import annotations

import logging
import re
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator, TypeVar, cast

from aragora.observability.config import get_metrics_config
from aragora.observability.metrics.base import NoOpMetric

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# =============================================================================
# Core Metrics Infrastructure
# =============================================================================

# Prometheus metrics - initialized lazily
_initialized = False
_metrics_server = None

# Core metric instances (will be set during initialization)
REQUEST_COUNT: Any = None
REQUEST_LATENCY: Any = None
AGENT_CALLS: Any = None
AGENT_LATENCY: Any = None
ACTIVE_DEBATES: Any = None
CONSENSUS_RATE: Any = None
MEMORY_OPERATIONS: Any = None
WEBSOCKET_CONNECTIONS: Any = None
DEBATE_DURATION: Any = None
DEBATE_ROUNDS: Any = None
DEBATE_PHASE_DURATION: Any = None
AGENT_PARTICIPATION: Any = None
CACHE_HITS: Any = None
CACHE_MISSES: Any = None

# Cross-functional feature metrics
KNOWLEDGE_CACHE_HITS: Any = None
KNOWLEDGE_CACHE_MISSES: Any = None
MEMORY_COORDINATOR_WRITES: Any = None
SELECTION_FEEDBACK_ADJUSTMENTS: Any = None
WORKFLOW_TRIGGERS: Any = None
EVIDENCE_STORED: Any = None
CULTURE_PATTERNS: Any = None

# Phase 9 Cross-Pollination metrics
RLM_CACHE_HITS: Any = None
RLM_CACHE_MISSES: Any = None
CALIBRATION_ADJUSTMENTS: Any = None
LEARNING_BONUSES: Any = None
VOTING_ACCURACY_UPDATES: Any = None
ADAPTIVE_ROUND_CHANGES: Any = None

# Phase 9 Bridge metrics
BRIDGE_SYNCS: Any = None
BRIDGE_SYNC_LATENCY: Any = None
BRIDGE_ERRORS: Any = None
PERFORMANCE_ROUTING_DECISIONS: Any = None
PERFORMANCE_ROUTING_LATENCY: Any = None
OUTCOME_COMPLEXITY_ADJUSTMENTS: Any = None
ANALYTICS_SELECTION_RECOMMENDATIONS: Any = None
NOVELTY_SCORE_CALCULATIONS: Any = None
NOVELTY_PENALTIES: Any = None
ECHO_CHAMBER_DETECTIONS: Any = None
RELATIONSHIP_BIAS_ADJUSTMENTS: Any = None
RLM_SELECTION_RECOMMENDATIONS: Any = None
CALIBRATION_COST_CALCULATIONS: Any = None
BUDGET_FILTERING_EVENTS: Any = None

# Slow debate detection metrics
SLOW_DEBATES_TOTAL: Any = None
SLOW_ROUNDS_TOTAL: Any = None
DEBATE_ROUND_LATENCY: Any = None

# New feature metrics (TTS, convergence, vote bonuses)
TTS_SYNTHESIS_TOTAL: Any = None
TTS_SYNTHESIS_LATENCY: Any = None
CONVERGENCE_CHECKS_TOTAL: Any = None
EVIDENCE_CITATION_BONUSES: Any = None
PROCESS_EVALUATION_BONUSES: Any = None
RLM_READY_QUORUM_EVENTS: Any = None

# Gauntlet metrics
GAUNTLET_EXPORTS_TOTAL: Any = None
GAUNTLET_EXPORT_LATENCY: Any = None
GAUNTLET_EXPORT_SIZE: Any = None

# Workflow Template metrics
WORKFLOW_TEMPLATES_CREATED: Any = None
WORKFLOW_TEMPLATE_EXECUTIONS: Any = None
WORKFLOW_TEMPLATE_EXECUTION_LATENCY: Any = None

# =============================================================================
# Submodule Imports - Delegate to specialized metrics modules
# =============================================================================

# Task Queue metrics
from aragora.observability.metrics.task_queue import (  # noqa: E402
    init_task_queue_metrics,
    record_task_queue_operation,
    set_task_queue_size,
    record_task_queue_recovery,
    record_task_queue_cleanup,
    track_task_queue_operation,
)

# Governance Store metrics
from aragora.observability.metrics.governance import (  # noqa: E402
    init_governance_metrics,
    record_governance_decision,
    record_governance_verification,
    record_governance_approval,
    record_governance_store_latency,
    set_governance_artifacts_active,
    track_governance_store_operation,
)

# User ID Mapping metrics
from aragora.observability.metrics.user_mapping import (  # noqa: E402
    init_user_mapping_metrics,
    record_user_mapping_operation,
    record_user_mapping_cache_hit,
    record_user_mapping_cache_miss,
    set_user_mappings_active,
)

# Checkpoint Store metrics
from aragora.observability.metrics.checkpoint import (  # noqa: E402
    init_checkpoint_metrics,
    record_checkpoint_operation,
    record_checkpoint_restore_result,
    track_checkpoint_operation,
)

# Consensus Ingestion metrics
from aragora.observability.metrics.consensus import (  # noqa: E402
    init_consensus_metrics,
    init_enhanced_consensus_metrics,
    record_consensus_ingestion,
    record_consensus_dissent,
    record_consensus_evolution,
    record_consensus_evidence_linked,
    record_consensus_agreement_ratio,
)

# Notification metrics
from aragora.observability.metrics.notification import (  # noqa: E402
    init_notification_metrics,
    record_notification_sent,
    record_notification_error,
    set_notification_queue_size,
    track_notification_delivery,
)

# Marketplace metrics
from aragora.observability.metrics.marketplace import (  # noqa: E402
    init_marketplace_metrics,
    set_marketplace_templates_count,
    record_marketplace_download,
    record_marketplace_rating,
    record_marketplace_review,
    record_marketplace_operation_latency,
    track_marketplace_operation,
)

# Knowledge Mound metrics
from aragora.observability.metrics.km import (  # noqa: E402
    init_km_metrics,
    record_km_operation,
    record_km_cache_access,
    set_km_health_status,
    record_km_adapter_sync,
    record_km_federated_query,
    record_km_event_emitted,
    set_km_active_adapters,
    sync_km_metrics_to_prometheus,
)

# Batch Explainability metrics
from aragora.observability.metrics.explainability import (  # noqa: E402
    init_explainability_metrics,
    set_batch_explainability_jobs_active,
    record_batch_explainability_job,
    record_batch_explainability_debate,
    record_batch_explainability_error,
    track_batch_explainability_debate,
)

# Security metrics (encryption, RBAC, migration)
from aragora.observability.metrics.security import (  # noqa: E402
    init_security_metrics,
    record_encryption_operation as _record_encryption_op,
    record_encryption_error as _record_encryption_err,
    track_encryption_operation as _track_encryption_op,
    record_rbac_decision,
    track_rbac_evaluation as track_rbac_check,
    record_migration_record,
    record_migration_error,
)

# Webhook metrics
from aragora.observability.metrics.webhook import (  # noqa: E402
    record_webhook_delivery as _record_webhook_delivery_impl,
    record_webhook_retry as _record_webhook_retry_impl,
    set_queue_size as _set_webhook_queue_impl,
    WebhookDeliveryTimer,
)

# =============================================================================
# Explicit exports for wildcard import
# =============================================================================

__all__ = [
    # Server startup
    "start_metrics_server",
    "init_core_metrics",
    # Metric instances (globals)
    "REQUEST_COUNT",
    "REQUEST_LATENCY",
    "AGENT_CALLS",
    "AGENT_LATENCY",
    "ACTIVE_DEBATES",
    "CONSENSUS_RATE",
    "MEMORY_OPERATIONS",
    "WEBSOCKET_CONNECTIONS",
    "DEBATE_DURATION",
    "DEBATE_ROUNDS",
    "DEBATE_PHASE_DURATION",
    "AGENT_PARTICIPATION",
    "CACHE_HITS",
    "CACHE_MISSES",
    # Cross-functional feature metrics
    "KNOWLEDGE_CACHE_HITS",
    "KNOWLEDGE_CACHE_MISSES",
    "MEMORY_COORDINATOR_WRITES",
    "SELECTION_FEEDBACK_ADJUSTMENTS",
    "WORKFLOW_TRIGGERS",
    "EVIDENCE_STORED",
    "CULTURE_PATTERNS",
    # Phase 9 metrics
    "RLM_CACHE_HITS",
    "RLM_CACHE_MISSES",
    "CALIBRATION_ADJUSTMENTS",
    "LEARNING_BONUSES",
    "VOTING_ACCURACY_UPDATES",
    "ADAPTIVE_ROUND_CHANGES",
    # Bridge metrics
    "BRIDGE_SYNCS",
    "BRIDGE_SYNC_LATENCY",
    "BRIDGE_ERRORS",
    # Slow debate metrics
    "SLOW_DEBATES_TOTAL",
    "SLOW_ROUNDS_TOTAL",
    "DEBATE_ROUND_LATENCY",
    # Feature metrics
    "TTS_SYNTHESIS_TOTAL",
    "TTS_SYNTHESIS_LATENCY",
    "CONVERGENCE_CHECKS_TOTAL",
    "EVIDENCE_CITATION_BONUSES",
    "PROCESS_EVALUATION_BONUSES",
    "RLM_READY_QUORUM_EVENTS",
    # Core recording functions
    "record_request",
    "record_agent_call",
    "track_debate",
    "set_consensus_rate",
    "record_memory_operation",
    "track_websocket_connection",
    "measure_latency",
    "measure_async_latency",
    "record_debate_completion",
    "record_phase_duration",
    "record_agent_participation",
    "track_phase",
    "record_cache_hit",
    "record_cache_miss",
    # Cross-functional recording
    "record_knowledge_cache_hit",
    "record_knowledge_cache_miss",
    "record_memory_coordinator_write",
    "record_selection_feedback_adjustment",
    "record_workflow_trigger",
    "record_evidence_stored",
    "record_culture_patterns",
    # Phase 9 recording
    "record_rlm_cache_hit",
    "record_rlm_cache_miss",
    "record_calibration_adjustment",
    "record_learning_bonus",
    "record_voting_accuracy_update",
    "record_adaptive_round_change",
    # Bridge recording
    "record_bridge_sync",
    "record_bridge_sync_latency",
    "record_bridge_error",
    "track_bridge_sync",
    # Slow debate recording
    "record_slow_debate",
    "record_slow_round",
    "record_round_latency",
    # Feature recording
    "record_tts_synthesis",
    "record_tts_latency",
    "record_convergence_check",
    "record_evidence_citation_bonus",
    "record_process_evaluation_bonus",
    "record_rlm_ready_quorum",
    # Gauntlet recording
    "record_gauntlet_export",
    "track_gauntlet_export",
    # Workflow template recording
    "record_workflow_template_created",
    "record_workflow_template_execution",
    "track_workflow_template_execution",
    # Re-exported from submodules (notification)
    "record_notification_sent",
    "record_notification_error",
    "set_notification_queue_size",
    "track_notification_delivery",
    # Re-exported from submodules (marketplace)
    "set_marketplace_templates_count",
    "record_marketplace_download",
    "record_marketplace_rating",
    "record_marketplace_review",
    "record_marketplace_operation_latency",
    "track_marketplace_operation",
    # Re-exported from submodules (KM)
    "record_km_operation",
    "record_km_cache_access",
    "set_km_health_status",
    "record_km_adapter_sync",
    "record_km_federated_query",
    "record_km_event_emitted",
    "set_km_active_adapters",
    "sync_km_metrics_to_prometheus",
    "record_km_inbound_event",
    # Re-exported from submodules (explainability)
    "set_batch_explainability_jobs_active",
    "record_batch_explainability_job",
    "record_batch_explainability_debate",
    "record_batch_explainability_error",
    "track_batch_explainability_debate",
    # Re-exported from submodules (security)
    "record_encryption_operation",
    "record_encryption_error",
    "track_encryption_operation",
    "record_rbac_check",
    "track_rbac_check",
    "record_migration_record",
    "record_migration_error",
    # Re-exported from submodules (webhook)
    "record_webhook_delivery",
    "record_webhook_failure",
    "record_webhook_retry",
    "set_webhook_queue_size",
    "track_webhook_delivery",
    "set_webhook_circuit_breaker_state",
    # Re-exported from submodules (task_queue)
    "record_task_queue_operation",
    "set_task_queue_size",
    "record_task_queue_recovery",
    "record_task_queue_cleanup",
    "track_task_queue_operation",
    # Re-exported from submodules (governance)
    "record_governance_decision",
    "record_governance_verification",
    "record_governance_approval",
    "record_governance_store_latency",
    "set_governance_artifacts_active",
    "track_governance_store_operation",
    # Re-exported from submodules (user_mapping)
    "record_user_mapping_operation",
    "record_user_mapping_cache_hit",
    "record_user_mapping_cache_miss",
    "set_user_mappings_active",
    # Re-exported from submodules (checkpoint)
    "record_checkpoint_operation",
    "record_checkpoint_restore_result",
    "track_checkpoint_operation",
    # Re-exported from submodules (consensus)
    "record_consensus_ingestion",
    "record_consensus_dissent",
    "record_consensus_evolution",
    "record_consensus_evidence_linked",
    "record_consensus_agreement_ratio",
]


# =============================================================================
# Core Initialization
# =============================================================================


def _init_metrics() -> bool:
    """Initialize Prometheus metrics lazily."""
    global _initialized
    global REQUEST_COUNT, REQUEST_LATENCY, AGENT_CALLS, AGENT_LATENCY
    global ACTIVE_DEBATES, CONSENSUS_RATE, MEMORY_OPERATIONS, WEBSOCKET_CONNECTIONS

    if _initialized:
        return True

    config = get_metrics_config()
    if not config.enabled:
        _init_noop_metrics()
        _initialized = True
        return False

    try:
        from prometheus_client import Counter, Gauge, Histogram

        # Request metrics
        REQUEST_COUNT = Counter(
            "aragora_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
        )

        REQUEST_LATENCY = Histogram(
            "aragora_request_latency_seconds",
            "HTTP request latency in seconds",
            ["endpoint"],
            buckets=config.histogram_buckets,
        )

        # Agent metrics
        AGENT_CALLS = Counter(
            "aragora_agent_calls_total",
            "Total agent API calls",
            ["agent", "status"],
        )

        AGENT_LATENCY = Histogram(
            "aragora_agent_latency_seconds",
            "Agent API call latency in seconds",
            ["agent"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
        )

        # Debate metrics
        ACTIVE_DEBATES = Gauge(
            "aragora_active_debates",
            "Number of currently active debates",
        )

        CONSENSUS_RATE = Gauge(
            "aragora_consensus_rate",
            "Rate of debates reaching consensus (0-1)",
        )

        # Memory metrics
        MEMORY_OPERATIONS = Counter(
            "aragora_memory_operations_total",
            "Total memory operations",
            ["operation", "tier"],
        )

        # WebSocket metrics
        WEBSOCKET_CONNECTIONS = Gauge(
            "aragora_websocket_connections",
            "Number of active WebSocket connections",
        )

        # Initialize internal core metrics (not in submodules)
        _init_debate_metrics_internal()
        _init_cache_metrics_internal()
        _init_cross_functional_metrics_internal()
        _init_phase9_metrics_internal()
        _init_slow_debate_metrics_internal()
        _init_feature_metrics_internal()
        _init_gauntlet_metrics_internal()
        _init_workflow_metrics_internal()

        # Initialize submodule metrics
        init_task_queue_metrics()
        init_governance_metrics()
        init_user_mapping_metrics()
        init_checkpoint_metrics()
        init_consensus_metrics()
        init_enhanced_consensus_metrics()
        init_notification_metrics()
        init_marketplace_metrics()
        init_km_metrics()
        init_explainability_metrics()
        init_security_metrics()

        _initialized = True
        logger.info("Prometheus metrics initialized")
        return True

    except ImportError as e:
        logger.warning(
            f"prometheus-client not installed, metrics disabled: {e}. "
            "Install with: pip install prometheus-client"
        )
        _init_noop_metrics()
        _initialized = True
        return False
    except Exception as e:
        logger.error(f"Failed to initialize metrics: {e}")
        _init_noop_metrics()
        _initialized = True
        return False


def _init_debate_metrics_internal() -> None:
    """Initialize debate-specific metrics."""
    global DEBATE_DURATION, DEBATE_ROUNDS, DEBATE_PHASE_DURATION, AGENT_PARTICIPATION
    from prometheus_client import Counter, Histogram

    DEBATE_DURATION = Histogram(
        "aragora_debate_duration_seconds",
        "Debate duration in seconds",
        ["outcome"],
        buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1200],
    )

    DEBATE_ROUNDS = Histogram(
        "aragora_debate_rounds_total",
        "Number of rounds per debate",
        ["outcome"],
        buckets=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )

    DEBATE_PHASE_DURATION = Histogram(
        "aragora_debate_phase_duration_seconds",
        "Duration of each debate phase",
        ["phase"],
        buckets=[0.5, 1, 2, 5, 10, 30, 60],
    )

    AGENT_PARTICIPATION = Counter(
        "aragora_agent_participation_total",
        "Agent participation in debates",
        ["agent", "phase"],
    )


def _init_cache_metrics_internal() -> None:
    """Initialize cache metrics."""
    global CACHE_HITS, CACHE_MISSES
    from prometheus_client import Counter

    CACHE_HITS = Counter(
        "aragora_cache_hits_total",
        "Cache hit count",
        ["cache_name"],
    )

    CACHE_MISSES = Counter(
        "aragora_cache_misses_total",
        "Cache miss count",
        ["cache_name"],
    )


def _init_cross_functional_metrics_internal() -> None:
    """Initialize cross-functional feature metrics."""
    global KNOWLEDGE_CACHE_HITS, KNOWLEDGE_CACHE_MISSES
    global MEMORY_COORDINATOR_WRITES, SELECTION_FEEDBACK_ADJUSTMENTS
    global WORKFLOW_TRIGGERS, EVIDENCE_STORED, CULTURE_PATTERNS
    from prometheus_client import Counter

    KNOWLEDGE_CACHE_HITS = Counter(
        "aragora_knowledge_cache_hits_total",
        "Knowledge query cache hits",
    )

    KNOWLEDGE_CACHE_MISSES = Counter(
        "aragora_knowledge_cache_misses_total",
        "Knowledge query cache misses",
    )

    MEMORY_COORDINATOR_WRITES = Counter(
        "aragora_memory_coordinator_writes_total",
        "Atomic memory coordinator writes",
        ["status"],
    )

    SELECTION_FEEDBACK_ADJUSTMENTS = Counter(
        "aragora_selection_feedback_adjustments_total",
        "Agent selection weight adjustments",
        ["agent", "direction"],
    )

    WORKFLOW_TRIGGERS = Counter(
        "aragora_workflow_triggers_total",
        "Post-debate workflow triggers",
        ["status"],
    )

    EVIDENCE_STORED = Counter(
        "aragora_evidence_stored_total",
        "Evidence items stored in knowledge mound",
    )

    CULTURE_PATTERNS = Counter(
        "aragora_culture_patterns_total",
        "Culture patterns extracted from debates",
    )


def _init_phase9_metrics_internal() -> None:
    """Initialize Phase 9 Cross-Pollination metrics."""
    global RLM_CACHE_HITS, RLM_CACHE_MISSES
    global CALIBRATION_ADJUSTMENTS, LEARNING_BONUSES
    global VOTING_ACCURACY_UPDATES, ADAPTIVE_ROUND_CHANGES
    global BRIDGE_SYNCS, BRIDGE_SYNC_LATENCY, BRIDGE_ERRORS
    global PERFORMANCE_ROUTING_DECISIONS, PERFORMANCE_ROUTING_LATENCY
    global OUTCOME_COMPLEXITY_ADJUSTMENTS, ANALYTICS_SELECTION_RECOMMENDATIONS
    global NOVELTY_SCORE_CALCULATIONS, NOVELTY_PENALTIES
    global ECHO_CHAMBER_DETECTIONS, RELATIONSHIP_BIAS_ADJUSTMENTS
    global RLM_SELECTION_RECOMMENDATIONS, CALIBRATION_COST_CALCULATIONS
    global BUDGET_FILTERING_EVENTS
    from prometheus_client import Counter, Histogram

    RLM_CACHE_HITS = Counter("aragora_rlm_cache_hits_total", "RLM compression cache hits")
    RLM_CACHE_MISSES = Counter("aragora_rlm_cache_misses_total", "RLM compression cache misses")

    CALIBRATION_ADJUSTMENTS = Counter(
        "aragora_calibration_adjustments_total",
        "Proposal confidence calibrations applied",
        ["agent"],
    )

    LEARNING_BONUSES = Counter(
        "aragora_learning_bonuses_total",
        "Learning efficiency ELO bonuses applied",
        ["agent", "category"],
    )

    VOTING_ACCURACY_UPDATES = Counter(
        "aragora_voting_accuracy_updates_total",
        "Voting accuracy records updated",
        ["result"],
    )

    ADAPTIVE_ROUND_CHANGES = Counter(
        "aragora_adaptive_round_changes_total",
        "Debate round count adjustments from memory strategy",
        ["direction"],
    )

    BRIDGE_SYNCS = Counter(
        "aragora_bridge_syncs_total",
        "Cross-pollination bridge sync operations",
        ["bridge", "status"],
    )

    BRIDGE_SYNC_LATENCY = Histogram(
        "aragora_bridge_sync_latency_seconds",
        "Bridge sync operation latency",
        ["bridge"],
        buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    )

    BRIDGE_ERRORS = Counter(
        "aragora_bridge_errors_total",
        "Cross-pollination bridge errors",
        ["bridge", "error_type"],
    )

    PERFORMANCE_ROUTING_DECISIONS = Counter(
        "aragora_performance_routing_decisions_total",
        "Performance-based routing decisions",
        ["task_type", "selected_agent"],
    )

    PERFORMANCE_ROUTING_LATENCY = Histogram(
        "aragora_performance_routing_latency_seconds",
        "Time to compute routing decision",
        buckets=[0.001, 0.005, 0.01, 0.05, 0.1],
    )

    OUTCOME_COMPLEXITY_ADJUSTMENTS = Counter(
        "aragora_outcome_complexity_adjustments_total",
        "Complexity budget adjustments from outcome patterns",
        ["direction"],
    )

    ANALYTICS_SELECTION_RECOMMENDATIONS = Counter(
        "aragora_analytics_selection_recommendations_total",
        "Analytics-driven team selection recommendations",
        ["recommendation_type"],
    )

    NOVELTY_SCORE_CALCULATIONS = Counter(
        "aragora_novelty_score_calculations_total",
        "Novelty score calculations performed",
        ["agent"],
    )

    NOVELTY_PENALTIES = Counter(
        "aragora_novelty_penalties_total",
        "Selection penalties for low novelty",
        ["agent"],
    )

    ECHO_CHAMBER_DETECTIONS = Counter(
        "aragora_echo_chamber_detections_total",
        "Echo chamber risk detections in team composition",
        ["risk_level"],
    )

    RELATIONSHIP_BIAS_ADJUSTMENTS = Counter(
        "aragora_relationship_bias_adjustments_total",
        "Voting weight adjustments for alliance bias",
        ["agent", "direction"],
    )

    RLM_SELECTION_RECOMMENDATIONS = Counter(
        "aragora_rlm_selection_recommendations_total",
        "RLM-efficient agent selection recommendations",
        ["agent"],
    )

    CALIBRATION_COST_CALCULATIONS = Counter(
        "aragora_calibration_cost_calculations_total",
        "Cost efficiency calculations with calibration",
        ["agent", "efficiency"],
    )

    BUDGET_FILTERING_EVENTS = Counter(
        "aragora_budget_filtering_events_total",
        "Agent filtering events due to budget constraints",
        ["outcome"],
    )


def _init_slow_debate_metrics_internal() -> None:
    """Initialize slow debate detection metrics."""
    global SLOW_DEBATES_TOTAL, SLOW_ROUNDS_TOTAL, DEBATE_ROUND_LATENCY
    from prometheus_client import Counter, Histogram

    SLOW_DEBATES_TOTAL = Counter(
        "aragora_slow_debates_total",
        "Number of debates flagged as slow (>30s per round)",
    )

    SLOW_ROUNDS_TOTAL = Counter(
        "aragora_slow_rounds_total",
        "Number of individual rounds flagged as slow",
        ["debate_outcome"],
    )

    DEBATE_ROUND_LATENCY = Histogram(
        "aragora_debate_round_latency_seconds",
        "Latency per debate round",
        buckets=[1, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300],
    )


def _init_feature_metrics_internal() -> None:
    """Initialize new feature metrics (TTS, convergence, vote bonuses)."""
    global TTS_SYNTHESIS_TOTAL, TTS_SYNTHESIS_LATENCY
    global CONVERGENCE_CHECKS_TOTAL, EVIDENCE_CITATION_BONUSES
    global PROCESS_EVALUATION_BONUSES, RLM_READY_QUORUM_EVENTS
    from prometheus_client import Counter, Histogram

    TTS_SYNTHESIS_TOTAL = Counter(
        "aragora_tts_synthesis_total",
        "Total TTS synthesis operations",
        ["voice", "platform"],
    )

    TTS_SYNTHESIS_LATENCY = Histogram(
        "aragora_tts_synthesis_latency_seconds",
        "TTS synthesis latency in seconds",
        buckets=[0.1, 0.5, 1, 2, 5, 10, 20],
    )

    CONVERGENCE_CHECKS_TOTAL = Counter(
        "aragora_convergence_checks_total",
        "Total convergence check events",
        ["status", "blocked"],
    )

    EVIDENCE_CITATION_BONUSES = Counter(
        "aragora_evidence_citation_bonuses_total",
        "Evidence citation vote bonuses applied",
        ["agent"],
    )

    PROCESS_EVALUATION_BONUSES = Counter(
        "aragora_process_evaluation_bonuses_total",
        "Process evaluation vote bonuses applied",
        ["agent"],
    )

    RLM_READY_QUORUM_EVENTS = Counter(
        "aragora_rlm_ready_quorum_total",
        "RLM ready signal quorum events",
    )


def _init_gauntlet_metrics_internal() -> None:
    """Initialize gauntlet export metrics."""
    global GAUNTLET_EXPORTS_TOTAL, GAUNTLET_EXPORT_LATENCY, GAUNTLET_EXPORT_SIZE
    from prometheus_client import Counter, Histogram

    GAUNTLET_EXPORTS_TOTAL = Counter(
        "aragora_gauntlet_exports_total",
        "Total Gauntlet exports",
        ["format", "type", "status"],
    )

    GAUNTLET_EXPORT_LATENCY = Histogram(
        "aragora_gauntlet_export_latency_seconds",
        "Gauntlet export latency",
        ["format", "type"],
        buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
    )

    GAUNTLET_EXPORT_SIZE = Histogram(
        "aragora_gauntlet_export_size_bytes",
        "Gauntlet export size",
        ["format", "type"],
        buckets=[1000, 10000, 100000, 1000000, 10000000],
    )


def _init_workflow_metrics_internal() -> None:
    """Initialize workflow template metrics."""
    global WORKFLOW_TEMPLATES_CREATED, WORKFLOW_TEMPLATE_EXECUTIONS
    global WORKFLOW_TEMPLATE_EXECUTION_LATENCY
    from prometheus_client import Counter, Histogram

    WORKFLOW_TEMPLATES_CREATED = Counter(
        "aragora_workflow_templates_created_total",
        "Workflow templates created",
        ["pattern", "template_id"],
    )

    WORKFLOW_TEMPLATE_EXECUTIONS = Counter(
        "aragora_workflow_template_executions_total",
        "Workflow template executions",
        ["pattern", "status"],
    )

    WORKFLOW_TEMPLATE_EXECUTION_LATENCY = Histogram(
        "aragora_workflow_template_execution_latency_seconds",
        "Workflow template execution latency",
        ["pattern"],
        buckets=[0.5, 1, 2.5, 5, 10, 30, 60, 120, 300],
    )


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global REQUEST_COUNT, REQUEST_LATENCY, AGENT_CALLS, AGENT_LATENCY
    global ACTIVE_DEBATES, CONSENSUS_RATE, MEMORY_OPERATIONS, WEBSOCKET_CONNECTIONS
    global DEBATE_DURATION, DEBATE_ROUNDS, DEBATE_PHASE_DURATION, AGENT_PARTICIPATION
    global CACHE_HITS, CACHE_MISSES
    global KNOWLEDGE_CACHE_HITS, KNOWLEDGE_CACHE_MISSES
    global MEMORY_COORDINATOR_WRITES, SELECTION_FEEDBACK_ADJUSTMENTS
    global WORKFLOW_TRIGGERS, EVIDENCE_STORED, CULTURE_PATTERNS
    global RLM_CACHE_HITS, RLM_CACHE_MISSES
    global CALIBRATION_ADJUSTMENTS, LEARNING_BONUSES
    global VOTING_ACCURACY_UPDATES, ADAPTIVE_ROUND_CHANGES
    global BRIDGE_SYNCS, BRIDGE_SYNC_LATENCY, BRIDGE_ERRORS
    global PERFORMANCE_ROUTING_DECISIONS, PERFORMANCE_ROUTING_LATENCY
    global OUTCOME_COMPLEXITY_ADJUSTMENTS, ANALYTICS_SELECTION_RECOMMENDATIONS
    global NOVELTY_SCORE_CALCULATIONS, NOVELTY_PENALTIES
    global ECHO_CHAMBER_DETECTIONS, RELATIONSHIP_BIAS_ADJUSTMENTS
    global RLM_SELECTION_RECOMMENDATIONS, CALIBRATION_COST_CALCULATIONS
    global BUDGET_FILTERING_EVENTS
    global SLOW_DEBATES_TOTAL, SLOW_ROUNDS_TOTAL, DEBATE_ROUND_LATENCY
    global TTS_SYNTHESIS_TOTAL, TTS_SYNTHESIS_LATENCY
    global CONVERGENCE_CHECKS_TOTAL, EVIDENCE_CITATION_BONUSES
    global PROCESS_EVALUATION_BONUSES, RLM_READY_QUORUM_EVENTS
    global GAUNTLET_EXPORTS_TOTAL, GAUNTLET_EXPORT_LATENCY, GAUNTLET_EXPORT_SIZE
    global WORKFLOW_TEMPLATES_CREATED, WORKFLOW_TEMPLATE_EXECUTIONS
    global WORKFLOW_TEMPLATE_EXECUTION_LATENCY

    noop = NoOpMetric()

    # Core metrics
    REQUEST_COUNT = noop
    REQUEST_LATENCY = noop
    AGENT_CALLS = noop
    AGENT_LATENCY = noop
    ACTIVE_DEBATES = noop
    CONSENSUS_RATE = noop
    MEMORY_OPERATIONS = noop
    WEBSOCKET_CONNECTIONS = noop

    # Debate metrics
    DEBATE_DURATION = noop
    DEBATE_ROUNDS = noop
    DEBATE_PHASE_DURATION = noop
    AGENT_PARTICIPATION = noop

    # Cache metrics
    CACHE_HITS = noop
    CACHE_MISSES = noop

    # Cross-functional metrics
    KNOWLEDGE_CACHE_HITS = noop
    KNOWLEDGE_CACHE_MISSES = noop
    MEMORY_COORDINATOR_WRITES = noop
    SELECTION_FEEDBACK_ADJUSTMENTS = noop
    WORKFLOW_TRIGGERS = noop
    EVIDENCE_STORED = noop
    CULTURE_PATTERNS = noop

    # Phase 9 metrics
    RLM_CACHE_HITS = noop
    RLM_CACHE_MISSES = noop
    CALIBRATION_ADJUSTMENTS = noop
    LEARNING_BONUSES = noop
    VOTING_ACCURACY_UPDATES = noop
    ADAPTIVE_ROUND_CHANGES = noop
    BRIDGE_SYNCS = noop
    BRIDGE_SYNC_LATENCY = noop
    BRIDGE_ERRORS = noop
    PERFORMANCE_ROUTING_DECISIONS = noop
    PERFORMANCE_ROUTING_LATENCY = noop
    OUTCOME_COMPLEXITY_ADJUSTMENTS = noop
    ANALYTICS_SELECTION_RECOMMENDATIONS = noop
    NOVELTY_SCORE_CALCULATIONS = noop
    NOVELTY_PENALTIES = noop
    ECHO_CHAMBER_DETECTIONS = noop
    RELATIONSHIP_BIAS_ADJUSTMENTS = noop
    RLM_SELECTION_RECOMMENDATIONS = noop
    CALIBRATION_COST_CALCULATIONS = noop
    BUDGET_FILTERING_EVENTS = noop

    # Slow debate metrics
    SLOW_DEBATES_TOTAL = noop
    SLOW_ROUNDS_TOTAL = noop
    DEBATE_ROUND_LATENCY = noop

    # Feature metrics
    TTS_SYNTHESIS_TOTAL = noop
    TTS_SYNTHESIS_LATENCY = noop
    CONVERGENCE_CHECKS_TOTAL = noop
    EVIDENCE_CITATION_BONUSES = noop
    PROCESS_EVALUATION_BONUSES = noop
    RLM_READY_QUORUM_EVENTS = noop

    # Gauntlet metrics
    GAUNTLET_EXPORTS_TOTAL = noop
    GAUNTLET_EXPORT_LATENCY = noop
    GAUNTLET_EXPORT_SIZE = noop

    # Workflow template metrics
    WORKFLOW_TEMPLATES_CREATED = noop
    WORKFLOW_TEMPLATE_EXECUTIONS = noop
    WORKFLOW_TEMPLATE_EXECUTION_LATENCY = noop


def init_core_metrics() -> bool:
    """Initialize core metrics (public API)."""
    return _init_metrics()


# =============================================================================
# Server and Endpoint Utilities
# =============================================================================


def start_metrics_server(port: int = 9090) -> bool:
    """Start a metrics server on the specified port."""
    global _metrics_server

    if not _init_metrics():
        logger.warning("Metrics not enabled, server not started")
        return False

    if _metrics_server is not None:
        logger.warning("Metrics server already running")
        return True

    try:
        from prometheus_client import start_http_server

        start_http_server(port)
        _metrics_server = port
        logger.info(f"Metrics server started on port {port}")
        return True
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")
        return False


# Endpoint normalization regex
_ENDPOINT_PATTERN = re.compile(
    r"(/api/v\d+)?/(debates|agents|workspaces|tenants|users|knowledge|receipts|memory)"
    r"/[a-zA-Z0-9_-]+(?:/|$)"
)


def _normalize_endpoint(endpoint: str) -> str:
    """Normalize endpoint paths to reduce cardinality."""
    match = _ENDPOINT_PATTERN.match(endpoint)
    if match:
        prefix = match.group(1) or ""
        resource = match.group(2)
        return f"{prefix}/{resource}/:id"
    return endpoint


# =============================================================================
# Core Recording Functions
# =============================================================================


def record_request(method: str, endpoint: str, status: int, latency_seconds: float) -> None:
    """Record an HTTP request."""
    _init_metrics()
    normalized = _normalize_endpoint(endpoint)
    REQUEST_COUNT.labels(method=method, endpoint=normalized, status=str(status)).inc()
    REQUEST_LATENCY.labels(endpoint=normalized).observe(latency_seconds)


def record_agent_call(agent: str, success: bool, latency: float) -> None:
    """Record an agent API call."""
    _init_metrics()
    status = "success" if success else "error"
    AGENT_CALLS.labels(agent=agent, status=status).inc()
    AGENT_LATENCY.labels(agent=agent).observe(latency)


def set_consensus_rate(rate: float) -> None:
    """Set the consensus rate (0-1)."""
    _init_metrics()
    CONSENSUS_RATE.set(rate)


def record_memory_operation(operation: str, tier: str) -> None:
    """Record a memory operation."""
    _init_metrics()
    MEMORY_OPERATIONS.labels(operation=operation, tier=tier).inc()


def record_debate_completion(outcome: str, duration_seconds: float, rounds: int) -> None:
    """Record debate completion."""
    _init_metrics()
    DEBATE_DURATION.labels(outcome=outcome).observe(duration_seconds)
    DEBATE_ROUNDS.labels(outcome=outcome).observe(rounds)


def record_phase_duration(phase: str, duration_seconds: float) -> None:
    """Record debate phase duration."""
    _init_metrics()
    DEBATE_PHASE_DURATION.labels(phase=phase).observe(duration_seconds)


def record_agent_participation(agent: str, phase: str) -> None:
    """Record agent participation in a debate phase."""
    _init_metrics()
    AGENT_PARTICIPATION.labels(agent=agent, phase=phase).inc()


def record_cache_hit(cache_name: str) -> None:
    """Record a cache hit."""
    _init_metrics()
    CACHE_HITS.labels(cache_name=cache_name).inc()


def record_cache_miss(cache_name: str) -> None:
    """Record a cache miss."""
    _init_metrics()
    CACHE_MISSES.labels(cache_name=cache_name).inc()


# =============================================================================
# Cross-Functional Recording Functions
# =============================================================================


def record_knowledge_cache_hit() -> None:
    """Record a knowledge query cache hit."""
    _init_metrics()
    KNOWLEDGE_CACHE_HITS.inc()


def record_knowledge_cache_miss() -> None:
    """Record a knowledge query cache miss."""
    _init_metrics()
    KNOWLEDGE_CACHE_MISSES.inc()


def record_memory_coordinator_write(success: bool) -> None:
    """Record a memory coordinator write."""
    _init_metrics()
    status = "success" if success else "error"
    MEMORY_COORDINATOR_WRITES.labels(status=status).inc()


def record_selection_feedback_adjustment(agent: str, direction: str) -> None:
    """Record a selection feedback adjustment."""
    _init_metrics()
    SELECTION_FEEDBACK_ADJUSTMENTS.labels(agent=agent, direction=direction).inc()


def record_workflow_trigger(success: bool) -> None:
    """Record a post-debate workflow trigger."""
    _init_metrics()
    status = "success" if success else "error"
    WORKFLOW_TRIGGERS.labels(status=status).inc()


def record_evidence_stored(count: int = 1) -> None:
    """Record an evidence item stored."""
    _init_metrics()
    EVIDENCE_STORED.inc(count)


def record_culture_patterns(count: int = 1) -> None:
    """Record a culture pattern extraction."""
    _init_metrics()
    CULTURE_PATTERNS.inc(count)


# =============================================================================
# Phase 9 Recording Functions
# =============================================================================


def record_rlm_cache_hit() -> None:
    """Record an RLM cache hit."""
    _init_metrics()
    RLM_CACHE_HITS.inc()


def record_rlm_cache_miss() -> None:
    """Record an RLM cache miss."""
    _init_metrics()
    RLM_CACHE_MISSES.inc()


def record_calibration_adjustment(agent: str) -> None:
    """Record a calibration adjustment."""
    _init_metrics()
    CALIBRATION_ADJUSTMENTS.labels(agent=agent).inc()


def record_learning_bonus(agent: str, category: str) -> None:
    """Record a learning efficiency bonus."""
    _init_metrics()
    LEARNING_BONUSES.labels(agent=agent, category=category).inc()


def record_voting_accuracy_update(result: str) -> None:
    """Record a voting accuracy update."""
    _init_metrics()
    VOTING_ACCURACY_UPDATES.labels(result=result).inc()


def record_adaptive_round_change(direction: str) -> None:
    """Record an adaptive round change."""
    _init_metrics()
    ADAPTIVE_ROUND_CHANGES.labels(direction=direction).inc()


def record_bridge_sync(bridge: str, success: bool) -> None:
    """Record a bridge sync operation."""
    _init_metrics()
    status = "success" if success else "error"
    BRIDGE_SYNCS.labels(bridge=bridge, status=status).inc()


def record_bridge_sync_latency(bridge: str, latency_seconds: float) -> None:
    """Record bridge sync latency."""
    _init_metrics()
    BRIDGE_SYNC_LATENCY.labels(bridge=bridge).observe(latency_seconds)


def record_bridge_error(bridge: str, error_type: str) -> None:
    """Record a bridge error."""
    _init_metrics()
    BRIDGE_ERRORS.labels(bridge=bridge, error_type=error_type).inc()


@contextmanager
def track_bridge_sync(bridge: str) -> Generator[None, None, None]:
    """Context manager to track bridge sync operations."""
    _init_metrics()
    start = time.perf_counter()
    success = True
    try:
        yield
    except Exception as e:
        logger.debug(f"Bridge sync failed for {bridge}: {e}")
        success = False
        raise
    finally:
        latency = time.perf_counter() - start
        record_bridge_sync(bridge, success)
        record_bridge_sync_latency(bridge, latency)


# =============================================================================
# Slow Debate Recording Functions
# =============================================================================


def record_slow_debate() -> None:
    """Record a debate flagged as slow."""
    _init_metrics()
    SLOW_DEBATES_TOTAL.inc()


def record_slow_round(debate_outcome: str = "in_progress") -> None:
    """Record a round flagged as slow."""
    _init_metrics()
    SLOW_ROUNDS_TOTAL.labels(debate_outcome=debate_outcome).inc()


def record_round_latency(latency_seconds: float) -> None:
    """Record latency for a debate round."""
    _init_metrics()
    DEBATE_ROUND_LATENCY.observe(latency_seconds)


# =============================================================================
# Feature Recording Functions (TTS, Convergence, Vote Bonuses)
# =============================================================================


def record_tts_synthesis(voice: str, platform: str = "unknown") -> None:
    """Record a TTS synthesis operation."""
    _init_metrics()
    TTS_SYNTHESIS_TOTAL.labels(voice=voice, platform=platform).inc()


def record_tts_latency(latency_seconds: float) -> None:
    """Record TTS synthesis latency."""
    _init_metrics()
    TTS_SYNTHESIS_LATENCY.observe(latency_seconds)


def record_convergence_check(status: str, blocked: bool = False) -> None:
    """Record a convergence check event."""
    _init_metrics()
    CONVERGENCE_CHECKS_TOTAL.labels(status=status, blocked=str(blocked)).inc()


def record_evidence_citation_bonus(agent: str) -> None:
    """Record an evidence citation vote bonus."""
    _init_metrics()
    EVIDENCE_CITATION_BONUSES.labels(agent=agent).inc()


def record_process_evaluation_bonus(agent: str) -> None:
    """Record a process evaluation vote bonus."""
    _init_metrics()
    PROCESS_EVALUATION_BONUSES.labels(agent=agent).inc()


def record_rlm_ready_quorum() -> None:
    """Record an RLM ready signal quorum event."""
    _init_metrics()
    RLM_READY_QUORUM_EVENTS.inc()


# =============================================================================
# Gauntlet Recording Functions
# =============================================================================


def record_gauntlet_export(
    format: str,
    export_type: str,
    success: bool,
    latency_seconds: float,
    size_bytes: int = 0,
) -> None:
    """Record a Gauntlet export operation."""
    _init_metrics()
    status = "success" if success else "error"
    GAUNTLET_EXPORTS_TOTAL.labels(format=format, type=export_type, status=status).inc()
    GAUNTLET_EXPORT_LATENCY.labels(format=format, type=export_type).observe(latency_seconds)
    if size_bytes > 0:
        GAUNTLET_EXPORT_SIZE.labels(format=format, type=export_type).observe(size_bytes)


@contextmanager
def track_gauntlet_export(format: str, export_type: str) -> Generator[dict, None, None]:
    """Context manager to track Gauntlet export operations."""
    _init_metrics()
    start = time.perf_counter()
    ctx: dict[str, Any] = {"size_bytes": 0}
    success = True
    try:
        yield ctx
    except Exception:
        success = False
        raise
    finally:
        latency = time.perf_counter() - start
        record_gauntlet_export(format, export_type, success, latency, ctx.get("size_bytes", 0))


# =============================================================================
# Workflow Template Recording Functions
# =============================================================================


def record_workflow_template_created(pattern: str, template_id: str) -> None:
    """Record a workflow template creation."""
    _init_metrics()
    WORKFLOW_TEMPLATES_CREATED.labels(pattern=pattern, template_id=template_id).inc()


def record_workflow_template_execution(
    pattern: str,
    success: bool,
    latency_seconds: float,
) -> None:
    """Record a workflow template execution."""
    _init_metrics()
    status = "success" if success else "error"
    WORKFLOW_TEMPLATE_EXECUTIONS.labels(pattern=pattern, status=status).inc()
    WORKFLOW_TEMPLATE_EXECUTION_LATENCY.labels(pattern=pattern).observe(latency_seconds)


@contextmanager
def track_workflow_template_execution(pattern: str) -> Generator[None, None, None]:
    """Context manager to track workflow template execution."""
    _init_metrics()
    start = time.perf_counter()
    success = True
    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        latency = time.perf_counter() - start
        record_workflow_template_execution(pattern, success, latency)


# =============================================================================
# KM Inbound Event (wrapper for convenience)
# =============================================================================


def record_km_inbound_event(event_type: str, source: str, success: bool = True) -> None:
    """Record an inbound event to Knowledge Mound."""
    record_km_operation("ingest", success, 0.0)
    record_km_event_emitted(f"inbound_{event_type}")


# =============================================================================
# Wrapper Functions for Webhook Metrics (compatibility layer)
# =============================================================================

# Webhook metrics - globals for compatibility
WEBHOOK_DELIVERIES_TOTAL: Any = None
WEBHOOK_DELIVERY_LATENCY: Any = None
WEBHOOK_FAILURES_BY_ENDPOINT: Any = None
WEBHOOK_RETRIES_TOTAL: Any = None
WEBHOOK_CIRCUIT_BREAKER_STATES: Any = None
WEBHOOK_QUEUE_SIZE: Any = None


def record_webhook_delivery(endpoint: str, success: bool, latency_seconds: float) -> None:
    """Record a webhook delivery attempt."""
    _record_webhook_delivery_impl(
        event_type=endpoint,
        success=success,
        duration_seconds=latency_seconds,
    )


def record_webhook_failure(endpoint: str, error_type: str) -> None:
    """Record a webhook delivery failure."""
    _record_webhook_delivery_impl(
        event_type=endpoint,
        success=False,
        duration_seconds=0.0,
        status_code=500 if error_type == "connection_error" else 400,
    )


def record_webhook_retry(endpoint: str, attempt: int) -> None:
    """Record a webhook delivery retry."""
    _record_webhook_retry_impl(event_type=endpoint, attempt=attempt)


def set_webhook_circuit_breaker_state(endpoint: str, state: str) -> None:
    """Set the circuit breaker state for a webhook endpoint."""
    # No-op for now - webhook module doesn't have this metric
    pass


def set_webhook_queue_size(size: int) -> None:
    """Set the current webhook delivery queue size."""
    _set_webhook_queue_impl(size)


@contextmanager
def track_webhook_delivery(endpoint: str) -> Generator[None, None, None]:
    """Context manager to track webhook delivery."""
    with WebhookDeliveryTimer(endpoint):
        yield


# =============================================================================
# Wrapper Functions for Security Metrics (compatibility layer)
# =============================================================================


def record_encryption_operation(operation: str, store: str, latency_seconds: float) -> None:
    """Record an encryption or decryption operation."""
    _record_encryption_op(operation=operation, success=True, latency_seconds=latency_seconds)


def record_encryption_error(operation: str, error_type: str) -> None:
    """Record an encryption or decryption error."""
    _record_encryption_err(operation=operation, error_type=error_type)


@contextmanager
def track_encryption_operation(operation: str, store: str) -> Generator[None, None, None]:
    """Context manager to track encryption/decryption operations."""
    with _track_encryption_op(operation):
        yield


def record_rbac_check(permission: str, allowed: bool, handler: str = "") -> None:
    """Record an RBAC permission check."""
    record_rbac_decision(permission=permission, granted=allowed)


# =============================================================================
# Context Managers
# =============================================================================


@contextmanager
def track_debate(outcome: str = "unknown") -> Generator[dict, None, None]:
    """Context manager to track debate execution."""
    _init_metrics()
    ACTIVE_DEBATES.inc()
    start = time.perf_counter()
    ctx: dict[str, Any] = {"outcome": outcome, "rounds": 0}
    try:
        yield ctx
    finally:
        ACTIVE_DEBATES.dec()
        duration = time.perf_counter() - start
        record_debate_completion(ctx.get("outcome", outcome), duration, ctx.get("rounds", 0))


@contextmanager
def track_phase(phase: str) -> Generator[None, None, None]:
    """Context manager to track debate phase duration."""
    _init_metrics()
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        record_phase_duration(phase, duration)


def track_websocket_connection(increment: bool = True) -> None:
    """Track WebSocket connections (increment or decrement).

    Args:
        increment: True to increment, False to decrement
    """
    _init_metrics()
    if increment:
        WEBSOCKET_CONNECTIONS.inc()
    else:
        WEBSOCKET_CONNECTIONS.dec()


def measure_latency(endpoint: str) -> Callable[[F], F]:
    """Decorator factory to measure function latency.

    Args:
        endpoint: The endpoint name for labeling

    Example:
        @measure_latency("my_endpoint")
        def my_func(x):
            return x * 2
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _init_metrics()
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                latency = time.perf_counter() - start
                REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)

        return cast(F, wrapper)

    return decorator


def measure_async_latency(endpoint: str) -> Callable[[F], F]:
    """Decorator factory to measure async function latency.

    Args:
        endpoint: The endpoint name for labeling

    Example:
        @measure_async_latency("my_endpoint")
        async def my_func(x):
            return x * 2
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            _init_metrics()
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                latency = time.perf_counter() - start
                REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)

        return cast(F, wrapper)

    return decorator
