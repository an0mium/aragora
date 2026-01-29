"""
Prometheus metrics for Aragora.

Provides metrics for monitoring request rates, latencies, agent performance,
and debate statistics.

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

This module serves as the facade for the metrics subsystem. Most functionality
is delegated to specialized submodules in aragora/observability/metrics/.
"""

from __future__ import annotations

import logging
import re
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator, Optional, TypeVar, cast

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

# Knowledge Mound metrics
KM_OPERATIONS_TOTAL: Any = None
KM_OPERATION_LATENCY: Any = None
KM_CACHE_HITS_TOTAL: Any = None
KM_CACHE_MISSES_TOTAL: Any = None
KM_HEALTH_STATUS: Any = None
KM_ADAPTER_SYNCS_TOTAL: Any = None
KM_FEDERATED_QUERIES_TOTAL: Any = None
KM_EVENTS_EMITTED_TOTAL: Any = None
KM_ACTIVE_ADAPTERS: Any = None

# Notification delivery metrics
NOTIFICATION_SENT_TOTAL: Any = None
NOTIFICATION_LATENCY: Any = None
NOTIFICATION_ERRORS_TOTAL: Any = None
NOTIFICATION_QUEUE_SIZE: Any = None

# Marketplace metrics
MARKETPLACE_TEMPLATES_TOTAL: Any = None
MARKETPLACE_DOWNLOADS_TOTAL: Any = None
MARKETPLACE_RATINGS_TOTAL: Any = None
MARKETPLACE_RATINGS_DISTRIBUTION: Any = None
MARKETPLACE_REVIEWS_TOTAL: Any = None
MARKETPLACE_OPERATION_LATENCY: Any = None

# Batch Explainability metrics
BATCH_EXPLAINABILITY_JOBS_ACTIVE: Any = None
BATCH_EXPLAINABILITY_JOBS_TOTAL: Any = None
BATCH_EXPLAINABILITY_DEBATES_PROCESSED: Any = None
BATCH_EXPLAINABILITY_PROCESSING_LATENCY: Any = None
BATCH_EXPLAINABILITY_ERRORS_TOTAL: Any = None

# Webhook Delivery metrics
WEBHOOK_DELIVERIES_TOTAL: Any = None
WEBHOOK_DELIVERY_LATENCY: Any = None
WEBHOOK_FAILURES_BY_ENDPOINT: Any = None
WEBHOOK_RETRIES_TOTAL: Any = None
WEBHOOK_CIRCUIT_BREAKER_STATES: Any = None
WEBHOOK_QUEUE_SIZE: Any = None

# Security & Governance Hardening metrics
ENCRYPTION_OPERATIONS_TOTAL: Any = None
ENCRYPTION_OPERATION_LATENCY: Any = None
ENCRYPTION_ERRORS_TOTAL: Any = None
RBAC_PERMISSION_CHECKS_TOTAL: Any = None
RBAC_PERMISSION_DENIED_TOTAL: Any = None
RBAC_CHECK_LATENCY: Any = None
MIGRATION_RECORDS_TOTAL: Any = None
MIGRATION_ERRORS_TOTAL: Any = None

# Gauntlet metrics
GAUNTLET_EXPORTS_TOTAL: Any = None
GAUNTLET_EXPORT_LATENCY: Any = None
GAUNTLET_EXPORT_SIZE: Any = None

# Workflow Template metrics
WORKFLOW_TEMPLATES_CREATED: Any = None
WORKFLOW_TEMPLATE_EXECUTIONS: Any = None
WORKFLOW_TEMPLATE_EXECUTION_LATENCY: Any = None

# =============================================================================
# Import metrics from submodules (delegating to avoid duplication)
# =============================================================================

# Task Queue metrics - delegated to submodule
from aragora.observability.metrics.task_queue import (  # noqa: E402
    TASK_QUEUE_OPERATIONS_TOTAL,
    TASK_QUEUE_OPERATION_LATENCY,
    TASK_QUEUE_SIZE,
    TASK_QUEUE_RECOVERED_TOTAL,
    TASK_QUEUE_CLEANUP_TOTAL,
    init_task_queue_metrics,
    record_task_queue_operation,
    set_task_queue_size,
    record_task_queue_recovery,
    record_task_queue_cleanup,
    track_task_queue_operation,
)

# Governance Store metrics - delegated to submodule
from aragora.observability.metrics.governance import (  # noqa: E402
    GOVERNANCE_DECISIONS_TOTAL,
    GOVERNANCE_VERIFICATIONS_TOTAL,
    GOVERNANCE_APPROVALS_TOTAL,
    GOVERNANCE_STORE_LATENCY,
    GOVERNANCE_ARTIFACTS_ACTIVE,
    init_governance_metrics,
    record_governance_decision,
    record_governance_verification,
    record_governance_approval,
    record_governance_store_latency,
    set_governance_artifacts_active,
    track_governance_store_operation,
)

# User ID Mapping metrics - delegated to submodule
from aragora.observability.metrics.user_mapping import (  # noqa: E402
    USER_MAPPING_OPERATIONS_TOTAL,
    USER_MAPPING_CACHE_HITS_TOTAL,
    USER_MAPPING_CACHE_MISSES_TOTAL,
    USER_MAPPINGS_ACTIVE,
    init_user_mapping_metrics,
    record_user_mapping_operation,
    record_user_mapping_cache_hit,
    record_user_mapping_cache_miss,
    set_user_mappings_active,
)

# Checkpoint Store metrics - delegated to submodule
from aragora.observability.metrics.checkpoint import (  # noqa: E402
    CHECKPOINT_OPERATIONS,
    CHECKPOINT_OPERATION_LATENCY,
    CHECKPOINT_SIZE,
    CHECKPOINT_RESTORE_RESULTS,
    init_checkpoint_metrics,
    record_checkpoint_operation,
    record_checkpoint_restore_result,
    track_checkpoint_operation,
)

# Consensus Ingestion metrics - delegated to submodule
from aragora.observability.metrics.consensus import (  # noqa: E402
    CONSENSUS_INGESTION_TOTAL,
    CONSENSUS_INGESTION_LATENCY,
    CONSENSUS_INGESTION_CLAIMS,
    CONSENSUS_DISSENT_INGESTED,
    CONSENSUS_EVOLUTION_TRACKED,
    CONSENSUS_EVIDENCE_LINKED,
    CONSENSUS_AGREEMENT_RATIO,
    init_consensus_metrics,
    init_enhanced_consensus_metrics,
    record_consensus_ingestion,
    record_consensus_dissent,
    record_consensus_evolution,
    record_consensus_evidence_linked,
    record_consensus_agreement_ratio,
)


# =============================================================================
# Explicit exports for wildcard import
# =============================================================================

__all__ = [
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
    "PERFORMANCE_ROUTING_DECISIONS",
    "PERFORMANCE_ROUTING_LATENCY",
    "OUTCOME_COMPLEXITY_ADJUSTMENTS",
    "ANALYTICS_SELECTION_RECOMMENDATIONS",
    "NOVELTY_SCORE_CALCULATIONS",
    "NOVELTY_PENALTIES",
    "ECHO_CHAMBER_DETECTIONS",
    "RELATIONSHIP_BIAS_ADJUSTMENTS",
    "RLM_SELECTION_RECOMMENDATIONS",
    "CALIBRATION_COST_CALCULATIONS",
    "BUDGET_FILTERING_EVENTS",
    # Slow debate detection
    "SLOW_DEBATES_TOTAL",
    "SLOW_ROUNDS_TOTAL",
    "DEBATE_ROUND_LATENCY",
    # Feature metrics (TTS, convergence, votes)
    "TTS_SYNTHESIS_TOTAL",
    "TTS_SYNTHESIS_LATENCY",
    "CONVERGENCE_CHECKS_TOTAL",
    "EVIDENCE_CITATION_BONUSES",
    "PROCESS_EVALUATION_BONUSES",
    "RLM_READY_QUORUM_EVENTS",
    # Knowledge Mound
    "KM_OPERATIONS_TOTAL",
    "KM_OPERATION_LATENCY",
    "KM_CACHE_HITS_TOTAL",
    "KM_CACHE_MISSES_TOTAL",
    "KM_HEALTH_STATUS",
    "KM_ADAPTER_SYNCS_TOTAL",
    "KM_FEDERATED_QUERIES_TOTAL",
    "KM_EVENTS_EMITTED_TOTAL",
    "KM_ACTIVE_ADAPTERS",
    # Notification
    "NOTIFICATION_SENT_TOTAL",
    "NOTIFICATION_LATENCY",
    "NOTIFICATION_ERRORS_TOTAL",
    "NOTIFICATION_QUEUE_SIZE",
    # Task Queue (from submodule)
    "TASK_QUEUE_OPERATIONS_TOTAL",
    "TASK_QUEUE_OPERATION_LATENCY",
    "TASK_QUEUE_SIZE",
    "TASK_QUEUE_RECOVERED_TOTAL",
    "TASK_QUEUE_CLEANUP_TOTAL",
    # Governance (from submodule)
    "GOVERNANCE_DECISIONS_TOTAL",
    "GOVERNANCE_VERIFICATIONS_TOTAL",
    "GOVERNANCE_APPROVALS_TOTAL",
    "GOVERNANCE_STORE_LATENCY",
    "GOVERNANCE_ARTIFACTS_ACTIVE",
    # User mapping (from submodule)
    "USER_MAPPING_OPERATIONS_TOTAL",
    "USER_MAPPING_CACHE_HITS_TOTAL",
    "USER_MAPPING_CACHE_MISSES_TOTAL",
    "USER_MAPPINGS_ACTIVE",
    # Checkpoint (from submodule)
    "CHECKPOINT_OPERATIONS",
    "CHECKPOINT_OPERATION_LATENCY",
    "CHECKPOINT_SIZE",
    "CHECKPOINT_RESTORE_RESULTS",
    # Consensus (from submodule)
    "CONSENSUS_INGESTION_TOTAL",
    "CONSENSUS_INGESTION_LATENCY",
    "CONSENSUS_INGESTION_CLAIMS",
    "CONSENSUS_DISSENT_INGESTED",
    "CONSENSUS_EVOLUTION_TRACKED",
    "CONSENSUS_EVIDENCE_LINKED",
    "CONSENSUS_AGREEMENT_RATIO",
    # Marketplace
    "MARKETPLACE_TEMPLATES_TOTAL",
    "MARKETPLACE_DOWNLOADS_TOTAL",
    "MARKETPLACE_RATINGS_TOTAL",
    "MARKETPLACE_RATINGS_DISTRIBUTION",
    "MARKETPLACE_REVIEWS_TOTAL",
    "MARKETPLACE_OPERATION_LATENCY",
    # Batch Explainability
    "BATCH_EXPLAINABILITY_JOBS_ACTIVE",
    "BATCH_EXPLAINABILITY_JOBS_TOTAL",
    "BATCH_EXPLAINABILITY_DEBATES_PROCESSED",
    "BATCH_EXPLAINABILITY_PROCESSING_LATENCY",
    "BATCH_EXPLAINABILITY_ERRORS_TOTAL",
    # Webhook
    "WEBHOOK_DELIVERIES_TOTAL",
    "WEBHOOK_DELIVERY_LATENCY",
    "WEBHOOK_FAILURES_BY_ENDPOINT",
    "WEBHOOK_RETRIES_TOTAL",
    "WEBHOOK_CIRCUIT_BREAKER_STATES",
    "WEBHOOK_QUEUE_SIZE",
    # Security & Governance Hardening
    "ENCRYPTION_OPERATIONS_TOTAL",
    "ENCRYPTION_OPERATION_LATENCY",
    "ENCRYPTION_ERRORS_TOTAL",
    "RBAC_PERMISSION_CHECKS_TOTAL",
    "RBAC_PERMISSION_DENIED_TOTAL",
    "RBAC_CHECK_LATENCY",
    "MIGRATION_RECORDS_TOTAL",
    "MIGRATION_ERRORS_TOTAL",
    # Gauntlet
    "GAUNTLET_EXPORTS_TOTAL",
    "GAUNTLET_EXPORT_LATENCY",
    "GAUNTLET_EXPORT_SIZE",
    # Workflow Template
    "WORKFLOW_TEMPLATES_CREATED",
    "WORKFLOW_TEMPLATE_EXECUTIONS",
    "WORKFLOW_TEMPLATE_EXECUTION_LATENCY",
    # Initialization
    "start_metrics_server",
    # Recording functions
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
    "record_performance_routing_decision",
    "record_performance_routing_latency",
    "record_outcome_complexity_adjustment",
    "record_analytics_selection_recommendation",
    "record_novelty_score_calculation",
    "record_novelty_penalty",
    "record_echo_chamber_detection",
    "record_relationship_bias_adjustment",
    "record_rlm_selection_recommendation",
    "record_calibration_cost_calculation",
    "record_budget_filtering_event",
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
    # Knowledge Mound recording
    "record_km_operation",
    "record_km_cache_access",
    "set_km_health_status",
    "record_km_adapter_sync",
    "record_km_federated_query",
    "record_km_event_emitted",
    "set_km_active_adapters",
    "sync_km_metrics_to_prometheus",
    # Notification recording
    "record_notification_sent",
    "record_notification_error",
    "set_notification_queue_size",
    "track_notification_delivery",
    # Task Queue recording (from submodule)
    "record_task_queue_operation",
    "set_task_queue_size",
    "record_task_queue_recovery",
    "record_task_queue_cleanup",
    "track_task_queue_operation",
    # Governance recording (from submodule)
    "record_governance_decision",
    "record_governance_verification",
    "record_governance_approval",
    "record_governance_store_latency",
    "set_governance_artifacts_active",
    "track_governance_store_operation",
    # User mapping recording (from submodule)
    "record_user_mapping_operation",
    "record_user_mapping_cache_hit",
    "record_user_mapping_cache_miss",
    "set_user_mappings_active",
    # Checkpoint recording (from submodule)
    "record_checkpoint_operation",
    "record_checkpoint_restore_result",
    "track_checkpoint_operation",
    # Consensus recording (from submodule)
    "record_consensus_ingestion",
    "record_consensus_dissent",
    "record_consensus_evolution",
    "record_consensus_evidence_linked",
    "record_consensus_agreement_ratio",
    "record_km_inbound_event",
    # Gauntlet
    "record_gauntlet_export",
    "track_gauntlet_export",
    # Workflow template
    "record_workflow_template_created",
    "record_workflow_template_execution",
    "track_workflow_template_execution",
    # Marketplace
    "set_marketplace_templates_count",
    "record_marketplace_download",
    "record_marketplace_rating",
    "record_marketplace_review",
    "record_marketplace_operation_latency",
    "track_marketplace_operation",
    # Batch Explainability
    "set_batch_explainability_jobs_active",
    "record_batch_explainability_job",
    "record_batch_explainability_debate",
    "record_batch_explainability_error",
    "track_batch_explainability_debate",
    # Webhook
    "record_webhook_delivery",
    "record_webhook_failure",
    "record_webhook_retry",
    "set_webhook_circuit_breaker_state",
    "set_webhook_queue_size",
    "track_webhook_delivery",
    # Security & Governance Hardening
    "record_encryption_operation",
    "record_encryption_error",
    "track_encryption_operation",
    "record_rbac_check",
    "track_rbac_check",
    "record_migration_record",
    "record_migration_error",
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

        # Initialize debate-specific metrics
        _init_debate_metrics_internal()

        # Initialize cache metrics
        _init_cache_metrics_internal()

        # Initialize cross-functional metrics
        _init_cross_functional_metrics_internal()

        # Initialize Phase 9 metrics
        _init_phase9_metrics_internal()

        # Initialize slow debate metrics
        _init_slow_debate_metrics_internal()

        # Initialize feature metrics
        _init_feature_metrics_internal()

        # Initialize KM metrics
        _init_km_metrics_internal()

        # Initialize notification metrics
        _init_notification_metrics_internal()

        # Initialize marketplace metrics
        _init_marketplace_metrics_internal()

        # Initialize batch explainability metrics
        _init_batch_explainability_metrics_internal()

        # Initialize webhook metrics
        _init_webhook_metrics_internal()

        # Initialize security metrics
        _init_security_metrics_internal()

        # Initialize gauntlet metrics
        _init_gauntlet_metrics_internal()

        # Initialize workflow template metrics
        _init_workflow_metrics_internal()

        # Initialize submodule metrics
        init_task_queue_metrics()
        init_governance_metrics()
        init_user_mapping_metrics()
        init_checkpoint_metrics()
        init_consensus_metrics()
        init_enhanced_consensus_metrics()

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
    """Initialize debate-specific metrics (internal helper)."""
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
    """Initialize cache metrics (internal helper)."""
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
    """Initialize cross-functional feature metrics (internal helper)."""
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
    """Initialize Phase 9 Cross-Pollination metrics (internal helper)."""
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

    RLM_CACHE_HITS = Counter(
        "aragora_rlm_cache_hits_total",
        "RLM compression cache hits",
    )

    RLM_CACHE_MISSES = Counter(
        "aragora_rlm_cache_misses_total",
        "RLM compression cache misses",
    )

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
    """Initialize slow debate detection metrics (internal helper)."""
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
    """Initialize new feature metrics (TTS, convergence, vote bonuses) (internal helper)."""
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


def _init_km_metrics_internal() -> None:
    """Initialize Knowledge Mound metrics (internal helper)."""
    global KM_OPERATIONS_TOTAL, KM_OPERATION_LATENCY
    global KM_CACHE_HITS_TOTAL, KM_CACHE_MISSES_TOTAL
    global KM_HEALTH_STATUS, KM_ADAPTER_SYNCS_TOTAL
    global KM_FEDERATED_QUERIES_TOTAL, KM_EVENTS_EMITTED_TOTAL
    global KM_ACTIVE_ADAPTERS
    from prometheus_client import Counter, Gauge, Histogram

    KM_OPERATIONS_TOTAL = Counter(
        "aragora_km_operations_total",
        "Total Knowledge Mound operations",
        ["operation", "status"],
    )

    KM_OPERATION_LATENCY = Histogram(
        "aragora_km_operation_latency_seconds",
        "Knowledge Mound operation latency",
        ["operation"],
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    )

    KM_CACHE_HITS_TOTAL = Counter(
        "aragora_km_cache_hits_total",
        "Knowledge Mound cache hits",
        ["adapter"],
    )

    KM_CACHE_MISSES_TOTAL = Counter(
        "aragora_km_cache_misses_total",
        "Knowledge Mound cache misses",
        ["adapter"],
    )

    KM_HEALTH_STATUS = Gauge(
        "aragora_km_health_status",
        "Knowledge Mound health status (0=unknown, 1=unhealthy, 2=degraded, 3=healthy)",
    )

    KM_ADAPTER_SYNCS_TOTAL = Counter(
        "aragora_km_adapter_syncs_total",
        "Knowledge Mound adapter sync operations",
        ["adapter", "direction", "status"],
    )

    KM_FEDERATED_QUERIES_TOTAL = Counter(
        "aragora_km_federated_queries_total",
        "Knowledge Mound federated query operations",
        ["adapters_queried", "status"],
    )

    KM_EVENTS_EMITTED_TOTAL = Counter(
        "aragora_km_events_emitted_total",
        "Knowledge Mound WebSocket events emitted",
        ["event_type"],
    )

    KM_ACTIVE_ADAPTERS = Gauge(
        "aragora_km_active_adapters",
        "Number of active Knowledge Mound adapters",
    )


def _init_notification_metrics_internal() -> None:
    """Initialize notification delivery metrics (internal helper)."""
    global NOTIFICATION_SENT_TOTAL, NOTIFICATION_LATENCY
    global NOTIFICATION_ERRORS_TOTAL, NOTIFICATION_QUEUE_SIZE
    from prometheus_client import Counter, Gauge, Histogram

    NOTIFICATION_SENT_TOTAL = Counter(
        "aragora_notification_sent_total",
        "Total notifications sent",
        ["channel", "severity", "priority", "status"],
    )

    NOTIFICATION_LATENCY = Histogram(
        "aragora_notification_latency_seconds",
        "Notification delivery latency in seconds",
        ["channel"],
        buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
    )

    NOTIFICATION_ERRORS_TOTAL = Counter(
        "aragora_notification_errors_total",
        "Total notification delivery errors",
        ["channel", "error_type"],
    )

    NOTIFICATION_QUEUE_SIZE = Gauge(
        "aragora_notification_queue_size",
        "Current notification queue size",
        ["channel"],
    )


def _init_marketplace_metrics_internal() -> None:
    """Initialize marketplace metrics (internal helper)."""
    global MARKETPLACE_TEMPLATES_TOTAL, MARKETPLACE_DOWNLOADS_TOTAL
    global MARKETPLACE_RATINGS_TOTAL, MARKETPLACE_RATINGS_DISTRIBUTION
    global MARKETPLACE_REVIEWS_TOTAL, MARKETPLACE_OPERATION_LATENCY
    from prometheus_client import Counter, Gauge, Histogram

    MARKETPLACE_TEMPLATES_TOTAL = Gauge(
        "aragora_marketplace_templates_total",
        "Total number of templates in the marketplace",
        ["category", "visibility"],
    )

    MARKETPLACE_DOWNLOADS_TOTAL = Counter(
        "aragora_marketplace_downloads_total",
        "Total template downloads",
        ["template_id", "category"],
    )

    MARKETPLACE_RATINGS_TOTAL = Counter(
        "aragora_marketplace_ratings_total",
        "Total template ratings submitted",
        ["template_id"],
    )

    MARKETPLACE_RATINGS_DISTRIBUTION = Histogram(
        "aragora_marketplace_ratings_distribution",
        "Distribution of template ratings",
        ["category"],
        buckets=[1, 2, 3, 4, 5],
    )

    MARKETPLACE_REVIEWS_TOTAL = Counter(
        "aragora_marketplace_reviews_total",
        "Total template reviews submitted",
        ["template_id", "status"],
    )

    MARKETPLACE_OPERATION_LATENCY = Histogram(
        "aragora_marketplace_operation_latency_seconds",
        "Marketplace operation latency in seconds",
        ["operation"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    )


def _init_batch_explainability_metrics_internal() -> None:
    """Initialize batch explainability metrics (internal helper)."""
    global BATCH_EXPLAINABILITY_JOBS_ACTIVE, BATCH_EXPLAINABILITY_JOBS_TOTAL
    global BATCH_EXPLAINABILITY_DEBATES_PROCESSED, BATCH_EXPLAINABILITY_PROCESSING_LATENCY
    global BATCH_EXPLAINABILITY_ERRORS_TOTAL
    from prometheus_client import Counter, Gauge, Histogram

    BATCH_EXPLAINABILITY_JOBS_ACTIVE = Gauge(
        "aragora_explainability_batch_jobs_active",
        "Number of active batch explainability jobs",
    )

    BATCH_EXPLAINABILITY_JOBS_TOTAL = Counter(
        "aragora_explainability_batch_jobs_total",
        "Total batch explainability jobs",
        ["status"],
    )

    BATCH_EXPLAINABILITY_DEBATES_PROCESSED = Counter(
        "aragora_explainability_batch_debates_processed_total",
        "Total debates processed in batch jobs",
        ["status"],
    )

    BATCH_EXPLAINABILITY_PROCESSING_LATENCY = Histogram(
        "aragora_explainability_batch_processing_latency_seconds",
        "Batch explainability processing latency per debate",
        buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
    )

    BATCH_EXPLAINABILITY_ERRORS_TOTAL = Counter(
        "aragora_explainability_batch_errors_total",
        "Total errors in batch explainability processing",
        ["error_type"],
    )


def _init_webhook_metrics_internal() -> None:
    """Initialize webhook delivery metrics (internal helper)."""
    global WEBHOOK_DELIVERIES_TOTAL, WEBHOOK_DELIVERY_LATENCY
    global WEBHOOK_FAILURES_BY_ENDPOINT, WEBHOOK_RETRIES_TOTAL
    global WEBHOOK_CIRCUIT_BREAKER_STATES, WEBHOOK_QUEUE_SIZE
    from prometheus_client import Counter, Gauge, Histogram

    WEBHOOK_DELIVERIES_TOTAL = Counter(
        "aragora_webhook_deliveries_total",
        "Total webhook delivery attempts",
        ["endpoint", "status"],
    )

    WEBHOOK_DELIVERY_LATENCY = Histogram(
        "aragora_webhook_delivery_latency_seconds",
        "Webhook delivery latency in seconds",
        ["endpoint"],
        buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
    )

    WEBHOOK_FAILURES_BY_ENDPOINT = Counter(
        "aragora_webhook_failures_by_endpoint_total",
        "Webhook failures by endpoint and error type",
        ["endpoint", "error_type"],
    )

    WEBHOOK_RETRIES_TOTAL = Counter(
        "aragora_webhook_retries_total",
        "Total webhook delivery retries",
        ["endpoint", "attempt"],
    )

    WEBHOOK_CIRCUIT_BREAKER_STATES = Gauge(
        "aragora_webhook_circuit_breaker_state",
        "Circuit breaker state per endpoint (0=closed, 1=half-open, 2=open)",
        ["endpoint"],
    )

    WEBHOOK_QUEUE_SIZE = Gauge(
        "aragora_webhook_queue_size",
        "Current size of the webhook delivery queue",
    )


def _init_security_metrics_internal() -> None:
    """Initialize security & governance hardening metrics (internal helper)."""
    global ENCRYPTION_OPERATIONS_TOTAL, ENCRYPTION_OPERATION_LATENCY, ENCRYPTION_ERRORS_TOTAL
    global RBAC_PERMISSION_CHECKS_TOTAL, RBAC_PERMISSION_DENIED_TOTAL, RBAC_CHECK_LATENCY
    global MIGRATION_RECORDS_TOTAL, MIGRATION_ERRORS_TOTAL
    from prometheus_client import Counter, Histogram

    ENCRYPTION_OPERATIONS_TOTAL = Counter(
        "aragora_encryption_operations_total",
        "Total encryption/decryption operations",
        ["operation", "store"],
    )

    ENCRYPTION_OPERATION_LATENCY = Histogram(
        "aragora_encryption_operation_latency_seconds",
        "Time spent on encryption/decryption operations",
        ["operation"],
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
    )

    ENCRYPTION_ERRORS_TOTAL = Counter(
        "aragora_encryption_errors_total",
        "Total encryption/decryption errors",
        ["operation", "error_type"],
    )

    RBAC_PERMISSION_CHECKS_TOTAL = Counter(
        "aragora_rbac_permission_checks_total",
        "Total RBAC permission checks",
        ["permission", "result"],
    )

    RBAC_PERMISSION_DENIED_TOTAL = Counter(
        "aragora_rbac_permission_denied_total",
        "Total RBAC permission denials",
        ["permission", "handler"],
    )

    RBAC_CHECK_LATENCY = Histogram(
        "aragora_rbac_check_latency_seconds",
        "Time spent on RBAC permission checks",
        buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
    )

    MIGRATION_RECORDS_TOTAL = Counter(
        "aragora_migration_records_total",
        "Total records processed during data migration",
        ["store", "status"],
    )

    MIGRATION_ERRORS_TOTAL = Counter(
        "aragora_migration_errors_total",
        "Total errors during data migration",
        ["store", "error_type"],
    )


def _init_gauntlet_metrics_internal() -> None:
    """Initialize Gauntlet export metrics (internal helper)."""
    global GAUNTLET_EXPORTS_TOTAL, GAUNTLET_EXPORT_LATENCY, GAUNTLET_EXPORT_SIZE
    from prometheus_client import Counter, Histogram

    GAUNTLET_EXPORTS_TOTAL = Counter(
        "aragora_gauntlet_exports_total",
        "Total Gauntlet export operations",
        ["format", "type", "status"],
    )

    GAUNTLET_EXPORT_LATENCY = Histogram(
        "aragora_gauntlet_export_latency_seconds",
        "Gauntlet export operation latency",
        ["format", "type"],
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    )

    GAUNTLET_EXPORT_SIZE = Histogram(
        "aragora_gauntlet_export_size_bytes",
        "Gauntlet export output size in bytes",
        ["format", "type"],
        buckets=[100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000],
    )


def _init_workflow_metrics_internal() -> None:
    """Initialize workflow template metrics (internal helper)."""
    global WORKFLOW_TEMPLATES_CREATED, WORKFLOW_TEMPLATE_EXECUTIONS
    global WORKFLOW_TEMPLATE_EXECUTION_LATENCY
    from prometheus_client import Counter, Histogram

    WORKFLOW_TEMPLATES_CREATED = Counter(
        "aragora_workflow_templates_created_total",
        "Total workflow templates created",
        ["pattern", "template_id"],
    )

    WORKFLOW_TEMPLATE_EXECUTIONS = Counter(
        "aragora_workflow_template_executions_total",
        "Total workflow template executions",
        ["pattern", "status"],
    )

    WORKFLOW_TEMPLATE_EXECUTION_LATENCY = Histogram(
        "aragora_workflow_template_execution_latency_seconds",
        "Workflow template execution latency",
        ["pattern"],
        buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0],
    )


def _init_noop_metrics() -> None:
    """Initialize no-op metrics for when prometheus is disabled."""
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
    global TTS_SYNTHESIS_TOTAL, TTS_SYNTHESIS_LATENCY
    global CONVERGENCE_CHECKS_TOTAL, EVIDENCE_CITATION_BONUSES
    global PROCESS_EVALUATION_BONUSES, RLM_READY_QUORUM_EVENTS
    global KM_OPERATIONS_TOTAL, KM_OPERATION_LATENCY
    global KM_CACHE_HITS_TOTAL, KM_CACHE_MISSES_TOTAL
    global KM_HEALTH_STATUS, KM_ADAPTER_SYNCS_TOTAL
    global KM_FEDERATED_QUERIES_TOTAL, KM_EVENTS_EMITTED_TOTAL
    global KM_ACTIVE_ADAPTERS
    global NOTIFICATION_SENT_TOTAL, NOTIFICATION_LATENCY
    global NOTIFICATION_ERRORS_TOTAL, NOTIFICATION_QUEUE_SIZE
    global SLOW_DEBATES_TOTAL, SLOW_ROUNDS_TOTAL, DEBATE_ROUND_LATENCY
    global MARKETPLACE_TEMPLATES_TOTAL, MARKETPLACE_DOWNLOADS_TOTAL
    global MARKETPLACE_RATINGS_TOTAL, MARKETPLACE_RATINGS_DISTRIBUTION
    global MARKETPLACE_REVIEWS_TOTAL, MARKETPLACE_OPERATION_LATENCY
    global BATCH_EXPLAINABILITY_JOBS_ACTIVE, BATCH_EXPLAINABILITY_JOBS_TOTAL
    global BATCH_EXPLAINABILITY_DEBATES_PROCESSED, BATCH_EXPLAINABILITY_PROCESSING_LATENCY
    global BATCH_EXPLAINABILITY_ERRORS_TOTAL
    global WEBHOOK_DELIVERIES_TOTAL, WEBHOOK_DELIVERY_LATENCY
    global WEBHOOK_FAILURES_BY_ENDPOINT, WEBHOOK_RETRIES_TOTAL
    global WEBHOOK_CIRCUIT_BREAKER_STATES, WEBHOOK_QUEUE_SIZE
    global ENCRYPTION_OPERATIONS_TOTAL, ENCRYPTION_OPERATION_LATENCY, ENCRYPTION_ERRORS_TOTAL
    global RBAC_PERMISSION_CHECKS_TOTAL, RBAC_PERMISSION_DENIED_TOTAL, RBAC_CHECK_LATENCY
    global MIGRATION_RECORDS_TOTAL, MIGRATION_ERRORS_TOTAL
    global GAUNTLET_EXPORTS_TOTAL, GAUNTLET_EXPORT_LATENCY, GAUNTLET_EXPORT_SIZE
    global WORKFLOW_TEMPLATES_CREATED, WORKFLOW_TEMPLATE_EXECUTIONS
    global WORKFLOW_TEMPLATE_EXECUTION_LATENCY

    # Core metrics
    REQUEST_COUNT = NoOpMetric()
    REQUEST_LATENCY = NoOpMetric()
    AGENT_CALLS = NoOpMetric()
    AGENT_LATENCY = NoOpMetric()
    ACTIVE_DEBATES = NoOpMetric()
    CONSENSUS_RATE = NoOpMetric()
    MEMORY_OPERATIONS = NoOpMetric()
    WEBSOCKET_CONNECTIONS = NoOpMetric()
    DEBATE_DURATION = NoOpMetric()
    DEBATE_ROUNDS = NoOpMetric()
    DEBATE_PHASE_DURATION = NoOpMetric()
    SLOW_DEBATES_TOTAL = NoOpMetric()
    SLOW_ROUNDS_TOTAL = NoOpMetric()
    DEBATE_ROUND_LATENCY = NoOpMetric()
    AGENT_PARTICIPATION = NoOpMetric()
    CACHE_HITS = NoOpMetric()
    CACHE_MISSES = NoOpMetric()
    KNOWLEDGE_CACHE_HITS = NoOpMetric()
    KNOWLEDGE_CACHE_MISSES = NoOpMetric()
    MEMORY_COORDINATOR_WRITES = NoOpMetric()
    SELECTION_FEEDBACK_ADJUSTMENTS = NoOpMetric()
    WORKFLOW_TRIGGERS = NoOpMetric()
    EVIDENCE_STORED = NoOpMetric()
    CULTURE_PATTERNS = NoOpMetric()
    RLM_CACHE_HITS = NoOpMetric()
    RLM_CACHE_MISSES = NoOpMetric()
    CALIBRATION_ADJUSTMENTS = NoOpMetric()
    LEARNING_BONUSES = NoOpMetric()
    VOTING_ACCURACY_UPDATES = NoOpMetric()
    ADAPTIVE_ROUND_CHANGES = NoOpMetric()
    BRIDGE_SYNCS = NoOpMetric()
    BRIDGE_SYNC_LATENCY = NoOpMetric()
    BRIDGE_ERRORS = NoOpMetric()
    PERFORMANCE_ROUTING_DECISIONS = NoOpMetric()
    PERFORMANCE_ROUTING_LATENCY = NoOpMetric()
    OUTCOME_COMPLEXITY_ADJUSTMENTS = NoOpMetric()
    ANALYTICS_SELECTION_RECOMMENDATIONS = NoOpMetric()
    NOVELTY_SCORE_CALCULATIONS = NoOpMetric()
    NOVELTY_PENALTIES = NoOpMetric()
    ECHO_CHAMBER_DETECTIONS = NoOpMetric()
    RELATIONSHIP_BIAS_ADJUSTMENTS = NoOpMetric()
    RLM_SELECTION_RECOMMENDATIONS = NoOpMetric()
    CALIBRATION_COST_CALCULATIONS = NoOpMetric()
    BUDGET_FILTERING_EVENTS = NoOpMetric()
    TTS_SYNTHESIS_TOTAL = NoOpMetric()
    TTS_SYNTHESIS_LATENCY = NoOpMetric()
    CONVERGENCE_CHECKS_TOTAL = NoOpMetric()
    EVIDENCE_CITATION_BONUSES = NoOpMetric()
    PROCESS_EVALUATION_BONUSES = NoOpMetric()
    RLM_READY_QUORUM_EVENTS = NoOpMetric()
    # Knowledge Mound
    KM_OPERATIONS_TOTAL = NoOpMetric()
    KM_OPERATION_LATENCY = NoOpMetric()
    KM_CACHE_HITS_TOTAL = NoOpMetric()
    KM_CACHE_MISSES_TOTAL = NoOpMetric()
    KM_HEALTH_STATUS = NoOpMetric()
    KM_ADAPTER_SYNCS_TOTAL = NoOpMetric()
    KM_FEDERATED_QUERIES_TOTAL = NoOpMetric()
    KM_EVENTS_EMITTED_TOTAL = NoOpMetric()
    KM_ACTIVE_ADAPTERS = NoOpMetric()
    # Notifications
    NOTIFICATION_SENT_TOTAL = NoOpMetric()
    NOTIFICATION_LATENCY = NoOpMetric()
    NOTIFICATION_ERRORS_TOTAL = NoOpMetric()
    NOTIFICATION_QUEUE_SIZE = NoOpMetric()
    # Marketplace
    MARKETPLACE_TEMPLATES_TOTAL = NoOpMetric()
    MARKETPLACE_DOWNLOADS_TOTAL = NoOpMetric()
    MARKETPLACE_RATINGS_TOTAL = NoOpMetric()
    MARKETPLACE_RATINGS_DISTRIBUTION = NoOpMetric()
    MARKETPLACE_REVIEWS_TOTAL = NoOpMetric()
    MARKETPLACE_OPERATION_LATENCY = NoOpMetric()
    # Batch Explainability
    BATCH_EXPLAINABILITY_JOBS_ACTIVE = NoOpMetric()
    BATCH_EXPLAINABILITY_JOBS_TOTAL = NoOpMetric()
    BATCH_EXPLAINABILITY_DEBATES_PROCESSED = NoOpMetric()
    BATCH_EXPLAINABILITY_PROCESSING_LATENCY = NoOpMetric()
    BATCH_EXPLAINABILITY_ERRORS_TOTAL = NoOpMetric()
    # Webhook Delivery
    WEBHOOK_DELIVERIES_TOTAL = NoOpMetric()
    WEBHOOK_DELIVERY_LATENCY = NoOpMetric()
    WEBHOOK_FAILURES_BY_ENDPOINT = NoOpMetric()
    WEBHOOK_RETRIES_TOTAL = NoOpMetric()
    WEBHOOK_CIRCUIT_BREAKER_STATES = NoOpMetric()
    WEBHOOK_QUEUE_SIZE = NoOpMetric()
    # Security & Governance Hardening
    ENCRYPTION_OPERATIONS_TOTAL = NoOpMetric()
    ENCRYPTION_OPERATION_LATENCY = NoOpMetric()
    ENCRYPTION_ERRORS_TOTAL = NoOpMetric()
    RBAC_PERMISSION_CHECKS_TOTAL = NoOpMetric()
    RBAC_PERMISSION_DENIED_TOTAL = NoOpMetric()
    RBAC_CHECK_LATENCY = NoOpMetric()
    MIGRATION_RECORDS_TOTAL = NoOpMetric()
    MIGRATION_ERRORS_TOTAL = NoOpMetric()
    # Gauntlet
    GAUNTLET_EXPORTS_TOTAL = NoOpMetric()
    GAUNTLET_EXPORT_LATENCY = NoOpMetric()
    GAUNTLET_EXPORT_SIZE = NoOpMetric()
    # Workflow Template
    WORKFLOW_TEMPLATES_CREATED = NoOpMetric()
    WORKFLOW_TEMPLATE_EXECUTIONS = NoOpMetric()
    WORKFLOW_TEMPLATE_EXECUTION_LATENCY = NoOpMetric()


def init_core_metrics() -> bool:
    """Initialize core metrics.

    Public alias for _init_metrics() for use by the metrics package.

    Returns:
        True if metrics were initialized successfully
    """
    return _init_metrics()


def start_metrics_server() -> Optional[Any]:
    """Start the Prometheus metrics HTTP server.

    Returns:
        The server instance, or None if metrics disabled
    """
    global _metrics_server

    if not _init_metrics():
        return None

    if _metrics_server is not None:
        return _metrics_server

    config = get_metrics_config()
    if not config.enabled:
        return None

    try:
        from prometheus_client import start_http_server

        _metrics_server = start_http_server(config.port)
        logger.info(f"Prometheus metrics server started on port {config.port}")
        return _metrics_server
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")
        return None


def _normalize_endpoint(endpoint: str) -> str:
    """Normalize endpoint path to control cardinality.

    Replaces dynamic path segments (IDs, UUIDs) with placeholders.

    Args:
        endpoint: Raw endpoint path

    Returns:
        Normalized endpoint path
    """
    # Replace UUIDs
    endpoint = re.sub(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        ":id",
        endpoint,
        flags=re.IGNORECASE,
    )

    # Replace numeric IDs
    endpoint = re.sub(r"/\d+", "/:id", endpoint)

    # Replace base64-like tokens
    endpoint = re.sub(r"/[A-Za-z0-9_-]{20,}", "/:token", endpoint)

    return endpoint


# =============================================================================
# Core Recording Functions
# =============================================================================


def record_request(
    method: str,
    endpoint: str,
    status: int,
    latency: float,
) -> None:
    """Record an HTTP request metric."""
    _init_metrics()
    normalized_endpoint = _normalize_endpoint(endpoint)
    REQUEST_COUNT.labels(method=method, endpoint=normalized_endpoint, status=str(status)).inc()
    REQUEST_LATENCY.labels(endpoint=normalized_endpoint).observe(latency)


def record_agent_call(
    agent: str,
    success: bool,
    latency: float,
) -> None:
    """Record an agent API call metric."""
    _init_metrics()
    status = "success" if success else "error"
    AGENT_CALLS.labels(agent=agent, status=status).inc()
    AGENT_LATENCY.labels(agent=agent).observe(latency)


@contextmanager
def track_debate() -> Generator[None, None, None]:
    """Context manager to track active debates."""
    _init_metrics()
    ACTIVE_DEBATES.inc()
    try:
        yield
    finally:
        ACTIVE_DEBATES.dec()


def set_consensus_rate(rate: float) -> None:
    """Set the consensus rate metric."""
    _init_metrics()
    CONSENSUS_RATE.set(rate)


def record_memory_operation(operation: str, tier: str) -> None:
    """Record a memory operation."""
    _init_metrics()
    MEMORY_OPERATIONS.labels(operation=operation, tier=tier).inc()


def track_websocket_connection(connected: bool) -> None:
    """Track WebSocket connection state."""
    _init_metrics()
    if connected:
        WEBSOCKET_CONNECTIONS.inc()
    else:
        WEBSOCKET_CONNECTIONS.dec()


def measure_latency(metric_name: str = "request") -> Callable[[F], F]:
    """Decorator to measure function latency."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _init_metrics()
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                latency = time.perf_counter() - start
                REQUEST_LATENCY.labels(endpoint=metric_name).observe(latency)

        return cast(F, wrapper)

    return decorator


def measure_async_latency(metric_name: str = "request") -> Callable[[F], F]:
    """Decorator to measure async function latency."""

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            _init_metrics()
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                latency = time.perf_counter() - start
                REQUEST_LATENCY.labels(endpoint=metric_name).observe(latency)

        return cast(F, wrapper)

    return decorator


# =============================================================================
# Debate-Specific Metrics
# =============================================================================


def record_debate_completion(
    duration_seconds: float,
    rounds: int,
    outcome: str,
) -> None:
    """Record metrics when a debate completes."""
    _init_metrics()
    DEBATE_DURATION.labels(outcome=outcome).observe(duration_seconds)
    DEBATE_ROUNDS.labels(outcome=outcome).observe(rounds)


def record_phase_duration(phase: str, duration_seconds: float) -> None:
    """Record the duration of a debate phase."""
    _init_metrics()
    DEBATE_PHASE_DURATION.labels(phase=phase).observe(duration_seconds)


def record_agent_participation(agent: str, phase: str) -> None:
    """Record agent participation in a debate phase."""
    _init_metrics()
    AGENT_PARTICIPATION.labels(agent=agent, phase=phase).inc()


@contextmanager
def track_phase(phase: str) -> Generator[None, None, None]:
    """Context manager to track phase duration."""
    _init_metrics()
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        DEBATE_PHASE_DURATION.labels(phase=phase).observe(duration)


# =============================================================================
# Cache Metrics
# =============================================================================


def record_cache_hit(cache_name: str) -> None:
    """Record a cache hit."""
    _init_metrics()
    CACHE_HITS.labels(cache_name=cache_name).inc()


def record_cache_miss(cache_name: str) -> None:
    """Record a cache miss."""
    _init_metrics()
    CACHE_MISSES.labels(cache_name=cache_name).inc()


# =============================================================================
# Cross-Functional Feature Metrics
# =============================================================================


def record_knowledge_cache_hit() -> None:
    """Record a knowledge query cache hit."""
    _init_metrics()
    KNOWLEDGE_CACHE_HITS.inc()


def record_knowledge_cache_miss() -> None:
    """Record a knowledge query cache miss."""
    _init_metrics()
    KNOWLEDGE_CACHE_MISSES.inc()


def record_memory_coordinator_write(status: str) -> None:
    """Record a memory coordinator write operation."""
    _init_metrics()
    MEMORY_COORDINATOR_WRITES.labels(status=status).inc()


def record_selection_feedback_adjustment(agent: str, direction: str) -> None:
    """Record an agent selection weight adjustment."""
    _init_metrics()
    SELECTION_FEEDBACK_ADJUSTMENTS.labels(agent=agent, direction=direction).inc()


def record_workflow_trigger(status: str) -> None:
    """Record a post-debate workflow trigger."""
    _init_metrics()
    WORKFLOW_TRIGGERS.labels(status=status).inc()


def record_evidence_stored(count: int = 1) -> None:
    """Record evidence items stored in knowledge mound."""
    _init_metrics()
    EVIDENCE_STORED.inc(count)


def record_culture_patterns(count: int = 1) -> None:
    """Record culture patterns extracted from debates."""
    _init_metrics()
    CULTURE_PATTERNS.inc(count)


# =============================================================================
# Phase 9 Cross-Pollination Metrics
# =============================================================================


def record_rlm_cache_hit() -> None:
    """Record an RLM compression cache hit."""
    _init_metrics()
    RLM_CACHE_HITS.inc()


def record_rlm_cache_miss() -> None:
    """Record an RLM compression cache miss."""
    _init_metrics()
    RLM_CACHE_MISSES.inc()


def record_calibration_adjustment(agent: str) -> None:
    """Record a proposal confidence calibration adjustment."""
    _init_metrics()
    CALIBRATION_ADJUSTMENTS.labels(agent=agent).inc()


def record_learning_bonus(agent: str, category: str) -> None:
    """Record a learning efficiency ELO bonus."""
    _init_metrics()
    LEARNING_BONUSES.labels(agent=agent, category=category).inc()


def record_voting_accuracy_update(result: str) -> None:
    """Record a voting accuracy update."""
    _init_metrics()
    VOTING_ACCURACY_UPDATES.labels(result=result).inc()


def record_adaptive_round_change(direction: str) -> None:
    """Record a debate round count adjustment."""
    _init_metrics()
    ADAPTIVE_ROUND_CHANGES.labels(direction=direction).inc()


# =============================================================================
# Phase 9 Bridge Metrics
# =============================================================================


def record_bridge_sync(bridge: str, success: bool) -> None:
    """Record a bridge sync operation."""
    _init_metrics()
    status = "success" if success else "error"
    BRIDGE_SYNCS.labels(bridge=bridge, status=status).inc()


def record_bridge_sync_latency(bridge: str, latency_seconds: float) -> None:
    """Record bridge sync operation latency."""
    _init_metrics()
    BRIDGE_SYNC_LATENCY.labels(bridge=bridge).observe(latency_seconds)


def record_bridge_error(bridge: str, error_type: str) -> None:
    """Record a bridge error."""
    _init_metrics()
    BRIDGE_ERRORS.labels(bridge=bridge, error_type=error_type).inc()


def record_performance_routing_decision(task_type: str, selected_agent: str) -> None:
    """Record a performance-based routing decision."""
    _init_metrics()
    PERFORMANCE_ROUTING_DECISIONS.labels(task_type=task_type, selected_agent=selected_agent).inc()


def record_performance_routing_latency(latency_seconds: float) -> None:
    """Record time to compute routing decision."""
    _init_metrics()
    PERFORMANCE_ROUTING_LATENCY.observe(latency_seconds)


def record_outcome_complexity_adjustment(direction: str) -> None:
    """Record a complexity budget adjustment."""
    _init_metrics()
    OUTCOME_COMPLEXITY_ADJUSTMENTS.labels(direction=direction).inc()


def record_analytics_selection_recommendation(recommendation_type: str) -> None:
    """Record an analytics-driven selection recommendation."""
    _init_metrics()
    ANALYTICS_SELECTION_RECOMMENDATIONS.labels(recommendation_type=recommendation_type).inc()


def record_novelty_score_calculation(agent: str) -> None:
    """Record a novelty score calculation."""
    _init_metrics()
    NOVELTY_SCORE_CALCULATIONS.labels(agent=agent).inc()


def record_novelty_penalty(agent: str) -> None:
    """Record a selection penalty for low novelty."""
    _init_metrics()
    NOVELTY_PENALTIES.labels(agent=agent).inc()


def record_echo_chamber_detection(risk_level: str) -> None:
    """Record an echo chamber risk detection."""
    _init_metrics()
    ECHO_CHAMBER_DETECTIONS.labels(risk_level=risk_level).inc()


def record_relationship_bias_adjustment(agent: str, direction: str) -> None:
    """Record a voting weight adjustment for alliance bias."""
    _init_metrics()
    RELATIONSHIP_BIAS_ADJUSTMENTS.labels(agent=agent, direction=direction).inc()


def record_rlm_selection_recommendation(agent: str) -> None:
    """Record an RLM-efficient agent selection recommendation."""
    _init_metrics()
    RLM_SELECTION_RECOMMENDATIONS.labels(agent=agent).inc()


def record_calibration_cost_calculation(agent: str, efficiency: str) -> None:
    """Record a cost efficiency calculation."""
    _init_metrics()
    CALIBRATION_COST_CALCULATIONS.labels(agent=agent, efficiency=efficiency).inc()


def record_budget_filtering_event(outcome: str) -> None:
    """Record an agent filtering event due to budget constraints."""
    _init_metrics()
    BUDGET_FILTERING_EVENTS.labels(outcome=outcome).inc()


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
# Slow Debate Detection Metrics
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
# New Feature Metrics (TTS, Convergence, Vote Bonuses)
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
# Knowledge Mound Metrics
# =============================================================================


def record_km_operation(operation: str, success: bool, latency_seconds: float) -> None:
    """Record a Knowledge Mound operation."""
    _init_metrics()
    status = "success" if success else "error"
    KM_OPERATIONS_TOTAL.labels(operation=operation, status=status).inc()
    KM_OPERATION_LATENCY.labels(operation=operation).observe(latency_seconds)


def record_km_cache_access(hit: bool, adapter: str = "global") -> None:
    """Record a Knowledge Mound cache access."""
    _init_metrics()
    if hit:
        KM_CACHE_HITS_TOTAL.labels(adapter=adapter).inc()
    else:
        KM_CACHE_MISSES_TOTAL.labels(adapter=adapter).inc()


def set_km_health_status(status: int) -> None:
    """Set the Knowledge Mound health status."""
    _init_metrics()
    KM_HEALTH_STATUS.set(status)


def record_km_adapter_sync(adapter: str, direction: str, success: bool) -> None:
    """Record a Knowledge Mound adapter sync operation."""
    _init_metrics()
    status = "success" if success else "error"
    KM_ADAPTER_SYNCS_TOTAL.labels(adapter=adapter, direction=direction, status=status).inc()


def record_km_federated_query(adapters_queried: int, success: bool) -> None:
    """Record a federated query operation."""
    _init_metrics()
    status = "success" if success else "error"
    KM_FEDERATED_QUERIES_TOTAL.labels(adapters_queried=str(adapters_queried), status=status).inc()


def record_km_event_emitted(event_type: str) -> None:
    """Record a Knowledge Mound WebSocket event emission."""
    _init_metrics()
    KM_EVENTS_EMITTED_TOTAL.labels(event_type=event_type).inc()


def set_km_active_adapters(count: int) -> None:
    """Set the number of active Knowledge Mound adapters."""
    _init_metrics()
    KM_ACTIVE_ADAPTERS.set(count)


def sync_km_metrics_to_prometheus() -> None:
    """Sync KMMetrics to Prometheus metrics."""
    _init_metrics()

    try:
        from aragora.knowledge.mound.metrics import get_metrics, HealthStatus

        km_metrics = get_metrics()
        health = km_metrics.get_health()

        status_map = {
            HealthStatus.UNKNOWN: 0,
            HealthStatus.UNHEALTHY: 1,
            HealthStatus.DEGRADED: 2,
            HealthStatus.HEALTHY: 3,
        }
        set_km_health_status(status_map.get(health.status, 0))

    except ImportError:
        logger.debug("KMMetrics not available for Prometheus sync")
    except (KeyError, AttributeError) as e:
        logger.debug(f"KM metrics data extraction failed: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error syncing KM metrics to Prometheus: {e}")


def record_km_inbound_event(
    event_type: str,
    source: str,
    success: bool = True,
) -> None:
    """Record an inbound event to Knowledge Mound."""
    _init_metrics()
    record_km_operation("ingest", success, 0.0)
    record_km_event_emitted(f"inbound_{event_type}")


# =============================================================================
# Notification Delivery Metrics
# =============================================================================


def record_notification_sent(
    channel: str,
    severity: str,
    priority: str,
    success: bool,
    latency_seconds: float,
) -> None:
    """Record a notification delivery attempt."""
    _init_metrics()
    status = "success" if success else "failed"
    NOTIFICATION_SENT_TOTAL.labels(
        channel=channel, severity=severity, priority=priority, status=status
    ).inc()
    NOTIFICATION_LATENCY.labels(channel=channel).observe(latency_seconds)


def record_notification_error(channel: str, error_type: str) -> None:
    """Record a notification delivery error."""
    _init_metrics()
    NOTIFICATION_ERRORS_TOTAL.labels(channel=channel, error_type=error_type).inc()


def set_notification_queue_size(channel: str, size: int) -> None:
    """Set the current notification queue size."""
    _init_metrics()
    NOTIFICATION_QUEUE_SIZE.labels(channel=channel).set(size)


@contextmanager
def track_notification_delivery(
    channel: str,
    severity: str = "info",
    priority: str = "normal",
) -> Generator[None, None, None]:
    """Context manager to track notification delivery."""
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
        record_notification_sent(channel, severity, priority, success, latency)


# =============================================================================
# Gauntlet Export Metrics
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
def track_gauntlet_export(
    format: str,
    export_type: str,
) -> Generator[dict, None, None]:
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
# Workflow Template Metrics
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
# Marketplace Metrics
# =============================================================================


def set_marketplace_templates_count(
    category: str,
    visibility: str,
    count: int,
) -> None:
    """Set the total number of templates in marketplace."""
    _init_metrics()
    MARKETPLACE_TEMPLATES_TOTAL.labels(category=category, visibility=visibility).set(count)


def record_marketplace_download(
    template_id: str,
    category: str,
) -> None:
    """Record a template download."""
    _init_metrics()
    MARKETPLACE_DOWNLOADS_TOTAL.labels(template_id=template_id, category=category).inc()


def record_marketplace_rating(
    template_id: str,
    category: str,
    rating: int,
) -> None:
    """Record a template rating."""
    _init_metrics()
    MARKETPLACE_RATINGS_TOTAL.labels(template_id=template_id).inc()
    MARKETPLACE_RATINGS_DISTRIBUTION.labels(category=category).observe(rating)


def record_marketplace_review(
    template_id: str,
    status: str,
) -> None:
    """Record a template review."""
    _init_metrics()
    MARKETPLACE_REVIEWS_TOTAL.labels(template_id=template_id, status=status).inc()


def record_marketplace_operation_latency(
    operation: str,
    latency_seconds: float,
) -> None:
    """Record marketplace operation latency."""
    _init_metrics()
    MARKETPLACE_OPERATION_LATENCY.labels(operation=operation).observe(latency_seconds)


@contextmanager
def track_marketplace_operation(operation: str) -> Generator[None, None, None]:
    """Context manager to track marketplace operations."""
    _init_metrics()
    start = time.perf_counter()
    try:
        yield
    finally:
        latency = time.perf_counter() - start
        record_marketplace_operation_latency(operation, latency)


# =============================================================================
# Batch Explainability Metrics
# =============================================================================


def set_batch_explainability_jobs_active(count: int) -> None:
    """Set the number of active batch explainability jobs."""
    _init_metrics()
    BATCH_EXPLAINABILITY_JOBS_ACTIVE.set(count)


def record_batch_explainability_job(status: str) -> None:
    """Record a batch explainability job."""
    _init_metrics()
    BATCH_EXPLAINABILITY_JOBS_TOTAL.labels(status=status).inc()


def record_batch_explainability_debate(
    status: str,
    latency_seconds: float,
) -> None:
    """Record a debate processed in a batch job."""
    _init_metrics()
    BATCH_EXPLAINABILITY_DEBATES_PROCESSED.labels(status=status).inc()
    BATCH_EXPLAINABILITY_PROCESSING_LATENCY.observe(latency_seconds)


def record_batch_explainability_error(error_type: str) -> None:
    """Record a batch explainability error."""
    _init_metrics()
    BATCH_EXPLAINABILITY_ERRORS_TOTAL.labels(error_type=error_type).inc()


@contextmanager
def track_batch_explainability_debate() -> Generator[None, None, None]:
    """Context manager to track debate processing in batch jobs."""
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
        status = "success" if success else "error"
        record_batch_explainability_debate(status, latency)


# =============================================================================
# Webhook Delivery Metrics
# =============================================================================


def record_webhook_delivery(
    endpoint: str,
    success: bool,
    latency_seconds: float,
) -> None:
    """Record a webhook delivery attempt."""
    _init_metrics()
    status = "success" if success else "failed"
    WEBHOOK_DELIVERIES_TOTAL.labels(endpoint=endpoint, status=status).inc()
    WEBHOOK_DELIVERY_LATENCY.labels(endpoint=endpoint).observe(latency_seconds)


def record_webhook_failure(
    endpoint: str,
    error_type: str,
) -> None:
    """Record a webhook delivery failure."""
    _init_metrics()
    WEBHOOK_FAILURES_BY_ENDPOINT.labels(endpoint=endpoint, error_type=error_type).inc()


def record_webhook_retry(
    endpoint: str,
    attempt: int,
) -> None:
    """Record a webhook delivery retry."""
    _init_metrics()
    WEBHOOK_RETRIES_TOTAL.labels(endpoint=endpoint, attempt=str(attempt)).inc()


def set_webhook_circuit_breaker_state(
    endpoint: str,
    state: str,
) -> None:
    """Set the circuit breaker state for a webhook endpoint."""
    _init_metrics()
    state_map = {"closed": 0, "half_open": 1, "open": 2}
    WEBHOOK_CIRCUIT_BREAKER_STATES.labels(endpoint=endpoint).set(state_map.get(state, 0))


def set_webhook_queue_size(size: int) -> None:
    """Set the current webhook delivery queue size."""
    _init_metrics()
    WEBHOOK_QUEUE_SIZE.set(size)


@contextmanager
def track_webhook_delivery(endpoint: str) -> Generator[None, None, None]:
    """Context manager to track webhook delivery."""
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
        record_webhook_delivery(endpoint, success, latency)


# =============================================================================
# Security & Governance Hardening Metrics
# =============================================================================


def record_encryption_operation(
    operation: str,
    store: str,
    latency_seconds: float,
) -> None:
    """Record an encryption or decryption operation."""
    _init_metrics()
    ENCRYPTION_OPERATIONS_TOTAL.labels(operation=operation, store=store).inc()
    ENCRYPTION_OPERATION_LATENCY.labels(operation=operation).observe(latency_seconds)


def record_encryption_error(
    operation: str,
    error_type: str,
) -> None:
    """Record an encryption or decryption error."""
    _init_metrics()
    ENCRYPTION_ERRORS_TOTAL.labels(operation=operation, error_type=error_type).inc()


@contextmanager
def track_encryption_operation(operation: str, store: str) -> Generator[None, None, None]:
    """Context manager to track encryption/decryption operations."""
    _init_metrics()
    start = time.perf_counter()
    try:
        yield
    except Exception as e:
        record_encryption_error(operation, type(e).__name__)
        raise
    finally:
        latency = time.perf_counter() - start
        record_encryption_operation(operation, store, latency)


def record_rbac_check(
    permission: str,
    allowed: bool,
    handler: str = "",
) -> None:
    """Record an RBAC permission check."""
    _init_metrics()
    result = "allowed" if allowed else "denied"
    RBAC_PERMISSION_CHECKS_TOTAL.labels(permission=permission, result=result).inc()
    if not allowed and handler:
        RBAC_PERMISSION_DENIED_TOTAL.labels(permission=permission, handler=handler).inc()


@contextmanager
def track_rbac_check(permission: str, handler: str = "") -> Generator[None, None, None]:
    """Context manager to track RBAC permission checks."""
    _init_metrics()
    start = time.perf_counter()
    try:
        yield
    finally:
        latency = time.perf_counter() - start
        RBAC_CHECK_LATENCY.observe(latency)


def record_migration_record(
    store: str,
    status: str,
) -> None:
    """Record a record processed during migration."""
    _init_metrics()
    MIGRATION_RECORDS_TOTAL.labels(store=store, status=status).inc()


def record_migration_error(
    store: str,
    error_type: str,
) -> None:
    """Record a migration error."""
    _init_metrics()
    MIGRATION_ERRORS_TOTAL.labels(store=store, error_type=error_type).inc()
