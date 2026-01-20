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
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator, Optional, TypeVar, cast

from aragora.observability.config import get_metrics_config

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Prometheus metrics - initialized lazily
_initialized = False
_metrics_server = None

# Metric instances (will be set during initialization)
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

# Persistent Task Queue metrics
TASK_QUEUE_OPERATIONS_TOTAL: Any = None
TASK_QUEUE_OPERATION_LATENCY: Any = None
TASK_QUEUE_SIZE: Any = None
TASK_QUEUE_RECOVERED_TOTAL: Any = None
TASK_QUEUE_CLEANUP_TOTAL: Any = None

# Governance Store metrics
GOVERNANCE_DECISIONS_TOTAL: Any = None
GOVERNANCE_VERIFICATIONS_TOTAL: Any = None
GOVERNANCE_APPROVALS_TOTAL: Any = None
GOVERNANCE_STORE_LATENCY: Any = None
GOVERNANCE_ARTIFACTS_ACTIVE: Any = None

# User ID Mapping metrics
USER_MAPPING_OPERATIONS_TOTAL: Any = None
USER_MAPPING_CACHE_HITS_TOTAL: Any = None
USER_MAPPING_CACHE_MISSES_TOTAL: Any = None
USER_MAPPINGS_ACTIVE: Any = None


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

        # Debate-specific metrics
        global DEBATE_DURATION, DEBATE_ROUNDS, DEBATE_PHASE_DURATION, AGENT_PARTICIPATION

        DEBATE_DURATION = Histogram(
            "aragora_debate_duration_seconds",
            "Debate duration in seconds",
            ["outcome"],  # consensus, no_consensus, error
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
            ["phase"],  # propose, critique, vote, consensus
            buckets=[0.5, 1, 2, 5, 10, 30, 60],
        )

        AGENT_PARTICIPATION = Counter(
            "aragora_agent_participation_total",
            "Agent participation in debates",
            ["agent", "phase"],
        )

        # Cache metrics
        global CACHE_HITS, CACHE_MISSES

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

        # Cross-functional feature metrics
        global KNOWLEDGE_CACHE_HITS, KNOWLEDGE_CACHE_MISSES
        global MEMORY_COORDINATOR_WRITES, SELECTION_FEEDBACK_ADJUSTMENTS
        global WORKFLOW_TRIGGERS, EVIDENCE_STORED, CULTURE_PATTERNS

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
            ["status"],  # success, failed, rolled_back
        )

        SELECTION_FEEDBACK_ADJUSTMENTS = Counter(
            "aragora_selection_feedback_adjustments_total",
            "Agent selection weight adjustments",
            ["agent", "direction"],  # up, down
        )

        WORKFLOW_TRIGGERS = Counter(
            "aragora_workflow_triggers_total",
            "Post-debate workflow triggers",
            ["status"],  # triggered, skipped, completed, failed
        )

        EVIDENCE_STORED = Counter(
            "aragora_evidence_stored_total",
            "Evidence items stored in knowledge mound",
        )

        CULTURE_PATTERNS = Counter(
            "aragora_culture_patterns_total",
            "Culture patterns extracted from debates",
        )

        # Phase 9 Cross-Pollination metrics
        global RLM_CACHE_HITS, RLM_CACHE_MISSES
        global CALIBRATION_ADJUSTMENTS, LEARNING_BONUSES
        global VOTING_ACCURACY_UPDATES, ADAPTIVE_ROUND_CHANGES

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
            ["agent", "category"],  # rapid, steady, slow
        )

        VOTING_ACCURACY_UPDATES = Counter(
            "aragora_voting_accuracy_updates_total",
            "Voting accuracy records updated",
            ["result"],  # correct, incorrect
        )

        ADAPTIVE_ROUND_CHANGES = Counter(
            "aragora_adaptive_round_changes_total",
            "Debate round count adjustments from memory strategy",
            ["direction"],  # increased, decreased, unchanged
        )

        # Phase 9 Bridge metrics
        global BRIDGE_SYNCS, BRIDGE_SYNC_LATENCY, BRIDGE_ERRORS
        global PERFORMANCE_ROUTING_DECISIONS, PERFORMANCE_ROUTING_LATENCY
        global OUTCOME_COMPLEXITY_ADJUSTMENTS, ANALYTICS_SELECTION_RECOMMENDATIONS
        global NOVELTY_SCORE_CALCULATIONS, NOVELTY_PENALTIES
        global ECHO_CHAMBER_DETECTIONS, RELATIONSHIP_BIAS_ADJUSTMENTS
        global RLM_SELECTION_RECOMMENDATIONS, CALIBRATION_COST_CALCULATIONS
        global BUDGET_FILTERING_EVENTS

        BRIDGE_SYNCS = Counter(
            "aragora_bridge_syncs_total",
            "Cross-pollination bridge sync operations",
            ["bridge", "status"],  # bridge name, success/error
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
            ["task_type", "selected_agent"],  # speed/precision/balanced
        )

        PERFORMANCE_ROUTING_LATENCY = Histogram(
            "aragora_performance_routing_latency_seconds",
            "Time to compute routing decision",
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1],
        )

        OUTCOME_COMPLEXITY_ADJUSTMENTS = Counter(
            "aragora_outcome_complexity_adjustments_total",
            "Complexity budget adjustments from outcome patterns",
            ["direction"],  # increased, decreased
        )

        ANALYTICS_SELECTION_RECOMMENDATIONS = Counter(
            "aragora_analytics_selection_recommendations_total",
            "Analytics-driven team selection recommendations",
            ["recommendation_type"],  # boost, penalty, neutral
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
            ["risk_level"],  # low, medium, high
        )

        RELATIONSHIP_BIAS_ADJUSTMENTS = Counter(
            "aragora_relationship_bias_adjustments_total",
            "Voting weight adjustments for alliance bias",
            ["agent", "direction"],  # up, down
        )

        RLM_SELECTION_RECOMMENDATIONS = Counter(
            "aragora_rlm_selection_recommendations_total",
            "RLM-efficient agent selection recommendations",
            ["agent"],
        )

        CALIBRATION_COST_CALCULATIONS = Counter(
            "aragora_calibration_cost_calculations_total",
            "Cost efficiency calculations with calibration",
            ["agent", "efficiency"],  # efficient, moderate, inefficient
        )

        BUDGET_FILTERING_EVENTS = Counter(
            "aragora_budget_filtering_events_total",
            "Agent filtering events due to budget constraints",
            ["outcome"],  # included, excluded
        )

        # Slow debate detection metrics
        global SLOW_DEBATES_TOTAL, SLOW_ROUNDS_TOTAL, DEBATE_ROUND_LATENCY

        SLOW_DEBATES_TOTAL = Counter(
            "aragora_slow_debates_total",
            "Number of debates flagged as slow (>30s per round)",
        )

        SLOW_ROUNDS_TOTAL = Counter(
            "aragora_slow_rounds_total",
            "Number of individual rounds flagged as slow",
            ["debate_outcome"],  # consensus, no_consensus, error
        )

        DEBATE_ROUND_LATENCY = Histogram(
            "aragora_debate_round_latency_seconds",
            "Latency per debate round",
            buckets=[1, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300],
        )

        # New feature metrics (TTS, convergence, vote bonuses)
        global TTS_SYNTHESIS_TOTAL, TTS_SYNTHESIS_LATENCY
        global CONVERGENCE_CHECKS_TOTAL, EVIDENCE_CITATION_BONUSES
        global PROCESS_EVALUATION_BONUSES, RLM_READY_QUORUM_EVENTS

        TTS_SYNTHESIS_TOTAL = Counter(
            "aragora_tts_synthesis_total",
            "Total TTS synthesis operations",
            ["voice", "platform"],  # voice type, chat platform
        )

        TTS_SYNTHESIS_LATENCY = Histogram(
            "aragora_tts_synthesis_latency_seconds",
            "TTS synthesis latency in seconds",
            buckets=[0.1, 0.5, 1, 2, 5, 10, 20],
        )

        CONVERGENCE_CHECKS_TOTAL = Counter(
            "aragora_convergence_checks_total",
            "Total convergence check events",
            ["status", "blocked"],  # converged/diverged, trickster_blocked
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

        # Knowledge Mound metrics
        global KM_OPERATIONS_TOTAL, KM_OPERATION_LATENCY
        global KM_CACHE_HITS_TOTAL, KM_CACHE_MISSES_TOTAL
        global KM_HEALTH_STATUS, KM_ADAPTER_SYNCS_TOTAL
        global KM_FEDERATED_QUERIES_TOTAL, KM_EVENTS_EMITTED_TOTAL
        global KM_ACTIVE_ADAPTERS

        KM_OPERATIONS_TOTAL = Counter(
            "aragora_km_operations_total",
            "Total Knowledge Mound operations",
            ["operation", "status"],  # query/store/get/delete/sync, success/error
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
            ["adapter"],  # source adapter or "global"
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
            ["adapter", "direction", "status"],  # adapter name, forward/reverse, success/error
        )

        KM_FEDERATED_QUERIES_TOTAL = Counter(
            "aragora_km_federated_queries_total",
            "Knowledge Mound federated query operations",
            ["adapters_queried", "status"],  # count of adapters, success/error
        )

        KM_EVENTS_EMITTED_TOTAL = Counter(
            "aragora_km_events_emitted_total",
            "Knowledge Mound WebSocket events emitted",
            ["event_type"],  # km_batch, knowledge_indexed, etc.
        )

        KM_ACTIVE_ADAPTERS = Gauge(
            "aragora_km_active_adapters",
            "Number of active Knowledge Mound adapters",
        )

        # Notification delivery metrics
        global NOTIFICATION_SENT_TOTAL, NOTIFICATION_LATENCY
        global NOTIFICATION_ERRORS_TOTAL, NOTIFICATION_QUEUE_SIZE

        NOTIFICATION_SENT_TOTAL = Counter(
            "aragora_notification_sent_total",
            "Total notifications sent",
            ["channel", "severity", "priority", "status"],  # slack/email/webhook, severity, priority, success/failed
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
            ["channel", "error_type"],  # channel, error category
        )

        NOTIFICATION_QUEUE_SIZE = Gauge(
            "aragora_notification_queue_size",
            "Current notification queue size",
            ["channel"],
        )

        # Persistent Task Queue metrics
        global TASK_QUEUE_OPERATIONS_TOTAL, TASK_QUEUE_OPERATION_LATENCY
        global TASK_QUEUE_SIZE, TASK_QUEUE_RECOVERED_TOTAL, TASK_QUEUE_CLEANUP_TOTAL

        TASK_QUEUE_OPERATIONS_TOTAL = Counter(
            "aragora_task_queue_operations_total",
            "Total task queue operations",
            ["operation", "status"],  # enqueue/dequeue/complete/fail/cancel, success/error
        )

        TASK_QUEUE_OPERATION_LATENCY = Histogram(
            "aragora_task_queue_operation_latency_seconds",
            "Task queue operation latency in seconds",
            ["operation"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        )

        TASK_QUEUE_SIZE = Gauge(
            "aragora_task_queue_size",
            "Current number of tasks in the queue",
            ["status"],  # pending/ready/running
        )

        TASK_QUEUE_RECOVERED_TOTAL = Counter(
            "aragora_task_queue_recovered_total",
            "Total tasks recovered on startup",
            ["original_status"],  # pending/ready/running
        )

        TASK_QUEUE_CLEANUP_TOTAL = Counter(
            "aragora_task_queue_cleanup_total",
            "Total completed tasks cleaned up",
        )

        # Governance Store metrics
        global GOVERNANCE_DECISIONS_TOTAL, GOVERNANCE_VERIFICATIONS_TOTAL
        global GOVERNANCE_APPROVALS_TOTAL, GOVERNANCE_STORE_LATENCY
        global GOVERNANCE_ARTIFACTS_ACTIVE

        GOVERNANCE_DECISIONS_TOTAL = Counter(
            "aragora_governance_decisions_total",
            "Total governance decisions stored",
            ["decision_type", "outcome"],  # manual/auto, approved/rejected
        )

        GOVERNANCE_VERIFICATIONS_TOTAL = Counter(
            "aragora_governance_verifications_total",
            "Total verifications stored",
            ["verification_type", "result"],  # formal/runtime, valid/invalid
        )

        GOVERNANCE_APPROVALS_TOTAL = Counter(
            "aragora_governance_approvals_total",
            "Total approvals stored",
            ["approval_type", "status"],  # nomic/deploy/change, granted/revoked
        )

        GOVERNANCE_STORE_LATENCY = Histogram(
            "aragora_governance_store_latency_seconds",
            "Governance store operation latency in seconds",
            ["operation"],  # save/get/list/delete
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
        )

        GOVERNANCE_ARTIFACTS_ACTIVE = Gauge(
            "aragora_governance_artifacts_active",
            "Current number of active governance artifacts",
            ["artifact_type"],  # decision/verification/approval
        )

        # User ID Mapping metrics
        global USER_MAPPING_OPERATIONS_TOTAL, USER_MAPPING_CACHE_HITS_TOTAL
        global USER_MAPPING_CACHE_MISSES_TOTAL, USER_MAPPINGS_ACTIVE

        USER_MAPPING_OPERATIONS_TOTAL = Counter(
            "aragora_user_mapping_operations_total",
            "Total user ID mapping operations",
            ["operation", "platform", "status"],  # save/get/delete, slack/discord/teams, success/not_found
        )

        USER_MAPPING_CACHE_HITS_TOTAL = Counter(
            "aragora_user_mapping_cache_hits_total",
            "User ID mapping cache hits",
            ["platform"],
        )

        USER_MAPPING_CACHE_MISSES_TOTAL = Counter(
            "aragora_user_mapping_cache_misses_total",
            "User ID mapping cache misses",
            ["platform"],
        )

        USER_MAPPINGS_ACTIVE = Gauge(
            "aragora_user_mappings_active",
            "Number of active user ID mappings",
            ["platform"],
        )

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
    # Persistent stores metrics
    global TASK_QUEUE_OPERATIONS_TOTAL, TASK_QUEUE_OPERATION_LATENCY
    global TASK_QUEUE_SIZE, TASK_QUEUE_RECOVERED_TOTAL, TASK_QUEUE_CLEANUP_TOTAL
    global GOVERNANCE_DECISIONS_TOTAL, GOVERNANCE_VERIFICATIONS_TOTAL
    global GOVERNANCE_APPROVALS_TOTAL, GOVERNANCE_STORE_LATENCY
    global GOVERNANCE_ARTIFACTS_ACTIVE
    global USER_MAPPING_OPERATIONS_TOTAL, USER_MAPPING_CACHE_HITS_TOTAL
    global USER_MAPPING_CACHE_MISSES_TOTAL, USER_MAPPINGS_ACTIVE

    class NoOpMetric:
        def labels(self, *args: Any, **kwargs: Any) -> "NoOpMetric":
            return self

        def inc(self, amount: float = 1) -> None:
            pass

        def dec(self, amount: float = 1) -> None:
            pass

        def set(self, value: float) -> None:
            pass

        def observe(self, value: float) -> None:
            pass

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
    global NOTIFICATION_SENT_TOTAL, NOTIFICATION_LATENCY
    global NOTIFICATION_ERRORS_TOTAL, NOTIFICATION_QUEUE_SIZE
    NOTIFICATION_SENT_TOTAL = NoOpMetric()
    NOTIFICATION_LATENCY = NoOpMetric()
    NOTIFICATION_ERRORS_TOTAL = NoOpMetric()
    NOTIFICATION_QUEUE_SIZE = NoOpMetric()
    # Persistent Task Queue
    TASK_QUEUE_OPERATIONS_TOTAL = NoOpMetric()
    TASK_QUEUE_OPERATION_LATENCY = NoOpMetric()
    TASK_QUEUE_SIZE = NoOpMetric()
    TASK_QUEUE_RECOVERED_TOTAL = NoOpMetric()
    TASK_QUEUE_CLEANUP_TOTAL = NoOpMetric()
    # Governance Store
    GOVERNANCE_DECISIONS_TOTAL = NoOpMetric()
    GOVERNANCE_VERIFICATIONS_TOTAL = NoOpMetric()
    GOVERNANCE_APPROVALS_TOTAL = NoOpMetric()
    GOVERNANCE_STORE_LATENCY = NoOpMetric()
    GOVERNANCE_ARTIFACTS_ACTIVE = NoOpMetric()
    # User ID Mapping
    USER_MAPPING_OPERATIONS_TOTAL = NoOpMetric()
    USER_MAPPING_CACHE_HITS_TOTAL = NoOpMetric()
    USER_MAPPING_CACHE_MISSES_TOTAL = NoOpMetric()
    USER_MAPPINGS_ACTIVE = NoOpMetric()


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


def record_request(
    method: str,
    endpoint: str,
    status: int,
    latency: float,
) -> None:
    """Record an HTTP request metric.

    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: Request endpoint path
        status: HTTP status code
        latency: Request latency in seconds
    """
    _init_metrics()

    # Normalize endpoint for cardinality control
    normalized_endpoint = _normalize_endpoint(endpoint)

    REQUEST_COUNT.labels(method=method, endpoint=normalized_endpoint, status=str(status)).inc()
    REQUEST_LATENCY.labels(endpoint=normalized_endpoint).observe(latency)


def record_agent_call(
    agent: str,
    success: bool,
    latency: float,
) -> None:
    """Record an agent API call metric.

    Args:
        agent: Agent name
        success: Whether the call succeeded
        latency: Call latency in seconds
    """
    _init_metrics()

    status = "success" if success else "error"
    AGENT_CALLS.labels(agent=agent, status=status).inc()
    AGENT_LATENCY.labels(agent=agent).observe(latency)


@contextmanager
def track_debate() -> Generator[None, None, None]:
    """Context manager to track active debates.

    Example:
        with track_debate():
            # Debate is running
            await arena.run()
    """
    _init_metrics()

    ACTIVE_DEBATES.inc()
    try:
        yield
    finally:
        ACTIVE_DEBATES.dec()


def set_consensus_rate(rate: float) -> None:
    """Set the consensus rate metric.

    Args:
        rate: Consensus rate between 0 and 1
    """
    _init_metrics()
    CONSENSUS_RATE.set(rate)


def record_memory_operation(operation: str, tier: str) -> None:
    """Record a memory operation.

    Args:
        operation: Operation type (store, query, promote, demote)
        tier: Memory tier (fast, medium, slow, glacial)
    """
    _init_metrics()
    MEMORY_OPERATIONS.labels(operation=operation, tier=tier).inc()


def track_websocket_connection(connected: bool) -> None:
    """Track WebSocket connection state.

    Args:
        connected: True if connected, False if disconnected
    """
    _init_metrics()
    if connected:
        WEBSOCKET_CONNECTIONS.inc()
    else:
        WEBSOCKET_CONNECTIONS.dec()


def measure_latency(metric_name: str = "request") -> Callable[[F], F]:
    """Decorator to measure function latency.

    Args:
        metric_name: Name for the latency metric

    Returns:
        Decorated function with latency measurement
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
                REQUEST_LATENCY.labels(endpoint=metric_name).observe(latency)

        return cast(F, wrapper)

    return decorator


def measure_async_latency(metric_name: str = "request") -> Callable[[F], F]:
    """Decorator to measure async function latency.

    Args:
        metric_name: Name for the latency metric

    Returns:
        Decorated async function with latency measurement
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
                REQUEST_LATENCY.labels(endpoint=metric_name).observe(latency)

        return cast(F, wrapper)

    return decorator


def _normalize_endpoint(endpoint: str) -> str:
    """Normalize endpoint path to control cardinality.

    Replaces dynamic path segments (IDs, UUIDs) with placeholders.

    Args:
        endpoint: Raw endpoint path

    Returns:
        Normalized endpoint path
    """
    import re

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
# Debate-Specific Metrics
# =============================================================================


def record_debate_completion(
    duration_seconds: float,
    rounds: int,
    outcome: str,
) -> None:
    """Record metrics when a debate completes.

    Args:
        duration_seconds: Total debate duration in seconds
        rounds: Number of rounds completed
        outcome: Debate outcome ("consensus", "no_consensus", "error")
    """
    _init_metrics()
    DEBATE_DURATION.labels(outcome=outcome).observe(duration_seconds)
    DEBATE_ROUNDS.labels(outcome=outcome).observe(rounds)


def record_phase_duration(phase: str, duration_seconds: float) -> None:
    """Record the duration of a debate phase.

    Args:
        phase: Phase name ("propose", "critique", "vote", "consensus")
        duration_seconds: Phase duration in seconds
    """
    _init_metrics()
    DEBATE_PHASE_DURATION.labels(phase=phase).observe(duration_seconds)


def record_agent_participation(agent: str, phase: str) -> None:
    """Record agent participation in a debate phase.

    Args:
        agent: Agent name
        phase: Phase name
    """
    _init_metrics()
    AGENT_PARTICIPATION.labels(agent=agent, phase=phase).inc()


@contextmanager
def track_phase(phase: str) -> Generator[None, None, None]:
    """Context manager to track phase duration.

    Args:
        phase: Phase name

    Example:
        with track_phase("propose"):
            # Phase is running
            await run_propose_phase()
    """
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
    """Record a cache hit.

    Args:
        cache_name: Name of the cache
    """
    _init_metrics()
    CACHE_HITS.labels(cache_name=cache_name).inc()


def record_cache_miss(cache_name: str) -> None:
    """Record a cache miss.

    Args:
        cache_name: Name of the cache
    """
    _init_metrics()
    CACHE_MISSES.labels(cache_name=cache_name).inc()


# Cross-functional feature metrics helpers


def record_knowledge_cache_hit() -> None:
    """Record a knowledge query cache hit."""
    _init_metrics()
    KNOWLEDGE_CACHE_HITS.inc()


def record_knowledge_cache_miss() -> None:
    """Record a knowledge query cache miss."""
    _init_metrics()
    KNOWLEDGE_CACHE_MISSES.inc()


def record_memory_coordinator_write(status: str) -> None:
    """Record a memory coordinator write operation.

    Args:
        status: Write status (success, failed, rolled_back)
    """
    _init_metrics()
    MEMORY_COORDINATOR_WRITES.labels(status=status).inc()


def record_selection_feedback_adjustment(agent: str, direction: str) -> None:
    """Record an agent selection weight adjustment.

    Args:
        agent: Agent name
        direction: Adjustment direction (up, down)
    """
    _init_metrics()
    SELECTION_FEEDBACK_ADJUSTMENTS.labels(agent=agent, direction=direction).inc()


def record_workflow_trigger(status: str) -> None:
    """Record a post-debate workflow trigger.

    Args:
        status: Trigger status (triggered, skipped, completed, failed)
    """
    _init_metrics()
    WORKFLOW_TRIGGERS.labels(status=status).inc()


def record_evidence_stored(count: int = 1) -> None:
    """Record evidence items stored in knowledge mound.

    Args:
        count: Number of evidence items stored
    """
    _init_metrics()
    EVIDENCE_STORED.inc(count)


def record_culture_patterns(count: int = 1) -> None:
    """Record culture patterns extracted from debates.

    Args:
        count: Number of patterns extracted
    """
    _init_metrics()
    CULTURE_PATTERNS.inc(count)


# Phase 9 Cross-Pollination metrics helpers


def record_rlm_cache_hit() -> None:
    """Record an RLM compression cache hit."""
    _init_metrics()
    RLM_CACHE_HITS.inc()


def record_rlm_cache_miss() -> None:
    """Record an RLM compression cache miss."""
    _init_metrics()
    RLM_CACHE_MISSES.inc()


def record_calibration_adjustment(agent: str) -> None:
    """Record a proposal confidence calibration adjustment.

    Args:
        agent: Agent name whose confidence was calibrated
    """
    _init_metrics()
    CALIBRATION_ADJUSTMENTS.labels(agent=agent).inc()


def record_learning_bonus(agent: str, category: str) -> None:
    """Record a learning efficiency ELO bonus.

    Args:
        agent: Agent name
        category: Learning category (rapid, steady, slow)
    """
    _init_metrics()
    LEARNING_BONUSES.labels(agent=agent, category=category).inc()


def record_voting_accuracy_update(result: str) -> None:
    """Record a voting accuracy update.

    Args:
        result: Vote result (correct, incorrect)
    """
    _init_metrics()
    VOTING_ACCURACY_UPDATES.labels(result=result).inc()


def record_adaptive_round_change(direction: str) -> None:
    """Record a debate round count adjustment.

    Args:
        direction: Change direction (increased, decreased, unchanged)
    """
    _init_metrics()
    ADAPTIVE_ROUND_CHANGES.labels(direction=direction).inc()


# =============================================================================
# Phase 9 Bridge Metrics
# =============================================================================


def record_bridge_sync(bridge: str, success: bool) -> None:
    """Record a bridge sync operation.

    Args:
        bridge: Bridge name (performance_router, relationship_bias, etc.)
        success: Whether the sync succeeded
    """
    _init_metrics()
    status = "success" if success else "error"
    BRIDGE_SYNCS.labels(bridge=bridge, status=status).inc()


def record_bridge_sync_latency(bridge: str, latency_seconds: float) -> None:
    """Record bridge sync operation latency.

    Args:
        bridge: Bridge name
        latency_seconds: Time taken for sync operation
    """
    _init_metrics()
    BRIDGE_SYNC_LATENCY.labels(bridge=bridge).observe(latency_seconds)


def record_bridge_error(bridge: str, error_type: str) -> None:
    """Record a bridge error.

    Args:
        bridge: Bridge name
        error_type: Type of error (e.g., "initialization", "sync", "compute")
    """
    _init_metrics()
    BRIDGE_ERRORS.labels(bridge=bridge, error_type=error_type).inc()


def record_performance_routing_decision(task_type: str, selected_agent: str) -> None:
    """Record a performance-based routing decision.

    Args:
        task_type: Task type (speed, precision, balanced)
        selected_agent: Agent selected for the task
    """
    _init_metrics()
    PERFORMANCE_ROUTING_DECISIONS.labels(task_type=task_type, selected_agent=selected_agent).inc()


def record_performance_routing_latency(latency_seconds: float) -> None:
    """Record time to compute routing decision.

    Args:
        latency_seconds: Time taken to compute routing
    """
    _init_metrics()
    PERFORMANCE_ROUTING_LATENCY.observe(latency_seconds)


def record_outcome_complexity_adjustment(direction: str) -> None:
    """Record a complexity budget adjustment.

    Args:
        direction: Adjustment direction (increased, decreased)
    """
    _init_metrics()
    OUTCOME_COMPLEXITY_ADJUSTMENTS.labels(direction=direction).inc()


def record_analytics_selection_recommendation(recommendation_type: str) -> None:
    """Record an analytics-driven selection recommendation.

    Args:
        recommendation_type: Type of recommendation (boost, penalty, neutral)
    """
    _init_metrics()
    ANALYTICS_SELECTION_RECOMMENDATIONS.labels(recommendation_type=recommendation_type).inc()


def record_novelty_score_calculation(agent: str) -> None:
    """Record a novelty score calculation.

    Args:
        agent: Agent name
    """
    _init_metrics()
    NOVELTY_SCORE_CALCULATIONS.labels(agent=agent).inc()


def record_novelty_penalty(agent: str) -> None:
    """Record a selection penalty for low novelty.

    Args:
        agent: Agent name
    """
    _init_metrics()
    NOVELTY_PENALTIES.labels(agent=agent).inc()


def record_echo_chamber_detection(risk_level: str) -> None:
    """Record an echo chamber risk detection.

    Args:
        risk_level: Risk level (low, medium, high)
    """
    _init_metrics()
    ECHO_CHAMBER_DETECTIONS.labels(risk_level=risk_level).inc()


def record_relationship_bias_adjustment(agent: str, direction: str) -> None:
    """Record a voting weight adjustment for alliance bias.

    Args:
        agent: Agent name
        direction: Adjustment direction (up, down)
    """
    _init_metrics()
    RELATIONSHIP_BIAS_ADJUSTMENTS.labels(agent=agent, direction=direction).inc()


def record_rlm_selection_recommendation(agent: str) -> None:
    """Record an RLM-efficient agent selection recommendation.

    Args:
        agent: Agent name recommended for RLM efficiency
    """
    _init_metrics()
    RLM_SELECTION_RECOMMENDATIONS.labels(agent=agent).inc()


def record_calibration_cost_calculation(agent: str, efficiency: str) -> None:
    """Record a cost efficiency calculation.

    Args:
        agent: Agent name
        efficiency: Efficiency category (efficient, moderate, inefficient)
    """
    _init_metrics()
    CALIBRATION_COST_CALCULATIONS.labels(agent=agent, efficiency=efficiency).inc()


def record_budget_filtering_event(outcome: str) -> None:
    """Record an agent filtering event due to budget constraints.

    Args:
        outcome: Filtering outcome (included, excluded)
    """
    _init_metrics()
    BUDGET_FILTERING_EVENTS.labels(outcome=outcome).inc()


@contextmanager
def track_bridge_sync(bridge: str) -> Generator[None, None, None]:
    """Context manager to track bridge sync operations.

    Automatically records sync success/failure and latency.

    Args:
        bridge: Bridge name

    Example:
        with track_bridge_sync("performance_router"):
            bridge.sync_to_router()
    """
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
    """Record a round flagged as slow.

    Args:
        debate_outcome: Current debate outcome (consensus, no_consensus, error, in_progress)
    """
    _init_metrics()
    SLOW_ROUNDS_TOTAL.labels(debate_outcome=debate_outcome).inc()


def record_round_latency(latency_seconds: float) -> None:
    """Record latency for a debate round.

    Args:
        latency_seconds: Round duration in seconds
    """
    _init_metrics()
    DEBATE_ROUND_LATENCY.observe(latency_seconds)


# =============================================================================
# New Feature Metrics (TTS, Convergence, Vote Bonuses)
# =============================================================================


def record_tts_synthesis(voice: str, platform: str = "unknown") -> None:
    """Record a TTS synthesis operation.

    Args:
        voice: Voice type used (e.g., narrator, moderator, analyst)
        platform: Chat platform (telegram, whatsapp, web)
    """
    _init_metrics()
    TTS_SYNTHESIS_TOTAL.labels(voice=voice, platform=platform).inc()


def record_tts_latency(latency_seconds: float) -> None:
    """Record TTS synthesis latency.

    Args:
        latency_seconds: Synthesis duration in seconds
    """
    _init_metrics()
    TTS_SYNTHESIS_LATENCY.observe(latency_seconds)


def record_convergence_check(status: str, blocked: bool = False) -> None:
    """Record a convergence check event.

    Args:
        status: Convergence status (converged, diverged, partial)
        blocked: Whether convergence was blocked by trickster
    """
    _init_metrics()
    CONVERGENCE_CHECKS_TOTAL.labels(status=status, blocked=str(blocked)).inc()


def record_evidence_citation_bonus(agent: str) -> None:
    """Record an evidence citation vote bonus.

    Args:
        agent: Agent name that received the bonus
    """
    _init_metrics()
    EVIDENCE_CITATION_BONUSES.labels(agent=agent).inc()


def record_process_evaluation_bonus(agent: str) -> None:
    """Record a process evaluation vote bonus.

    Args:
        agent: Agent name that received the bonus
    """
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
    """Record a Knowledge Mound operation.

    Args:
        operation: Operation type (query, store, get, delete, sync)
        success: Whether the operation succeeded
        latency_seconds: Operation latency in seconds
    """
    _init_metrics()
    status = "success" if success else "error"
    KM_OPERATIONS_TOTAL.labels(operation=operation, status=status).inc()
    KM_OPERATION_LATENCY.labels(operation=operation).observe(latency_seconds)


def record_km_cache_access(hit: bool, adapter: str = "global") -> None:
    """Record a Knowledge Mound cache access.

    Args:
        hit: Whether it was a cache hit
        adapter: Adapter name or "global" for general cache
    """
    _init_metrics()
    if hit:
        KM_CACHE_HITS_TOTAL.labels(adapter=adapter).inc()
    else:
        KM_CACHE_MISSES_TOTAL.labels(adapter=adapter).inc()


def set_km_health_status(status: int) -> None:
    """Set the Knowledge Mound health status.

    Args:
        status: Health status (0=unknown, 1=unhealthy, 2=degraded, 3=healthy)
    """
    _init_metrics()
    KM_HEALTH_STATUS.set(status)


def record_km_adapter_sync(adapter: str, direction: str, success: bool) -> None:
    """Record a Knowledge Mound adapter sync operation.

    Args:
        adapter: Adapter name (continuum, consensus, elo, etc.)
        direction: Sync direction (forward, reverse)
        success: Whether the sync succeeded
    """
    _init_metrics()
    status = "success" if success else "error"
    KM_ADAPTER_SYNCS_TOTAL.labels(adapter=adapter, direction=direction, status=status).inc()


def record_km_federated_query(adapters_queried: int, success: bool) -> None:
    """Record a federated query operation.

    Args:
        adapters_queried: Number of adapters queried
        success: Whether the query succeeded
    """
    _init_metrics()
    status = "success" if success else "error"
    KM_FEDERATED_QUERIES_TOTAL.labels(adapters_queried=str(adapters_queried), status=status).inc()


def record_km_event_emitted(event_type: str) -> None:
    """Record a Knowledge Mound WebSocket event emission.

    Args:
        event_type: Event type (km_batch, knowledge_indexed, etc.)
    """
    _init_metrics()
    KM_EVENTS_EMITTED_TOTAL.labels(event_type=event_type).inc()


def set_km_active_adapters(count: int) -> None:
    """Set the number of active Knowledge Mound adapters.

    Args:
        count: Number of active adapters
    """
    _init_metrics()
    KM_ACTIVE_ADAPTERS.set(count)


def sync_km_metrics_to_prometheus() -> None:
    """Sync KMMetrics to Prometheus metrics.

    Reads the current state from the global KMMetrics instance and
    updates Prometheus metrics accordingly. Call this periodically
    (e.g., every 30 seconds) to keep Prometheus in sync.
    """
    _init_metrics()

    try:
        from aragora.knowledge.mound.metrics import get_metrics, HealthStatus

        km_metrics = get_metrics()
        health = km_metrics.get_health()

        # Map health status to numeric value
        status_map = {
            HealthStatus.UNKNOWN: 0,
            HealthStatus.UNHEALTHY: 1,
            HealthStatus.DEGRADED: 2,
            HealthStatus.HEALTHY: 3,
        }
        set_km_health_status(status_map.get(health.status, 0))

        # Sync operation stats
        stats = km_metrics.get_stats()
        for op_name, op_stats in stats.items():
            # The stats are already aggregated, so we just ensure they're recorded
            # Note: Prometheus counters are cumulative, so we'd need to track deltas
            # For simplicity, health status and gauges are the main sync targets
            pass

    except ImportError:
        logger.debug("KMMetrics not available for Prometheus sync")
    except (KeyError, AttributeError) as e:
        logger.debug(f"KM metrics data extraction failed: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error syncing KM metrics to Prometheus: {e}")


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
    """Record a notification delivery attempt.

    Args:
        channel: Notification channel (slack, email, webhook, in_app)
        severity: Notification severity (info, warning, error, critical)
        priority: Notification priority (low, normal, high, urgent)
        success: Whether the delivery succeeded
        latency_seconds: Delivery latency in seconds
    """
    _init_metrics()
    status = "success" if success else "failed"
    NOTIFICATION_SENT_TOTAL.labels(
        channel=channel, severity=severity, priority=priority, status=status
    ).inc()
    NOTIFICATION_LATENCY.labels(channel=channel).observe(latency_seconds)


def record_notification_error(channel: str, error_type: str) -> None:
    """Record a notification delivery error.

    Args:
        channel: Notification channel (slack, email, webhook)
        error_type: Error category (timeout, auth_failed, rate_limited, connection_error, etc.)
    """
    _init_metrics()
    NOTIFICATION_ERRORS_TOTAL.labels(channel=channel, error_type=error_type).inc()


def set_notification_queue_size(channel: str, size: int) -> None:
    """Set the current notification queue size.

    Args:
        channel: Notification channel
        size: Current queue size
    """
    _init_metrics()
    NOTIFICATION_QUEUE_SIZE.labels(channel=channel).set(size)


@contextmanager
def track_notification_delivery(
    channel: str,
    severity: str = "info",
    priority: str = "normal",
) -> Generator[None, None, None]:
    """Context manager to track notification delivery.

    Automatically records latency and success/failure.

    Args:
        channel: Notification channel
        severity: Notification severity
        priority: Notification priority

    Example:
        with track_notification_delivery("slack", "warning", "high"):
            await send_slack_message(...)
    """
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
# Persistent Task Queue Metrics
# =============================================================================


def record_task_queue_operation(
    operation: str,
    success: bool,
    latency_seconds: float,
) -> None:
    """Record a task queue operation.

    Args:
        operation: Operation type (enqueue, dequeue, complete, fail, cancel)
        success: Whether the operation succeeded
        latency_seconds: Operation latency in seconds
    """
    _init_metrics()
    status = "success" if success else "error"
    TASK_QUEUE_OPERATIONS_TOTAL.labels(operation=operation, status=status).inc()
    TASK_QUEUE_OPERATION_LATENCY.labels(operation=operation).observe(latency_seconds)


def set_task_queue_size(pending: int, ready: int, running: int) -> None:
    """Set the current task queue sizes by status.

    Args:
        pending: Number of pending tasks
        ready: Number of ready tasks
        running: Number of running tasks
    """
    _init_metrics()
    TASK_QUEUE_SIZE.labels(status="pending").set(pending)
    TASK_QUEUE_SIZE.labels(status="ready").set(ready)
    TASK_QUEUE_SIZE.labels(status="running").set(running)


def record_task_queue_recovery(original_status: str, count: int = 1) -> None:
    """Record recovered tasks on startup.

    Args:
        original_status: Original status of recovered task (pending, ready, running)
        count: Number of tasks recovered
    """
    _init_metrics()
    TASK_QUEUE_RECOVERED_TOTAL.labels(original_status=original_status).inc(count)


def record_task_queue_cleanup(count: int) -> None:
    """Record completed tasks cleaned up.

    Args:
        count: Number of tasks cleaned up
    """
    _init_metrics()
    TASK_QUEUE_CLEANUP_TOTAL.inc(count)


@contextmanager
def track_task_queue_operation(operation: str) -> Generator[None, None, None]:
    """Context manager to track task queue operations.

    Automatically records latency and success/failure.

    Args:
        operation: Operation type (enqueue, dequeue, complete, fail)

    Example:
        with track_task_queue_operation("enqueue"):
            await queue.enqueue(task)
    """
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
        record_task_queue_operation(operation, success, latency)


# =============================================================================
# Governance Store Metrics
# =============================================================================


def record_governance_decision(decision_type: str, outcome: str) -> None:
    """Record a governance decision stored.

    Args:
        decision_type: Type of decision (manual, auto)
        outcome: Decision outcome (approved, rejected)
    """
    _init_metrics()
    GOVERNANCE_DECISIONS_TOTAL.labels(decision_type=decision_type, outcome=outcome).inc()


def record_governance_verification(verification_type: str, result: str) -> None:
    """Record a verification stored.

    Args:
        verification_type: Type of verification (formal, runtime)
        result: Verification result (valid, invalid)
    """
    _init_metrics()
    GOVERNANCE_VERIFICATIONS_TOTAL.labels(
        verification_type=verification_type, result=result
    ).inc()


def record_governance_approval(approval_type: str, status: str) -> None:
    """Record an approval stored.

    Args:
        approval_type: Type of approval (nomic, deploy, change)
        status: Approval status (granted, revoked)
    """
    _init_metrics()
    GOVERNANCE_APPROVALS_TOTAL.labels(approval_type=approval_type, status=status).inc()


def record_governance_store_latency(operation: str, latency_seconds: float) -> None:
    """Record governance store operation latency.

    Args:
        operation: Operation type (save, get, list, delete)
        latency_seconds: Operation latency in seconds
    """
    _init_metrics()
    GOVERNANCE_STORE_LATENCY.labels(operation=operation).observe(latency_seconds)


def set_governance_artifacts_active(
    decisions: int, verifications: int, approvals: int
) -> None:
    """Set the current number of active governance artifacts.

    Args:
        decisions: Number of active decisions
        verifications: Number of active verifications
        approvals: Number of active approvals
    """
    _init_metrics()
    GOVERNANCE_ARTIFACTS_ACTIVE.labels(artifact_type="decision").set(decisions)
    GOVERNANCE_ARTIFACTS_ACTIVE.labels(artifact_type="verification").set(verifications)
    GOVERNANCE_ARTIFACTS_ACTIVE.labels(artifact_type="approval").set(approvals)


@contextmanager
def track_governance_store_operation(operation: str) -> Generator[None, None, None]:
    """Context manager to track governance store operations.

    Automatically records latency.

    Args:
        operation: Operation type (save, get, list, delete)

    Example:
        with track_governance_store_operation("save"):
            await store.save_verification(...)
    """
    _init_metrics()
    start = time.perf_counter()
    try:
        yield
    finally:
        latency = time.perf_counter() - start
        record_governance_store_latency(operation, latency)


# =============================================================================
# User ID Mapping Metrics
# =============================================================================


def record_user_mapping_operation(
    operation: str, platform: str, found: bool
) -> None:
    """Record a user ID mapping operation.

    Args:
        operation: Operation type (save, get, delete)
        platform: Platform name (slack, discord, teams)
        found: Whether the mapping was found (for get operations)
    """
    _init_metrics()
    status = "success" if found else "not_found"
    USER_MAPPING_OPERATIONS_TOTAL.labels(
        operation=operation, platform=platform, status=status
    ).inc()


def record_user_mapping_cache_hit(platform: str) -> None:
    """Record a user ID mapping cache hit.

    Args:
        platform: Platform name (slack, discord, teams)
    """
    _init_metrics()
    USER_MAPPING_CACHE_HITS_TOTAL.labels(platform=platform).inc()


def record_user_mapping_cache_miss(platform: str) -> None:
    """Record a user ID mapping cache miss.

    Args:
        platform: Platform name (slack, discord, teams)
    """
    _init_metrics()
    USER_MAPPING_CACHE_MISSES_TOTAL.labels(platform=platform).inc()


def set_user_mappings_active(platform: str, count: int) -> None:
    """Set the number of active user ID mappings for a platform.

    Args:
        platform: Platform name (slack, discord, teams)
        count: Number of active mappings
    """
    _init_metrics()
    USER_MAPPINGS_ACTIVE.labels(platform=platform).set(count)


# =============================================================================
# Gauntlet Export Metrics
# =============================================================================

# Metric instances (will be set during initialization)
GAUNTLET_EXPORTS_TOTAL: Any = None
GAUNTLET_EXPORT_LATENCY: Any = None
GAUNTLET_EXPORT_SIZE: Any = None


def _init_gauntlet_metrics() -> None:
    """Initialize Gauntlet export metrics."""
    global GAUNTLET_EXPORTS_TOTAL, GAUNTLET_EXPORT_LATENCY, GAUNTLET_EXPORT_SIZE

    config = get_metrics_config()
    if not config.enabled:
        _init_gauntlet_noop_metrics()
        return

    try:
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

    except ImportError:
        _init_gauntlet_noop_metrics()


def _init_gauntlet_noop_metrics() -> None:
    """Initialize no-op Gauntlet metrics."""
    global GAUNTLET_EXPORTS_TOTAL, GAUNTLET_EXPORT_LATENCY, GAUNTLET_EXPORT_SIZE

    class NoOpMetric:
        def labels(self, *args: Any, **kwargs: Any) -> "NoOpMetric":
            return self

        def inc(self, amount: float = 1) -> None:
            pass

        def observe(self, value: float) -> None:
            pass

    GAUNTLET_EXPORTS_TOTAL = NoOpMetric()
    GAUNTLET_EXPORT_LATENCY = NoOpMetric()
    GAUNTLET_EXPORT_SIZE = NoOpMetric()


def record_gauntlet_export(
    format: str,
    export_type: str,
    success: bool,
    latency_seconds: float,
    size_bytes: int = 0,
) -> None:
    """Record a Gauntlet export operation.

    Args:
        format: Export format (json, csv, html, markdown, sarif)
        export_type: Type of export (receipt, heatmap, bundle)
        success: Whether the export succeeded
        latency_seconds: Operation latency in seconds
        size_bytes: Size of exported content in bytes
    """
    _init_metrics()
    if GAUNTLET_EXPORTS_TOTAL is None:
        _init_gauntlet_metrics()

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
    """Context manager to track Gauntlet export operations.

    Args:
        format: Export format (json, csv, html, markdown, sarif)
        export_type: Type of export (receipt, heatmap, bundle)

    Example:
        with track_gauntlet_export("json", "receipt") as ctx:
            result = export_receipt(receipt, format=ReceiptExportFormat.JSON)
            ctx["size_bytes"] = len(result)
    """
    _init_metrics()
    if GAUNTLET_EXPORTS_TOTAL is None:
        _init_gauntlet_metrics()

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

WORKFLOW_TEMPLATES_CREATED: Any = None
WORKFLOW_TEMPLATE_EXECUTIONS: Any = None
WORKFLOW_TEMPLATE_EXECUTION_LATENCY: Any = None


def _init_workflow_metrics() -> None:
    """Initialize workflow template metrics."""
    global WORKFLOW_TEMPLATES_CREATED, WORKFLOW_TEMPLATE_EXECUTIONS
    global WORKFLOW_TEMPLATE_EXECUTION_LATENCY

    config = get_metrics_config()
    if not config.enabled:
        _init_workflow_noop_metrics()
        return

    try:
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

    except ImportError:
        _init_workflow_noop_metrics()


def _init_workflow_noop_metrics() -> None:
    """Initialize no-op workflow metrics."""
    global WORKFLOW_TEMPLATES_CREATED, WORKFLOW_TEMPLATE_EXECUTIONS
    global WORKFLOW_TEMPLATE_EXECUTION_LATENCY

    class NoOpMetric:
        def labels(self, *args: Any, **kwargs: Any) -> "NoOpMetric":
            return self

        def inc(self, amount: float = 1) -> None:
            pass

        def observe(self, value: float) -> None:
            pass

    WORKFLOW_TEMPLATES_CREATED = NoOpMetric()
    WORKFLOW_TEMPLATE_EXECUTIONS = NoOpMetric()
    WORKFLOW_TEMPLATE_EXECUTION_LATENCY = NoOpMetric()


def record_workflow_template_created(pattern: str, template_id: str) -> None:
    """Record a workflow template creation.

    Args:
        pattern: Workflow pattern (hive_mind, map_reduce, review_cycle)
        template_id: Template identifier
    """
    _init_metrics()
    if WORKFLOW_TEMPLATES_CREATED is None:
        _init_workflow_metrics()

    WORKFLOW_TEMPLATES_CREATED.labels(pattern=pattern, template_id=template_id).inc()


def record_workflow_template_execution(
    pattern: str,
    success: bool,
    latency_seconds: float,
) -> None:
    """Record a workflow template execution.

    Args:
        pattern: Workflow pattern (hive_mind, map_reduce, review_cycle)
        success: Whether the execution succeeded
        latency_seconds: Execution latency in seconds
    """
    _init_metrics()
    if WORKFLOW_TEMPLATE_EXECUTIONS is None:
        _init_workflow_metrics()

    status = "success" if success else "error"
    WORKFLOW_TEMPLATE_EXECUTIONS.labels(pattern=pattern, status=status).inc()
    WORKFLOW_TEMPLATE_EXECUTION_LATENCY.labels(pattern=pattern).observe(latency_seconds)


@contextmanager
def track_workflow_template_execution(pattern: str) -> Generator[None, None, None]:
    """Context manager to track workflow template execution.

    Args:
        pattern: Workflow pattern (hive_mind, map_reduce, review_cycle)

    Example:
        with track_workflow_template_execution("hive_mind"):
            await workflow.execute()
    """
    _init_metrics()
    if WORKFLOW_TEMPLATE_EXECUTIONS is None:
        _init_workflow_metrics()

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
# Checkpoint Store Metrics
# =============================================================================

CHECKPOINT_OPERATIONS: Any = None
CHECKPOINT_OPERATION_LATENCY: Any = None
CHECKPOINT_SIZE: Any = None
CHECKPOINT_RESTORE_RESULTS: Any = None


def _init_checkpoint_metrics() -> None:
    """Initialize checkpoint store metrics."""
    global CHECKPOINT_OPERATIONS, CHECKPOINT_OPERATION_LATENCY
    global CHECKPOINT_SIZE, CHECKPOINT_RESTORE_RESULTS

    config = get_metrics_config()
    if not config.enabled:
        _init_checkpoint_noop_metrics()
        return

    try:
        from prometheus_client import Counter, Histogram

        CHECKPOINT_OPERATIONS = Counter(
            "aragora_checkpoint_operations_total",
            "Total checkpoint operations",
            ["operation", "status"],
        )

        CHECKPOINT_OPERATION_LATENCY = Histogram(
            "aragora_checkpoint_operation_latency_seconds",
            "Checkpoint operation latency",
            ["operation"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
        )

        CHECKPOINT_SIZE = Histogram(
            "aragora_checkpoint_size_bytes",
            "Checkpoint file size in bytes",
            buckets=[1000, 10000, 100000, 1000000, 10000000, 100000000],
        )

        CHECKPOINT_RESTORE_RESULTS = Counter(
            "aragora_checkpoint_restore_results_total",
            "Checkpoint restore results",
            ["result"],
        )

    except ImportError:
        _init_checkpoint_noop_metrics()


def _init_checkpoint_noop_metrics() -> None:
    """Initialize no-op checkpoint metrics."""
    global CHECKPOINT_OPERATIONS, CHECKPOINT_OPERATION_LATENCY
    global CHECKPOINT_SIZE, CHECKPOINT_RESTORE_RESULTS

    class NoOpMetric:
        def labels(self, *args: Any, **kwargs: Any) -> "NoOpMetric":
            return self

        def inc(self, amount: float = 1) -> None:
            pass

        def observe(self, value: float) -> None:
            pass

    CHECKPOINT_OPERATIONS = NoOpMetric()
    CHECKPOINT_OPERATION_LATENCY = NoOpMetric()
    CHECKPOINT_SIZE = NoOpMetric()
    CHECKPOINT_RESTORE_RESULTS = NoOpMetric()


def record_checkpoint_operation(
    operation: str,
    success: bool,
    latency_seconds: float,
    size_bytes: int = 0,
) -> None:
    """Record a checkpoint operation.

    Args:
        operation: Operation type (create, restore, delete, list, compare)
        success: Whether the operation succeeded
        latency_seconds: Operation latency in seconds
        size_bytes: Checkpoint size in bytes (for create operations)
    """
    _init_metrics()
    if CHECKPOINT_OPERATIONS is None:
        _init_checkpoint_metrics()

    status = "success" if success else "error"
    CHECKPOINT_OPERATIONS.labels(operation=operation, status=status).inc()
    CHECKPOINT_OPERATION_LATENCY.labels(operation=operation).observe(latency_seconds)
    if size_bytes > 0:
        CHECKPOINT_SIZE.observe(size_bytes)


def record_checkpoint_restore_result(
    nodes_restored: int,
    nodes_skipped: int,
    errors: int,
) -> None:
    """Record checkpoint restore results.

    Args:
        nodes_restored: Number of nodes successfully restored
        nodes_skipped: Number of nodes skipped (duplicates)
        errors: Number of errors during restore
    """
    _init_metrics()
    if CHECKPOINT_RESTORE_RESULTS is None:
        _init_checkpoint_metrics()

    if nodes_restored > 0:
        CHECKPOINT_RESTORE_RESULTS.labels(result="nodes_restored").inc(nodes_restored)
    if nodes_skipped > 0:
        CHECKPOINT_RESTORE_RESULTS.labels(result="nodes_skipped").inc(nodes_skipped)
    if errors > 0:
        CHECKPOINT_RESTORE_RESULTS.labels(result="errors").inc(errors)


@contextmanager
def track_checkpoint_operation(operation: str) -> Generator[dict, None, None]:
    """Context manager to track checkpoint operations.

    Args:
        operation: Operation type (create, restore, delete, list, compare)

    Example:
        with track_checkpoint_operation("create") as ctx:
            checkpoint = store.create_checkpoint("my_checkpoint")
            ctx["size_bytes"] = checkpoint.size_bytes
    """
    _init_metrics()
    if CHECKPOINT_OPERATIONS is None:
        _init_checkpoint_metrics()

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
        record_checkpoint_operation(operation, success, latency, ctx.get("size_bytes", 0))


# =============================================================================
# Consensus Ingestion Metrics
# =============================================================================

CONSENSUS_INGESTION_TOTAL: Any = None
CONSENSUS_INGESTION_LATENCY: Any = None
CONSENSUS_INGESTION_CLAIMS: Any = None


def _init_consensus_ingestion_metrics() -> None:
    """Initialize consensus ingestion metrics."""
    global CONSENSUS_INGESTION_TOTAL, CONSENSUS_INGESTION_LATENCY
    global CONSENSUS_INGESTION_CLAIMS

    config = get_metrics_config()
    if not config.enabled:
        _init_consensus_ingestion_noop_metrics()
        return

    try:
        from prometheus_client import Counter, Histogram

        CONSENSUS_INGESTION_TOTAL = Counter(
            "aragora_consensus_ingestion_total",
            "Total consensus ingestion events",
            ["strength", "tier", "status"],
        )

        CONSENSUS_INGESTION_LATENCY = Histogram(
            "aragora_consensus_ingestion_latency_seconds",
            "Consensus ingestion latency",
            ["strength"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        )

        CONSENSUS_INGESTION_CLAIMS = Counter(
            "aragora_consensus_ingestion_claims_total",
            "Total key claims ingested from consensus",
            ["tier"],
        )

    except ImportError:
        _init_consensus_ingestion_noop_metrics()


def _init_consensus_ingestion_noop_metrics() -> None:
    """Initialize no-op consensus ingestion metrics."""
    global CONSENSUS_INGESTION_TOTAL, CONSENSUS_INGESTION_LATENCY
    global CONSENSUS_INGESTION_CLAIMS

    class NoOpMetric:
        def labels(self, *args: Any, **kwargs: Any) -> "NoOpMetric":
            return self

        def inc(self, amount: float = 1) -> None:
            pass

        def observe(self, value: float) -> None:
            pass

    CONSENSUS_INGESTION_TOTAL = NoOpMetric()
    CONSENSUS_INGESTION_LATENCY = NoOpMetric()
    CONSENSUS_INGESTION_CLAIMS = NoOpMetric()


def record_consensus_ingestion(
    strength: str,
    tier: str,
    success: bool,
    latency_seconds: float,
    claims_count: int = 0,
) -> None:
    """Record a consensus ingestion event.

    Args:
        strength: Consensus strength (unanimous, strong, moderate, weak, split, contested)
        tier: KM tier used (glacial, slow, medium, fast)
        success: Whether the ingestion succeeded
        latency_seconds: Ingestion latency in seconds
        claims_count: Number of key claims ingested
    """
    _init_metrics()
    if CONSENSUS_INGESTION_TOTAL is None:
        _init_consensus_ingestion_metrics()

    status = "success" if success else "error"
    CONSENSUS_INGESTION_TOTAL.labels(strength=strength, tier=tier, status=status).inc()
    CONSENSUS_INGESTION_LATENCY.labels(strength=strength).observe(latency_seconds)
    if claims_count > 0:
        CONSENSUS_INGESTION_CLAIMS.labels(tier=tier).inc(claims_count)


# =============================================================================
# Enhanced Consensus Ingestion Metrics (Dissent, Evolution, Linking)
# =============================================================================

CONSENSUS_DISSENT_INGESTED: Any = None
CONSENSUS_EVOLUTION_TRACKED: Any = None
CONSENSUS_EVIDENCE_LINKED: Any = None
CONSENSUS_AGREEMENT_RATIO: Any = None


def _init_enhanced_consensus_metrics() -> None:
    """Initialize enhanced consensus ingestion metrics for dissent, evolution, and linking."""
    global CONSENSUS_DISSENT_INGESTED, CONSENSUS_EVOLUTION_TRACKED
    global CONSENSUS_EVIDENCE_LINKED, CONSENSUS_AGREEMENT_RATIO

    config = get_metrics_config()
    if not config.enabled:
        _init_enhanced_consensus_noop_metrics()
        return

    try:
        from prometheus_client import Counter, Histogram

        CONSENSUS_DISSENT_INGESTED = Counter(
            "aragora_consensus_dissent_ingested_total",
            "Total dissenting views ingested from consensus debates",
            ["dissent_type", "acknowledged"],
        )

        CONSENSUS_EVOLUTION_TRACKED = Counter(
            "aragora_consensus_evolution_tracked_total",
            "Total consensus evolution events (supersedes relationships)",
            ["evolution_type"],  # new_supersedes, found_similar, no_evolution
        )

        CONSENSUS_EVIDENCE_LINKED = Counter(
            "aragora_consensus_evidence_linked_total",
            "Total evidence items linked to consensus nodes",
            ["tier"],
        )

        CONSENSUS_AGREEMENT_RATIO = Histogram(
            "aragora_consensus_agreement_ratio",
            "Distribution of agreement ratios in ingested consensus",
            ["strength"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )

    except ImportError:
        _init_enhanced_consensus_noop_metrics()


def _init_enhanced_consensus_noop_metrics() -> None:
    """Initialize no-op enhanced consensus metrics."""
    global CONSENSUS_DISSENT_INGESTED, CONSENSUS_EVOLUTION_TRACKED
    global CONSENSUS_EVIDENCE_LINKED, CONSENSUS_AGREEMENT_RATIO

    class NoOpMetric:
        def labels(self, *args: Any, **kwargs: Any) -> "NoOpMetric":
            return self

        def inc(self, amount: float = 1) -> None:
            pass

        def observe(self, value: float) -> None:
            pass

    CONSENSUS_DISSENT_INGESTED = NoOpMetric()
    CONSENSUS_EVOLUTION_TRACKED = NoOpMetric()
    CONSENSUS_EVIDENCE_LINKED = NoOpMetric()
    CONSENSUS_AGREEMENT_RATIO = NoOpMetric()


def record_consensus_dissent(
    dissent_type: str,
    acknowledged: bool = False,
    count: int = 1,
) -> None:
    """Record ingestion of dissenting views from consensus.

    Args:
        dissent_type: Type of dissent (risk_warning, fundamental_disagreement, etc.)
        acknowledged: Whether the dissent was acknowledged by majority
        count: Number of dissents ingested (default: 1)
    """
    _init_metrics()
    if CONSENSUS_DISSENT_INGESTED is None:
        _init_enhanced_consensus_metrics()

    CONSENSUS_DISSENT_INGESTED.labels(
        dissent_type=dissent_type,
        acknowledged="true" if acknowledged else "false",
    ).inc(count)


def record_consensus_evolution(
    evolution_type: str,
) -> None:
    """Record consensus evolution tracking event.

    Args:
        evolution_type: Type of evolution:
            - new_supersedes: New consensus supersedes an existing one
            - found_similar: Found similar prior consensus (potential supersedes)
            - no_evolution: No evolution relationship detected
    """
    _init_metrics()
    if CONSENSUS_EVOLUTION_TRACKED is None:
        _init_enhanced_consensus_metrics()

    CONSENSUS_EVOLUTION_TRACKED.labels(evolution_type=evolution_type).inc()


def record_consensus_evidence_linked(
    tier: str,
    count: int = 1,
) -> None:
    """Record evidence items linked to consensus.

    Args:
        tier: KM tier for the evidence
        count: Number of evidence items linked
    """
    _init_metrics()
    if CONSENSUS_EVIDENCE_LINKED is None:
        _init_enhanced_consensus_metrics()

    CONSENSUS_EVIDENCE_LINKED.labels(tier=tier).inc(count)


def record_consensus_agreement_ratio(
    strength: str,
    agreement_ratio: float,
) -> None:
    """Record the agreement ratio for a consensus.

    Args:
        strength: Consensus strength
        agreement_ratio: Ratio of agreeing agents to total participants (0.0-1.0)
    """
    _init_metrics()
    if CONSENSUS_AGREEMENT_RATIO is None:
        _init_enhanced_consensus_metrics()

    CONSENSUS_AGREEMENT_RATIO.labels(strength=strength).observe(agreement_ratio)


def record_km_inbound_event(
    event_type: str,
    source: str,
    success: bool = True,
) -> None:
    """Record an inbound event to Knowledge Mound.

    This is a general-purpose metric for tracking all events flowing into KM.

    Args:
        event_type: Type of event (consensus, belief, elo, insight, etc.)
        source: Source of the event (debate_orchestrator, belief_network, etc.)
        success: Whether the event was processed successfully
    """
    _init_metrics()
    # Uses existing KM metrics
    record_km_operation("ingest", success, 0.0)
    record_km_event_emitted(f"inbound_{event_type}")


# =============================================================================
# Explainability API Metrics
# =============================================================================

EXPLAINABILITY_REQUESTS: Any = None
EXPLAINABILITY_LATENCY: Any = None
EXPLAINABILITY_FACTORS: Any = None
EXPLAINABILITY_COUNTERFACTUALS: Any = None


def _init_explainability_metrics() -> None:
    """Initialize explainability API metrics."""
    global EXPLAINABILITY_REQUESTS, EXPLAINABILITY_LATENCY
    global EXPLAINABILITY_FACTORS, EXPLAINABILITY_COUNTERFACTUALS

    config = get_metrics_config()
    if not config.enabled:
        _init_explainability_noop_metrics()
        return

    try:
        from prometheus_client import Counter, Histogram

        EXPLAINABILITY_REQUESTS = Counter(
            "aragora_explainability_requests_total",
            "Total explainability API requests",
            ["endpoint", "status"],
        )

        EXPLAINABILITY_LATENCY = Histogram(
            "aragora_explainability_latency_seconds",
            "Explainability API latency",
            ["endpoint"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
        )

        EXPLAINABILITY_FACTORS = Histogram(
            "aragora_explainability_factors_count",
            "Number of factors returned in explainability response",
            buckets=[1, 3, 5, 10, 15, 20, 30, 50],
        )

        EXPLAINABILITY_COUNTERFACTUALS = Histogram(
            "aragora_explainability_counterfactuals_count",
            "Number of counterfactuals generated",
            buckets=[1, 2, 3, 5, 10, 15, 20],
        )

    except ImportError:
        _init_explainability_noop_metrics()


def _init_explainability_noop_metrics() -> None:
    """Initialize no-op explainability metrics."""
    global EXPLAINABILITY_REQUESTS, EXPLAINABILITY_LATENCY
    global EXPLAINABILITY_FACTORS, EXPLAINABILITY_COUNTERFACTUALS

    class NoOpMetric:
        def labels(self, *args: Any, **kwargs: Any) -> "NoOpMetric":
            return self

        def inc(self, amount: float = 1) -> None:
            pass

        def observe(self, value: float) -> None:
            pass

    EXPLAINABILITY_REQUESTS = NoOpMetric()
    EXPLAINABILITY_LATENCY = NoOpMetric()
    EXPLAINABILITY_FACTORS = NoOpMetric()
    EXPLAINABILITY_COUNTERFACTUALS = NoOpMetric()


def record_explainability_request(
    endpoint: str,
    success: bool,
    latency_seconds: float,
    factors_count: int = 0,
    counterfactuals_count: int = 0,
) -> None:
    """Record an explainability API request.

    Args:
        endpoint: API endpoint (full, factors, counterfactual, provenance, narrative)
        success: Whether the request succeeded
        latency_seconds: Request latency in seconds
        factors_count: Number of factors returned (for factors endpoint)
        counterfactuals_count: Number of counterfactuals generated
    """
    _init_metrics()
    if EXPLAINABILITY_REQUESTS is None:
        _init_explainability_metrics()

    status = "success" if success else "error"
    EXPLAINABILITY_REQUESTS.labels(endpoint=endpoint, status=status).inc()
    EXPLAINABILITY_LATENCY.labels(endpoint=endpoint).observe(latency_seconds)
    if factors_count > 0:
        EXPLAINABILITY_FACTORS.observe(factors_count)
    if counterfactuals_count > 0:
        EXPLAINABILITY_COUNTERFACTUALS.observe(counterfactuals_count)


@contextmanager
def track_explainability_request(endpoint: str) -> Generator[dict, None, None]:
    """Context manager to track explainability API requests.

    Args:
        endpoint: API endpoint name

    Example:
        with track_explainability_request("factors") as ctx:
            factors = await get_factors(debate_id)
            ctx["factors_count"] = len(factors)
    """
    _init_metrics()
    if EXPLAINABILITY_REQUESTS is None:
        _init_explainability_metrics()

    start = time.perf_counter()
    ctx: dict[str, Any] = {"factors_count": 0, "counterfactuals_count": 0}
    success = True
    try:
        yield ctx
    except Exception:
        success = False
        raise
    finally:
        latency = time.perf_counter() - start
        record_explainability_request(
            endpoint, success, latency,
            ctx.get("factors_count", 0),
            ctx.get("counterfactuals_count", 0),
        )


# =============================================================================
# KM Resilience Metrics
# =============================================================================

KM_CIRCUIT_BREAKER_STATE: Any = None
KM_CIRCUIT_BREAKER_TRIPS: Any = None
KM_RETRY_TOTAL: Any = None
KM_RETRY_LATENCY: Any = None
KM_CACHE_INVALIDATION: Any = None
KM_CONNECTION_POOL_USAGE: Any = None
KM_INTEGRITY_CHECK_ERRORS: Any = None
KM_INTEGRITY_CHECK_REPAIRS: Any = None


def _init_km_resilience_metrics() -> None:
    """Initialize KM resilience metrics."""
    global KM_CIRCUIT_BREAKER_STATE, KM_CIRCUIT_BREAKER_TRIPS
    global KM_RETRY_TOTAL, KM_RETRY_LATENCY
    global KM_CACHE_INVALIDATION, KM_CONNECTION_POOL_USAGE
    global KM_INTEGRITY_CHECK_ERRORS, KM_INTEGRITY_CHECK_REPAIRS

    config = get_metrics_config()
    if not config.enabled:
        _init_km_resilience_noop_metrics()
        return

    try:
        from prometheus_client import Counter, Gauge, Histogram

        KM_CIRCUIT_BREAKER_STATE = Gauge(
            "aragora_km_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=half_open, 2=open)",
            ["service"],
        )

        KM_CIRCUIT_BREAKER_TRIPS = Counter(
            "aragora_km_circuit_breaker_trips_total",
            "Total circuit breaker trips",
            ["service"],
        )

        KM_RETRY_TOTAL = Counter(
            "aragora_km_retry_total",
            "Total retry operations",
            ["outcome"],  # success, exhausted, error
        )

        KM_RETRY_LATENCY = Histogram(
            "aragora_km_retry_latency_seconds",
            "Retry operation latency (including all retries)",
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
        )

        KM_CACHE_INVALIDATION = Counter(
            "aragora_km_cache_invalidation_total",
            "Total cache invalidation events",
            ["event_type"],  # node_updated, node_deleted, bulk_refresh
        )

        KM_CONNECTION_POOL_USAGE = Gauge(
            "aragora_km_connection_pool_usage_ratio",
            "Connection pool usage ratio (0.0-1.0)",
        )

        KM_INTEGRITY_CHECK_ERRORS = Counter(
            "aragora_km_integrity_check_errors_total",
            "Total integrity check errors detected",
        )

        KM_INTEGRITY_CHECK_REPAIRS = Counter(
            "aragora_km_integrity_check_repairs_total",
            "Total integrity repairs applied",
        )

    except ImportError:
        _init_km_resilience_noop_metrics()


def _init_km_resilience_noop_metrics() -> None:
    """Initialize no-op KM resilience metrics."""
    global KM_CIRCUIT_BREAKER_STATE, KM_CIRCUIT_BREAKER_TRIPS
    global KM_RETRY_TOTAL, KM_RETRY_LATENCY
    global KM_CACHE_INVALIDATION, KM_CONNECTION_POOL_USAGE
    global KM_INTEGRITY_CHECK_ERRORS, KM_INTEGRITY_CHECK_REPAIRS

    class NoOpMetric:
        def labels(self, *args: Any, **kwargs: Any) -> "NoOpMetric":
            return self

        def inc(self, amount: float = 1) -> None:
            pass

        def set(self, value: float) -> None:
            pass

        def observe(self, value: float) -> None:
            pass

    KM_CIRCUIT_BREAKER_STATE = NoOpMetric()
    KM_CIRCUIT_BREAKER_TRIPS = NoOpMetric()
    KM_RETRY_TOTAL = NoOpMetric()
    KM_RETRY_LATENCY = NoOpMetric()
    KM_CACHE_INVALIDATION = NoOpMetric()
    KM_CONNECTION_POOL_USAGE = NoOpMetric()
    KM_INTEGRITY_CHECK_ERRORS = NoOpMetric()
    KM_INTEGRITY_CHECK_REPAIRS = NoOpMetric()


def set_km_circuit_breaker_state(service: str, state: str) -> None:
    """Set the circuit breaker state for a service.

    Args:
        service: Service name (postgres, redis)
        state: Circuit breaker state (closed, half_open, open)
    """
    _init_metrics()
    if KM_CIRCUIT_BREAKER_STATE is None:
        _init_km_resilience_metrics()

    state_map = {"closed": 0, "half_open": 1, "open": 2}
    KM_CIRCUIT_BREAKER_STATE.labels(service=service).set(state_map.get(state, 0))


def record_km_circuit_breaker_trip(service: str) -> None:
    """Record a circuit breaker trip.

    Args:
        service: Service name (postgres, redis)
    """
    _init_metrics()
    if KM_CIRCUIT_BREAKER_TRIPS is None:
        _init_km_resilience_metrics()

    KM_CIRCUIT_BREAKER_TRIPS.labels(service=service).inc()


def record_km_retry(outcome: str, latency_seconds: float) -> None:
    """Record a retry operation.

    Args:
        outcome: Retry outcome (success, exhausted, error)
        latency_seconds: Total latency including all retries
    """
    _init_metrics()
    if KM_RETRY_TOTAL is None:
        _init_km_resilience_metrics()

    KM_RETRY_TOTAL.labels(outcome=outcome).inc()
    KM_RETRY_LATENCY.observe(latency_seconds)


def record_km_cache_invalidation(event_type: str) -> None:
    """Record a cache invalidation event.

    Args:
        event_type: Type of invalidation (node_updated, node_deleted, bulk_refresh)
    """
    _init_metrics()
    if KM_CACHE_INVALIDATION is None:
        _init_km_resilience_metrics()

    KM_CACHE_INVALIDATION.labels(event_type=event_type).inc()


def set_km_connection_pool_usage(usage_ratio: float) -> None:
    """Set the connection pool usage ratio.

    Args:
        usage_ratio: Usage ratio (0.0-1.0)
    """
    _init_metrics()
    if KM_CONNECTION_POOL_USAGE is None:
        _init_km_resilience_metrics()

    KM_CONNECTION_POOL_USAGE.set(usage_ratio)


def record_km_integrity_error() -> None:
    """Record an integrity check error."""
    _init_metrics()
    if KM_INTEGRITY_CHECK_ERRORS is None:
        _init_km_resilience_metrics()

    KM_INTEGRITY_CHECK_ERRORS.inc()


def record_km_integrity_repair() -> None:
    """Record an integrity repair."""
    _init_metrics()
    if KM_INTEGRITY_CHECK_REPAIRS is None:
        _init_km_resilience_metrics()

    KM_INTEGRITY_CHECK_REPAIRS.inc()


@contextmanager
def track_km_retry(service: str = "postgres") -> Generator[None, None, None]:
    """Context manager to track KM retry operations.

    Args:
        service: Service name for circuit breaker tracking

    Example:
        with track_km_retry("postgres"):
            await resilient_store.query(...)
    """
    _init_metrics()
    if KM_RETRY_TOTAL is None:
        _init_km_resilience_metrics()

    start = time.perf_counter()
    outcome = "success"
    try:
        yield
    except Exception as e:
        outcome = "exhausted" if "retry" in str(e).lower() else "error"
        raise
    finally:
        latency = time.perf_counter() - start
        record_km_retry(outcome, latency)
