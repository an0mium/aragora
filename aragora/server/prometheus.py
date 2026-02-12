"""
Prometheus metrics for Aragora server.

Provides OpenMetrics-compliant metrics for monitoring:
- Debate operations (latency, token usage, outcomes)
- Agent performance (generation time, failures)
- HTTP request metrics (latency per endpoint)
- WebSocket connections
- Rate limiter state
- Cache statistics
"""

import logging
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Callable

logger = logging.getLogger(__name__)

# Try to import prometheus_client, fall back to simple implementation
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        REGISTRY,
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

# ============================================================================
# Metric Definitions (when prometheus_client is available)
# ============================================================================

if PROMETHEUS_AVAILABLE:
    # Debate metrics
    DEBATE_DURATION = Histogram(
        "aragora_debate_duration_seconds",
        "Time spent in debate execution",
        ["outcome", "agent_count"],
        buckets=[1, 5, 10, 30, 60, 120, 300, 600],
    )

    DEBATE_ROUNDS = Histogram(
        "aragora_debate_rounds_total",
        "Number of rounds per debate",
        ["outcome"],
        buckets=[1, 2, 3, 4, 5, 7, 10],
    )

    DEBATE_TOKENS = Counter(
        "aragora_debate_tokens_total",
        "Total tokens used in debates",
        ["model", "direction"],  # direction: input/output
    )

    DEBATES_TOTAL = Counter(
        "aragora_debates_total",
        "Total number of debates",
        ["outcome"],  # consensus, no_consensus, error, timeout
    )

    # Cost metrics
    COST_USD_TOTAL = Counter(
        "aragora_cost_usd_total",
        "Total cost in USD (multiply by 1e-6 for actual dollars)",
        ["provider", "model", "agent_id"],
    )

    COST_PER_DEBATE = Histogram(
        "aragora_debate_cost_usd",
        "Cost per debate in USD (multiply by 1e-6 for actual dollars)",
        ["provider"],
        buckets=[1000, 5000, 10000, 50000, 100000, 500000, 1000000],  # micro-dollars
    )

    BUDGET_UTILIZATION = Gauge(
        "aragora_budget_utilization_percent",
        "Current budget utilization percentage",
        ["workspace_id", "budget_type"],  # budget_type: daily, monthly, per_debate
    )

    # Agent metrics
    AGENT_GENERATION_DURATION = Histogram(
        "aragora_agent_generation_seconds",
        "Time spent generating agent responses",
        ["agent_type", "model"],
        buckets=[0.5, 1, 2, 5, 10, 30, 60],
    )

    AGENT_FAILURES = Counter(
        "aragora_agent_failures_total",
        "Total agent failures",
        ["agent_type", "error_type"],
    )

    AGENT_CIRCUIT_BREAKER = Gauge(
        "aragora_agent_circuit_breaker_state",
        "Circuit breaker state (0=closed, 1=open, 2=half-open)",
        ["agent_type"],
    )

    # HTTP metrics
    HTTP_REQUEST_DURATION = Histogram(
        "aragora_http_request_duration_seconds",
        "HTTP request latency",
        ["method", "endpoint", "status"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
    )

    HTTP_REQUESTS_TOTAL = Counter(
        "aragora_http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status"],
    )

    # WebSocket metrics
    WEBSOCKET_CONNECTIONS = Gauge(
        "aragora_websocket_connections_active",
        "Number of active WebSocket connections",
    )

    WEBSOCKET_MESSAGES = Counter(
        "aragora_websocket_messages_total",
        "Total WebSocket messages",
        ["direction", "message_type"],  # direction: sent/received
    )

    # Rate limiter metrics
    RATE_LIMIT_HITS = Counter(
        "aragora_rate_limit_hits_total",
        "Rate limit hits",
        ["limit_type"],  # token, ip
    )

    RATE_LIMIT_TOKENS_TRACKED = Gauge(
        "aragora_rate_limit_tokens_tracked",
        "Number of tokens being rate-limited",
    )

    # Cache metrics
    CACHE_SIZE = Gauge(
        "aragora_cache_size_entries",
        "Number of entries in cache",
        ["cache_name"],
    )

    CACHE_HITS = Counter(
        "aragora_cache_hits_total",
        "Cache hits",
        ["cache_name"],
    )

    CACHE_MISSES = Counter(
        "aragora_cache_misses_total",
        "Cache misses",
        ["cache_name"],
    )

    # System info
    ARAGORA_INFO = Info(
        "aragora",
        "Aragora server information",
    )

    # Database metrics
    DB_QUERY_DURATION = Histogram(
        "aragora_db_query_duration_seconds",
        "Database query execution time",
        ["operation", "table"],  # operation: select, insert, update, delete
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5],
    )

    DB_QUERY_TOTAL = Counter(
        "aragora_db_queries_total",
        "Total database queries",
        ["operation", "table"],
    )

    DB_ERRORS_TOTAL = Counter(
        "aragora_db_errors_total",
        "Total database errors",
        ["error_type", "operation"],
    )

    DB_CONNECTION_POOL_SIZE = Gauge(
        "aragora_db_connection_pool_size",
        "Database connection pool size",
        ["state"],  # active, idle
    )

    # Memory tier metrics
    MEMORY_TIER_SIZE = Gauge(
        "aragora_memory_tier_size",
        "Number of memories in each tier",
        ["tier"],  # fast, medium, slow, glacial
    )

    MEMORY_TIER_TRANSITIONS = Counter(
        "aragora_memory_tier_transitions_total",
        "Memory tier transitions",
        ["from_tier", "to_tier"],  # e.g., fast->medium, medium->slow
    )

    MEMORY_OPERATIONS = Counter(
        "aragora_memory_operations_total",
        "Memory operations by type",
        ["operation"],  # store, retrieve, consolidate, prune
    )

    # Nomic loop phase metrics
    NOMIC_PHASE_DURATION = Histogram(
        "aragora_nomic_phase_duration_seconds",
        "Time spent in each nomic loop phase",
        ["phase", "outcome"],  # phase: context, debate, design, implement, verify, commit
        buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1200],
    )

    NOMIC_PHASE_TOTAL = Counter(
        "aragora_nomic_phases_total",
        "Total nomic phases executed",
        ["phase", "outcome"],  # outcome: success, failure, skipped
    )

    NOMIC_CYCLE_DURATION = Histogram(
        "aragora_nomic_cycle_duration_seconds",
        "Total time for a complete nomic cycle",
        ["outcome"],  # outcome: success, failure, partial
        buckets=[60, 120, 300, 600, 1200, 1800, 3600],
    )

    NOMIC_CYCLE_TOTAL = Counter(
        "aragora_nomic_cycles_total",
        "Total nomic cycles executed",
        ["outcome"],
    )

    NOMIC_AGENT_PHASE_DURATION = Histogram(
        "aragora_nomic_agent_phase_seconds",
        "Time spent by each agent in a phase",
        ["phase", "agent"],
        buckets=[1, 5, 10, 30, 60, 120, 300],
    )

    # Control Plane metrics
    CONTROL_PLANE_TASKS_TOTAL = Counter(
        "aragora_control_plane_tasks_total",
        "Total tasks submitted to the control plane",
        ["task_type", "priority"],
    )

    CONTROL_PLANE_TASK_STATUS = Gauge(
        "aragora_control_plane_task_status_count",
        "Number of tasks by status",
        ["status"],  # pending, running, completed, failed, cancelled
    )

    CONTROL_PLANE_TASK_DURATION = Histogram(
        "aragora_control_plane_task_duration_seconds",
        "Task execution duration",
        ["task_type", "outcome"],  # outcome: completed, failed, timeout
        buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800],
    )

    CONTROL_PLANE_QUEUE_DEPTH = Gauge(
        "aragora_control_plane_queue_depth",
        "Number of tasks in queue by priority",
        ["priority"],  # low, normal, high, urgent
    )

    CONTROL_PLANE_AGENTS_REGISTERED = Gauge(
        "aragora_control_plane_agents_registered",
        "Number of registered agents",
        ["status"],  # available, busy, offline
    )

    CONTROL_PLANE_AGENT_HEALTH = Gauge(
        "aragora_control_plane_agent_health",
        "Agent health status (0=unhealthy, 1=degraded, 2=healthy)",
        ["agent_id"],
    )

    CONTROL_PLANE_AGENT_LATENCY = Histogram(
        "aragora_control_plane_agent_latency_seconds",
        "Agent health check latency",
        ["agent_id"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5],
    )

    CONTROL_PLANE_TASK_RETRIES = Counter(
        "aragora_control_plane_task_retries_total",
        "Total task retries",
        ["task_type", "reason"],  # reason: timeout, error, capability_mismatch
    )

    CONTROL_PLANE_DEAD_LETTER_QUEUE = Gauge(
        "aragora_control_plane_dead_letter_queue_size",
        "Number of tasks in dead letter queue",
    )

    CONTROL_PLANE_CLAIM_LATENCY = Histogram(
        "aragora_control_plane_claim_latency_seconds",
        "Time to claim a task from queue",
        ["priority"],
        buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5],
    )

    # Deliberation metrics
    DELIBERATION_DURATION = Histogram(
        "aragora_deliberation_duration_seconds",
        "Deliberation execution time",
        ["status", "consensus_reached"],
        buckets=[10, 30, 60, 120, 300, 600, 900, 1800],
    )

    DELIBERATION_SLA_COMPLIANCE = Counter(
        "aragora_deliberation_sla_total",
        "SLA compliance counts",
        ["level"],  # compliant, warning, critical, violated
    )

    DELIBERATION_CONSENSUS_CONFIDENCE = Histogram(
        "aragora_deliberation_consensus_confidence",
        "Distribution of consensus confidence scores",
        buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
    )

    DELIBERATION_ROUNDS = Histogram(
        "aragora_deliberation_rounds_total",
        "Number of rounds per deliberation",
        ["outcome"],
        buckets=[1, 2, 3, 4, 5, 7, 10, 15],
    )

    DELIBERATION_AGENTS = Histogram(
        "aragora_deliberation_agents_count",
        "Number of agents per deliberation",
        buckets=[2, 3, 4, 5, 6, 8, 10],
    )

    # External agent gateway metrics
    EXTERNAL_AGENT_TASKS_TOTAL = Counter(
        "aragora_external_agent_tasks_total",
        "Total external agent tasks",
        ["adapter", "status"],
    )

    EXTERNAL_AGENT_TASK_DURATION = Histogram(
        "aragora_external_agent_task_duration_seconds",
        "External agent task execution time",
        ["adapter", "task_type"],
        buckets=[1, 5, 10, 30, 60, 300, 600, 1800],
    )

    EXTERNAL_AGENT_TOKENS_TOTAL = Counter(
        "aragora_external_agent_tokens_total",
        "Tokens used by external agents",
        ["adapter"],
    )

    EXTERNAL_AGENT_TOOLS_BLOCKED = Counter(
        "aragora_external_agent_tools_blocked_total",
        "Tool calls blocked by security policy",
        ["adapter", "tool"],
    )

    EXTERNAL_AGENT_COST_TOTAL = Counter(
        "aragora_external_agent_cost_microdollars_total",
        "Cost in micro-dollars for external agent tasks",
        ["adapter"],
    )

    DELIBERATION_TOTAL = Counter(
        "aragora_deliberation_total",
        "Total deliberations",
        ["outcome"],  # consensus, no_consensus, failed, timeout
    )

    AGENT_UTILIZATION = Gauge(
        "aragora_agent_utilization_ratio",
        "Agent busy vs available ratio",
        ["agent_id"],
    )

    POLICY_DECISIONS = Counter(
        "aragora_policy_decisions_total",
        "Policy evaluation decisions",
        ["decision", "policy_type"],  # decision: allow, deny, warn
    )

    # RLM (Recursive Language Models) metrics
    RLM_COMPRESSIONS = Counter(
        "aragora_rlm_compressions_total",
        "Total RLM compression operations",
        ["source_type", "status"],
    )

    RLM_COMPRESSION_RATIO = Histogram(
        "aragora_rlm_compression_ratio",
        "Compression ratio (compressed/original tokens)",
        ["source_type"],
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )

    RLM_TOKENS_SAVED = Counter(
        "aragora_rlm_tokens_saved_total",
        "Total tokens saved through compression",
        ["source_type"],
    )

    RLM_COMPRESSION_DURATION = Histogram(
        "aragora_rlm_compression_duration_seconds",
        "Time taken for compression operations",
        ["source_type", "levels"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )

    RLM_QUERIES = Counter(
        "aragora_rlm_queries_total",
        "Total RLM context queries",
        ["query_type", "level"],
    )

    RLM_QUERY_DURATION = Histogram(
        "aragora_rlm_query_duration_seconds",
        "Time taken for context queries",
        ["query_type"],
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
    )

    RLM_CACHE_HITS = Counter(
        "aragora_rlm_cache_hits_total",
        "RLM compression cache hits",
    )

    RLM_CACHE_MISSES = Counter(
        "aragora_rlm_cache_misses_total",
        "RLM compression cache misses",
    )

    RLM_CONTEXT_LEVELS = Histogram(
        "aragora_rlm_context_levels",
        "Number of abstraction levels created",
        ["source_type"],
        buckets=[1, 2, 3, 4, 5],
    )

    RLM_MEMORY_USAGE = Gauge(
        "aragora_rlm_memory_bytes",
        "Memory used by RLM context cache",
    )

    # RLM Iterative refinement metrics (Prime Intellect alignment)
    RLM_REFINEMENT_ITERATIONS = Histogram(
        "aragora_rlm_refinement_iterations",
        "Number of refinement iterations until ready=True",
        ["strategy"],
        buckets=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )

    RLM_REFINEMENT_SUCCESS = Counter(
        "aragora_rlm_refinement_success_total",
        "Successful refinements (ready=True before max iterations)",
        ["strategy"],
    )

    RLM_REFINEMENT_DURATION = Histogram(
        "aragora_rlm_refinement_duration_seconds",
        "Total time for refinement loop (all iterations)",
        ["strategy"],
        buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
    )

    RLM_READY_FALSE_RATE = Counter(
        "aragora_rlm_ready_false_total",
        "Times LLM signaled ready=False (needs refinement)",
        ["iteration"],
    )

    # Knowledge Mound metrics
    KNOWLEDGE_VISIBILITY_CHANGES = Counter(
        "aragora_knowledge_visibility_changes_total",
        "Visibility level changes on knowledge items",
        ["from_level", "to_level", "workspace_id"],
    )

    KNOWLEDGE_ACCESS_GRANTS = Counter(
        "aragora_knowledge_access_grants_total",
        "Access grants created or revoked",
        ["action", "grantee_type", "workspace_id"],  # action: grant/revoke
    )

    KNOWLEDGE_SHARES = Counter(
        "aragora_knowledge_shares_total",
        "Knowledge sharing operations",
        ["action", "target_type"],  # action: share/accept/decline/revoke
    )

    KNOWLEDGE_SHARED_ITEMS = Gauge(
        "aragora_knowledge_shared_items_count",
        "Shared items pending acceptance per workspace",
        ["workspace_id"],
    )

    KNOWLEDGE_GLOBAL_FACTS = Counter(
        "aragora_knowledge_global_facts_total",
        "Global/verified facts stored or promoted",
        ["action"],  # action: stored/promoted/queried
    )

    KNOWLEDGE_GLOBAL_QUERIES = Counter(
        "aragora_knowledge_global_queries_total",
        "Queries against global knowledge",
        ["has_results"],  # has_results: true/false
    )

    KNOWLEDGE_FEDERATION_SYNCS = Counter(
        "aragora_knowledge_federation_syncs_total",
        "Federation sync operations",
        ["region_id", "direction", "status"],  # direction: push/pull
    )

    KNOWLEDGE_FEDERATION_NODES = Counter(
        "aragora_knowledge_federation_nodes_total",
        "Nodes synced via federation",
        ["region_id", "direction"],
    )

    KNOWLEDGE_FEDERATION_LATENCY = Histogram(
        "aragora_knowledge_federation_latency_seconds",
        "Federation sync operation latency",
        ["region_id", "direction"],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    )

    KNOWLEDGE_FEDERATION_REGIONS = Gauge(
        "aragora_knowledge_federation_regions_count",
        "Federated regions by status",
        ["status"],  # status: enabled/disabled/healthy/unhealthy
    )

    # ========================================================================
    # Cross-Pollination Event Metrics
    # ========================================================================

    CROSS_POLL_EVENTS_TOTAL = Counter(
        "aragora_cross_pollination_events_total",
        "Total cross-pollination events dispatched",
        ["event_type"],
    )

    CROSS_POLL_HANDLER_CALLS = Counter(
        "aragora_cross_pollination_handler_calls_total",
        "Total handler invocations",
        ["handler", "status"],  # status: success, failure, skipped
    )

    CROSS_POLL_HANDLER_DURATION = Histogram(
        "aragora_cross_pollination_handler_duration_seconds",
        "Handler execution time",
        ["handler"],
        buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5],
    )

    CROSS_POLL_CIRCUIT_BREAKER = Gauge(
        "aragora_cross_pollination_circuit_breaker_state",
        "Circuit breaker state (0=closed, 1=open)",
        ["handler"],
    )

    CROSS_POLL_SUBSCRIBERS = Gauge(
        "aragora_cross_pollination_subscribers",
        "Number of registered subscribers",
        ["event_type"],
    )

    # ========================================================================
    # Knowledge Mound Bidirectional Flow Metrics
    # ========================================================================

    KM_INBOUND_EVENTS = Counter(
        "aragora_km_inbound_events_total",
        "Events flowing INTO Knowledge Mound from other subsystems",
        ["source", "event_type"],  # source: memory, belief, rlm, ranking, insights, etc.
    )

    KM_OUTBOUND_EVENTS = Counter(
        "aragora_km_outbound_events_total",
        "Events flowing OUT of Knowledge Mound to other subsystems",
        ["target", "event_type"],  # target: memory, belief, debate, trickster, etc.
    )

    KM_ADAPTER_SYNC = Counter(
        "aragora_km_adapter_sync_total",
        "Adapter sync operations to/from Knowledge Mound",
        ["adapter", "direction", "status"],  # direction: to_mound, from_mound
    )

    KM_ADAPTER_SYNC_DURATION = Histogram(
        "aragora_km_adapter_sync_duration_seconds",
        "Adapter sync operation duration",
        ["adapter", "direction"],
        buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30],
    )

    KM_STALENESS_CHECKS = Counter(
        "aragora_km_staleness_checks_total",
        "Staleness check operations",
        ["workspace", "status"],  # status: completed, failed, skipped
    )

    KM_STALE_NODES_FOUND = Gauge(
        "aragora_km_stale_nodes_found",
        "Number of stale nodes found in last check",
        ["workspace"],
    )

    KM_NODES_BY_SOURCE = Gauge(
        "aragora_km_nodes_by_source",
        "Knowledge nodes by source type",
        ["source"],
    )

    # ========================================================================
    # Checkpoint Bridge metrics (unified recovery)
    # ========================================================================
    CHECKPOINT_BRIDGE_SAVES = Counter(
        "aragora_checkpoint_bridge_saves_total",
        "Total checkpoint bridge save operations",
        ["debate_id", "phase"],
    )

    CHECKPOINT_BRIDGE_RESTORES = Counter(
        "aragora_checkpoint_bridge_restores_total",
        "Total checkpoint bridge restore operations",
        ["status"],  # success, not_found, failed
    )

    CHECKPOINT_BRIDGE_MOLECULE_RECOVERIES = Counter(
        "aragora_checkpoint_bridge_molecule_recoveries_total",
        "Molecule recovery operations from checkpoints",
        ["status"],  # success, not_found, no_state
    )

    CHECKPOINT_BRIDGE_SAVE_DURATION = Histogram(
        "aragora_checkpoint_bridge_save_duration_seconds",
        "Checkpoint bridge save operation duration",
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    )

    # ========================================================================
    # Agent Channel metrics (peer-to-peer messaging)
    # ========================================================================
    AGENT_CHANNEL_MESSAGES = Counter(
        "aragora_agent_channel_messages_total",
        "Total messages sent through agent channels",
        ["message_type", "channel"],  # message_type: proposal, critique, query, signal
    )

    AGENT_CHANNEL_SETUPS = Counter(
        "aragora_agent_channel_setups_total",
        "Channel setup operations",
        ["status"],  # success, failed, disabled
    )

    AGENT_CHANNEL_TEARDOWNS = Counter(
        "aragora_agent_channel_teardowns_total",
        "Channel teardown operations",
    )

    AGENT_CHANNEL_ACTIVE = Gauge(
        "aragora_agent_channel_active",
        "Number of active agent channels",
    )

    AGENT_CHANNEL_HISTORY_SIZE = Histogram(
        "aragora_agent_channel_history_size",
        "Message history size at channel teardown",
        buckets=[5, 10, 25, 50, 100, 250, 500],
    )

    # ========================================================================
    # Session Management metrics (multi-channel sessions)
    # ========================================================================
    SESSION_CREATED = Counter(
        "aragora_session_created_total",
        "Sessions created",
        ["channel"],  # slack, telegram, whatsapp, api
    )

    SESSION_DEBATES_LINKED = Counter(
        "aragora_session_debates_linked_total",
        "Debates linked to sessions",
        ["channel"],
    )

    SESSION_HANDOFFS = Counter(
        "aragora_session_handoffs_total",
        "Session handoffs between channels",
        ["from_channel", "to_channel"],
    )

    SESSION_RESULT_ROUTES = Counter(
        "aragora_session_result_routes_total",
        "Debate results routed to sessions",
        ["channel", "status"],  # status: success, failed, no_channel
    )

    SESSION_ACTIVE = Gauge(
        "aragora_sessions_active",
        "Number of active sessions by channel",
        ["channel"],
    )

    # ========================================================================
    # V1 API Deprecation Metrics (sunset June 2026)
    # ========================================================================

    V1_API_REQUESTS = Counter(
        "aragora_v1_api_requests_total",
        "Total requests to deprecated V1 API endpoints",
        ["endpoint", "method"],
    )

    V1_API_DAYS_UNTIL_SUNSET = Gauge(
        "aragora_v1_api_days_until_sunset",
        "Days remaining until V1 API sunset (0 if past)",
    )

    V1_API_SUNSET_BLOCKED = Counter(
        "aragora_v1_api_sunset_blocked_total",
        "Requests blocked due to V1 API sunset",
        ["endpoint", "method"],
    )

# ============================================================================
# Fallback Implementation (when prometheus_client not available)
# ============================================================================


@dataclass
class SimpleMetrics:
    """Simple metrics storage when prometheus_client is not available."""

    counters: dict[str, float] = field(default_factory=dict)
    gauges: dict[str, float] = field(default_factory=dict)
    histograms: dict[str, list] = field(default_factory=dict)
    info: dict[str, dict[str, str]] = field(default_factory=dict)

    def inc_counter(
        self, name: str, labels: dict[str, str] | None = None, value: float = 1
    ) -> None:
        key = self._make_key(name, labels)
        self.counters[key] = self.counters.get(key, 0) + value

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = self._make_key(name, labels)
        self.gauges[key] = value

    def observe_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        key = self._make_key(name, labels)
        if key not in self.histograms:
            self.histograms[key] = []
        self.histograms[key].append(value)

    def set_info(self, name: str, info: dict[str, str]) -> None:
        self.info[name] = info

    def _make_key(self, name: str, labels: dict[str, str] | None = None) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def generate_output(self) -> str:
        """Generate Prometheus-format output."""
        lines = []

        # Counters
        for key, value in sorted(self.counters.items()):
            lines.append(f"{key} {value}")

        # Gauges
        for key, value in sorted(self.gauges.items()):
            lines.append(f"{key} {value}")

        # Histograms (simplified - just count and sum)
        for key, values in sorted(self.histograms.items()):
            if values:
                lines.append(f"{key}_count {len(values)}")
                lines.append(f"{key}_sum {sum(values)}")

        # Info
        for name, info in self.info.items():
            info_str = ",".join(f'{k}="{v}"' for k, v in info.items())
            lines.append(f"{name}_info{{{info_str}}} 1")

        return "\n".join(lines) + "\n"


# Global simple metrics instance (fallback)
_simple_metrics = SimpleMetrics()

# ============================================================================
# Public API
# ============================================================================


def get_metrics_output() -> tuple[str, str]:
    """
    Get metrics in Prometheus format.

    Returns:
        Tuple of (content, content_type)
    """
    if PROMETHEUS_AVAILABLE:
        return generate_latest(REGISTRY).decode("utf-8"), CONTENT_TYPE_LATEST
    else:
        return _simple_metrics.generate_output(), CONTENT_TYPE_LATEST


def is_prometheus_available() -> bool:
    """Check if prometheus_client is installed."""
    return PROMETHEUS_AVAILABLE


def get_prometheus_metrics() -> str:
    """Get metrics text in Prometheus format."""
    content, _ = get_metrics_output()
    return content


# ============================================================================
# Recording Functions (delegated to prometheus_recording.py)
# ============================================================================

from aragora.server.prometheus_recording import (  # noqa: F401, E402
    record_debate_completed,
    record_tokens_used,
    record_cost_usd,
    record_debate_cost,
    set_budget_utilization,
    record_agent_generation,
    record_agent_failure,
    set_circuit_breaker_state,
    initialize_circuit_breaker_metrics,
    export_circuit_breaker_metrics,
    record_http_request,
    set_websocket_connections,
    record_websocket_message,
    record_rate_limit_hit,
    set_rate_limit_tokens_tracked,
    set_cache_size,
    record_cache_hit,
    record_cache_miss,
    set_server_info,
    record_db_query,
    record_db_error,
    set_db_pool_size,
    set_memory_tier_size,
    record_memory_tier_transition,
    record_memory_operation,
    record_external_agent_task,
    record_external_agent_duration,
    record_external_agent_tokens,
    record_external_agent_tool_blocked,
    record_external_agent_cost,
    record_v1_api_request,
    update_v1_days_until_sunset,
    record_v1_api_sunset_blocked,
)

# ============================================================================
# Decorators (delegated to prometheus_decorators.py)
# ============================================================================

from aragora.server.prometheus_decorators import (  # noqa: F401, E402
    timed_http_request,
    timed_agent_generation,
    timed_db_query,
    timed_db_query_async,
)


# ============================================================================
# Extracted modules (import directly to avoid circular imports)
# ============================================================================
# Nomic metrics: from aragora.server.prometheus_nomic import record_nomic_phase, ...
# Control Plane metrics: from aragora.server.prometheus_control_plane import ...
# RLM metrics: from aragora.server.prometheus_rlm import record_rlm_compression, ...
# Knowledge metrics: from aragora.server.prometheus_knowledge import ...
# Recording functions: from aragora.server.prometheus_recording import ...
# Decorators: from aragora.server.prometheus_decorators import ...

__all__ = [
    # Core
    "PROMETHEUS_AVAILABLE",
    "get_metrics_output",
    "is_prometheus_available",
    "get_prometheus_metrics",
    # Core recording functions (from prometheus_recording)
    "record_debate_completed",
    "record_tokens_used",
    "record_cost_usd",
    "record_debate_cost",
    "set_budget_utilization",
    "record_agent_generation",
    "record_agent_failure",
    "set_circuit_breaker_state",
    "initialize_circuit_breaker_metrics",
    "export_circuit_breaker_metrics",
    "record_http_request",
    "set_websocket_connections",
    "record_websocket_message",
    "record_rate_limit_hit",
    "set_rate_limit_tokens_tracked",
    "set_cache_size",
    "record_cache_hit",
    "record_cache_miss",
    "set_server_info",
    "record_db_query",
    "record_db_error",
    "set_db_pool_size",
    "set_memory_tier_size",
    "record_memory_tier_transition",
    "record_memory_operation",
    "record_external_agent_task",
    "record_external_agent_duration",
    "record_external_agent_tokens",
    "record_external_agent_tool_blocked",
    "record_external_agent_cost",
    # Decorators (from prometheus_decorators)
    "timed_http_request",
    "timed_agent_generation",
    "timed_db_query",
    "timed_db_query_async",
    # V1 API Deprecation (from prometheus_recording)
    "record_v1_api_request",
    "update_v1_days_until_sunset",
    "record_v1_api_sunset_blocked",
    # Note: Nomic, Control Plane, RLM, and Knowledge metrics are in extracted modules.
    # Import directly from: prometheus_nomic, prometheus_control_plane,
    # prometheus_rlm, prometheus_knowledge
]
