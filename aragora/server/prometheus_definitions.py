"""
Prometheus metric definitions and fallback implementation.

Extracted from prometheus.py to break circular imports.
Both prometheus.py and prometheus_recording.py import from here.

This module contains:
- PROMETHEUS_AVAILABLE flag
- All Prometheus metric objects (Counter, Gauge, Histogram, Info)
- SimpleMetrics fallback class for when prometheus_client is not installed
- Global _simple_metrics instance
"""

import logging
from dataclasses import dataclass, field

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
# Safe metric creation helpers (prevent ValueError on duplicate registration)
# ============================================================================

def _get_existing(name: str):
    """Look up an already-registered collector by metric name."""
    try:
        return REGISTRY._names_to_collectors.get(name)
    except (AttributeError, NameError):
        return None


def _safe_counter(name, doc, labels=None):
    if not PROMETHEUS_AVAILABLE:
        return None
    try:
        if labels:
            return Counter(name, doc, labels)
        return Counter(name, doc)
    except ValueError:
        return _get_existing(name)


def _safe_gauge(name, doc, labels=None):
    if not PROMETHEUS_AVAILABLE:
        return None
    try:
        if labels:
            return Gauge(name, doc, labels)
        return Gauge(name, doc)
    except ValueError:
        return _get_existing(name)


def _safe_histogram(name, doc, labels=None, buckets=None):
    if not PROMETHEUS_AVAILABLE:
        return None
    try:
        kwargs = {}
        if labels:
            kwargs["labelnames"] = labels
        if buckets:
            kwargs["buckets"] = buckets
        return Histogram(name, doc, **kwargs)
    except ValueError:
        return _get_existing(name)


def _safe_info(name, doc):
    if not PROMETHEUS_AVAILABLE:
        return None
    try:
        return Info(name, doc)
    except ValueError:
        return _get_existing(name)


# ============================================================================
# Metric Definitions (when prometheus_client is available)
# ============================================================================

if PROMETHEUS_AVAILABLE:
    # Debate metrics
    DEBATE_DURATION = _safe_histogram(
        "aragora_debate_duration_seconds",
        "Time spent in debate execution",
        ["outcome", "agent_count"],
        buckets=[1, 5, 10, 30, 60, 120, 300, 600],
    )

    DEBATE_ROUNDS = _safe_histogram(
        "aragora_debate_rounds_total",
        "Number of rounds per debate",
        ["outcome"],
        buckets=[1, 2, 3, 4, 5, 7, 10],
    )

    DEBATE_TOKENS = _safe_counter(
        "aragora_debate_tokens_total",
        "Total tokens used in debates",
        ["model", "direction"],
    )

    DEBATES_TOTAL = _safe_counter(
        "aragora_debates_total",
        "Total number of debates",
        ["outcome"],
    )

    # Cost metrics
    COST_USD_TOTAL = _safe_counter(
        "aragora_cost_usd_total",
        "Total cost in USD (multiply by 1e-6 for actual dollars)",
        ["provider", "model", "agent_id"],
    )

    COST_PER_DEBATE = _safe_histogram(
        "aragora_debate_cost_usd",
        "Cost per debate in USD (multiply by 1e-6 for actual dollars)",
        ["provider"],
        buckets=[1000, 5000, 10000, 50000, 100000, 500000, 1000000],
    )

    BUDGET_UTILIZATION = _safe_gauge(
        "aragora_budget_utilization_percent",
        "Current budget utilization percentage",
        ["workspace_id", "budget_type"],
    )

    # Agent metrics
    AGENT_GENERATION_DURATION = _safe_histogram(
        "aragora_agent_generation_seconds",
        "Time spent generating agent responses",
        ["agent_type", "model"],
        buckets=[0.5, 1, 2, 5, 10, 30, 60],
    )

    AGENT_FAILURES = _safe_counter(
        "aragora_agent_failures_total",
        "Total agent failures",
        ["agent_type", "error_type"],
    )

    AGENT_CIRCUIT_BREAKER = _safe_gauge(
        "aragora_agent_circuit_breaker_state",
        "Circuit breaker state (0=closed, 1=open, 2=half-open)",
        ["agent_type"],
    )

    # HTTP metrics
    HTTP_REQUEST_DURATION = _safe_histogram(
        "aragora_http_request_duration_seconds",
        "HTTP request latency",
        ["method", "endpoint", "status"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
    )

    HTTP_REQUESTS_TOTAL = _safe_counter(
        "aragora_http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status"],
    )

    # WebSocket metrics
    WEBSOCKET_CONNECTIONS = _safe_gauge(
        "aragora_websocket_connections_active",
        "Number of active WebSocket connections",
    )

    WEBSOCKET_MESSAGES = _safe_counter(
        "aragora_websocket_messages_total",
        "Total WebSocket messages",
        ["direction", "message_type"],
    )

    # Rate limiter metrics
    RATE_LIMIT_HITS = _safe_counter(
        "aragora_rate_limit_hits_total",
        "Rate limit hits",
        ["limit_type"],
    )

    RATE_LIMIT_TOKENS_TRACKED = _safe_gauge(
        "aragora_rate_limit_tokens_tracked",
        "Number of tokens being rate-limited",
    )

    # Cache metrics
    CACHE_SIZE = _safe_gauge(
        "aragora_cache_size_entries",
        "Number of entries in cache",
        ["cache_name"],
    )

    CACHE_HITS = _safe_counter(
        "aragora_cache_hits_total",
        "Cache hits",
        ["cache_name"],
    )

    CACHE_MISSES = _safe_counter(
        "aragora_cache_misses_total",
        "Cache misses",
        ["cache_name"],
    )

    # System info
    ARAGORA_INFO = _safe_info(
        "aragora",
        "Aragora server information",
    )

    # Database metrics
    DB_QUERY_DURATION = _safe_histogram(
        "aragora_db_query_duration_seconds",
        "Database query execution time",
        ["operation", "table"],
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5],
    )

    DB_QUERY_TOTAL = _safe_counter(
        "aragora_db_queries_total",
        "Total database queries",
        ["operation", "table"],
    )

    DB_ERRORS_TOTAL = _safe_counter(
        "aragora_db_errors_total",
        "Total database errors",
        ["error_type", "operation"],
    )

    DB_CONNECTION_POOL_SIZE = _safe_gauge(
        "aragora_db_connection_pool_size",
        "Database connection pool size",
        ["state"],
    )

    # Memory tier metrics
    MEMORY_TIER_SIZE = _safe_gauge(
        "aragora_memory_tier_size",
        "Number of memories in each tier",
        ["tier"],
    )

    MEMORY_TIER_TRANSITIONS = _safe_counter(
        "aragora_memory_tier_transitions_total",
        "Memory tier transitions",
        ["from_tier", "to_tier"],
    )

    MEMORY_OPERATIONS = _safe_counter(
        "aragora_memory_operations_total",
        "Memory operations by type",
        ["operation"],
    )

    # Nomic loop phase metrics
    NOMIC_PHASE_DURATION = _safe_histogram(
        "aragora_nomic_phase_duration_seconds",
        "Time spent in each nomic loop phase",
        ["phase", "outcome"],
        buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1200],
    )

    NOMIC_PHASE_TOTAL = _safe_counter(
        "aragora_nomic_phases_total",
        "Total nomic phases executed",
        ["phase", "outcome"],
    )

    NOMIC_CYCLE_DURATION = _safe_histogram(
        "aragora_nomic_cycle_duration_seconds",
        "Total time for a complete nomic cycle",
        ["outcome"],
        buckets=[60, 120, 300, 600, 1200, 1800, 3600],
    )

    NOMIC_CYCLE_TOTAL = _safe_counter(
        "aragora_nomic_cycles_total",
        "Total nomic cycles executed",
        ["outcome"],
    )

    NOMIC_AGENT_PHASE_DURATION = _safe_histogram(
        "aragora_nomic_agent_phase_seconds",
        "Time spent by each agent in a phase",
        ["phase", "agent"],
        buckets=[1, 5, 10, 30, 60, 120, 300],
    )

    # Control Plane metrics
    CONTROL_PLANE_TASKS_TOTAL = _safe_counter(
        "aragora_control_plane_tasks_total",
        "Total tasks submitted to the control plane",
        ["task_type", "priority"],
    )

    CONTROL_PLANE_TASK_STATUS = _safe_gauge(
        "aragora_control_plane_task_status_count",
        "Number of tasks by status",
        ["status"],
    )

    CONTROL_PLANE_TASK_DURATION = _safe_histogram(
        "aragora_control_plane_task_duration_seconds",
        "Task execution duration",
        ["task_type", "outcome"],
        buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800],
    )

    CONTROL_PLANE_QUEUE_DEPTH = _safe_gauge(
        "aragora_control_plane_queue_depth",
        "Number of tasks in queue by priority",
        ["priority"],
    )

    CONTROL_PLANE_AGENTS_REGISTERED = _safe_gauge(
        "aragora_control_plane_agents_registered",
        "Number of registered agents",
        ["status"],
    )

    CONTROL_PLANE_AGENT_HEALTH = _safe_gauge(
        "aragora_control_plane_agent_health",
        "Agent health status (0=unhealthy, 1=degraded, 2=healthy)",
        ["agent_id"],
    )

    CONTROL_PLANE_AGENT_LATENCY = _safe_histogram(
        "aragora_control_plane_agent_latency_seconds",
        "Agent health check latency",
        ["agent_id"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5],
    )

    CONTROL_PLANE_TASK_RETRIES = _safe_counter(
        "aragora_control_plane_task_retries_total",
        "Total task retries",
        ["task_type", "reason"],
    )

    CONTROL_PLANE_DEAD_LETTER_QUEUE = _safe_gauge(
        "aragora_control_plane_dead_letter_queue_size",
        "Number of tasks in dead letter queue",
    )

    CONTROL_PLANE_CLAIM_LATENCY = _safe_histogram(
        "aragora_control_plane_claim_latency_seconds",
        "Time to claim a task from queue",
        ["priority"],
        buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5],
    )

    # Deliberation metrics
    DELIBERATION_DURATION = _safe_histogram(
        "aragora_deliberation_duration_seconds",
        "Deliberation execution time",
        ["status", "consensus_reached"],
        buckets=[10, 30, 60, 120, 300, 600, 900, 1800],
    )

    DELIBERATION_SLA_COMPLIANCE = _safe_counter(
        "aragora_deliberation_sla_total",
        "SLA compliance counts",
        ["level"],
    )

    DELIBERATION_CONSENSUS_CONFIDENCE = _safe_histogram(
        "aragora_deliberation_consensus_confidence",
        "Distribution of consensus confidence scores",
        buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
    )

    DELIBERATION_ROUNDS = _safe_histogram(
        "aragora_deliberation_rounds_total",
        "Number of rounds per deliberation",
        ["outcome"],
        buckets=[1, 2, 3, 4, 5, 7, 10, 15],
    )

    DELIBERATION_AGENTS = _safe_histogram(
        "aragora_deliberation_agents_count",
        "Number of agents per deliberation",
        buckets=[2, 3, 4, 5, 6, 8, 10],
    )

    # External agent gateway metrics
    EXTERNAL_AGENT_TASKS_TOTAL = _safe_counter(
        "aragora_external_agent_tasks_total",
        "Total external agent tasks",
        ["adapter", "status"],
    )

    EXTERNAL_AGENT_TASK_DURATION = _safe_histogram(
        "aragora_external_agent_task_duration_seconds",
        "External agent task execution time",
        ["adapter", "task_type"],
        buckets=[1, 5, 10, 30, 60, 300, 600, 1800],
    )

    EXTERNAL_AGENT_TOKENS_TOTAL = _safe_counter(
        "aragora_external_agent_tokens_total",
        "Tokens used by external agents",
        ["adapter"],
    )

    EXTERNAL_AGENT_TOOLS_BLOCKED = _safe_counter(
        "aragora_external_agent_tools_blocked_total",
        "Tool calls blocked by security policy",
        ["adapter", "tool"],
    )

    EXTERNAL_AGENT_COST_TOTAL = _safe_counter(
        "aragora_external_agent_cost_microdollars_total",
        "Cost in micro-dollars for external agent tasks",
        ["adapter"],
    )

    DELIBERATION_TOTAL = _safe_counter(
        "aragora_deliberation_total",
        "Total deliberations",
        ["outcome"],
    )

    AGENT_UTILIZATION = _safe_gauge(
        "aragora_agent_utilization_ratio",
        "Agent busy vs available ratio",
        ["agent_id"],
    )

    POLICY_DECISIONS = _safe_counter(
        "aragora_policy_decisions_total",
        "Policy evaluation decisions",
        ["decision", "policy_type"],
    )

    # RLM metrics
    RLM_COMPRESSIONS = _safe_counter("aragora_rlm_compressions_total", "Total RLM compression operations", ["source_type", "status"])
    RLM_COMPRESSION_RATIO = _safe_histogram("aragora_rlm_compression_ratio", "Compression ratio", ["source_type"], buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    RLM_TOKENS_SAVED = _safe_counter("aragora_rlm_tokens_saved_total", "Total tokens saved through compression", ["source_type"])
    RLM_COMPRESSION_DURATION = _safe_histogram("aragora_rlm_compression_duration_seconds", "Time taken for compression", ["source_type", "levels"], buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
    RLM_QUERIES = _safe_counter("aragora_rlm_queries_total", "Total RLM context queries", ["query_type", "level"])
    RLM_QUERY_DURATION = _safe_histogram("aragora_rlm_query_duration_seconds", "Time taken for context queries", ["query_type"], buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5])
    RLM_CACHE_HITS = _safe_counter("aragora_rlm_cache_hits_total", "RLM compression cache hits")
    RLM_CACHE_MISSES = _safe_counter("aragora_rlm_cache_misses_total", "RLM compression cache misses")
    RLM_CONTEXT_LEVELS = _safe_histogram("aragora_rlm_context_levels", "Number of abstraction levels created", ["source_type"], buckets=[1, 2, 3, 4, 5])
    RLM_MEMORY_USAGE = _safe_gauge("aragora_rlm_memory_bytes", "Memory used by RLM context cache")
    RLM_REFINEMENT_ITERATIONS = _safe_histogram("aragora_rlm_refinement_iterations", "Refinement iterations", ["strategy"], buckets=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    RLM_REFINEMENT_SUCCESS = _safe_counter("aragora_rlm_refinement_success_total", "Successful refinements", ["strategy"])
    RLM_REFINEMENT_DURATION = _safe_histogram("aragora_rlm_refinement_duration_seconds", "Refinement loop time", ["strategy"], buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0])
    RLM_READY_FALSE_RATE = _safe_counter("aragora_rlm_ready_false_total", "Times LLM signaled ready=False", ["iteration"])

    # Knowledge Mound metrics
    KNOWLEDGE_VISIBILITY_CHANGES = _safe_counter("aragora_knowledge_visibility_changes_total", "Visibility level changes", ["from_level", "to_level", "workspace_id"])
    KNOWLEDGE_ACCESS_GRANTS = _safe_counter("aragora_knowledge_access_grants_total", "Access grants", ["action", "grantee_type", "workspace_id"])
    KNOWLEDGE_SHARES = _safe_counter("aragora_knowledge_shares_total", "Knowledge sharing operations", ["action", "target_type"])
    KNOWLEDGE_SHARED_ITEMS = _safe_gauge("aragora_knowledge_shared_items_count", "Shared items pending", ["workspace_id"])
    KNOWLEDGE_GLOBAL_FACTS = _safe_counter("aragora_knowledge_global_facts_total", "Global/verified facts", ["action"])
    KNOWLEDGE_GLOBAL_QUERIES = _safe_counter("aragora_knowledge_global_queries_total", "Queries against global knowledge", ["has_results"])
    KNOWLEDGE_FEDERATION_SYNCS = _safe_counter("aragora_knowledge_federation_syncs_total", "Federation sync operations", ["region_id", "direction", "status"])
    KNOWLEDGE_FEDERATION_NODES = _safe_counter("aragora_knowledge_federation_nodes_total", "Nodes synced via federation", ["region_id", "direction"])
    KNOWLEDGE_FEDERATION_LATENCY = _safe_histogram("aragora_knowledge_federation_latency_seconds", "Federation sync latency", ["region_id", "direction"], buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0])
    KNOWLEDGE_FEDERATION_REGIONS = _safe_gauge("aragora_knowledge_federation_regions_count", "Federated regions by status", ["status"])

    # Cross-Pollination Event Metrics
    CROSS_POLL_EVENTS_TOTAL = _safe_counter("aragora_cross_pollination_events_total", "Cross-pollination events dispatched", ["event_type"])
    CROSS_POLL_HANDLER_CALLS = _safe_counter("aragora_cross_pollination_handler_calls_total", "Handler invocations", ["handler", "status"])
    CROSS_POLL_HANDLER_DURATION = _safe_histogram("aragora_cross_pollination_handler_duration_seconds", "Handler execution time", ["handler"], buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5])
    CROSS_POLL_CIRCUIT_BREAKER = _safe_gauge("aragora_cross_pollination_circuit_breaker_state", "Circuit breaker state", ["handler"])
    CROSS_POLL_SUBSCRIBERS = _safe_gauge("aragora_cross_pollination_subscribers", "Registered subscribers", ["event_type"])

    # KM Bidirectional Flow Metrics
    KM_INBOUND_EVENTS = _safe_counter("aragora_km_inbound_events_total", "Events INTO Knowledge Mound", ["source", "event_type"])
    KM_OUTBOUND_EVENTS = _safe_counter("aragora_km_outbound_events_total", "Events OUT of Knowledge Mound", ["target", "event_type"])
    KM_ADAPTER_SYNC = _safe_counter("aragora_km_adapter_sync_total", "Adapter sync operations", ["adapter", "direction", "status"])
    KM_ADAPTER_SYNC_DURATION = _safe_histogram("aragora_km_adapter_sync_duration_seconds", "Adapter sync duration", ["adapter", "direction"], buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30])
    KM_STALENESS_CHECKS = _safe_counter("aragora_km_staleness_checks_total", "Staleness check operations", ["workspace", "status"])
    KM_STALE_NODES_FOUND = _safe_gauge("aragora_km_stale_nodes_found", "Stale nodes found", ["workspace"])
    KM_NODES_BY_SOURCE = _safe_gauge("aragora_km_nodes_by_source", "Knowledge nodes by source type", ["source"])

    # Checkpoint Bridge metrics
    CHECKPOINT_BRIDGE_SAVES = _safe_counter("aragora_checkpoint_bridge_saves_total", "Checkpoint bridge save operations", ["debate_id", "phase"])
    CHECKPOINT_BRIDGE_RESTORES = _safe_counter("aragora_checkpoint_bridge_restores_total", "Checkpoint bridge restore operations", ["status"])
    CHECKPOINT_BRIDGE_MOLECULE_RECOVERIES = _safe_counter("aragora_checkpoint_bridge_molecule_recoveries_total", "Molecule recovery operations", ["status"])
    CHECKPOINT_BRIDGE_SAVE_DURATION = _safe_histogram("aragora_checkpoint_bridge_save_duration_seconds", "Checkpoint bridge save duration", buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0])

    # Agent Channel metrics
    AGENT_CHANNEL_MESSAGES = _safe_counter("aragora_agent_channel_messages_total", "Agent channel messages", ["message_type", "channel"])
    AGENT_CHANNEL_SETUPS = _safe_counter("aragora_agent_channel_setups_total", "Channel setup operations", ["status"])
    AGENT_CHANNEL_TEARDOWNS = _safe_counter("aragora_agent_channel_teardowns_total", "Channel teardown operations")
    AGENT_CHANNEL_ACTIVE = _safe_gauge("aragora_agent_channel_active", "Active agent channels")
    AGENT_CHANNEL_HISTORY_SIZE = _safe_histogram("aragora_agent_channel_history_size", "Message history size at teardown", buckets=[5, 10, 25, 50, 100, 250, 500])

    # Session Management metrics
    SESSION_CREATED = _safe_counter("aragora_session_created_total", "Sessions created", ["channel"])
    SESSION_DEBATES_LINKED = _safe_counter("aragora_session_debates_linked_total", "Debates linked to sessions", ["channel"])
    SESSION_HANDOFFS = _safe_counter("aragora_session_handoffs_total", "Session handoffs", ["from_channel", "to_channel"])
    SESSION_RESULT_ROUTES = _safe_counter("aragora_session_result_routes_total", "Debate results routed", ["channel", "status"])
    SESSION_ACTIVE = _safe_gauge("aragora_sessions_active", "Active sessions by channel", ["channel"])

    # V1 API Deprecation Metrics
    V1_API_REQUESTS = _safe_counter("aragora_v1_api_requests_total", "V1 API requests", ["endpoint", "method"])
    V1_API_DAYS_UNTIL_SUNSET = _safe_gauge("aragora_v1_api_days_until_sunset", "Days until V1 API sunset")
    V1_API_SUNSET_BLOCKED = _safe_counter("aragora_v1_api_sunset_blocked_total", "Requests blocked by sunset", ["endpoint", "method"])


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
        for key, value in sorted(self.counters.items()):
            lines.append(f"{key} {value}")
        for key, value in sorted(self.gauges.items()):
            lines.append(f"{key} {value}")
        for key, values in sorted(self.histograms.items()):
            if values:
                lines.append(f"{key}_count {len(values)}")
                lines.append(f"{key}_sum {sum(values)}")
        for name, info in self.info.items():
            info_str = ",".join(f'{k}="{v}"' for k, v in info.items())
            lines.append(f"{name}_info{{{info_str}}} 1")
        return "\n".join(lines) + "\n"


# Global simple metrics instance (fallback)
_simple_metrics = SimpleMetrics()
