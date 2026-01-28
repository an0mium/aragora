"""
Prometheus Metrics for Aragora.

Exposes metrics for monitoring:
- Billing and subscription events
- API usage and latency
- Debate throughput
- Agent performance

Usage:
    from aragora.server.metrics import (
        SUBSCRIPTION_EVENTS,
        API_REQUESTS,
        track_request,
    )

    # Track subscription event
    SUBSCRIPTION_EVENTS.labels(event="created", tier="starter").inc()

    # Track API request
    with track_request("/api/debates", "POST"):
        # handle request

Metrics endpoint: GET /metrics
"""

# Re-export all symbols for backward compatibility
from .types import (
    Counter,
    Gauge,
    Histogram,
    LabeledCounter,
    LabeledGauge,
    LabeledHistogram,
    get_percentile,
    get_percentiles,
)
from .billing import (
    SUBSCRIPTION_EVENTS,
    SUBSCRIPTION_ACTIVE,
    USAGE_DEBATES,
    USAGE_TOKENS,
    BILLING_REVENUE,
    PAYMENT_FAILURES,
    track_subscription_event,
    track_debate,
    track_tokens,
)
from .api import (
    API_REQUESTS,
    API_LATENCY,
    ACTIVE_DEBATES,
    WEBSOCKET_CONNECTIONS,
    track_request,
)
from .security import (
    AUTH_FAILURES,
    RATE_LIMIT_HITS,
    SECURITY_VIOLATIONS,
    track_auth_failure,
    track_rate_limit_hit,
    track_security_violation,
)
from .debate import (
    DEBATES_TOTAL,
    CONSENSUS_REACHED,
    DEBATE_CONFIDENCE,
    AGENT_PARTICIPATION,
    LAST_DEBATE_TIMESTAMP,
    DEBATE_DURATION,
    CONSENSUS_QUALITY,
    CIRCUIT_BREAKERS_OPEN,
    AGENT_ERRORS,
    track_debate_outcome,
    track_circuit_breaker_state,
    track_agent_error,
    classify_agent_error,
    track_agent_participation,
    track_debate_execution,
)
from .agents import (
    AGENT_REQUESTS,
    AGENT_LATENCY,
    AGENT_TOKENS,
    track_agent_call,
)
from .vector import (
    VECTOR_OPERATIONS,
    VECTOR_LATENCY,
    VECTOR_RESULTS,
    VECTOR_INDEX_BATCH_SIZE,
    track_vector_operation,
    track_vector_search_results,
    track_vector_index_batch,
)
from .knowledge_mound import (
    KNOWLEDGE_VISIBILITY_CHANGES,
    KNOWLEDGE_ACCESS_GRANTS,
    KNOWLEDGE_SHARES,
    KNOWLEDGE_SHARED_ITEMS,
    KNOWLEDGE_GLOBAL_FACTS,
    KNOWLEDGE_GLOBAL_QUERIES,
    KNOWLEDGE_FEDERATION_SYNCS,
    KNOWLEDGE_FEDERATION_NODES,
    KNOWLEDGE_FEDERATION_LATENCY,
    KNOWLEDGE_FEDERATION_REGIONS,
    track_visibility_change,
    track_access_grant,
    track_share,
    track_shared_items_count,
    track_global_fact,
    track_global_query,
    track_federation_sync,
    track_federation_regions,
)
from .export import generate_metrics

__all__ = [
    # Metric types
    "Counter",
    "Gauge",
    "Histogram",
    "LabeledCounter",
    "LabeledGauge",
    "LabeledHistogram",
    # Percentile helpers
    "get_percentile",
    "get_percentiles",
    # Billing metrics
    "SUBSCRIPTION_EVENTS",
    "SUBSCRIPTION_ACTIVE",
    "USAGE_DEBATES",
    "USAGE_TOKENS",
    "BILLING_REVENUE",
    "PAYMENT_FAILURES",
    # API metrics
    "API_REQUESTS",
    "API_LATENCY",
    "ACTIVE_DEBATES",
    "WEBSOCKET_CONNECTIONS",
    # Security metrics
    "AUTH_FAILURES",
    "RATE_LIMIT_HITS",
    "SECURITY_VIOLATIONS",
    # Business metrics (debate outcomes)
    "DEBATES_TOTAL",
    "CONSENSUS_REACHED",
    "DEBATE_DURATION",
    "DEBATE_CONFIDENCE",
    "CONSENSUS_QUALITY",
    "CIRCUIT_BREAKERS_OPEN",
    "AGENT_ERRORS",
    "AGENT_PARTICIPATION",
    "LAST_DEBATE_TIMESTAMP",
    # Agent metrics
    "AGENT_REQUESTS",
    "AGENT_LATENCY",
    "AGENT_TOKENS",
    # Vector store metrics
    "VECTOR_OPERATIONS",
    "VECTOR_LATENCY",
    "VECTOR_RESULTS",
    "VECTOR_INDEX_BATCH_SIZE",
    # Knowledge Mound metrics
    "KNOWLEDGE_VISIBILITY_CHANGES",
    "KNOWLEDGE_ACCESS_GRANTS",
    "KNOWLEDGE_SHARES",
    "KNOWLEDGE_SHARED_ITEMS",
    "KNOWLEDGE_GLOBAL_FACTS",
    "KNOWLEDGE_GLOBAL_QUERIES",
    "KNOWLEDGE_FEDERATION_SYNCS",
    "KNOWLEDGE_FEDERATION_NODES",
    "KNOWLEDGE_FEDERATION_LATENCY",
    "KNOWLEDGE_FEDERATION_REGIONS",
    # Helpers
    "track_request",
    "track_subscription_event",
    "track_debate",
    "track_tokens",
    "track_agent_call",
    "track_auth_failure",
    "track_rate_limit_hit",
    "track_security_violation",
    "track_debate_outcome",
    "track_circuit_breaker_state",
    "track_agent_error",
    "classify_agent_error",
    "track_agent_participation",
    "track_debate_execution",
    # Vector helpers
    "track_vector_operation",
    "track_vector_search_results",
    "track_vector_index_batch",
    # Knowledge Mound helpers
    "track_visibility_change",
    "track_access_grant",
    "track_share",
    "track_shared_items_count",
    "track_global_fact",
    "track_global_query",
    "track_federation_sync",
    "track_federation_regions",
    # Export
    "generate_metrics",
]
