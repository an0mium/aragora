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

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Metric Types
# =============================================================================


@dataclass
class Counter:
    """Simple counter metric."""

    name: str
    help: str
    label_names: list[str] = field(default_factory=list)
    _values: dict[tuple, float] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def labels(self, **kwargs: str) -> "LabeledCounter":
        """Get a labeled instance of this counter."""
        return LabeledCounter(self, tuple(sorted(kwargs.items())))

    def inc(self, value: float = 1.0, **labels) -> None:
        """Increment the counter."""
        key = tuple(sorted(labels.items())) if labels else ()
        with self._lock:
            self._values[key] = self._values.get(key, 0) + value

    def get(self, **labels) -> float:
        """Get current value."""
        key = tuple(sorted(labels.items())) if labels else ()
        return self._values.get(key, 0)

    def collect(self) -> list[tuple[dict, float]]:
        """Collect all values for export."""
        with self._lock:
            return [(dict(k), v) for k, v in self._values.items()]


@dataclass
class LabeledCounter:
    """A counter with specific label values."""

    counter: Counter
    label_values: tuple

    def inc(self, value: float = 1.0) -> None:
        with self.counter._lock:
            self.counter._values[self.label_values] = (
                self.counter._values.get(self.label_values, 0) + value
            )


@dataclass
class Gauge:
    """Simple gauge metric (can go up or down)."""

    name: str
    help: str
    label_names: list[str] = field(default_factory=list)
    _values: dict[tuple, float] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def labels(self, **kwargs: str) -> "LabeledGauge":
        """Get a labeled instance of this gauge."""
        return LabeledGauge(self, tuple(sorted(kwargs.items())))

    def set(self, value: float, **labels) -> None:
        """Set gauge value."""
        key = tuple(sorted(labels.items())) if labels else ()
        with self._lock:
            self._values[key] = value

    def inc(self, value: float = 1.0, **labels) -> None:
        """Increment gauge."""
        key = tuple(sorted(labels.items())) if labels else ()
        with self._lock:
            self._values[key] = self._values.get(key, 0) + value

    def dec(self, value: float = 1.0, **labels) -> None:
        """Decrement gauge."""
        self.inc(-value, **labels)

    def get(self, **labels) -> float:
        """Get current value."""
        key = tuple(sorted(labels.items())) if labels else ()
        return self._values.get(key, 0)

    def collect(self) -> list[tuple[dict, float]]:
        """Collect all values for export."""
        with self._lock:
            return [(dict(k), v) for k, v in self._values.items()]


@dataclass
class LabeledGauge:
    """A gauge with specific label values."""

    gauge: Gauge
    label_values: tuple

    def set(self, value: float) -> None:
        with self.gauge._lock:
            self.gauge._values[self.label_values] = value

    def inc(self, value: float = 1.0) -> None:
        with self.gauge._lock:
            self.gauge._values[self.label_values] = (
                self.gauge._values.get(self.label_values, 0) + value
            )

    def dec(self, value: float = 1.0) -> None:
        self.inc(-value)


@dataclass
class Histogram:
    """Simple histogram for tracking distributions."""

    name: str
    help: str
    label_names: list[str] = field(default_factory=list)
    buckets: list[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    _counts: dict[tuple, list[int]] = field(default_factory=dict)
    _sums: dict[tuple, float] = field(default_factory=dict)
    _totals: dict[tuple, int] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def labels(self, **kwargs: str) -> "LabeledHistogram":
        """Get a labeled instance of this histogram."""
        return LabeledHistogram(self, tuple(sorted(kwargs.items())))

    def observe(self, value: float, **labels) -> None:
        """Record an observation."""
        key = tuple(sorted(labels.items())) if labels else ()
        with self._lock:
            if key not in self._counts:
                self._counts[key] = [0] * len(self.buckets)
                self._sums[key] = 0
                self._totals[key] = 0

            # Update bucket counts
            for i, bucket in enumerate(self.buckets):
                if value <= bucket:
                    self._counts[key][i] += 1

            self._sums[key] += value
            self._totals[key] += 1

    def collect(self) -> list[tuple[dict, dict]]:
        """Collect all values for export."""
        with self._lock:
            results = []
            for key, counts in self._counts.items():
                results.append(
                    (
                        dict(key),
                        {
                            "buckets": list(zip(self.buckets, counts)),
                            "sum": self._sums.get(key, 0),
                            "count": self._totals.get(key, 0),
                        },
                    )
                )
            return results


@dataclass
class LabeledHistogram:
    """A histogram with specific label values."""

    histogram: Histogram
    label_values: tuple

    def observe(self, value: float) -> None:
        key = self.label_values
        with self.histogram._lock:
            if key not in self.histogram._counts:
                self.histogram._counts[key] = [0] * len(self.histogram.buckets)
                self.histogram._sums[key] = 0
                self.histogram._totals[key] = 0

            for i, bucket in enumerate(self.histogram.buckets):
                if value <= bucket:
                    self.histogram._counts[key][i] += 1

            self.histogram._sums[key] += value
            self.histogram._totals[key] += 1


def get_percentile(histogram: Histogram, percentile: float, **labels) -> Optional[float]:
    """Estimate a percentile from histogram buckets.

    Uses linear interpolation between bucket boundaries.

    Args:
        histogram: The histogram to query
        percentile: The percentile to compute (0-100, e.g., 50, 95, 99)
        **labels: Label filters

    Returns:
        Estimated percentile value, or None if no data
    """
    key = tuple(sorted(labels.items())) if labels else ()

    with histogram._lock:
        if key not in histogram._totals or histogram._totals[key] == 0:
            return None

        total = histogram._totals[key]
        target = total * (percentile / 100.0)
        counts = histogram._counts[key]

        # Find the bucket containing the target percentile
        cumulative = 0
        for i, (bucket, count) in enumerate(zip(histogram.buckets, counts)):
            if cumulative + count >= target:
                if i == 0:
                    # In first bucket, estimate using bucket boundary
                    return bucket * (target / count) if count > 0 else bucket
                else:
                    # Linear interpolation between buckets
                    prev_bucket = histogram.buckets[i - 1]
                    prev_cumulative = cumulative
                    within_bucket = (target - prev_cumulative) / count if count > 0 else 0
                    return prev_bucket + (bucket - prev_bucket) * within_bucket
            cumulative += count

        # Above all buckets, return highest bucket boundary
        return histogram.buckets[-1]


def get_percentiles(histogram: Histogram, **labels) -> dict[str, Optional[float]]:
    """Get common percentiles (p50, p90, p95, p99) from a histogram.

    Args:
        histogram: The histogram to query
        **labels: Label filters

    Returns:
        Dict with keys 'p50', 'p90', 'p95', 'p99' and their values
    """
    return {
        "p50": get_percentile(histogram, 50, **labels),
        "p90": get_percentile(histogram, 90, **labels),
        "p95": get_percentile(histogram, 95, **labels),
        "p99": get_percentile(histogram, 99, **labels),
    }


# =============================================================================
# Billing Metrics
# =============================================================================

SUBSCRIPTION_EVENTS = Counter(
    name="aragora_subscription_events_total",
    help="Total subscription events by type and tier",
    label_names=["event", "tier"],
)

SUBSCRIPTION_ACTIVE = Gauge(
    name="aragora_subscriptions_active",
    help="Currently active subscriptions by tier",
    label_names=["tier"],
)

USAGE_DEBATES = Counter(
    name="aragora_debates_total",
    help="Total debates run by tier",
    label_names=["tier", "org_id"],
)

USAGE_TOKENS = Counter(
    name="aragora_tokens_total",
    help="Total tokens used by provider",
    label_names=["provider", "tier"],
)

BILLING_REVENUE = Counter(
    name="aragora_revenue_cents_total",
    help="Total revenue in cents by tier",
    label_names=["tier"],
)

PAYMENT_FAILURES = Counter(
    name="aragora_payment_failures_total",
    help="Payment failure count by tier",
    label_names=["tier"],
)


# =============================================================================
# API Metrics
# =============================================================================

API_REQUESTS = Counter(
    name="aragora_api_requests_total",
    help="Total API requests by endpoint and status",
    label_names=["endpoint", "method", "status"],
)

API_LATENCY = Histogram(
    name="aragora_api_latency_seconds",
    help="API request latency in seconds",
    label_names=["endpoint", "method"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

ACTIVE_DEBATES = Gauge(
    name="aragora_active_debates",
    help="Currently running debates",
    label_names=[],
)

WEBSOCKET_CONNECTIONS = Gauge(
    name="aragora_websocket_connections",
    help="Active WebSocket connections",
    label_names=[],
)


# =============================================================================
# Security Metrics
# =============================================================================

AUTH_FAILURES = Counter(
    name="aragora_auth_failures_total",
    help="Authentication failures by reason and endpoint",
    label_names=["reason", "endpoint"],
)

RATE_LIMIT_HITS = Counter(
    name="aragora_rate_limit_hits_total",
    help="Rate limit hits by endpoint and limit type",
    label_names=["endpoint", "limit_type"],
)

SECURITY_VIOLATIONS = Counter(
    name="aragora_security_violations_total",
    help="Security violations by type (path_traversal, xss_attempt, etc)",
    label_names=["type"],
)


# =============================================================================
# Business Metrics (Debate Outcomes)
# =============================================================================

DEBATES_TOTAL = Counter(
    name="aragora_debates_completed_total",
    help="Total completed debates by status and domain",
    label_names=["status", "domain"],
)

CONSENSUS_REACHED = Counter(
    name="aragora_consensus_reached_total",
    help="Total consensus events by domain and type",
    label_names=["domain", "consensus_type"],  # consensus_type: majority, supermajority, unanimous
)

# Debate confidence score distribution
DEBATE_CONFIDENCE = Histogram(
    name="aragora_debate_confidence_score",
    help="Confidence score of debate conclusions",
    label_names=["domain"],
    buckets=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
)

# Agent participation and outcomes
AGENT_PARTICIPATION = Counter(
    name="aragora_agent_participation_total",
    help="Agent participation in debates by outcome",
    label_names=["agent_name", "outcome"],  # outcome: won, lost, abstained, contributed
)

# Last debate timestamp for staleness detection
LAST_DEBATE_TIMESTAMP = Gauge(
    name="aragora_last_debate_timestamp",
    help="Unix timestamp of the last completed debate",
    label_names=[],
)

DEBATE_DURATION = Histogram(
    name="aragora_debate_duration_seconds",
    help="Debate duration in seconds",
    label_names=["domain", "status"],
    buckets=[5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0],
)

CONSENSUS_QUALITY = Gauge(
    name="aragora_consensus_quality",
    help="Average consensus confidence by domain (0.0-1.0)",
    label_names=["domain"],
)

CIRCUIT_BREAKERS_OPEN = Gauge(
    name="aragora_circuit_breakers_open",
    help="Number of circuit breakers in open state",
    label_names=[],
)

AGENT_ERRORS = Counter(
    name="aragora_agent_errors_total",
    help="Agent error count by agent name and error type",
    label_names=["agent", "error_type"],
)


# =============================================================================
# Agent Metrics
# =============================================================================

AGENT_REQUESTS = Counter(
    name="aragora_agent_requests_total",
    help="Agent API requests by agent and status",
    label_names=["agent", "status"],
)

AGENT_LATENCY = Histogram(
    name="aragora_agent_latency_seconds",
    help="Agent response latency",
    label_names=["agent"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

AGENT_TOKENS = Counter(
    name="aragora_agent_tokens_total",
    help="Tokens used by agent",
    label_names=["agent", "direction"],  # direction: input/output
)


# =============================================================================
# Vector Store Metrics
# =============================================================================

VECTOR_OPERATIONS = Counter(
    name="aragora_vector_operations_total",
    help="Total vector store operations",
    label_names=["operation", "store", "status"],  # operation: search/index/delete
)

VECTOR_LATENCY = Histogram(
    name="aragora_vector_latency_seconds",
    help="Vector operation latency",
    label_names=["operation", "store"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

VECTOR_RESULTS = Histogram(
    name="aragora_vector_results_count",
    help="Number of results returned by vector search",
    label_names=["store", "search_type"],  # search_type: semantic/keyword/relationship
    buckets=[0, 1, 5, 10, 25, 50, 100, 250, 500],
)

VECTOR_INDEX_BATCH_SIZE = Histogram(
    name="aragora_vector_index_batch_size",
    help="Batch size for vector indexing operations",
    label_names=["store"],
    buckets=[1, 10, 25, 50, 100, 250, 500, 1000],
)


# =============================================================================
# Knowledge Mound Metrics
# =============================================================================

# Visibility operations
KNOWLEDGE_VISIBILITY_CHANGES = Counter(
    name="aragora_knowledge_visibility_changes_total",
    help="Number of visibility level changes on knowledge items",
    label_names=["from_level", "to_level", "workspace_id"],
)

KNOWLEDGE_ACCESS_GRANTS = Counter(
    name="aragora_knowledge_access_grants_total",
    help="Number of access grants created/revoked",
    label_names=["action", "grantee_type", "workspace_id"],  # action: grant/revoke
)

# Sharing operations
KNOWLEDGE_SHARES = Counter(
    name="aragora_knowledge_shares_total",
    help="Number of knowledge sharing operations",
    label_names=["action", "target_type"],  # action: share/accept/decline/revoke, target_type: workspace/user
)

KNOWLEDGE_SHARED_ITEMS = Gauge(
    name="aragora_knowledge_shared_items_count",
    help="Current number of shared items pending acceptance",
    label_names=["workspace_id"],
)

# Global knowledge operations
KNOWLEDGE_GLOBAL_FACTS = Counter(
    name="aragora_knowledge_global_facts_total",
    help="Number of global/verified facts stored or promoted",
    label_names=["action"],  # action: stored/promoted/queried
)

KNOWLEDGE_GLOBAL_QUERIES = Counter(
    name="aragora_knowledge_global_queries_total",
    help="Number of queries against global knowledge",
    label_names=["has_results"],  # has_results: true/false
)

# Federation operations
KNOWLEDGE_FEDERATION_SYNCS = Counter(
    name="aragora_knowledge_federation_syncs_total",
    help="Number of federation sync operations",
    label_names=["region_id", "direction", "status"],  # direction: push/pull, status: success/failed
)

KNOWLEDGE_FEDERATION_NODES = Counter(
    name="aragora_knowledge_federation_nodes_total",
    help="Number of nodes synced via federation",
    label_names=["region_id", "direction"],
)

KNOWLEDGE_FEDERATION_LATENCY = Histogram(
    name="aragora_knowledge_federation_latency_seconds",
    help="Latency of federation sync operations",
    label_names=["region_id", "direction"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

KNOWLEDGE_FEDERATION_REGIONS = Gauge(
    name="aragora_knowledge_federation_regions_count",
    help="Number of federated regions by status",
    label_names=["status"],  # status: enabled/disabled/healthy/unhealthy
)


# =============================================================================
# Helpers
# =============================================================================


@contextmanager
def track_request(endpoint: str, method: str = "GET") -> Generator[None, None, None]:
    """Context manager to track request latency."""
    start = time.perf_counter()
    status = "success"
    try:
        yield
    except Exception:  # noqa: BLE001 - Intentionally broad for metrics tracking
        # Note: We catch Exception (not BaseException) to track application errors
        # while letting KeyboardInterrupt/SystemExit propagate as intentional terminations
        status = "error"
        raise
    finally:
        duration = time.perf_counter() - start
        API_REQUESTS.inc(endpoint=endpoint, method=method, status=status)
        API_LATENCY.observe(duration, endpoint=endpoint, method=method)


def track_subscription_event(event: str, tier: str) -> None:
    """Track a subscription event."""
    SUBSCRIPTION_EVENTS.inc(event=event, tier=tier)


def track_debate(tier: str, org_id: str) -> None:
    """Track a debate execution."""
    USAGE_DEBATES.inc(tier=tier, org_id=org_id)


def track_tokens(provider: str, tier: str, count: int) -> None:
    """Track token usage."""
    USAGE_TOKENS.inc(count, provider=provider, tier=tier)


def track_agent_call(
    agent: str, latency: float, tokens_in: int, tokens_out: int, success: bool
) -> None:
    """Track an agent API call."""
    AGENT_REQUESTS.inc(agent=agent, status="success" if success else "error")
    AGENT_LATENCY.observe(latency, agent=agent)
    AGENT_TOKENS.inc(tokens_in, agent=agent, direction="input")
    AGENT_TOKENS.inc(tokens_out, agent=agent, direction="output")


def track_auth_failure(reason: str, endpoint: str = "unknown") -> None:
    """Track an authentication failure.

    Args:
        reason: Why authentication failed (invalid_token, expired, wrong_password, etc)
        endpoint: The endpoint where the failure occurred
    """
    AUTH_FAILURES.inc(reason=reason, endpoint=endpoint)


def track_rate_limit_hit(endpoint: str, limit_type: str = "request") -> None:
    """Track when a rate limit is hit.

    Args:
        endpoint: The rate-limited endpoint
        limit_type: Type of limit (request, upload, etc)
    """
    RATE_LIMIT_HITS.inc(endpoint=endpoint, limit_type=limit_type)


def track_security_violation(violation_type: str) -> None:
    """Track a security violation attempt.

    Args:
        violation_type: Type of violation (path_traversal, xss_attempt, sql_injection, etc)
    """
    SECURITY_VIOLATIONS.inc(type=violation_type)


@contextmanager
def track_vector_operation(
    operation: str, store: str = "weaviate"
) -> Generator[None, None, None]:
    """Context manager to track vector store operation latency.

    Args:
        operation: Operation type (search_semantic, search_keyword, index, delete)
        store: Store name (weaviate, qdrant, chroma)

    Usage:
        with track_vector_operation("search_semantic", "weaviate"):
            results = await store.search_semantic(embedding, limit=10)
    """
    start = time.perf_counter()
    status = "success"
    try:
        yield
    except Exception:  # noqa: BLE001 - Re-raised after recording status
        status = "error"
        raise
    finally:
        duration = time.perf_counter() - start
        VECTOR_OPERATIONS.inc(operation=operation, store=store, status=status)
        VECTOR_LATENCY.observe(duration, operation=operation, store=store)
        # Log slow queries (>500ms)
        if duration > 0.5:
            logger.warning(
                f"Slow vector operation: {operation} on {store} took {duration:.3f}s"
            )


def track_vector_search_results(
    result_count: int, store: str = "weaviate", search_type: str = "semantic"
) -> None:
    """Track number of results returned by vector search.

    Args:
        result_count: Number of results returned
        store: Store name
        search_type: Type of search (semantic, keyword, relationship)
    """
    VECTOR_RESULTS.observe(result_count, store=store, search_type=search_type)


def track_vector_index_batch(batch_size: int, store: str = "weaviate") -> None:
    """Track batch size for vector indexing.

    Args:
        batch_size: Number of items indexed
        store: Store name
    """
    VECTOR_INDEX_BATCH_SIZE.observe(batch_size, store=store)


def track_debate_outcome(
    status: str,
    domain: str,
    duration_seconds: float,
    consensus_reached: bool = False,
    confidence: float = 0.0,
    consensus_type: str = "majority",
) -> None:
    """Track debate outcome metrics.

    Args:
        status: Debate status (completed, timeout, error, aborted)
        domain: Debate domain (security, performance, testing, etc)
        duration_seconds: Time taken for the debate
        consensus_reached: Whether consensus was reached
        confidence: Consensus confidence (0.0-1.0)
        consensus_type: Type of consensus (majority, supermajority, unanimous)
    """
    import time as time_module

    # Increment debate counter
    DEBATES_TOTAL.inc(status=status, domain=domain)

    # Record duration
    DEBATE_DURATION.observe(duration_seconds, domain=domain, status=status)

    # Update last debate timestamp
    LAST_DEBATE_TIMESTAMP.set(time_module.time())

    # Track confidence distribution
    if confidence > 0:
        DEBATE_CONFIDENCE.observe(confidence, domain=domain)

    # Track consensus metrics
    if consensus_reached:
        CONSENSUS_REACHED.inc(domain=domain, consensus_type=consensus_type)
        # Update rolling average confidence (simplified: just set to latest)
        if confidence > 0:
            CONSENSUS_QUALITY.set(confidence, domain=domain)


def track_circuit_breaker_state(open_count: int) -> None:
    """Track number of open circuit breakers.

    Args:
        open_count: Number of circuit breakers currently in open state
    """
    CIRCUIT_BREAKERS_OPEN.set(open_count)


def track_agent_error(agent: str, error_type: str = "unknown") -> None:
    """Track an agent error with type classification.

    Args:
        agent: Name of the agent that encountered an error
        error_type: Type of error (timeout, rate_limit, auth, network, api, validation, unknown)
    """
    AGENT_ERRORS.inc(agent=agent, error_type=error_type)


def classify_agent_error(error: Exception) -> str:
    """Classify an exception into an error type for metrics.

    Args:
        error: The exception to classify

    Returns:
        Error type string for metrics labeling
    """
    error_class = type(error).__name__.lower()

    # Check for common error patterns
    if "timeout" in error_class:
        return "timeout"
    if "ratelimit" in error_class or "429" in str(error):
        return "rate_limit"
    if "auth" in error_class or "401" in str(error) or "403" in str(error):
        return "auth"
    if "connection" in error_class or "network" in error_class:
        return "network"
    if "validation" in error_class or "value" in error_class:
        return "validation"
    if "api" in error_class or "500" in str(error):
        return "api"

    return "unknown"


def track_agent_participation(agent_name: str, outcome: str) -> None:
    """Track an agent's participation in a debate.

    Args:
        agent_name: Name of the agent
        outcome: Participation outcome (won, lost, abstained, contributed)
    """
    AGENT_PARTICIPATION.inc(agent_name=agent_name, outcome=outcome)


@contextmanager
def track_debate_execution(domain: str = "general") -> Generator[dict, None, None]:
    """Context manager to track debate execution metrics.

    Usage:
        with track_debate_execution(domain="security") as ctx:
            # run debate
            ctx["consensus"] = True
            ctx["confidence"] = 0.85
            ctx["status"] = "completed"

    Args:
        domain: The debate domain

    Yields:
        Dict to populate with outcome data (consensus, confidence, status)
    """
    start = time.perf_counter()
    ctx: dict[str, Any] = {
        "status": "completed",
        "consensus": False,
        "confidence": 0.0,
    }
    ACTIVE_DEBATES.inc()
    try:
        yield ctx
    except asyncio.TimeoutError:
        ctx["status"] = "timeout"
        raise
    except Exception:  # noqa: BLE001 - Intentionally broad for metrics tracking
        # Note: Catches any application error to track status, then re-raises
        ctx["status"] = "error"
        raise
    finally:
        ACTIVE_DEBATES.dec()
        duration = time.perf_counter() - start
        track_debate_outcome(
            status=str(ctx["status"]),
            domain=domain,
            duration_seconds=duration,
            consensus_reached=bool(ctx["consensus"]),
            confidence=float(ctx["confidence"]),
        )


# =============================================================================
# Knowledge Mound Helpers
# =============================================================================


def track_visibility_change(
    node_id: str,
    from_level: str,
    to_level: str,
    workspace_id: str,
) -> None:
    """Track a visibility level change on a knowledge item."""
    KNOWLEDGE_VISIBILITY_CHANGES.inc(
        from_level=from_level,
        to_level=to_level,
        workspace_id=workspace_id,
    )


def track_access_grant(
    action: str,
    grantee_type: str,
    workspace_id: str,
) -> None:
    """Track an access grant create/revoke operation."""
    KNOWLEDGE_ACCESS_GRANTS.inc(
        action=action,
        grantee_type=grantee_type,
        workspace_id=workspace_id,
    )


def track_share(action: str, target_type: str) -> None:
    """Track a sharing operation (share/accept/decline/revoke)."""
    KNOWLEDGE_SHARES.inc(action=action, target_type=target_type)


def track_shared_items_count(workspace_id: str, count: int) -> None:
    """Update the count of pending shared items for a workspace."""
    KNOWLEDGE_SHARED_ITEMS.set(count, workspace_id=workspace_id)


def track_global_fact(action: str) -> None:
    """Track a global knowledge operation (stored/promoted/queried)."""
    KNOWLEDGE_GLOBAL_FACTS.inc(action=action)


def track_global_query(has_results: bool) -> None:
    """Track a query against global knowledge."""
    KNOWLEDGE_GLOBAL_QUERIES.inc(has_results=str(has_results).lower())


@contextmanager
def track_federation_sync(
    region_id: str,
    direction: str,
) -> Generator[dict, None, None]:
    """Context manager to track federation sync operations.

    Usage:
        with track_federation_sync("region-1", "push") as ctx:
            # perform sync
            ctx["nodes_synced"] = 42
            ctx["status"] = "success"

    Args:
        region_id: The federated region ID
        direction: Sync direction ("push" or "pull")

    Yields:
        Dict to populate with sync results (nodes_synced, status)
    """
    start = time.perf_counter()
    ctx: dict[str, Any] = {"status": "success", "nodes_synced": 0}
    try:
        yield ctx
    except Exception:  # noqa: BLE001 - Intentionally broad for metrics tracking
        ctx["status"] = "failed"
        raise
    finally:
        duration = time.perf_counter() - start
        KNOWLEDGE_FEDERATION_SYNCS.inc(
            region_id=region_id,
            direction=direction,
            status=str(ctx["status"]),
        )
        KNOWLEDGE_FEDERATION_NODES.inc(
            region_id=region_id,
            direction=direction,
            value=int(ctx.get("nodes_synced", 0)),
        )
        KNOWLEDGE_FEDERATION_LATENCY.observe(
            duration,
            region_id=region_id,
            direction=direction,
        )


def track_federation_regions(
    enabled: int = 0,
    disabled: int = 0,
    healthy: int = 0,
    unhealthy: int = 0,
) -> None:
    """Update federation region counts by status."""
    KNOWLEDGE_FEDERATION_REGIONS.set(enabled, status="enabled")
    KNOWLEDGE_FEDERATION_REGIONS.set(disabled, status="disabled")
    KNOWLEDGE_FEDERATION_REGIONS.set(healthy, status="healthy")
    KNOWLEDGE_FEDERATION_REGIONS.set(unhealthy, status="unhealthy")


# =============================================================================
# Prometheus Format Export
# =============================================================================


def _format_labels(labels: dict) -> str:
    """Format labels for Prometheus output."""
    if not labels:
        return ""
    parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
    return "{" + ",".join(parts) + "}"


def generate_metrics() -> str:
    """Generate Prometheus-format metrics output.

    This combines:
    1. Built-in server metrics (billing, API, agent)
    2. Nomic loop metrics
    3. Observability metrics from aragora.observability.metrics (if available)
    """
    lines = []

    # All metrics to export
    counters = [
        SUBSCRIPTION_EVENTS,
        USAGE_DEBATES,
        USAGE_TOKENS,
        BILLING_REVENUE,
        PAYMENT_FAILURES,
        API_REQUESTS,
        AGENT_REQUESTS,
        AGENT_TOKENS,
        # Security metrics
        AUTH_FAILURES,
        RATE_LIMIT_HITS,
        SECURITY_VIOLATIONS,
        # Business metrics
        DEBATES_TOTAL,
        CONSENSUS_REACHED,
        AGENT_ERRORS,
        AGENT_PARTICIPATION,
        # Knowledge Mound metrics
        KNOWLEDGE_VISIBILITY_CHANGES,
        KNOWLEDGE_ACCESS_GRANTS,
        KNOWLEDGE_SHARES,
        KNOWLEDGE_GLOBAL_FACTS,
        KNOWLEDGE_GLOBAL_QUERIES,
        KNOWLEDGE_FEDERATION_SYNCS,
        KNOWLEDGE_FEDERATION_NODES,
    ]

    gauges = [
        SUBSCRIPTION_ACTIVE,
        ACTIVE_DEBATES,
        WEBSOCKET_CONNECTIONS,
        # Business metrics
        CONSENSUS_QUALITY,
        CIRCUIT_BREAKERS_OPEN,
        LAST_DEBATE_TIMESTAMP,
        # Knowledge Mound metrics
        KNOWLEDGE_SHARED_ITEMS,
        KNOWLEDGE_FEDERATION_REGIONS,
    ]

    histograms = [
        API_LATENCY,
        AGENT_LATENCY,
        # Business metrics
        DEBATE_DURATION,
        DEBATE_CONFIDENCE,
        # Knowledge Mound metrics
        KNOWLEDGE_FEDERATION_LATENCY,
    ]

    # Add nomic loop metrics if available
    try:
        from aragora.nomic.metrics import (
            NOMIC_CIRCUIT_BREAKERS_OPEN,
            NOMIC_CURRENT_PHASE,
            NOMIC_CYCLES_IN_PROGRESS,
            NOMIC_CYCLES_TOTAL,
            NOMIC_ERRORS,
            NOMIC_PHASE_DURATION,
            NOMIC_PHASE_LAST_TRANSITION,
            NOMIC_PHASE_TRANSITIONS,
            NOMIC_RECOVERY_DECISIONS,
            NOMIC_RETRIES,
        )

        counters.extend(
            [
                NOMIC_PHASE_TRANSITIONS,
                NOMIC_CYCLES_TOTAL,
                NOMIC_ERRORS,
                NOMIC_RECOVERY_DECISIONS,
                NOMIC_RETRIES,
            ]
        )
        gauges.extend(
            [
                NOMIC_CURRENT_PHASE,
                NOMIC_CYCLES_IN_PROGRESS,
                NOMIC_PHASE_LAST_TRANSITION,
                NOMIC_CIRCUIT_BREAKERS_OPEN,
            ]
        )
        histograms.append(NOMIC_PHASE_DURATION)
    except ImportError:
        # Nomic metrics not available
        pass

    # Export counters
    for counter in counters:
        lines.append(f"# HELP {counter.name} {counter.help}")
        lines.append(f"# TYPE {counter.name} counter")
        for labels, value in counter.collect():
            lines.append(f"{counter.name}{_format_labels(labels)} {value}")
        lines.append("")

    # Export gauges
    for gauge in gauges:
        lines.append(f"# HELP {gauge.name} {gauge.help}")
        lines.append(f"# TYPE {gauge.name} gauge")
        for labels, value in gauge.collect():
            lines.append(f"{gauge.name}{_format_labels(labels)} {value}")
        lines.append("")

    # Export histograms
    for histogram in histograms:
        lines.append(f"# HELP {histogram.name} {histogram.help}")
        lines.append(f"# TYPE {histogram.name} histogram")
        for labels, data in histogram.collect():
            for bucket, count in data["buckets"]:
                bucket_labels = {**labels, "le": str(bucket)}
                lines.append(f"{histogram.name}_bucket{_format_labels(bucket_labels)} {count}")
            inf_labels = {**labels, "le": "+Inf"}
            lines.append(f"{histogram.name}_bucket{_format_labels(inf_labels)} {data['count']}")
            lines.append(f"{histogram.name}_sum{_format_labels(labels)} {data['sum']}")
            lines.append(f"{histogram.name}_count{_format_labels(labels)} {data['count']}")
        lines.append("")

    # Include observability metrics if prometheus_client is available
    try:
        from prometheus_client import REGISTRY, generate_latest

        observability_metrics = generate_latest(REGISTRY).decode("utf-8")
        if observability_metrics.strip():
            lines.append("# Observability metrics (from prometheus_client)")
            lines.append(observability_metrics)
    except ImportError:
        # prometheus_client not installed, skip observability metrics
        pass
    except Exception as e:
        lines.append(f"# Error collecting observability metrics: {e}")

    return "\n".join(lines)


__all__ = [
    # Metric types
    "Counter",
    "Gauge",
    "Histogram",
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
