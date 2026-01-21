"""
Prometheus telemetry metrics for social media handler integrations.

Provides OpenMetrics-compliant metrics for monitoring:
- Webhook requests (counts, latency by platform)
- Message handling (by type: command, text, callback)
- Commands executed (by platform and command)
- Debates and gauntlets (started, completed, failed)
- Votes recorded
- Error tracking
"""

from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

# Try to import prometheus_client, fall back gracefully
try:
    from prometheus_client import Counter, Gauge, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# ============================================================================
# Metric Definitions (when prometheus_client is available)
# ============================================================================

if PROMETHEUS_AVAILABLE:
    # Webhook request metrics
    SOCIAL_WEBHOOK_REQUESTS_TOTAL = Counter(
        "aragora_social_webhook_requests_total",
        "Total webhook requests received",
        ["platform", "status"],  # platform: telegram, whatsapp; status: success, error, unauthorized
    )

    SOCIAL_WEBHOOK_LATENCY = Histogram(
        "aragora_social_webhook_latency_seconds",
        "Webhook request processing latency",
        ["platform"],
        buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )

    # Message handling metrics
    SOCIAL_MESSAGES_TOTAL = Counter(
        "aragora_social_messages_total",
        "Total messages processed",
        ["platform", "message_type"],  # message_type: text, command, callback, inline, interactive
    )

    SOCIAL_COMMANDS_TOTAL = Counter(
        "aragora_social_commands_total",
        "Total commands executed",
        ["platform", "command"],  # command: start, help, status, agents, debate, gauntlet, unknown
    )

    # Debate metrics
    SOCIAL_DEBATES_STARTED = Counter(
        "aragora_social_debates_started_total",
        "Total debates started from social platforms",
        ["platform"],
    )

    SOCIAL_DEBATES_COMPLETED = Counter(
        "aragora_social_debates_completed_total",
        "Total debates completed successfully",
        ["platform", "consensus"],  # consensus: reached, not_reached
    )

    SOCIAL_DEBATES_FAILED = Counter(
        "aragora_social_debates_failed_total",
        "Total debates that failed",
        ["platform"],
    )

    SOCIAL_DEBATES_IN_PROGRESS = Gauge(
        "aragora_social_debates_in_progress",
        "Number of debates currently in progress",
        ["platform"],
    )

    # Gauntlet metrics
    SOCIAL_GAUNTLETS_STARTED = Counter(
        "aragora_social_gauntlets_started_total",
        "Total gauntlets started from social platforms",
        ["platform"],
    )

    SOCIAL_GAUNTLETS_COMPLETED = Counter(
        "aragora_social_gauntlets_completed_total",
        "Total gauntlets completed",
        ["platform", "result"],  # result: passed, failed
    )

    SOCIAL_GAUNTLETS_FAILED = Counter(
        "aragora_social_gauntlets_failed_total",
        "Total gauntlets that failed (error)",
        ["platform"],
    )

    # Vote metrics
    SOCIAL_VOTES_TOTAL = Counter(
        "aragora_social_votes_total",
        "Total votes recorded",
        ["platform", "vote"],  # vote: agree, disagree
    )

    # Error metrics
    SOCIAL_ERRORS_TOTAL = Counter(
        "aragora_social_errors_total",
        "Total errors encountered",
        ["platform", "error_type"],  # error_type: json_parse, api_call, auth, unknown
    )

    # API call metrics (outbound to Telegram/WhatsApp APIs)
    SOCIAL_API_CALLS_TOTAL = Counter(
        "aragora_social_api_calls_total",
        "Total outbound API calls",
        ["platform", "method", "status"],  # method: sendMessage, answerCallback, etc.
    )

    SOCIAL_API_LATENCY = Histogram(
        "aragora_social_api_latency_seconds",
        "Outbound API call latency",
        ["platform", "method"],
        buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
    )


# ============================================================================
# Fallback Implementation (when prometheus_client not available)
# ============================================================================


class FallbackSocialMetrics:
    """Simple metrics accumulator when prometheus_client is unavailable."""

    def __init__(self) -> None:
        self.webhook_requests: Dict[str, Dict[str, int]] = {}
        self.webhook_latencies: Dict[str, list] = {}
        self.messages: Dict[str, Dict[str, int]] = {}
        self.commands: Dict[str, Dict[str, int]] = {}
        self.debates_started: Dict[str, int] = {}
        self.debates_completed: Dict[str, Dict[str, int]] = {}
        self.debates_failed: Dict[str, int] = {}
        self.debates_in_progress: Dict[str, int] = {}
        self.gauntlets_started: Dict[str, int] = {}
        self.gauntlets_completed: Dict[str, Dict[str, int]] = {}
        self.gauntlets_failed: Dict[str, int] = {}
        self.votes: Dict[str, Dict[str, int]] = {}
        self.errors: Dict[str, Dict[str, int]] = {}
        self.api_calls: Dict[str, Dict[str, Dict[str, int]]] = {}
        self.api_latencies: Dict[str, Dict[str, list]] = {}


_fallback_metrics: Optional[FallbackSocialMetrics] = None


def _get_fallback_metrics() -> FallbackSocialMetrics:
    """Get or create fallback metrics instance."""
    global _fallback_metrics
    if _fallback_metrics is None:
        _fallback_metrics = FallbackSocialMetrics()
    return _fallback_metrics


# ============================================================================
# Metric Recording Functions
# ============================================================================


def record_webhook_request(platform: str, status: str) -> None:
    """Record a webhook request."""
    if PROMETHEUS_AVAILABLE:
        SOCIAL_WEBHOOK_REQUESTS_TOTAL.labels(platform=platform, status=status).inc()
    else:
        fb = _get_fallback_metrics()
        if platform not in fb.webhook_requests:
            fb.webhook_requests[platform] = {}
        fb.webhook_requests[platform][status] = fb.webhook_requests[platform].get(status, 0) + 1


def record_webhook_latency(platform: str, latency_seconds: float) -> None:
    """Record webhook processing latency."""
    if PROMETHEUS_AVAILABLE:
        SOCIAL_WEBHOOK_LATENCY.labels(platform=platform).observe(latency_seconds)
    else:
        fb = _get_fallback_metrics()
        if platform not in fb.webhook_latencies:
            fb.webhook_latencies[platform] = []
        fb.webhook_latencies[platform].append(latency_seconds)
        # Keep only last 1000 samples
        if len(fb.webhook_latencies[platform]) > 1000:
            fb.webhook_latencies[platform] = fb.webhook_latencies[platform][-1000:]


def record_message(platform: str, message_type: str) -> None:
    """Record a processed message."""
    if PROMETHEUS_AVAILABLE:
        SOCIAL_MESSAGES_TOTAL.labels(platform=platform, message_type=message_type).inc()
    else:
        fb = _get_fallback_metrics()
        if platform not in fb.messages:
            fb.messages[platform] = {}
        fb.messages[platform][message_type] = fb.messages[platform].get(message_type, 0) + 1


def record_command(platform: str, command: str) -> None:
    """Record a command execution."""
    if PROMETHEUS_AVAILABLE:
        SOCIAL_COMMANDS_TOTAL.labels(platform=platform, command=command).inc()
    else:
        fb = _get_fallback_metrics()
        if platform not in fb.commands:
            fb.commands[platform] = {}
        fb.commands[platform][command] = fb.commands[platform].get(command, 0) + 1


def record_debate_started(platform: str) -> None:
    """Record a debate being started."""
    if PROMETHEUS_AVAILABLE:
        SOCIAL_DEBATES_STARTED.labels(platform=platform).inc()
        SOCIAL_DEBATES_IN_PROGRESS.labels(platform=platform).inc()
    else:
        fb = _get_fallback_metrics()
        fb.debates_started[platform] = fb.debates_started.get(platform, 0) + 1
        fb.debates_in_progress[platform] = fb.debates_in_progress.get(platform, 0) + 1


def record_debate_completed(platform: str, consensus_reached: bool) -> None:
    """Record a debate completion."""
    consensus = "reached" if consensus_reached else "not_reached"
    if PROMETHEUS_AVAILABLE:
        SOCIAL_DEBATES_COMPLETED.labels(platform=platform, consensus=consensus).inc()
        SOCIAL_DEBATES_IN_PROGRESS.labels(platform=platform).dec()
    else:
        fb = _get_fallback_metrics()
        if platform not in fb.debates_completed:
            fb.debates_completed[platform] = {}
        fb.debates_completed[platform][consensus] = fb.debates_completed[platform].get(consensus, 0) + 1
        fb.debates_in_progress[platform] = max(0, fb.debates_in_progress.get(platform, 0) - 1)


def record_debate_failed(platform: str) -> None:
    """Record a debate failure."""
    if PROMETHEUS_AVAILABLE:
        SOCIAL_DEBATES_FAILED.labels(platform=platform).inc()
        SOCIAL_DEBATES_IN_PROGRESS.labels(platform=platform).dec()
    else:
        fb = _get_fallback_metrics()
        fb.debates_failed[platform] = fb.debates_failed.get(platform, 0) + 1
        fb.debates_in_progress[platform] = max(0, fb.debates_in_progress.get(platform, 0) - 1)


def record_gauntlet_started(platform: str) -> None:
    """Record a gauntlet being started."""
    if PROMETHEUS_AVAILABLE:
        SOCIAL_GAUNTLETS_STARTED.labels(platform=platform).inc()
    else:
        fb = _get_fallback_metrics()
        fb.gauntlets_started[platform] = fb.gauntlets_started.get(platform, 0) + 1


def record_gauntlet_completed(platform: str, passed: bool) -> None:
    """Record a gauntlet completion."""
    result = "passed" if passed else "failed"
    if PROMETHEUS_AVAILABLE:
        SOCIAL_GAUNTLETS_COMPLETED.labels(platform=platform, result=result).inc()
    else:
        fb = _get_fallback_metrics()
        if platform not in fb.gauntlets_completed:
            fb.gauntlets_completed[platform] = {}
        fb.gauntlets_completed[platform][result] = fb.gauntlets_completed[platform].get(result, 0) + 1


def record_gauntlet_failed(platform: str) -> None:
    """Record a gauntlet failure (error, not test failure)."""
    if PROMETHEUS_AVAILABLE:
        SOCIAL_GAUNTLETS_FAILED.labels(platform=platform).inc()
    else:
        fb = _get_fallback_metrics()
        fb.gauntlets_failed[platform] = fb.gauntlets_failed.get(platform, 0) + 1


def record_vote(platform: str, vote: str) -> None:
    """Record a vote."""
    if PROMETHEUS_AVAILABLE:
        SOCIAL_VOTES_TOTAL.labels(platform=platform, vote=vote).inc()
    else:
        fb = _get_fallback_metrics()
        if platform not in fb.votes:
            fb.votes[platform] = {}
        fb.votes[platform][vote] = fb.votes[platform].get(vote, 0) + 1


def record_error(platform: str, error_type: str) -> None:
    """Record an error."""
    if PROMETHEUS_AVAILABLE:
        SOCIAL_ERRORS_TOTAL.labels(platform=platform, error_type=error_type).inc()
    else:
        fb = _get_fallback_metrics()
        if platform not in fb.errors:
            fb.errors[platform] = {}
        fb.errors[platform][error_type] = fb.errors[platform].get(error_type, 0) + 1


def record_api_call(platform: str, method: str, status: str) -> None:
    """Record an outbound API call."""
    if PROMETHEUS_AVAILABLE:
        SOCIAL_API_CALLS_TOTAL.labels(platform=platform, method=method, status=status).inc()
    else:
        fb = _get_fallback_metrics()
        if platform not in fb.api_calls:
            fb.api_calls[platform] = {}
        if method not in fb.api_calls[platform]:
            fb.api_calls[platform][method] = {}
        fb.api_calls[platform][method][status] = fb.api_calls[platform][method].get(status, 0) + 1


def record_api_latency(platform: str, method: str, latency_seconds: float) -> None:
    """Record outbound API call latency."""
    if PROMETHEUS_AVAILABLE:
        SOCIAL_API_LATENCY.labels(platform=platform, method=method).observe(latency_seconds)
    else:
        fb = _get_fallback_metrics()
        if platform not in fb.api_latencies:
            fb.api_latencies[platform] = {}
        if method not in fb.api_latencies[platform]:
            fb.api_latencies[platform][method] = []
        fb.api_latencies[platform][method].append(latency_seconds)
        # Keep only last 1000 samples
        if len(fb.api_latencies[platform][method]) > 1000:
            fb.api_latencies[platform][method] = fb.api_latencies[platform][method][-1000:]


# ============================================================================
# Decorators
# ============================================================================


def with_webhook_metrics(platform: str) -> Callable:
    """Decorator to add webhook metrics to a handler method."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            status = "success"
            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                status = "error"
                record_error(platform, "handler_exception")
                raise
            finally:
                latency = time.time() - start_time
                record_webhook_request(platform, status)
                record_webhook_latency(platform, latency)

        return wrapper

    return decorator


def with_api_metrics(platform: str, method: str) -> Callable:
    """Decorator to add API call metrics to async methods."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            status = "success"
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception:
                status = "error"
                raise
            finally:
                latency = time.time() - start_time
                record_api_call(platform, method, status)
                record_api_latency(platform, method, latency)

        return wrapper

    return decorator


# ============================================================================
# Utility Functions
# ============================================================================


def get_metrics_summary() -> Dict[str, Any]:
    """Get a summary of all social handler metrics."""
    if PROMETHEUS_AVAILABLE:
        # When prometheus is available, metrics are exported via /metrics endpoint
        return {"prometheus_available": True, "export_endpoint": "/metrics"}
    else:
        fb = _get_fallback_metrics()
        return {
            "prometheus_available": False,
            "webhook_requests": fb.webhook_requests,
            "messages": fb.messages,
            "commands": fb.commands,
            "debates_started": fb.debates_started,
            "debates_completed": fb.debates_completed,
            "debates_failed": fb.debates_failed,
            "debates_in_progress": fb.debates_in_progress,
            "gauntlets_started": fb.gauntlets_started,
            "gauntlets_completed": fb.gauntlets_completed,
            "gauntlets_failed": fb.gauntlets_failed,
            "votes": fb.votes,
            "errors": fb.errors,
            "api_calls": fb.api_calls,
        }


def reset_fallback_metrics() -> None:
    """Reset fallback metrics (for testing)."""
    global _fallback_metrics
    _fallback_metrics = None


__all__ = [
    "PROMETHEUS_AVAILABLE",
    "record_webhook_request",
    "record_webhook_latency",
    "record_message",
    "record_command",
    "record_debate_started",
    "record_debate_completed",
    "record_debate_failed",
    "record_gauntlet_started",
    "record_gauntlet_completed",
    "record_gauntlet_failed",
    "record_vote",
    "record_error",
    "record_api_call",
    "record_api_latency",
    "with_webhook_metrics",
    "with_api_metrics",
    "get_metrics_summary",
    "reset_fallback_metrics",
]
