"""
Oracle streaming observability metrics.

Tracks server-side Oracle consultation health signals:
- consultations started/completed/cancelled/errored
- active in-flight consultations
- time-to-first-token (TTFT)
- per-phase duration
- stream stall reasons
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock
from typing import Any

from aragora.observability.metrics.base import (
    NoOpMetric,
    get_metrics_enabled,
    get_or_create_counter,
    get_or_create_gauge,
    get_or_create_histogram,
)

logger = logging.getLogger(__name__)


# Prometheus collectors
ORACLE_STREAM_SESSIONS_TOTAL: Any = None
ORACLE_STREAM_ACTIVE_SESSIONS: Any = None
ORACLE_STREAM_TTFT_SECONDS: Any = None
ORACLE_STREAM_PHASE_DURATION_SECONDS: Any = None
ORACLE_STREAM_STALLS_TOTAL: Any = None


@dataclass
class _OracleSnapshot:
    sessions_started: int = 0
    sessions_completed: int = 0
    sessions_cancelled: int = 0
    sessions_errors: int = 0
    active_sessions: int = 0
    stalls_waiting_first_token: int = 0
    stalls_stream_inactive: int = 0
    ttft_samples: int = 0
    ttft_total_seconds: float = 0.0
    ttft_last_seconds: float | None = None


_snapshot = _OracleSnapshot()
_snapshot_lock = Lock()
_initialized = False

_VALID_OUTCOMES = {"completed", "cancelled", "error"}
_VALID_STALL_REASONS = {"waiting_first_token", "stream_inactive"}


def init_oracle_metrics() -> None:
    """Initialize Oracle streaming metrics."""
    global _initialized
    global ORACLE_STREAM_SESSIONS_TOTAL
    global ORACLE_STREAM_ACTIVE_SESSIONS
    global ORACLE_STREAM_TTFT_SECONDS
    global ORACLE_STREAM_PHASE_DURATION_SECONDS
    global ORACLE_STREAM_STALLS_TOTAL

    if _initialized:
        return

    if not get_metrics_enabled():
        _init_noop_metrics()
        _initialized = True
        return

    try:
        ORACLE_STREAM_SESSIONS_TOTAL = get_or_create_counter(
            "aragora_oracle_stream_sessions_total",
            "Total Oracle consultations by outcome",
            ["outcome"],
        )
        ORACLE_STREAM_ACTIVE_SESSIONS = get_or_create_gauge(
            "aragora_oracle_stream_active_sessions",
            "Current number of in-flight Oracle consultations",
        )
        ORACLE_STREAM_TTFT_SECONDS = get_or_create_histogram(
            "aragora_oracle_ttft_seconds",
            "Oracle time-to-first-token in seconds",
            ["phase"],
            buckets=[0.1, 0.25, 0.5, 1, 1.5, 2, 3, 5, 8, 13, 21, 34],
        )
        ORACLE_STREAM_PHASE_DURATION_SECONDS = get_or_create_histogram(
            "aragora_oracle_phase_duration_seconds",
            "Oracle streaming phase duration in seconds",
            ["phase"],
            buckets=[0.25, 0.5, 1, 2, 3, 5, 8, 13, 21, 34, 55],
        )
        ORACLE_STREAM_STALLS_TOTAL = get_or_create_counter(
            "aragora_oracle_stream_stalls_total",
            "Oracle stream stalls by reason and phase",
            ["reason", "phase"],
        )
        _initialized = True
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning("Failed to initialize Oracle metrics: %s", exc)
        _init_noop_metrics()
        _initialized = True


def _init_noop_metrics() -> None:
    """Initialize no-op collectors when Prometheus is unavailable/disabled."""
    global ORACLE_STREAM_SESSIONS_TOTAL
    global ORACLE_STREAM_ACTIVE_SESSIONS
    global ORACLE_STREAM_TTFT_SECONDS
    global ORACLE_STREAM_PHASE_DURATION_SECONDS
    global ORACLE_STREAM_STALLS_TOTAL

    ORACLE_STREAM_SESSIONS_TOTAL = NoOpMetric()
    ORACLE_STREAM_ACTIVE_SESSIONS = NoOpMetric()
    ORACLE_STREAM_TTFT_SECONDS = NoOpMetric()
    ORACLE_STREAM_PHASE_DURATION_SECONDS = NoOpMetric()
    ORACLE_STREAM_STALLS_TOTAL = NoOpMetric()


def _ensure_init() -> None:
    if not _initialized:
        init_oracle_metrics()


def record_oracle_session_started() -> None:
    """Record the start of an Oracle consultation."""
    _ensure_init()
    ORACLE_STREAM_SESSIONS_TOTAL.labels(outcome="started").inc()
    with _snapshot_lock:
        _snapshot.sessions_started += 1
        _snapshot.active_sessions += 1
        ORACLE_STREAM_ACTIVE_SESSIONS.set(_snapshot.active_sessions)


def record_oracle_session_outcome(outcome: str) -> None:
    """Record final outcome for an Oracle consultation."""
    _ensure_init()
    normalized = outcome if outcome in _VALID_OUTCOMES else "error"
    ORACLE_STREAM_SESSIONS_TOTAL.labels(outcome=normalized).inc()

    with _snapshot_lock:
        if _snapshot.active_sessions > 0:
            _snapshot.active_sessions -= 1

        if normalized == "completed":
            _snapshot.sessions_completed += 1
        elif normalized == "cancelled":
            _snapshot.sessions_cancelled += 1
        else:
            _snapshot.sessions_errors += 1

        ORACLE_STREAM_ACTIVE_SESSIONS.set(_snapshot.active_sessions)


def record_oracle_time_to_first_token(phase: str, latency_seconds: float) -> None:
    """Record Oracle time-to-first-token for a phase."""
    _ensure_init()
    safe_phase = phase or "unknown"
    safe_latency = max(latency_seconds, 0.0)
    ORACLE_STREAM_TTFT_SECONDS.labels(phase=safe_phase).observe(safe_latency)
    with _snapshot_lock:
        _snapshot.ttft_samples += 1
        _snapshot.ttft_total_seconds += safe_latency
        _snapshot.ttft_last_seconds = safe_latency


def record_oracle_stream_phase_duration(phase: str, duration_seconds: float) -> None:
    """Record full phase duration for an Oracle stream phase."""
    _ensure_init()
    safe_phase = phase or "unknown"
    ORACLE_STREAM_PHASE_DURATION_SECONDS.labels(phase=safe_phase).observe(
        max(duration_seconds, 0.0)
    )


def record_oracle_stream_stall(reason: str, *, phase: str = "unknown") -> None:
    """Record an Oracle stream stall reason."""
    _ensure_init()
    normalized_reason = reason if reason in _VALID_STALL_REASONS else "stream_inactive"
    safe_phase = phase or "unknown"
    ORACLE_STREAM_STALLS_TOTAL.labels(reason=normalized_reason, phase=safe_phase).inc()

    with _snapshot_lock:
        if normalized_reason == "waiting_first_token":
            _snapshot.stalls_waiting_first_token += 1
        else:
            _snapshot.stalls_stream_inactive += 1


def get_oracle_stream_metrics_summary() -> dict[str, Any]:
    """Return a lightweight aggregate summary for dashboard consumption."""
    _ensure_init()
    with _snapshot_lock:
        ttft_avg_ms = (
            round((_snapshot.ttft_total_seconds / _snapshot.ttft_samples) * 1000, 1)
            if _snapshot.ttft_samples > 0
            else None
        )
        ttft_last_ms = (
            round(_snapshot.ttft_last_seconds * 1000, 1)
            if _snapshot.ttft_last_seconds is not None
            else None
        )

        return {
            "sessions_started": _snapshot.sessions_started,
            "sessions_completed": _snapshot.sessions_completed,
            "sessions_cancelled": _snapshot.sessions_cancelled,
            "sessions_errors": _snapshot.sessions_errors,
            "active_sessions": _snapshot.active_sessions,
            "stalls_waiting_first_token": _snapshot.stalls_waiting_first_token,
            "stalls_stream_inactive": _snapshot.stalls_stream_inactive,
            "stalls_total": _snapshot.stalls_waiting_first_token + _snapshot.stalls_stream_inactive,
            "ttft_samples": _snapshot.ttft_samples,
            "ttft_avg_ms": ttft_avg_ms,
            "ttft_last_ms": ttft_last_ms,
            "available": True,
        }


__all__ = [
    "ORACLE_STREAM_SESSIONS_TOTAL",
    "ORACLE_STREAM_ACTIVE_SESSIONS",
    "ORACLE_STREAM_TTFT_SECONDS",
    "ORACLE_STREAM_PHASE_DURATION_SECONDS",
    "ORACLE_STREAM_STALLS_TOTAL",
    "init_oracle_metrics",
    "record_oracle_session_started",
    "record_oracle_session_outcome",
    "record_oracle_time_to_first_token",
    "record_oracle_stream_phase_duration",
    "record_oracle_stream_stall",
    "get_oracle_stream_metrics_summary",
]
