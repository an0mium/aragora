"""Observability dashboard endpoint.

Endpoints:
- GET /api/observability/dashboard - Aggregated system observability metrics

Returns debate performance, agent rankings, circuit breaker states,
self-improvement cycle status, and error rates from available subsystems.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from aragora.server.versioning.compat import strip_version_prefix

from ..base import HandlerResult, json_response
from ..secure import SecureHandler
from ..utils.auth_mixins import SecureEndpointMixin, require_permission
from ..utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


_ALERT_SETTLEMENT_SUCCESS_RATE_MIN = _env_float(
    "ARAGORA_ALERT_SETTLEMENT_SUCCESS_RATE_MIN", 0.95
)
_ALERT_SETTLEMENT_UNRESOLVED_DUE_MAX = _env_int(
    "ARAGORA_ALERT_SETTLEMENT_UNRESOLVED_DUE_MAX", 10
)
_ALERT_ORACLE_STALLS_TOTAL_MAX = _env_int("ARAGORA_ALERT_ORACLE_STALLS_TOTAL_MAX", 5)
_ALERT_ORACLE_TTFT_AVG_MS_MAX = _env_float("ARAGORA_ALERT_ORACLE_TTFT_AVG_MS_MAX", 2000.0)
_ALERT_ORACLE_TTFT_MIN_SAMPLES = _env_int("ARAGORA_ALERT_ORACLE_TTFT_MIN_SAMPLES", 5)


class ObservabilityDashboardHandler(SecureEndpointMixin, SecureHandler):  # type: ignore[misc]
    """Aggregated observability dashboard handler.

    Collects metrics from available subsystems with graceful fallbacks
    when individual systems are unavailable.

    RBAC Permissions:
    - observability:read - View dashboard metrics
    """

    RESOURCE_TYPE = "observability"

    ROUTES = [
        "/api/observability/dashboard",
        "/api/observability/metrics",
    ]

    def __init__(self, server_context: dict[str, Any]) -> None:
        super().__init__(server_context)
        self._elo = server_context.get("elo_system")
        self._storage = server_context.get("storage")

    def can_handle(self, path: str, method: str = "GET") -> bool:
        path = strip_version_prefix(path)
        return path in self.ROUTES

    @require_permission("observability:read")
    @rate_limit(requests_per_minute=30)
    async def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        path = strip_version_prefix(path)
        if path in self.ROUTES:
            return self._get_dashboard()
        return None

    def _get_dashboard(self) -> HandlerResult:
        """Build aggregated dashboard payload."""
        t0 = time.monotonic()
        oracle_stream = self._collect_oracle_stream()
        settlement_review = self._collect_settlement_review()

        data: dict[str, Any] = {
            "timestamp": time.time(),
            "debate_metrics": self._collect_debate_metrics(),
            "agent_rankings": self._collect_agent_rankings(),
            "circuit_breakers": self._collect_circuit_breakers(),
            "self_improve": self._collect_self_improve(),
            "oracle_stream": oracle_stream,
            "settlement_review": settlement_review,
            "alerts": self._collect_operational_alerts(oracle_stream, settlement_review),
            "system_health": self._collect_system_health(),
            "error_rates": self._collect_error_rates(),
        }

        elapsed_ms = (time.monotonic() - t0) * 1000
        data["collection_time_ms"] = round(elapsed_ms, 1)

        return json_response(data)

    # ------------------------------------------------------------------
    # Debate metrics
    # ------------------------------------------------------------------

    def _collect_debate_metrics(self) -> dict[str, Any]:
        """Collect debate count, avg duration, consensus rate."""
        fallback = {
            "total_debates": 0,
            "avg_duration_seconds": 0,
            "consensus_rate": 0,
            "available": False,
        }
        if not self._storage:
            return fallback
        try:
            debates = self._storage.list_debates() if hasattr(self._storage, "list_debates") else []
            total = len(debates)
            if total == 0:
                return {**fallback, "available": True}

            durations = []
            consensus_count = 0
            for d in debates[-100:]:
                meta = d if isinstance(d, dict) else getattr(d, "__dict__", {})
                dur = meta.get("duration") or meta.get("duration_seconds")
                if dur is not None:
                    durations.append(float(dur))
                if meta.get("consensus_reached") or meta.get("consensus"):
                    consensus_count += 1

            sample = len(debates[-100:])
            return {
                "total_debates": total,
                "avg_duration_seconds": round(sum(durations) / len(durations), 1)
                if durations
                else 0,
                "consensus_rate": round(consensus_count / sample, 3) if sample else 0,
                "available": True,
            }
        except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError):
            logger.debug("Failed to collect debate metrics", exc_info=True)
            return fallback

    # ------------------------------------------------------------------
    # Agent rankings (top 10)
    # ------------------------------------------------------------------

    def _collect_agent_rankings(self) -> dict[str, Any]:
        """Collect top 10 agents by ELO rating."""
        fallback: dict[str, Any] = {"top_agents": [], "available": False}
        if not self._elo:
            return fallback
        try:
            lb = (
                self._elo.get_leaderboard(limit=10) if hasattr(self._elo, "get_leaderboard") else []
            )
            agents = []
            for entry in lb:
                if isinstance(entry, dict):
                    agents.append(
                        {
                            "name": entry.get("agent") or entry.get("name", "unknown"),
                            "rating": entry.get("rating") or entry.get("elo", 1500),
                            "matches": entry.get("matches") or entry.get("total_matches", 0),
                            "win_rate": entry.get("win_rate", 0),
                        }
                    )
                else:
                    agents.append(
                        {
                            "name": getattr(entry, "agent", getattr(entry, "name", "unknown")),
                            "rating": getattr(entry, "rating", getattr(entry, "elo", 1500)),
                            "matches": getattr(
                                entry, "matches", getattr(entry, "total_matches", 0)
                            ),
                            "win_rate": getattr(entry, "win_rate", 0),
                        }
                    )
            return {"top_agents": agents, "available": True}
        except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError):
            logger.debug("Failed to collect agent rankings", exc_info=True)
            return fallback

    # ------------------------------------------------------------------
    # Circuit breaker states
    # ------------------------------------------------------------------

    def _collect_circuit_breakers(self) -> dict[str, Any]:
        """Collect circuit breaker states from the resilience registry."""
        fallback: dict[str, Any] = {"breakers": [], "available": False}
        try:
            from aragora.resilience.registry import get_circuit_breakers

            all_breakers = get_circuit_breakers()
            breakers = []
            for name, cb in all_breakers.items():
                breakers.append(
                    {
                        "name": name,
                        "state": cb.state if hasattr(cb, "state") else "unknown",
                        "failure_count": getattr(cb, "failure_count", 0),
                        "success_count": getattr(cb, "success_count", 0),
                    }
                )
            return {"breakers": breakers, "available": True}
        except (ImportError, AttributeError):
            pass

        try:
            from aragora.resilience import CircuitBreaker

            if hasattr(CircuitBreaker, "_instances"):
                breakers = []
                for name, cb in CircuitBreaker._instances.items():
                    breakers.append(
                        {
                            "name": name,
                            "state": cb.state if hasattr(cb, "state") else "unknown",
                            "failure_count": getattr(cb, "failure_count", 0),
                        }
                    )
                return {"breakers": breakers, "available": True}
        except (ImportError, AttributeError):
            pass

        return fallback

    # ------------------------------------------------------------------
    # Self-improvement cycles
    # ------------------------------------------------------------------

    def _collect_self_improve(self) -> dict[str, Any]:
        """Collect self-improvement run history."""
        fallback: dict[str, Any] = {
            "total_cycles": 0,
            "successful": 0,
            "failed": 0,
            "recent_runs": [],
            "available": False,
        }
        try:
            from aragora.nomic.stores.run_store import SelfImproveRunStore

            store = SelfImproveRunStore()
            runs = store.list_runs() if hasattr(store, "list_runs") else []
            successful = sum(
                1
                for r in runs
                if (r.get("status") if isinstance(r, dict) else getattr(r, "status", ""))
                == "completed"
            )
            failed = sum(
                1
                for r in runs
                if (r.get("status") if isinstance(r, dict) else getattr(r, "status", ""))
                == "failed"
            )

            recent = []
            for r in runs[-5:]:
                if isinstance(r, dict):
                    recent.append(
                        {
                            "id": r.get("id", ""),
                            "goal": r.get("goal", ""),
                            "status": r.get("status", ""),
                            "started_at": r.get("started_at", ""),
                        }
                    )
                else:
                    recent.append(
                        {
                            "id": getattr(r, "id", ""),
                            "goal": getattr(r, "goal", ""),
                            "status": getattr(r, "status", ""),
                            "started_at": str(getattr(r, "started_at", "")),
                        }
                    )

            return {
                "total_cycles": len(runs),
                "successful": successful,
                "failed": failed,
                "recent_runs": recent,
                "available": True,
            }
        except (ImportError, OSError):
            logger.debug("Self-improve store unavailable", exc_info=True)
            return fallback

    # ------------------------------------------------------------------
    # Oracle streaming metrics
    # ------------------------------------------------------------------

    def _collect_oracle_stream(self) -> dict[str, Any]:
        """Collect Oracle stream TTFT/stall/session metrics."""
        fallback: dict[str, Any] = {
            "sessions_started": 0,
            "sessions_completed": 0,
            "sessions_cancelled": 0,
            "sessions_errors": 0,
            "active_sessions": 0,
            "stalls_waiting_first_token": 0,
            "stalls_stream_inactive": 0,
            "stalls_total": 0,
            "ttft_samples": 0,
            "ttft_avg_ms": None,
            "ttft_last_ms": None,
            "available": False,
        }
        try:
            from aragora.observability.metrics.oracle import (
                get_oracle_stream_metrics_summary,
            )

            return get_oracle_stream_metrics_summary()
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return fallback

    # ------------------------------------------------------------------
    # Settlement review scheduler
    # ------------------------------------------------------------------

    def _collect_settlement_review(self) -> dict[str, Any]:
        """Collect settlement review scheduler status and rollup stats."""
        fallback: dict[str, Any] = {
            "running": False,
            "interval_hours": None,
            "max_receipts_per_run": None,
            "startup_delay_seconds": None,
            "stats": None,
            "available": False,
        }
        try:
            from aragora.scheduler.settlement_review import get_settlement_review_scheduler

            scheduler = get_settlement_review_scheduler()
            if scheduler is None:
                return fallback

            status = scheduler.get_status()
            status["available"] = True
            return status
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return fallback

    def _collect_operational_alerts(
        self,
        oracle_stream: dict[str, Any],
        settlement_review: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate targeted operational threshold alerts for critical quality signals."""
        alerts: list[dict[str, Any]] = []

        if settlement_review.get("available"):
            stats = settlement_review.get("stats")
            if isinstance(stats, dict):
                success_rate = _as_float(stats.get("success_rate"))
                if (
                    success_rate is not None
                    and success_rate < _ALERT_SETTLEMENT_SUCCESS_RATE_MIN
                ):
                    alerts.append(
                        {
                            "id": "settlement_review.success_rate.low",
                            "severity": "warning",
                            "metric": "settlement_review.success_rate",
                            "operator": "<",
                            "threshold": _ALERT_SETTLEMENT_SUCCESS_RATE_MIN,
                            "value": round(success_rate, 4),
                            "message": "Settlement review success rate is below threshold.",
                        }
                    )

                unresolved_due = None
                last_result = stats.get("last_result")
                if isinstance(last_result, dict):
                    unresolved_due = _as_int(last_result.get("unresolved_due"))
                if (
                    unresolved_due is not None
                    and unresolved_due > _ALERT_SETTLEMENT_UNRESOLVED_DUE_MAX
                ):
                    alerts.append(
                        {
                            "id": "settlement_review.stats.last_result.unresolved_due.high",
                            "severity": "warning",
                            "metric": "settlement_review.stats.last_result.unresolved_due",
                            "operator": ">",
                            "threshold": _ALERT_SETTLEMENT_UNRESOLVED_DUE_MAX,
                            "value": unresolved_due,
                            "message": "Settlement review has too many unresolved due receipts.",
                        }
                    )

        if oracle_stream.get("available"):
            stalls_total = _as_int(oracle_stream.get("stalls_total"))
            if stalls_total is not None and stalls_total > _ALERT_ORACLE_STALLS_TOTAL_MAX:
                alerts.append(
                    {
                        "id": "oracle_stream.stalls_total.high",
                        "severity": "warning",
                        "metric": "oracle_stream.stalls_total",
                        "operator": ">",
                        "threshold": _ALERT_ORACLE_STALLS_TOTAL_MAX,
                        "value": stalls_total,
                        "message": "Oracle stream stalls exceeded threshold.",
                    }
                )

            ttft_avg_ms = _as_float(oracle_stream.get("ttft_avg_ms"))
            ttft_samples = _as_int(oracle_stream.get("ttft_samples")) or 0
            if (
                ttft_avg_ms is not None
                and ttft_samples >= _ALERT_ORACLE_TTFT_MIN_SAMPLES
                and ttft_avg_ms > _ALERT_ORACLE_TTFT_AVG_MS_MAX
            ):
                alerts.append(
                    {
                        "id": "oracle_stream.ttft_avg_ms.high",
                        "severity": "warning",
                        "metric": "oracle_stream.ttft_avg_ms",
                        "operator": ">",
                        "threshold": _ALERT_ORACLE_TTFT_AVG_MS_MAX,
                        "value": round(ttft_avg_ms, 1),
                        "message": "Oracle stream average TTFT exceeded threshold.",
                    }
                )

        return {
            "active": alerts,
            "total": len(alerts),
            "available": True,
        }

    # ------------------------------------------------------------------
    # System health
    # ------------------------------------------------------------------

    def _collect_system_health(self) -> dict[str, Any]:
        """Collect basic system health indicators."""
        import os

        try:
            import psutil

            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0)
            return {
                "memory_percent": mem.percent,
                "cpu_percent": cpu,
                "pid": os.getpid(),
                "available": True,
            }
        except ImportError:
            return {
                "memory_percent": None,
                "cpu_percent": None,
                "pid": os.getpid(),
                "available": False,
            }

    # ------------------------------------------------------------------
    # Error rates
    # ------------------------------------------------------------------

    def _collect_error_rates(self) -> dict[str, Any]:
        """Collect error rate info from observability metrics if available."""
        try:
            from aragora.observability.metrics.request import (
                REQUEST_COUNT,
                _ensure_init,
            )

            _ensure_init()
            # REQUEST_COUNT is a Prometheus Counter with labels (method, endpoint, status)
            # or a NoOpMetric if Prometheus is not available
            total = 0
            errors = 0
            if hasattr(REQUEST_COUNT, "collect"):
                for metric_family in REQUEST_COUNT.collect():
                    for sample in metric_family.samples:
                        val = int(sample.value)
                        total += val
                        status = sample.labels.get("status", "200")
                        if status.startswith("5"):
                            errors += val
            rate = round(errors / total, 4) if total > 0 else 0
            return {
                "total_requests": total,
                "total_errors": errors,
                "error_rate": rate,
                "available": True,
            }
        except (ImportError, AttributeError):
            return {
                "total_requests": 0,
                "total_errors": 0,
                "error_rate": 0,
                "available": False,
            }
