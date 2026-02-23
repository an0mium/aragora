"""Observability dashboard endpoint.

Endpoints:
- GET /api/observability/dashboard - Aggregated system observability metrics

Returns debate performance, agent rankings, circuit breaker states,
self-improvement cycle status, and error rates from available subsystems.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from aragora.server.versioning.compat import strip_version_prefix

from ..base import HandlerResult, json_response
from ..secure import SecureHandler
from ..utils.auth_mixins import SecureEndpointMixin, require_permission
from ..utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)


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

        data: dict[str, Any] = {
            "timestamp": time.time(),
            "debate_metrics": self._collect_debate_metrics(),
            "agent_rankings": self._collect_agent_rankings(),
            "circuit_breakers": self._collect_circuit_breakers(),
            "self_improve": self._collect_self_improve(),
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
            from aragora.resilience.registry import get_registry

            registry = get_registry()
            breakers = []
            for name, cb in registry.get_all().items():
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
            from aragora.observability.metrics.server import (
                get_request_error_count,
                get_request_total_count,
            )

            total = get_request_total_count()
            errors = get_request_error_count()
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
