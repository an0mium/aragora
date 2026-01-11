"""
Operational metrics endpoint handlers.

Endpoints:
- GET /api/metrics - Get operational metrics for monitoring
- GET /api/metrics/health - Detailed health check
- GET /api/metrics/cache - Cache statistics
- GET /api/metrics/verification - Z3 formal verification statistics
- GET /api/metrics/system - System information
- GET /metrics - Prometheus-format metrics (OpenMetrics)
"""

from __future__ import annotations

import logging
import os
import platform
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

from aragora.config import DB_ELO_PATH, DB_INSIGHTS_PATH
from .base import BaseHandler, HandlerResult, json_response, error_response, safe_error_message
from .cache import _cache, get_cache_stats
from ..prometheus import (
    get_metrics_output,
    is_prometheus_available,
    set_cache_size,
    set_server_info,
)


# Request tracking for metrics (thread-safe)
_request_counts: dict[str, int] = {}
_error_counts: dict[str, int] = {}
_metrics_lock = threading.Lock()
_start_time = time.time()

# Verification metrics tracking (thread-safe)
_verification_stats: dict[str, int | float] = {
    "total_claims_processed": 0,
    "z3_verified": 0,
    "z3_disproved": 0,
    "z3_timeout": 0,
    "z3_translation_failed": 0,
    "confidence_fallback": 0,
    "total_verification_time_ms": 0.0,
}
_verification_lock = threading.Lock()


def track_verification(
    status: str,
    verification_time_ms: float = 0.0,
) -> None:
    """Track verification outcome (thread-safe).

    Args:
        status: One of 'z3_verified', 'z3_disproved', 'z3_timeout',
                'z3_translation_failed', 'confidence_fallback'
        verification_time_ms: Time taken for verification in milliseconds
    """
    with _verification_lock:
        _verification_stats["total_claims_processed"] += 1
        if status in _verification_stats:
            _verification_stats[status] += 1
        _verification_stats["total_verification_time_ms"] += verification_time_ms


def get_verification_stats() -> dict:
    """Get verification statistics (thread-safe snapshot)."""
    with _verification_lock:
        stats = dict(_verification_stats)

    # Calculate derived metrics
    total = stats["total_claims_processed"]
    if total > 0:
        stats["avg_verification_time_ms"] = round(
            stats["total_verification_time_ms"] / total, 2
        )
        stats["z3_success_rate"] = round(
            stats["z3_verified"] / total, 4
        )
    else:
        stats["avg_verification_time_ms"] = 0.0
        stats["z3_success_rate"] = 0.0

    return stats


def track_request(endpoint: str, is_error: bool = False):
    """Track a request for metrics (thread-safe)."""
    with _metrics_lock:
        _request_counts[endpoint] = _request_counts.get(endpoint, 0) + 1
        if is_error:
            _error_counts[endpoint] = _error_counts.get(endpoint, 0) + 1


class MetricsHandler(BaseHandler):
    """Handler for operational metrics endpoints."""

    ROUTES = [
        "/api/metrics",
        "/api/metrics/health",
        "/api/metrics/cache",
        "/api/metrics/verification",
        "/api/metrics/system",
        "/api/metrics/background",
        "/metrics",  # Prometheus-format endpoint
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route metrics requests to appropriate methods."""
        if path == "/api/metrics":
            return self._get_metrics()

        if path == "/api/metrics/health":
            return self._get_health()

        if path == "/api/metrics/cache":
            return self._get_cache_stats()

        if path == "/api/metrics/verification":
            return self._get_verification_stats()

        if path == "/api/metrics/system":
            return self._get_system_info()

        if path == "/api/metrics/background":
            return self._get_background_stats()

        if path == "/metrics":
            return self._get_prometheus_metrics()

        return None

    def _get_prometheus_metrics(self) -> HandlerResult:
        """Get metrics in Prometheus/OpenMetrics format."""
        try:
            import sys

            # Update cache size metric
            set_cache_size("handler_cache", len(_cache))

            # Update server info
            set_server_info(
                version="0.07",
                python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                start_time=_start_time,
            )

            # Generate output
            content, content_type = get_metrics_output()

            return HandlerResult(
                status_code=200,
                content_type=content_type,
                body=content.encode("utf-8"),
            )
        except Exception as e:
            logger.error("Failed to get Prometheus metrics: %s", e, exc_info=True)
            return error_response(safe_error_message(e, "get Prometheus metrics"), 500)

    def _get_metrics(self) -> HandlerResult:
        """Get comprehensive operational metrics."""
        try:
            uptime = time.time() - _start_time

            # Calculate request rates (thread-safe snapshot)
            with _metrics_lock:
                total_requests = sum(_request_counts.values())
                total_errors = sum(_error_counts.values())
                # Take snapshot of counts for sorting
                counts_snapshot = list(_request_counts.items())

            error_rate = total_errors / total_requests if total_requests > 0 else 0.0

            # Top endpoints by request count
            top_endpoints = sorted(
                counts_snapshot,
                key=lambda x: x[1],
                reverse=True
            )[:10]

            # Database sizes
            db_stats = self._get_database_sizes()

            # Cache stats
            cache_size = len(_cache)

            metrics = {
                "uptime_seconds": round(uptime, 1),
                "uptime_human": self._format_uptime(uptime),
                "requests": {
                    "total": total_requests,
                    "errors": total_errors,
                    "error_rate": round(error_rate, 4),
                    "top_endpoints": [
                        {"endpoint": ep, "count": count}
                        for ep, count in top_endpoints
                    ],
                },
                "cache": {
                    "entries": cache_size,
                },
                "databases": db_stats,
                "timestamp": datetime.now().isoformat(),
            }

            return json_response(metrics)
        except Exception as e:
            logger.error("Failed to get operational metrics: %s", e, exc_info=True)
            return error_response(safe_error_message(e, "get metrics"), 500)

    def _get_health(self) -> HandlerResult:
        """Get detailed health check status."""
        try:
            checks: dict[str, dict[str, Any]] = {}
            status = "healthy"

            # Check storage
            storage = self.get_storage()
            if storage:
                try:
                    storage.list_debates(limit=1)
                    checks["storage"] = {"status": "healthy"}
                except Exception as e:
                    checks["storage"] = {"status": "unhealthy", "error": str(e)}
                    status = "degraded"
            else:
                checks["storage"] = {"status": "unavailable"}

            # Check ELO system
            elo = self.get_elo_system()
            if elo:
                try:
                    elo.get_leaderboard(limit=1)
                    checks["elo_system"] = {"status": "healthy"}
                except Exception as e:
                    checks["elo_system"] = {"status": "unhealthy", "error": str(e)}
                    status = "degraded"
            else:
                checks["elo_system"] = {"status": "unavailable"}

            # Check nomic directory
            nomic_dir = self.get_nomic_dir()
            if nomic_dir and nomic_dir.exists():
                checks["nomic_dir"] = {
                    "status": "healthy",
                    "path": str(nomic_dir),
                }
            else:
                checks["nomic_dir"] = {"status": "unavailable"}

            health: dict[str, Any] = {
                "status": status,
                "checks": checks,
            }

            # Overall status code
            status_code = 200 if health["status"] == "healthy" else 503

            return json_response(health, status=status_code)
        except Exception as e:
            logger.error("Health check failed: %s", e, exc_info=True)
            return error_response(safe_error_message(e, "health check"), 500)

    def _get_cache_stats(self) -> HandlerResult:
        """Get cache statistics."""
        try:
            now = time.time()

            # Get basic stats from cache
            cache_stats = get_cache_stats()

            # Analyze cache entries by prefix
            entries_by_prefix: dict[str, int] = {}
            oldest_entry = now
            newest_entry = 0

            for key, (cached_time, _) in _cache.items():
                # Extract prefix from cache key
                prefix = key.split(":")[0] if ":" in key else "default"
                entries_by_prefix[prefix] = entries_by_prefix.get(prefix, 0) + 1

                if cached_time < oldest_entry:
                    oldest_entry = cached_time
                if cached_time > newest_entry:
                    newest_entry = cached_time

            stats = {
                "total_entries": cache_stats["entries"],
                "max_entries": cache_stats["max_entries"],
                "hit_rate": round(cache_stats["hit_rate"], 4),
                "hits": cache_stats["hits"],
                "misses": cache_stats["misses"],
                "entries_by_prefix": entries_by_prefix,
                "oldest_entry_age_seconds": round(now - oldest_entry, 1) if len(_cache) > 0 else 0,
                "newest_entry_age_seconds": round(now - newest_entry, 1) if len(_cache) > 0 else 0,
            }

            return json_response(stats)
        except Exception as e:
            logger.error("Failed to get cache stats: %s", e, exc_info=True)
            return error_response(safe_error_message(e, "get cache stats"), 500)

    def _get_verification_stats(self) -> HandlerResult:
        """Get formal verification statistics.

        Returns metrics on Z3 claim verification including:
        - Total claims processed
        - Z3 verification outcomes (verified, disproved, timeout, failed)
        - Confidence fallback count
        - Average verification time
        """
        try:
            stats = get_verification_stats()
            return json_response(stats)
        except Exception as e:
            logger.error("Failed to get verification stats: %s", e, exc_info=True)
            return error_response(safe_error_message(e, "get verification stats"), 500)

    def _get_system_info(self) -> HandlerResult:
        """Get system information."""
        try:
            import sys

            info = {
                "python_version": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "pid": os.getpid(),
            }

            # Memory usage (if psutil available)
            try:
                import psutil
                process = psutil.Process()
                info["memory"] = {
                    "rss_mb": round(process.memory_info().rss / 1024 / 1024, 2),
                    "vms_mb": round(process.memory_info().vms / 1024 / 1024, 2),
                }
            except ImportError:
                info["memory"] = {"available": False, "reason": "psutil not installed"}

            return json_response(info)
        except Exception as e:
            logger.error("Failed to get system info: %s", e, exc_info=True)
            return error_response(safe_error_message(e, "get system info"), 500)

    def _get_background_stats(self) -> HandlerResult:
        """Get background task statistics."""
        try:
            from aragora.server.background import get_background_manager
            manager = get_background_manager()
            stats = manager.get_stats()
            return json_response(stats)
        except ImportError:
            return json_response({
                "running": False,
                "task_count": 0,
                "tasks": {},
                "message": "Background task manager not available",
            })
        except Exception as e:
            logger.error("Failed to get background stats: %s", e, exc_info=True)
            return error_response(safe_error_message(e, "get background stats"), 500)

    def _get_database_sizes(self) -> dict[str, Any]:
        """Get database file sizes."""
        sizes: dict[str, dict[str, Any]] = {}
        nomic_dir = self.get_nomic_dir()

        if not nomic_dir or not nomic_dir.exists():
            return sizes

        # Common database files
        db_files = [
            DB_ELO_PATH,
            "debate_storage.db",
            "debate_embeddings.db",
            DB_INSIGHTS_PATH,
            "continuum_memory.db",
            "grounded_positions.db",
        ]

        for db_file in db_files:
            db_path = nomic_dir / db_file
            if db_path.exists():
                size_bytes = db_path.stat().st_size
                sizes[db_file] = {
                    "bytes": size_bytes,
                    "human": self._format_size(size_bytes),
                }

        return sizes

    def _format_uptime(self, seconds: float) -> str:
        """Format uptime as human-readable string."""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

    def _format_size(self, size_bytes: int) -> str:
        """Format size as human-readable string."""
        size_float = float(size_bytes)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_float < 1024:
                return f"{size_float:.1f} {unit}"
            size_float /= 1024
        return f"{size_float:.1f} TB"
