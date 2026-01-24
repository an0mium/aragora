"""
System and utility endpoint handlers.

Endpoints:
- GET /api/debug/test - Debug test endpoint
- GET /api/history/cycles - Get cycle history
- GET /api/history/events - Get event history
- GET /api/history/debates - Get debate history
- GET /api/history/summary - Get history summary
- GET /api/system/maintenance?task=<task> - Run database maintenance
- GET /api/auth/stats - Get authentication statistics
- POST /api/auth/revoke - Revoke a token to invalidate it
- GET /api/circuit-breakers - Circuit breaker metrics
- GET /metrics - Prometheus metrics

Note: Health, nomic, and docs endpoints have been extracted to separate handlers:
- HealthHandler: /healthz, /readyz, /api/health/*
- NomicHandler: /api/nomic/*, /api/modes
- DocsHandler: /api/openapi*, /api/docs*, /api/redoc*, /api/postman.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)
from aragora.billing.jwt_auth import extract_user_from_request
from aragora.exceptions import DatabaseError, StorageError

from ..base import (
    SAFE_ID_PATTERN,
    BaseHandler,
    HandlerResult,
    error_response,
    get_clamped_int_param,
    get_string_param,
    handle_errors,
    json_response,
    safe_error_message,
    ttl_cache,
    validate_path_segment,
)
from ..utils.rate_limit import rate_limit
from ..utils.decorators import require_permission

# Cache TTLs for system endpoints (in seconds)
CACHE_TTL_HISTORY = 60  # History queries


class SystemHandler(BaseHandler):
    """Handler for system-related endpoints."""

    ROUTES = [
        # Debug endpoint
        "/api/v1/debug/test",
        # History endpoints
        "/api/v1/history/cycles",
        "/api/v1/history/events",
        "/api/v1/history/debates",
        "/api/v1/history/summary",
        # System maintenance
        "/api/v1/system/maintenance",
        # Auth stats
        "/api/v1/auth/stats",
        "/api/v1/auth/revoke",
        # Resilience monitoring
        "/api/v1/circuit-breakers",
        # Prometheus metrics
        "/metrics",
    ]

    # History endpoints require authentication (can expose debate data)
    HISTORY_ENDPOINTS = [
        "/api/v1/history/cycles",
        "/api/v1/history/events",
        "/api/v1/history/debates",
        "/api/v1/history/summary",
    ]

    # History endpoint configuration: path -> (method_name, default_limit, max_limit)
    _HISTORY_CONFIG: dict[str, tuple[str, int, int]] = {
        "/api/v1/history/cycles": ("_get_history_cycles", 50, 200),
        "/api/v1/history/events": ("_get_history_events", 100, 500),
        "/api/v1/history/debates": ("_get_history_debates", 50, 200),
        "/api/v1/history/summary": ("_get_history_summary", 0, 0),  # No limit param
    }

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def _check_history_auth(self, handler) -> Optional[HandlerResult]:
        """Check authentication for history endpoints.

        History endpoints can expose sensitive debate data and require
        authentication when auth is enabled.

        Returns:
            None if auth passes or is disabled, HandlerResult with 401 if auth fails.
        """
        from aragora.server.auth import auth_config

        # If auth is disabled globally, allow access
        if not auth_config.enabled:
            return None

        # Try JWT/API key auth first
        user_store = getattr(self.__class__, "user_store", None)
        if user_store is None:
            user_store = self.ctx.get("user_store")

        user_ctx = extract_user_from_request(handler, user_store)
        if user_ctx.is_authenticated:
            return None

        # Fall back to legacy API token
        if auth_config.api_token:
            auth_header = (
                handler.headers.get("Authorization", "") if hasattr(handler, "headers") else ""
            )
            token = auth_header[7:] if auth_header.startswith("Bearer ") else None
            if token and auth_config.validate_token(token):
                return None

        return error_response("Authentication required for history endpoints", 401)

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route system requests to appropriate methods.

        Uses dispatch tables to reduce cyclomatic complexity.
        Note: Health, nomic, and docs endpoints are now handled by dedicated handlers.
        """
        # Simple routes with no parameters
        simple_routes = {
            "/api/v1/debug/test": lambda: self._handle_debug_test(handler),
            "/api/v1/auth/stats": lambda: self._get_auth_stats(handler),
            "/metrics": self._get_prometheus_metrics,
            "/api/v1/circuit-breakers": self._get_circuit_breaker_metrics,
        }

        if path in simple_routes:
            return simple_routes[path]()

        # System maintenance (requires admin:system permission)
        if path == "/api/v1/system/maintenance":
            return self._handle_maintenance(query_params, handler)

        # History endpoints (require auth)
        if path in self._HISTORY_CONFIG:
            return self._handle_history_endpoint(path, query_params, handler)

        return None

    def _handle_debug_test(self, handler) -> HandlerResult:
        """Handle debug test endpoint."""
        method = getattr(handler, "command", "GET")
        return json_response({"status": "ok", "method": method, "message": "Modular handler works"})

    @require_permission("admin:system")
    def _handle_maintenance(self, query_params: dict, handler=None, user=None) -> HandlerResult:
        """Handle system maintenance endpoint.

        Requires admin:system permission.
        """
        task = get_string_param(query_params, "task", "status")
        valid_tasks = ("status", "vacuum", "analyze", "checkpoint", "full")
        if task not in valid_tasks:
            return error_response(f"Invalid task. Use: {', '.join(valid_tasks)}", 400)
        return self._run_maintenance(task)

    def _handle_history_endpoint(self, path: str, query_params: dict, handler) -> HandlerResult:
        """Handle history endpoints with common auth and validation pattern."""
        # Require auth for history endpoints
        auth_error = self._check_history_auth(handler)
        if auth_error:
            return auth_error

        # Validate loop_id parameter
        loop_id = get_string_param(query_params, "loop_id")
        if loop_id:
            is_valid, err = validate_path_segment(loop_id, "loop_id", SAFE_ID_PATTERN)
            if not is_valid:
                return error_response(err, 400)

        # Get config for this endpoint
        method_name, default_limit, max_limit = self._HISTORY_CONFIG[path]
        method = getattr(self, method_name)

        # Summary endpoint doesn't use limit
        if max_limit == 0:
            return method(handler, loop_id)

        limit = get_clamped_int_param(query_params, "limit", default_limit, 1, max_limit)
        return method(handler, loop_id, limit)

    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Handle POST requests for auth endpoints."""
        if path == "/api/v1/auth/revoke":
            return self._revoke_token(handler, handler)
        return None

    def _load_filtered_json(
        self, file_path: Path, loop_id: Optional[str] = None, limit: int = 100
    ) -> list:
        """Load JSON file with optional filtering and early termination.

        Filters during load to avoid loading entire large datasets when
        only a subset is needed.

        Args:
            file_path: Path to JSON file
            loop_id: Optional loop_id filter
            limit: Maximum items to return

        Returns:
            List of matching items, limited to `limit` entries

        Raises:
            json.JSONDecodeError: If file contains invalid JSON
            OSError: If file cannot be read
        """
        if not file_path.exists():
            return []

        with open(file_path) as f:
            data = json.load(f)

        if loop_id:
            # Filter with early termination
            filtered = []
            for item in data:
                if item.get("loop_id") == loop_id:
                    filtered.append(item)
                    if len(filtered) >= limit:
                        break
            return filtered
        else:
            return data[:limit]

    @rate_limit(rpm=30, limiter_name="history_cycles")
    @ttl_cache(ttl_seconds=CACHE_TTL_HISTORY, key_prefix="history_cycles", skip_first=True)
    @handle_errors("get cycles")
    def _get_history_cycles(self, handler, loop_id: Optional[str], limit: int) -> HandlerResult:
        """Get cycle history from Supabase or local storage."""
        nomic_dir = self.get_nomic_dir()
        if nomic_dir:
            cycles_file = nomic_dir / "cycles.json"
            cycles = self._load_filtered_json(cycles_file, loop_id, limit)
            return json_response({"cycles": cycles})

        return json_response({"cycles": []})

    @rate_limit(rpm=30, limiter_name="history_events")
    @ttl_cache(ttl_seconds=CACHE_TTL_HISTORY, key_prefix="history_events", skip_first=True)
    @handle_errors("get events")
    def _get_history_events(self, handler, loop_id: Optional[str], limit: int) -> HandlerResult:
        """Get event history."""
        nomic_dir = self.get_nomic_dir()
        if nomic_dir:
            events_file = nomic_dir / "events.json"
            events = self._load_filtered_json(events_file, loop_id, limit)
            return json_response({"events": events})

        return json_response({"events": []})

    @rate_limit(rpm=20, limiter_name="history_debates")
    @ttl_cache(ttl_seconds=CACHE_TTL_HISTORY, key_prefix="history_debates", skip_first=True)
    @handle_errors("get debates")
    def _get_history_debates(self, handler, loop_id: Optional[str], limit: int) -> HandlerResult:
        """Get debate history."""
        storage = self.get_storage()
        if not storage:
            return json_response({"debates": []})

        # When filtering, fetch more to account for non-matching items
        fetch_limit = limit * 3 if loop_id else limit
        debate_metadata = storage.list_recent(limit=fetch_limit)

        if loop_id:
            # Filter with early termination
            debates: list[dict[str, Any]] = []
            for d in debate_metadata:
                item = vars(d)
                if item.get("loop_id") == loop_id:
                    debates.append(item)
                    if len(debates) >= limit:
                        break
        else:
            debates = [vars(d) for d in debate_metadata[:limit]]

        return json_response({"debates": debates})

    @rate_limit(rpm=30, limiter_name="history_summary")
    @ttl_cache(ttl_seconds=CACHE_TTL_HISTORY, key_prefix="history_summary", skip_first=True)
    def _get_history_summary(self, handler, loop_id: Optional[str]) -> HandlerResult:
        """Get history summary statistics."""
        storage = self.get_storage()
        elo = self.get_elo_system()

        summary = {
            "total_debates": 0,
            "total_agents": 0,
            "total_matches": 0,
        }

        try:
            if storage:
                debates = storage.list_recent(limit=1000)
                summary["total_debates"] = len(debates)

            if elo:
                rankings = elo.get_leaderboard(limit=100)
                summary["total_agents"] = len(rankings)

            return json_response(summary)
        except (StorageError, DatabaseError) as e:
            logger.error("Database error getting history summary: %s: %s", type(e).__name__, e)
            return error_response("Database error retrieving history summary", 500)
        except Exception as e:
            logger.error("Failed to get history summary: %s", e, exc_info=True)
            return error_response(safe_error_message(e, "get summary"), 500)

    def _run_maintenance(self, task: str) -> HandlerResult:
        """Run database maintenance tasks.

        Args:
            task: One of 'status', 'vacuum', 'analyze', 'checkpoint', 'full'

        Returns:
            Maintenance results including affected databases and stats.
        """
        nomic_dir = self.get_nomic_dir()
        if not nomic_dir:
            return error_response("Nomic directory not configured", 503)

        try:
            from aragora.maintenance import DatabaseMaintenance

            maintenance = DatabaseMaintenance(nomic_dir)
            result: dict[str, Any] = {"task": task}

            if task == "status":
                result["stats"] = maintenance.get_stats()

            elif task == "checkpoint":
                result["checkpoint"] = maintenance.checkpoint_all_wal()
                result["stats"] = maintenance.get_stats()

            elif task == "analyze":
                result["analyze"] = maintenance.analyze_all()
                result["stats"] = maintenance.get_stats()

            elif task == "vacuum":
                result["vacuum"] = maintenance.vacuum_all()
                result["stats"] = maintenance.get_stats()

            elif task == "full":
                # Run all maintenance tasks
                result["checkpoint"] = maintenance.checkpoint_all_wal()
                result["analyze"] = maintenance.analyze_all()
                result["vacuum"] = maintenance.vacuum_all()
                result["stats"] = maintenance.get_stats()

            return json_response(result)

        except ImportError:
            return error_response("Maintenance module not available", 503)
        except (StorageError, DatabaseError) as e:
            logger.error(
                "Database error during maintenance '%s': %s: %s", task, type(e).__name__, e
            )
            return error_response(f"Database error during maintenance task '{task}'", 500)
        except OSError as e:
            logger.error("Filesystem error during maintenance '%s': %s", task, e)
            return error_response(f"Filesystem error during maintenance task '{task}'", 500)
        except Exception as e:
            logger.exception(f"Maintenance task '{task}' failed: {e}")
            return error_response(safe_error_message(e, "maintenance"), 500)

    @require_permission("admin:security")
    def _get_auth_stats(self, handler=None, user=None) -> HandlerResult:
        """Get authentication and rate limiting statistics.

        Requires admin:security permission.

        Returns:
            enabled: Whether auth is enabled
            rate_limit_stats: Current rate limiting counters
            revoked_tokens: Number of revoked tokens being tracked
        """
        from aragora.server.auth import auth_config

        stats = auth_config.get_rate_limit_stats()

        return json_response(
            {
                "enabled": auth_config.enabled,
                "rate_limit_per_minute": auth_config.rate_limit_per_minute,
                "ip_rate_limit_per_minute": auth_config.ip_rate_limit_per_minute,
                "token_ttl_seconds": auth_config.token_ttl,
                "stats": stats,
            }
        )

    @require_permission("admin:security")
    def _revoke_token(self, handler, _handler=None, user=None) -> HandlerResult:
        """Revoke a token to invalidate it immediately.

        Requires admin:security permission.

        POST body:
            token: The token to revoke (required)
            reason: Optional reason for revocation

        Returns:
            success: Whether revocation succeeded
            revoked_count: Total revoked tokens being tracked
        """
        from aragora.server.auth import auth_config

        # Read request body
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        token = body.get("token")
        if not token:
            return error_response("Token is required", 400)

        reason = body.get("reason", "")

        # Revoke the token using both in-memory and persistent backends
        success = auth_config.revoke_token(token, reason)

        # Also persist revocation for multi-instance consistency
        try:
            from aragora.billing.jwt_auth import revoke_token_persistent

            persistent_ok = revoke_token_persistent(token)
            if not persistent_ok:
                logger.warning("Token revoked in-memory but persistent revocation failed")
        except ImportError:
            logger.debug("Persistent token blacklist not available")

        if success:
            logger.info("Token revoked: reason=%s", reason or "not specified")

        return json_response(
            {
                "success": success,
                "revoked_count": auth_config.get_revocation_count(),
            }
        )

    def _get_prometheus_metrics(self) -> HandlerResult:
        """Get Prometheus-format metrics.

        Exposes metrics for:
        - Subscription events and active subscriptions by tier
        - Usage (debates, tokens) by tier
        - API request counts and latency
        - Agent performance metrics

        Returns:
            Prometheus text format metrics
        """
        try:
            from aragora.server.metrics import generate_metrics

            metrics_text = generate_metrics()
            return HandlerResult(
                status_code=200,
                content_type="text/plain; version=0.0.4; charset=utf-8",
                body=metrics_text.encode("utf-8"),
            )
        except ImportError:
            return error_response("Metrics module not available", 503)
        except Exception as e:
            logger.exception(f"Metrics generation failed: {e}")
            return error_response(safe_error_message(e, "metrics"), 500)

    def _get_circuit_breaker_metrics(self) -> HandlerResult:
        """Get circuit breaker metrics for monitoring.

        Returns comprehensive metrics for all registered circuit breakers:
        - summary: Aggregate counts (total, open, closed, half-open)
        - circuit_breakers: Per-circuit-breaker details with timing info
        - health: Overall health status and warnings for cascading failure detection

        Health status values:
        - healthy: All circuits closed
        - degraded: 1-2 circuits open
        - critical: 3+ circuits open (potential cascading failure)

        Returns:
            JSON metrics suitable for Prometheus/Grafana integration
        """
        try:
            from aragora.resilience import get_circuit_breaker_metrics

            metrics = get_circuit_breaker_metrics()
            return json_response(metrics)
        except ImportError:
            return error_response("Resilience module not available", 503)
        except Exception as e:
            logger.exception(f"Circuit breaker metrics failed: {e}")
            return error_response(safe_error_message(e, "circuit breaker metrics"), 500)
