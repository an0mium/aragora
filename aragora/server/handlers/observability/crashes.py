"""Frontend crash telemetry collector.

Stability: STABLE

Receives crash reports from the frontend error boundary / CrashReporter,
stores them in a bounded in-memory buffer, and exposes admin endpoints
for listing and aggregating crash statistics.

Endpoints:
- POST /api/observability/crashes         - Ingest crash reports
- GET  /api/observability/crashes         - List recent crashes (admin)
- GET  /api/observability/crashes/stats   - Crash frequency stats (admin)
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import deque
from typing import Any

from aragora.server.versioning.compat import strip_version_prefix

from ..base import (
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)
from ..secure import SecureHandler
from ..utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory crash store (bounded)
# ---------------------------------------------------------------------------

MAX_STORED_CRASHES = 1_000
MAX_REPORTS_PER_BATCH = 50

# Module-level store so it persists across requests
_crash_store: deque[dict[str, Any]] = deque(maxlen=MAX_STORED_CRASHES)
_seen_fingerprints: dict[str, int] = {}  # fingerprint -> count
_ingest_count: int = 0
_duplicate_count: int = 0
_rate_limited_count: int = 0

# Rate limit: max ingestions per source IP per minute
_ip_timestamps: dict[str, list[float]] = {}
IP_RATE_LIMIT = 30  # per minute


def get_crash_store() -> deque[dict[str, Any]]:
    """Return the module-level crash store (useful for testing)."""
    return _crash_store


def reset_crash_store() -> None:
    """Reset the crash store and counters (for testing)."""
    global _ingest_count, _duplicate_count, _rate_limited_count
    _crash_store.clear()
    _seen_fingerprints.clear()
    _ip_timestamps.clear()
    _ingest_count = 0
    _duplicate_count = 0
    _rate_limited_count = 0


def _check_ip_rate_limit(ip: str) -> bool:
    """Return True if the IP is within its rate limit."""
    now = time.monotonic()
    timestamps = _ip_timestamps.get(ip, [])
    timestamps = [t for t in timestamps if now - t < 60.0]
    _ip_timestamps[ip] = timestamps
    if len(timestamps) >= IP_RATE_LIMIT:
        return False
    timestamps.append(now)
    return True


def _compute_fingerprint(message: str, stack: str | None) -> str:
    """Compute a stable fingerprint from error message and stack."""
    raw = f"{message}::{(stack or '')[:500]}"
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class CrashTelemetryHandler(SecureHandler):
    """Handler for frontend crash telemetry collection and querying.

    RBAC Permissions:
    - (none) for POST /api/observability/crashes  (public ingestion)
    - observability:read for GET endpoints (admin listing / stats)
    """

    ROUTES = [
        "/api/observability/crashes",
        "/api/observability/crashes/stats",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        path = strip_version_prefix(path)
        return path in self.ROUTES

    # ------------------------------------------------------------------
    # GET
    # ------------------------------------------------------------------

    @handle_errors("crash telemetry read")
    @rate_limit(requests_per_minute=30)
    def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        path = strip_version_prefix(path)

        if path == "/api/observability/crashes/stats":
            return self._get_stats(query_params, handler)

        if path == "/api/observability/crashes":
            return self._list_crashes(query_params, handler)

        return None

    # ------------------------------------------------------------------
    # POST
    # ------------------------------------------------------------------

    @handle_errors("crash telemetry ingest")
    @rate_limit(requests_per_minute=60)
    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        path = strip_version_prefix(path)

        if path != "/api/observability/crashes":
            return None

        return self._ingest(handler)

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def _ingest(self, handler: Any) -> HandlerResult:
        global _ingest_count, _duplicate_count, _rate_limited_count

        body = self.read_json_body(handler, max_size=256 * 1024)  # 256 KB max
        if body is None:
            return error_response("Invalid JSON body", 400)

        # Per-IP rate limit
        client_ip = "unknown"
        if hasattr(handler, "client_address"):
            addr = handler.client_address
            client_ip = addr[0] if isinstance(addr, (tuple, list)) else str(addr)
        elif hasattr(handler, "headers"):
            client_ip = (
                handler.headers.get("X-Forwarded-For", "").split(",")[0].strip()
                or handler.headers.get("X-Real-IP", "unknown")
            )

        if not _check_ip_rate_limit(client_ip):
            _rate_limited_count += 1
            return error_response("Rate limit exceeded", 429)

        reports = body.get("reports", [])
        if not isinstance(reports, list):
            return error_response("'reports' must be a list", 400)

        if len(reports) > MAX_REPORTS_PER_BATCH:
            return error_response(
                f"Batch too large (max {MAX_REPORTS_PER_BATCH})", 400
            )

        accepted = 0
        duplicates = 0

        for report in reports:
            if not isinstance(report, dict):
                continue

            message = str(report.get("message", ""))[:2000]
            stack = report.get("stack")
            if isinstance(stack, str):
                stack = stack[:5000]
            else:
                stack = None

            fingerprint = report.get("fingerprint") or _compute_fingerprint(
                message, stack
            )

            # Deduplicate
            if fingerprint in _seen_fingerprints:
                _seen_fingerprints[fingerprint] += 1
                _duplicate_count += 1
                duplicates += 1
                continue

            _seen_fingerprints[fingerprint] = 1

            entry: dict[str, Any] = {
                "fingerprint": fingerprint,
                "message": message,
                "component_stack": str(report.get("componentStack", ""))[:5000]
                if report.get("componentStack")
                else None,
                "stack": stack,
                "url": str(report.get("url", ""))[:2000],
                "timestamp": str(report.get("timestamp", "")),
                "user_agent": str(report.get("userAgent", ""))[:500],
                "session_id": str(report.get("sessionId", ""))[:200],
                "component_name": str(report.get("componentName", ""))[:200]
                if report.get("componentName")
                else None,
                "received_at": time.time(),
                "client_ip": client_ip,
            }

            _crash_store.append(entry)
            _ingest_count += 1
            accepted += 1

            logger.info(
                "Crash report ingested: fingerprint=%s component=%s message=%.80s",
                fingerprint,
                entry.get("component_name", "unknown"),
                message,
            )

        return json_response(
            {
                "accepted": accepted,
                "duplicates": duplicates,
                "total_stored": len(_crash_store),
            },
            status=202,
        )

    # ------------------------------------------------------------------
    # List (admin)
    # ------------------------------------------------------------------

    def _list_crashes(
        self, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult:
        # Admin check
        _user, err = self.require_admin_or_error(handler)
        if err:
            return err

        limit = min(int(query_params.get("limit", 50)), 200)
        offset = max(int(query_params.get("offset", 0)), 0)

        # Convert deque to list for slicing (most recent first)
        all_crashes = list(reversed(_crash_store))
        page = all_crashes[offset : offset + limit]

        return json_response(
            {
                "crashes": page,
                "total": len(_crash_store),
                "limit": limit,
                "offset": offset,
            }
        )

    # ------------------------------------------------------------------
    # Stats (admin)
    # ------------------------------------------------------------------

    def _get_stats(self, query_params: dict[str, Any], handler: Any) -> HandlerResult:
        # Admin check
        user, err = self.require_admin_or_error(handler)
        if err:
            return err

        now = time.time()
        # Count crashes in last hour / last 24h
        last_hour = sum(
            1
            for c in _crash_store
            if now - c.get("received_at", 0) < 3600
        )
        last_24h = sum(
            1
            for c in _crash_store
            if now - c.get("received_at", 0) < 86400
        )

        # Top fingerprints by occurrence count
        top_fingerprints = sorted(
            _seen_fingerprints.items(), key=lambda kv: kv[1], reverse=True
        )[:10]

        # Top components
        component_counts: dict[str, int] = {}
        for crash in _crash_store:
            comp = crash.get("component_name") or "unknown"
            component_counts[comp] = component_counts.get(comp, 0) + 1
        top_components = sorted(
            component_counts.items(), key=lambda kv: kv[1], reverse=True
        )[:10]

        return json_response(
            {
                "total_ingested": _ingest_count,
                "total_stored": len(_crash_store),
                "total_duplicates": _duplicate_count,
                "total_rate_limited": _rate_limited_count,
                "unique_fingerprints": len(_seen_fingerprints),
                "last_hour": last_hour,
                "last_24h": last_24h,
                "top_fingerprints": [
                    {"fingerprint": fp, "count": cnt} for fp, cnt in top_fingerprints
                ],
                "top_components": [
                    {"component": comp, "count": cnt}
                    for comp, cnt in top_components
                ],
            }
        )
