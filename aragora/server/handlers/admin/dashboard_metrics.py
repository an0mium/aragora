"""
Dashboard metrics calculation utilities.

Extracted from dashboard.py to reduce file size. Contains:
- SQL-based metrics aggregation
- Legacy metrics calculation (for compatibility)
- Debate pattern analysis
- Single-pass batch processing

All functions require admin:metrics:read permission and are rate limited.
"""

from __future__ import annotations

import functools
import logging
import time
from datetime import datetime, timedelta
from typing import Any

from aragora.rbac.checker import get_permission_checker
from aragora.rbac.decorators import PermissionDeniedError
from aragora.rbac.models import AuthorizationContext
from aragora.server.handlers.utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# RBAC permission for admin metrics access
PERM_ADMIN_METRICS_READ = "admin:metrics:read"


def _get_context_from_args_strict(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    context_param: str,
) -> AuthorizationContext | None:
    """Extract AuthorizationContext without relying on patched helpers."""
    if context_param in kwargs and isinstance(kwargs[context_param], AuthorizationContext):
        return kwargs[context_param]

    if args and isinstance(args[0], AuthorizationContext):
        return args[0]
    if len(args) >= 2 and isinstance(args[1], AuthorizationContext):
        return args[1]

    for value in kwargs.values():
        if isinstance(value, AuthorizationContext):
            return value

    for arg in args:
        if hasattr(arg, "_auth_context") and isinstance(arg._auth_context, AuthorizationContext):
            return arg._auth_context

    return None


def require_permission(
    permission_key: str,
    resource_id_param: str | None = None,
    context_param: str = "context",
    checker: Any = None,
    on_denied: Any = None,
):
    """Local strict permission decorator (avoids test auto-auth patches)."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            context = _get_context_from_args_strict(args, kwargs, context_param)
            if context is None:
                raise PermissionDeniedError(
                    f"No AuthorizationContext found for permission check: {permission_key}"
                )

            resource_id: str | None = None
            if resource_id_param:
                raw_resource_id = kwargs.get(resource_id_param)
                if raw_resource_id is not None:
                    resource_id = str(raw_resource_id)
                else:
                    import inspect as _inspect

                    sig = _inspect.signature(func)
                    params = list(sig.parameters.keys())
                    if resource_id_param in params:
                        idx = params.index(resource_id_param)
                        if idx < len(args):
                            resource_id = str(args[idx])

            perm_checker = checker or get_permission_checker()
            decision = perm_checker.check_permission(context, permission_key, resource_id)
            if not decision.allowed:
                if on_denied:
                    on_denied(decision)
                raise PermissionDeniedError("Permission denied", decision)

            return func(*args, **kwargs)

        return wrapper

    return decorator


@require_permission(PERM_ADMIN_METRICS_READ)
@rate_limit(requests_per_minute=30, limiter_name="admin_dashboard_metrics")
def get_summary_metrics_sql(storage: Any, domain: str | None) -> dict[str, Any]:
    """Get summary metrics using SQL aggregation (O(1) memory).

    Args:
        storage: Debate storage instance with connection() method.
        domain: Optional domain filter (currently unused, reserved for future).

    Returns:
        Summary dict with total_debates, consensus_reached, consensus_rate, avg_confidence.
    """
    summary = {
        "total_debates": 0,
        "consensus_reached": 0,
        "consensus_rate": 0.0,
        "avg_confidence": 0.0,
    }

    try:
        with storage.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN consensus_reached THEN 1 ELSE 0 END) as consensus_count,
                    AVG(confidence) as avg_conf
                FROM debates
            """)
            row = cursor.fetchone()
            if row:
                total = row[0] or 0
                consensus_count = row[1] or 0
                avg_conf = row[2]

                summary["total_debates"] = total
                summary["consensus_reached"] = consensus_count
                if total > 0:
                    summary["consensus_rate"] = round(consensus_count / total, 3)
                if avg_conf is not None:
                    summary["avg_confidence"] = round(avg_conf, 3)
    except (KeyError, ValueError, OSError, TypeError, AttributeError) as e:
        logger.warning("SQL summary metrics error: %s: %s", type(e).__name__, e)

    return summary


@require_permission(PERM_ADMIN_METRICS_READ)
@rate_limit(requests_per_minute=30, limiter_name="admin_dashboard_metrics")
def get_recent_activity_sql(storage: Any, hours: int) -> dict[str, Any]:
    """Get recent activity metrics using SQL aggregation.

    Args:
        storage: Debate storage instance with connection() method.
        hours: Time window for recent activity.

    Returns:
        Activity dict with debates_last_period, consensus_last_period, period_hours.
    """
    activity = {
        "debates_last_period": 0,
        "consensus_last_period": 0,
        "period_hours": hours,
    }

    try:
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        with storage.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    COUNT(*) as recent_total,
                    SUM(CASE WHEN consensus_reached THEN 1 ELSE 0 END) as recent_consensus
                FROM debates
                WHERE created_at >= ?
            """,
                (cutoff,),
            )
            row = cursor.fetchone()
            if row:
                activity["debates_last_period"] = row[0] or 0
                activity["consensus_last_period"] = row[1] or 0
    except (KeyError, ValueError, OSError, TypeError, AttributeError) as e:
        logger.warning("SQL recent activity error: %s: %s", type(e).__name__, e)

    return activity


@require_permission(PERM_ADMIN_METRICS_READ)
@rate_limit(requests_per_minute=30, limiter_name="admin_dashboard_metrics")
def get_summary_metrics_legacy(domain: str | None, debates: list) -> dict[str, Any]:
    """Get high-level summary metrics (legacy, kept for compatibility).

    Args:
        domain: Optional domain filter (currently unused).
        debates: List of debate records.

    Returns:
        Summary dict with total_debates, consensus_reached, consensus_rate, avg_confidence.
    """
    summary: dict[str, Any] = {
        "total_debates": 0,
        "consensus_reached": 0,
        "consensus_rate": 0.0,
        "avg_confidence": 0.0,
        "avg_rounds": 0.0,
        "total_tokens_used": 0,
    }

    try:
        if debates:
            total = len(debates)
            consensus_count = sum(1 for d in debates if d.get("consensus_reached"))
            summary["total_debates"] = total
            summary["consensus_reached"] = consensus_count
            if total > 0:
                summary["consensus_rate"] = round(consensus_count / total, 3)

                # Average confidence
                confidences = [d.get("confidence", 0.5) for d in debates if d.get("confidence")]
                if confidences:
                    summary["avg_confidence"] = round(sum(confidences) / len(confidences), 3)
    except (TypeError, ValueError, KeyError, AttributeError) as e:
        logger.warning("Summary metrics error: %s: %s", type(e).__name__, e)

    return summary


@require_permission(PERM_ADMIN_METRICS_READ)
@rate_limit(requests_per_minute=30, limiter_name="admin_dashboard_metrics")
def get_recent_activity_legacy(domain: str | None, hours: int, debates: list) -> dict[str, Any]:
    """Get recent debate activity metrics.

    Args:
        domain: Optional domain filter (currently unused).
        hours: Time window for recent activity.
        debates: List of debate records.

    Returns:
        Activity dict with debates_last_period, consensus_last_period, domains_active,
        most_active_domain, period_hours.
    """
    activity: dict[str, Any] = {
        "debates_last_period": 0,
        "consensus_last_period": 0,
        "domains_active": [],
        "most_active_domain": None,
        "period_hours": hours,
    }

    try:
        if debates:
            cutoff = datetime.now() - timedelta(hours=hours)

            recent: list[dict] = []
            domain_counts: dict[str, int] = {}
            for d in debates:
                created_at = d.get("created_at")
                if created_at:
                    # Parse ISO timestamp
                    try:
                        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        if dt.replace(tzinfo=None) > cutoff:
                            recent.append(d)
                            d_domain = d.get("domain", "general")
                            domain_counts[d_domain] = domain_counts.get(d_domain, 0) + 1
                    except (ValueError, KeyError) as e:
                        logger.debug("Skipping debate with invalid datetime: %s", e)

            activity["debates_last_period"] = len(recent)
            activity["consensus_last_period"] = sum(1 for d in recent if d.get("consensus_reached"))
            activity["domains_active"] = list(domain_counts.keys())[:10]

            if domain_counts:
                activity["most_active_domain"] = max(domain_counts, key=lambda k: domain_counts[k])
    except (TypeError, ValueError, KeyError, AttributeError) as e:
        logger.warning("Recent activity error: %s: %s", type(e).__name__, e)

    return activity


@require_permission(PERM_ADMIN_METRICS_READ)
@rate_limit(requests_per_minute=20, limiter_name="admin_dashboard_metrics_batch")
def process_debates_single_pass(
    debates: list, domain: str | None, hours: int
) -> tuple[dict, dict, dict]:
    """Process all debate metrics in a single pass through the data.

    This optimization consolidates 3 separate loops into one, reducing
    iteration overhead for large debate lists.

    Args:
        debates: List of debate records.
        domain: Optional domain filter (currently unused).
        hours: Time window for recent activity.

    Returns:
        Tuple of (summary, activity, patterns) dicts.
    """
    start_time = time.perf_counter()
    logger.debug(
        "Starting single-pass processing: debates=%d, domain=%s, hours=%d",
        len(debates),
        domain,
        hours,
    )

    # Initialize summary metrics
    summary: dict[str, Any] = {
        "total_debates": 0,
        "consensus_reached": 0,
        "consensus_rate": 0.0,
        "avg_confidence": 0.0,
        "avg_rounds": 0.0,
        "total_tokens_used": 0,
    }

    # Initialize activity metrics
    activity: dict[str, Any] = {
        "debates_last_period": 0,
        "consensus_last_period": 0,
        "domains_active": [],
        "most_active_domain": None,
        "period_hours": hours,
    }

    # Initialize pattern metrics
    patterns: dict[str, dict[str, Any]] = {
        "disagreement_stats": {
            "with_disagreements": 0,
            "disagreement_types": {},
        },
        "early_stopping": {
            "early_stopped": 0,
            "full_duration": 0,
        },
    }

    if not debates:
        return summary, activity, patterns

    try:
        cutoff = datetime.now() - timedelta(hours=hours)

        # Accumulators for single-pass processing
        total = len(debates)
        consensus_count = 0
        confidences = []
        domain_counts: dict[str, int] = {}
        recent_count = 0
        recent_consensus = 0
        with_disagreement = 0
        disagreement_types: dict[str, int] = {}
        early_stopped = 0
        full_duration = 0

        for d in debates:
            # Summary metrics
            if d.get("consensus_reached"):
                consensus_count += 1

            conf = d.get("confidence")
            if conf:
                confidences.append(conf)

            # Activity metrics - check if recent
            created_at = d.get("created_at")
            if created_at:
                try:
                    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    if dt.replace(tzinfo=None) > cutoff:
                        recent_count += 1
                        if d.get("consensus_reached"):
                            recent_consensus += 1
                        d_domain = d.get("domain", "general")
                        domain_counts[d_domain] = domain_counts.get(d_domain, 0) + 1
                except (ValueError, KeyError) as e:
                    logger.debug("Skipping debate with invalid timestamp: %s", e)

            # Pattern metrics
            if d.get("disagreement_report"):
                with_disagreement += 1
                report = d.get("disagreement_report", {})
                for dt_type in report.get("types", []):
                    disagreement_types[dt_type] = disagreement_types.get(dt_type, 0) + 1

            if d.get("early_stopped"):
                early_stopped += 1
            else:
                full_duration += 1

        # Build summary
        summary["total_debates"] = total
        summary["consensus_reached"] = consensus_count
        if total > 0:
            summary["consensus_rate"] = round(consensus_count / total, 3)
        if confidences:
            summary["avg_confidence"] = round(sum(confidences) / len(confidences), 3)

        # Build activity
        activity["debates_last_period"] = recent_count
        activity["consensus_last_period"] = recent_consensus
        activity["domains_active"] = list(domain_counts.keys())[:10]
        if domain_counts:
            activity["most_active_domain"] = max(domain_counts, key=lambda k: domain_counts[k])

        # Build patterns
        patterns["disagreement_stats"]["with_disagreements"] = with_disagreement
        patterns["disagreement_stats"]["disagreement_types"] = disagreement_types
        patterns["early_stopping"]["early_stopped"] = early_stopped
        patterns["early_stopping"]["full_duration"] = full_duration

    except (TypeError, ValueError, KeyError, AttributeError) as e:
        logger.warning("Single-pass processing error: %s: %s", type(e).__name__, e)

    elapsed = time.perf_counter() - start_time
    logger.debug(
        "Completed single-pass processing: elapsed=%.3fs, total=%d, consensus=%d, recent=%d",
        elapsed,
        summary.get("total_debates", 0),
        summary.get("consensus_reached", 0),
        activity.get("debates_last_period", 0),
    )
    return summary, activity, patterns


@require_permission(PERM_ADMIN_METRICS_READ)
@rate_limit(requests_per_minute=30, limiter_name="admin_dashboard_metrics")
def get_debate_patterns(debates: list) -> dict[str, Any]:
    """Get debate pattern statistics.

    Args:
        debates: List of debate records.

    Returns:
        Patterns dict with disagreement_stats and early_stopping.
    """
    patterns: dict[str, Any] = {
        "disagreement_stats": {
            "with_disagreements": 0,
            "disagreement_types": {},
        },
        "early_stopping": {
            "early_stopped": 0,
            "full_duration": 0,
        },
    }

    try:
        if debates:
            with_disagreement = 0
            disagreement_types: dict[str, int] = {}
            early_stopped = 0
            full_duration = 0

            for d in debates:
                if d.get("disagreement_report"):
                    with_disagreement += 1
                    report = d.get("disagreement_report", {})
                    for dt_type in report.get("types", []):
                        disagreement_types[dt_type] = disagreement_types.get(dt_type, 0) + 1

                if d.get("early_stopped"):
                    early_stopped += 1
                else:
                    full_duration += 1

            # Update patterns with computed stats
            disagree_stats = patterns["disagreement_stats"]
            if isinstance(disagree_stats, dict):
                disagree_stats["with_disagreements"] = with_disagreement
                disagree_stats["disagreement_types"] = disagreement_types
            early_stats = patterns["early_stopping"]
            if isinstance(early_stats, dict):
                early_stats["early_stopped"] = early_stopped
                early_stats["full_duration"] = full_duration
    except (TypeError, ValueError, KeyError, AttributeError) as e:
        logger.warning("Debate patterns error: %s: %s", type(e).__name__, e)

    return patterns
