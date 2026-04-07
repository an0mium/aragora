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
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from aragora.rbac.checker import get_permission_checker
from aragora.rbac.decorators import PermissionDeniedError
from aragora.rbac.models import AuthorizationContext
from aragora.server.handlers.utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# RBAC permission for admin metrics access
PERM_ADMIN_METRICS_READ = "admin:metrics:read"
ACTIVE_DEBATE_STATUSES = frozenset(
    {"pending", "in_progress", "starting", "initializing", "running", "active"}
)
IN_PROGRESS_DEBATE_STATUSES = frozenset(
    {"in_progress", "starting", "initializing", "running", "active"}
)
LOW_CONFIDENCE_THRESHOLD = 0.5
HIGH_CONFIDENCE_THRESHOLD = 0.8
URGENT_CONFIDENCE_THRESHOLD = 0.3


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


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    numeric = _coerce_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _coerce_text(value: Any) -> str | None:
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    text = text.strip()
    return text or None


def _load_json_payload(raw_payload: Any) -> dict[str, Any] | None:
    if raw_payload is None:
        return None
    if isinstance(raw_payload, dict):
        return raw_payload
    if isinstance(raw_payload, str):
        try:
            decoded = json.loads(raw_payload)
            return decoded if isinstance(decoded, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _artifact_size_bytes(raw_payload: Any) -> int:
    if raw_payload is None:
        return 0
    if isinstance(raw_payload, str):
        return len(raw_payload.encode("utf-8"))
    try:
        return len(json.dumps(raw_payload).encode("utf-8"))
    except (TypeError, ValueError):
        return 0


def _get_nested(payload: dict[str, Any] | None, *path: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _get_mapping_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _find_first_numeric(payload: Any, keys: tuple[str, ...]) -> float | None:
    if isinstance(payload, dict):
        for key in keys:
            numeric = _coerce_float(payload.get(key))
            if numeric is not None:
                return numeric

        for value in payload.values():
            numeric = _find_first_numeric(value, keys)
            if numeric is not None:
                return numeric
    elif isinstance(payload, list):
        for item in payload:
            numeric = _find_first_numeric(item, keys)
            if numeric is not None:
                return numeric
    return None


def _normalize_string_list(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []

    normalized: list[str] = []
    for value in values:
        text = _coerce_text(value)
        if text is not None:
            normalized.append(text)
    return normalized


def _normalize_float_dict(values: Any) -> dict[str, float]:
    if not isinstance(values, dict):
        return {}

    normalized: dict[str, float] = {}
    for key, value in values.items():
        key_text = _coerce_text(key)
        numeric = _coerce_float(value)
        if key_text is not None and numeric is not None:
            normalized[key_text] = numeric
    return normalized


def _get_receipt_store() -> Any:
    from aragora.storage.receipt_store import get_receipt_store

    return get_receipt_store()


def _extract_receipt_per_agent_cost(cost_summary: Any) -> dict[str, float]:
    if not isinstance(cost_summary, dict):
        return {}

    per_agent = cost_summary.get("per_agent")
    if not isinstance(per_agent, dict):
        return {}

    costs: dict[str, float] = {}
    for agent_name, raw_value in per_agent.items():
        agent_key = _coerce_text(agent_name)
        if agent_key is None:
            continue
        if isinstance(raw_value, dict):
            numeric = _coerce_float(raw_value.get("total_cost_usd"))
            if numeric is None:
                numeric = _coerce_float(raw_value.get("cost"))
        else:
            numeric = _coerce_float(raw_value)
        if numeric is not None:
            costs[agent_key] = numeric
    return costs


def _extract_dashboard_proof(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None

    cost_summary = _get_nested(payload, "receipt", "cost_summary")

    proof: dict[str, Any] = {}

    for candidate in (
        _get_nested(payload, "receipt", "receipt_id"),
        payload.get("receipt_id"),
        _get_nested(payload, "result", "receipt_id"),
        _get_nested(payload, "metadata", "receipt_id"),
    ):
        receipt_id = _coerce_text(candidate)
        if receipt_id is not None:
            proof["receipt_id"] = receipt_id
            break

    for candidate in (
        payload.get("receipt_hash"),
        _get_nested(payload, "result", "receipt_hash"),
        _get_nested(payload, "receipt", "artifact_hash"),
        _get_nested(payload, "receipt", "checksum"),
        _get_nested(payload, "receipt", "hash"),
        _get_nested(payload, "receipt", "signature"),
    ):
        receipt_hash = _coerce_text(candidate)
        if receipt_hash is not None:
            proof["receipt_hash"] = receipt_hash
            break

    for candidate in (
        _get_nested(payload, "receipt", "timestamp"),
        _get_nested(payload, "receipt", "created_at"),
        _get_nested(payload, "result", "receipt_timestamp"),
    ):
        receipt_timestamp = _coerce_text(candidate)
        if receipt_timestamp is not None:
            proof["receipt_timestamp"] = receipt_timestamp
            break

    for candidate in (
        payload.get("provider"),
        _get_nested(payload, "metadata", "provider"),
    ):
        provider = _coerce_text(candidate)
        if provider is not None:
            proof["provider"] = provider
            break

    for candidate in (
        payload.get("provider_route"),
        _get_nested(payload, "metadata", "provider_route"),
    ):
        provider_route = _coerce_text(candidate)
        if provider_route is not None:
            proof["provider_route"] = provider_route
            break

    provider_names = _normalize_string_list(
        payload.get("provider_names")
        or _get_nested(payload, "result", "provider_names")
        or _get_nested(payload, "metadata", "provider_names")
    )
    if provider_names:
        proof["provider_names"] = provider_names

    provider_hints = _normalize_string_list(
        payload.get("provider_hints")
        or _get_nested(payload, "result", "provider_hints")
        or _get_nested(payload, "metadata", "provider_hints")
    )
    if provider_hints:
        proof["provider_hints"] = provider_hints

    provider_routing = (
        payload.get("provider_routing")
        or _get_nested(payload, "result", "provider_routing")
        or _get_nested(payload, "metadata", "provider_routing")
    )
    if isinstance(provider_routing, dict) and provider_routing:
        proof["provider_routing"] = provider_routing

    for candidate in (
        payload.get("total_cost_usd"),
        _get_nested(payload, "result", "total_cost_usd"),
        _get_nested(payload, "cost", "total_cost_usd"),
        _get_nested(payload, "receipt", "cost_summary", "total_cost_usd"),
        payload.get("cost_usd"),
        _get_nested(payload, "result", "cost_usd"),
    ):
        total_cost_usd = _coerce_float(candidate)
        if total_cost_usd is not None:
            proof["total_cost_usd"] = total_cost_usd
            break

    per_agent_cost = _normalize_float_dict(
        payload.get("per_agent_cost")
        or _get_nested(payload, "result", "per_agent_cost")
        or _get_nested(payload, "cost", "per_agent_cost")
    )
    if not per_agent_cost:
        per_agent_cost = _extract_receipt_per_agent_cost(cost_summary)
    if per_agent_cost:
        proof["per_agent_cost"] = per_agent_cost

    return proof or None


def _receipt_lookup_candidates(debate_id: str) -> list[str]:
    candidates = [debate_id]
    if debate_id and not debate_id.startswith("debate-"):
        candidates.append(f"debate-{debate_id}")
    return candidates


def build_debate_proof(record: dict[str, Any]) -> dict[str, Any] | None:
    proof = dict(record.get("proof") or {}) if isinstance(record.get("proof"), dict) else {}
    debate_id = _coerce_text(record.get("id"))
    if debate_id is None:
        return proof or None

    needs_receipt_backfill = any(
        field not in proof for field in ("receipt_id", "receipt_hash", "total_cost_usd")
    )
    if not needs_receipt_backfill:
        return proof or None

    try:
        store = _get_receipt_store()
    except (ImportError, RuntimeError, OSError):
        return proof or None

    receipt = None
    for candidate in _receipt_lookup_candidates(debate_id):
        try:
            receipt = store.get_by_gauntlet(candidate)
        except AttributeError:
            return proof or None
        if receipt is not None:
            break

    if receipt is None:
        return proof or None

    receipt_id = _coerce_text(_get_mapping_value(receipt, "receipt_id"))
    if receipt_id is not None and "receipt_id" not in proof:
        proof["receipt_id"] = receipt_id

    receipt_hash = _coerce_text(
        _get_mapping_value(receipt, "checksum") or _get_mapping_value(receipt, "artifact_hash")
    )
    if receipt_hash is not None and "receipt_hash" not in proof:
        proof["receipt_hash"] = receipt_hash

    receipt_timestamp = _get_mapping_value(receipt, "created_at")
    if receipt_timestamp is not None and "receipt_timestamp" not in proof:
        if hasattr(receipt_timestamp, "isoformat"):
            proof["receipt_timestamp"] = receipt_timestamp.isoformat()
        else:
            text = _coerce_text(receipt_timestamp)
            if text is not None:
                proof["receipt_timestamp"] = text

    cost_summary = _get_mapping_value(receipt, "cost_summary")
    if "total_cost_usd" not in proof:
        total_cost_usd = _find_first_numeric(cost_summary, ("total_cost_usd", "cost_usd"))
        if total_cost_usd is not None:
            proof["total_cost_usd"] = total_cost_usd

    if "per_agent_cost" not in proof:
        per_agent_cost = _extract_receipt_per_agent_cost(cost_summary)
        if per_agent_cost:
            proof["per_agent_cost"] = per_agent_cost

    return proof or None


def _extract_total_tokens(payload: dict[str, Any] | None) -> int:
    if not payload:
        return 0

    for key_group in (
        ("total_tokens",),
        ("tokens_used",),
        ("total_tokens_in", "total_tokens_out"),
        ("tokens_in", "tokens_out"),
    ):
        values = [
            value
            for key in key_group
            if (value := _find_first_numeric(payload, (key,))) is not None
        ]
        if values:
            return int(sum(values))
    return 0


def _extract_domain(payload: dict[str, Any] | None, fallback: Any) -> str | None:
    for candidate in (
        fallback,
        payload.get("domain") if payload else None,
        payload.get("classification", {}).get("domain") if payload else None,
        payload.get("metadata", {}).get("domain") if payload else None,
    ):
        if candidate not in (None, ""):
            return str(candidate)
    return None


def _extract_status(
    raw_status: Any,
    payload: dict[str, Any] | None,
    consensus_reached: bool,
    completed_at_dt: datetime | None,
) -> str:
    for candidate in (
        raw_status,
        payload.get("status") if payload else None,
        payload.get("state") if payload else None,
    ):
        if candidate not in (None, ""):
            return str(candidate).strip().lower()
    if completed_at_dt is not None or consensus_reached:
        return "completed"
    return "pending"


def _extract_rounds_used(payload: dict[str, Any] | None, raw_rounds: Any) -> int | None:
    direct_rounds = _coerce_int(raw_rounds)
    if direct_rounds is not None:
        return direct_rounds
    if not payload:
        return None
    for key in ("rounds_used", "rounds_completed", "rounds"):
        rounds = _find_first_numeric(payload, (key,))
        if rounds is not None:
            return int(rounds)
    return None


def _extract_duration_seconds(
    payload: dict[str, Any] | None,
    created_at_dt: datetime | None,
    completed_at_dt: datetime | None,
) -> float | None:
    if payload:
        duration_seconds = _find_first_numeric(payload, ("duration_seconds",))
        if duration_seconds is not None:
            return duration_seconds

        duration_ms = _find_first_numeric(payload, ("duration_ms", "execution_time_ms"))
        if duration_ms is not None:
            return duration_ms / 1000.0

        duration = _find_first_numeric(payload, ("duration",))
        if duration is not None:
            return duration

    if created_at_dt is not None and completed_at_dt is not None:
        return max((completed_at_dt - created_at_dt).total_seconds(), 0.0)
    return None


def load_debate_records(storage: Any) -> list[dict[str, Any]]:
    """Load dashboard-ready debate records across sqlite/postgres variants."""
    records: list[dict[str, Any]] = []
    min_dt = datetime.min.replace(tzinfo=timezone.utc)

    try:
        with storage.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM debates")
            columns = [str(column[0]) for column in (cursor.description or [])]

            for row in cursor.fetchall():
                row_map = dict(zip(columns, row, strict=False))
                raw_payload = row_map.get("artifact_json")
                if raw_payload is None:
                    raw_payload = row_map.get("result")
                payload = _load_json_payload(raw_payload)

                debate_id = (
                    row_map.get("id")
                    or row_map.get("slug")
                    or (payload.get("id") if payload else None)
                    or ""
                )

                consensus_value = row_map.get("consensus_reached")
                if consensus_value is None and payload:
                    consensus_value = payload.get("consensus_reached")
                consensus_reached = bool(consensus_value)

                confidence_value = row_map.get("confidence")
                if confidence_value is None and payload:
                    confidence_value = payload.get("confidence")
                confidence = _coerce_float(confidence_value)

                created_at_value = row_map.get("created_at")
                if created_at_value is None and payload:
                    created_at_value = payload.get("created_at")
                created_at_dt = _parse_datetime(created_at_value)

                completed_at_value = row_map.get("completed_at")
                if completed_at_value is None and payload:
                    completed_at_value = payload.get("completed_at")
                completed_at_dt = _parse_datetime(completed_at_value)

                status = _extract_status(
                    row_map.get("status"),
                    payload,
                    consensus_reached,
                    completed_at_dt,
                )
                domain_name = _extract_domain(payload, row_map.get("domain"))
                rounds_used = _extract_rounds_used(payload, row_map.get("rounds_used"))
                duration_seconds = _extract_duration_seconds(
                    payload,
                    created_at_dt,
                    completed_at_dt,
                )
                total_tokens = _extract_total_tokens(payload)
                review_state = (
                    str(payload.get("dashboard_review_state", "")).strip().lower()
                    if payload
                    else ""
                )

                if isinstance(created_at_value, datetime):
                    created_at_text = created_at_value.isoformat()
                elif created_at_value is not None:
                    created_at_text = str(created_at_value)
                elif created_at_dt is not None:
                    created_at_text = created_at_dt.isoformat()
                else:
                    created_at_text = None

                if isinstance(completed_at_value, datetime):
                    completed_at_text = completed_at_value.isoformat()
                elif completed_at_value is not None:
                    completed_at_text = str(completed_at_value)
                elif completed_at_dt is not None:
                    completed_at_text = completed_at_dt.isoformat()
                else:
                    completed_at_text = None

                records.append(
                    {
                        "id": str(debate_id),
                        "domain": domain_name,
                        "domain_label": domain_name or "general",
                        "status": status,
                        "consensus_reached": consensus_reached,
                        "confidence": confidence,
                        "created_at": created_at_text,
                        "completed_at": completed_at_text,
                        "rounds_used": rounds_used,
                        "duration_seconds": duration_seconds,
                        "total_tokens": total_tokens,
                        "artifact_bytes": _artifact_size_bytes(raw_payload),
                        "task": str(
                            row_map.get("task") or (payload.get("task") if payload else "") or ""
                        ),
                        "proof": _extract_dashboard_proof(payload),
                        "review_state": review_state,
                        "needs_attention": review_state != "dismissed"
                        and (
                            not consensus_reached
                            or (confidence is not None and confidence < URGENT_CONFIDENCE_THRESHOLD)
                        ),
                        "_sort_created_at": created_at_dt or min_dt,
                    }
                )
    except (KeyError, ValueError, OSError, TypeError, AttributeError) as e:
        logger.warning("Dashboard debate load error: %s: %s", type(e).__name__, e)
        return []

    records.sort(key=lambda record: record.get("_sort_created_at", min_dt), reverse=True)
    return records


def find_debate_record(storage: Any, debate_id: str) -> dict[str, Any] | None:
    """Find a single debate record by identifier."""
    for record in load_debate_records(storage):
        if record.get("id") == debate_id:
            return record
    return None


def summarize_debate_records(debates: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute dashboard summary metrics from normalized debate records."""
    summary: dict[str, Any] = {
        "total_debates": 0,
        "consensus_reached": 0,
        "consensus_rate": 0.0,
        "avg_confidence": 0.0,
        "avg_rounds": 0.0,
        "avg_duration_ms": 0.0,
        "total_tokens_used": 0,
        "completed_debates": 0,
        "open_debates": 0,
        "pending_debates": 0,
        "in_progress_debates": 0,
        "low_confidence_debates": 0,
        "needs_attention_debates": 0,
        "urgent_debates": 0,
        "high_confidence_consensus_count": 0,
        "high_confidence_consensus_rate": 0.0,
    }

    if not debates:
        return summary

    total = len(debates)
    consensus_count = sum(1 for debate in debates if debate.get("consensus_reached"))
    confidence_values = [
        confidence
        for debate in debates
        if (confidence := _coerce_float(debate.get("confidence"))) is not None
    ]
    rounds_values = [
        rounds
        for debate in debates
        if (rounds := _coerce_int(debate.get("rounds_used"))) is not None
    ]
    duration_values_ms = [
        duration * 1000.0
        for debate in debates
        if (duration := _coerce_float(debate.get("duration_seconds"))) is not None and duration > 0
    ]

    summary["total_debates"] = total
    summary["consensus_reached"] = consensus_count
    summary["consensus_rate"] = round(consensus_count / total, 3)
    if confidence_values:
        summary["avg_confidence"] = round(sum(confidence_values) / len(confidence_values), 3)
    if rounds_values:
        summary["avg_rounds"] = round(sum(rounds_values) / len(rounds_values), 3)
    if duration_values_ms:
        summary["avg_duration_ms"] = round(sum(duration_values_ms) / len(duration_values_ms), 3)

    summary["total_tokens_used"] = sum(
        int(_coerce_float(debate.get("total_tokens")) or 0) for debate in debates
    )
    summary["completed_debates"] = sum(
        1 for debate in debates if str(debate.get("status") or "") == "completed"
    )
    summary["pending_debates"] = sum(
        1 for debate in debates if str(debate.get("status") or "") == "pending"
    )
    summary["in_progress_debates"] = sum(
        1 for debate in debates if str(debate.get("status") or "") in IN_PROGRESS_DEBATE_STATUSES
    )
    summary["open_debates"] = sum(
        1 for debate in debates if str(debate.get("status") or "") in ACTIVE_DEBATE_STATUSES
    )
    summary["low_confidence_debates"] = sum(
        1
        for debate in debates
        if (
            (confidence := _coerce_float(debate.get("confidence"))) is not None
            and confidence < LOW_CONFIDENCE_THRESHOLD
        )
    )
    summary["needs_attention_debates"] = sum(
        1 for debate in debates if bool(debate.get("needs_attention"))
    )
    summary["urgent_debates"] = summary["needs_attention_debates"]
    summary["high_confidence_consensus_count"] = sum(
        1
        for debate in debates
        if debate.get("consensus_reached")
        and (
            (confidence := _coerce_float(debate.get("confidence"))) is not None
            and confidence >= HIGH_CONFIDENCE_THRESHOLD
        )
    )
    summary["high_confidence_consensus_rate"] = round(
        summary["high_confidence_consensus_count"] / total,
        3,
    )
    return summary


def recent_activity_from_debate_records(
    debates: list[dict[str, Any]], hours: int
) -> dict[str, Any]:
    """Compute recent activity metrics from normalized debate records."""
    activity: dict[str, Any] = {
        "debates_last_period": 0,
        "consensus_last_period": 0,
        "domains_active": [],
        "most_active_domain": None,
        "period_hours": hours,
    }

    if not debates:
        return activity

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    recent_debates = [
        debate
        for debate in debates
        if isinstance(debate.get("_sort_created_at"), datetime)
        and debate["_sort_created_at"] >= cutoff
    ]
    domain_counts: dict[str, int] = {}
    for debate in recent_debates:
        domain_name = str(debate.get("domain") or "general")
        domain_counts[domain_name] = domain_counts.get(domain_name, 0) + 1

    activity["debates_last_period"] = len(recent_debates)
    activity["consensus_last_period"] = sum(
        1 for debate in recent_debates if debate.get("consensus_reached")
    )
    activity["domains_active"] = list(domain_counts.keys())[:10]
    if domain_counts:
        activity["most_active_domain"] = max(
            domain_counts,
            key=lambda domain_name: domain_counts.get(domain_name, 0),
        )
    return activity


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
    try:
        return summarize_debate_records(load_debate_records(storage))
    except (KeyError, ValueError, OSError, TypeError, AttributeError) as e:
        logger.warning("SQL summary metrics error: %s: %s", type(e).__name__, e)
        return summarize_debate_records([])


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
    try:
        return recent_activity_from_debate_records(load_debate_records(storage), hours)
    except (KeyError, ValueError, OSError, TypeError, AttributeError) as e:
        logger.warning("SQL recent activity error: %s: %s", type(e).__name__, e)
        return recent_activity_from_debate_records([], hours)


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
