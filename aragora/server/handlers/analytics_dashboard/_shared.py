"""Shared imports, constants, and utilities for analytics dashboard handlers."""

from __future__ import annotations

import logging
from typing import Any, TypeVar
from collections.abc import Coroutine

from aragora.server.errors import safe_error_message
from aragora.server.handlers.analytics.cache import cached_analytics, cached_analytics_org
from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    get_clamped_int_param,
    get_string_param,
    handle_errors,
    json_response,
    rate_limit,
    require_permission,
    require_user_auth,
)
from aragora.server.http_utils import run_async
from aragora.server.versioning.compat import strip_version_prefix


# RBAC Permission constants
PERM_ANALYTICS_READ = "analytics:dashboard:read"
PERM_ANALYTICS_WRITE = "analytics:dashboard:write"
PERM_ANALYTICS_EXPORT = "analytics:export"
PERM_ANALYTICS_ADMIN = "analytics:admin"
PERM_ANALYTICS_COST = "analytics:cost:read"
PERM_ANALYTICS_COMPLIANCE = "analytics:compliance:read"
PERM_ANALYTICS_TOKENS = "analytics:tokens:read"
PERM_ANALYTICS_FLIPS = "analytics:flips:read"
PERM_ANALYTICS_DELIBERATIONS = "analytics:deliberations:read"

# RBAC imports (optional - graceful degradation in dev, fail-closed in production)
from aragora.server.handlers.utils.rbac_guard import rbac_fail_closed

try:
    from aragora.rbac import (
        AuthorizationContext,
        check_permission,
        PermissionDeniedError,
    )
    from aragora.billing.auth import extract_user_from_request

    RBAC_AVAILABLE = True
except ImportError:
    # SECURITY: In production, rbac_fail_closed() returns True and handlers
    # using RBAC_AVAILABLE must deny access rather than skip checks.
    RBAC_AVAILABLE = False
    AuthorizationContext = None  # type: ignore[misc]
    check_permission = None  # type: ignore[misc]
    PermissionDeniedError = Exception  # type: ignore[misc, assignment]
    extract_user_from_request = None

# Metrics imports (optional)
try:
    from aragora.observability.metrics import record_rbac_check

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

    def record_rbac_check(*args, **kwargs):
        pass


logger = logging.getLogger(__name__)

T = TypeVar("T")


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run async coroutine in sync context."""
    return run_async(coro)


ANALYTICS_STUB_RESPONSES = {
    "/api/analytics/summary": {
        "summary": {
            "total_debates": 0,
            "total_messages": 0,
            "consensus_rate": 0,
            "avg_debate_duration_ms": 0,
            "active_users_24h": 0,
            "top_topics": [],
        }
    },
    "/api/analytics/trends/findings": {"trends": []},
    "/api/analytics/remediation": {
        "metrics": {
            "total_findings": 0,
            "remediated": 0,
            "pending": 0,
            "avg_remediation_time_hours": 0,
            "remediation_rate": 0,
        }
    },
    "/api/analytics/agents": {"agents": []},
    "/api/analytics/cost": {
        "analysis": {
            "total_cost_usd": 0,
            "cost_by_model": {},
            "cost_by_debate_type": {},
            "projected_monthly_cost": 0,
            "cost_trend": [],
        }
    },
    "/api/analytics/cost/breakdown": {
        "breakdown": {
            "total_spend_usd": 0,
            "agents": [],
            "budget_utilization_pct": 0,
        }
    },
    "/api/analytics/compliance": {
        "compliance": {
            "overall_score": 0,
            "categories": [],
            "last_audit": "",
        }
    },
    "/api/analytics/heatmap": {
        "heatmap": {
            "x_labels": [],
            "y_labels": [],
            "values": [],
            "max_value": 0,
        }
    },
    "/api/analytics/tokens": {
        "summary": {
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "total_tokens": 0,
            "avg_tokens_per_day": 0,
        },
        "by_agent": {},
        "by_model": {},
    },
    "/api/analytics/tokens/trends": {"trends": []},
    "/api/analytics/tokens/providers": {"providers": []},
    "/api/analytics/flips/summary": {"summary": {"total": 0, "consistent": 0, "inconsistent": 0}},
    "/api/analytics/flips/recent": {"flips": []},
    "/api/analytics/flips/consistency": {"consistency": []},
    "/api/analytics/flips/trends": {"trends": []},
    "/api/analytics/deliberations": {"summary": {"total": 0, "consensus_rate": 0}},
    "/api/analytics/deliberations/channels": {"channels": []},
    "/api/analytics/deliberations/consensus": {"consensus": []},
    "/api/analytics/deliberations/performance": {"performance": []},
}

__all__ = [
    "ANALYTICS_STUB_RESPONSES",
    "AuthorizationContext",
    "BaseHandler",
    "HandlerResult",
    "METRICS_AVAILABLE",
    "PERM_ANALYTICS_ADMIN",
    "PERM_ANALYTICS_COMPLIANCE",
    "PERM_ANALYTICS_COST",
    "PERM_ANALYTICS_DELIBERATIONS",
    "PERM_ANALYTICS_EXPORT",
    "PERM_ANALYTICS_FLIPS",
    "PERM_ANALYTICS_READ",
    "PERM_ANALYTICS_TOKENS",
    "PERM_ANALYTICS_WRITE",
    "PermissionDeniedError",
    "RBAC_AVAILABLE",
    "rbac_fail_closed",
    "_run_async",
    "cached_analytics",
    "cached_analytics_org",
    "check_permission",
    "error_response",
    "extract_user_from_request",
    "get_clamped_int_param",
    "get_string_param",
    "handle_errors",
    "json_response",
    "logger",
    "rate_limit",
    "record_rbac_check",
    "require_permission",
    "require_user_auth",
    "safe_error_message",
    "strip_version_prefix",
]
