"""Tests for MetricsDashboardMixin in metrics_dashboard.py.

Comprehensive coverage of all 3 admin metrics endpoints:
- GET /api/v1/admin/stats          (_get_stats)
- GET /api/v1/admin/system/metrics (_get_system_metrics)
- GET /api/v1/admin/revenue        (_get_revenue_stats)

Also covers:
- Admin auth gate (401) for every endpoint
- RBAC permission gate (403) for every endpoint
- User store absence / None handling
- Debate storage presence, absence, and error paths
- Circuit breaker stats (import success, import failure, runtime error)
- Cache stats (import success, import failure, runtime error)
- Rate limiter stats (import success, import failure, limiter absent, no get_stats)
- Revenue calculation with patched TIER_LIMITS (string keys for JSON compat)
- Revenue calculation with string keys (no match in real TIER_LIMITS)
- Tier breakdown, MRR, ARR, paying/total organizations
- Empty tier distribution
- Free-only organizations (0 paying)
- Mixed tiers with correct MRR aggregation
- handle_errors decorator coverage
- Module exports
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.admin.metrics_dashboard import MetricsDashboardMixin
from aragora.server.handlers.utils.responses import HandlerResult, error_response


# ===========================================================================
# Helpers
# ===========================================================================


def _body(result: HandlerResult) -> dict:
    """Parse JSON body from a HandlerResult."""
    if result and result.body:
        return json.loads(result.body.decode("utf-8"))
    return {}


def _status(result: HandlerResult) -> int:
    """Extract status code from a HandlerResult."""
    return result.status_code


# ===========================================================================
# Mock TierLimits for revenue calculation tests
# ===========================================================================


@dataclass
class MockTierLimits:
    """Lightweight mock matching the price_monthly_cents attribute."""

    price_monthly_cents: int


# String-keyed TIER_LIMITS so json_response can serialize the dict keys
MOCK_TIER_LIMITS: dict[str, MockTierLimits] = {
    "free": MockTierLimits(price_monthly_cents=0),
    "starter": MockTierLimits(price_monthly_cents=9900),
    "professional": MockTierLimits(price_monthly_cents=29900),
    "enterprise": MockTierLimits(price_monthly_cents=99900),
}


# ===========================================================================
# Mock classes
# ===========================================================================


class MockAuthContext:
    """Minimal mock auth context for the mixin tests."""

    def __init__(self, user_id: str = "admin-001", org_id: str = "org-001"):
        self.user_id = user_id
        self.org_id = org_id


class MockHTTPHandler:
    """Mock HTTP handler with minimal attributes."""

    def __init__(self, path: str = "/api/v1/admin/stats", method: str = "GET"):
        self.path = path
        self.command = method
        self.headers = {"Content-Type": "application/json"}
        self.client_address = ("127.0.0.1", 12345)


class TestableHandler(MetricsDashboardMixin):
    """Concrete class wiring the mixin to controllable auth stubs."""

    def __init__(
        self,
        ctx: dict[str, Any] | None = None,
        admin_result: tuple[MockAuthContext | None, HandlerResult | None] | None = None,
        rbac_result: HandlerResult | None = None,
        user_store: Any | None = None,
    ):
        self.ctx = ctx or {}
        self._admin_result = admin_result or (MockAuthContext(), None)
        self._rbac_result = rbac_result
        self._user_store_val = user_store or MagicMock()

    def _require_admin(self, handler: Any) -> tuple[MockAuthContext | None, HandlerResult | None]:
        return self._admin_result

    def _check_rbac_permission(
        self, auth_ctx: Any, permission: str, resource_id: str | None = None
    ) -> HandlerResult | None:
        return self._rbac_result

    def _get_user_store(self) -> Any:
        return self._user_store_val


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def http():
    """Factory for MockHTTPHandler."""

    def _make(path: str = "/api/v1/admin/stats", method: str = "GET"):
        return MockHTTPHandler(path=path, method=method)

    return _make


@pytest.fixture
def user_store():
    """Default user store with admin stats."""
    store = MagicMock()
    store.get_admin_stats.return_value = {
        "total_users": 100,
        "active_users": 80,
        "total_organizations": 10,
        "tier_distribution": {},
    }
    return store


@pytest.fixture
def handler(user_store):
    """Handler with default user store and empty ctx."""
    return TestableHandler(user_store=user_store)


# ===========================================================================
# Module Exports
# ===========================================================================


class TestModuleExports:
    def test_all_exports(self):
        from aragora.server.handlers.admin.metrics_dashboard import __all__

        assert "MetricsDashboardMixin" in __all__

    def test_mixin_is_importable(self):
        assert MetricsDashboardMixin is not None

    def test_mixin_has_expected_methods(self):
        assert hasattr(MetricsDashboardMixin, "_get_stats")
        assert hasattr(MetricsDashboardMixin, "_get_system_metrics")
        assert hasattr(MetricsDashboardMixin, "_get_revenue_stats")


# ===========================================================================
# GET /api/v1/admin/stats
# ===========================================================================


class TestGetStats:
    """Tests for _get_stats endpoint."""

    def test_returns_401_when_admin_check_fails(self, http):
        h = TestableHandler(admin_result=(None, error_response("Unauthorized", 401)))
        result = h._get_stats(http())
        assert _status(result) == 401

    def test_returns_403_when_rbac_denied(self, http):
        h = TestableHandler(rbac_result=error_response("Forbidden", 403))
        result = h._get_stats(http())
        assert _status(result) == 403

    def test_success_returns_200(self, http, handler):
        result = handler._get_stats(http())
        assert _status(result) == 200

    def test_success_returns_stats_key(self, http, handler):
        result = handler._get_stats(http())
        data = _body(result)
        assert "stats" in data

    def test_returns_user_store_stats(self, http, user_store):
        user_store.get_admin_stats.return_value = {
            "total_users": 42,
            "active_users": 30,
        }
        h = TestableHandler(user_store=user_store)
        result = h._get_stats(http())
        data = _body(result)
        assert data["stats"]["total_users"] == 42
        assert data["stats"]["active_users"] == 30

    def test_calls_get_admin_stats(self, http, user_store):
        h = TestableHandler(user_store=user_store)
        h._get_stats(http())
        user_store.get_admin_stats.assert_called_once()

    def test_rbac_permission_name(self, http, user_store):
        """Verify the correct RBAC permission is checked."""
        permissions_checked = []

        class TrackingHandler(TestableHandler):
            def _check_rbac_permission(self, auth_ctx, permission, resource_id=None):
                permissions_checked.append(permission)
                return None

        h = TrackingHandler(user_store=user_store)
        h._get_stats(http())
        assert "admin.stats.read" in permissions_checked

    def test_stats_with_empty_store_result(self, http):
        store = MagicMock()
        store.get_admin_stats.return_value = {}
        h = TestableHandler(user_store=store)
        result = h._get_stats(http())
        assert _status(result) == 200
        data = _body(result)
        assert data["stats"] == {}

    def test_stats_with_rich_data(self, http):
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "total_users": 500,
            "active_users": 350,
            "total_organizations": 25,
            "tier_distribution": {"free": 10, "starter": 8, "professional": 5, "enterprise": 2},
        }
        h = TestableHandler(user_store=store)
        result = h._get_stats(http())
        data = _body(result)
        assert data["stats"]["total_users"] == 500
        assert data["stats"]["tier_distribution"]["free"] == 10

    def test_admin_error_returned_before_rbac(self, http):
        """When admin check fails, RBAC should not be checked."""
        rbac_called = []

        class TrackingHandler(TestableHandler):
            def _check_rbac_permission(self, auth_ctx, permission, resource_id=None):
                rbac_called.append(True)
                return None

        h = TrackingHandler(
            admin_result=(None, error_response("Unauthorized", 401)),
        )
        result = h._get_stats(http())
        assert _status(result) == 401
        assert len(rbac_called) == 0

    def test_rbac_error_returned_before_store_access(self, http):
        """When RBAC fails, user store should not be called."""
        store = MagicMock()
        h = TestableHandler(
            rbac_result=error_response("Forbidden", 403),
            user_store=store,
        )
        h._get_stats(http())
        store.get_admin_stats.assert_not_called()

    def test_stats_nested_objects(self, http):
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "nested": {"deeply": {"value": 123}},
        }
        h = TestableHandler(user_store=store)
        result = h._get_stats(http())
        data = _body(result)
        assert data["stats"]["nested"]["deeply"]["value"] == 123

    def test_stats_with_list_values(self, http):
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "recent_users": ["alice", "bob", "charlie"],
        }
        h = TestableHandler(user_store=store)
        result = h._get_stats(http())
        data = _body(result)
        assert data["stats"]["recent_users"] == ["alice", "bob", "charlie"]


# ===========================================================================
# GET /api/v1/admin/system/metrics
# ===========================================================================


class TestGetSystemMetrics:
    """Tests for _get_system_metrics endpoint."""

    def test_returns_401_when_admin_check_fails(self, http):
        h = TestableHandler(admin_result=(None, error_response("Unauthorized", 401)))
        result = h._get_system_metrics(http())
        assert _status(result) == 401

    def test_returns_403_when_rbac_denied(self, http):
        h = TestableHandler(rbac_result=error_response("Forbidden", 403))
        result = h._get_system_metrics(http())
        assert _status(result) == 403

    def test_success_returns_200(self, http, handler):
        result = handler._get_system_metrics(http())
        assert _status(result) == 200

    def test_response_contains_metrics_key(self, http, handler):
        result = handler._get_system_metrics(http())
        data = _body(result)
        assert "metrics" in data

    def test_metrics_has_timestamp(self, http, handler):
        result = handler._get_system_metrics(http())
        data = _body(result)
        assert "timestamp" in data["metrics"]

    def test_timestamp_is_iso_format(self, http, handler):
        result = handler._get_system_metrics(http())
        data = _body(result)
        ts = data["metrics"]["timestamp"]
        # Should be parseable ISO format
        assert "T" in ts

    def test_includes_users_when_user_store_present(self, http, user_store):
        user_store.get_admin_stats.return_value = {"total_users": 50}
        h = TestableHandler(user_store=user_store)
        result = h._get_system_metrics(http())
        data = _body(result)
        assert "users" in data["metrics"]
        assert data["metrics"]["users"]["total_users"] == 50

    def test_no_users_when_user_store_is_none(self, http):
        h = TestableHandler()
        # _get_user_store returns None
        h._get_user_store = lambda: None
        result = h._get_system_metrics(http())
        data = _body(result)
        assert "users" not in data["metrics"]

    def test_rbac_permission_name(self, http, user_store):
        """Verify the correct RBAC permission is checked."""
        permissions_checked = []

        class TrackingHandler(TestableHandler):
            def _check_rbac_permission(self, auth_ctx, permission, resource_id=None):
                permissions_checked.append(permission)
                return None

        h = TrackingHandler(user_store=user_store)
        h._get_system_metrics(http())
        assert "admin.metrics.read" in permissions_checked

    # --- Debate storage ---

    def test_includes_debates_when_storage_has_get_statistics(self, http, user_store):
        debate_storage = MagicMock()
        debate_storage.get_statistics.return_value = {"total": 200, "completed": 180}
        h = TestableHandler(
            ctx={"debate_storage": debate_storage},
            user_store=user_store,
        )
        result = h._get_system_metrics(http())
        data = _body(result)
        assert data["metrics"]["debates"]["total"] == 200

    def test_no_debates_when_no_debate_storage_in_ctx(self, http, handler):
        result = handler._get_system_metrics(http())
        data = _body(result)
        assert "debates" not in data["metrics"]

    def test_no_debates_when_storage_lacks_get_statistics(self, http, user_store):
        debate_storage = MagicMock(spec=[])  # no get_statistics
        h = TestableHandler(
            ctx={"debate_storage": debate_storage},
            user_store=user_store,
        )
        result = h._get_system_metrics(http())
        data = _body(result)
        assert "debates" not in data["metrics"]

    def test_debates_error_yields_unavailable(self, http, user_store):
        debate_storage = MagicMock()
        debate_storage.get_statistics.side_effect = OSError("disk error")
        h = TestableHandler(
            ctx={"debate_storage": debate_storage},
            user_store=user_store,
        )
        result = h._get_system_metrics(http())
        data = _body(result)
        assert data["metrics"]["debates"] == {"error": "unavailable"}

    def test_debates_key_error_yields_unavailable(self, http, user_store):
        debate_storage = MagicMock()
        debate_storage.get_statistics.side_effect = KeyError("missing")
        h = TestableHandler(
            ctx={"debate_storage": debate_storage},
            user_store=user_store,
        )
        result = h._get_system_metrics(http())
        data = _body(result)
        assert data["metrics"]["debates"]["error"] == "unavailable"

    def test_debates_value_error_yields_unavailable(self, http, user_store):
        debate_storage = MagicMock()
        debate_storage.get_statistics.side_effect = ValueError("bad data")
        h = TestableHandler(
            ctx={"debate_storage": debate_storage},
            user_store=user_store,
        )
        result = h._get_system_metrics(http())
        data = _body(result)
        assert data["metrics"]["debates"]["error"] == "unavailable"

    def test_debates_type_error_yields_unavailable(self, http, user_store):
        debate_storage = MagicMock()
        debate_storage.get_statistics.side_effect = TypeError("wrong type")
        h = TestableHandler(
            ctx={"debate_storage": debate_storage},
            user_store=user_store,
        )
        result = h._get_system_metrics(http())
        data = _body(result)
        assert data["metrics"]["debates"]["error"] == "unavailable"

    def test_debates_attribute_error_yields_unavailable(self, http, user_store):
        debate_storage = MagicMock()
        debate_storage.get_statistics.side_effect = AttributeError("no attr")
        h = TestableHandler(
            ctx={"debate_storage": debate_storage},
            user_store=user_store,
        )
        result = h._get_system_metrics(http())
        data = _body(result)
        assert data["metrics"]["debates"]["error"] == "unavailable"

    def test_debate_storage_none_value_in_ctx(self, http, user_store):
        """When debate_storage is explicitly None, debates should not appear."""
        h = TestableHandler(
            ctx={"debate_storage": None},
            user_store=user_store,
        )
        result = h._get_system_metrics(http())
        data = _body(result)
        assert "debates" not in data["metrics"]

    # --- Circuit breakers ---

    def test_includes_circuit_breakers(self, http, user_store):
        h = TestableHandler(user_store=user_store)
        with patch(
            "aragora.resilience.get_circuit_breaker_status",
            return_value={"breaker1": "closed", "breaker2": "open"},
        ):
            result = h._get_system_metrics(http())
        data = _body(result)
        assert "circuit_breakers" in data["metrics"]
        assert data["metrics"]["circuit_breakers"]["breaker1"] == "closed"

    def test_circuit_breakers_import_error_silenced(self, http, user_store):
        h = TestableHandler(user_store=user_store)
        with patch.dict("sys.modules", {"aragora.resilience": None}):
            result = h._get_system_metrics(http())
        # The key test is that no exception propagates
        assert _status(result) == 200

    def test_circuit_breakers_runtime_error_silenced(self, http, user_store):
        h = TestableHandler(user_store=user_store)
        with patch(
            "aragora.resilience.get_circuit_breaker_status",
            side_effect=RuntimeError("breaker down"),
        ):
            result = h._get_system_metrics(http())
        data = _body(result)
        assert _status(result) == 200
        assert "circuit_breakers" not in data["metrics"]

    def test_circuit_breakers_value_error_silenced(self, http, user_store):
        h = TestableHandler(user_store=user_store)
        with patch(
            "aragora.resilience.get_circuit_breaker_status",
            side_effect=ValueError("bad"),
        ):
            result = h._get_system_metrics(http())
        assert _status(result) == 200

    def test_circuit_breakers_type_error_silenced(self, http, user_store):
        h = TestableHandler(user_store=user_store)
        with patch(
            "aragora.resilience.get_circuit_breaker_status",
            side_effect=TypeError("type err"),
        ):
            result = h._get_system_metrics(http())
        assert _status(result) == 200

    def test_circuit_breakers_attribute_error_silenced(self, http, user_store):
        h = TestableHandler(user_store=user_store)
        with patch(
            "aragora.resilience.get_circuit_breaker_status",
            side_effect=AttributeError("attr err"),
        ):
            result = h._get_system_metrics(http())
        assert _status(result) == 200

    # --- Cache stats ---

    def test_includes_cache_stats(self, http, user_store):
        h = TestableHandler(user_store=user_store)
        with patch(
            "aragora.server.handlers.admin.cache.get_cache_stats",
            return_value={"hits": 100, "misses": 20},
        ):
            result = h._get_system_metrics(http())
        data = _body(result)
        assert "cache" in data["metrics"]
        assert data["metrics"]["cache"]["hits"] == 100

    def test_cache_stats_runtime_error_silenced(self, http, user_store):
        h = TestableHandler(user_store=user_store)
        with patch(
            "aragora.server.handlers.admin.cache.get_cache_stats",
            side_effect=RuntimeError("cache down"),
        ):
            result = h._get_system_metrics(http())
        assert _status(result) == 200

    def test_cache_stats_value_error_silenced(self, http, user_store):
        h = TestableHandler(user_store=user_store)
        with patch(
            "aragora.server.handlers.admin.cache.get_cache_stats",
            side_effect=ValueError("bad cache"),
        ):
            result = h._get_system_metrics(http())
        assert _status(result) == 200

    def test_cache_stats_type_error_silenced(self, http, user_store):
        h = TestableHandler(user_store=user_store)
        with patch(
            "aragora.server.handlers.admin.cache.get_cache_stats",
            side_effect=TypeError("type err"),
        ):
            result = h._get_system_metrics(http())
        assert _status(result) == 200

    # --- Rate limiter stats ---

    def test_includes_rate_limit_stats(self, http, user_store):
        mock_limiter = MagicMock()
        mock_limiter.get_stats.return_value = {"requests": 5000, "limited": 50}
        h = TestableHandler(user_store=user_store)
        with patch(
            "aragora.server.middleware.rate_limit.get_rate_limiter",
            return_value=mock_limiter,
        ):
            result = h._get_system_metrics(http())
        data = _body(result)
        assert "rate_limits" in data["metrics"]
        assert data["metrics"]["rate_limits"]["requests"] == 5000

    def test_rate_limiter_none_no_rate_limits(self, http, user_store):
        h = TestableHandler(user_store=user_store)
        with patch(
            "aragora.server.middleware.rate_limit.get_rate_limiter",
            return_value=None,
        ):
            result = h._get_system_metrics(http())
        data = _body(result)
        assert "rate_limits" not in data["metrics"]

    def test_rate_limiter_no_get_stats(self, http, user_store):
        mock_limiter = MagicMock(spec=[])  # no get_stats attribute
        h = TestableHandler(user_store=user_store)
        with patch(
            "aragora.server.middleware.rate_limit.get_rate_limiter",
            return_value=mock_limiter,
        ):
            result = h._get_system_metrics(http())
        data = _body(result)
        assert "rate_limits" not in data["metrics"]

    def test_rate_limiter_import_error_silenced(self, http, user_store):
        h = TestableHandler(user_store=user_store)
        with patch(
            "aragora.server.middleware.rate_limit.get_rate_limiter",
            side_effect=ImportError("no module"),
        ):
            result = h._get_system_metrics(http())
        assert _status(result) == 200

    def test_rate_limiter_runtime_error_silenced(self, http, user_store):
        h = TestableHandler(user_store=user_store)
        with patch(
            "aragora.server.middleware.rate_limit.get_rate_limiter",
            side_effect=RuntimeError("limiter error"),
        ):
            result = h._get_system_metrics(http())
        assert _status(result) == 200

    def test_rate_limiter_value_error_silenced(self, http, user_store):
        h = TestableHandler(user_store=user_store)
        with patch(
            "aragora.server.middleware.rate_limit.get_rate_limiter",
            side_effect=ValueError("val error"),
        ):
            result = h._get_system_metrics(http())
        assert _status(result) == 200

    # --- Admin + RBAC ordering ---

    def test_admin_error_returned_before_rbac(self, http):
        rbac_called = []

        class TrackingHandler(TestableHandler):
            def _check_rbac_permission(self, auth_ctx, permission, resource_id=None):
                rbac_called.append(True)
                return None

        h = TrackingHandler(
            admin_result=(None, error_response("Unauthorized", 401)),
        )
        result = h._get_system_metrics(http())
        assert _status(result) == 401
        assert len(rbac_called) == 0

    def test_rbac_error_before_data_collection(self, http):
        store = MagicMock()
        h = TestableHandler(
            rbac_result=error_response("Forbidden", 403),
            user_store=store,
        )
        h._get_system_metrics(http())
        store.get_admin_stats.assert_not_called()

    # --- Combined sources ---

    def test_all_sources_present(self, http, user_store):
        user_store.get_admin_stats.return_value = {"total_users": 99}
        debate_storage = MagicMock()
        debate_storage.get_statistics.return_value = {"total": 50}
        mock_limiter = MagicMock()
        mock_limiter.get_stats.return_value = {"limited": 5}

        h = TestableHandler(
            ctx={"debate_storage": debate_storage},
            user_store=user_store,
        )
        with (
            patch(
                "aragora.resilience.get_circuit_breaker_status",
                return_value={"status": "healthy"},
            ),
            patch(
                "aragora.server.handlers.admin.cache.get_cache_stats",
                return_value={"hits": 10},
            ),
            patch(
                "aragora.server.middleware.rate_limit.get_rate_limiter",
                return_value=mock_limiter,
            ),
        ):
            result = h._get_system_metrics(http())

        data = _body(result)
        assert _status(result) == 200
        metrics = data["metrics"]
        assert "timestamp" in metrics
        assert metrics["users"]["total_users"] == 99
        assert metrics["debates"]["total"] == 50
        assert metrics["circuit_breakers"]["status"] == "healthy"
        assert metrics["cache"]["hits"] == 10
        assert metrics["rate_limits"]["limited"] == 5

    def test_empty_ctx_no_optional_sources(self, http, user_store):
        """With empty ctx, only timestamp + users should appear."""
        h = TestableHandler(ctx={}, user_store=user_store)
        result = h._get_system_metrics(http())
        data = _body(result)
        metrics = data["metrics"]
        assert "timestamp" in metrics
        assert "users" in metrics

    def test_debate_statistics_empty_dict_returned(self, http, user_store):
        debate_storage = MagicMock()
        debate_storage.get_statistics.return_value = {}
        h = TestableHandler(
            ctx={"debate_storage": debate_storage},
            user_store=user_store,
        )
        result = h._get_system_metrics(http())
        data = _body(result)
        assert data["metrics"]["debates"] == {}

    def test_debate_statistics_rich_data(self, http, user_store):
        debate_storage = MagicMock()
        debate_storage.get_statistics.return_value = {
            "total": 500,
            "completed": 450,
            "active": 30,
            "consensus_rate": 0.85,
        }
        h = TestableHandler(
            ctx={"debate_storage": debate_storage},
            user_store=user_store,
        )
        result = h._get_system_metrics(http())
        data = _body(result)
        assert data["metrics"]["debates"]["total"] == 500
        assert data["metrics"]["debates"]["consensus_rate"] == 0.85


# ===========================================================================
# GET /api/v1/admin/revenue
# ===========================================================================


class TestGetRevenueStats:
    """Tests for _get_revenue_stats endpoint.

    Revenue calculation tests patch TIER_LIMITS to use string keys
    (matching what get_admin_stats returns in practice) so json_response
    can serialize the tier_breakdown dict keys.
    """

    def test_returns_401_when_admin_check_fails(self, http):
        h = TestableHandler(admin_result=(None, error_response("Unauthorized", 401)))
        result = h._get_revenue_stats(http())
        assert _status(result) == 401

    def test_returns_403_when_rbac_denied(self, http):
        h = TestableHandler(rbac_result=error_response("Forbidden", 403))
        result = h._get_revenue_stats(http())
        assert _status(result) == 403

    def test_success_returns_200(self, http, handler):
        result = handler._get_revenue_stats(http())
        assert _status(result) == 200

    def test_response_contains_revenue_key(self, http, handler):
        result = handler._get_revenue_stats(http())
        data = _body(result)
        assert "revenue" in data

    def test_revenue_has_expected_fields(self, http, handler):
        result = handler._get_revenue_stats(http())
        data = _body(result)
        rev = data["revenue"]
        assert "mrr_cents" in rev
        assert "mrr_dollars" in rev
        assert "arr_dollars" in rev
        assert "tier_breakdown" in rev
        assert "total_organizations" in rev
        assert "paying_organizations" in rev

    def test_empty_tier_distribution(self, http):
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "tier_distribution": {},
            "total_organizations": 0,
        }
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        rev = data["revenue"]
        assert rev["mrr_cents"] == 0
        assert rev["mrr_dollars"] == 0.0
        assert rev["arr_dollars"] == 0.0
        assert rev["tier_breakdown"] == {}
        assert rev["paying_organizations"] == 0

    @patch("aragora.billing.models.TIER_LIMITS", MOCK_TIER_LIMITS)
    def test_revenue_starter_tier(self, http):
        """Starter tier MRR calculation."""
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "tier_distribution": {"starter": 5},
            "total_organizations": 5,
        }
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        rev = data["revenue"]
        assert rev["mrr_cents"] == 5 * 9900
        assert rev["mrr_dollars"] == 5 * 9900 / 100

    @patch("aragora.billing.models.TIER_LIMITS", MOCK_TIER_LIMITS)
    def test_revenue_professional_tier(self, http):
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "tier_distribution": {"professional": 3},
            "total_organizations": 3,
        }
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        assert data["revenue"]["mrr_cents"] == 3 * 29900

    @patch("aragora.billing.models.TIER_LIMITS", MOCK_TIER_LIMITS)
    def test_revenue_enterprise_tier(self, http):
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "tier_distribution": {"enterprise": 2},
            "total_organizations": 2,
        }
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        assert data["revenue"]["mrr_cents"] == 2 * 99900

    @patch("aragora.billing.models.TIER_LIMITS", MOCK_TIER_LIMITS)
    def test_revenue_enterprise_tier_custom_pricing(self, http):
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "tier_distribution": {"enterprise": 1},
            "total_organizations": 1,
        }
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        rev = data["revenue"]
        assert rev["mrr_cents"] == 99900
        assert rev["mrr_dollars"] == 999.0
        assert rev["arr_dollars"] == 11988.0

    @patch("aragora.billing.models.TIER_LIMITS", MOCK_TIER_LIMITS)
    def test_revenue_free_tier_zero_contribution(self, http):
        """Free tier has 0 price, so MRR contribution is 0."""
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "tier_distribution": {"free": 100},
            "total_organizations": 100,
        }
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        rev = data["revenue"]
        assert rev["mrr_cents"] == 0
        assert rev["mrr_dollars"] == 0.0

    @patch("aragora.billing.models.TIER_LIMITS", MOCK_TIER_LIMITS)
    def test_revenue_tier_breakdown_structure(self, http):
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "tier_distribution": {"enterprise": 2},
            "total_organizations": 2,
        }
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        breakdown = data["revenue"]["tier_breakdown"]
        assert len(breakdown) == 1
        assert "enterprise" in breakdown
        tier_entry = breakdown["enterprise"]
        assert tier_entry["count"] == 2
        assert tier_entry["price_cents"] == 99900
        assert tier_entry["mrr_cents"] == 2 * 99900

    @patch("aragora.billing.models.TIER_LIMITS", MOCK_TIER_LIMITS)
    def test_revenue_multiple_tiers_combined(self, http):
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "tier_distribution": {
                "free": 50,
                "starter": 10,
                "professional": 5,
                "enterprise": 2,
                "enterprise_plus": 1,
            },
            "total_organizations": 68,
        }
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        rev = data["revenue"]
        expected_mrr = (
            0 * 50  # free
            + 9900 * 10  # starter
            + 29900 * 5  # professional
            + 99900 * 2  # enterprise
            + 500000 * 1  # enterprise_plus
        )
        assert rev["mrr_cents"] == expected_mrr
        assert rev["mrr_dollars"] == expected_mrr / 100
        assert rev["arr_dollars"] == expected_mrr * 12 / 100

    @patch("aragora.billing.models.TIER_LIMITS", MOCK_TIER_LIMITS)
    def test_arr_is_12x_mrr(self, http):
        """ARR should always be 12 times MRR."""
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "tier_distribution": {"starter": 1},
            "total_organizations": 1,
        }
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        rev = data["revenue"]
        assert rev["arr_dollars"] == rev["mrr_dollars"] * 12

    @patch("aragora.billing.models.TIER_LIMITS", MOCK_TIER_LIMITS)
    def test_revenue_single_org_per_tier(self, http):
        """One organization per tier to verify price lookup."""
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "tier_distribution": {
                "free": 1,
                "starter": 1,
                "professional": 1,
                "enterprise": 1,
                "enterprise_plus": 1,
            },
            "total_organizations": 5,
        }
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        expected_mrr = sum(tl.price_monthly_cents for tl in MOCK_TIER_LIMITS.values())
        assert data["revenue"]["mrr_cents"] == expected_mrr

    @patch("aragora.billing.models.TIER_LIMITS", MOCK_TIER_LIMITS)
    def test_revenue_zero_count_tiers(self, http):
        """Tiers with count=0 should produce 0 MRR contribution."""
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "tier_distribution": {"enterprise": 0},
            "total_organizations": 0,
        }
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        assert data["revenue"]["mrr_cents"] == 0

    @patch("aragora.billing.models.TIER_LIMITS", MOCK_TIER_LIMITS)
    def test_revenue_negative_count_still_computes(self, http):
        """Negative counts are not realistic but shouldn't crash."""
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "tier_distribution": {"starter": -1},
            "total_organizations": 0,
        }
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        assert data["revenue"]["mrr_cents"] == -9900

    def test_revenue_with_string_keys_no_match_in_real_tier_limits(self, http):
        """String keys won't match SubscriptionTier enum keys in real TIER_LIMITS."""
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "tier_distribution": {"starter": 10, "professional": 5},
            "total_organizations": 15,
        }
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        rev = data["revenue"]
        # String keys don't match SubscriptionTier enum keys -> no matches
        assert rev["mrr_cents"] == 0
        assert rev["tier_breakdown"] == {}

    def test_paying_organizations_excludes_free(self, http):
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "tier_distribution": {"free": 10, "starter": 5, "enterprise": 2},
            "total_organizations": 17,
        }
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        assert data["revenue"]["paying_organizations"] == 7  # 5 + 2

    def test_paying_organizations_all_free(self, http):
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "tier_distribution": {"free": 20},
            "total_organizations": 20,
        }
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        assert data["revenue"]["paying_organizations"] == 0

    def test_total_organizations_from_stats(self, http):
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "tier_distribution": {},
            "total_organizations": 42,
        }
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        assert data["revenue"]["total_organizations"] == 42

    def test_total_organizations_defaults_to_zero(self, http):
        store = MagicMock()
        store.get_admin_stats.return_value = {"tier_distribution": {}}
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        assert data["revenue"]["total_organizations"] == 0

    def test_tier_distribution_missing_defaults_to_empty(self, http):
        store = MagicMock()
        store.get_admin_stats.return_value = {}
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        assert data["revenue"]["tier_breakdown"] == {}
        assert data["revenue"]["mrr_cents"] == 0

    def test_rbac_permission_name(self, http, user_store):
        permissions_checked = []

        class TrackingHandler(TestableHandler):
            def _check_rbac_permission(self, auth_ctx, permission, resource_id=None):
                permissions_checked.append(permission)
                return None

        h = TrackingHandler(user_store=user_store)
        h._get_revenue_stats(http())
        assert "admin.revenue.read" in permissions_checked

    def test_admin_error_returned_before_rbac(self, http):
        rbac_called = []

        class TrackingHandler(TestableHandler):
            def _check_rbac_permission(self, auth_ctx, permission, resource_id=None):
                rbac_called.append(True)
                return None

        h = TrackingHandler(
            admin_result=(None, error_response("Unauthorized", 401)),
        )
        result = h._get_revenue_stats(http())
        assert _status(result) == 401
        assert len(rbac_called) == 0

    def test_rbac_error_before_store_access(self, http):
        store = MagicMock()
        h = TestableHandler(
            rbac_result=error_response("Forbidden", 403),
            user_store=store,
        )
        h._get_revenue_stats(http())
        store.get_admin_stats.assert_not_called()

    def test_unknown_tier_in_distribution_skipped(self, http):
        """Unrecognized tier names (not in TIER_LIMITS) should be skipped."""
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "tier_distribution": {"custom_tier": 5, "unknown": 3},
            "total_organizations": 8,
        }
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        assert data["revenue"]["mrr_cents"] == 0
        assert data["revenue"]["tier_breakdown"] == {}

    def test_paying_orgs_with_mixed_string_keys(self, http):
        """Paying organizations counts all non-'free' keys regardless of TIER_LIMITS match."""
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "tier_distribution": {
                "free": 10,
                "gold": 5,
                "silver": 3,
            },
            "total_organizations": 18,
        }
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        # gold + silver = 8
        assert data["revenue"]["paying_organizations"] == 8

    @patch("aragora.billing.models.TIER_LIMITS", MOCK_TIER_LIMITS)
    def test_revenue_only_free_with_patched_limits(self, http):
        """With only free orgs, MRR should be 0 even with patched TIER_LIMITS."""
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "tier_distribution": {"free": 50},
            "total_organizations": 50,
        }
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        assert data["revenue"]["mrr_cents"] == 0
        assert data["revenue"]["paying_organizations"] == 0

    @patch("aragora.billing.models.TIER_LIMITS", MOCK_TIER_LIMITS)
    def test_revenue_tier_breakdown_has_all_matched_tiers(self, http):
        """All tiers present in both distribution and TIER_LIMITS appear in breakdown."""
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "tier_distribution": {
                "starter": 3,
                "enterprise": 1,
            },
            "total_organizations": 4,
        }
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        breakdown = data["revenue"]["tier_breakdown"]
        assert "starter" in breakdown
        assert "enterprise" in breakdown
        assert len(breakdown) == 2

    @patch("aragora.billing.models.TIER_LIMITS", MOCK_TIER_LIMITS)
    def test_revenue_tier_breakdown_count_matches(self, http):
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "tier_distribution": {"professional": 7},
            "total_organizations": 7,
        }
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        assert data["revenue"]["tier_breakdown"]["professional"]["count"] == 7

    @patch("aragora.billing.models.TIER_LIMITS", MOCK_TIER_LIMITS)
    def test_revenue_tier_breakdown_price_cents_correct(self, http):
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "tier_distribution": {"professional": 1},
            "total_organizations": 1,
        }
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        assert data["revenue"]["tier_breakdown"]["professional"]["price_cents"] == 29900

    @patch("aragora.billing.models.TIER_LIMITS", MOCK_TIER_LIMITS)
    def test_revenue_mrr_dollars_precision(self, http):
        """MRR dollars should be mrr_cents / 100."""
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "tier_distribution": {"starter": 1},
            "total_organizations": 1,
        }
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        assert data["revenue"]["mrr_dollars"] == 9900 / 100  # 99.0


# ===========================================================================
# Error Handling / Edge Cases
# ===========================================================================


class TestErrorHandling:
    """Tests for error-handling paths across all endpoints."""

    def test_stats_user_store_exception_handled(self, http):
        """handle_errors decorator should catch and return 500."""
        store = MagicMock()
        store.get_admin_stats.side_effect = RuntimeError("store crashed")
        h = TestableHandler(user_store=store)
        result = h._get_stats(http())
        assert _status(result) == 500

    def test_system_metrics_user_store_exception_handled(self, http):
        store = MagicMock()
        store.get_admin_stats.side_effect = RuntimeError("store crashed")
        h = TestableHandler(user_store=store)
        result = h._get_system_metrics(http())
        assert _status(result) == 500

    def test_revenue_user_store_exception_handled(self, http):
        store = MagicMock()
        store.get_admin_stats.side_effect = RuntimeError("store crashed")
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        assert _status(result) == 500

    def test_stats_with_empty_body_error_result(self, http):
        """Admin returning a body-less error result."""
        err = HandlerResult(status_code=401, content_type="application/json", body=b"")
        h = TestableHandler(admin_result=(None, err))
        result = h._get_stats(http())
        assert _status(result) == 401

    def test_system_metrics_with_empty_body_error_result(self, http):
        err = HandlerResult(status_code=403, content_type="application/json", body=b"")
        h = TestableHandler(admin_result=(None, err))
        result = h._get_system_metrics(http())
        assert _status(result) == 403

    def test_revenue_with_empty_body_error_result(self, http):
        err = HandlerResult(status_code=403, content_type="application/json", body=b"")
        h = TestableHandler(admin_result=(None, err))
        result = h._get_revenue_stats(http())
        assert _status(result) == 403

    def test_stats_attribute_error_handled(self, http):
        store = MagicMock()
        store.get_admin_stats.side_effect = AttributeError("missing attr")
        h = TestableHandler(user_store=store)
        result = h._get_stats(http())
        assert _status(result) == 500

    def test_system_metrics_attribute_error_handled(self, http):
        store = MagicMock()
        store.get_admin_stats.side_effect = AttributeError("missing attr")
        h = TestableHandler(user_store=store)
        result = h._get_system_metrics(http())
        assert _status(result) == 500

    def test_revenue_attribute_error_handled(self, http):
        store = MagicMock()
        store.get_admin_stats.side_effect = AttributeError("missing attr")
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        assert _status(result) == 500


# ===========================================================================
# Content Type / Response Format
# ===========================================================================


class TestResponseFormat:
    """Tests for response content type and format."""

    def test_stats_content_type(self, http, handler):
        result = handler._get_stats(http())
        assert result.content_type == "application/json"

    def test_system_metrics_content_type(self, http, handler):
        result = handler._get_system_metrics(http())
        assert result.content_type == "application/json"

    def test_revenue_content_type(self, http, handler):
        result = handler._get_revenue_stats(http())
        assert result.content_type == "application/json"

    def test_stats_body_is_valid_json(self, http, handler):
        result = handler._get_stats(http())
        data = json.loads(result.body.decode("utf-8"))
        assert isinstance(data, dict)

    def test_system_metrics_body_is_valid_json(self, http, handler):
        result = handler._get_system_metrics(http())
        data = json.loads(result.body.decode("utf-8"))
        assert isinstance(data, dict)

    def test_revenue_body_is_valid_json(self, http, handler):
        result = handler._get_revenue_stats(http())
        data = json.loads(result.body.decode("utf-8"))
        assert isinstance(data, dict)

    def test_error_response_content_type(self, http):
        h = TestableHandler(admin_result=(None, error_response("Unauthorized", 401)))
        result = h._get_stats(http())
        assert result.content_type == "application/json"

    def test_500_response_has_body(self, http):
        store = MagicMock()
        store.get_admin_stats.side_effect = RuntimeError("boom")
        h = TestableHandler(user_store=store)
        result = h._get_stats(http())
        assert result.body is not None
        assert len(result.body) > 0


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_stats_large_stats_object(self, http):
        store = MagicMock()
        store.get_admin_stats.return_value = {f"key_{i}": i for i in range(100)}
        h = TestableHandler(user_store=store)
        result = h._get_stats(http())
        data = _body(result)
        assert len(data["stats"]) == 100

    def test_system_metrics_debate_storage_falsy_zero(self, http, user_store):
        """When debate_storage is 0 (falsy), debates should not appear."""
        h = TestableHandler(
            ctx={"debate_storage": 0},
            user_store=user_store,
        )
        result = h._get_system_metrics(http())
        data = _body(result)
        assert "debates" not in data["metrics"]

    def test_system_metrics_debate_storage_empty_string(self, http, user_store):
        h = TestableHandler(
            ctx={"debate_storage": ""},
            user_store=user_store,
        )
        result = h._get_system_metrics(http())
        data = _body(result)
        assert "debates" not in data["metrics"]

    def test_multiple_calls_to_same_endpoint(self, http, handler):
        """Calling the same endpoint multiple times should be safe."""
        for _ in range(5):
            result = handler._get_stats(http())
            assert _status(result) == 200

    def test_stats_handler_path_does_not_matter(self, http, handler):
        """The handler path is irrelevant for mixin methods called directly."""
        mock = MockHTTPHandler(path="/something/else")
        result = handler._get_stats(mock)
        assert _status(result) == 200

    def test_system_metrics_with_all_errors(self, http, user_store):
        """All optional sources fail but endpoint still returns 200."""
        debate_storage = MagicMock()
        debate_storage.get_statistics.side_effect = OSError("fail")

        h = TestableHandler(
            ctx={"debate_storage": debate_storage},
            user_store=user_store,
        )
        with (
            patch(
                "aragora.resilience.get_circuit_breaker_status",
                side_effect=RuntimeError("fail"),
            ),
            patch(
                "aragora.server.handlers.admin.cache.get_cache_stats",
                side_effect=RuntimeError("fail"),
            ),
            patch(
                "aragora.server.middleware.rate_limit.get_rate_limiter",
                side_effect=RuntimeError("fail"),
            ),
        ):
            result = h._get_system_metrics(http())

        assert _status(result) == 200
        data = _body(result)
        # Should still have timestamp and users
        assert "timestamp" in data["metrics"]
        assert "users" in data["metrics"]
        # Debates shows error
        assert data["metrics"]["debates"]["error"] == "unavailable"
        # Others silenced
        assert "circuit_breakers" not in data["metrics"]
        assert "rate_limits" not in data["metrics"]

    def test_system_metrics_ctx_without_debate_storage_key(self, http, user_store):
        """ctx that has other keys but no debate_storage."""
        h = TestableHandler(
            ctx={"some_other_key": "value"},
            user_store=user_store,
        )
        result = h._get_system_metrics(http())
        data = _body(result)
        assert "debates" not in data["metrics"]

    @patch("aragora.billing.models.TIER_LIMITS", MOCK_TIER_LIMITS)
    def test_revenue_large_org_count(self, http):
        """Large org counts compute correctly."""
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "tier_distribution": {"enterprise_plus": 1000},
            "total_organizations": 1000,
        }
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        assert data["revenue"]["mrr_cents"] == 1000 * 500000

    def test_revenue_free_key_literal_in_paying_orgs(self, http):
        """Only the exact string 'free' is excluded from paying_organizations."""
        store = MagicMock()
        store.get_admin_stats.return_value = {
            "tier_distribution": {"free": 10, "freemium": 5},
            "total_organizations": 15,
        }
        h = TestableHandler(user_store=store)
        result = h._get_revenue_stats(http())
        data = _body(result)
        # "freemium" != "free", so it counts as paying
        assert data["revenue"]["paying_organizations"] == 5

    def test_system_metrics_multiple_users_calls_not_duplicated(self, http, user_store):
        """user_store.get_admin_stats should be called exactly once."""
        h = TestableHandler(user_store=user_store)
        h._get_system_metrics(http())
        user_store.get_admin_stats.assert_called_once()
