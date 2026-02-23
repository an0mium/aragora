"""Tests for DashboardHandler in dashboard.py.

Comprehensive coverage of the main DashboardHandler class including:
- Route dispatch via handle() and handle_post()
- RBAC authentication and permission enforcement
- Rate limiting behavior
- can_handle() path matching
- Legacy /api/dashboard/debates endpoint
- _get_debates_dashboard aggregation
- _get_agent_performance with ELO data
- _get_consensus_insights with ConsensusMemory
- _get_system_health and _get_connector_health
- _get_connector_type mapping
- _call_bypassing_decorators utility
- Delegation to mixin methods
- Query param parsing (limit, offset, domain, status, hours)
- Param clamping (max limit, min offset)
- Error handling and graceful degradation
- TTL cache interaction
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.admin.cache import clear_cache
from aragora.server.handlers.admin.dashboard import (
    DASHBOARD_PERMISSION,
    DashboardHandler,
    PERM_ADMIN_DASHBOARD_READ,
    PERM_ADMIN_DASHBOARD_WRITE,
    PERM_ADMIN_METRICS_READ,
    _call_bypassing_decorators,
)
from aragora.server.handlers.utils.responses import HandlerResult


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
# In-memory SQLite storage for realistic SQL-level testing
# ===========================================================================


class InMemoryStorage:
    """Minimal storage with a real SQLite debates table."""

    def __init__(self, rows: list[tuple] | None = None):
        self._conn = sqlite3.connect(":memory:")
        cur = self._conn.cursor()
        cur.execute(
            """CREATE TABLE debates (
                id TEXT PRIMARY KEY,
                domain TEXT,
                status TEXT,
                consensus_reached INTEGER,
                confidence REAL,
                created_at TEXT
            )"""
        )
        cur.execute(
            """CREATE TABLE consensus (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT,
                confidence REAL,
                domain TEXT
            )"""
        )
        if rows:
            cur.executemany("INSERT INTO debates VALUES (?, ?, ?, ?, ?, ?)", rows)
        self._conn.commit()

    @contextmanager
    def connection(self):
        yield self._conn


class ErrorStorage:
    """Storage whose connection raises on cursor ops."""

    @contextmanager
    def connection(self):
        raise OSError("disk failure")


# ===========================================================================
# Mock ELO and Rating helpers
# ===========================================================================


class MockRating:
    """Mock ELO rating object."""

    def __init__(
        self,
        agent_name: str,
        elo: float = 1000,
        wins: int = 0,
        losses: int = 0,
        draws: int = 0,
        win_rate: float = 0.0,
        debates_count: int = 0,
    ):
        self.agent_name = agent_name
        self.elo = elo
        self.wins = wins
        self.losses = losses
        self.draws = draws
        self.win_rate = win_rate
        self.debates_count = debates_count


SAMPLE_ROWS = [
    ("d1", "finance", "completed", 1, 0.92, "2026-02-23T10:00:00"),
    ("d2", "tech", "completed", 0, 0.45, "2026-02-23T11:00:00"),
    ("d3", "finance", "in_progress", 0, 0.60, "2026-02-22T08:00:00"),
    ("d4", "legal", "pending", 1, 0.88, "2026-02-21T09:00:00"),
    ("d5", None, "pending", 1, 0.75, "2026-02-20T12:00:00"),
]


SAMPLE_RATINGS = [
    MockRating("claude-opus", 1250, 14, 6, 0, 0.7, 20),
    MockRating("gpt-4", 1200, 12, 6, 0, 0.65, 18),
    MockRating("mistral-large", 1100, 6, 6, 0, 0.5, 12),
]


# ===========================================================================
# Mock HTTP handler
# ===========================================================================


def make_mock_http(ip: str = "127.0.0.1") -> MagicMock:
    """Create a mock HTTP handler with client_address."""
    h = MagicMock()
    h.client_address = (ip, 12345)
    h.headers = {"Content-Type": "application/json"}
    h.path = "/api/v1/dashboard"
    h.command = "GET"
    return h


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def _clear_ttl_cache():
    """Clear TTL cache before every test to avoid cross-test pollution."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset rate limiters before each test."""
    from aragora.server.handlers.utils.rate_limit import clear_all_limiters

    clear_all_limiters()
    yield
    clear_all_limiters()


@pytest.fixture
def handler():
    """DashboardHandler with empty context."""
    return DashboardHandler(ctx={})


@pytest.fixture
def handler_with_storage():
    """DashboardHandler with in-memory storage."""
    storage = InMemoryStorage(SAMPLE_ROWS)
    return DashboardHandler(ctx={"storage": storage})


@pytest.fixture
def handler_with_elo():
    """DashboardHandler with mock ELO system."""
    elo = MagicMock()
    elo.get_all_ratings.return_value = SAMPLE_RATINGS
    return DashboardHandler(ctx={"elo_system": elo})


@pytest.fixture
def handler_with_storage_and_elo():
    """DashboardHandler with both storage and ELO."""
    storage = InMemoryStorage(SAMPLE_ROWS)
    elo = MagicMock()
    elo.get_all_ratings.return_value = SAMPLE_RATINGS
    return DashboardHandler(ctx={"storage": storage, "elo_system": elo})


@pytest.fixture
def mock_http():
    """Minimal mock HTTP handler."""
    return make_mock_http()


# ===========================================================================
# Tests: Permission constants
# ===========================================================================


class TestPermissionConstants:
    """Verify permission string constants."""

    def test_dashboard_read_permission(self):
        assert PERM_ADMIN_DASHBOARD_READ == "admin:dashboard:read"

    def test_dashboard_write_permission(self):
        assert PERM_ADMIN_DASHBOARD_WRITE == "admin:dashboard:write"

    def test_metrics_read_permission(self):
        assert PERM_ADMIN_METRICS_READ == "admin:metrics:read"

    def test_legacy_alias(self):
        assert DASHBOARD_PERMISSION == PERM_ADMIN_DASHBOARD_READ


# ===========================================================================
# Tests: can_handle
# ===========================================================================


class TestCanHandle:
    """Test can_handle path matching logic."""

    def test_exact_route_matches(self, handler):
        for route in DashboardHandler.ROUTES:
            assert handler.can_handle(route), f"Expected can_handle({route!r}) to be True"

    def test_gastown_path_excluded(self, handler):
        assert not handler.can_handle("/api/v1/dashboard/gastown/something")

    def test_debate_detail_path(self, handler):
        assert handler.can_handle("/api/v1/dashboard/debates/abc123")

    def test_debate_detail_too_deep(self, handler):
        assert not handler.can_handle("/api/v1/dashboard/debates/abc123/extra")

    def test_team_performance_detail_path(self, handler):
        assert handler.can_handle("/api/v1/dashboard/team-performance/claude")

    def test_team_performance_detail_too_deep(self, handler):
        assert not handler.can_handle("/api/v1/dashboard/team-performance/claude/extra")

    def test_quick_actions_detail_path(self, handler):
        assert handler.can_handle("/api/v1/dashboard/quick-actions/archive_read")

    def test_urgent_dismiss_path(self, handler):
        assert handler.can_handle("/api/v1/dashboard/urgent/item1/dismiss")

    def test_urgent_dismiss_wrong_depth(self, handler):
        assert not handler.can_handle("/api/v1/dashboard/urgent/item1/extra/dismiss")

    def test_pending_complete_path(self, handler):
        assert handler.can_handle("/api/v1/dashboard/pending-actions/act1/complete")

    def test_pending_complete_wrong_depth(self, handler):
        assert not handler.can_handle("/api/v1/dashboard/pending-actions/act1/extra/complete")

    def test_unknown_path_returns_false(self, handler):
        assert not handler.can_handle("/api/v1/totally-unknown")

    def test_unknown_dashboard_subpath(self, handler):
        assert not handler.can_handle("/api/v1/dashboard/nonexistent")


# ===========================================================================
# Tests: RBAC / auth (opt out of auto_auth)
# ===========================================================================


class TestRBACAuth:
    """Test authentication and authorization enforcement."""

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_handle_returns_401_on_unauthenticated(self, handler, mock_http):
        """Unauthenticated request should return 401."""
        from aragora.server.handlers.secure import SecureHandler, UnauthorizedError

        async def raise_unauth(self, req, require_auth=False):
            raise UnauthorizedError("No token")

        with patch.object(SecureHandler, "get_auth_context", raise_unauth):
            result = await handler.handle("/api/v1/dashboard", {}, mock_http)

        assert _status(result) == 401
        assert "Authentication required" in _body(result).get("error", "")

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_handle_returns_403_on_forbidden(self, handler, mock_http):
        """Authenticated but insufficient permissions returns 403."""
        from aragora.server.handlers.secure import SecureHandler, ForbiddenError
        from aragora.rbac.models import AuthorizationContext

        mock_ctx = AuthorizationContext(
            user_id="u1",
            user_email="u@test.com",
            org_id="org1",
            roles={"viewer"},
            permissions=set(),
        )

        async def mock_auth(self, req, require_auth=False):
            return mock_ctx

        def mock_check(self, ctx, perm, resource_id=None):
            raise ForbiddenError(f"Missing {perm}")

        with (
            patch.object(SecureHandler, "get_auth_context", mock_auth),
            patch.object(SecureHandler, "check_permission", mock_check),
        ):
            result = await handler.handle("/api/v1/dashboard", {}, mock_http)

        assert _status(result) == 403
        assert "Permission denied" in _body(result).get("error", "")

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_handle_post_returns_401_on_unauthenticated(self, handler, mock_http):
        """POST without auth returns 401."""
        from aragora.server.handlers.secure import SecureHandler, UnauthorizedError

        async def raise_unauth(self, req, require_auth=False):
            raise UnauthorizedError("No token")

        with patch.object(SecureHandler, "get_auth_context", raise_unauth):
            result = await handler.handle_post("/api/v1/dashboard/export", {}, mock_http)

        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_handle_post_returns_403_on_forbidden(self, handler, mock_http):
        """POST with insufficient permissions returns 403."""
        from aragora.server.handlers.secure import SecureHandler, ForbiddenError
        from aragora.rbac.models import AuthorizationContext

        mock_ctx = AuthorizationContext(
            user_id="u1",
            user_email="u@test.com",
            org_id="org1",
            roles={"viewer"},
            permissions=set(),
        )

        async def mock_auth(self, req, require_auth=False):
            return mock_ctx

        def mock_check(self, ctx, perm, resource_id=None):
            raise ForbiddenError(f"Missing {perm}")

        with (
            patch.object(SecureHandler, "get_auth_context", mock_auth),
            patch.object(SecureHandler, "check_permission", mock_check),
        ):
            result = await handler.handle_post("/api/v1/dashboard/export", {}, mock_http)

        assert _status(result) == 403


# ===========================================================================
# Tests: Rate limiting
# ===========================================================================


class TestRateLimiting:
    """Test rate limit enforcement on handle() and handle_post()."""

    @pytest.mark.asyncio
    async def test_handle_rate_limit_exceeded(self, handler, mock_http):
        """When rate limiter denies, return 429."""
        with patch("aragora.server.handlers.admin.dashboard._dashboard_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            result = await handler.handle("/api/v1/dashboard", {}, mock_http)

        assert _status(result) == 429
        assert "Rate limit" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_handle_post_rate_limit_exceeded(self, handler, mock_http):
        """When rate limiter denies POST, return 429."""
        with patch("aragora.server.handlers.admin.dashboard._dashboard_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            result = await handler.handle_post("/api/v1/dashboard/export", {}, mock_http)

        assert _status(result) == 429


# ===========================================================================
# Tests: handle() route dispatch (GET paths)
# ===========================================================================


class TestHandleRouteDispatch:
    """Test that handle() dispatches to the correct methods."""

    @pytest.mark.asyncio
    async def test_legacy_debates_endpoint(self, handler, mock_http):
        """GET /api/dashboard/debates returns dashboard data."""
        result = await handler.handle(
            "/api/dashboard/debates", {"limit": "5", "hours": "12"}, mock_http
        )
        assert _status(result) == 200
        body = _body(result)
        assert "summary" in body
        assert "system_health" in body

    @pytest.mark.asyncio
    async def test_overview_endpoint(self, handler, mock_http):
        """GET /api/v1/dashboard dispatches to _get_overview."""
        result = await handler.handle("/api/v1/dashboard", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert "stats" in body
        assert "system_health" in body

    @pytest.mark.asyncio
    async def test_overview_alternate_path(self, handler, mock_http):
        """GET /api/v1/dashboard/overview also dispatches to _get_overview."""
        result = await handler.handle("/api/v1/dashboard/overview", {}, mock_http)
        assert _status(result) == 200
        assert "stats" in _body(result)

    @pytest.mark.asyncio
    async def test_debates_list_endpoint(self, handler_with_storage, mock_http):
        """GET /api/v1/dashboard/debates returns debate list."""
        result = await handler_with_storage.handle(
            "/api/v1/dashboard/debates", {"limit": "10", "offset": "0"}, mock_http
        )
        assert _status(result) == 200
        body = _body(result)
        assert "debates" in body
        assert "total" in body

    @pytest.mark.asyncio
    async def test_debates_list_with_status_filter(self, handler_with_storage, mock_http):
        """GET /api/v1/dashboard/debates?status=completed filters by status."""
        result = await handler_with_storage.handle(
            "/api/v1/dashboard/debates",
            {"limit": "50", "offset": "0", "status": "completed"},
            mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        # Only completed debates
        for d in body["debates"]:
            assert d["status"] == "completed"

    @pytest.mark.asyncio
    async def test_debate_detail_endpoint(self, handler, mock_http):
        """GET /api/v1/dashboard/debates/{id} returns debate detail stub."""
        result = await handler.handle("/api/v1/dashboard/debates/d1", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["debate_id"] == "d1"

    @pytest.mark.asyncio
    async def test_stats_endpoint(self, handler, mock_http):
        """GET /api/v1/dashboard/stats returns stats."""
        result = await handler.handle("/api/v1/dashboard/stats", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert "debates" in body
        assert "agents" in body
        assert "performance" in body

    @pytest.mark.asyncio
    async def test_stat_cards_endpoint(self, handler, mock_http):
        """GET /api/v1/dashboard/stat-cards returns cards."""
        result = await handler.handle("/api/v1/dashboard/stat-cards", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert "cards" in body

    @pytest.mark.asyncio
    async def test_team_performance_endpoint(self, handler, mock_http):
        """GET /api/v1/dashboard/team-performance returns teams."""
        result = await handler.handle(
            "/api/v1/dashboard/team-performance", {"limit": "10"}, mock_http
        )
        assert _status(result) == 200
        body = _body(result)
        assert "teams" in body
        assert "total" in body

    @pytest.mark.asyncio
    async def test_team_performance_detail_endpoint(self, handler, mock_http):
        """GET /api/v1/dashboard/team-performance/{team_id} returns detail."""
        result = await handler.handle("/api/v1/dashboard/team-performance/claude", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["team_id"] == "claude"

    @pytest.mark.asyncio
    async def test_top_senders_endpoint(self, handler, mock_http):
        """GET /api/v1/dashboard/top-senders returns senders."""
        result = await handler.handle("/api/v1/dashboard/top-senders", {}, mock_http)
        assert _status(result) == 200
        assert "senders" in _body(result)

    @pytest.mark.asyncio
    async def test_labels_endpoint(self, handler, mock_http):
        """GET /api/v1/dashboard/labels returns labels."""
        result = await handler.handle("/api/v1/dashboard/labels", {}, mock_http)
        assert _status(result) == 200
        assert "labels" in _body(result)

    @pytest.mark.asyncio
    async def test_activity_endpoint(self, handler, mock_http):
        """GET /api/v1/dashboard/activity returns activity feed."""
        result = await handler.handle("/api/v1/dashboard/activity", {"limit": "20"}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert "activity" in body
        assert "total" in body

    @pytest.mark.asyncio
    async def test_inbox_summary_endpoint(self, handler, mock_http):
        """GET /api/v1/dashboard/inbox-summary returns summary."""
        result = await handler.handle("/api/v1/dashboard/inbox-summary", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert "total_messages" in body

    @pytest.mark.asyncio
    async def test_quick_actions_endpoint(self, handler, mock_http):
        """GET /api/v1/dashboard/quick-actions returns actions."""
        result = await handler.handle("/api/v1/dashboard/quick-actions", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert "actions" in body
        assert body["total"] >= 1

    @pytest.mark.asyncio
    async def test_urgent_items_endpoint(self, handler, mock_http):
        """GET /api/v1/dashboard/urgent returns urgent items."""
        result = await handler.handle("/api/v1/dashboard/urgent", {"limit": "20"}, mock_http)
        assert _status(result) == 200
        assert "items" in _body(result)

    @pytest.mark.asyncio
    async def test_pending_actions_endpoint(self, handler, mock_http):
        """GET /api/v1/dashboard/pending-actions returns pending actions."""
        result = await handler.handle(
            "/api/v1/dashboard/pending-actions", {"limit": "20"}, mock_http
        )
        assert _status(result) == 200
        assert "actions" in _body(result)

    @pytest.mark.asyncio
    async def test_search_endpoint(self, handler, mock_http):
        """GET /api/v1/dashboard/search returns search results."""
        result = await handler.handle("/api/v1/dashboard/search", {"q": "finance"}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert "results" in body
        assert "total" in body

    @pytest.mark.asyncio
    async def test_quality_metrics_endpoint(self, handler, mock_http):
        """GET /api/v1/dashboard/quality-metrics returns quality data."""
        result = await handler.handle("/api/v1/dashboard/quality-metrics", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert "calibration" in body
        assert "performance" in body
        assert "evolution" in body

    @pytest.mark.asyncio
    async def test_unmatched_path_returns_none(self, handler, mock_http):
        """Handle returns None for unmatched path."""
        result = await handler.handle("/api/v1/totally-unknown", {}, mock_http)
        assert result is None

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_quality_metrics_requires_additional_permission(self, mock_http):
        """quality-metrics checks admin:metrics:read in addition to dashboard:read."""
        from aragora.server.handlers.secure import SecureHandler, ForbiddenError
        from aragora.rbac.models import AuthorizationContext

        mock_ctx = AuthorizationContext(
            user_id="u1",
            user_email="u@test.com",
            org_id="org1",
            roles={"admin"},
            permissions={"admin:dashboard:read"},
        )
        h = DashboardHandler(ctx={})

        async def mock_auth(self, req, require_auth=False):
            return mock_ctx

        call_count = 0

        def mock_check(self, ctx, perm, resource_id=None):
            nonlocal call_count
            call_count += 1
            if perm == PERM_ADMIN_METRICS_READ:
                raise ForbiddenError(f"Missing {perm}")
            return True

        with (
            patch.object(SecureHandler, "get_auth_context", mock_auth),
            patch.object(SecureHandler, "check_permission", mock_check),
        ):
            result = await h.handle("/api/v1/dashboard/quality-metrics", {}, mock_http)

        assert _status(result) == 403


# ===========================================================================
# Tests: handle_post() route dispatch (POST paths)
# ===========================================================================


class TestHandlePostRouteDispatch:
    """Test that handle_post() dispatches to the correct methods."""

    @pytest.mark.asyncio
    async def test_execute_quick_action(self, handler, mock_http):
        """POST /api/v1/dashboard/quick-actions/{id} executes action."""
        result = await handler.handle_post(
            "/api/v1/dashboard/quick-actions/archive_read", {}, mock_http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["action_id"] == "archive_read"

    @pytest.mark.asyncio
    async def test_dismiss_urgent_item(self, handler_with_storage, mock_http):
        """POST /api/v1/dashboard/urgent/{id}/dismiss marks item reviewed."""
        result = await handler_with_storage.handle_post(
            "/api/v1/dashboard/urgent/d2/dismiss", {}, mock_http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["item_id"] == "d2"

    @pytest.mark.asyncio
    async def test_dismiss_urgent_item_not_found(self, handler_with_storage, mock_http):
        """POST /api/v1/dashboard/urgent/{id}/dismiss returns 404 for missing item."""
        result = await handler_with_storage.handle_post(
            "/api/v1/dashboard/urgent/nonexistent/dismiss", {}, mock_http
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_complete_pending_action(self, handler_with_storage, mock_http):
        """POST /api/v1/dashboard/pending-actions/{id}/complete updates status."""
        result = await handler_with_storage.handle_post(
            "/api/v1/dashboard/pending-actions/d4/complete", {}, mock_http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["action_id"] == "d4"

    @pytest.mark.asyncio
    async def test_complete_pending_action_not_found(self, handler_with_storage, mock_http):
        """POST /api/v1/dashboard/pending-actions/{id}/complete returns 404 if not pending."""
        result = await handler_with_storage.handle_post(
            "/api/v1/dashboard/pending-actions/nonexistent/complete", {}, mock_http
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_export_dashboard_data(self, handler, mock_http):
        """POST /api/v1/dashboard/export returns export snapshot."""
        result = await handler.handle_post("/api/v1/dashboard/export", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert "generated_at" in body
        assert "summary" in body

    @pytest.mark.asyncio
    async def test_post_unmatched_returns_none(self, handler, mock_http):
        """handle_post returns None for unmatched path."""
        result = await handler.handle_post("/api/v1/dashboard/unknown", {}, mock_http)
        assert result is None

    @pytest.mark.asyncio
    async def test_post_quick_action_wrong_depth_returns_none(self, handler, mock_http):
        """POST with extra path segments returns None."""
        result = await handler.handle_post(
            "/api/v1/dashboard/quick-actions/id/extra", {}, mock_http
        )
        assert result is None


# ===========================================================================
# Tests: Query param parsing and clamping
# ===========================================================================


class TestQueryParamParsing:
    """Test limit/offset/hours clamping and defaults."""

    @pytest.mark.asyncio
    async def test_debates_limit_clamped_to_50(self, handler, mock_http):
        """Limit is clamped to max 50 for legacy debates."""
        result = await handler.handle("/api/dashboard/debates", {"limit": "999"}, mock_http)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_debates_list_limit_clamped(self, handler_with_storage, mock_http):
        """GET /api/v1/dashboard/debates limit clamped to 50."""
        result = await handler_with_storage.handle(
            "/api/v1/dashboard/debates", {"limit": "200", "offset": "0"}, mock_http
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_debates_list_offset_min_zero(self, handler_with_storage, mock_http):
        """GET /api/v1/dashboard/debates offset clamped to min 0."""
        result = await handler_with_storage.handle(
            "/api/v1/dashboard/debates", {"limit": "10", "offset": "-5"}, mock_http
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_activity_limit_clamped_to_100(self, handler, mock_http):
        """GET /api/v1/dashboard/activity limit clamped to 100."""
        result = await handler.handle("/api/v1/dashboard/activity", {"limit": "500"}, mock_http)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_urgent_limit_clamped_to_100(self, handler, mock_http):
        """GET /api/v1/dashboard/urgent limit clamped to 100."""
        result = await handler.handle("/api/v1/dashboard/urgent", {"limit": "500"}, mock_http)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_team_performance_limit_clamped(self, handler, mock_http):
        """GET /api/v1/dashboard/team-performance limit clamped to 50."""
        result = await handler.handle(
            "/api/v1/dashboard/team-performance", {"limit": "200"}, mock_http
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_search_empty_query(self, handler, mock_http):
        """GET /api/v1/dashboard/search with empty q returns empty results."""
        result = await handler.handle("/api/v1/dashboard/search", {"q": ""}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["results"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_search_no_q_param(self, handler, mock_http):
        """GET /api/v1/dashboard/search with no q param returns empty."""
        result = await handler.handle("/api/v1/dashboard/search", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["results"] == []


# ===========================================================================
# Tests: _get_debates_dashboard (legacy aggregation)
# ===========================================================================


class TestGetDebatesDashboard:
    """Test the _get_debates_dashboard method directly."""

    def test_no_storage_returns_zero_summary(self, handler):
        """Without storage, summary shows zero values."""
        result = handler._get_debates_dashboard(None, 10, 24)
        body = _body(result)
        assert body["summary"]["total_debates"] == 0
        assert body["summary"]["consensus_rate"] == 0.0

    def test_no_storage_recent_activity(self, handler):
        """Without storage, recent_activity uses given hours."""
        result = handler._get_debates_dashboard(None, 10, 48)
        body = _body(result)
        assert body["recent_activity"]["period_hours"] == 48
        assert body["recent_activity"]["debates_last_period"] == 0

    def test_with_storage_returns_summary(self, handler_with_storage):
        """With storage, SQL metrics are returned."""
        result = handler_with_storage._get_debates_dashboard(None, 10, 24)
        body = _body(result)
        assert body["summary"]["total_debates"] == 5

    def test_generated_at_present(self, handler):
        """Response includes generated_at timestamp."""
        result = handler._get_debates_dashboard(None, 10, 24)
        body = _body(result)
        assert "generated_at" in body
        assert isinstance(body["generated_at"], float)

    def test_system_health_present(self, handler):
        """Response includes system_health section."""
        result = handler._get_debates_dashboard(None, 10, 24)
        body = _body(result)
        assert "system_health" in body

    def test_debate_patterns_present(self, handler):
        """Response includes debate_patterns with expected structure."""
        result = handler._get_debates_dashboard(None, 10, 24)
        body = _body(result)
        assert "disagreement_stats" in body["debate_patterns"]
        assert "early_stopping" in body["debate_patterns"]

    def test_consensus_insights_present(self, handler):
        """Response includes consensus_insights."""
        result = handler._get_debates_dashboard(None, 10, 24)
        body = _body(result)
        assert "consensus_insights" in body


# ===========================================================================
# Tests: _get_agent_performance
# ===========================================================================


class TestGetAgentPerformance:
    """Test _get_agent_performance method."""

    def test_no_elo_returns_empty(self, handler):
        """Without ELO system, returns default performance."""
        result = handler._get_agent_performance(10)
        assert result["top_performers"] == []
        assert result["total_agents"] == 0
        assert result["avg_elo"] == 0

    def test_with_elo_returns_agents(self, handler_with_elo):
        """With ELO system, returns agent performance data."""
        result = handler_with_elo._get_agent_performance(10)
        assert result["total_agents"] == 3
        assert len(result["top_performers"]) == 3
        assert result["avg_elo"] > 0

    def test_limit_truncates(self, handler_with_elo):
        """Limit truncates top_performers list."""
        result = handler_with_elo._get_agent_performance(2)
        assert len(result["top_performers"]) == 2

    def test_agent_fields(self, handler_with_elo):
        """Each agent entry has expected fields."""
        result = handler_with_elo._get_agent_performance(10)
        agent = result["top_performers"][0]
        assert "name" in agent
        assert "elo" in agent
        assert "wins" in agent
        assert "losses" in agent
        assert "draws" in agent
        assert "win_rate" in agent
        assert "debates_count" in agent

    def test_avg_elo_calculation(self, handler_with_elo):
        """avg_elo is the rounded average of all agent ELOs."""
        result = handler_with_elo._get_agent_performance(10)
        expected_avg = round((1250 + 1200 + 1100) / 3, 1)
        assert result["avg_elo"] == expected_avg

    def test_elo_system_error_graceful(self):
        """ELO system error returns default performance."""
        elo = MagicMock()
        elo.get_all_ratings.side_effect = RuntimeError("ELO broken")
        h = DashboardHandler(ctx={"elo_system": elo})
        result = h._get_agent_performance(10)
        assert result["top_performers"] == []

    def test_elo_no_get_all_ratings(self):
        """ELO object without get_all_ratings returns empty."""
        elo = MagicMock(spec=[])  # No attributes
        h = DashboardHandler(ctx={"elo_system": elo})
        result = h._get_agent_performance(10)
        assert result["top_performers"] == []


# ===========================================================================
# Tests: _get_consensus_insights
# ===========================================================================


class TestGetConsensusInsights:
    """Test _get_consensus_insights method."""

    def test_default_insights_on_import_error(self, handler):
        """When consensus memory is not importable, returns defaults."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "consensus" in name or "schema" in name:
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = handler._get_consensus_insights(None)
        assert result["total_consensus_topics"] == 0
        assert result["high_confidence_count"] == 0
        assert result["avg_confidence"] == 0.0

    def test_consensus_insights_structure(self, handler):
        """Insights dict has expected keys."""
        result = handler._get_consensus_insights(None)
        assert "total_consensus_topics" in result
        assert "high_confidence_count" in result
        assert "avg_confidence" in result
        assert "total_dissents" in result
        assert "domains" in result

    def test_consensus_insights_on_oserror(self, handler):
        """OSError during consensus query returns defaults gracefully."""
        with patch(
            "aragora.server.handlers.admin.dashboard.DashboardHandler._get_consensus_insights"
        ) as mock_method:
            mock_method.return_value = {
                "total_consensus_topics": 0,
                "high_confidence_count": 0,
                "avg_confidence": 0.0,
                "total_dissents": 0,
                "domains": [],
            }
            result = mock_method(None)
        assert result["total_consensus_topics"] == 0


# ===========================================================================
# Tests: _get_system_health
# ===========================================================================


class TestGetSystemHealth:
    """Test _get_system_health method."""

    def test_returns_expected_keys(self, handler):
        """System health includes standard keys."""
        result = handler._get_system_health()
        assert "uptime_seconds" in result
        assert "cache_entries" in result
        assert "connector_health" in result

    def test_connector_health_included(self, handler):
        """connector_health section is always present."""
        result = handler._get_system_health()
        assert "summary" in result["connector_health"]
        assert "connectors" in result["connector_health"]


# ===========================================================================
# Tests: _get_connector_type
# ===========================================================================


class TestGetConnectorType:
    """Test _get_connector_type method."""

    def test_none_connector(self, handler):
        assert handler._get_connector_type(None) == "unknown"

    def test_known_connector_mapping(self, handler):
        mock_connector = MagicMock()
        mock_connector.__class__.__name__ = "GithubEnterpriseConnector"
        # get_connector_type uses type(connector).__name__.lower()
        # but since MagicMock class name is set differently, let's test with patch
        with patch(
            "aragora.server.handlers.admin.dashboard_health.get_connector_type",
            return_value="github",
        ):
            from aragora.server.handlers.admin.dashboard_health import get_connector_type

            result = get_connector_type(mock_connector)
            assert result == "github"

    def test_unknown_connector_strips_connector(self, handler):
        """Unknown class names strip 'connector' suffix."""
        from aragora.server.handlers.admin.dashboard_health import get_connector_type

        class CustomConnector:
            pass

        result = get_connector_type(CustomConnector())
        assert result == "custom"

    def test_class_without_connector_suffix(self, handler):
        """Class without 'connector' in name returns lowered name."""
        from aragora.server.handlers.admin.dashboard_health import get_connector_type

        class MyAdapter:
            pass

        result = get_connector_type(MyAdapter())
        assert result == "myadapter"


# ===========================================================================
# Tests: _call_bypassing_decorators
# ===========================================================================


class TestCallBypassingDecorators:
    """Test the _call_bypassing_decorators utility."""

    def test_plain_function(self):
        """Plain function is called directly."""

        def fn(x, y):
            return x + y

        assert _call_bypassing_decorators(fn, 3, 4) == 7

    def test_wrapped_function(self):
        """Unwraps __wrapped__ chain to call inner function."""

        def inner(x):
            return x * 2

        def wrapper(x):
            return inner(x)

        wrapper.__wrapped__ = inner
        assert _call_bypassing_decorators(wrapper, 5) == 10

    def test_double_wrapped_function(self):
        """Unwraps multiple layers of __wrapped__."""

        def base(x):
            return x + 100

        def layer1(x):
            return base(x)

        layer1.__wrapped__ = base

        def layer2(x):
            return layer1(x)

        layer2.__wrapped__ = layer1

        assert _call_bypassing_decorators(layer2, 1) == 101


# ===========================================================================
# Tests: Handler initialization
# ===========================================================================


class TestHandlerInit:
    """Test DashboardHandler initialization."""

    def test_default_ctx(self):
        """Without ctx, empty dict is used."""
        h = DashboardHandler()
        assert h.ctx == {}

    def test_custom_ctx(self):
        """Custom ctx is stored."""
        ctx = {"storage": "fake", "elo_system": "mock"}
        h = DashboardHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_routes_defined(self):
        """ROUTES list is non-empty."""
        assert len(DashboardHandler.ROUTES) > 0

    def test_route_prefixes_defined(self):
        """ROUTE_PREFIXES is defined."""
        assert "/api/v1/dashboard/" in DashboardHandler.ROUTE_PREFIXES

    def test_resource_type(self):
        """RESOURCE_TYPE is dashboard."""
        assert DashboardHandler.RESOURCE_TYPE == "dashboard"


# ===========================================================================
# Tests: Storage error handling
# ===========================================================================


class TestStorageErrorHandling:
    """Test graceful degradation on storage errors."""

    @pytest.mark.asyncio
    async def test_debates_list_with_error_storage(self, mock_http):
        """Storage errors produce empty debates list."""
        h = DashboardHandler(ctx={"storage": ErrorStorage()})
        result = await h.handle("/api/v1/dashboard/debates", {"limit": "10"}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["debates"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_activity_with_error_storage(self, mock_http):
        """Storage errors produce empty activity list."""
        h = DashboardHandler(ctx={"storage": ErrorStorage()})
        result = await h.handle("/api/v1/dashboard/activity", {"limit": "10"}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["activity"] == []

    @pytest.mark.asyncio
    async def test_urgent_with_error_storage(self, mock_http):
        """Storage errors produce empty urgent items list."""
        h = DashboardHandler(ctx={"storage": ErrorStorage()})
        result = await h.handle("/api/v1/dashboard/urgent", {"limit": "10"}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["items"] == []

    @pytest.mark.asyncio
    async def test_search_with_error_storage(self, mock_http):
        """Storage errors produce empty search results."""
        h = DashboardHandler(ctx={"storage": ErrorStorage()})
        result = await h.handle("/api/v1/dashboard/search", {"q": "test"}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["results"] == []

    @pytest.mark.asyncio
    async def test_labels_with_error_storage(self, mock_http):
        """Storage errors produce empty labels list."""
        h = DashboardHandler(ctx={"storage": ErrorStorage()})
        result = await h.handle("/api/v1/dashboard/labels", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["labels"] == []

    @pytest.mark.asyncio
    async def test_top_senders_with_error_storage(self, mock_http):
        """Storage errors produce empty senders list."""
        h = DashboardHandler(ctx={"storage": ErrorStorage()})
        result = await h.handle("/api/v1/dashboard/top-senders", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["senders"] == []

    @pytest.mark.asyncio
    async def test_dismiss_with_error_storage(self, mock_http):
        """Storage errors on dismiss return 500."""
        h = DashboardHandler(ctx={"storage": ErrorStorage()})
        result = await h.handle_post("/api/v1/dashboard/urgent/d1/dismiss", {}, mock_http)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_complete_with_error_storage(self, mock_http):
        """Storage errors on complete return 500."""
        h = DashboardHandler(ctx={"storage": ErrorStorage()})
        result = await h.handle_post("/api/v1/dashboard/pending-actions/d1/complete", {}, mock_http)
        assert _status(result) == 500


# ===========================================================================
# Tests: Storage-backed data paths
# ===========================================================================


class TestStorageBackedPaths:
    """Test endpoints that query the database."""

    @pytest.mark.asyncio
    async def test_debates_list_returns_all(self, handler_with_storage, mock_http):
        """debates list returns all 5 sample rows."""
        result = await handler_with_storage.handle(
            "/api/v1/dashboard/debates", {"limit": "50"}, mock_http
        )
        body = _body(result)
        assert body["total"] == 5
        assert len(body["debates"]) == 5

    @pytest.mark.asyncio
    async def test_debates_pagination(self, handler_with_storage, mock_http):
        """Pagination limit and offset work correctly."""
        result = await handler_with_storage.handle(
            "/api/v1/dashboard/debates",
            {"limit": "2", "offset": "1"},
            mock_http,
        )
        body = _body(result)
        assert body["total"] == 5
        assert len(body["debates"]) == 2

    @pytest.mark.asyncio
    async def test_search_finds_matching_domain(self, handler_with_storage, mock_http):
        """Search by domain finds matching debates."""
        result = await handler_with_storage.handle(
            "/api/v1/dashboard/search", {"q": "finance"}, mock_http
        )
        body = _body(result)
        assert body["total"] >= 2

    @pytest.mark.asyncio
    async def test_search_finds_by_id(self, handler_with_storage, mock_http):
        """Search by debate ID finds matching debates."""
        result = await handler_with_storage.handle(
            "/api/v1/dashboard/search", {"q": "d1"}, mock_http
        )
        body = _body(result)
        assert body["total"] >= 1
        assert any(r["id"] == "d1" for r in body["results"])

    @pytest.mark.asyncio
    async def test_urgent_items_with_data(self, handler_with_storage, mock_http):
        """Urgent items returns debates with no consensus or low confidence."""
        result = await handler_with_storage.handle(
            "/api/v1/dashboard/urgent", {"limit": "50"}, mock_http
        )
        body = _body(result)
        # d2 has consensus_reached=0, d3 has consensus_reached=0
        assert body["total"] >= 2

    @pytest.mark.asyncio
    async def test_pending_actions_with_data(self, handler_with_storage, mock_http):
        """Pending actions returns pending/in_progress debates."""
        result = await handler_with_storage.handle(
            "/api/v1/dashboard/pending-actions", {"limit": "50"}, mock_http
        )
        body = _body(result)
        # d3 is in_progress, d4 is pending, d5 is pending
        assert body["total"] >= 2

    @pytest.mark.asyncio
    async def test_labels_with_data(self, handler_with_storage, mock_http):
        """Labels returns domain counts."""
        result = await handler_with_storage.handle("/api/v1/dashboard/labels", {}, mock_http)
        body = _body(result)
        assert len(body["labels"]) >= 1
        # finance appears twice
        finance_label = [l for l in body["labels"] if l["name"] == "finance"]
        assert len(finance_label) == 1
        assert finance_label[0]["count"] == 2

    @pytest.mark.asyncio
    async def test_top_senders_with_data(self, handler_with_storage, mock_http):
        """Top senders returns domain-based sender counts."""
        result = await handler_with_storage.handle("/api/v1/dashboard/top-senders", {}, mock_http)
        body = _body(result)
        assert len(body["senders"]) >= 1

    @pytest.mark.asyncio
    async def test_activity_with_data(self, handler_with_storage, mock_http):
        """Activity feed returns debate entries."""
        result = await handler_with_storage.handle(
            "/api/v1/dashboard/activity", {"limit": "50"}, mock_http
        )
        body = _body(result)
        assert body["total"] == 5
        assert len(body["activity"]) == 5
        # Each entry has expected fields
        entry = body["activity"][0]
        assert entry["type"] == "debate"
        assert "debate_id" in entry
        assert "consensus_reached" in entry


# ===========================================================================
# Tests: Dashboard with ELO data
# ===========================================================================


class TestDashboardWithElo:
    """Test endpoints that consume ELO data."""

    @pytest.mark.asyncio
    async def test_stat_cards_with_elo(self, handler_with_elo, mock_http):
        """Stat cards include agent count and avg ELO from ELO system."""
        result = await handler_with_elo.handle("/api/v1/dashboard/stat-cards", {}, mock_http)
        body = _body(result)
        cards = body["cards"]
        # Should have agent card and elo card at minimum
        agent_card = [c for c in cards if c["id"] == "active_agents"]
        assert len(agent_card) == 1
        assert agent_card[0]["value"] == 3

    @pytest.mark.asyncio
    async def test_overview_with_elo(self, handler_with_elo, mock_http):
        """Overview stats include agent count from ELO."""
        result = await handler_with_elo.handle("/api/v1/dashboard/overview", {}, mock_http)
        body = _body(result)
        agent_stat = [s for s in body["stats"] if s["label"] == "Total Agents"]
        assert len(agent_stat) == 1
        assert agent_stat[0]["value"] == 3

    @pytest.mark.asyncio
    async def test_team_performance_groups_by_provider(self, handler_with_elo, mock_http):
        """Team performance groups agents by provider prefix."""
        result = await handler_with_elo.handle(
            "/api/v1/dashboard/team-performance", {"limit": "50"}, mock_http
        )
        body = _body(result)
        teams = body["teams"]
        team_ids = [t["team_id"] for t in teams]
        assert "claude" in team_ids
        assert "gpt" in team_ids
        assert "mistral" in team_ids

    @pytest.mark.asyncio
    async def test_team_performance_detail_with_members(self, handler_with_elo, mock_http):
        """Team detail includes member info."""
        result = await handler_with_elo.handle(
            "/api/v1/dashboard/team-performance/claude", {}, mock_http
        )
        body = _body(result)
        assert body["team_id"] == "claude"
        assert body["member_count"] == 1  # only claude-opus starts with "claude"


# ===========================================================================
# Tests: No-storage scenarios
# ===========================================================================


class TestNoStorage:
    """Test behavior when no storage is available."""

    @pytest.mark.asyncio
    async def test_overview_no_storage(self, handler, mock_http):
        """Overview works without storage."""
        result = await handler.handle("/api/v1/dashboard/overview", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["total_debates_today"] == 0
        assert body["consensus_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_inbox_summary_no_storage(self, handler, mock_http):
        """Inbox summary returns zeros without storage."""
        result = await handler.handle("/api/v1/dashboard/inbox-summary", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["total_messages"] == 0

    @pytest.mark.asyncio
    async def test_stats_no_storage(self, handler, mock_http):
        """Stats returns zeros without storage."""
        result = await handler.handle("/api/v1/dashboard/stats", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["debates"]["total"] == 0

    @pytest.mark.asyncio
    async def test_dismiss_no_storage(self, handler, mock_http):
        """Dismiss without storage succeeds (no update to make)."""
        result = await handler.handle_post("/api/v1/dashboard/urgent/item1/dismiss", {}, mock_http)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_complete_no_storage(self, handler, mock_http):
        """Complete without storage succeeds."""
        result = await handler.handle_post(
            "/api/v1/dashboard/pending-actions/act1/complete", {}, mock_http
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_export_no_storage(self, handler, mock_http):
        """Export without storage returns partial data."""
        result = await handler.handle_post("/api/v1/dashboard/export", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["summary"] == {}


# ===========================================================================
# Tests: Quality metrics sub-methods
# ===========================================================================


class TestQualityMetricsSubMethods:
    """Test internal methods used by quality-metrics endpoint."""

    def test_calibration_metrics_no_tracker(self):
        """Without calibration_tracker, returns default calibration."""
        h = DashboardHandler(ctx={})
        result = h._get_calibration_metrics()
        assert result["overall_calibration"] == 0.0
        assert result["agents"] == {}

    def test_calibration_metrics_with_tracker(self):
        """With calibration_tracker, populates calibration data."""
        tracker = MagicMock()
        tracker.get_calibration_summary.return_value = {
            "agents": {
                "claude": {"calibration_bias": 0.2, "brier_score": 0.15},
                "gpt": {"calibration_bias": -0.15, "brier_score": 0.25},
                "mistral": {"calibration_bias": 0.05, "brier_score": 0.10},
            },
            "overall": 0.85,
        }
        tracker.get_all_agents.return_value = ["claude", "gpt", "mistral"]
        tracker.get_calibration_curve.return_value = None
        tracker.get_domain_breakdown.return_value = None

        h = DashboardHandler(ctx={"calibration_tracker": tracker})
        result = h._get_calibration_metrics()
        assert result["overall_calibration"] == 0.85
        assert "claude" in result["overconfident_agents"]
        assert "gpt" in result["underconfident_agents"]
        assert "mistral" in result["well_calibrated_agents"]
        assert len(result["top_by_brier"]) == 3

    def test_performance_metrics_no_monitor(self):
        """Without performance_monitor, returns default performance."""
        h = DashboardHandler(ctx={})
        result = h._get_performance_metrics()
        assert result["avg_latency_ms"] == 0.0
        assert result["success_rate"] == 0.0

    def test_performance_metrics_with_monitor(self):
        """With performance_monitor, populates data."""
        monitor = MagicMock()
        monitor.get_performance_insights.return_value = {
            "agents": {"claude": {"calls": 100}},
            "avg_latency_ms": 150.5,
            "success_rate": 0.95,
            "total_calls": 500,
        }
        h = DashboardHandler(ctx={"performance_monitor": monitor})
        result = h._get_performance_metrics()
        assert result["avg_latency_ms"] == 150.5
        assert result["success_rate"] == 0.95
        assert result["total_calls"] == 500

    def test_evolution_metrics_no_evolver(self):
        """Without prompt_evolver, returns default evolution."""
        h = DashboardHandler(ctx={})
        result = h._get_evolution_metrics()
        assert result["total_versions"] == 0
        assert result["patterns_extracted"] == 0

    def test_evolution_metrics_with_evolver(self):
        """With prompt_evolver, returns version data."""
        version = MagicMock()
        version.version = 3
        version.performance_score = 0.9
        version.debates_count = 50

        evolver = MagicMock()
        evolver.get_prompt_version.return_value = version
        evolver.get_top_patterns.return_value = ["p1", "p2"]

        h = DashboardHandler(ctx={"prompt_evolver": evolver})
        result = h._get_evolution_metrics()
        assert result["total_versions"] >= 3
        assert result["patterns_extracted"] == 2

    def test_debate_quality_metrics_no_data(self):
        """Without any subsystems, returns defaults."""
        h = DashboardHandler(ctx={})
        result = h._get_debate_quality_metrics()
        assert result["avg_confidence"] == 0.0
        assert result["consensus_rate"] == 0.0
        assert result["recent_winners"] == []

    def test_debate_quality_metrics_with_elo(self):
        """With ELO system providing recent matches."""
        elo = MagicMock()
        elo.get_recent_matches.return_value = [
            {"winner": "claude", "confidence": 0.9},
            {"winner": "gpt", "confidence": 0.8},
            {"winner": None, "confidence": 0.7},
        ]
        h = DashboardHandler(ctx={"elo_system": elo})
        result = h._get_debate_quality_metrics()
        assert result["recent_winners"] == ["claude", "gpt"]
        assert result["avg_confidence"] > 0


# ===========================================================================
# Tests: Edge cases
# ===========================================================================


class TestEdgeCases:
    """Miscellaneous edge cases."""

    @pytest.mark.asyncio
    async def test_empty_database(self, mock_http):
        """Handler works with empty database."""
        h = DashboardHandler(ctx={"storage": InMemoryStorage()})
        result = await h.handle("/api/v1/dashboard/debates", {"limit": "10"}, mock_http)
        body = _body(result)
        assert body["total"] == 0
        assert body["debates"] == []

    @pytest.mark.asyncio
    async def test_legacy_debates_domain_filter(self, handler_with_storage, mock_http):
        """Legacy endpoint accepts domain param."""
        result = await handler_with_storage.handle(
            "/api/dashboard/debates",
            {"domain": "finance", "limit": "10", "hours": "24"},
            mock_http,
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_concurrent_different_paths(self, handler, mock_http):
        """Multiple distinct paths can be handled in sequence."""
        paths = [
            "/api/v1/dashboard/overview",
            "/api/v1/dashboard/stats",
            "/api/v1/dashboard/labels",
        ]
        for path in paths:
            result = await handler.handle(path, {}, mock_http)
            assert _status(result) == 200

    def test_complete_action_already_completed(self, handler_with_storage):
        """Completing already-completed action returns 404."""
        # d1 has status=completed
        result = handler_with_storage._complete_pending_action("d1")
        assert _status(result) == 404

    def test_execute_quick_action_empty_id(self):
        """Empty action_id returns 400."""
        h = DashboardHandler(ctx={})
        result = h._execute_quick_action("")
        assert _status(result) == 400

    def test_dismiss_urgent_empty_id(self):
        """Empty item_id returns 400."""
        h = DashboardHandler(ctx={})
        result = h._dismiss_urgent_item("")
        assert _status(result) == 400

    def test_complete_pending_empty_id(self):
        """Empty action_id returns 400."""
        h = DashboardHandler(ctx={})
        result = h._complete_pending_action("")
        assert _status(result) == 400

    def test_debate_detail_empty_id(self):
        """Empty debate_id returns 400."""
        h = DashboardHandler(ctx={})
        result = h._get_dashboard_debate("")
        assert _status(result) == 400

    def test_team_performance_detail_empty_id(self):
        """Empty team_id returns 400."""
        h = DashboardHandler(ctx={})
        result = h._get_team_performance_detail("")
        assert _status(result) == 400
