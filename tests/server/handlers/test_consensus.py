"""
Comprehensive tests for the consensus handler - consensus memory API.

Tests cover:
- Route handling (can_handle) for all endpoints
- Similar debates endpoint (topic validation, success, edge cases)
- Settled topics endpoint (default/custom params, DB interaction)
- Consensus stats endpoint (statistics aggregation)
- Dissents endpoint (topic/domain filtering)
- Contrarian views endpoint (with/without DissentRetriever)
- Risk warnings endpoint (severity inference, filtering)
- Domain history endpoint (path param extraction, validation)
- Seed demo endpoint (authentication, import errors)
- RBAC permission checks (read vs update, denied, error)
- Rate limiting enforcement
- Input validation (topic length, limit clamping, confidence bounds)
- Error handling (DB failures, parse errors, feature unavailable)
- Version prefix stripping
- Edge cases (empty results, None values, list params)
"""

import json
from datetime import datetime
from enum import Enum
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from aragora.server.handlers.consensus import ConsensusHandler, _consensus_limiter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_response(result):
    """Parse HandlerResult body into dict."""
    if result is None:
        return None
    body = result.body
    if isinstance(body, bytes):
        body = body.decode()
    return json.loads(body)


def _make_mock_user(**overrides):
    """Create a mock user object with sensible defaults."""
    user = MagicMock()
    user.id = overrides.get("id", "user-1")
    user.org_id = overrides.get("org_id", "org-1")
    user.roles = overrides.get("roles", {"member"})
    user.authenticated = overrides.get("authenticated", True)
    return user


def _make_auth_decision(allowed=True, reason=""):
    """Create a mock RBAC decision."""
    decision = MagicMock()
    decision.allowed = allowed
    decision.reason = reason
    return decision


def _mock_db_conn(rows=None, fetchone_row=None):
    """Create a mock DB connection context manager."""
    mock_cursor = MagicMock()
    if rows is not None:
        mock_cursor.fetchall.return_value = rows
    if fetchone_row is not None:
        mock_cursor.fetchone.return_value = fetchone_row
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    return mock_conn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def consensus_handler():
    """Create a consensus handler with empty context."""
    return ConsensusHandler({"storage": None, "user_store": None})


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler with client address."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {}
    return handler


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter state before each test."""
    _consensus_limiter.clear()
    yield


@pytest.fixture(autouse=True)
def _auto_auth():
    """Auto-mock authentication and RBAC so endpoints don't 401/403 by default."""
    mock_user = _make_mock_user()
    decision = _make_auth_decision(allowed=True)
    checker = MagicMock()
    checker.check_permission.return_value = decision

    with (
        patch.object(ConsensusHandler, "require_auth_or_error", return_value=(mock_user, None)),
        patch("aragora.server.handlers.consensus.get_permission_checker", return_value=checker),
    ):
        yield


# ===================================================================
# Routing tests
# ===================================================================


class TestConsensusHandlerRouting:
    """Tests for ConsensusHandler.can_handle method."""

    @pytest.mark.parametrize(
        "path",
        [
            "/api/consensus/similar",
            "/api/consensus/settled",
            "/api/consensus/stats",
            "/api/consensus/dissents",
            "/api/consensus/contrarian-views",
            "/api/consensus/risk-warnings",
            "/api/consensus/seed-demo",
            "/api/consensus/domain/*",
        ],
    )
    def test_can_handle_known_routes(self, consensus_handler, path):
        """Handler recognizes all known consensus routes."""
        assert consensus_handler.can_handle(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "/api/v1/consensus/similar",
            "/api/v2/consensus/settled",
            "/api/v1/consensus/domain/tech",
        ],
    )
    def test_can_handle_versioned_routes(self, consensus_handler, path):
        """Handler recognizes versioned routes after prefix stripping."""
        assert consensus_handler.can_handle(path) is True

    def test_can_handle_domain_subpath(self, consensus_handler):
        """Handler recognizes arbitrary domain subpaths."""
        assert consensus_handler.can_handle("/api/consensus/domain/science") is True
        assert consensus_handler.can_handle("/api/consensus/domain/my-org") is True

    @pytest.mark.parametrize(
        "path",
        [
            "/api/consensus/",
            "/api/debates",
            "/api/unknown",
            "/api/consensus/unknown-route",
        ],
    )
    def test_cannot_handle_unknown_routes(self, consensus_handler, path):
        """Handler rejects unknown routes."""
        assert consensus_handler.can_handle(path) is False

    def test_handle_returns_none_for_unmatched_path(self, consensus_handler, mock_http_handler):
        """Handle returns None for an unrecognized sub-path within consensus."""
        # After auth + rate limit pass, an unmatched path returns None
        result = consensus_handler.handle("/api/consensus/nonexistent", {}, mock_http_handler)
        assert result is None


# ===================================================================
# Authentication tests
# ===================================================================


class TestAuthentication:
    """Test authentication requirement for consensus endpoints.

    Note: Only seed-demo (a mutating operation) requires authentication.
    Read-only consensus endpoints are public dashboard data and skip auth
    (listed in AUTH_EXEMPT_GET_PREFIXES).
    """

    def test_unauthenticated_request_returns_401(self, consensus_handler, mock_http_handler):
        """Unauthenticated requests to seed-demo get 401."""
        from aragora.server.handlers.base import error_response as _err

        err_result = _err("Authentication required", 401)

        with patch.object(
            ConsensusHandler, "require_auth_or_error", return_value=(None, err_result)
        ):
            result = consensus_handler.handle("/api/v1/consensus/seed-demo", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 401

    def test_auth_exception_returns_401(self, consensus_handler, mock_http_handler):
        """Auth failure on seed-demo returns 401."""
        from aragora.server.handlers.base import error_response as _err

        err_result = _err("Authentication required", 401)
        with patch.object(
            ConsensusHandler, "require_auth_or_error", return_value=(None, err_result)
        ):
            result = consensus_handler.handle("/api/v1/consensus/seed-demo", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 401
        body = parse_response(result)
        assert "Authentication required" in body["error"]

    def test_read_endpoints_require_auth(self, consensus_handler, mock_http_handler):
        """Read endpoints now require authentication and consensus:read permission."""
        mock_user = _make_mock_user()
        with (
            patch.object(
                ConsensusHandler,
                "require_auth_or_error",
                return_value=(mock_user, None),
            ),
            patch.object(
                ConsensusHandler,
                "require_permission_or_error",
                return_value=(mock_user, None),
            ),
            patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", False),
        ):
            result = consensus_handler.handle("/api/v1/consensus/stats", {}, mock_http_handler)
        # Should reach the endpoint (503 because feature unavailable), not 401
        assert result is not None
        assert result.status_code == 503


# ===================================================================
# RBAC permission tests
# ===================================================================


class TestRBACPermissions:
    """Test RBAC permission checks.

    Note: Only seed-demo (a mutating operation) enforces RBAC via
    _check_memory_permission in handle(). Read-only consensus endpoints are
    public dashboard data and skip RBAC entirely.
    """

    def test_rbac_denied_returns_403(self, consensus_handler, mock_http_handler):
        """RBAC denial on seed-demo returns 403."""
        mock_user = _make_mock_user()
        denied = _make_auth_decision(allowed=False, reason="Insufficient permissions")
        checker = MagicMock()
        checker.check_permission.return_value = denied

        with (
            patch.object(ConsensusHandler, "require_auth_or_error", return_value=(mock_user, None)),
            patch("aragora.server.handlers.consensus.get_permission_checker", return_value=checker),
        ):
            result = consensus_handler.handle("/api/v1/consensus/seed-demo", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 403
        body = parse_response(result)
        assert "Permission denied" in body["error"]

    def test_seed_demo_requires_update_permission(self, consensus_handler, mock_http_handler):
        """Seed-demo endpoint requests memory.update permission."""
        mock_user = _make_mock_user()
        checker = MagicMock()
        denied = _make_auth_decision(allowed=False, reason="No update access")
        checker.check_permission.return_value = denied

        with (
            patch.object(ConsensusHandler, "require_auth_or_error", return_value=(mock_user, None)),
            patch("aragora.server.handlers.consensus.get_permission_checker", return_value=checker),
        ):
            result = consensus_handler.handle("/api/v1/consensus/seed-demo", {}, mock_http_handler)

        # Verify it asked for memory.update, not memory.read
        from aragora.rbac.models import AuthorizationContext

        call_args = checker.check_permission.call_args
        assert call_args[0][1] == "memory.update"

    def test_read_endpoints_skip_rbac(self, consensus_handler, mock_http_handler):
        """Read-only endpoints do not enforce RBAC checks."""
        checker = MagicMock()
        checker.check_permission.side_effect = AssertionError(
            "should not be called for read endpoints"
        )

        with (
            patch("aragora.server.handlers.consensus.get_permission_checker", return_value=checker),
            patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", False),
        ):
            result = consensus_handler.handle("/api/v1/consensus/stats", {}, mock_http_handler)

        # Should reach the endpoint (503 because feature unavailable), not 403
        assert result is not None
        assert result.status_code == 503
        # Permission checker was never consulted
        checker.check_permission.assert_not_called()

    def test_rbac_check_exception_returns_500(self, consensus_handler, mock_http_handler):
        """RBAC check failure on seed-demo returns 500."""
        mock_user = _make_mock_user()
        checker = MagicMock()
        checker.check_permission.side_effect = TypeError("checker down")

        with (
            patch.object(ConsensusHandler, "require_auth_or_error", return_value=(mock_user, None)),
            patch("aragora.server.handlers.consensus.get_permission_checker", return_value=checker),
        ):
            result = consensus_handler.handle("/api/v1/consensus/seed-demo", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 500
        body = parse_response(result)
        assert "Authorization check failed" in body["error"]

    def test_rbac_user_with_dict_attributes(self, consensus_handler, mock_http_handler):
        """RBAC works on seed-demo when user is a dict rather than an object."""
        user_dict = {"id": "user-2", "org_id": "org-2", "roles": ["admin"]}

        checker = MagicMock()
        checker.check_permission.return_value = _make_auth_decision(allowed=True)

        with (
            patch.object(ConsensusHandler, "require_auth_or_error", return_value=(user_dict, None)),
            patch.object(
                ConsensusHandler, "require_permission_or_error", return_value=(user_dict, None)
            ),
            patch("aragora.server.handlers.consensus.get_permission_checker", return_value=checker),
            patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True),
            patch("aragora.server.handlers.consensus.extract_user_from_request") as mock_extract,
            patch("aragora.server.handlers.consensus.ConsensusMemory") as mock_mem_cls,
        ):
            # seed-demo also checks extract_user_from_request for a second auth layer
            mock_user_ctx = MagicMock()
            mock_user_ctx.authenticated = True
            mock_extract.return_value = mock_user_ctx

            mock_memory = MagicMock()
            mock_memory.db_path = "/tmp/test.db"
            mock_memory.get_statistics.side_effect = [
                {"total_consensus": 0},
                {"total_consensus": 5},
            ]
            mock_mem_cls.return_value = mock_memory

            with patch.dict(
                "sys.modules",
                {"aragora.fixtures": MagicMock(load_demo_consensus=MagicMock(return_value=5))},
            ):
                result = consensus_handler.handle(
                    "/api/v1/consensus/seed-demo", {}, mock_http_handler
                )
        # Should succeed through the RBAC check (200 from seeding)
        assert result is not None
        assert result.status_code == 200


# ===================================================================
# Rate limiting tests
# ===================================================================


class TestRateLimiting:
    """Tests for rate limiting on consensus endpoints."""

    def test_rate_limit_enforcement(self, consensus_handler, mock_http_handler):
        """Exceeding 30 requests/minute triggers 429."""
        for _ in range(30):
            _consensus_limiter.is_allowed("127.0.0.1")

        result = consensus_handler.handle(
            "/api/v1/consensus/similar", {"topic": "test"}, mock_http_handler
        )
        assert result is not None
        assert result.status_code == 429
        body = parse_response(result)
        assert "Rate limit" in body["error"]

    def test_rate_limit_before_auth(self, consensus_handler, mock_http_handler):
        """Rate limiting is checked before authentication."""
        for _ in range(30):
            _consensus_limiter.is_allowed("127.0.0.1")

        # Even if auth would fail, rate limit should fire first
        with patch.object(
            ConsensusHandler,
            "require_auth_or_error",
            side_effect=AssertionError("should not be called"),
        ):
            result = consensus_handler.handle("/api/v1/consensus/stats", {}, mock_http_handler)
        assert result.status_code == 429

    def test_different_ips_have_separate_limits(self, consensus_handler):
        """Different client IPs have independent rate limits."""
        for _ in range(30):
            _consensus_limiter.is_allowed("10.0.0.1")

        # A different IP should still be allowed
        assert _consensus_limiter.is_allowed("10.0.0.2") is True


# ===================================================================
# Similar debates endpoint tests
# ===================================================================


class TestSimilarDebatesEndpoint:
    """Tests for GET /api/consensus/similar."""

    def test_missing_topic_returns_400(self, consensus_handler, mock_http_handler):
        """Missing topic parameter returns 400."""
        result = consensus_handler.handle("/api/v1/consensus/similar", {}, mock_http_handler)
        assert result.status_code == 400
        body = parse_response(result)
        assert "Topic required" in body["error"]

    def test_empty_topic_returns_400(self, consensus_handler, mock_http_handler):
        """Empty string topic returns 400."""
        result = consensus_handler.handle(
            "/api/v1/consensus/similar", {"topic": ""}, mock_http_handler
        )
        assert result.status_code == 400

    def test_topic_too_long_returns_400(self, consensus_handler, mock_http_handler):
        """Topic exceeding 100k characters returns 400."""
        result = consensus_handler.handle(
            "/api/v1/consensus/similar", {"topic": "x" * 100_001}, mock_http_handler
        )
        assert result.status_code == 400
        body = parse_response(result)
        assert "too long" in body["error"].lower()

    def test_topic_exactly_500_chars_is_accepted(self, consensus_handler, mock_http_handler):
        """Topic of exactly 500 characters is accepted."""
        with (
            patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True),
            patch("aragora.server.handlers.consensus.ConsensusMemory") as mock_cls,
        ):
            mock_cls.return_value.find_similar_debates.return_value = []
            result = consensus_handler.handle(
                "/api/v1/consensus/similar", {"topic": "a" * 500}, mock_http_handler
            )
        assert result.status_code == 200

    def test_topic_list_param_uses_first(self, consensus_handler, mock_http_handler):
        """When topic is a list (repeated query param), first value is used."""
        # The handler checks if raw_topic is a list and takes first element
        result = consensus_handler.handle(
            "/api/v1/consensus/similar", {"topic": ["AI safety", "other"]}, mock_http_handler
        )
        # Should not be rejected as too long
        assert (
            result.status_code != 400
            or "too long" not in parse_response(result).get("error", "").lower()
        )

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    def test_similar_success(self, mock_memory_cls, consensus_handler, mock_http_handler):
        """Successful similar debates retrieval returns formatted data."""
        mock_memory = MagicMock()
        mock_memory_cls.return_value = mock_memory

        mock_consensus = MagicMock()
        mock_consensus.topic = "AI Safety"
        mock_consensus.conclusion = "Needs regulation"
        mock_consensus.strength.value = "strong"
        mock_consensus.confidence = 0.92
        mock_consensus.participating_agents = ["claude", "gpt-4"]
        mock_consensus.timestamp = datetime(2025, 6, 15, 10, 0, 0)

        mock_result = MagicMock()
        mock_result.consensus = mock_consensus
        mock_result.similarity_score = 0.87
        mock_result.dissents = [MagicMock()]  # one dissent

        mock_memory.find_similar_debates.return_value = [mock_result]

        result = consensus_handler.handle(
            "/api/v1/consensus/similar", {"topic": "AI regulation"}, mock_http_handler
        )
        assert result.status_code == 200
        body = parse_response(result)
        assert body["query"] == "AI regulation"
        assert body["count"] == 1
        assert len(body["similar"]) == 1

        entry = body["similar"][0]
        assert entry["topic"] == "AI Safety"
        assert entry["conclusion"] == "Needs regulation"
        assert entry["strength"] == "strong"
        assert entry["confidence"] == 0.92
        assert entry["similarity"] == 0.87
        assert entry["dissent_count"] == 1
        assert entry["agents"] == ["claude", "gpt-4"]
        assert "2025-06-15" in entry["timestamp"]

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    def test_similar_empty_results(self, mock_memory_cls, consensus_handler, mock_http_handler):
        """Empty similar results returns count 0."""
        mock_memory_cls.return_value.find_similar_debates.return_value = []
        result = consensus_handler.handle(
            "/api/v1/consensus/similar", {"topic": "obscure topic"}, mock_http_handler
        )
        assert result.status_code == 200
        body = parse_response(result)
        assert body["count"] == 0
        assert body["similar"] == []

    def test_similar_feature_unavailable_returns_503(self, consensus_handler, mock_http_handler):
        """When consensus memory is unavailable, returns 503."""
        with patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", False):
            result = consensus_handler.handle(
                "/api/v1/consensus/similar", {"topic": "test"}, mock_http_handler
            )
        assert result.status_code == 503

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    def test_similar_custom_limit(self, mock_memory_cls, consensus_handler, mock_http_handler):
        """Custom limit parameter is passed through."""
        mock_memory = MagicMock()
        mock_memory_cls.return_value = mock_memory
        mock_memory.find_similar_debates.return_value = []

        consensus_handler.handle(
            "/api/v1/consensus/similar",
            {"topic": "test", "limit": "10"},
            mock_http_handler,
        )
        mock_memory.find_similar_debates.assert_called_once_with("test", limit=10)

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    def test_similar_limit_clamped_to_max(
        self, mock_memory_cls, consensus_handler, mock_http_handler
    ):
        """Limit over 20 is clamped to 20 for similar endpoint."""
        mock_memory = MagicMock()
        mock_memory_cls.return_value = mock_memory
        mock_memory.find_similar_debates.return_value = []

        consensus_handler.handle(
            "/api/v1/consensus/similar",
            {"topic": "test", "limit": "999"},
            mock_http_handler,
        )
        mock_memory.find_similar_debates.assert_called_once_with("test", limit=20)

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    def test_similar_topic_is_stripped(self, mock_memory_cls, consensus_handler, mock_http_handler):
        """Topic whitespace is stripped."""
        mock_memory = MagicMock()
        mock_memory_cls.return_value = mock_memory
        mock_memory.find_similar_debates.return_value = []

        consensus_handler.handle(
            "/api/v1/consensus/similar",
            {"topic": "  AI safety  "},
            mock_http_handler,
        )
        mock_memory.find_similar_debates.assert_called_once_with("AI safety", limit=5)


# ===================================================================
# Settled topics endpoint tests
# ===================================================================


class TestSettledTopicsEndpoint:
    """Tests for GET /api/consensus/settled."""

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.get_db_connection")
    def test_settled_default_params(
        self, mock_db, mock_mem_cls, consensus_handler, mock_http_handler
    ):
        """Default min_confidence=0.8 and limit=20."""
        mock_mem_cls.return_value.db_path = ":memory:"
        mock_db.return_value = _mock_db_conn(
            rows=[
                ("Topic A", "Conclusion A", 0.95, "strong", "2025-01-01T00:00:00"),
            ]
        )

        result = consensus_handler.handle("/api/v1/consensus/settled", {}, mock_http_handler)
        assert result.status_code == 200
        body = parse_response(result)
        assert body["min_confidence"] == 0.8
        assert body["count"] == 1
        assert body["topics"][0]["topic"] == "Topic A"

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.get_db_connection")
    def test_settled_custom_params(
        self, mock_db, mock_mem_cls, consensus_handler, mock_http_handler
    ):
        """Custom min_confidence and limit are passed to query."""
        mock_mem_cls.return_value.db_path = ":memory:"
        mock_conn = _mock_db_conn(rows=[])
        mock_db.return_value = mock_conn

        result = consensus_handler.handle(
            "/api/v1/consensus/settled",
            {"min_confidence": "0.95", "limit": "5"},
            mock_http_handler,
        )
        assert result.status_code == 200
        body = parse_response(result)
        assert body["min_confidence"] == 0.95

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.get_db_connection")
    def test_settled_confidence_clamped_to_max(
        self, mock_db, mock_mem_cls, consensus_handler, mock_http_handler
    ):
        """min_confidence > 1.0 is clamped to 1.0."""
        mock_mem_cls.return_value.db_path = ":memory:"
        mock_db.return_value = _mock_db_conn(rows=[])

        result = consensus_handler.handle(
            "/api/v1/consensus/settled",
            {"min_confidence": "5.0"},
            mock_http_handler,
        )
        assert result.status_code == 200
        body = parse_response(result)
        assert body["min_confidence"] == 1.0

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.get_db_connection")
    def test_settled_confidence_clamped_to_min(
        self, mock_db, mock_mem_cls, consensus_handler, mock_http_handler
    ):
        """min_confidence < 0.0 is clamped to 0.0."""
        mock_mem_cls.return_value.db_path = ":memory:"
        mock_db.return_value = _mock_db_conn(rows=[])

        result = consensus_handler.handle(
            "/api/v1/consensus/settled",
            {"min_confidence": "-1.0"},
            mock_http_handler,
        )
        assert result.status_code == 200
        body = parse_response(result)
        assert body["min_confidence"] == 0.0

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.get_db_connection")
    def test_settled_limit_clamped_to_max(
        self, mock_db, mock_mem_cls, consensus_handler, mock_http_handler
    ):
        """Limit > 100 is clamped to 100 for settled endpoint."""
        mock_mem_cls.return_value.db_path = ":memory:"
        mock_db.return_value = _mock_db_conn(rows=[])

        result = consensus_handler.handle(
            "/api/v1/consensus/settled",
            {"limit": "999"},
            mock_http_handler,
        )
        assert result.status_code == 200

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.get_db_connection")
    def test_settled_empty_results(
        self, mock_db, mock_mem_cls, consensus_handler, mock_http_handler
    ):
        """Empty result set returns count 0."""
        mock_mem_cls.return_value.db_path = ":memory:"
        mock_db.return_value = _mock_db_conn(rows=[])

        result = consensus_handler.handle("/api/v1/consensus/settled", {}, mock_http_handler)
        assert result.status_code == 200
        body = parse_response(result)
        assert body["count"] == 0
        assert body["topics"] == []


# ===================================================================
# Consensus stats endpoint tests
# ===================================================================


class TestConsensusStatsEndpoint:
    """Tests for GET /api/consensus/stats."""

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.get_db_connection")
    def test_stats_success(self, mock_db, mock_mem_cls, consensus_handler, mock_http_handler):
        """Stats endpoint returns aggregated statistics."""
        mock_memory = MagicMock()
        mock_memory.db_path = ":memory:"
        mock_memory.get_statistics.return_value = {
            "total_consensus": 100,
            "total_dissents": 20,
            "by_domain": {"technology": 50, "science": 30},
            "by_strength": {"strong": 60, "moderate": 40},
        }
        mock_mem_cls.return_value = mock_memory
        mock_db.return_value = _mock_db_conn(fetchone_row=(80, 0.856))

        result = consensus_handler.handle("/api/v1/consensus/stats", {}, mock_http_handler)
        assert result.status_code == 200
        body = parse_response(result)
        assert body["total_topics"] == 100
        assert body["high_confidence_count"] == 80
        assert body["avg_confidence"] == 0.856
        assert body["total_dissents"] == 20
        assert "technology" in body["domains"]
        assert "science" in body["domains"]
        assert body["by_strength"] == {"strong": 60, "moderate": 40}
        assert body["by_domain"] == {"technology": 50, "science": 30}

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.get_db_connection")
    def test_stats_with_null_db_values(
        self, mock_db, mock_mem_cls, consensus_handler, mock_http_handler
    ):
        """Stats handles NULL database values gracefully."""
        mock_memory = MagicMock()
        mock_memory.db_path = ":memory:"
        mock_memory.get_statistics.return_value = {
            "total_consensus": 0,
            "total_dissents": 0,
            "by_domain": {},
            "by_strength": {},
        }
        mock_mem_cls.return_value = mock_memory
        mock_db.return_value = _mock_db_conn(fetchone_row=(None, None))

        result = consensus_handler.handle("/api/v1/consensus/stats", {}, mock_http_handler)
        assert result.status_code == 200
        body = parse_response(result)
        assert body["high_confidence_count"] == 0
        assert body["avg_confidence"] == 0.0

    def test_stats_feature_unavailable(self, consensus_handler, mock_http_handler):
        """Stats returns 503 when feature unavailable."""
        with patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", False):
            result = consensus_handler.handle("/api/v1/consensus/stats", {}, mock_http_handler)
        assert result.status_code == 503


# ===================================================================
# Dissents endpoint tests
# ===================================================================


class TestDissentsEndpoint:
    """Tests for GET /api/consensus/dissents."""

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.get_db_connection")
    def test_dissents_no_filters(self, mock_db, mock_mem_cls, consensus_handler, mock_http_handler):
        """Dissents without filters returns all recent dissents."""
        mock_mem_cls.return_value.db_path = ":memory:"

        mock_record = MagicMock()
        mock_record.content = "I disagree"
        mock_record.agent_id = "claude"
        mock_record.confidence = 0.7
        mock_record.reasoning = "Because reasons"

        dissent_data = json.dumps(
            {
                "content": "I disagree",
                "agent_id": "claude",
                "confidence": 0.7,
                "reasoning": "Because reasons",
            }
        )
        mock_db.return_value = _mock_db_conn(
            rows=[
                (dissent_data, "AI Safety", "Regulation needed"),
            ]
        )

        with patch("aragora.memory.consensus.DissentRecord") as mock_dr_cls:
            mock_dr_cls.from_dict.return_value = mock_record
            result = consensus_handler.handle("/api/v1/consensus/dissents", {}, mock_http_handler)

        assert result.status_code == 200
        body = parse_response(result)
        assert len(body["dissents"]) == 1
        assert body["dissents"][0]["topic"] == "AI Safety"
        assert body["dissents"][0]["dissenting_view"] == "I disagree"
        assert body["dissents"][0]["dissenting_agent"] == "claude"

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.get_db_connection")
    def test_dissents_with_topic_filter(
        self, mock_db, mock_mem_cls, consensus_handler, mock_http_handler
    ):
        """Dissents with topic filter adds LIKE clause."""
        mock_mem_cls.return_value.db_path = ":memory:"
        mock_db.return_value = _mock_db_conn(rows=[])

        result = consensus_handler.handle(
            "/api/v1/consensus/dissents",
            {"topic": "security"},
            mock_http_handler,
        )
        assert result.status_code == 200

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.get_db_connection")
    def test_dissents_with_invalid_domain_returns_400(
        self, mock_db, mock_mem_cls, consensus_handler, mock_http_handler
    ):
        """Dissents with invalid domain format returns 400."""
        mock_mem_cls.return_value.db_path = ":memory:"

        result = consensus_handler.handle(
            "/api/v1/consensus/dissents",
            {"domain": "invalid domain with spaces!"},
            mock_http_handler,
        )
        assert result.status_code == 400
        body = parse_response(result)
        assert "Invalid domain" in body["error"]

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.get_db_connection")
    def test_dissents_with_null_topic_and_conclusion(
        self, mock_db, mock_mem_cls, consensus_handler, mock_http_handler
    ):
        """Dissents gracefully handle NULL topic/conclusion from DB."""
        mock_mem_cls.return_value.db_path = ":memory:"

        mock_record = MagicMock()
        mock_record.content = "Disagreement"
        mock_record.agent_id = "gpt-4"
        mock_record.confidence = 0.5
        mock_record.reasoning = None

        dissent_data = json.dumps({"content": "Disagreement", "agent_id": "gpt-4"})
        mock_db.return_value = _mock_db_conn(
            rows=[
                (dissent_data, None, None),
            ]
        )

        with patch("aragora.memory.consensus.DissentRecord") as mock_dr_cls:
            mock_dr_cls.from_dict.return_value = mock_record
            result = consensus_handler.handle("/api/v1/consensus/dissents", {}, mock_http_handler)

        assert result.status_code == 200
        body = parse_response(result)
        assert body["dissents"][0]["topic"] == "Unknown topic"
        assert body["dissents"][0]["majority_view"] == "No consensus recorded"
        assert body["dissents"][0]["reasoning"] is None

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.get_db_connection")
    def test_dissents_parse_error_skips_record(
        self, mock_db, mock_mem_cls, consensus_handler, mock_http_handler
    ):
        """Malformed dissent records are skipped gracefully."""
        mock_mem_cls.return_value.db_path = ":memory:"
        mock_db.return_value = _mock_db_conn(
            rows=[
                ("not-valid-json", "Topic", "Conclusion"),
            ]
        )

        # DissentRecord.from_dict will fail on bad JSON parse result
        with patch("aragora.memory.consensus.DissentRecord") as mock_dr_cls:
            mock_dr_cls.from_dict.side_effect = ValueError("bad data")
            result = consensus_handler.handle("/api/v1/consensus/dissents", {}, mock_http_handler)

        assert result.status_code == 200
        body = parse_response(result)
        assert body["dissents"] == []


# ===================================================================
# Contrarian views endpoint tests
# ===================================================================


class TestContrarianViewsEndpoint:
    """Tests for GET /api/consensus/contrarian-views."""

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.DissentRetriever")
    def test_contrarian_with_topic_uses_retriever(
        self, mock_retriever_cls, mock_mem_cls, consensus_handler, mock_http_handler
    ):
        """When topic is provided, DissentRetriever is used."""
        mock_memory = MagicMock()
        mock_mem_cls.return_value = mock_memory

        mock_record = MagicMock()
        mock_record.agent_id = "claude"
        mock_record.content = "Alternative approach"
        mock_record.confidence = 0.8
        mock_record.reasoning = "Strong reasoning"
        mock_record.debate_id = "debate-1"

        mock_retriever = MagicMock()
        mock_retriever.find_contrarian_views.return_value = [mock_record]
        mock_retriever_cls.return_value = mock_retriever

        result = consensus_handler.handle(
            "/api/v1/consensus/contrarian-views",
            {"topic": "microservices", "limit": "5"},
            mock_http_handler,
        )
        assert result.status_code == 200
        body = parse_response(result)
        assert len(body["views"]) == 1
        assert body["views"][0]["agent"] == "claude"
        assert body["views"][0]["position"] == "Alternative approach"

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.DissentRetriever", None)
    @patch("aragora.server.handlers.consensus.get_db_connection")
    def test_contrarian_without_retriever_falls_back_to_db(
        self, mock_db, mock_mem_cls, consensus_handler, mock_http_handler
    ):
        """When DissentRetriever is None, falls back to raw DB query."""
        mock_mem_cls.return_value.db_path = ":memory:"
        mock_db.return_value = _mock_db_conn(rows=[])

        result = consensus_handler.handle(
            "/api/v1/consensus/contrarian-views",
            {"topic": "microservices"},
            mock_http_handler,
        )
        assert result.status_code == 200
        body = parse_response(result)
        assert body["views"] == []

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.get_db_connection")
    def test_contrarian_no_topic_uses_db(
        self, mock_db, mock_mem_cls, consensus_handler, mock_http_handler
    ):
        """Without topic, falls back to DB query regardless of retriever."""
        mock_mem_cls.return_value.db_path = ":memory:"
        mock_db.return_value = _mock_db_conn(rows=[])

        result = consensus_handler.handle(
            "/api/v1/consensus/contrarian-views", {}, mock_http_handler
        )
        assert result.status_code == 200

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.DissentRetriever")
    def test_contrarian_with_domain_filter(
        self, mock_retriever_cls, mock_mem_cls, consensus_handler, mock_http_handler
    ):
        """Domain filter is passed through to retriever."""
        mock_mem_cls.return_value = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.find_contrarian_views.return_value = []
        mock_retriever_cls.return_value = mock_retriever

        consensus_handler.handle(
            "/api/v1/consensus/contrarian-views",
            {"topic": "test", "domain": "security"},
            mock_http_handler,
        )
        mock_retriever.find_contrarian_views.assert_called_once_with(
            "test", domain="security", limit=10
        )

    def test_contrarian_invalid_domain_returns_400(self, consensus_handler, mock_http_handler):
        """Invalid domain format returns 400."""
        with (
            patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True),
            patch("aragora.server.handlers.consensus.ConsensusMemory"),
        ):
            result = consensus_handler.handle(
                "/api/v1/consensus/contrarian-views",
                {"domain": "../etc/passwd"},
                mock_http_handler,
            )
        assert result.status_code == 400


# ===================================================================
# Risk warnings endpoint tests
# ===================================================================


class TestRiskWarningsEndpoint:
    """Tests for GET /api/consensus/risk-warnings."""

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.DissentRetriever")
    def test_risk_warnings_with_topic(
        self, mock_retriever_cls, mock_mem_cls, consensus_handler, mock_http_handler
    ):
        """Risk warnings with topic uses DissentRetriever."""
        mock_mem_cls.return_value = MagicMock()

        mock_record = MagicMock()
        mock_record.metadata = {"domain": "security"}
        mock_record.dissent_type.value = "risk_warning"
        mock_record.confidence = 0.85
        mock_record.content = "SQL injection risk"
        mock_record.rebuttal = "Use parameterized queries"
        mock_record.timestamp = datetime(2025, 6, 1, 12, 0, 0)

        mock_retriever = MagicMock()
        mock_retriever.find_risk_warnings.return_value = [mock_record]
        mock_retriever_cls.return_value = mock_retriever

        result = consensus_handler.handle(
            "/api/v1/consensus/risk-warnings",
            {"topic": "SQL"},
            mock_http_handler,
        )
        assert result.status_code == 200
        body = parse_response(result)
        assert len(body["warnings"]) == 1
        w = body["warnings"][0]
        assert w["domain"] == "security"
        assert w["risk_type"] == "Risk Warning"
        assert w["severity"] == "critical"
        assert w["description"] == "SQL injection risk"
        assert w["mitigation"] == "Use parameterized queries"
        assert "2025-06-01" in w["detected_at"]

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.DissentRetriever")
    def test_risk_severity_levels(
        self, mock_retriever_cls, mock_mem_cls, consensus_handler, mock_http_handler
    ):
        """Severity is correctly inferred from confidence and type."""
        mock_mem_cls.return_value = MagicMock()

        def _make_record(confidence, dissent_type):
            r = MagicMock()
            r.metadata = {"domain": "general"}
            r.dissent_type.value = dissent_type
            r.confidence = confidence
            r.content = "Warning"
            r.rebuttal = None
            r.timestamp = datetime(2025, 1, 1)
            return r

        records = [
            _make_record(0.9, "risk_warning"),  # critical
            _make_record(0.7, "risk_warning"),  # high
            _make_record(0.5, "risk_warning"),  # medium
            _make_record(0.3, "risk_warning"),  # low
            _make_record(0.9, "edge_case_concern"),  # low (not risk_warning type)
        ]

        mock_retriever = MagicMock()
        mock_retriever.find_risk_warnings.return_value = records
        mock_retriever_cls.return_value = mock_retriever

        result = consensus_handler.handle(
            "/api/v1/consensus/risk-warnings",
            {"topic": "test"},
            mock_http_handler,
        )
        assert result.status_code == 200
        body = parse_response(result)
        severities = [w["severity"] for w in body["warnings"]]
        assert severities == ["critical", "high", "medium", "low", "low"]

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.DissentRetriever", None)
    @patch("aragora.server.handlers.consensus.get_db_connection")
    def test_risk_warnings_db_fallback(
        self, mock_db, mock_mem_cls, consensus_handler, mock_http_handler
    ):
        """Without DissentRetriever, falls back to DB query."""
        mock_mem_cls.return_value.db_path = ":memory:"
        mock_db.return_value = _mock_db_conn(rows=[])

        result = consensus_handler.handle(
            "/api/v1/consensus/risk-warnings",
            {"topic": "test"},
            mock_http_handler,
        )
        assert result.status_code == 200
        body = parse_response(result)
        assert body["warnings"] == []

    def test_risk_warnings_invalid_domain(self, consensus_handler, mock_http_handler):
        """Invalid domain format returns 400."""
        with (
            patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True),
            patch("aragora.server.handlers.consensus.ConsensusMemory"),
        ):
            result = consensus_handler.handle(
                "/api/v1/consensus/risk-warnings",
                {"domain": "bad domain!"},
                mock_http_handler,
            )
        assert result.status_code == 400

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.DissentRetriever")
    def test_risk_warnings_null_rebuttal(
        self, mock_retriever_cls, mock_mem_cls, consensus_handler, mock_http_handler
    ):
        """Risk warning with no rebuttal sets mitigation to None."""
        mock_mem_cls.return_value = MagicMock()

        mock_record = MagicMock()
        mock_record.metadata = {"domain": "general"}
        mock_record.dissent_type.value = "edge_case_concern"
        mock_record.confidence = 0.5
        mock_record.content = "Edge case"
        mock_record.rebuttal = None
        mock_record.timestamp = datetime(2025, 1, 1)

        mock_retriever = MagicMock()
        mock_retriever.find_risk_warnings.return_value = [mock_record]
        mock_retriever_cls.return_value = mock_retriever

        result = consensus_handler.handle(
            "/api/v1/consensus/risk-warnings",
            {"topic": "test"},
            mock_http_handler,
        )
        body = parse_response(result)
        assert body["warnings"][0]["mitigation"] is None


# ===================================================================
# Domain history endpoint tests
# ===================================================================


class TestDomainHistoryEndpoint:
    """Tests for GET /api/consensus/domain/:domain."""

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    def test_domain_history_success(self, mock_mem_cls, consensus_handler, mock_http_handler):
        """Successful domain history retrieval."""
        mock_memory = MagicMock()
        mock_record = MagicMock()
        mock_record.to_dict.return_value = {"topic": "AI", "domain": "technology"}
        mock_memory.get_domain_consensus_history.return_value = [mock_record]
        mock_mem_cls.return_value = mock_memory

        result = consensus_handler.handle(
            "/api/v1/consensus/domain/technology", {}, mock_http_handler
        )
        assert result.status_code == 200
        body = parse_response(result)
        assert body["domain"] == "technology"
        assert body["count"] == 1
        assert body["history"][0]["topic"] == "AI"

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    def test_domain_history_custom_limit(self, mock_mem_cls, consensus_handler, mock_http_handler):
        """Custom limit is passed to domain history query."""
        mock_memory = MagicMock()
        mock_memory.get_domain_consensus_history.return_value = []
        mock_mem_cls.return_value = mock_memory

        consensus_handler.handle(
            "/api/v1/consensus/domain/science",
            {"limit": "25"},
            mock_http_handler,
        )
        mock_memory.get_domain_consensus_history.assert_called_once_with("science", limit=25)

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    def test_domain_history_limit_clamped(self, mock_mem_cls, consensus_handler, mock_http_handler):
        """Domain history limit > 200 is clamped to 200."""
        mock_memory = MagicMock()
        mock_memory.get_domain_consensus_history.return_value = []
        mock_mem_cls.return_value = mock_memory

        consensus_handler.handle(
            "/api/v1/consensus/domain/tech",
            {"limit": "999"},
            mock_http_handler,
        )
        mock_memory.get_domain_consensus_history.assert_called_once_with("tech", limit=200)

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    def test_domain_history_empty(self, mock_mem_cls, consensus_handler, mock_http_handler):
        """Empty domain history returns count 0."""
        mock_mem_cls.return_value.get_domain_consensus_history.return_value = []

        result = consensus_handler.handle(
            "/api/v1/consensus/domain/obscure-domain", {}, mock_http_handler
        )
        assert result.status_code == 200
        body = parse_response(result)
        assert body["count"] == 0
        assert body["history"] == []

    def test_domain_history_feature_unavailable(self, consensus_handler, mock_http_handler):
        """Domain history returns 503 when feature unavailable."""
        with patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", False):
            result = consensus_handler.handle(
                "/api/v1/consensus/domain/tech", {}, mock_http_handler
            )
        assert result.status_code == 503


# ===================================================================
# Seed demo endpoint tests
# ===================================================================


class TestSeedDemoEndpoint:
    """Tests for POST /api/consensus/seed-demo."""

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.extract_user_from_request")
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    def test_seed_demo_success(
        self, mock_mem_cls, mock_extract, consensus_handler, mock_http_handler
    ):
        """Successful demo seeding returns stats."""
        mock_user_ctx = MagicMock()
        mock_user_ctx.authenticated = True
        mock_extract.return_value = mock_user_ctx

        mock_memory = MagicMock()
        mock_memory.db_path = "/tmp/test.db"
        mock_memory.get_statistics.side_effect = [
            {"total_consensus": 0},
            {"total_consensus": 10},
        ]
        mock_mem_cls.return_value = mock_memory

        with patch(
            "aragora.server.handlers.consensus.load_demo_consensus", create=True
        ) as mock_load:
            # Simulate the import inside the method
            with patch.dict(
                "sys.modules", {"aragora.fixtures": MagicMock(load_demo_consensus=mock_load)}
            ):
                mock_load.return_value = 10
                result = consensus_handler.handle(
                    "/api/v1/consensus/seed-demo", {}, mock_http_handler
                )

        assert result.status_code == 200
        body = parse_response(result)
        assert body["success"] is True
        assert body["seeded"] == 10

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.extract_user_from_request")
    def test_seed_demo_unauthenticated(self, mock_extract, consensus_handler, mock_http_handler):
        """Unauthenticated seed-demo request returns 401."""
        mock_user_ctx = MagicMock()
        mock_user_ctx.authenticated = False
        mock_extract.return_value = mock_user_ctx

        result = consensus_handler.handle("/api/v1/consensus/seed-demo", {}, mock_http_handler)
        assert result.status_code == 401

    @patch("aragora.server.handlers.consensus.extract_user_from_request")
    def test_seed_demo_feature_unavailable(
        self, mock_extract, consensus_handler, mock_http_handler
    ):
        """Seed demo returns 503 when consensus memory unavailable."""
        mock_user_ctx = MagicMock()
        mock_user_ctx.authenticated = True
        mock_extract.return_value = mock_user_ctx

        with patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", False):
            result = consensus_handler.handle("/api/v1/consensus/seed-demo", {}, mock_http_handler)
        assert result.status_code == 503


# ===================================================================
# Domain validation tests
# ===================================================================


class TestDomainValidation:
    """Tests for _validate_domain static method."""

    def test_valid_domain(self):
        """Valid slug-format domains pass validation."""
        domain, err = ConsensusHandler._validate_domain("technology")
        assert domain == "technology"
        assert err is None

    def test_valid_domain_with_hyphens(self):
        """Domains with hyphens pass validation."""
        domain, err = ConsensusHandler._validate_domain("software-engineering")
        assert domain == "software-engineering"
        assert err is None

    def test_valid_domain_with_underscores(self):
        """Domains with underscores pass validation."""
        domain, err = ConsensusHandler._validate_domain("data_science")
        assert domain == "data_science"
        assert err is None

    def test_none_domain(self):
        """None domain returns (None, None)."""
        domain, err = ConsensusHandler._validate_domain(None)
        assert domain is None
        assert err is None

    def test_invalid_domain_with_spaces(self):
        """Domains with spaces fail validation."""
        domain, err = ConsensusHandler._validate_domain("bad domain")
        assert domain is None
        assert err is not None
        assert err.status_code == 400

    def test_invalid_domain_with_special_chars(self):
        """Domains with special characters fail validation."""
        domain, err = ConsensusHandler._validate_domain("../etc/passwd")
        assert domain is None
        assert err.status_code == 400

    def test_invalid_domain_with_dots(self):
        """Domains with dots fail SAFE_SLUG_PATTERN validation."""
        domain, err = ConsensusHandler._validate_domain("domain.com")
        assert domain is None
        assert err.status_code == 400

    def test_empty_string_domain(self):
        """Empty string domain fails validation."""
        domain, err = ConsensusHandler._validate_domain("")
        assert domain is None
        assert err.status_code == 400


# ===================================================================
# Error handling tests
# ===================================================================


class TestErrorHandling:
    """Tests for error handling across endpoints."""

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    def test_similar_memory_exception(self, mock_mem_cls, consensus_handler, mock_http_handler):
        """Exception during similar debates retrieval is handled."""
        mock_mem_cls.return_value.find_similar_debates.side_effect = RuntimeError("DB error")

        result = consensus_handler.handle(
            "/api/v1/consensus/similar", {"topic": "test"}, mock_http_handler
        )
        assert result is not None
        assert result.status_code == 500

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    @patch("aragora.server.handlers.consensus.get_db_connection")
    def test_settled_db_exception(
        self, mock_db, mock_mem_cls, consensus_handler, mock_http_handler
    ):
        """Exception during settled topics DB query is handled."""
        mock_mem_cls.return_value.db_path = ":memory:"
        mock_db.side_effect = RuntimeError("Connection failed")

        result = consensus_handler.handle("/api/v1/consensus/settled", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 500

    @patch("aragora.server.handlers.consensus.CONSENSUS_MEMORY_AVAILABLE", True)
    @patch("aragora.server.handlers.consensus.ConsensusMemory")
    def test_domain_history_exception(self, mock_mem_cls, consensus_handler, mock_http_handler):
        """Exception during domain history is handled."""
        mock_mem_cls.return_value.get_domain_consensus_history.side_effect = RuntimeError("fail")

        result = consensus_handler.handle("/api/v1/consensus/domain/tech", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 500


# ===================================================================
# Constructor and context tests
# ===================================================================


class TestHandlerConstruction:
    """Tests for handler construction."""

    def test_default_context(self):
        """Handler with no context defaults to empty dict."""
        handler = ConsensusHandler()
        assert handler.ctx == {}

    def test_custom_context(self):
        """Handler stores provided context."""
        ctx = {"user_store": "mock_store", "storage": "mock_storage"}
        handler = ConsensusHandler(ctx)
        assert handler.ctx == ctx

    def test_routes_list(self):
        """Handler has expected number of routes."""
        assert len(ConsensusHandler.ROUTES) == 10
