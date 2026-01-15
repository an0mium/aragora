"""
Tests for EvolutionABTestingHandler - A/B testing endpoints for evolution.

Tests cover:
- GET /api/evolution/ab-tests - List all tests
- GET /api/evolution/ab-tests/{agent}/active - Get active test
- POST /api/evolution/ab-tests - Create new test
- GET /api/evolution/ab-tests/{id} - Get specific test
- POST /api/evolution/ab-tests/{id}/record - Record result
- POST /api/evolution/ab-tests/{id}/conclude - Conclude test
- DELETE /api/evolution/ab-tests/{id} - Cancel test
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from aragora.server.handlers.evolution import EvolutionABTestingHandler
from aragora.server.handlers.base import clear_cache
import aragora.server.handlers.evolution.ab_testing as ab_mod


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def handler(tmp_path):
    """Create handler with temp database."""
    ctx = {"ab_tests_db": str(tmp_path / "test_ab.db")}
    return EvolutionABTestingHandler(ctx)


@pytest.fixture
def mock_test():
    """Create mock ABTest."""
    test = Mock()
    test.id = "test-123"
    test.agent = "claude"
    test.baseline_prompt_version = 1
    test.evolved_prompt_version = 2
    test.baseline_wins = 5
    test.evolved_wins = 7
    test.baseline_debates = 10
    test.evolved_debates = 10
    test.evolved_win_rate = 0.58
    test.baseline_win_rate = 0.42
    test.total_debates = 20
    test.sample_size = 12
    test.is_significant = False
    test.started_at = "2024-01-01T00:00:00Z"
    test.concluded_at = None
    test.status = Mock()
    test.status.value = "active"
    test.metadata = {}
    test.to_dict.return_value = {
        "id": "test-123",
        "agent": "claude",
        "baseline_prompt_version": 1,
        "evolved_prompt_version": 2,
        "baseline_wins": 5,
        "evolved_wins": 7,
        "status": "active",
    }
    return test


@pytest.fixture
def mock_manager(mock_test):
    """Create mock ABTestManager."""
    manager = Mock()
    manager.db_path = Path("test.db")
    manager.get_test.return_value = mock_test
    manager.get_active_test.return_value = mock_test
    manager.get_agent_tests.return_value = [mock_test]
    manager.start_test.return_value = mock_test
    manager.record_result.return_value = mock_test
    manager.cancel_test.return_value = True

    result = Mock()
    result.test_id = "test-123"
    result.winner = "evolved"
    result.confidence = 0.7
    result.recommendation = "Adopt evolved prompt"
    result.stats = {"evolved_win_rate": 0.58}
    manager.conclude_test.return_value = result

    return manager


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before and after each test."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def enabled_handler(handler, mock_manager):
    """Handler with A/B testing enabled and mocked manager."""
    original = ab_mod.AB_TESTING_AVAILABLE
    ab_mod.AB_TESTING_AVAILABLE = True
    handler._manager = mock_manager
    yield handler
    ab_mod.AB_TESTING_AVAILABLE = original


# ============================================================================
# Route Recognition Tests
# ============================================================================


class TestABTestingRouting:
    """Tests for A/B testing route recognition."""

    def test_routes_defined(self, handler):
        """Test handler has routes defined."""
        assert "/api/evolution/ab-tests" in handler.ROUTES

    def test_can_handle_base_route(self, handler):
        """Test can_handle for base route."""
        assert handler.can_handle("/api/evolution/ab-tests") is True

    def test_can_handle_test_route(self, handler):
        """Test can_handle for test ID route."""
        assert handler.can_handle("/api/evolution/ab-tests/test-123") is True

    def test_can_handle_active_route(self, handler):
        """Test can_handle for active test route."""
        assert handler.can_handle("/api/evolution/ab-tests/claude/active") is True

    def test_cannot_handle_unrelated_routes(self, handler):
        """Test can_handle returns False for unrelated routes."""
        assert handler.can_handle("/api/evolution/history") is False
        assert handler.can_handle("/api/debates") is False


# ============================================================================
# GET /api/evolution/ab-tests Tests
# ============================================================================


class TestListTests:
    """Tests for listing A/B tests."""

    def test_list_tests_unavailable(self, handler):
        """Test 503 when module not available."""
        original = ab_mod.AB_TESTING_AVAILABLE
        ab_mod.AB_TESTING_AVAILABLE = False
        try:
            result = handler.handle("/api/evolution/ab-tests", {}, None)
            assert result.status_code == 503
        finally:
            ab_mod.AB_TESTING_AVAILABLE = original

    def test_list_tests_success(self, enabled_handler, mock_test):
        """Test successful test listing."""
        with patch.object(enabled_handler, "_get_all_tests", return_value=[mock_test]):
            result = enabled_handler.handle("/api/evolution/ab-tests", {}, None)

            assert result.status_code == 200
            data = json.loads(result.body)
            assert "tests" in data
            assert data["count"] >= 0

    def test_list_tests_with_agent_filter(self, enabled_handler, mock_manager):
        """Test listing with agent filter."""
        result = enabled_handler.handle("/api/evolution/ab-tests", {"agent": "claude"}, None)

        assert result.status_code == 200
        mock_manager.get_agent_tests.assert_called()


# ============================================================================
# GET /api/evolution/ab-tests/{id} Tests
# ============================================================================


class TestGetTest:
    """Tests for getting specific test."""

    def test_get_test_success(self, enabled_handler):
        """Test successful test retrieval."""
        result = enabled_handler.handle("/api/evolution/ab-tests/test-123", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["id"] == "test-123"

    def test_get_test_not_found(self, enabled_handler, mock_manager):
        """Test 404 for non-existent test."""
        mock_manager.get_test.return_value = None

        result = enabled_handler.handle("/api/evolution/ab-tests/nonexistent", {}, None)

        assert result.status_code == 404


# ============================================================================
# GET /api/evolution/ab-tests/{agent}/active Tests
# ============================================================================


class TestGetActiveTest:
    """Tests for getting active test for agent."""

    def test_get_active_test_success(self, enabled_handler):
        """Test successful active test retrieval."""
        result = enabled_handler.handle("/api/evolution/ab-tests/claude/active", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["agent"] == "claude"
        assert data["has_active_test"] is True

    def test_get_active_test_none(self, enabled_handler, mock_manager):
        """Test when no active test exists."""
        mock_manager.get_active_test.return_value = None

        result = enabled_handler.handle("/api/evolution/ab-tests/claude/active", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["has_active_test"] is False


# ============================================================================
# POST /api/evolution/ab-tests Tests
# ============================================================================


class TestCreateTest:
    """Tests for creating new tests."""

    def test_create_test_success(self, enabled_handler):
        """Test successful test creation."""
        result = enabled_handler.handle_post(
            "/api/evolution/ab-tests",
            {
                "agent": "claude",
                "baseline_version": 1,
                "evolved_version": 2,
            },
            None,
        )

        assert result.status_code == 201
        data = json.loads(result.body)
        assert "test" in data

    def test_create_test_missing_agent(self, enabled_handler):
        """Test 400 when agent is missing."""
        result = enabled_handler.handle_post(
            "/api/evolution/ab-tests",
            {"baseline_version": 1, "evolved_version": 2},
            None,
        )

        assert result.status_code == 400

    def test_create_test_missing_versions(self, enabled_handler):
        """Test 400 when versions are missing."""
        result = enabled_handler.handle_post(
            "/api/evolution/ab-tests",
            {"agent": "claude"},
            None,
        )

        assert result.status_code == 400

    def test_create_test_conflict(self, enabled_handler, mock_manager):
        """Test 409 when agent already has active test."""
        mock_manager.start_test.side_effect = ValueError("Already has active test")

        result = enabled_handler.handle_post(
            "/api/evolution/ab-tests",
            {
                "agent": "claude",
                "baseline_version": 1,
                "evolved_version": 2,
            },
            None,
        )

        assert result.status_code == 409

    def test_create_test_invalid_version_format_returns_400(self, enabled_handler):
        """Test 400 when version is not a valid integer."""
        result = enabled_handler.handle_post(
            "/api/evolution/ab-tests",
            {
                "agent": "claude",
                "baseline_version": "not-a-number",
                "evolved_version": 2,
            },
            None,
        )

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "integer" in data.get("error", "").lower()

    def test_create_test_float_version_returns_400(self, enabled_handler):
        """Test 400 when version is a float string."""
        result = enabled_handler.handle_post(
            "/api/evolution/ab-tests",
            {
                "agent": "claude",
                "baseline_version": "1.5",
                "evolved_version": 2,
            },
            None,
        )

        assert result.status_code == 400


# ============================================================================
# POST /api/evolution/ab-tests/{id}/record Tests
# ============================================================================


class TestRecordResult:
    """Tests for recording debate results."""

    def test_record_result_success(self, enabled_handler):
        """Test successful result recording."""
        result = enabled_handler.handle_post(
            "/api/evolution/ab-tests/test-123/record",
            {
                "debate_id": "debate-456",
                "variant": "evolved",
                "won": True,
            },
            None,
        )

        assert result.status_code == 200

    def test_record_result_missing_fields(self, enabled_handler):
        """Test 400 for missing required fields."""
        # Missing debate_id
        result = enabled_handler.handle_post(
            "/api/evolution/ab-tests/test-123/record",
            {"variant": "evolved", "won": True},
            None,
        )
        assert result.status_code == 400

    def test_record_result_invalid_variant(self, enabled_handler):
        """Test 400 for invalid variant."""
        result = enabled_handler.handle_post(
            "/api/evolution/ab-tests/test-123/record",
            {
                "debate_id": "debate-456",
                "variant": "invalid",
                "won": True,
            },
            None,
        )

        assert result.status_code == 400


# ============================================================================
# POST /api/evolution/ab-tests/{id}/conclude Tests
# ============================================================================


class TestConcludeTest:
    """Tests for concluding tests."""

    def test_conclude_test_success(self, enabled_handler):
        """Test successful test conclusion."""
        result = enabled_handler.handle_post(
            "/api/evolution/ab-tests/test-123/conclude",
            {},
            None,
        )

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "result" in data
        assert data["result"]["winner"] == "evolved"

    def test_conclude_test_with_force(self, enabled_handler, mock_manager):
        """Test conclusion with force flag."""
        result = enabled_handler.handle_post(
            "/api/evolution/ab-tests/test-123/conclude",
            {"force": True},
            None,
        )

        assert result.status_code == 200
        mock_manager.conclude_test.assert_called_with("test-123", force=True)


# ============================================================================
# DELETE /api/evolution/ab-tests/{id} Tests
# ============================================================================


class TestCancelTest:
    """Tests for cancelling tests."""

    def _make_auth_handler(self, authenticated=True, client_ip="127.0.0.1"):
        """Create mock request handler with auth headers."""
        mock_handler = Mock()
        mock_handler.headers = {"Authorization": "Bearer test_token"} if authenticated else {}
        mock_handler.client_address = (client_ip, 12345)
        return mock_handler

    def test_cancel_test_success(self, enabled_handler):
        """Test successful test cancellation."""
        mock_handler = self._make_auth_handler()

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_extract.return_value = mock_auth_ctx
            result = enabled_handler.handle_delete("/api/evolution/ab-tests/test-123", mock_handler)

        assert result.status_code == 200

    def test_cancel_test_not_found(self, enabled_handler, mock_manager):
        """Test 404 when test not found or already concluded."""
        mock_manager.cancel_test.return_value = False
        mock_handler = self._make_auth_handler()

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_extract.return_value = mock_auth_ctx
            result = enabled_handler.handle_delete("/api/evolution/ab-tests/test-123", mock_handler)

        assert result.status_code == 404

    def test_cancel_test_requires_auth(self, enabled_handler):
        """Test 401 when not authenticated."""
        mock_handler = self._make_auth_handler(authenticated=False)

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = False
            mock_extract.return_value = mock_auth_ctx
            result = enabled_handler.handle_delete("/api/evolution/ab-tests/test-123", mock_handler)

        assert result is not None
        assert result.status_code == 401
        data = json.loads(result.body)
        assert "authentication" in data["error"].lower()

    def test_cancel_test_rate_limited(self, enabled_handler, mock_manager):
        """Test 429 when rate limit exceeded."""
        mock_manager.cancel_test.return_value = True

        # Make more than 10 requests from same IP
        rate_limited_count = 0
        for i in range(15):
            mock_handler = self._make_auth_handler(client_ip="192.168.1.200")

            with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
                mock_auth_ctx = Mock()
                mock_auth_ctx.is_authenticated = True
                mock_extract.return_value = mock_auth_ctx
                result = enabled_handler.handle_delete(
                    f"/api/evolution/ab-tests/test-{i}", mock_handler
                )

            # After 10 requests, should get rate limited
            if result is not None and result.status_code == 429:
                rate_limited_count += 1

        # At least some requests should be rate limited
        assert rate_limited_count > 0

    def test_cancel_test_invalid_id(self, enabled_handler):
        """Test handling of invalid test ID format."""
        mock_handler = self._make_auth_handler()

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_extract.return_value = mock_auth_ctx
            # Path traversal attempt
            result = enabled_handler.handle_delete(
                "/api/evolution/ab-tests/../../../etc/passwd", mock_handler
            )

        # Should either return None (not matched) or 400/404
        assert result is None or result.status_code in [400, 404]

    def test_cancel_test_returns_none_for_wrong_path(self, enabled_handler):
        """Test that unrelated paths return None."""
        mock_handler = self._make_auth_handler()

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_extract.return_value = mock_auth_ctx
            # Use a completely different path that's not under /api/evolution/ab-tests
            result = enabled_handler.handle_delete("/api/debates/test-123", mock_handler)

        assert result is None

    def test_cancel_test_handles_exception(self, enabled_handler, mock_manager):
        """Test 500 when cancel raises exception."""
        mock_manager.cancel_test.side_effect = Exception("Database error")
        mock_handler = self._make_auth_handler()

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_auth_ctx = Mock()
            mock_auth_ctx.is_authenticated = True
            mock_extract.return_value = mock_auth_ctx
            result = enabled_handler.handle_delete("/api/evolution/ab-tests/test-123", mock_handler)

        assert result is not None
        assert result.status_code == 500
