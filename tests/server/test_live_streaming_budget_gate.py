"""
Tests for the LiveStreamingBudgetGate and budget-gated auth checks.

Covers:
- LiveStreamingBudgetGate: usage counting, tier limits, budget exhaustion
- _check_live_streaming_budget: auth + budget pipeline integration
- BUDGET_GATED_PATHS: correct path membership
- Graceful degradation when billing modules are unavailable
"""

from __future__ import annotations

import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.auth_checks import (
    BUDGET_GATED_PATHS,
    AuthChecksMixin,
    LiveStreamingBudgetGate,
)


# ---------------------------------------------------------------------------
# LiveStreamingBudgetGate unit tests
# ---------------------------------------------------------------------------


class TestLiveStreamingBudgetGate:
    """Unit tests for the SQLite-backed budget gate."""

    @pytest.fixture(autouse=True)
    def gate(self, tmp_path):
        """Create a gate with a temporary database."""
        db = str(tmp_path / "test_budget.db")
        self._gate = LiveStreamingBudgetGate(db_path=db)
        yield self._gate

    def test_initial_usage_is_zero(self):
        assert self._gate.get_usage("org_1") == 0

    def test_increment_increases_count(self):
        assert self._gate.increment("org_1") == 1
        assert self._gate.increment("org_1") == 2
        assert self._gate.get_usage("org_1") == 2

    def test_separate_orgs_tracked_independently(self):
        self._gate.increment("org_a")
        self._gate.increment("org_a")
        self._gate.increment("org_b")
        assert self._gate.get_usage("org_a") == 2
        assert self._gate.get_usage("org_b") == 1

    def test_check_budget_allows_under_limit(self):
        allowed, error = self._gate.check_budget("org_1", "free")
        assert allowed is True
        assert error is None

    def test_check_budget_blocks_at_limit(self):
        # Free tier = 10 debates/month
        for _ in range(10):
            self._gate.increment("org_1")

        allowed, error = self._gate.check_budget("org_1", "free")
        assert allowed is False
        assert error is not None
        assert error["code"] == "live_budget_exceeded"
        assert error["limit"] == 10
        assert error["current_usage"] == 10
        assert "/pricing" in error["upgrade_url"]

    def test_check_budget_uses_tier_limit(self):
        # Starter = 50
        for _ in range(10):
            self._gate.increment("org_1")

        allowed, _ = self._gate.check_budget("org_1", "starter")
        assert allowed is True  # 10 < 50

    def test_check_budget_enterprise_virtually_unlimited(self):
        for _ in range(100):
            self._gate.increment("org_1")

        allowed, _ = self._gate.check_budget("org_1", "enterprise")
        assert allowed is True

    def test_record_usage_returns_new_count(self):
        count = self._gate.record_usage("org_1")
        assert count == 1
        count = self._gate.record_usage("org_1")
        assert count == 2

    def test_get_tier_limit_fallback_on_import_error(self):
        """When billing models are unavailable, use hardcoded defaults."""
        with patch.dict("sys.modules", {"aragora.billing.models": None}):
            # Force re-evaluation by calling the static method
            limit = LiveStreamingBudgetGate.get_tier_limit("free")
            assert limit == 10

    def test_get_tier_limit_unknown_tier_defaults_to_free(self):
        limit = LiveStreamingBudgetGate.get_tier_limit("nonexistent_tier")
        assert limit == 10  # falls back to free tier default


class TestLiveStreamingBudgetGateSingleton:
    """Tests for the singleton accessor."""

    def setup_method(self):
        LiveStreamingBudgetGate.reset_instance()

    def teardown_method(self):
        LiveStreamingBudgetGate.reset_instance()

    def test_get_instance_returns_same_object(self):
        a = LiveStreamingBudgetGate.get_instance()
        b = LiveStreamingBudgetGate.get_instance()
        assert a is b

    def test_reset_clears_instance(self):
        a = LiveStreamingBudgetGate.get_instance()
        LiveStreamingBudgetGate.reset_instance()
        b = LiveStreamingBudgetGate.get_instance()
        assert a is not b


# ---------------------------------------------------------------------------
# BUDGET_GATED_PATHS membership
# ---------------------------------------------------------------------------


class TestBudgetGatedPaths:
    """Ensure the correct paths are budget-gated."""

    def test_live_debate_is_gated(self):
        assert "/api/v1/playground/debate/live" in BUDGET_GATED_PATHS

    def test_cost_estimate_is_gated(self):
        assert "/api/v1/playground/debate/live/cost-estimate" in BUDGET_GATED_PATHS

    def test_tts_is_gated(self):
        assert "/api/v1/playground/tts" in BUDGET_GATED_PATHS

    def test_mock_debate_is_not_gated(self):
        assert "/api/v1/playground/debate" not in BUDGET_GATED_PATHS

    def test_playground_status_is_not_gated(self):
        assert "/api/v1/playground/status" not in BUDGET_GATED_PATHS


# ---------------------------------------------------------------------------
# _check_live_streaming_budget integration tests
# ---------------------------------------------------------------------------


class _FakeHandler(AuthChecksMixin):
    """Minimal handler stub for testing the mixin method."""

    def __init__(self, path: str = "/api/v1/playground/debate/live"):
        self.path = path
        self.command = "POST"
        self.headers = {}
        self.user_store = None
        self.rbac = None
        self._rate_limit_result = None
        self._responses: list[tuple[dict, int]] = []

    def _send_json(self, data, status=200):
        self._responses.append((data, status))


class TestCheckLiveStreamingBudget:
    """Integration tests for _check_live_streaming_budget on the mixin."""

    def setup_method(self):
        LiveStreamingBudgetGate.reset_instance()

    def teardown_method(self):
        LiveStreamingBudgetGate.reset_instance()

    def test_non_gated_path_passes_through(self):
        handler = _FakeHandler(path="/api/v1/playground/debate")
        assert handler._check_live_streaming_budget() is True
        assert len(handler._responses) == 0

    @patch("aragora.server.auth_checks.auth_config", new_callable=lambda: type("AC", (), {"enabled": False}))
    def test_auth_disabled_allows_access(self, _mock_config):
        """When auth is disabled (dev/demo), live streaming is allowed."""
        handler = _FakeHandler()
        with patch("aragora.server.auth_checks.auth_config", MagicMock(enabled=False)):
            assert handler._check_live_streaming_budget() is True

    def test_unauthenticated_user_blocked(self):
        """Unauthenticated users get 401."""
        handler = _FakeHandler()
        mock_ctx = MagicMock(authenticated=False, user_id=None)
        with (
            patch("aragora.server.auth_checks.auth_config", MagicMock(enabled=True)),
            patch(
                "aragora.server.auth_checks.extract_user_from_request",
                return_value=mock_ctx,
            ),
        ):
            result = handler._check_live_streaming_budget()
            assert result is False
            assert len(handler._responses) == 1
            assert handler._responses[0][1] == 401
            assert "auth_required" in handler._responses[0][0]["code"]

    def test_authenticated_user_within_budget_allowed(self, tmp_path):
        """Authenticated user with remaining budget is allowed."""
        db = str(tmp_path / "budget.db")
        gate = LiveStreamingBudgetGate(db_path=db)
        LiveStreamingBudgetGate._instance = gate

        handler = _FakeHandler()
        mock_ctx = MagicMock(
            authenticated=True, user_id="user_1", org_id="org_1", role="member"
        )
        with (
            patch("aragora.server.auth_checks.auth_config", MagicMock(enabled=True)),
            patch(
                "aragora.server.auth_checks.extract_user_from_request",
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.auth_checks._resolve_org_tier",
                return_value="free",
            ),
        ):
            result = handler._check_live_streaming_budget()
            assert result is True
            # Usage should be recorded (path is /api/v1/playground/debate/live)
            assert gate.get_usage("org_1") == 1

    def test_authenticated_user_over_budget_blocked(self, tmp_path):
        """Authenticated user who exhausted budget gets 429."""
        db = str(tmp_path / "budget.db")
        gate = LiveStreamingBudgetGate(db_path=db)
        LiveStreamingBudgetGate._instance = gate

        # Exhaust free tier limit
        for _ in range(10):
            gate.increment("org_1")

        handler = _FakeHandler()
        mock_ctx = MagicMock(
            authenticated=True, user_id="user_1", org_id="org_1", role="member"
        )
        with (
            patch("aragora.server.auth_checks.auth_config", MagicMock(enabled=True)),
            patch(
                "aragora.server.auth_checks.extract_user_from_request",
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.auth_checks._resolve_org_tier",
                return_value="free",
            ),
        ):
            result = handler._check_live_streaming_budget()
            assert result is False
            assert handler._responses[0][1] == 429
            assert handler._responses[0][0]["code"] == "live_budget_exceeded"

    def test_cost_estimate_does_not_increment_usage(self, tmp_path):
        """Cost estimate requests check budget but don't record usage."""
        db = str(tmp_path / "budget.db")
        gate = LiveStreamingBudgetGate(db_path=db)
        LiveStreamingBudgetGate._instance = gate

        handler = _FakeHandler(path="/api/v1/playground/debate/live/cost-estimate")
        mock_ctx = MagicMock(
            authenticated=True, user_id="user_1", org_id="org_1", role="member"
        )
        with (
            patch("aragora.server.auth_checks.auth_config", MagicMock(enabled=True)),
            patch(
                "aragora.server.auth_checks.extract_user_from_request",
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.auth_checks._resolve_org_tier",
                return_value="free",
            ),
        ):
            result = handler._check_live_streaming_budget()
            assert result is True
            # Usage should NOT be incremented for cost estimates
            assert gate.get_usage("org_1") == 0

    def test_tts_does_not_increment_usage(self, tmp_path):
        """TTS requests check budget but don't increment the debate counter."""
        db = str(tmp_path / "budget.db")
        gate = LiveStreamingBudgetGate(db_path=db)
        LiveStreamingBudgetGate._instance = gate

        handler = _FakeHandler(path="/api/v1/playground/tts")
        mock_ctx = MagicMock(
            authenticated=True, user_id="user_1", org_id="org_1", role="member"
        )
        with (
            patch("aragora.server.auth_checks.auth_config", MagicMock(enabled=True)),
            patch(
                "aragora.server.auth_checks.extract_user_from_request",
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.auth_checks._resolve_org_tier",
                return_value="free",
            ),
        ):
            result = handler._check_live_streaming_budget()
            assert result is True
            assert gate.get_usage("org_1") == 0

    def test_paid_tier_has_higher_limit(self, tmp_path):
        """Professional tier users have a higher budget."""
        db = str(tmp_path / "budget.db")
        gate = LiveStreamingBudgetGate(db_path=db)
        LiveStreamingBudgetGate._instance = gate

        # Use 15 debates (exceeds free=10, but under professional=200)
        for _ in range(15):
            gate.increment("org_1")

        handler = _FakeHandler()
        mock_ctx = MagicMock(
            authenticated=True, user_id="user_1", org_id="org_1", role="member"
        )
        with (
            patch("aragora.server.auth_checks.auth_config", MagicMock(enabled=True)),
            patch(
                "aragora.server.auth_checks.extract_user_from_request",
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.auth_checks._resolve_org_tier",
                return_value="professional",
            ),
        ):
            result = handler._check_live_streaming_budget()
            assert result is True

    def test_fallback_to_user_id_when_no_org(self, tmp_path):
        """When org_id is None, falls back to user_id as workspace key."""
        db = str(tmp_path / "budget.db")
        gate = LiveStreamingBudgetGate(db_path=db)
        LiveStreamingBudgetGate._instance = gate

        handler = _FakeHandler()
        mock_ctx = MagicMock(
            authenticated=True, user_id="user_solo", org_id=None, role="member"
        )
        with (
            patch("aragora.server.auth_checks.auth_config", MagicMock(enabled=True)),
            patch(
                "aragora.server.auth_checks.extract_user_from_request",
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.auth_checks._resolve_org_tier",
                return_value=None,
            ),
        ):
            result = handler._check_live_streaming_budget()
            assert result is True
            # Usage recorded against user_id as workspace fallback
            assert gate.get_usage("user_solo") == 1


# ---------------------------------------------------------------------------
# Auth exemption regression tests
# ---------------------------------------------------------------------------


class TestLivePathsNotExempt:
    """Ensure live/TTS paths are NOT in AUTH_EXEMPT_PATHS anymore."""

    @pytest.fixture
    def exempt_paths(self):
        from aragora.server.auth_checks import AuthChecksMixin

        return AuthChecksMixin.AUTH_EXEMPT_PATHS

    def test_live_debate_not_exempt(self, exempt_paths):
        assert "/api/v1/playground/debate/live" not in exempt_paths

    def test_cost_estimate_not_exempt(self, exempt_paths):
        assert "/api/v1/playground/debate/live/cost-estimate" not in exempt_paths

    def test_tts_not_exempt(self, exempt_paths):
        assert "/api/v1/playground/tts" not in exempt_paths

    def test_mock_debate_still_exempt(self, exempt_paths):
        """The mock debate endpoint should remain publicly accessible."""
        assert "/api/v1/playground/debate" in exempt_paths

    def test_playground_status_still_exempt(self, exempt_paths):
        assert "/api/v1/playground/status" in exempt_paths
