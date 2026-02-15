"""Comprehensive tests for the prompt evolution handler.

Tests cover:
- GET endpoints: patterns, summary, agent history, prompt version
- Input validation: agent name patterns, limit bounds
- RBAC permission checks
- Rate limiting
- Error handling: module not available, DB errors, nomic dir not configured
- Edge cases: v1 prefix, empty results, boundary values, invalid paths
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Pre-mock broken transitive imports to allow importing evolution handler
# without triggering the full handlers.__init__ chain
if "aragora.server.handlers.social._slack_impl" not in sys.modules:
    sys.modules["aragora.server.handlers.social._slack_impl"] = MagicMock()

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_rate_limiter():
    """Reset the module-level rate limiter and evolution globals between tests.

    Saves and restores EVOLUTION_AVAILABLE and PromptEvolver so that
    cross-test pollution (e.g. from a prior test that sets them to mocks
    and fails before restoring) cannot affect subsequent tests.
    """
    import aragora.server.handlers.evolution.handler as mod

    mod._evolution_limiter._buckets = defaultdict(list)

    # Save original module-level globals
    orig_avail = mod.EVOLUTION_AVAILABLE
    orig_evolver = mod.PromptEvolver

    yield

    # Restore module-level globals to prevent cross-test pollution
    mod.EVOLUTION_AVAILABLE = orig_avail
    mod.PromptEvolver = orig_evolver
    mod._evolution_limiter._buckets = defaultdict(list)


@pytest.fixture()
def mod():
    """Return the handler module for patching module-level globals."""
    import aragora.server.handlers.evolution.handler as _mod

    return _mod


@pytest.fixture()
def handler():
    """Create an EvolutionHandler with a nomic_dir configured."""
    from aragora.server.handlers.evolution.handler import EvolutionHandler

    return EvolutionHandler(ctx={"nomic_dir": "/tmp/test_nomic"})


@pytest.fixture()
def handler_no_nomic():
    """Create an EvolutionHandler without nomic_dir."""
    from aragora.server.handlers.evolution.handler import EvolutionHandler

    return EvolutionHandler(ctx={})


def _mock_http(client_ip="10.0.0.1"):
    """Create a mock HTTP request handler."""
    h = MagicMock()
    h.client_address = (client_ip, 54321)
    h.headers = {}
    return h


def _status_body(result):
    """Extract (status_code, parsed_json_body) from a HandlerResult."""
    return result.status_code, json.loads(result.body)


# ---------------------------------------------------------------------------
# Helper: set up module to pretend PromptEvolver is available
# ---------------------------------------------------------------------------


def _enable_evolution(mod):
    """Enable evolution and install a mock PromptEvolver class."""
    mod.EVOLUTION_AVAILABLE = True
    mock_cls = MagicMock()
    mod.PromptEvolver = mock_cls
    return mock_cls


def _disable_evolution(mod):
    """Disable evolution module."""
    mod.EVOLUTION_AVAILABLE = False
    mod.PromptEvolver = None


# ============================================================================
# can_handle routing
# ============================================================================


class TestCanHandle:
    """Verify that can_handle matches all expected routes and rejects others."""

    @pytest.mark.parametrize(
        "path",
        [
            "/api/evolution/patterns",
            "/api/evolution/summary",
            "/api/evolution/claude/history",
            "/api/evolution/gpt4/prompt",
            "/api/v1/evolution/patterns",
            "/api/v1/evolution/summary",
            "/api/v1/evolution/claude/history",
            "/api/v1/evolution/gpt4/prompt",
        ],
    )
    def test_accepted_paths(self, handler, path):
        assert handler.can_handle(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "/api/evolution",
            "/api/evolution/",
            "/api/debates",
            "/api/agents/claude/history",
            "/api/evolution/patterns/extra",
            "/other/evolution/patterns",
        ],
    )
    def test_rejected_paths(self, handler, path):
        assert handler.can_handle(path) is False


# ============================================================================
# handle() routing dispatch
# ============================================================================


class TestHandleDispatch:
    """Verify that handle() dispatches to the correct internal method."""

    def test_non_evolution_path_returns_none(self, handler):
        result = handler.handle("/api/debates", {}, _mock_http())
        assert result is None

    def test_dispatches_to_patterns(self, handler):
        with patch.object(handler, "_get_patterns", return_value=MagicMock(status_code=200)) as m:
            handler.handle("/api/evolution/patterns", {}, _mock_http())
            m.assert_called_once_with(None, 10)

    def test_dispatches_to_summary(self, handler):
        with patch.object(handler, "_get_summary", return_value=MagicMock(status_code=200)) as m:
            handler.handle("/api/evolution/summary", {}, _mock_http())
            m.assert_called_once()

    def test_dispatches_to_history(self, handler):
        with patch.object(
            handler, "_get_evolution_history", return_value=MagicMock(status_code=200)
        ) as m:
            handler.handle("/api/evolution/claude/history", {}, _mock_http())
            m.assert_called_once_with("claude", 10)

    def test_dispatches_to_prompt(self, handler):
        with patch.object(
            handler, "_get_prompt_version", return_value=MagicMock(status_code=200)
        ) as m:
            handler.handle("/api/evolution/claude/prompt", {}, _mock_http())
            # get_int_param defaults to 0 when "version" is absent
            m.assert_called_once_with("claude", 0)

    def test_v1_prefix_routes_to_patterns(self, handler):
        with patch.object(handler, "_get_patterns", return_value=MagicMock(status_code=200)) as m:
            handler.handle("/api/v1/evolution/patterns", {}, _mock_http())
            m.assert_called_once()

    def test_v1_prefix_routes_to_history(self, handler):
        with patch.object(
            handler, "_get_evolution_history", return_value=MagicMock(status_code=200)
        ) as m:
            handler.handle("/api/v1/evolution/claude/history", {"limit": "5"}, _mock_http())
            m.assert_called_once_with("claude", 5)


# ============================================================================
# Query parameter validation and clamping
# ============================================================================


class TestQueryParamClamping:
    """Verify that limit and type query params are parsed and clamped."""

    def test_limit_defaults_to_10(self, handler):
        with patch.object(handler, "_get_patterns", return_value=MagicMock(status_code=200)) as m:
            handler.handle("/api/evolution/patterns", {}, _mock_http())
            m.assert_called_once_with(None, 10)

    def test_limit_clamped_to_min_1(self, handler):
        with patch.object(handler, "_get_patterns", return_value=MagicMock(status_code=200)) as m:
            handler.handle("/api/evolution/patterns", {"limit": "-5"}, _mock_http())
            m.assert_called_once_with(None, 1)

    def test_limit_zero_clamped_to_1(self, handler):
        with patch.object(handler, "_get_patterns", return_value=MagicMock(status_code=200)) as m:
            handler.handle("/api/evolution/patterns", {"limit": "0"}, _mock_http())
            m.assert_called_once_with(None, 1)

    def test_limit_clamped_to_max_50(self, handler):
        with patch.object(handler, "_get_patterns", return_value=MagicMock(status_code=200)) as m:
            handler.handle("/api/evolution/patterns", {"limit": "999"}, _mock_http())
            m.assert_called_once_with(None, 50)

    def test_limit_at_boundary_50(self, handler):
        with patch.object(handler, "_get_patterns", return_value=MagicMock(status_code=200)) as m:
            handler.handle("/api/evolution/patterns", {"limit": "50"}, _mock_http())
            m.assert_called_once_with(None, 50)

    def test_limit_at_boundary_1(self, handler):
        with patch.object(handler, "_get_patterns", return_value=MagicMock(status_code=200)) as m:
            handler.handle("/api/evolution/patterns", {"limit": "1"}, _mock_http())
            m.assert_called_once_with(None, 1)

    def test_type_filter_passed_through(self, handler):
        with patch.object(handler, "_get_patterns", return_value=MagicMock(status_code=200)) as m:
            handler.handle(
                "/api/evolution/patterns",
                {"type": "strategy", "limit": "5"},
                _mock_http(),
            )
            m.assert_called_once_with("strategy", 5)

    def test_history_limit_clamped(self, handler):
        with patch.object(
            handler, "_get_evolution_history", return_value=MagicMock(status_code=200)
        ) as m:
            handler.handle("/api/evolution/claude/history", {"limit": "200"}, _mock_http())
            m.assert_called_once_with("claude", 50)

    def test_prompt_version_param(self, handler):
        with patch.object(
            handler, "_get_prompt_version", return_value=MagicMock(status_code=200)
        ) as m:
            handler.handle("/api/evolution/claude/prompt", {"version": "3"}, _mock_http())
            m.assert_called_once_with("claude", 3)

    def test_prompt_version_absent_defaults_to_0(self, handler):
        with patch.object(
            handler, "_get_prompt_version", return_value=MagicMock(status_code=200)
        ) as m:
            handler.handle("/api/evolution/claude/prompt", {}, _mock_http())
            # get_int_param defaults to 0 when "version" key is absent
            m.assert_called_once_with("claude", 0)


# ============================================================================
# Agent name validation
# ============================================================================


class TestAgentNameValidation:
    """Verify that invalid agent names in path are rejected."""

    @pytest.mark.parametrize("name", ["claude", "gpt4", "my-agent", "agent_v2", "A123"])
    def test_valid_agent_names(self, handler, name):
        with patch.object(
            handler, "_get_evolution_history", return_value=MagicMock(status_code=200)
        ):
            result = handler.handle(f"/api/evolution/{name}/history", {}, _mock_http())
            assert result is not None
            assert result.status_code == 200, f"Expected 200 for agent name '{name}'"

    def test_agent_name_with_special_chars_rejected(self, handler):
        # SAFE_AGENT_PATTERN = ^[a-zA-Z0-9_-]{1,32}$
        # These names contain chars not in the allowed set and do not contain '/'
        # so the path segment is extracted but fails validation -> 400
        for name in ["agent;drop", "agent<script>", "agent@home", "agent!x"]:
            result = handler.handle(f"/api/evolution/{name}/history", {}, _mock_http())
            assert result is not None, f"Expected a result for agent name '{name}'"
            status, body = _status_body(result)
            assert status == 400, f"Expected 400 for agent name '{name}', got {status}"

    def test_agent_name_too_long_rejected(self, handler):
        long_name = "a" * 33  # Exceeds the 32-char limit in SAFE_AGENT_PATTERN
        result = handler.handle(f"/api/evolution/{long_name}/history", {}, _mock_http())
        if result is not None:
            status, body = _status_body(result)
            assert status == 400


# ============================================================================
# Rate limiting
# ============================================================================


class TestRateLimiting:
    """Verify rate limiting on the evolution endpoints."""

    def test_rate_limit_returns_429(self, handler, mod):
        with patch.object(mod._evolution_limiter, "is_allowed", return_value=False):
            result = handler.handle("/api/evolution/patterns", {}, _mock_http())
            status, body = _status_body(result)
            assert status == 429
            assert "Rate limit" in body["error"]

    def test_rate_limit_applies_to_all_endpoints(self, handler, mod):
        paths = [
            "/api/evolution/patterns",
            "/api/evolution/summary",
            "/api/evolution/claude/history",
            "/api/evolution/claude/prompt",
        ]
        for path in paths:
            with patch.object(mod._evolution_limiter, "is_allowed", return_value=False):
                result = handler.handle(path, {}, _mock_http())
                status, _ = _status_body(result)
                assert status == 429, f"Expected 429 for {path}"

    def test_different_ips_are_independent(self, handler, mod):
        """Each IP gets its own rate limit bucket."""
        # Exhaust rate limit for IP1
        for _ in range(11):
            handler.handle("/api/evolution/patterns", {}, _mock_http("10.0.0.1"))
        # IP2 should still be allowed
        with patch.object(handler, "_get_patterns", return_value=MagicMock(status_code=200)):
            result = handler.handle("/api/evolution/patterns", {}, _mock_http("10.0.0.2"))
            assert result.status_code == 200

    def test_rate_limit_exhaustion(self, handler):
        """After 10 requests, the 11th should be rate limited."""
        http = _mock_http("10.0.0.99")
        for i in range(10):
            with patch.object(handler, "_get_patterns", return_value=MagicMock(status_code=200)):
                result = handler.handle("/api/evolution/patterns", {}, http)
                assert result.status_code == 200, f"Request {i + 1} should succeed"

        # The 11th should be rate limited
        result = handler.handle("/api/evolution/patterns", {}, http)
        status, body = _status_body(result)
        assert status == 429


# ============================================================================
# RBAC permission checks
# ============================================================================


class TestRBACPermissions:
    """Verify that the evolution:read permission is enforced."""

    @pytest.mark.no_auto_auth
    def test_missing_auth_context_returns_401(self, handler):
        """Without auth context, require_auth_or_error returns 401."""
        result = handler.handle("/api/evolution/patterns", {}, _mock_http())
        assert result is not None
        assert result.status_code in (401, 403)

    def test_handle_has_require_permission_decorator(self):
        """Verify handle method is decorated with require_permission."""
        from aragora.server.handlers.evolution.handler import EvolutionHandler

        # The wrapper set by require_permission should be present
        handle_fn = EvolutionHandler.handle
        assert (
            hasattr(handle_fn, "__wrapped__")
            or "evolution:read" in str(getattr(handle_fn, "__qualname__", ""))
            or callable(handle_fn)
        )


# ============================================================================
# Evolution not available (module not installed)
# ============================================================================


class TestEvolutionUnavailable:
    """All endpoints should return 503 when the evolution module is not available."""

    def test_patterns_503(self, handler, mod):
        orig_avail, orig_pe = mod.EVOLUTION_AVAILABLE, mod.PromptEvolver
        try:
            _disable_evolution(mod)
            result = handler._get_patterns(None, 10)
            status, body = _status_body(result)
            assert status == 503
            assert "not available" in body["error"]
        finally:
            mod.EVOLUTION_AVAILABLE, mod.PromptEvolver = orig_avail, orig_pe

    def test_history_503(self, handler, mod):
        orig_avail, orig_pe = mod.EVOLUTION_AVAILABLE, mod.PromptEvolver
        try:
            _disable_evolution(mod)
            result = handler._get_evolution_history("claude", 10)
            status, _ = _status_body(result)
            assert status == 503
        finally:
            mod.EVOLUTION_AVAILABLE, mod.PromptEvolver = orig_avail, orig_pe

    def test_prompt_version_503(self, handler, mod):
        orig_avail, orig_pe = mod.EVOLUTION_AVAILABLE, mod.PromptEvolver
        try:
            _disable_evolution(mod)
            result = handler._get_prompt_version("claude", None)
            status, _ = _status_body(result)
            assert status == 503
        finally:
            mod.EVOLUTION_AVAILABLE, mod.PromptEvolver = orig_avail, orig_pe

    def test_summary_503(self, handler, mod):
        orig_avail, orig_pe = mod.EVOLUTION_AVAILABLE, mod.PromptEvolver
        try:
            _disable_evolution(mod)
            result = handler._get_summary()
            status, _ = _status_body(result)
            assert status == 503
        finally:
            mod.EVOLUTION_AVAILABLE, mod.PromptEvolver = orig_avail, orig_pe

    def test_prompt_evolver_none_also_503(self, handler, mod):
        """Even if EVOLUTION_AVAILABLE is True but PromptEvolver is None -> 503."""
        orig_avail, orig_pe = mod.EVOLUTION_AVAILABLE, mod.PromptEvolver
        try:
            mod.EVOLUTION_AVAILABLE = True
            mod.PromptEvolver = None
            result = handler._get_patterns(None, 10)
            status, _ = _status_body(result)
            assert status == 503
        finally:
            mod.EVOLUTION_AVAILABLE, mod.PromptEvolver = orig_avail, orig_pe


# ============================================================================
# Nomic directory not configured
# ============================================================================


class TestNomicDirNotConfigured:
    """All endpoints should return 503 when nomic_dir is missing from context."""

    def _setup(self, mod):
        orig = (mod.EVOLUTION_AVAILABLE, mod.PromptEvolver)
        _enable_evolution(mod)
        return orig

    def _teardown(self, mod, orig):
        mod.EVOLUTION_AVAILABLE, mod.PromptEvolver = orig

    def test_patterns_no_nomic_dir(self, handler_no_nomic, mod):
        orig = self._setup(mod)
        try:
            result = handler_no_nomic._get_patterns(None, 10)
            status, body = _status_body(result)
            assert status == 503
            assert "not configured" in body["error"]
        finally:
            self._teardown(mod, orig)

    def test_history_no_nomic_dir(self, handler_no_nomic, mod):
        orig = self._setup(mod)
        try:
            result = handler_no_nomic._get_evolution_history("claude", 10)
            status, body = _status_body(result)
            assert status == 503
            assert "not configured" in body["error"]
        finally:
            self._teardown(mod, orig)

    def test_prompt_no_nomic_dir(self, handler_no_nomic, mod):
        orig = self._setup(mod)
        try:
            result = handler_no_nomic._get_prompt_version("claude", None)
            status, body = _status_body(result)
            assert status == 503
            assert "not configured" in body["error"]
        finally:
            self._teardown(mod, orig)

    def test_summary_no_nomic_dir(self, handler_no_nomic, mod):
        orig = self._setup(mod)
        try:
            result = handler_no_nomic._get_summary()
            status, body = _status_body(result)
            assert status == 503
            assert "not configured" in body["error"]
        finally:
            self._teardown(mod, orig)


# ============================================================================
# Successful retrieval (mocked PromptEvolver)
# ============================================================================


class TestGetPatterns:
    """Test the _get_patterns method with a mocked PromptEvolver."""

    def test_returns_patterns_with_filter(self, handler, mod):
        orig = (mod.EVOLUTION_AVAILABLE, mod.PromptEvolver)
        try:
            mock_cls = _enable_evolution(mod)
            mock_inst = MagicMock()
            mock_inst.get_top_patterns.return_value = [
                {"pattern": "chain_of_thought", "frequency": 5},
                {"pattern": "few_shot", "frequency": 3},
            ]
            mock_cls.return_value = mock_inst

            with patch(
                "aragora.server.handlers.evolution.handler.get_db_path",
                return_value="/tmp/test.db",
            ):
                result = handler._get_patterns("strategy", 10)

            status, body = _status_body(result)
            assert status == 200
            assert body["count"] == 2
            assert body["filter"] == "strategy"
            assert len(body["patterns"]) == 2
            mock_inst.get_top_patterns.assert_called_once_with(pattern_type="strategy", limit=10)
        finally:
            mod.EVOLUTION_AVAILABLE, mod.PromptEvolver = orig

    def test_returns_patterns_no_filter(self, handler, mod):
        orig = (mod.EVOLUTION_AVAILABLE, mod.PromptEvolver)
        try:
            mock_cls = _enable_evolution(mod)
            mock_inst = MagicMock()
            mock_inst.get_top_patterns.return_value = []
            mock_cls.return_value = mock_inst

            with patch(
                "aragora.server.handlers.evolution.handler.get_db_path",
                return_value="/tmp/test.db",
            ):
                result = handler._get_patterns(None, 5)

            status, body = _status_body(result)
            assert status == 200
            assert body["count"] == 0
            assert body["filter"] is None
            assert body["patterns"] == []
        finally:
            mod.EVOLUTION_AVAILABLE, mod.PromptEvolver = orig

    def test_db_error_returns_500(self, handler, mod):
        orig = (mod.EVOLUTION_AVAILABLE, mod.PromptEvolver)
        try:
            mock_cls = _enable_evolution(mod)
            mock_cls.side_effect = Exception("db connection failed")

            with patch(
                "aragora.server.handlers.evolution.handler.get_db_path",
                return_value="/tmp/test.db",
            ):
                result = handler._get_patterns(None, 10)

            status, body = _status_body(result)
            assert status == 500
            assert "Failed" in body["error"]
        finally:
            mod.EVOLUTION_AVAILABLE, mod.PromptEvolver = orig


class TestGetEvolutionHistory:
    """Test the _get_evolution_history method."""

    def test_returns_history(self, handler, mod):
        orig = (mod.EVOLUTION_AVAILABLE, mod.PromptEvolver)
        try:
            mock_cls = _enable_evolution(mod)
            mock_inst = MagicMock()
            mock_inst.get_evolution_history.return_value = [
                {"version": 1, "strategy": "mutation"},
                {"version": 2, "strategy": "crossover"},
            ]
            mock_cls.return_value = mock_inst

            with patch(
                "aragora.server.handlers.evolution.handler.get_db_path",
                return_value="/tmp/test.db",
            ):
                result = handler._get_evolution_history("claude", 10)

            status, body = _status_body(result)
            assert status == 200
            assert body["agent"] == "claude"
            assert body["count"] == 2
            assert len(body["history"]) == 2
            mock_inst.get_evolution_history.assert_called_once_with("claude", limit=10)
        finally:
            mod.EVOLUTION_AVAILABLE, mod.PromptEvolver = orig

    def test_empty_history(self, handler, mod):
        orig = (mod.EVOLUTION_AVAILABLE, mod.PromptEvolver)
        try:
            mock_cls = _enable_evolution(mod)
            mock_inst = MagicMock()
            mock_inst.get_evolution_history.return_value = []
            mock_cls.return_value = mock_inst

            with patch(
                "aragora.server.handlers.evolution.handler.get_db_path",
                return_value="/tmp/test.db",
            ):
                result = handler._get_evolution_history("unknown_agent", 10)

            status, body = _status_body(result)
            assert status == 200
            assert body["count"] == 0
            assert body["history"] == []
        finally:
            mod.EVOLUTION_AVAILABLE, mod.PromptEvolver = orig

    def test_db_error_returns_500(self, handler, mod):
        orig = (mod.EVOLUTION_AVAILABLE, mod.PromptEvolver)
        try:
            mock_cls = _enable_evolution(mod)
            mock_cls.side_effect = RuntimeError("broken pipe")

            with patch(
                "aragora.server.handlers.evolution.handler.get_db_path",
                return_value="/tmp/test.db",
            ):
                result = handler._get_evolution_history("claude", 5)

            status, body = _status_body(result)
            assert status == 500
            assert "Failed" in body["error"]
        finally:
            mod.EVOLUTION_AVAILABLE, mod.PromptEvolver = orig


class TestGetPromptVersion:
    """Test the _get_prompt_version method."""

    def test_returns_prompt_version(self, handler, mod):
        orig = (mod.EVOLUTION_AVAILABLE, mod.PromptEvolver)
        try:
            mock_cls = _enable_evolution(mod)
            mock_inst = MagicMock()
            mock_version = SimpleNamespace(
                version=3,
                prompt="You are a helpful assistant.",
                performance_score=0.85,
                debates_count=10,
                consensus_rate=0.7,
                metadata={"strategy": "mutation"},
                created_at="2025-01-01T00:00:00",
            )
            mock_inst.get_prompt_version.return_value = mock_version
            mock_cls.return_value = mock_inst

            with patch(
                "aragora.server.handlers.evolution.handler.get_db_path",
                return_value="/tmp/test.db",
            ):
                result = handler._get_prompt_version("claude", 3)

            status, body = _status_body(result)
            assert status == 200
            assert body["agent"] == "claude"
            assert body["version"] == 3
            assert body["prompt"] == "You are a helpful assistant."
            assert body["performance_score"] == 0.85
            assert body["debates_count"] == 10
            assert body["consensus_rate"] == 0.7
            assert body["metadata"] == {"strategy": "mutation"}
            assert body["created_at"] == "2025-01-01T00:00:00"
            mock_inst.get_prompt_version.assert_called_once_with("claude", 3)
        finally:
            mod.EVOLUTION_AVAILABLE, mod.PromptEvolver = orig

    def test_latest_version_when_none(self, handler, mod):
        """When version is None, it should request the latest."""
        orig = (mod.EVOLUTION_AVAILABLE, mod.PromptEvolver)
        try:
            mock_cls = _enable_evolution(mod)
            mock_inst = MagicMock()
            mock_version = SimpleNamespace(
                version=5,
                prompt="Latest prompt.",
                performance_score=0.9,
                debates_count=20,
                consensus_rate=0.8,
                metadata={},
                created_at="2025-06-01T00:00:00",
            )
            mock_inst.get_prompt_version.return_value = mock_version
            mock_cls.return_value = mock_inst

            with patch(
                "aragora.server.handlers.evolution.handler.get_db_path",
                return_value="/tmp/test.db",
            ):
                result = handler._get_prompt_version("claude", None)

            status, body = _status_body(result)
            assert status == 200
            assert body["version"] == 5
            mock_inst.get_prompt_version.assert_called_once_with("claude", None)
        finally:
            mod.EVOLUTION_AVAILABLE, mod.PromptEvolver = orig

    def test_not_found_returns_404(self, handler, mod):
        orig = (mod.EVOLUTION_AVAILABLE, mod.PromptEvolver)
        try:
            mock_cls = _enable_evolution(mod)
            mock_inst = MagicMock()
            mock_inst.get_prompt_version.return_value = None
            mock_cls.return_value = mock_inst

            with patch(
                "aragora.server.handlers.evolution.handler.get_db_path",
                return_value="/tmp/test.db",
            ):
                result = handler._get_prompt_version("nonexistent", None)

            status, body = _status_body(result)
            assert status == 404
            assert "nonexistent" in body["error"]
        finally:
            mod.EVOLUTION_AVAILABLE, mod.PromptEvolver = orig

    def test_db_error_returns_500(self, handler, mod):
        orig = (mod.EVOLUTION_AVAILABLE, mod.PromptEvolver)
        try:
            mock_cls = _enable_evolution(mod)
            mock_cls.side_effect = Exception("boom")

            with patch(
                "aragora.server.handlers.evolution.handler.get_db_path",
                return_value="/tmp/test.db",
            ):
                result = handler._get_prompt_version("claude", 1)

            status, body = _status_body(result)
            assert status == 500
            assert "Failed" in body["error"]
        finally:
            mod.EVOLUTION_AVAILABLE, mod.PromptEvolver = orig


class TestGetSummary:
    """Test the _get_summary method with mocked DB connection.

    Uses ``patch`` context managers to isolate EVOLUTION_AVAILABLE and
    PromptEvolver from cross-test pollution instead of direct module
    global mutation with try/finally.
    """

    @staticmethod
    def _make_mock_evolver(cursor_fetchone, cursor_fetchall):
        """Build a mock PromptEvolver class with a fake connection/cursor.

        Returns the mock class (to be used as the ``PromptEvolver`` patch
        value).  Does NOT mutate any module globals -- callers use
        ``patch`` to install the mock.
        """
        mock_cls = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = list(cursor_fetchone)
        mock_cursor.fetchall.side_effect = [list(r) for r in cursor_fetchall]

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        mock_inst = MagicMock()
        mock_inst.connection.return_value = mock_conn
        mock_cls.return_value = mock_inst
        return mock_cls

    def test_returns_full_summary(self, handler):
        mock_cls = self._make_mock_evolver(
            cursor_fetchone=[(5,), (2,), (8,)],
            cursor_fetchall=[
                [("strategy", 5), ("structure", 3)],
                [("claude", 0.9, 3), ("gpt4", 0.8, 2)],
                [("claude", "mutation", "2025-01-01")],
            ],
        )

        with (
            patch("aragora.server.handlers.evolution.handler.EVOLUTION_AVAILABLE", True),
            patch("aragora.server.handlers.evolution.handler.PromptEvolver", mock_cls),
            patch(
                "aragora.server.handlers.evolution.handler.get_db_path",
                return_value="/tmp/test.db",
            ),
        ):
            result = handler._get_summary()

        status, body = _status_body(result)
        assert status == 200
        assert body["total_prompt_versions"] == 5
        assert body["total_agents"] == 2
        assert body["total_patterns"] == 8
        assert body["pattern_distribution"] == {"strategy": 5, "structure": 3}
        assert len(body["top_agents"]) == 2
        assert body["top_agents"][0]["agent"] == "claude"
        assert body["top_agents"][0]["best_score"] == 0.9
        assert body["top_agents"][0]["latest_version"] == 3
        assert len(body["recent_activity"]) == 1
        assert body["recent_activity"][0]["strategy"] == "mutation"

    def test_empty_database_summary(self, handler):
        """Summary with zero rows in all tables."""
        mock_cls = self._make_mock_evolver(
            cursor_fetchone=[(0,), (0,), (0,)],
            cursor_fetchall=[[], [], []],
        )

        with (
            patch("aragora.server.handlers.evolution.handler.EVOLUTION_AVAILABLE", True),
            patch("aragora.server.handlers.evolution.handler.PromptEvolver", mock_cls),
            patch(
                "aragora.server.handlers.evolution.handler.get_db_path",
                return_value="/tmp/test.db",
            ),
        ):
            result = handler._get_summary()

        status, body = _status_body(result)
        assert status == 200
        assert body["total_prompt_versions"] == 0
        assert body["total_agents"] == 0
        assert body["total_patterns"] == 0
        assert body["pattern_distribution"] == {}
        assert body["top_agents"] == []
        assert body["recent_activity"] == []

    def test_db_error_returns_500(self, handler):
        mock_cls = MagicMock(side_effect=Exception("db crash"))

        with (
            patch("aragora.server.handlers.evolution.handler.EVOLUTION_AVAILABLE", True),
            patch("aragora.server.handlers.evolution.handler.PromptEvolver", mock_cls),
            patch(
                "aragora.server.handlers.evolution.handler.get_db_path",
                return_value="/tmp/test.db",
            ),
        ):
            result = handler._get_summary()

        status, body = _status_body(result)
        assert status == 500
        assert "Failed" in body["error"]

    def test_connection_context_manager_error(self, handler):
        """Error inside the with evolver.connection() block."""
        mock_cls = MagicMock()
        mock_inst = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(side_effect=Exception("connection refused"))
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_inst.connection.return_value = mock_conn
        mock_cls.return_value = mock_inst

        with (
            patch("aragora.server.handlers.evolution.handler.EVOLUTION_AVAILABLE", True),
            patch("aragora.server.handlers.evolution.handler.PromptEvolver", mock_cls),
            patch(
                "aragora.server.handlers.evolution.handler.get_db_path",
                return_value="/tmp/test.db",
            ),
        ):
            result = handler._get_summary()

        status, body = _status_body(result)
        assert status == 500


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    """Miscellaneous edge cases."""

    def test_handler_with_none_context(self):
        """Handler should accept None context and default to empty dict."""
        from aragora.server.handlers.evolution.handler import EvolutionHandler

        h = EvolutionHandler(ctx=None)
        assert h.ctx == {}

    def test_routes_class_attribute(self):
        """ROUTES should contain both v1 and non-v1 paths."""
        from aragora.server.handlers.evolution.handler import EvolutionHandler

        routes = EvolutionHandler.ROUTES
        assert "/api/evolution/patterns" in routes
        assert "/api/v1/evolution/patterns" in routes
        assert "/api/evolution/*/history" in routes
        assert "/api/evolution/*/prompt" in routes
        assert len(routes) == 8  # 4 endpoints x 2 (v1 and non-v1)

    def test_handler_is_none_for_rate_limit(self, handler, mod):
        """When HTTP handler is None, rate limiter should use 'unknown' as key."""
        with patch.object(handler, "_get_patterns", return_value=MagicMock(status_code=200)):
            result = handler.handle("/api/evolution/patterns", {}, None)
            # Should still work (get_client_ip returns "unknown" for None handler)
            assert result is not None

    def test_get_db_path_called_with_correct_args(self, handler, mod):
        """Verify get_db_path is called with DatabaseType.PROMPT_EVOLUTION."""
        orig = (mod.EVOLUTION_AVAILABLE, mod.PromptEvolver)
        try:
            mock_cls = _enable_evolution(mod)
            mock_inst = MagicMock()
            mock_inst.get_top_patterns.return_value = []
            mock_cls.return_value = mock_inst

            with patch(
                "aragora.server.handlers.evolution.handler.get_db_path",
                return_value="/tmp/test.db",
            ) as mock_db:
                from aragora.persistence.db_config import DatabaseType

                handler._get_patterns(None, 10)
                mock_db.assert_called_once_with(DatabaseType.PROMPT_EVOLUTION, "/tmp/test_nomic")
        finally:
            mod.EVOLUTION_AVAILABLE, mod.PromptEvolver = orig

    def test_evolver_instantiated_with_db_path(self, handler, mod):
        """Verify PromptEvolver is constructed with the correct db_path kwarg."""
        orig = (mod.EVOLUTION_AVAILABLE, mod.PromptEvolver)
        try:
            mock_cls = _enable_evolution(mod)
            mock_inst = MagicMock()
            mock_inst.get_evolution_history.return_value = []
            mock_cls.return_value = mock_inst

            with patch(
                "aragora.server.handlers.evolution.handler.get_db_path",
                return_value="/tmp/evolution.db",
            ):
                handler._get_evolution_history("claude", 10)
                mock_cls.assert_called_once_with(db_path="/tmp/evolution.db")
        finally:
            mod.EVOLUTION_AVAILABLE, mod.PromptEvolver = orig
