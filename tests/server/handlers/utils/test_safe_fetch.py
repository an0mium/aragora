"""Tests for safe_fetch module."""

from __future__ import annotations

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import logging
from unittest.mock import MagicMock

import pytest

from aragora.server.handlers.utils.safe_fetch import (
    safe_fetch,
    safe_fetch_async,
    SafeFetchContext,
    fetch_multiple,
    fetch_multiple_async,
    DATA_EXCEPTIONS,
    SYSTEM_EXCEPTIONS,
    _make_fallback,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def logger():
    """Create a mock logger for testing."""
    return MagicMock(spec=logging.Logger)


# =============================================================================
# Test safe_fetch
# =============================================================================


class TestSafeFetch:
    """Tests for safe_fetch function."""

    def test_returns_value_on_success(self, logger):
        """Should return fetched value on success."""
        result = safe_fetch(
            lambda: {"data": "value"},
            fallback={"available": False},
            context="test data",
            logger=logger,
        )

        assert result == {"data": "value"}
        logger.log.assert_not_called()

    def test_returns_fallback_on_data_exception(self, logger):
        """Should return fallback on data exceptions."""
        result = safe_fetch(
            lambda: (_ for _ in ()).throw(KeyError("missing")),
            fallback={"available": False},
            context="test data",
            logger=logger,
        )

        assert result["available"] is False
        assert "error" in result

    def test_returns_fallback_on_value_error(self, logger):
        """Should return fallback on ValueError."""

        def raises_value_error():
            raise ValueError("invalid")

        result = safe_fetch(
            raises_value_error,
            fallback={"available": False},
            context="test data",
            logger=logger,
        )

        assert result["available"] is False

    def test_returns_fallback_on_unexpected_error(self, logger):
        """Should return fallback on unexpected errors."""

        def raises_runtime():
            raise RuntimeError("unexpected")

        result = safe_fetch(
            raises_runtime,
            fallback={"available": False},
            context="test data",
            logger=logger,
        )

        assert result["available"] is False
        assert result["error"] == "Internal error"

    def test_logs_data_errors_at_warning(self, logger):
        """Should log data errors at warning level by default."""

        def raises_key_error():
            raise KeyError("key")

        safe_fetch(
            raises_key_error,
            fallback={},
            context="test",
            logger=logger,
        )

        logger.log.assert_called()

    def test_can_disable_error_in_fallback(self, logger):
        """Should not add error to fallback when disabled."""

        def raises():
            raise ValueError("test")

        result = safe_fetch(
            raises,
            fallback={"original": True},
            context="test",
            logger=logger,
            include_error_in_fallback=False,
        )

        assert result == {"original": True}
        assert "error" not in result

    def test_non_dict_fallback_returned_as_is(self, logger):
        """Should return non-dict fallback as-is."""

        def raises():
            raise KeyError()

        result = safe_fetch(
            raises,
            fallback="string_fallback",
            context="test",
            logger=logger,
        )

        assert result == "string_fallback"


# =============================================================================
# Test safe_fetch_async
# =============================================================================


class TestSafeFetchAsync:
    """Tests for safe_fetch_async function."""

    @pytest.mark.asyncio
    async def test_returns_value_on_success(self, logger):
        """Should return fetched value on success."""

        async def fetch():
            return {"async": "data"}

        result = await safe_fetch_async(
            fetch,
            fallback={"available": False},
            context="async test",
            logger=logger,
        )

        assert result == {"async": "data"}

    @pytest.mark.asyncio
    async def test_returns_fallback_on_exception(self, logger):
        """Should return fallback on exception."""

        async def raises():
            raise ValueError("async error")

        result = await safe_fetch_async(
            raises,
            fallback={"available": False},
            context="async test",
            logger=logger,
        )

        assert result["available"] is False

    @pytest.mark.asyncio
    async def test_handles_async_runtime_error(self, logger):
        """Should handle async runtime errors."""

        async def raises():
            raise RuntimeError("runtime")

        result = await safe_fetch_async(
            raises,
            fallback={"default": True},
            context="test",
            logger=logger,
        )

        assert result["error"] == "Internal error"


# =============================================================================
# Test SafeFetchContext
# =============================================================================


class TestSafeFetchContext:
    """Tests for SafeFetchContext class."""

    def test_context_manager_returns_self(self, logger):
        """Should return self from __enter__."""
        ctx = SafeFetchContext(logger=logger, context_prefix="test")
        with ctx as c:
            assert c is ctx

    def test_fetch_returns_value_on_success(self, logger):
        """Should return fetched value on success."""
        with SafeFetchContext(logger=logger, context_prefix="debate-123") as ctx:
            result = ctx.fetch(
                lambda: {"memory": "data"},
                fallback={"available": False},
                name="memory",
            )

        assert result == {"memory": "data"}
        assert ctx.has_errors is False

    def test_fetch_returns_fallback_on_error(self, logger):
        """Should return fallback on error."""
        with SafeFetchContext(logger=logger, context_prefix="debate-123") as ctx:
            result = ctx.fetch(
                lambda: (_ for _ in ()).throw(KeyError("missing")),
                fallback={"available": False},
                name="memory",
            )

        assert result["available"] is False
        assert ctx.has_errors is True

    def test_tracks_errors(self, logger):
        """Should track errors from failed fetches."""
        with SafeFetchContext(logger=logger, context_prefix="test") as ctx:
            ctx.fetch(lambda: (_ for _ in ()).throw(ValueError("err1")), {"available": False}, "a")
            ctx.fetch(lambda: (_ for _ in ()).throw(KeyError("err2")), {"available": False}, "b")

        assert len(ctx.errors) == 2

    @pytest.mark.asyncio
    async def test_fetch_async_returns_value(self, logger):
        """Should return async fetched value."""

        async def async_fetch():
            return {"async": True}

        with SafeFetchContext(logger=logger) as ctx:
            result = await ctx.fetch_async(
                async_fetch,
                fallback={"available": False},
                name="async_data",
            )

        assert result == {"async": True}


# =============================================================================
# Test fetch_multiple
# =============================================================================


class TestFetchMultiple:
    """Tests for fetch_multiple function."""

    def test_fetches_all_successfully(self, logger):
        """Should fetch all values successfully."""
        fetchers = {
            "a": lambda: {"value": "A"},
            "b": lambda: {"value": "B"},
            "c": lambda: {"value": "C"},
        }

        results = fetch_multiple(
            fetchers,
            fallback={"available": False},
            context_prefix="test",
            logger=logger,
        )

        assert results["a"]["value"] == "A"
        assert results["b"]["value"] == "B"
        assert results["c"]["value"] == "C"

    def test_uses_fallback_for_failed_fetches(self, logger):
        """Should use fallback for failed fetches."""

        def fails():
            raise ValueError("failed")

        fetchers = {
            "success": lambda: {"value": "ok"},
            "failure": fails,
        }

        results = fetch_multiple(
            fetchers,
            fallback={"available": False},
            context_prefix="test",
            logger=logger,
        )

        assert results["success"]["value"] == "ok"
        assert results["failure"]["available"] is False


# =============================================================================
# Test fetch_multiple_async
# =============================================================================


class TestFetchMultipleAsync:
    """Tests for fetch_multiple_async function."""

    @pytest.mark.asyncio
    async def test_fetches_all_async(self, logger):
        """Should fetch all values asynchronously."""

        async def fetch_a():
            return {"value": "A"}

        async def fetch_b():
            return {"value": "B"}

        fetchers = {
            "a": fetch_a,
            "b": fetch_b,
        }

        results = await fetch_multiple_async(
            fetchers,
            fallback={"available": False},
            context_prefix="test",
            logger=logger,
        )

        assert results["a"]["value"] == "A"
        assert results["b"]["value"] == "B"


# =============================================================================
# Test _make_fallback
# =============================================================================


class TestMakeFallback:
    """Tests for _make_fallback function."""

    def test_adds_error_to_dict_fallback(self):
        """Should add error info to dict fallback."""
        fallback = {"original": True}
        result = _make_fallback(fallback, "test error", include_error=True)

        assert result["original"] is True
        assert result["error"] == "test error"
        assert result["available"] is False

    def test_does_not_mutate_original(self):
        """Should not mutate original fallback dict."""
        fallback = {"original": True}
        _make_fallback(fallback, "error", include_error=True)

        assert "error" not in fallback

    def test_returns_fallback_as_is_when_disabled(self):
        """Should return fallback as-is when include_error=False."""
        fallback = {"original": True}
        result = _make_fallback(fallback, "error", include_error=False)

        assert result == {"original": True}
        assert "error" not in result

    def test_returns_non_dict_fallback_unchanged(self):
        """Should return non-dict fallback unchanged."""
        result = _make_fallback("string", "error", include_error=True)
        assert result == "string"


# =============================================================================
# Test Exception Constants
# =============================================================================


class TestExceptionConstants:
    """Tests for exception constant tuples."""

    def test_data_exceptions_includes_common(self):
        """Should include common data access exceptions."""
        assert KeyError in DATA_EXCEPTIONS
        assert ValueError in DATA_EXCEPTIONS
        assert TypeError in DATA_EXCEPTIONS
        assert AttributeError in DATA_EXCEPTIONS

    def test_system_exceptions_includes_io(self):
        """Should include I/O related exceptions."""
        assert OSError in SYSTEM_EXCEPTIONS
        assert IOError in SYSTEM_EXCEPTIONS
        assert RuntimeError in SYSTEM_EXCEPTIONS
        assert ConnectionError in SYSTEM_EXCEPTIONS
