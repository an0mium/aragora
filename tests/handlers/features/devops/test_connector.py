"""Tests for PagerDuty connector instance management.

Covers all public functions and edge cases in
``aragora.server.handlers.features.devops.connector``:

- get_pagerduty_connector(): creation, caching, env-var gating, error paths
- clear_connector_instances(): cleanup of both internal dicts
"""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.devops.connector import (
    _active_contexts,
    _connector_instances,
    clear_connector_instances,
    get_pagerduty_connector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_connector_state():
    """Ensure connector module-level dicts are empty before and after each test."""
    _connector_instances.clear()
    _active_contexts.clear()
    yield
    _connector_instances.clear()
    _active_contexts.clear()


def _make_mock_connector():
    """Build an AsyncMock that behaves like PagerDutyConnector."""
    connector = AsyncMock()
    connector.__aenter__ = AsyncMock(return_value=connector)
    connector.__aexit__ = AsyncMock(return_value=None)
    return connector


# ---------------------------------------------------------------------------
# Tests: get_pagerduty_connector — happy path
# ---------------------------------------------------------------------------


class TestGetPagerdutyConnectorCreation:
    """Tests for creating a new connector when none is cached."""

    @pytest.mark.asyncio
    async def test_creates_connector_with_valid_env(self):
        """A new connector is created when api_key and email are set."""
        mock_connector = _make_mock_connector()
        mock_creds_cls = MagicMock()
        mock_connector_cls = MagicMock(return_value=mock_connector)

        env = {
            "PAGERDUTY_API_KEY": "pk_test_key",
            "PAGERDUTY_EMAIL": "ops@example.com",
            "PAGERDUTY_WEBHOOK_SECRET": "whsec_123",
        }

        with (
            patch.dict("os.environ", env, clear=False),
            patch(
                "aragora.server.handlers.features.devops.connector.PagerDutyConnector",
                mock_connector_cls,
                create=True,
            ),
            patch(
                "aragora.server.handlers.features.devops.connector.PagerDutyCredentials",
                mock_creds_cls,
                create=True,
            ),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyConnector",
                mock_connector_cls,
            ),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyCredentials",
                mock_creds_cls,
            ),
        ):
            result = await get_pagerduty_connector("tenant-1")

        assert result is mock_connector
        mock_connector.__aenter__.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_creates_credentials_with_all_env_vars(self):
        """PagerDutyCredentials receives api_key, email, webhook_secret."""
        mock_connector = _make_mock_connector()
        captured_creds = {}

        def capture_creds(**kwargs):
            captured_creds.update(kwargs)
            return MagicMock()

        mock_connector_cls = MagicMock(return_value=mock_connector)

        env = {
            "PAGERDUTY_API_KEY": "pk_abc",
            "PAGERDUTY_EMAIL": "team@acme.io",
            "PAGERDUTY_WEBHOOK_SECRET": "whsec_xyz",
        }

        with (
            patch.dict("os.environ", env, clear=False),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyConnector",
                mock_connector_cls,
            ),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyCredentials",
                capture_creds,
            ),
        ):
            await get_pagerduty_connector("tenant-cred")

        assert captured_creds["api_key"] == "pk_abc"
        assert captured_creds["email"] == "team@acme.io"
        assert captured_creds["webhook_secret"] == "whsec_xyz"

    @pytest.mark.asyncio
    async def test_webhook_secret_can_be_none(self):
        """Connector is still created when PAGERDUTY_WEBHOOK_SECRET is absent."""
        mock_connector = _make_mock_connector()
        mock_connector_cls = MagicMock(return_value=mock_connector)
        mock_creds_cls = MagicMock()

        env = {
            "PAGERDUTY_API_KEY": "pk_nowebhook",
            "PAGERDUTY_EMAIL": "admin@test.com",
        }

        with (
            patch.dict("os.environ", env, clear=True),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyConnector",
                mock_connector_cls,
            ),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyCredentials",
                mock_creds_cls,
            ),
        ):
            result = await get_pagerduty_connector("tenant-nohook")

        assert result is mock_connector
        # webhook_secret should be None from os.getenv default
        call_kwargs = mock_creds_cls.call_args
        assert (
            call_kwargs[1]["webhook_secret"] is None
            or call_kwargs.kwargs.get("webhook_secret") is None
        )

    @pytest.mark.asyncio
    async def test_stores_connector_in_module_dicts(self):
        """Created connector is cached in both _connector_instances and _active_contexts."""
        mock_connector = _make_mock_connector()
        mock_connector_cls = MagicMock(return_value=mock_connector)
        mock_creds_cls = MagicMock()

        env = {
            "PAGERDUTY_API_KEY": "pk_store",
            "PAGERDUTY_EMAIL": "cache@test.com",
        }

        with (
            patch.dict("os.environ", env, clear=False),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyConnector",
                mock_connector_cls,
            ),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyCredentials",
                mock_creds_cls,
            ),
        ):
            await get_pagerduty_connector("tenant-store")

        assert "tenant-store" in _connector_instances
        assert "tenant-store" in _active_contexts
        assert _connector_instances["tenant-store"] is mock_connector
        assert _active_contexts["tenant-store"] is mock_connector


# ---------------------------------------------------------------------------
# Tests: get_pagerduty_connector — caching behaviour
# ---------------------------------------------------------------------------


class TestGetPagerdutyConnectorCaching:
    """Tests that connectors are cached by tenant_id."""

    @pytest.mark.asyncio
    async def test_returns_cached_connector_on_second_call(self):
        """Second call for same tenant_id returns the cached instance."""
        sentinel = MagicMock(name="cached-connector")
        _connector_instances["tenant-cached"] = sentinel

        result = await get_pagerduty_connector("tenant-cached")
        assert result is sentinel

    @pytest.mark.asyncio
    async def test_does_not_re_create_for_cached_tenant(self):
        """No import / construction happens when connector already cached."""
        _connector_instances["tenant-existing"] = MagicMock()

        with patch(
            "aragora.connectors.devops.pagerduty.PagerDutyConnector",
            side_effect=AssertionError("Should not be called"),
        ):
            result = await get_pagerduty_connector("tenant-existing")

        assert result is _connector_instances["tenant-existing"]

    @pytest.mark.asyncio
    async def test_different_tenants_get_different_connectors(self):
        """Each tenant_id gets its own connector instance."""
        mock_a = _make_mock_connector()
        mock_b = _make_mock_connector()
        call_count = 0

        def connector_factory(creds):
            nonlocal call_count
            call_count += 1
            return mock_a if call_count == 1 else mock_b

        mock_creds_cls = MagicMock()
        env = {
            "PAGERDUTY_API_KEY": "pk_multi",
            "PAGERDUTY_EMAIL": "multi@test.com",
        }

        with (
            patch.dict("os.environ", env, clear=False),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyConnector",
                side_effect=connector_factory,
            ),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyCredentials",
                mock_creds_cls,
            ),
        ):
            result_a = await get_pagerduty_connector("tenant-a")
            result_b = await get_pagerduty_connector("tenant-b")

        assert result_a is mock_a
        assert result_b is mock_b
        assert result_a is not result_b


# ---------------------------------------------------------------------------
# Tests: get_pagerduty_connector — missing env vars
# ---------------------------------------------------------------------------


class TestGetPagerdutyConnectorMissingEnv:
    """Connector creation fails gracefully when env vars are absent."""

    @pytest.mark.asyncio
    async def test_returns_none_when_api_key_missing(self):
        """Returns None when PAGERDUTY_API_KEY is not set."""
        env = {"PAGERDUTY_EMAIL": "ops@test.com"}

        with patch.dict("os.environ", env, clear=True):
            result = await get_pagerduty_connector("tenant-nokey")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_email_missing(self):
        """Returns None when PAGERDUTY_EMAIL is not set."""
        env = {"PAGERDUTY_API_KEY": "pk_test"}

        with patch.dict("os.environ", env, clear=True):
            result = await get_pagerduty_connector("tenant-nomail")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_both_missing(self):
        """Returns None when neither key nor email is set."""
        with patch.dict("os.environ", {}, clear=True):
            result = await get_pagerduty_connector("tenant-empty")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_api_key_empty_string(self):
        """Empty string for api_key is treated as missing."""
        env = {"PAGERDUTY_API_KEY": "", "PAGERDUTY_EMAIL": "ops@test.com"}

        with patch.dict("os.environ", env, clear=True):
            result = await get_pagerduty_connector("tenant-emptykey")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_email_empty_string(self):
        """Empty string for email is treated as missing."""
        env = {"PAGERDUTY_API_KEY": "pk_real", "PAGERDUTY_EMAIL": ""}

        with patch.dict("os.environ", env, clear=True):
            result = await get_pagerduty_connector("tenant-emptyemail")

        assert result is None

    @pytest.mark.asyncio
    async def test_no_cache_entry_on_missing_env(self):
        """When env vars are missing, nothing is stored in module dicts."""
        with patch.dict("os.environ", {}, clear=True):
            await get_pagerduty_connector("tenant-nocache")

        assert "tenant-nocache" not in _connector_instances
        assert "tenant-nocache" not in _active_contexts


# ---------------------------------------------------------------------------
# Tests: get_pagerduty_connector — error handling
# ---------------------------------------------------------------------------


class TestGetPagerdutyConnectorErrors:
    """Tests for ImportError and connection-related exception handling."""

    @pytest.mark.asyncio
    async def test_returns_none_on_import_error(self):
        """ImportError from missing pagerduty package returns None."""
        with patch.dict(
            "os.environ",
            {"PAGERDUTY_API_KEY": "pk_imp", "PAGERDUTY_EMAIL": "err@test.com"},
            clear=True,
        ):
            with patch.dict("sys.modules", {"aragora.connectors.devops.pagerduty": None}):
                # Force ImportError by making the import fail
                import builtins

                original_import = builtins.__import__

                def fail_import(name, *args, **kwargs):
                    if "pagerduty" in name:
                        raise ImportError("No module named pagerduty")
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=fail_import):
                    result = await get_pagerduty_connector("tenant-imp")

        assert result is None
        assert "tenant-imp" not in _connector_instances

    @pytest.mark.asyncio
    async def test_returns_none_on_connection_error(self):
        """ConnectionError during __aenter__ returns None."""
        mock_connector = _make_mock_connector()
        mock_connector.__aenter__ = AsyncMock(side_effect=ConnectionError("refused"))
        mock_connector_cls = MagicMock(return_value=mock_connector)
        mock_creds_cls = MagicMock()

        env = {"PAGERDUTY_API_KEY": "pk_conn", "PAGERDUTY_EMAIL": "conn@test.com"}

        with (
            patch.dict("os.environ", env, clear=False),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyConnector",
                mock_connector_cls,
            ),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyCredentials",
                mock_creds_cls,
            ),
        ):
            result = await get_pagerduty_connector("tenant-conn")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_timeout_error(self):
        """TimeoutError during initialization returns None."""
        mock_connector = _make_mock_connector()
        mock_connector.__aenter__ = AsyncMock(side_effect=TimeoutError("timed out"))
        mock_connector_cls = MagicMock(return_value=mock_connector)
        mock_creds_cls = MagicMock()

        env = {"PAGERDUTY_API_KEY": "pk_tout", "PAGERDUTY_EMAIL": "tout@test.com"}

        with (
            patch.dict("os.environ", env, clear=False),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyConnector",
                mock_connector_cls,
            ),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyCredentials",
                mock_creds_cls,
            ),
        ):
            result = await get_pagerduty_connector("tenant-tout")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_os_error(self):
        """OSError during initialization returns None."""
        mock_connector = _make_mock_connector()
        mock_connector.__aenter__ = AsyncMock(side_effect=OSError("network unreachable"))
        mock_connector_cls = MagicMock(return_value=mock_connector)
        mock_creds_cls = MagicMock()

        env = {"PAGERDUTY_API_KEY": "pk_os", "PAGERDUTY_EMAIL": "os@test.com"}

        with (
            patch.dict("os.environ", env, clear=False),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyConnector",
                mock_connector_cls,
            ),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyCredentials",
                mock_creds_cls,
            ),
        ):
            result = await get_pagerduty_connector("tenant-os")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_value_error(self):
        """ValueError during initialization returns None."""
        mock_connector = _make_mock_connector()
        mock_connector.__aenter__ = AsyncMock(side_effect=ValueError("bad config"))
        mock_connector_cls = MagicMock(return_value=mock_connector)
        mock_creds_cls = MagicMock()

        env = {"PAGERDUTY_API_KEY": "pk_val", "PAGERDUTY_EMAIL": "val@test.com"}

        with (
            patch.dict("os.environ", env, clear=False),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyConnector",
                mock_connector_cls,
            ),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyCredentials",
                mock_creds_cls,
            ),
        ):
            result = await get_pagerduty_connector("tenant-val")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_runtime_error(self):
        """RuntimeError during initialization returns None."""
        mock_connector = _make_mock_connector()
        mock_connector.__aenter__ = AsyncMock(side_effect=RuntimeError("init failed"))
        mock_connector_cls = MagicMock(return_value=mock_connector)
        mock_creds_cls = MagicMock()

        env = {"PAGERDUTY_API_KEY": "pk_rt", "PAGERDUTY_EMAIL": "rt@test.com"}

        with (
            patch.dict("os.environ", env, clear=False),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyConnector",
                mock_connector_cls,
            ),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyCredentials",
                mock_creds_cls,
            ),
        ):
            result = await get_pagerduty_connector("tenant-rt")

        assert result is None

    @pytest.mark.asyncio
    async def test_logs_error_on_connection_failure(self, caplog):
        """Connection errors are logged at ERROR level."""
        mock_connector = _make_mock_connector()
        mock_connector.__aenter__ = AsyncMock(side_effect=ConnectionError("connection refused"))
        mock_connector_cls = MagicMock(return_value=mock_connector)
        mock_creds_cls = MagicMock()

        env = {"PAGERDUTY_API_KEY": "pk_log", "PAGERDUTY_EMAIL": "log@test.com"}

        with (
            patch.dict("os.environ", env, clear=False),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyConnector",
                mock_connector_cls,
            ),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyCredentials",
                mock_creds_cls,
            ),
            caplog.at_level(
                logging.ERROR, logger="aragora.server.handlers.features.devops.connector"
            ),
        ):
            await get_pagerduty_connector("tenant-log")

        assert any(
            "Failed to initialize PagerDuty connector" in rec.message for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_no_cache_entry_on_error(self):
        """When an error occurs, nothing is stored in module dicts."""
        mock_connector = _make_mock_connector()
        mock_connector.__aenter__ = AsyncMock(side_effect=RuntimeError("boom"))
        mock_connector_cls = MagicMock(return_value=mock_connector)
        mock_creds_cls = MagicMock()

        env = {"PAGERDUTY_API_KEY": "pk_err", "PAGERDUTY_EMAIL": "err@test.com"}

        with (
            patch.dict("os.environ", env, clear=False),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyConnector",
                mock_connector_cls,
            ),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyCredentials",
                mock_creds_cls,
            ),
        ):
            await get_pagerduty_connector("tenant-nocache-err")

        assert "tenant-nocache-err" not in _connector_instances
        assert "tenant-nocache-err" not in _active_contexts

    @pytest.mark.asyncio
    async def test_import_error_does_not_log_as_error(self, caplog):
        """ImportError is silently returned as None (no ERROR log)."""
        import builtins

        original_import = builtins.__import__

        def fail_import(name, *args, **kwargs):
            if "pagerduty" in name:
                raise ImportError("no pagerduty module")
            return original_import(name, *args, **kwargs)

        env = {"PAGERDUTY_API_KEY": "pk_imp2", "PAGERDUTY_EMAIL": "imp2@test.com"}

        with (
            patch.dict("os.environ", env, clear=True),
            patch("builtins.__import__", side_effect=fail_import),
            caplog.at_level(
                logging.ERROR, logger="aragora.server.handlers.features.devops.connector"
            ),
        ):
            result = await get_pagerduty_connector("tenant-imp2")

        assert result is None
        # ImportError branch does NOT call logger.error
        error_records = [
            r for r in caplog.records if "Failed to initialize PagerDuty connector" in r.message
        ]
        assert len(error_records) == 0


# ---------------------------------------------------------------------------
# Tests: get_pagerduty_connector — credential construction error
# ---------------------------------------------------------------------------


class TestGetPagerdutyConnectorCredentialError:
    """Errors raised during PagerDutyCredentials construction."""

    @pytest.mark.asyncio
    async def test_returns_none_when_credentials_raise_value_error(self):
        """ValueError from PagerDutyCredentials constructor returns None."""
        mock_creds_cls = MagicMock(side_effect=ValueError("invalid api key format"))

        env = {"PAGERDUTY_API_KEY": "pk_cred", "PAGERDUTY_EMAIL": "cred@test.com"}

        with (
            patch.dict("os.environ", env, clear=False),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyCredentials",
                mock_creds_cls,
            ),
        ):
            result = await get_pagerduty_connector("tenant-cred-err")

        assert result is None


# ---------------------------------------------------------------------------
# Tests: clear_connector_instances
# ---------------------------------------------------------------------------


class TestClearConnectorInstances:
    """Tests for the clear_connector_instances cleanup function."""

    def test_clears_empty_dicts(self):
        """Clearing already-empty dicts is a no-op."""
        clear_connector_instances()
        assert len(_connector_instances) == 0
        assert len(_active_contexts) == 0

    def test_clears_populated_connector_instances(self):
        """_connector_instances is emptied."""
        _connector_instances["t1"] = MagicMock()
        _connector_instances["t2"] = MagicMock()
        clear_connector_instances()
        assert len(_connector_instances) == 0

    def test_clears_populated_active_contexts(self):
        """_active_contexts is emptied."""
        _active_contexts["t1"] = MagicMock()
        _active_contexts["t3"] = MagicMock()
        clear_connector_instances()
        assert len(_active_contexts) == 0

    def test_clears_both_dicts_simultaneously(self):
        """Both dicts are cleared in a single call."""
        _connector_instances["a"] = MagicMock()
        _active_contexts["b"] = MagicMock()
        clear_connector_instances()
        assert len(_connector_instances) == 0
        assert len(_active_contexts) == 0

    def test_double_clear_is_safe(self):
        """Calling clear twice does not raise."""
        _connector_instances["x"] = MagicMock()
        clear_connector_instances()
        clear_connector_instances()
        assert len(_connector_instances) == 0

    @pytest.mark.asyncio
    async def test_clear_then_recreate(self):
        """After clearing, a new connector can be created for the same tenant."""
        sentinel = MagicMock()
        _connector_instances["tenant-re"] = sentinel

        clear_connector_instances()
        assert "tenant-re" not in _connector_instances

        mock_connector = _make_mock_connector()
        mock_connector_cls = MagicMock(return_value=mock_connector)
        mock_creds_cls = MagicMock()

        env = {"PAGERDUTY_API_KEY": "pk_re", "PAGERDUTY_EMAIL": "re@test.com"}

        with (
            patch.dict("os.environ", env, clear=False),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyConnector",
                mock_connector_cls,
            ),
            patch(
                "aragora.connectors.devops.pagerduty.PagerDutyCredentials",
                mock_creds_cls,
            ),
        ):
            result = await get_pagerduty_connector("tenant-re")

        assert result is mock_connector
        assert result is not sentinel
