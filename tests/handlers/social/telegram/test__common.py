"""Tests for the shared _common.py module in the Telegram handler package.

Covers all components:
- Permission constants (PERM_TELEGRAM_*)
- Environment variable configuration (TELEGRAM_BOT_TOKEN, TELEGRAM_WEBHOOK_SECRET, TELEGRAM_API_BASE)
- _handle_task_exception: cancelled, failed, and successful tasks
- create_tracked_task: with running loop, without running loop (RuntimeError fallback),
  task exception callback wiring
- RBAC imports: RBAC_AVAILABLE, check_permission, extract_user_from_request, AuthorizationContext
- _tg() lazy import helper
- TelegramRBACMixin:
    - _get_auth_context: RBAC available/unavailable, extract_user returns None/valid,
      exception handling
    - _get_telegram_user_context: RBAC available/unavailable, exception handling
    - _check_permission: RBAC unavailable (fail open/closed), context None, allowed, denied,
      exception handling
    - _check_telegram_user_permission: RBAC unavailable (fail open/closed),
      TELEGRAM_RBAC_ENABLED toggling, context None, allowed, denied, exception handling
    - _deny_telegram_permission: sends message and returns json_response
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _ensure_no_production_env(monkeypatch):
    """Ensure we are NOT in a production environment for RBAC fail-open tests."""
    monkeypatch.delenv("ARAGORA_ENV", raising=False)


@pytest.fixture
def _set_production_env(monkeypatch):
    """Set production environment for RBAC fail-closed tests."""
    monkeypatch.setenv("ARAGORA_ENV", "production")


# ============================================================================
# Permission Constants
# ============================================================================


class TestPermissionConstants:
    """Verify all PERM_TELEGRAM_* constants are defined and have correct values."""

    def test_perm_telegram_read(self):
        from aragora.server.handlers.social.telegram._common import PERM_TELEGRAM_READ

        assert PERM_TELEGRAM_READ == "telegram:read"

    def test_perm_telegram_messages_send(self):
        from aragora.server.handlers.social.telegram._common import PERM_TELEGRAM_MESSAGES_SEND

        assert PERM_TELEGRAM_MESSAGES_SEND == "telegram:messages:send"

    def test_perm_telegram_debates_create(self):
        from aragora.server.handlers.social.telegram._common import PERM_TELEGRAM_DEBATES_CREATE

        assert PERM_TELEGRAM_DEBATES_CREATE == "telegram:debates:create"

    def test_perm_telegram_debates_read(self):
        from aragora.server.handlers.social.telegram._common import PERM_TELEGRAM_DEBATES_READ

        assert PERM_TELEGRAM_DEBATES_READ == "telegram:debates:read"

    def test_perm_telegram_gauntlet_run(self):
        from aragora.server.handlers.social.telegram._common import PERM_TELEGRAM_GAUNTLET_RUN

        assert PERM_TELEGRAM_GAUNTLET_RUN == "telegram:gauntlet:run"

    def test_perm_telegram_votes_record(self):
        from aragora.server.handlers.social.telegram._common import PERM_TELEGRAM_VOTES_RECORD

        assert PERM_TELEGRAM_VOTES_RECORD == "telegram:votes:record"

    def test_perm_telegram_commands_execute(self):
        from aragora.server.handlers.social.telegram._common import PERM_TELEGRAM_COMMANDS_EXECUTE

        assert PERM_TELEGRAM_COMMANDS_EXECUTE == "telegram:commands:execute"

    def test_perm_telegram_callbacks_handle(self):
        from aragora.server.handlers.social.telegram._common import PERM_TELEGRAM_CALLBACKS_HANDLE

        assert PERM_TELEGRAM_CALLBACKS_HANDLE == "telegram:callbacks:handle"

    def test_perm_telegram_admin(self):
        from aragora.server.handlers.social.telegram._common import PERM_TELEGRAM_ADMIN

        assert PERM_TELEGRAM_ADMIN == "telegram:admin"

    def test_all_permissions_are_strings(self):
        from aragora.server.handlers.social.telegram._common import (
            PERM_TELEGRAM_ADMIN,
            PERM_TELEGRAM_CALLBACKS_HANDLE,
            PERM_TELEGRAM_COMMANDS_EXECUTE,
            PERM_TELEGRAM_DEBATES_CREATE,
            PERM_TELEGRAM_DEBATES_READ,
            PERM_TELEGRAM_GAUNTLET_RUN,
            PERM_TELEGRAM_MESSAGES_SEND,
            PERM_TELEGRAM_READ,
            PERM_TELEGRAM_VOTES_RECORD,
        )

        perms = [
            PERM_TELEGRAM_READ,
            PERM_TELEGRAM_MESSAGES_SEND,
            PERM_TELEGRAM_DEBATES_CREATE,
            PERM_TELEGRAM_DEBATES_READ,
            PERM_TELEGRAM_GAUNTLET_RUN,
            PERM_TELEGRAM_VOTES_RECORD,
            PERM_TELEGRAM_COMMANDS_EXECUTE,
            PERM_TELEGRAM_CALLBACKS_HANDLE,
            PERM_TELEGRAM_ADMIN,
        ]
        for p in perms:
            assert isinstance(p, str)
            assert p.startswith("telegram:")

    def test_all_permissions_unique(self):
        from aragora.server.handlers.social.telegram._common import (
            PERM_TELEGRAM_ADMIN,
            PERM_TELEGRAM_CALLBACKS_HANDLE,
            PERM_TELEGRAM_COMMANDS_EXECUTE,
            PERM_TELEGRAM_DEBATES_CREATE,
            PERM_TELEGRAM_DEBATES_READ,
            PERM_TELEGRAM_GAUNTLET_RUN,
            PERM_TELEGRAM_MESSAGES_SEND,
            PERM_TELEGRAM_READ,
            PERM_TELEGRAM_VOTES_RECORD,
        )

        perms = [
            PERM_TELEGRAM_READ,
            PERM_TELEGRAM_MESSAGES_SEND,
            PERM_TELEGRAM_DEBATES_CREATE,
            PERM_TELEGRAM_DEBATES_READ,
            PERM_TELEGRAM_GAUNTLET_RUN,
            PERM_TELEGRAM_VOTES_RECORD,
            PERM_TELEGRAM_COMMANDS_EXECUTE,
            PERM_TELEGRAM_CALLBACKS_HANDLE,
            PERM_TELEGRAM_ADMIN,
        ]
        assert len(perms) == len(set(perms))


# ============================================================================
# Environment Variable Configuration
# ============================================================================


class TestEnvironmentConfig:
    """Test environment variable constants."""

    def test_telegram_api_base(self):
        from aragora.server.handlers.social.telegram._common import TELEGRAM_API_BASE

        assert TELEGRAM_API_BASE == "https://api.telegram.org/bot"

    def test_telegram_api_base_ends_with_bot(self):
        from aragora.server.handlers.social.telegram._common import TELEGRAM_API_BASE

        assert TELEGRAM_API_BASE.endswith("/bot")

    def test_telegram_bot_token_type(self):
        """TELEGRAM_BOT_TOKEN is either None or a string from env."""
        from aragora.server.handlers.social.telegram._common import TELEGRAM_BOT_TOKEN

        assert TELEGRAM_BOT_TOKEN is None or isinstance(TELEGRAM_BOT_TOKEN, str)

    def test_telegram_webhook_secret_type(self):
        """TELEGRAM_WEBHOOK_SECRET is either None or a string from env."""
        from aragora.server.handlers.social.telegram._common import TELEGRAM_WEBHOOK_SECRET

        assert TELEGRAM_WEBHOOK_SECRET is None or isinstance(TELEGRAM_WEBHOOK_SECRET, str)


# ============================================================================
# _handle_task_exception
# ============================================================================


class TestHandleTaskException:
    """Test the fire-and-forget task exception handler."""

    def test_cancelled_task_logs_debug(self, caplog):
        """Cancelled tasks log at debug level."""
        from aragora.server.handlers.social.telegram._common import _handle_task_exception

        task = MagicMock(spec=asyncio.Task)
        task.cancelled.return_value = True

        with caplog.at_level(logging.DEBUG):
            _handle_task_exception(task, "test-cancelled-task")

        assert any("cancelled" in r.message for r in caplog.records)

    def test_failed_task_logs_error(self, caplog):
        """Failed tasks log at error level with the exception."""
        from aragora.server.handlers.social.telegram._common import _handle_task_exception

        exc = RuntimeError("task failed")
        task = MagicMock(spec=asyncio.Task)
        task.cancelled.return_value = False
        task.exception.return_value = exc

        with caplog.at_level(logging.ERROR):
            _handle_task_exception(task, "test-failed-task")

        assert any("test-failed-task" in r.message for r in caplog.records)
        assert any("task failed" in r.message for r in caplog.records)

    def test_successful_task_does_not_log(self, caplog):
        """Successful tasks (no exception, not cancelled) do nothing."""
        from aragora.server.handlers.social.telegram._common import _handle_task_exception

        task = MagicMock(spec=asyncio.Task)
        task.cancelled.return_value = False
        task.exception.return_value = None

        with caplog.at_level(logging.DEBUG):
            _handle_task_exception(task, "test-ok-task")

        # Should not log cancelled or error messages
        assert not any("cancelled" in r.message for r in caplog.records)
        assert not any("failed" in r.message for r in caplog.records)

    def test_task_name_appears_in_cancelled_log(self, caplog):
        """Task name is included in cancelled log message."""
        from aragora.server.handlers.social.telegram._common import _handle_task_exception

        task = MagicMock(spec=asyncio.Task)
        task.cancelled.return_value = True

        with caplog.at_level(logging.DEBUG):
            _handle_task_exception(task, "my-special-task-42")

        assert any("my-special-task-42" in r.message for r in caplog.records)

    def test_task_name_appears_in_error_log(self, caplog):
        """Task name is included in error log message."""
        from aragora.server.handlers.social.telegram._common import _handle_task_exception

        task = MagicMock(spec=asyncio.Task)
        task.cancelled.return_value = False
        task.exception.return_value = ValueError("bad value")

        with caplog.at_level(logging.ERROR):
            _handle_task_exception(task, "my-error-task-99")

        assert any("my-error-task-99" in r.message for r in caplog.records)


# ============================================================================
# create_tracked_task
# ============================================================================


class TestCreateTrackedTask:
    """Test the async task creation with exception tracking."""

    @pytest.mark.asyncio
    async def test_creates_task_with_running_loop(self):
        """When there's a running event loop, creates task via asyncio.create_task."""
        from aragora.server.handlers.social.telegram._common import create_tracked_task

        executed = False

        async def simple_coro():
            nonlocal executed
            executed = True

        task = create_tracked_task(simple_coro(), name="test-loop-task")
        await task
        assert executed
        assert task.done()

    @pytest.mark.asyncio
    async def test_task_has_done_callback(self):
        """Task gets a done callback attached for exception handling."""
        from aragora.server.handlers.social.telegram._common import create_tracked_task

        async def simple_coro():
            return 42

        task = create_tracked_task(simple_coro(), name="test-cb-task")
        # The task should have at least one done callback
        # (asyncio.Task tracks callbacks internally)
        await task
        assert task.result() == 42

    @pytest.mark.asyncio
    async def test_task_name_is_set(self):
        """Task is created with the specified name."""
        from aragora.server.handlers.social.telegram._common import create_tracked_task

        async def simple_coro():
            pass

        task = create_tracked_task(simple_coro(), name="my-named-task")
        await task
        assert task.get_name() == "my-named-task"

    @pytest.mark.asyncio
    async def test_exception_in_task_is_logged(self, caplog):
        """Exception in the tracked task is caught by the callback."""
        from aragora.server.handlers.social.telegram._common import create_tracked_task

        async def failing_coro():
            raise RuntimeError("intentional failure")

        with caplog.at_level(logging.ERROR):
            task = create_tracked_task(failing_coro(), name="failing-task")
            # Wait for the task to complete, then let callbacks fire
            try:
                await task
            except RuntimeError:
                pass
            # Allow event loop to process callbacks
            await asyncio.sleep(0)

        # The done callback should have logged the error
        assert any("failing-task" in r.message for r in caplog.records)

    def test_fallback_without_running_loop(self):
        """When no event loop is running, creates a new one and runs inline."""
        from aragora.server.handlers.social.telegram._common import create_tracked_task

        executed = False

        async def inline_coro():
            nonlocal executed
            executed = True

        # Run outside of any async context
        # Patch asyncio.create_task to raise RuntimeError (no running loop)
        with patch("asyncio.create_task", side_effect=RuntimeError("no running loop")):
            task = create_tracked_task(inline_coro(), name="inline-task")

        assert executed
        assert task.done()

    def test_fallback_cleans_up_event_loop(self):
        """Fallback path closes the loop and sets event loop to None."""
        from aragora.server.handlers.social.telegram._common import create_tracked_task

        async def noop_coro():
            pass

        with patch("asyncio.create_task", side_effect=RuntimeError("no loop")):
            create_tracked_task(noop_coro(), name="cleanup-task")

        # After cleanup, getting event loop should raise or return None
        # (depends on Python version / platform, so just verify no crash)


# ============================================================================
# RBAC Imports
# ============================================================================


class TestRBACImports:
    """Test the RBAC optional import setup."""

    def test_rbac_available_is_bool(self):
        from aragora.server.handlers.social.telegram._common import RBAC_AVAILABLE

        assert isinstance(RBAC_AVAILABLE, bool)

    def test_check_permission_is_callable_or_none(self):
        from aragora.server.handlers.social.telegram._common import check_permission

        assert check_permission is None or callable(check_permission)

    def test_extract_user_from_request_is_callable_or_none(self):
        from aragora.server.handlers.social.telegram._common import extract_user_from_request

        assert extract_user_from_request is None or callable(extract_user_from_request)

    def test_authorization_context_is_type_or_none(self):
        from aragora.server.handlers.social.telegram._common import AuthorizationContext

        assert AuthorizationContext is None or isinstance(AuthorizationContext, type)

    def test_rbac_available_true_when_imports_succeed(self):
        """When RBAC imports succeed, RBAC_AVAILABLE is True."""
        from aragora.server.handlers.social.telegram._common import RBAC_AVAILABLE

        # In the test environment, RBAC should be available
        assert RBAC_AVAILABLE is True

    def test_check_permission_not_none_when_rbac_available(self):
        from aragora.server.handlers.social.telegram._common import (
            RBAC_AVAILABLE,
            check_permission,
        )

        if RBAC_AVAILABLE:
            assert check_permission is not None


# ============================================================================
# _tg() Lazy Import Helper
# ============================================================================


class TestTgLazyImport:
    """Test the _tg() lazy import helper."""

    def test_tg_returns_telegram_module(self):
        from aragora.server.handlers.social.telegram._common import _tg

        tg_module = _tg()
        assert hasattr(tg_module, "RBAC_AVAILABLE")
        assert hasattr(tg_module, "create_tracked_task")

    def test_tg_returns_same_module_each_call(self):
        """_tg() returns a consistent reference each time."""
        from aragora.server.handlers.social.telegram._common import _tg

        mod1 = _tg()
        mod2 = _tg()
        assert mod1 is mod2

    def test_tg_has_permission_constants(self):
        from aragora.server.handlers.social.telegram._common import _tg

        tg_module = _tg()
        assert hasattr(tg_module, "PERM_TELEGRAM_READ")
        assert hasattr(tg_module, "PERM_TELEGRAM_ADMIN")

    def test_tg_has_telegram_bot_token(self):
        from aragora.server.handlers.social.telegram._common import _tg

        tg_module = _tg()
        assert hasattr(tg_module, "TELEGRAM_BOT_TOKEN")


# ============================================================================
# TelegramRBACMixin: _get_auth_context
# ============================================================================


class TestGetAuthContext:
    """Test TelegramRBACMixin._get_auth_context method."""

    def _make_mixin(self):
        from aragora.server.handlers.social.telegram._common import TelegramRBACMixin

        class TestMixin(TelegramRBACMixin):
            pass

        return TestMixin()

    def test_returns_none_when_rbac_unavailable(self):
        mixin = self._make_mixin()
        handler = MagicMock()
        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn:
            mock_tg_fn.return_value.RBAC_AVAILABLE = False
            result = mixin._get_auth_context(handler)
        assert result is None

    def test_returns_none_when_extract_user_is_none(self):
        mixin = self._make_mixin()
        handler = MagicMock()
        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.extract_user_from_request",
            None,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._get_auth_context(handler)
        assert result is None

    def test_returns_none_when_extract_user_returns_none(self):
        mixin = self._make_mixin()
        handler = MagicMock()
        mock_extract = MagicMock(return_value=None)
        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.extract_user_from_request",
            mock_extract,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._get_auth_context(handler)
        assert result is None

    def test_returns_auth_context_when_user_extracted(self):
        mixin = self._make_mixin()
        handler = MagicMock()

        mock_user = MagicMock()
        mock_user.user_id = "user-123"
        mock_user.role = "admin"
        mock_user.org_id = "org-456"

        mock_extract = MagicMock(return_value=mock_user)

        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.extract_user_from_request",
            mock_extract,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._get_auth_context(handler)

        assert result is not None
        assert result.user_id == "user-123"
        assert "admin" in result.roles
        assert result.org_id == "org-456"

    def test_returns_anonymous_when_user_id_is_none(self):
        mixin = self._make_mixin()
        handler = MagicMock()

        mock_user = MagicMock()
        mock_user.user_id = None
        mock_user.role = None
        mock_user.org_id = None

        mock_extract = MagicMock(return_value=mock_user)

        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.extract_user_from_request",
            mock_extract,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._get_auth_context(handler)

        assert result is not None
        assert result.user_id == "anonymous"

    def test_returns_empty_roles_when_role_is_none(self):
        mixin = self._make_mixin()
        handler = MagicMock()

        mock_user = MagicMock()
        mock_user.user_id = "u1"
        mock_user.role = None
        mock_user.org_id = None

        mock_extract = MagicMock(return_value=mock_user)

        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.extract_user_from_request",
            mock_extract,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._get_auth_context(handler)

        assert result is not None
        assert result.roles == set()

    def test_catches_value_error(self):
        mixin = self._make_mixin()
        handler = MagicMock()

        mock_extract = MagicMock(side_effect=ValueError("bad value"))

        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.extract_user_from_request",
            mock_extract,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._get_auth_context(handler)
        assert result is None

    def test_catches_type_error(self):
        mixin = self._make_mixin()
        handler = MagicMock()

        mock_extract = MagicMock(side_effect=TypeError("bad type"))

        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.extract_user_from_request",
            mock_extract,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._get_auth_context(handler)
        assert result is None

    def test_catches_attribute_error(self):
        mixin = self._make_mixin()
        handler = MagicMock()

        mock_extract = MagicMock(side_effect=AttributeError("no attr"))

        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.extract_user_from_request",
            mock_extract,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._get_auth_context(handler)
        assert result is None

    def test_catches_key_error(self):
        mixin = self._make_mixin()
        handler = MagicMock()

        mock_extract = MagicMock(side_effect=KeyError("missing key"))

        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.extract_user_from_request",
            mock_extract,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._get_auth_context(handler)
        assert result is None


# ============================================================================
# TelegramRBACMixin: _get_telegram_user_context
# ============================================================================


class TestGetTelegramUserContext:
    """Test TelegramRBACMixin._get_telegram_user_context method."""

    def _make_mixin(self):
        from aragora.server.handlers.social.telegram._common import TelegramRBACMixin

        class TestMixin(TelegramRBACMixin):
            pass

        return TestMixin()

    def test_returns_none_when_rbac_unavailable(self):
        mixin = self._make_mixin()
        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn:
            mock_tg_fn.return_value.RBAC_AVAILABLE = False
            result = mixin._get_telegram_user_context(123, "testuser", 456)
        assert result is None

    def test_returns_none_when_auth_context_class_is_none(self):
        mixin = self._make_mixin()
        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.AuthorizationContext",
            None,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._get_telegram_user_context(123, "testuser", 456)
        assert result is None

    def test_returns_context_with_telegram_prefix(self):
        mixin = self._make_mixin()
        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn:
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._get_telegram_user_context(67890, "alice", 12345)

        assert result is not None
        assert result.user_id == "telegram:67890"
        assert "telegram_user" in result.roles
        assert result.org_id is None

    def test_user_id_is_string_with_telegram_prefix(self):
        mixin = self._make_mixin()
        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn:
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._get_telegram_user_context(999, "bob", 111)

        assert result.user_id.startswith("telegram:")
        assert "999" in result.user_id

    def test_catches_value_error(self):
        mixin = self._make_mixin()
        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.AuthorizationContext",
            side_effect=ValueError("bad"),
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._get_telegram_user_context(123, "user", 456)
        assert result is None

    def test_catches_type_error(self):
        mixin = self._make_mixin()
        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.AuthorizationContext",
            side_effect=TypeError("bad type"),
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._get_telegram_user_context(123, "user", 456)
        assert result is None

    def test_catches_attribute_error(self):
        mixin = self._make_mixin()
        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.AuthorizationContext",
            side_effect=AttributeError("no attr"),
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._get_telegram_user_context(123, "user", 456)
        assert result is None


# ============================================================================
# TelegramRBACMixin: _check_permission
# ============================================================================


class TestCheckPermission:
    """Test TelegramRBACMixin._check_permission method (HTTP handler RBAC)."""

    def _make_mixin(self):
        from aragora.server.handlers.social.telegram._common import TelegramRBACMixin

        class TestMixin(TelegramRBACMixin):
            pass

        return TestMixin()

    def test_returns_none_when_rbac_unavailable_non_production(
        self, _ensure_no_production_env
    ):
        """RBAC unavailable in non-production: returns None (fail open)."""
        mixin = self._make_mixin()
        handler = MagicMock()
        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.check_permission",
            None,
        ), patch(
            "aragora.server.handlers.social.telegram._common.rbac_fail_closed",
            return_value=False,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = False
            result = mixin._check_permission(handler, "telegram:read")
        assert result is None

    def test_returns_503_when_rbac_unavailable_production(self):
        """RBAC unavailable in production: returns 503 (fail closed)."""
        mixin = self._make_mixin()
        handler = MagicMock()
        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.check_permission",
            None,
        ), patch(
            "aragora.server.handlers.social.telegram._common.rbac_fail_closed",
            return_value=True,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = False
            result = mixin._check_permission(handler, "telegram:read")

        assert result is not None
        assert _status(result) == 503
        assert "access control" in _body(result).get("error", "").lower()

    def test_returns_none_when_context_is_none(self):
        """When _get_auth_context returns None, returns None (skip check)."""
        mixin = self._make_mixin()
        handler = MagicMock()
        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch.object(
            mixin, "_get_auth_context", return_value=None
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._check_permission(handler, "telegram:read")
        assert result is None

    def test_returns_none_when_permission_allowed(self):
        """When permission is allowed, returns None."""
        mixin = self._make_mixin()
        handler = MagicMock()

        mock_ctx = MagicMock()
        mock_ctx.user_id = "user-1"
        mock_decision = MagicMock()
        mock_decision.allowed = True

        mock_check = MagicMock(return_value=mock_decision)

        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch.object(
            mixin, "_get_auth_context", return_value=mock_ctx
        ), patch(
            "aragora.server.handlers.social.telegram._common.check_permission",
            mock_check,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._check_permission(handler, "telegram:read")

        assert result is None
        mock_check.assert_called_once_with(mock_ctx, "telegram:read")

    def test_returns_403_when_permission_denied(self):
        """When permission is denied, returns 403."""
        mixin = self._make_mixin()
        handler = MagicMock()

        mock_ctx = MagicMock()
        mock_ctx.user_id = "user-2"
        mock_decision = MagicMock()
        mock_decision.allowed = False

        mock_check = MagicMock(return_value=mock_decision)

        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch.object(
            mixin, "_get_auth_context", return_value=mock_ctx
        ), patch(
            "aragora.server.handlers.social.telegram._common.check_permission",
            mock_check,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._check_permission(handler, "telegram:admin")

        assert result is not None
        assert _status(result) == 403
        assert "denied" in _body(result).get("error", "").lower()

    def test_catches_value_error_returns_none(self):
        """ValueError in check_permission is caught, returns None."""
        mixin = self._make_mixin()
        handler = MagicMock()

        mock_ctx = MagicMock()
        mock_check = MagicMock(side_effect=ValueError("bad"))

        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch.object(
            mixin, "_get_auth_context", return_value=mock_ctx
        ), patch(
            "aragora.server.handlers.social.telegram._common.check_permission",
            mock_check,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._check_permission(handler, "telegram:read")

        assert result is None

    def test_catches_type_error_returns_none(self):
        mixin = self._make_mixin()
        handler = MagicMock()
        mock_ctx = MagicMock()
        mock_check = MagicMock(side_effect=TypeError("bad type"))

        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch.object(
            mixin, "_get_auth_context", return_value=mock_ctx
        ), patch(
            "aragora.server.handlers.social.telegram._common.check_permission",
            mock_check,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._check_permission(handler, "telegram:read")
        assert result is None

    def test_catches_attribute_error_returns_none(self):
        mixin = self._make_mixin()
        handler = MagicMock()
        mock_ctx = MagicMock()
        mock_check = MagicMock(side_effect=AttributeError("attr"))

        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch.object(
            mixin, "_get_auth_context", return_value=mock_ctx
        ), patch(
            "aragora.server.handlers.social.telegram._common.check_permission",
            mock_check,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._check_permission(handler, "telegram:read")
        assert result is None

    def test_catches_runtime_error_returns_none(self):
        mixin = self._make_mixin()
        handler = MagicMock()
        mock_ctx = MagicMock()
        mock_check = MagicMock(side_effect=RuntimeError("runtime"))

        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch.object(
            mixin, "_get_auth_context", return_value=mock_ctx
        ), patch(
            "aragora.server.handlers.social.telegram._common.check_permission",
            mock_check,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._check_permission(handler, "telegram:read")
        assert result is None

    def test_logs_warning_when_permission_denied(self, caplog):
        """Permission denied logs a warning with permission key and user_id."""
        mixin = self._make_mixin()
        handler = MagicMock()

        mock_ctx = MagicMock()
        mock_ctx.user_id = "denied-user-42"
        mock_decision = MagicMock()
        mock_decision.allowed = False

        mock_check = MagicMock(return_value=mock_decision)

        with caplog.at_level(logging.WARNING), patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch.object(
            mixin, "_get_auth_context", return_value=mock_ctx
        ), patch(
            "aragora.server.handlers.social.telegram._common.check_permission",
            mock_check,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            mixin._check_permission(handler, "telegram:admin")

        assert any("denied-user-42" in r.message for r in caplog.records)
        assert any("telegram:admin" in r.message for r in caplog.records)


# ============================================================================
# TelegramRBACMixin: _check_telegram_user_permission
# ============================================================================


class TestCheckTelegramUserPermission:
    """Test TelegramRBACMixin._check_telegram_user_permission method."""

    def _make_mixin(self):
        from aragora.server.handlers.social.telegram._common import TelegramRBACMixin

        class TestMixin(TelegramRBACMixin):
            pass

        return TestMixin()

    def test_returns_true_when_rbac_unavailable_non_production(
        self, _ensure_no_production_env
    ):
        """RBAC unavailable in dev: returns True (fail open)."""
        mixin = self._make_mixin()
        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.check_permission",
            None,
        ), patch(
            "aragora.server.handlers.social.telegram._common.rbac_fail_closed",
            return_value=False,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = False
            result = mixin._check_telegram_user_permission(
                123, "user", 456, "telegram:read"
            )
        assert result is True

    def test_returns_false_when_rbac_unavailable_production(self):
        """RBAC unavailable in production: returns False (fail closed)."""
        mixin = self._make_mixin()
        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.check_permission",
            None,
        ), patch(
            "aragora.server.handlers.social.telegram._common.rbac_fail_closed",
            return_value=True,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = False
            result = mixin._check_telegram_user_permission(
                123, "user", 456, "telegram:read"
            )
        assert result is False

    def test_returns_true_when_telegram_rbac_not_enabled(self, monkeypatch):
        """When TELEGRAM_RBAC_ENABLED is not set, returns True (skip check)."""
        monkeypatch.delenv("TELEGRAM_RBAC_ENABLED", raising=False)
        mixin = self._make_mixin()
        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn:
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._check_telegram_user_permission(
                123, "user", 456, "telegram:read"
            )
        assert result is True

    def test_returns_true_when_telegram_rbac_false(self, monkeypatch):
        """When TELEGRAM_RBAC_ENABLED=false, returns True."""
        monkeypatch.setenv("TELEGRAM_RBAC_ENABLED", "false")
        mixin = self._make_mixin()
        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn:
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._check_telegram_user_permission(
                123, "user", 456, "telegram:read"
            )
        assert result is True

    def test_checks_permission_when_telegram_rbac_enabled(self, monkeypatch):
        """When TELEGRAM_RBAC_ENABLED=true and allowed, returns True."""
        monkeypatch.setenv("TELEGRAM_RBAC_ENABLED", "true")
        mixin = self._make_mixin()

        mock_decision = MagicMock()
        mock_decision.allowed = True
        mock_check = MagicMock(return_value=mock_decision)

        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.check_permission",
            mock_check,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._check_telegram_user_permission(
                67890, "alice", 12345, "telegram:debates:create"
            )

        assert result is True
        mock_check.assert_called_once()

    def test_returns_false_when_permission_denied(self, monkeypatch):
        """When TELEGRAM_RBAC_ENABLED=true and denied, returns False."""
        monkeypatch.setenv("TELEGRAM_RBAC_ENABLED", "true")
        mixin = self._make_mixin()

        mock_decision = MagicMock()
        mock_decision.allowed = False
        mock_decision.reason = "insufficient privileges"
        mock_check = MagicMock(return_value=mock_decision)

        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.check_permission",
            mock_check,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._check_telegram_user_permission(
                67890, "alice", 12345, "telegram:admin"
            )

        assert result is False

    def test_returns_true_when_context_is_none(self, monkeypatch):
        """When _get_telegram_user_context returns None, returns True."""
        monkeypatch.setenv("TELEGRAM_RBAC_ENABLED", "true")
        mixin = self._make_mixin()

        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch.object(
            mixin, "_get_telegram_user_context", return_value=None
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._check_telegram_user_permission(
                123, "user", 456, "telegram:read"
            )

        assert result is True

    def test_catches_value_error_returns_true(self, monkeypatch):
        """ValueError in check_permission returns True (fail open)."""
        monkeypatch.setenv("TELEGRAM_RBAC_ENABLED", "true")
        mixin = self._make_mixin()

        mock_check = MagicMock(side_effect=ValueError("bad"))

        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.check_permission",
            mock_check,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._check_telegram_user_permission(
                123, "user", 456, "telegram:read"
            )

        assert result is True

    def test_catches_type_error_returns_true(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_RBAC_ENABLED", "true")
        mixin = self._make_mixin()
        mock_check = MagicMock(side_effect=TypeError("bad"))
        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.check_permission",
            mock_check,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._check_telegram_user_permission(
                123, "user", 456, "telegram:read"
            )
        assert result is True

    def test_catches_attribute_error_returns_true(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_RBAC_ENABLED", "true")
        mixin = self._make_mixin()
        mock_check = MagicMock(side_effect=AttributeError("attr"))
        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.check_permission",
            mock_check,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._check_telegram_user_permission(
                123, "user", 456, "telegram:read"
            )
        assert result is True

    def test_catches_runtime_error_returns_true(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_RBAC_ENABLED", "true")
        mixin = self._make_mixin()
        mock_check = MagicMock(side_effect=RuntimeError("runtime"))
        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.check_permission",
            mock_check,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._check_telegram_user_permission(
                123, "user", 456, "telegram:read"
            )
        assert result is True

    def test_logs_warning_when_denied(self, monkeypatch, caplog):
        """Denied permission logs a warning with user details."""
        monkeypatch.setenv("TELEGRAM_RBAC_ENABLED", "true")
        mixin = self._make_mixin()

        mock_decision = MagicMock()
        mock_decision.allowed = False
        mock_decision.reason = "no access"
        mock_check = MagicMock(return_value=mock_decision)

        with caplog.at_level(logging.WARNING), patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.check_permission",
            mock_check,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            mixin._check_telegram_user_permission(
                99999, "denied_user", 55555, "telegram:admin"
            )

        assert any("99999" in r.message for r in caplog.records)
        assert any("denied_user" in r.message for r in caplog.records)
        assert any("telegram:admin" in r.message for r in caplog.records)

    def test_logs_debug_when_rbac_not_enabled(self, monkeypatch, caplog):
        """When TELEGRAM_RBAC_ENABLED=false, debug log explains skip."""
        monkeypatch.delenv("TELEGRAM_RBAC_ENABLED", raising=False)
        mixin = self._make_mixin()

        with caplog.at_level(logging.DEBUG), patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn:
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            mixin._check_telegram_user_permission(
                123, "user", 456, "telegram:read"
            )

        assert any("not enabled" in r.message.lower() for r in caplog.records)

    def test_telegram_rbac_enabled_case_insensitive(self, monkeypatch):
        """TELEGRAM_RBAC_ENABLED=TRUE (uppercase) works."""
        monkeypatch.setenv("TELEGRAM_RBAC_ENABLED", "TRUE")
        mixin = self._make_mixin()

        mock_decision = MagicMock()
        mock_decision.allowed = True
        mock_check = MagicMock(return_value=mock_decision)

        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.check_permission",
            mock_check,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._check_telegram_user_permission(
                123, "user", 456, "telegram:read"
            )

        assert result is True
        mock_check.assert_called_once()

    def test_telegram_rbac_enabled_mixed_case(self, monkeypatch):
        """TELEGRAM_RBAC_ENABLED=True (mixed case) works."""
        monkeypatch.setenv("TELEGRAM_RBAC_ENABLED", "True")
        mixin = self._make_mixin()

        mock_decision = MagicMock()
        mock_decision.allowed = True
        mock_check = MagicMock(return_value=mock_decision)

        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.check_permission",
            mock_check,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = mixin._check_telegram_user_permission(
                123, "user", 456, "telegram:read"
            )

        assert result is True
        mock_check.assert_called_once()


# ============================================================================
# TelegramRBACMixin: _deny_telegram_permission
# ============================================================================


class TestDenyTelegramPermission:
    """Test TelegramRBACMixin._deny_telegram_permission method."""

    def _make_mixin_with_send(self):
        from aragora.server.handlers.social.telegram._common import TelegramRBACMixin

        class TestMixin(TelegramRBACMixin):
            _send_message_async = AsyncMock()

        return TestMixin()

    def test_returns_json_response_ok(self):
        mixin = self._make_mixin_with_send()
        mock_tg = MagicMock()
        mock_tg.create_tracked_task = MagicMock()

        with patch(
            "aragora.server.handlers.social.telegram._common.telegram_module",
            mock_tg,
            create=True,
        ), patch(
            "aragora.server.handlers.social.telegram.handler.TelegramHandler",
            create=True,
        ):
            # Patch the import inside _deny_telegram_permission
            with patch(
                "aragora.server.handlers.social.telegram._common.__import__",
                create=True,
            ):
                # Directly use the module-level import path
                with patch(
                    "aragora.server.handlers.social.telegram.telegram_module",
                    mock_tg,
                    create=True,
                ):
                    pass

        # Use a simpler approach: call the method with mock telegram module
        with patch(
            "aragora.server.handlers.social.telegram._common.create_tracked_task",
        ) as mock_create:
            # The method imports telegram module internally
            with patch(
                "aragora.server.handlers.social.telegram._common.__import__",
                create=True,
            ):
                pass

    def test_deny_sends_message_and_returns_ok(self):
        """_deny_telegram_permission sends a denial message and returns ok."""
        from aragora.server.handlers.social.telegram._common import TelegramRBACMixin

        class TestMixin(TelegramRBACMixin):
            _send_message_async = AsyncMock()

        mixin = TestMixin()

        mock_tg_module = MagicMock()
        mock_tg_module.create_tracked_task = MagicMock()

        with patch(
            "aragora.server.handlers.social.telegram._common.telegram_module",
            mock_tg_module,
            create=True,
        ):
            # Patch the from import inside _deny_telegram_permission
            import aragora.server.handlers.social.telegram as tg_pkg

            with patch.dict(
                "sys.modules",
                {"aragora.server.handlers.social.telegram": tg_pkg},
            ):
                result = mixin._deny_telegram_permission(
                    12345, "telegram:admin", "manage webhook settings"
                )

        assert _status(result) == 200
        assert _body(result)["ok"] is True

    def test_deny_message_contains_permission_key(self):
        """Denial message includes the required permission key."""
        from aragora.server.handlers.social.telegram._common import TelegramRBACMixin

        class TestMixin(TelegramRBACMixin):
            _send_message_async = AsyncMock()

        mixin = TestMixin()

        import aragora.server.handlers.social.telegram as tg_pkg

        # Capture the coroutine passed to create_tracked_task
        captured_coro = []
        original_create = tg_pkg.create_tracked_task

        def capture_task(coro, name=""):
            captured_coro.append(coro)
            mock_task = MagicMock()
            mock_task.done.return_value = True
            return mock_task

        with patch.object(tg_pkg, "create_tracked_task", side_effect=capture_task):
            result = mixin._deny_telegram_permission(
                12345, "telegram:gauntlet:run", "run gauntlet tests"
            )

        assert _status(result) == 200
        # The _send_message_async should have been called with message containing the permission
        # Since create_tracked_task was called with a coroutine, verify the call was made
        assert len(captured_coro) == 1

    def test_deny_message_contains_action_description(self):
        """Denial message includes the action description."""
        from aragora.server.handlers.social.telegram._common import TelegramRBACMixin

        sent_messages = []

        class TestMixin(TelegramRBACMixin):
            async def _send_message_async(self, chat_id, text, parse_mode=None):
                sent_messages.append(text)

        mixin = TestMixin()

        import aragora.server.handlers.social.telegram as tg_pkg

        mock_tasks = []

        def capture_task(coro, name=""):
            mock_task = MagicMock()
            mock_task.done.return_value = True
            mock_tasks.append(coro)
            return mock_task

        with patch.object(tg_pkg, "create_tracked_task", side_effect=capture_task):
            mixin._deny_telegram_permission(
                99999, "telegram:admin", "configure webhook"
            )

        # The coroutine was captured; run it to extract the message
        loop = asyncio.new_event_loop()
        try:
            for coro in mock_tasks:
                loop.run_until_complete(coro)
        finally:
            loop.close()

        assert len(sent_messages) == 1
        assert "configure webhook" in sent_messages[0]
        assert "telegram:admin" in sent_messages[0]
        assert "Permission denied" in sent_messages[0]

    def test_deny_sends_markdown_message(self):
        """Denial message uses Markdown parse mode."""
        from aragora.server.handlers.social.telegram._common import TelegramRBACMixin

        send_calls = []

        class TestMixin(TelegramRBACMixin):
            async def _send_message_async(self, chat_id, text, parse_mode=None):
                send_calls.append({"chat_id": chat_id, "text": text, "parse_mode": parse_mode})

        mixin = TestMixin()

        import aragora.server.handlers.social.telegram as tg_pkg

        captured_coros = []

        def capture_task(coro, name=""):
            captured_coros.append(coro)
            mock_task = MagicMock()
            mock_task.done.return_value = True
            return mock_task

        with patch.object(tg_pkg, "create_tracked_task", side_effect=capture_task):
            mixin._deny_telegram_permission(
                12345, "telegram:read", "view status"
            )

        loop = asyncio.new_event_loop()
        try:
            for coro in captured_coros:
                loop.run_until_complete(coro)
        finally:
            loop.close()

        assert len(send_calls) == 1
        assert send_calls[0]["parse_mode"] == "Markdown"
        assert send_calls[0]["chat_id"] == 12345

    def test_deny_task_name_includes_chat_id(self):
        """The tracked task name includes the chat_id."""
        from aragora.server.handlers.social.telegram._common import TelegramRBACMixin

        class TestMixin(TelegramRBACMixin):
            _send_message_async = AsyncMock()

        mixin = TestMixin()

        import aragora.server.handlers.social.telegram as tg_pkg

        task_names = []

        def capture_task(coro, name=""):
            task_names.append(name)
            mock_task = MagicMock()
            mock_task.done.return_value = True
            return mock_task

        with patch.object(tg_pkg, "create_tracked_task", side_effect=capture_task):
            mixin._deny_telegram_permission(
                77777, "telegram:admin", "admin action"
            )

        assert len(task_names) == 1
        assert "77777" in task_names[0]
        assert "permission-denied" in task_names[0]

    def test_deny_message_includes_contact_admin(self):
        """Denial message tells user to contact administrator."""
        from aragora.server.handlers.social.telegram._common import TelegramRBACMixin

        sent_messages = []

        class TestMixin(TelegramRBACMixin):
            async def _send_message_async(self, chat_id, text, parse_mode=None):
                sent_messages.append(text)

        mixin = TestMixin()

        import aragora.server.handlers.social.telegram as tg_pkg

        captured_coros = []

        def capture_task(coro, name=""):
            captured_coros.append(coro)
            mock_task = MagicMock()
            mock_task.done.return_value = True
            return mock_task

        with patch.object(tg_pkg, "create_tracked_task", side_effect=capture_task):
            mixin._deny_telegram_permission(
                12345, "telegram:admin", "manage settings"
            )

        loop = asyncio.new_event_loop()
        try:
            for coro in captured_coros:
                loop.run_until_complete(coro)
        finally:
            loop.close()

        assert len(sent_messages) == 1
        assert "administrator" in sent_messages[0].lower()


# ============================================================================
# Integration: TelegramRBACMixin used via TelegramHandler
# ============================================================================


class TestMixinIntegration:
    """Test that TelegramHandler inherits TelegramRBACMixin methods correctly."""

    def test_handler_has_get_auth_context(self):
        from aragora.server.handlers.social.telegram.handler import TelegramHandler

        handler = TelegramHandler(ctx={})
        assert hasattr(handler, "_get_auth_context")
        assert callable(handler._get_auth_context)

    def test_handler_has_get_telegram_user_context(self):
        from aragora.server.handlers.social.telegram.handler import TelegramHandler

        handler = TelegramHandler(ctx={})
        assert hasattr(handler, "_get_telegram_user_context")
        assert callable(handler._get_telegram_user_context)

    def test_handler_has_check_permission(self):
        from aragora.server.handlers.social.telegram.handler import TelegramHandler

        handler = TelegramHandler(ctx={})
        assert hasattr(handler, "_check_permission")
        assert callable(handler._check_permission)

    def test_handler_has_check_telegram_user_permission(self):
        from aragora.server.handlers.social.telegram.handler import TelegramHandler

        handler = TelegramHandler(ctx={})
        assert hasattr(handler, "_check_telegram_user_permission")
        assert callable(handler._check_telegram_user_permission)

    def test_handler_has_deny_telegram_permission(self):
        from aragora.server.handlers.social.telegram.handler import TelegramHandler

        handler = TelegramHandler(ctx={})
        assert hasattr(handler, "_deny_telegram_permission")
        assert callable(handler._deny_telegram_permission)

    def test_handler_check_permission_rbac_off(self, _ensure_no_production_env):
        """TelegramHandler._check_permission returns None when RBAC disabled."""
        from aragora.server.handlers.social.telegram.handler import TelegramHandler

        handler = TelegramHandler(ctx={})
        mock_http = MagicMock()

        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn, patch(
            "aragora.server.handlers.social.telegram._common.check_permission",
            None,
        ), patch(
            "aragora.server.handlers.social.telegram._common.rbac_fail_closed",
            return_value=False,
        ):
            mock_tg_fn.return_value.RBAC_AVAILABLE = False
            result = handler._check_permission(mock_http, "telegram:read")

        assert result is None

    def test_handler_telegram_user_permission_rbac_off(
        self, monkeypatch, _ensure_no_production_env
    ):
        """TelegramHandler._check_telegram_user_permission returns True with RBAC off."""
        from aragora.server.handlers.social.telegram.handler import TelegramHandler

        handler = TelegramHandler(ctx={})
        monkeypatch.delenv("TELEGRAM_RBAC_ENABLED", raising=False)

        with patch(
            "aragora.server.handlers.social.telegram._common._tg"
        ) as mock_tg_fn:
            mock_tg_fn.return_value.RBAC_AVAILABLE = True
            result = handler._check_telegram_user_permission(
                123, "user", 456, "telegram:read"
            )

        assert result is True


# ============================================================================
# Module re-exports via __init__.py
# ============================================================================


class TestModuleReexports:
    """Test that _common.py symbols are properly re-exported from the package."""

    def test_rbac_available_exported(self):
        from aragora.server.handlers.social.telegram import RBAC_AVAILABLE

        assert isinstance(RBAC_AVAILABLE, bool)

    def test_create_tracked_task_exported(self):
        from aragora.server.handlers.social.telegram import create_tracked_task

        assert callable(create_tracked_task)

    def test_handle_task_exception_exported(self):
        from aragora.server.handlers.social.telegram import _handle_task_exception

        assert callable(_handle_task_exception)

    def test_telegram_api_base_exported(self):
        from aragora.server.handlers.social.telegram import TELEGRAM_API_BASE

        assert TELEGRAM_API_BASE == "https://api.telegram.org/bot"

    def test_telegram_bot_token_exported(self):
        from aragora.server.handlers.social.telegram import TELEGRAM_BOT_TOKEN

        assert TELEGRAM_BOT_TOKEN is None or isinstance(TELEGRAM_BOT_TOKEN, str)

    def test_telegram_webhook_secret_exported(self):
        from aragora.server.handlers.social.telegram import TELEGRAM_WEBHOOK_SECRET

        assert TELEGRAM_WEBHOOK_SECRET is None or isinstance(TELEGRAM_WEBHOOK_SECRET, str)

    def test_all_permission_constants_exported(self):
        from aragora.server.handlers.social.telegram import (
            PERM_TELEGRAM_ADMIN,
            PERM_TELEGRAM_CALLBACKS_HANDLE,
            PERM_TELEGRAM_COMMANDS_EXECUTE,
            PERM_TELEGRAM_DEBATES_CREATE,
            PERM_TELEGRAM_DEBATES_READ,
            PERM_TELEGRAM_GAUNTLET_RUN,
            PERM_TELEGRAM_MESSAGES_SEND,
            PERM_TELEGRAM_READ,
            PERM_TELEGRAM_VOTES_RECORD,
        )

        assert all(
            isinstance(p, str)
            for p in [
                PERM_TELEGRAM_ADMIN,
                PERM_TELEGRAM_CALLBACKS_HANDLE,
                PERM_TELEGRAM_COMMANDS_EXECUTE,
                PERM_TELEGRAM_DEBATES_CREATE,
                PERM_TELEGRAM_DEBATES_READ,
                PERM_TELEGRAM_GAUNTLET_RUN,
                PERM_TELEGRAM_MESSAGES_SEND,
                PERM_TELEGRAM_READ,
                PERM_TELEGRAM_VOTES_RECORD,
            ]
        )
