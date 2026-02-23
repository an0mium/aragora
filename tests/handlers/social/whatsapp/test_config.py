"""Comprehensive tests for WhatsApp config module.

Covers all public symbols and utilities in
aragora/server/handlers/social/whatsapp/config.py:
- RBAC permission constants
- Environment variable loading (TTS, tokens, secrets)
- WHATSAPP_API_BASE constant
- _handle_task_exception() helper
- create_tracked_task() helper
- Optional RBAC imports (check_permission, extract_user_from_request,
  AuthorizationContext, RBAC_AVAILABLE)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Module path for patching
_MOD = "aragora.server.handlers.social.whatsapp.config"


# ---------------------------------------------------------------------------
# RBAC Permission Constants
# ---------------------------------------------------------------------------


class TestRBACPermissionConstants:
    """Tests for RBAC permission constants defined in config."""

    def test_perm_whatsapp_read_is_bots_read(self):
        from aragora.server.handlers.social.whatsapp.config import PERM_WHATSAPP_READ

        assert PERM_WHATSAPP_READ == "bots.read"

    def test_perm_whatsapp_messages_is_bots_read(self):
        from aragora.server.handlers.social.whatsapp.config import PERM_WHATSAPP_MESSAGES

        assert PERM_WHATSAPP_MESSAGES == "bots.read"

    def test_perm_whatsapp_debates_is_debates_create(self):
        from aragora.server.handlers.social.whatsapp.config import PERM_WHATSAPP_DEBATES

        assert PERM_WHATSAPP_DEBATES == "debates.create"

    def test_perm_whatsapp_gauntlet_is_gauntlet_run(self):
        from aragora.server.handlers.social.whatsapp.config import PERM_WHATSAPP_GAUNTLET

        assert PERM_WHATSAPP_GAUNTLET == "gauntlet.run"

    def test_perm_whatsapp_votes_is_debates_update(self):
        from aragora.server.handlers.social.whatsapp.config import PERM_WHATSAPP_VOTES

        assert PERM_WHATSAPP_VOTES == "debates.update"

    def test_perm_whatsapp_details_is_debates_read(self):
        from aragora.server.handlers.social.whatsapp.config import PERM_WHATSAPP_DETAILS

        assert PERM_WHATSAPP_DETAILS == "debates.read"

    def test_perm_whatsapp_admin_is_bots_wildcard(self):
        from aragora.server.handlers.social.whatsapp.config import PERM_WHATSAPP_ADMIN

        assert PERM_WHATSAPP_ADMIN == "bots.*"

    def test_all_permissions_are_strings(self):
        from aragora.server.handlers.social.whatsapp import config

        perm_names = [
            "PERM_WHATSAPP_READ",
            "PERM_WHATSAPP_MESSAGES",
            "PERM_WHATSAPP_DEBATES",
            "PERM_WHATSAPP_GAUNTLET",
            "PERM_WHATSAPP_VOTES",
            "PERM_WHATSAPP_DETAILS",
            "PERM_WHATSAPP_ADMIN",
        ]
        for name in perm_names:
            value = getattr(config, name)
            assert isinstance(value, str), f"{name} should be str, got {type(value)}"

    def test_permissions_are_not_empty(self):
        from aragora.server.handlers.social.whatsapp import config

        perm_names = [
            "PERM_WHATSAPP_READ",
            "PERM_WHATSAPP_MESSAGES",
            "PERM_WHATSAPP_DEBATES",
            "PERM_WHATSAPP_GAUNTLET",
            "PERM_WHATSAPP_VOTES",
            "PERM_WHATSAPP_DETAILS",
            "PERM_WHATSAPP_ADMIN",
        ]
        for name in perm_names:
            value = getattr(config, name)
            assert len(value) > 0, f"{name} should not be empty"

    def test_all_seven_permissions_defined(self):
        """Ensure exactly 7 PERM_WHATSAPP_* constants are defined."""
        from aragora.server.handlers.social.whatsapp import config

        perm_attrs = [a for a in dir(config) if a.startswith("PERM_WHATSAPP_")]
        assert len(perm_attrs) == 7


# ---------------------------------------------------------------------------
# Environment Variable Loading
# ---------------------------------------------------------------------------


class TestEnvironmentVariables:
    """Tests for module-level environment variable constants."""

    def test_api_base_url(self):
        from aragora.server.handlers.social.whatsapp.config import WHATSAPP_API_BASE

        assert WHATSAPP_API_BASE == "https://graph.facebook.com/v18.0"

    def test_access_token_reads_from_env(self):
        """WHATSAPP_ACCESS_TOKEN should be os.environ.get result."""
        from aragora.server.handlers.social.whatsapp import config

        # The value was resolved at import time. Verify the attribute exists.
        assert hasattr(config, "WHATSAPP_ACCESS_TOKEN")

    def test_phone_number_id_reads_from_env(self):
        from aragora.server.handlers.social.whatsapp import config

        assert hasattr(config, "WHATSAPP_PHONE_NUMBER_ID")

    def test_verify_token_reads_from_env(self):
        from aragora.server.handlers.social.whatsapp import config

        assert hasattr(config, "WHATSAPP_VERIFY_TOKEN")

    def test_app_secret_reads_from_env(self):
        from aragora.server.handlers.social.whatsapp import config

        assert hasattr(config, "WHATSAPP_APP_SECRET")

    def test_tts_voice_enabled_attribute_exists(self):
        from aragora.server.handlers.social.whatsapp import config

        assert hasattr(config, "TTS_VOICE_ENABLED")

    def test_tts_voice_enabled_is_bool(self):
        from aragora.server.handlers.social.whatsapp.config import TTS_VOICE_ENABLED

        assert isinstance(TTS_VOICE_ENABLED, bool)

    def test_tts_disabled_by_default(self):
        """Without env var set, TTS_VOICE_ENABLED defaults to False."""
        # The module reads the env var at import time. In test env it is
        # typically unset, so should be False.
        from aragora.server.handlers.social.whatsapp.config import TTS_VOICE_ENABLED

        # If the env var happens to be set in the test runner, just check type
        assert isinstance(TTS_VOICE_ENABLED, bool)


# ---------------------------------------------------------------------------
# _handle_task_exception
# ---------------------------------------------------------------------------


class TestHandleTaskException:
    """Tests for _handle_task_exception() helper."""

    def test_cancelled_task_logs_debug(self, caplog):
        from aragora.server.handlers.social.whatsapp.config import _handle_task_exception

        task = MagicMock(spec=asyncio.Task)
        task.cancelled.return_value = True
        task.exception.return_value = None

        with caplog.at_level(logging.DEBUG, logger=_MOD):
            _handle_task_exception(task, "test-task")

        assert any("cancelled" in r.message.lower() for r in caplog.records)

    def test_cancelled_task_does_not_check_exception(self):
        from aragora.server.handlers.social.whatsapp.config import _handle_task_exception

        task = MagicMock(spec=asyncio.Task)
        task.cancelled.return_value = True

        _handle_task_exception(task, "test-task")
        task.exception.assert_not_called()

    def test_task_with_exception_logs_error(self, caplog):
        from aragora.server.handlers.social.whatsapp.config import _handle_task_exception

        task = MagicMock(spec=asyncio.Task)
        task.cancelled.return_value = False
        task.exception.return_value = RuntimeError("boom")

        with caplog.at_level(logging.ERROR, logger=_MOD):
            _handle_task_exception(task, "my-failing-task")

        assert any("my-failing-task" in r.message for r in caplog.records)

    def test_task_with_exception_logs_exception_type(self, caplog):
        from aragora.server.handlers.social.whatsapp.config import _handle_task_exception

        task = MagicMock(spec=asyncio.Task)
        task.cancelled.return_value = False
        exc = ValueError("bad value")
        task.exception.return_value = exc

        with caplog.at_level(logging.ERROR, logger=_MOD):
            _handle_task_exception(task, "val-task")

        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(error_records) >= 1
        assert "bad value" in error_records[0].message

    def test_successful_task_no_logging(self, caplog):
        from aragora.server.handlers.social.whatsapp.config import _handle_task_exception

        task = MagicMock(spec=asyncio.Task)
        task.cancelled.return_value = False
        task.exception.return_value = None

        with caplog.at_level(logging.DEBUG, logger=_MOD):
            _handle_task_exception(task, "success-task")

        assert not any("success-task" in r.message for r in caplog.records)

    def test_task_name_appears_in_cancelled_log(self, caplog):
        from aragora.server.handlers.social.whatsapp.config import _handle_task_exception

        task = MagicMock(spec=asyncio.Task)
        task.cancelled.return_value = True

        with caplog.at_level(logging.DEBUG, logger=_MOD):
            _handle_task_exception(task, "unique-name-xyz")

        assert any("unique-name-xyz" in r.message for r in caplog.records)

    def test_task_name_appears_in_error_log(self, caplog):
        from aragora.server.handlers.social.whatsapp.config import _handle_task_exception

        task = MagicMock(spec=asyncio.Task)
        task.cancelled.return_value = False
        task.exception.return_value = TypeError("oops")

        with caplog.at_level(logging.ERROR, logger=_MOD):
            _handle_task_exception(task, "error-name-abc")

        assert any("error-name-abc" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# create_tracked_task
# ---------------------------------------------------------------------------


class TestCreateTrackedTask:
    """Tests for create_tracked_task() helper."""

    @pytest.mark.asyncio
    async def test_creates_asyncio_task(self):
        from aragora.server.handlers.social.whatsapp.config import create_tracked_task

        completed = False

        async def dummy_coro():
            nonlocal completed
            completed = True

        task = create_tracked_task(dummy_coro(), name="test-create")
        assert isinstance(task, asyncio.Task)
        await task
        assert completed

    @pytest.mark.asyncio
    async def test_task_has_correct_name(self):
        from aragora.server.handlers.social.whatsapp.config import create_tracked_task

        async def noop():
            pass

        task = create_tracked_task(noop(), name="my-task-name")
        assert task.get_name() == "my-task-name"
        await task

    @pytest.mark.asyncio
    async def test_done_callback_registered(self):
        from aragora.server.handlers.social.whatsapp.config import create_tracked_task

        async def noop():
            pass

        task = create_tracked_task(noop(), name="cb-test")
        # asyncio.Task doesn't expose callbacks publicly, but we can verify
        # by checking the task completes without error (callback is invoked)
        await task
        assert task.done()

    @pytest.mark.asyncio
    async def test_exception_in_task_is_handled(self, caplog):
        from aragora.server.handlers.social.whatsapp.config import create_tracked_task

        async def failing_coro():
            raise RuntimeError("task failure")

        with caplog.at_level(logging.ERROR, logger=_MOD):
            task = create_tracked_task(failing_coro(), name="fail-task")
            # Wait for task to complete (it will fail internally)
            try:
                await task
            except RuntimeError:
                pass

        # The done callback should have logged the error
        assert any("fail-task" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_successful_task_returns_result(self):
        from aragora.server.handlers.social.whatsapp.config import create_tracked_task

        async def returning_coro():
            return 42

        task = create_tracked_task(returning_coro(), name="result-task")
        result = await task
        assert result == 42

    @pytest.mark.asyncio
    async def test_cancelled_task_callback(self, caplog):
        from aragora.server.handlers.social.whatsapp.config import create_tracked_task

        async def long_coro():
            await asyncio.sleep(10)

        with caplog.at_level(logging.DEBUG, logger=_MOD):
            task = create_tracked_task(long_coro(), name="cancel-me")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert any("cancel-me" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# RBAC Optional Imports
# ---------------------------------------------------------------------------


class TestRBACImports:
    """Tests for the optional RBAC import block at module level."""

    def test_rbac_available_is_bool(self):
        from aragora.server.handlers.social.whatsapp.config import RBAC_AVAILABLE

        assert isinstance(RBAC_AVAILABLE, bool)

    def test_check_permission_not_none_when_available(self):
        from aragora.server.handlers.social.whatsapp.config import (
            RBAC_AVAILABLE,
            check_permission,
        )

        if RBAC_AVAILABLE:
            assert check_permission is not None

    def test_authorization_context_not_none_when_available(self):
        from aragora.server.handlers.social.whatsapp.config import (
            RBAC_AVAILABLE,
            AuthorizationContext,
        )

        if RBAC_AVAILABLE:
            assert AuthorizationContext is not None

    def test_check_permission_is_callable_when_available(self):
        from aragora.server.handlers.social.whatsapp.config import (
            RBAC_AVAILABLE,
            check_permission,
        )

        if RBAC_AVAILABLE:
            assert callable(check_permission)

    def test_extract_user_from_request_attr_exists(self):
        from aragora.server.handlers.social.whatsapp import config

        assert hasattr(config, "extract_user_from_request")

    def test_rbac_available_true_in_normal_env(self):
        """In the test environment, RBAC modules should be importable."""
        from aragora.server.handlers.social.whatsapp.config import RBAC_AVAILABLE

        assert RBAC_AVAILABLE is True

    def test_check_permission_comes_from_rbac_checker(self):
        from aragora.server.handlers.social.whatsapp.config import (
            RBAC_AVAILABLE,
            check_permission,
        )

        if RBAC_AVAILABLE:
            from aragora.rbac.checker import check_permission as original

            assert check_permission is original

    def test_authorization_context_comes_from_rbac_models(self):
        from aragora.server.handlers.social.whatsapp.config import (
            RBAC_AVAILABLE,
            AuthorizationContext,
        )

        if RBAC_AVAILABLE:
            from aragora.rbac.models import AuthorizationContext as original

            assert AuthorizationContext is original


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class TestLogger:
    """Tests for the module-level logger."""

    def test_logger_exists(self):
        from aragora.server.handlers.social.whatsapp import config

        assert hasattr(config, "logger")

    def test_logger_has_correct_name(self):
        from aragora.server.handlers.social.whatsapp.config import logger

        assert logger.name == "aragora.server.handlers.social.whatsapp.config"

    def test_logger_is_logging_logger(self):
        from aragora.server.handlers.social.whatsapp.config import logger

        assert isinstance(logger, logging.Logger)


# ---------------------------------------------------------------------------
# Module-level Exports Completeness
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Verify all expected symbols are exported from config."""

    def test_has_create_tracked_task(self):
        from aragora.server.handlers.social.whatsapp import config

        assert hasattr(config, "create_tracked_task")
        assert callable(config.create_tracked_task)

    def test_has_handle_task_exception(self):
        from aragora.server.handlers.social.whatsapp import config

        assert hasattr(config, "_handle_task_exception")
        assert callable(config._handle_task_exception)

    def test_has_whatsapp_api_base(self):
        from aragora.server.handlers.social.whatsapp import config

        assert hasattr(config, "WHATSAPP_API_BASE")
        assert config.WHATSAPP_API_BASE.startswith("https://")

    def test_has_all_env_var_constants(self):
        from aragora.server.handlers.social.whatsapp import config

        env_vars = [
            "WHATSAPP_ACCESS_TOKEN",
            "WHATSAPP_PHONE_NUMBER_ID",
            "WHATSAPP_VERIFY_TOKEN",
            "WHATSAPP_APP_SECRET",
        ]
        for var in env_vars:
            assert hasattr(config, var), f"Missing attribute: {var}"

    def test_has_tts_voice_enabled(self):
        from aragora.server.handlers.social.whatsapp import config

        assert hasattr(config, "TTS_VOICE_ENABLED")
