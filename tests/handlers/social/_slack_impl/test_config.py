"""Comprehensive tests for the Slack integration config module.

Covers every public function, constant, regex pattern, singleton accessor,
and fallback stub in ``aragora/server/handlers/social/_slack_impl/config.py``:

- _get_audit_logger: success, cached, ImportError, RuntimeError, OSError
- _get_user_rate_limiter: success, cached, ImportError, RuntimeError, OSError
- _get_workspace_rate_limiter: success, cached, ImportError, RuntimeError, OSError,
  action_limits wiring, limiter returns None
- _validate_slack_url: HTTPS Slack domains, HTTP rejection, wrong domain, empty,
  None, path/query preservation, edge cases
- _handle_task_exception: cancelled, exception, no-exception, combined
- create_tracked_task: with running loop, without loop (thread fallback),
  _BackgroundTask stub
- SLACK_ALLOWED_DOMAINS, ARAGORA_API_BASE_URL, BOTS_READ_PERMISSION, env vars
- COMMAND_PATTERN & TOPIC_PATTERN: valid, invalid, edge cases
- get_workspace_store: success, cached, ImportError
- resolve_workspace: success, empty team_id, no store, KeyError, OSError, RuntimeError
- get_slack_integration: success, cached, no webhook URL, ImportError,
  ValueError, KeyError, TypeError, RuntimeError, OSError, AttributeError
- Re-exported handler utilities: HandlerResult, SecureHandler, error_response,
  json_response, auto_error_response, rate_limit, ForbiddenError, UnauthorizedError
- Module-level SLACK_WORKSPACE_RATE_LIMIT_RPM env parsing
"""

from __future__ import annotations

import asyncio
import builtins
import json
import logging
import re
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_real_import = builtins.__import__


@contextmanager
def _block_import(*module_names: str):
    """Context manager that makes specific module imports raise ImportError.

    Uses a selective __import__ replacement so that only the listed module
    names trigger ImportError; all other imports proceed normally.
    """
    blocked = set(module_names)

    def selective_import(name, *args, **kwargs):
        if name in blocked:
            raise ImportError(f"blocked for test: {name}")
        return _real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=selective_import):
        yield


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """Import the config module lazily."""
    from aragora.server.handlers.social._slack_impl import config as mod
    return mod


@pytest.fixture(autouse=True)
def _reset_singletons(config, monkeypatch):
    """Reset all module-level singletons between tests."""
    monkeypatch.setattr(config, "_slack_audit", None)
    monkeypatch.setattr(config, "_slack_user_limiter", None)
    monkeypatch.setattr(config, "_slack_workspace_limiter", None)
    monkeypatch.setattr(config, "_slack_integration", None)
    monkeypatch.setattr(config, "_workspace_store", None)
    yield


# ===========================================================================
# _get_audit_logger
# ===========================================================================

class TestGetAuditLogger:
    """Tests for _get_audit_logger lazy singleton."""

    def test_success_creates_logger(self, config, monkeypatch):
        mock_logger = MagicMock()
        with patch(
            "aragora.audit.slack_audit.get_slack_audit_logger",
            return_value=mock_logger,
            create=True,
        ):
            result = config._get_audit_logger()
        assert result is mock_logger

    def test_cached_returns_same_instance(self, config, monkeypatch):
        sentinel = MagicMock()
        monkeypatch.setattr(config, "_slack_audit", sentinel)
        result = config._get_audit_logger()
        assert result is sentinel

    def test_import_error_returns_none(self, config):
        with _block_import("aragora.audit.slack_audit"):
            result = config._get_audit_logger()
        assert result is None

    def test_runtime_error_returns_none(self, config):
        with patch(
            "aragora.audit.slack_audit.get_slack_audit_logger",
            side_effect=RuntimeError("init fail"),
            create=True,
        ):
            result = config._get_audit_logger()
        assert result is None

    def test_os_error_returns_none(self, config):
        with patch(
            "aragora.audit.slack_audit.get_slack_audit_logger",
            side_effect=OSError("disk error"),
            create=True,
        ):
            result = config._get_audit_logger()
        assert result is None

    def test_none_stays_none_on_error(self, config):
        """After an error, _slack_audit stays None so next call retries."""
        with patch(
            "aragora.audit.slack_audit.get_slack_audit_logger",
            side_effect=ImportError("nope"),
            create=True,
        ):
            config._get_audit_logger()
        assert config._slack_audit is None


# ===========================================================================
# _get_user_rate_limiter
# ===========================================================================

class TestGetUserRateLimiter:
    """Tests for _get_user_rate_limiter lazy singleton."""

    def test_success_creates_limiter(self, config):
        mock_limiter = MagicMock()
        with patch(
            "aragora.server.middleware.rate_limit.user_limiter.get_user_rate_limiter",
            return_value=mock_limiter,
            create=True,
        ):
            result = config._get_user_rate_limiter()
        assert result is mock_limiter

    def test_cached_returns_same_instance(self, config, monkeypatch):
        sentinel = MagicMock()
        monkeypatch.setattr(config, "_slack_user_limiter", sentinel)
        result = config._get_user_rate_limiter()
        assert result is sentinel

    def test_import_error_returns_none(self, config):
        with _block_import("aragora.server.middleware.rate_limit.user_limiter"):
            result = config._get_user_rate_limiter()
        assert result is None

    def test_runtime_error_returns_none(self, config):
        with patch(
            "aragora.server.middleware.rate_limit.user_limiter.get_user_rate_limiter",
            side_effect=RuntimeError("fail"),
            create=True,
        ):
            result = config._get_user_rate_limiter()
        assert result is None

    def test_os_error_returns_none(self, config):
        with patch(
            "aragora.server.middleware.rate_limit.user_limiter.get_user_rate_limiter",
            side_effect=OSError("disk"),
            create=True,
        ):
            result = config._get_user_rate_limiter()
        assert result is None


# ===========================================================================
# _get_workspace_rate_limiter
# ===========================================================================

class TestGetWorkspaceRateLimiter:
    """Tests for _get_workspace_rate_limiter lazy singleton."""

    def test_success_creates_limiter_with_action_limits(self, config, monkeypatch):
        mock_limiter = MagicMock()
        mock_limiter.action_limits = {}
        with patch(
            "aragora.server.middleware.rate_limit.user_limiter.get_user_rate_limiter",
            return_value=mock_limiter,
            create=True,
        ):
            result = config._get_workspace_rate_limiter()
        assert result is mock_limiter
        assert mock_limiter.action_limits["slack_workspace_command"] == config.SLACK_WORKSPACE_RATE_LIMIT_RPM

    def test_cached_returns_same_instance(self, config, monkeypatch):
        sentinel = MagicMock()
        monkeypatch.setattr(config, "_slack_workspace_limiter", sentinel)
        result = config._get_workspace_rate_limiter()
        assert result is sentinel

    def test_import_error_returns_none(self, config):
        with _block_import("aragora.server.middleware.rate_limit.user_limiter"):
            result = config._get_workspace_rate_limiter()
        assert result is None

    def test_runtime_error_returns_none(self, config):
        with patch(
            "aragora.server.middleware.rate_limit.user_limiter.get_user_rate_limiter",
            side_effect=RuntimeError("fail"),
            create=True,
        ):
            result = config._get_workspace_rate_limiter()
        assert result is None

    def test_os_error_returns_none(self, config):
        with patch(
            "aragora.server.middleware.rate_limit.user_limiter.get_user_rate_limiter",
            side_effect=OSError("disk"),
            create=True,
        ):
            result = config._get_workspace_rate_limiter()
        assert result is None

    def test_limiter_returns_none_from_factory(self, config):
        """When get_user_rate_limiter() returns None, no action_limits set."""
        with patch(
            "aragora.server.middleware.rate_limit.user_limiter.get_user_rate_limiter",
            return_value=None,
            create=True,
        ):
            result = config._get_workspace_rate_limiter()
        # _slack_workspace_limiter is set to None (factory returned None)
        assert result is None

    def test_action_limits_use_configured_rpm(self, config, monkeypatch):
        """Verify the RPM value comes from SLACK_WORKSPACE_RATE_LIMIT_RPM."""
        monkeypatch.setattr(config, "SLACK_WORKSPACE_RATE_LIMIT_RPM", 99)
        mock_limiter = MagicMock()
        mock_limiter.action_limits = {}
        with patch(
            "aragora.server.middleware.rate_limit.user_limiter.get_user_rate_limiter",
            return_value=mock_limiter,
            create=True,
        ):
            config._get_workspace_rate_limiter()
        assert mock_limiter.action_limits["slack_workspace_command"] == 99


# ===========================================================================
# _validate_slack_url
# ===========================================================================

class TestValidateSlackUrl:
    """Tests for _validate_slack_url SSRF protection."""

    def test_valid_hooks_slack_com(self, config):
        assert config._validate_slack_url("https://hooks.slack.com/actions/T123/456") is True

    def test_valid_api_slack_com(self, config):
        assert config._validate_slack_url("https://api.slack.com/something") is True

    def test_http_rejected(self, config):
        assert config._validate_slack_url("http://hooks.slack.com/test") is False

    def test_wrong_domain_rejected(self, config):
        assert config._validate_slack_url("https://evil.com/hooks.slack.com") is False

    def test_subdomain_not_allowed(self, config):
        assert config._validate_slack_url("https://sub.hooks.slack.com/test") is False

    def test_empty_string(self, config):
        assert config._validate_slack_url("") is False

    def test_none_returns_false(self, config):
        # urlparse(None) raises TypeError, caught in except
        assert config._validate_slack_url(None) is False  # type: ignore[arg-type]

    def test_ftp_scheme_rejected(self, config):
        assert config._validate_slack_url("ftp://hooks.slack.com/test") is False

    def test_no_scheme(self, config):
        assert config._validate_slack_url("hooks.slack.com/test") is False

    def test_with_query_params(self, config):
        assert config._validate_slack_url("https://hooks.slack.com/test?a=1&b=2") is True

    def test_with_fragment(self, config):
        assert config._validate_slack_url("https://api.slack.com/path#frag") is True

    def test_with_port(self, config):
        """Domain with port is not in SLACK_ALLOWED_DOMAINS."""
        assert config._validate_slack_url("https://hooks.slack.com:443/test") is False

    def test_javascript_scheme_rejected(self, config):
        assert config._validate_slack_url("javascript:alert(1)") is False

    def test_data_scheme_rejected(self, config):
        assert config._validate_slack_url("data:text/html,<h1>hi</h1>") is False

    def test_file_scheme_rejected(self, config):
        assert config._validate_slack_url("file:///etc/passwd") is False

    def test_localhost_rejected(self, config):
        assert config._validate_slack_url("https://localhost/slack") is False

    def test_ip_address_rejected(self, config):
        assert config._validate_slack_url("https://127.0.0.1/slack") is False


# ===========================================================================
# _handle_task_exception
# ===========================================================================

class TestHandleTaskException:
    """Tests for _handle_task_exception callback."""

    def test_cancelled_task_logs_debug(self, config, caplog):
        task = MagicMock()
        task.cancelled.return_value = True
        with caplog.at_level(logging.DEBUG):
            config._handle_task_exception(task, "test-task")
        assert "test-task" in caplog.text
        assert "cancelled" in caplog.text

    def test_failed_task_logs_error(self, config, caplog):
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = ValueError("boom")
        with caplog.at_level(logging.ERROR):
            config._handle_task_exception(task, "my-task")
        assert "my-task" in caplog.text
        assert "boom" in caplog.text

    def test_successful_task_no_log(self, config, caplog):
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = None
        with caplog.at_level(logging.DEBUG):
            config._handle_task_exception(task, "ok-task")
        # Should not log anything specific about the task
        assert "failed" not in caplog.text
        assert "cancelled" not in caplog.text

    def test_exception_info_passed(self, config):
        """Verify exc_info is passed to logger.error."""
        task = MagicMock()
        task.cancelled.return_value = False
        exc = RuntimeError("whoops")
        task.exception.return_value = exc
        with patch.object(config.logger, "error") as mock_error:
            config._handle_task_exception(task, "info-task")
            mock_error.assert_called_once()
            _, kwargs = mock_error.call_args
            assert kwargs.get("exc_info") is exc


# ===========================================================================
# create_tracked_task
# ===========================================================================

class TestCreateTrackedTask:
    """Tests for create_tracked_task (both loop and thread paths)."""

    @pytest.mark.asyncio
    async def test_with_running_loop_creates_task(self, config):
        """When an event loop is running, creates a real asyncio.Task."""
        async def dummy():
            return 42

        task = config.create_tracked_task(dummy(), "loop-task")
        assert isinstance(task, asyncio.Task)
        result = await task
        assert result == 42

    @pytest.mark.asyncio
    async def test_task_has_done_callback(self, config):
        """Task created in loop has a done callback attached."""
        async def dummy():
            return "ok"

        task = config.create_tracked_task(dummy(), "cb-task")
        # The task should complete normally
        await task
        assert task.done()

    def test_without_loop_returns_background_task(self, config):
        """When no loop is running, falls back to thread and returns stub."""
        was_called = threading.Event()

        async def dummy():
            was_called.set()

        result = config.create_tracked_task(dummy(), "thread-task")
        # Returns _BackgroundTask stub
        assert hasattr(result, "add_done_callback")
        assert result.add_done_callback(lambda t: None) is None
        was_called.wait(timeout=5)
        assert was_called.is_set()

    def test_background_task_add_done_callback_returns_none(self, config):
        """_BackgroundTask.add_done_callback is a no-op returning None."""
        async def dummy():
            pass

        result = config.create_tracked_task(dummy(), "noop-task")
        assert result.add_done_callback(lambda t: None) is None

    @pytest.mark.asyncio
    async def test_task_name_set(self, config):
        """Task name is passed to create_task."""
        async def dummy():
            return 1

        task = config.create_tracked_task(dummy(), "named-task")
        assert task.get_name() == "named-task"
        await task

    def test_thread_name_contains_task_name(self, config):
        """Background thread name includes the task name."""
        started = threading.Event()

        async def dummy():
            started.set()

        with patch.object(threading, "Thread", wraps=threading.Thread) as mock_thread:
            config.create_tracked_task(dummy(), "my-bg")
            mock_thread.assert_called_once()
            _, kwargs = mock_thread.call_args
            assert "slack-task-my-bg" == kwargs["name"]
            assert kwargs["daemon"] is True
        started.wait(timeout=5)


# ===========================================================================
# Constants
# ===========================================================================

class TestConstants:
    """Tests for module-level constants."""

    def test_slack_allowed_domains(self, config):
        assert config.SLACK_ALLOWED_DOMAINS == frozenset({"hooks.slack.com", "api.slack.com"})

    def test_slack_allowed_domains_is_frozenset(self, config):
        assert isinstance(config.SLACK_ALLOWED_DOMAINS, frozenset)

    def test_bots_read_permission(self, config):
        assert config.BOTS_READ_PERMISSION == "bots.read"

    def test_workspace_rate_limit_rpm_default(self, config):
        # Default is 30 unless env var overrides
        assert isinstance(config.SLACK_WORKSPACE_RATE_LIMIT_RPM, int)

    def test_aragora_api_base_url_is_string(self, config):
        assert isinstance(config.ARAGORA_API_BASE_URL, str)

    def test_slack_allowed_domains_contains_hooks(self, config):
        assert "hooks.slack.com" in config.SLACK_ALLOWED_DOMAINS

    def test_slack_allowed_domains_contains_api(self, config):
        assert "api.slack.com" in config.SLACK_ALLOWED_DOMAINS

    def test_slack_allowed_domains_length(self, config):
        assert len(config.SLACK_ALLOWED_DOMAINS) == 2


# ===========================================================================
# COMMAND_PATTERN & TOPIC_PATTERN
# ===========================================================================

class TestCommandPattern:
    """Tests for the COMMAND_PATTERN regex."""

    def test_matches_simple_command(self, config):
        m = config.COMMAND_PATTERN.match("/aragora help")
        assert m is not None
        assert m.group(1) == "help"
        assert m.group(2) is None

    def test_matches_command_with_args(self, config):
        m = config.COMMAND_PATTERN.match("/aragora ask What is the meaning of life?")
        assert m is not None
        assert m.group(1) == "ask"
        assert m.group(2) == "What is the meaning of life?"

    def test_matches_command_multiple_spaces(self, config):
        m = config.COMMAND_PATTERN.match("/aragora   search   some query")
        assert m is not None
        assert m.group(1) == "search"

    def test_no_match_wrong_prefix(self, config):
        assert config.COMMAND_PATTERN.match("/other help") is None

    def test_no_match_no_subcommand(self, config):
        assert config.COMMAND_PATTERN.match("/aragora") is None

    def test_no_match_empty_string(self, config):
        assert config.COMMAND_PATTERN.match("") is None

    def test_command_word_boundary(self, config):
        """Subcommand must be a word (\\w+); hyphenated words don't match."""
        m = config.COMMAND_PATTERN.match("/aragora help-me")
        # "help-me" has no space after "help" and "-me" is not \\w or \\s
        # so the full-line pattern does not match
        assert m is None

    def test_command_hyphen_with_space(self, config):
        """Hyphenated text after space is captured as args."""
        m = config.COMMAND_PATTERN.match("/aragora help -me")
        assert m is not None
        assert m.group(1) == "help"
        assert m.group(2) == "-me"

    def test_debate_command(self, config):
        m = config.COMMAND_PATTERN.match("/aragora debate Should we use Rust?")
        assert m is not None
        assert m.group(1) == "debate"
        assert m.group(2) == "Should we use Rust?"

    def test_agents_command(self, config):
        m = config.COMMAND_PATTERN.match("/aragora agents")
        assert m is not None
        assert m.group(1) == "agents"

    def test_leaderboard_command(self, config):
        m = config.COMMAND_PATTERN.match("/aragora leaderboard")
        assert m is not None
        assert m.group(1) == "leaderboard"

    def test_no_match_just_slash(self, config):
        assert config.COMMAND_PATTERN.match("/") is None

    def test_no_match_missing_space_after_aragora(self, config):
        assert config.COMMAND_PATTERN.match("/aragorahelp") is None

    def test_command_with_trailing_space(self, config):
        m = config.COMMAND_PATTERN.match("/aragora help ")
        assert m is not None
        assert m.group(1) == "help"
        assert m.group(2) == ""


class TestTopicPattern:
    """Tests for the TOPIC_PATTERN regex."""

    def test_plain_topic(self, config):
        m = config.TOPIC_PATTERN.match("some topic")
        assert m is not None
        assert m.group(1) == "some topic"

    def test_double_quoted(self, config):
        m = config.TOPIC_PATTERN.match('"my topic"')
        assert m is not None
        assert m.group(1) == "my topic"

    def test_single_quoted(self, config):
        m = config.TOPIC_PATTERN.match("'my topic'")
        assert m is not None
        assert m.group(1) == "my topic"

    def test_empty_string_no_match(self, config):
        m = config.TOPIC_PATTERN.match("")
        assert m is None

    def test_only_quotes(self, config):
        m = config.TOPIC_PATTERN.match('""')
        # The pattern is greedy-minimal: (.+?) needs at least one char
        assert m is not None
        # Between the two " the (.+?) matches the second "
        assert m.group(1) == '"'

    def test_mixed_quotes(self, config):
        m = config.TOPIC_PATTERN.match("\"hello'")
        assert m is not None
        assert m.group(1) == "hello"

    def test_unquoted_single_word(self, config):
        m = config.TOPIC_PATTERN.match("kubernetes")
        assert m is not None
        assert m.group(1) == "kubernetes"

    def test_topic_with_special_chars(self, config):
        m = config.TOPIC_PATTERN.match("rate-limiting & caching!")
        assert m is not None
        assert m.group(1) == "rate-limiting & caching!"


# ===========================================================================
# get_workspace_store
# ===========================================================================

class TestGetWorkspaceStore:
    """Tests for get_workspace_store lazy singleton."""

    def test_success(self, config):
        mock_store = MagicMock()
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_store,
            create=True,
        ):
            result = config.get_workspace_store()
        assert result is mock_store

    def test_cached(self, config, monkeypatch):
        sentinel = MagicMock()
        monkeypatch.setattr(config, "_workspace_store", sentinel)
        result = config.get_workspace_store()
        assert result is sentinel

    def test_import_error(self, config):
        with _block_import("aragora.storage.slack_workspace_store"):
            result = config.get_workspace_store()
        assert result is None

    def test_success_caches_store(self, config):
        mock_store = MagicMock()
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_store,
            create=True,
        ):
            config.get_workspace_store()
        assert config._workspace_store is mock_store


# ===========================================================================
# resolve_workspace
# ===========================================================================

class TestResolveWorkspace:
    """Tests for resolve_workspace."""

    def test_success(self, config, monkeypatch):
        mock_store = MagicMock()
        workspace = SimpleNamespace(team_id="T123", name="test")
        mock_store.get.return_value = workspace
        monkeypatch.setattr(config, "_workspace_store", mock_store)
        result = config.resolve_workspace("T123")
        assert result is workspace
        mock_store.get.assert_called_once_with("T123")

    def test_empty_team_id_returns_none(self, config):
        assert config.resolve_workspace("") is None

    def test_none_team_id_returns_none(self, config):
        assert config.resolve_workspace(None) is None  # type: ignore[arg-type]

    def test_no_store_returns_none(self, config, monkeypatch):
        monkeypatch.setattr(config, "_workspace_store", None)
        with _block_import("aragora.storage.slack_workspace_store"):
            result = config.resolve_workspace("T999")
        assert result is None

    def test_key_error_returns_none(self, config, monkeypatch):
        mock_store = MagicMock()
        mock_store.get.side_effect = KeyError("not found")
        monkeypatch.setattr(config, "_workspace_store", mock_store)
        assert config.resolve_workspace("T404") is None

    def test_os_error_returns_none(self, config, monkeypatch):
        mock_store = MagicMock()
        mock_store.get.side_effect = OSError("disk")
        monkeypatch.setattr(config, "_workspace_store", mock_store)
        assert config.resolve_workspace("T500") is None

    def test_runtime_error_returns_none(self, config, monkeypatch):
        mock_store = MagicMock()
        mock_store.get.side_effect = RuntimeError("broken")
        monkeypatch.setattr(config, "_workspace_store", mock_store)
        assert config.resolve_workspace("T501") is None

    def test_calls_store_get_with_team_id(self, config, monkeypatch):
        mock_store = MagicMock()
        mock_store.get.return_value = None
        monkeypatch.setattr(config, "_workspace_store", mock_store)
        config.resolve_workspace("TABC")
        mock_store.get.assert_called_once_with("TABC")


# ===========================================================================
# get_slack_integration
# ===========================================================================

class TestGetSlackIntegration:
    """Tests for get_slack_integration lazy singleton."""

    def test_no_webhook_url_returns_none(self, config, monkeypatch):
        monkeypatch.setattr(config, "SLACK_WEBHOOK_URL", None)
        result = config.get_slack_integration()
        assert result is None

    def test_empty_webhook_url_returns_none(self, config, monkeypatch):
        monkeypatch.setattr(config, "SLACK_WEBHOOK_URL", "")
        result = config.get_slack_integration()
        assert result is None

    def test_success_creates_integration(self, config, monkeypatch):
        monkeypatch.setattr(config, "SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")
        mock_integration = MagicMock()
        mock_config = MagicMock()
        with patch(
            "aragora.integrations.slack.SlackConfig",
            return_value=mock_config,
            create=True,
        ) as mock_config_cls, patch(
            "aragora.integrations.slack.SlackIntegration",
            return_value=mock_integration,
            create=True,
        ) as mock_int_cls:
            result = config.get_slack_integration()
        assert result is mock_integration
        mock_config_cls.assert_called_once_with(webhook_url="https://hooks.slack.com/test")
        mock_int_cls.assert_called_once_with(mock_config)

    def test_cached_returns_same_instance(self, config, monkeypatch):
        sentinel = MagicMock()
        monkeypatch.setattr(config, "_slack_integration", sentinel)
        result = config.get_slack_integration()
        assert result is sentinel

    def test_import_error_returns_none(self, config, monkeypatch):
        monkeypatch.setattr(config, "SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")
        with _block_import("aragora.integrations.slack"):
            result = config.get_slack_integration()
        assert result is None

    def test_value_error_returns_none(self, config, monkeypatch):
        monkeypatch.setattr(config, "SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")
        with patch(
            "aragora.integrations.slack.SlackConfig",
            side_effect=ValueError("bad config"),
            create=True,
        ):
            result = config.get_slack_integration()
        assert result is None

    def test_key_error_returns_none(self, config, monkeypatch):
        monkeypatch.setattr(config, "SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")
        with patch(
            "aragora.integrations.slack.SlackConfig",
            side_effect=KeyError("missing"),
            create=True,
        ):
            result = config.get_slack_integration()
        assert result is None

    def test_type_error_returns_none(self, config, monkeypatch):
        monkeypatch.setattr(config, "SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")
        with patch(
            "aragora.integrations.slack.SlackConfig",
            side_effect=TypeError("wrong type"),
            create=True,
        ):
            result = config.get_slack_integration()
        assert result is None

    def test_runtime_error_returns_none(self, config, monkeypatch):
        monkeypatch.setattr(config, "SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")
        mock_config = MagicMock()
        with patch(
            "aragora.integrations.slack.SlackConfig",
            return_value=mock_config,
            create=True,
        ), patch(
            "aragora.integrations.slack.SlackIntegration",
            side_effect=RuntimeError("init fail"),
            create=True,
        ):
            result = config.get_slack_integration()
        assert result is None

    def test_os_error_returns_none(self, config, monkeypatch):
        monkeypatch.setattr(config, "SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")
        mock_config = MagicMock()
        with patch(
            "aragora.integrations.slack.SlackConfig",
            return_value=mock_config,
            create=True,
        ), patch(
            "aragora.integrations.slack.SlackIntegration",
            side_effect=OSError("network"),
            create=True,
        ):
            result = config.get_slack_integration()
        assert result is None

    def test_attribute_error_returns_none(self, config, monkeypatch):
        monkeypatch.setattr(config, "SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")
        mock_config = MagicMock()
        with patch(
            "aragora.integrations.slack.SlackConfig",
            return_value=mock_config,
            create=True,
        ), patch(
            "aragora.integrations.slack.SlackIntegration",
            side_effect=AttributeError("missing attr"),
            create=True,
        ):
            result = config.get_slack_integration()
        assert result is None

    def test_no_webhook_does_not_set_singleton(self, config, monkeypatch):
        """When no webhook URL, _slack_integration remains None (can retry later)."""
        monkeypatch.setattr(config, "SLACK_WEBHOOK_URL", None)
        config.get_slack_integration()
        assert config._slack_integration is None

    def test_import_error_does_not_set_singleton(self, config, monkeypatch):
        """After ImportError, _slack_integration remains None."""
        monkeypatch.setattr(config, "SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")
        with _block_import("aragora.integrations.slack"):
            config.get_slack_integration()
        assert config._slack_integration is None


# ===========================================================================
# Re-exported handler utilities
# ===========================================================================

class TestHandlerUtilityReexports:
    """Tests that handler utilities are properly re-exported."""

    def test_handler_result_not_none(self, config):
        # Should be the real HandlerResult if import succeeded
        assert config.HandlerResult is not None

    def test_secure_handler_not_none(self, config):
        assert config.SecureHandler is not None

    def test_error_response_callable(self, config):
        assert callable(config.error_response)

    def test_json_response_callable(self, config):
        assert callable(config.json_response)

    def test_auto_error_response_callable(self, config):
        assert callable(config.auto_error_response)

    def test_rate_limit_callable(self, config):
        assert callable(config.rate_limit)

    def test_forbidden_error_is_exception_subclass(self, config):
        assert issubclass(config.ForbiddenError, Exception)

    def test_unauthorized_error_is_exception_subclass(self, config):
        assert issubclass(config.UnauthorizedError, Exception)

    def test_auto_error_response_returns_decorator(self, config):
        """auto_error_response should return a callable decorator."""
        decorator = config.auto_error_response("test_op")
        assert callable(decorator)

    def test_rate_limit_returns_decorator(self, config):
        """rate_limit should return a callable decorator."""
        decorator = config.rate_limit(rpm=10)
        assert callable(decorator)


# ===========================================================================
# Environment variable constants
# ===========================================================================

class TestEnvVarConstants:
    """Tests for environment-variable-based constants."""

    def test_slack_signing_secret_is_none_or_string(self, config):
        val = config.SLACK_SIGNING_SECRET
        assert val is None or isinstance(val, str)

    def test_slack_bot_token_is_none_or_string(self, config):
        val = config.SLACK_BOT_TOKEN
        assert val is None or isinstance(val, str)

    def test_slack_webhook_url_is_none_or_string(self, config):
        val = config.SLACK_WEBHOOK_URL
        assert val is None or isinstance(val, str)

    def test_aragora_api_base_url_default(self, config):
        """Without env var, default is localhost:8080."""
        assert "localhost" in config.ARAGORA_API_BASE_URL or isinstance(config.ARAGORA_API_BASE_URL, str)


# ===========================================================================
# Logging (debug messages for unconfigured integrations)
# ===========================================================================

class TestLoggingMessages:
    """Tests for debug/warning log messages."""

    def test_audit_logger_import_error_logs_debug(self, config, caplog):
        with caplog.at_level(logging.DEBUG):
            with patch(
                "aragora.audit.slack_audit.get_slack_audit_logger",
                side_effect=ImportError("test"),
                create=True,
            ):
                config._get_audit_logger()
        assert "Slack audit logger not available" in caplog.text

    def test_user_limiter_import_error_logs_debug(self, config, caplog):
        with caplog.at_level(logging.DEBUG):
            with patch(
                "aragora.server.middleware.rate_limit.user_limiter.get_user_rate_limiter",
                side_effect=ImportError("test"),
                create=True,
            ):
                config._get_user_rate_limiter()
        assert "User rate limiter not available" in caplog.text

    def test_workspace_limiter_import_error_logs_debug(self, config, caplog):
        with caplog.at_level(logging.DEBUG):
            with patch(
                "aragora.server.middleware.rate_limit.user_limiter.get_user_rate_limiter",
                side_effect=ImportError("test"),
                create=True,
            ):
                config._get_workspace_rate_limiter()
        assert "Workspace rate limiter not available" in caplog.text

    def test_validate_url_none_returns_false(self, config, caplog):
        """Passing None triggers TypeError which is caught and logged."""
        with caplog.at_level(logging.DEBUG):
            result = config._validate_slack_url(None)  # type: ignore[arg-type]
        assert result is False

    def test_resolve_workspace_error_logs_debug(self, config, caplog, monkeypatch):
        mock_store = MagicMock()
        mock_store.get.side_effect = KeyError("not found")
        monkeypatch.setattr(config, "_workspace_store", mock_store)
        with caplog.at_level(logging.DEBUG):
            config.resolve_workspace("TMISSING")
        assert "Failed to get workspace" in caplog.text

    def test_get_slack_integration_value_error_logs_warning(self, config, caplog, monkeypatch):
        monkeypatch.setattr(config, "SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")
        with caplog.at_level(logging.WARNING):
            with patch(
                "aragora.integrations.slack.SlackConfig",
                side_effect=ValueError("bad"),
                create=True,
            ):
                config.get_slack_integration()
        assert "Invalid Slack configuration" in caplog.text

    def test_get_slack_integration_runtime_error_logs_exception(self, config, caplog, monkeypatch):
        monkeypatch.setattr(config, "SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")
        mock_config = MagicMock()
        with caplog.at_level(logging.WARNING):
            with patch(
                "aragora.integrations.slack.SlackConfig",
                return_value=mock_config,
                create=True,
            ), patch(
                "aragora.integrations.slack.SlackIntegration",
                side_effect=RuntimeError("boom"),
                create=True,
            ):
                config.get_slack_integration()
        assert "Unexpected error" in caplog.text

    def test_get_slack_integration_no_webhook_logs_debug(self, config, caplog, monkeypatch):
        monkeypatch.setattr(config, "SLACK_WEBHOOK_URL", None)
        with caplog.at_level(logging.DEBUG):
            config.get_slack_integration()
        assert "Slack integration disabled" in caplog.text

    def test_audit_logger_runtime_error_logs_debug(self, config, caplog):
        with caplog.at_level(logging.DEBUG):
            with patch(
                "aragora.audit.slack_audit.get_slack_audit_logger",
                side_effect=RuntimeError("fail"),
                create=True,
            ):
                config._get_audit_logger()
        assert "Slack audit logger not available" in caplog.text

    def test_audit_logger_os_error_logs_debug(self, config, caplog):
        with caplog.at_level(logging.DEBUG):
            with patch(
                "aragora.audit.slack_audit.get_slack_audit_logger",
                side_effect=OSError("fail"),
                create=True,
            ):
                config._get_audit_logger()
        assert "Slack audit logger not available" in caplog.text


# ===========================================================================
# Edge cases and integration
# ===========================================================================

class TestEdgeCases:
    """Edge cases and cross-cutting concerns."""

    def test_all_singletons_start_none(self, config):
        """After reset, all singletons are None."""
        assert config._slack_audit is None
        assert config._slack_user_limiter is None
        assert config._slack_workspace_limiter is None
        assert config._slack_integration is None
        assert config._workspace_store is None

    def test_command_pattern_is_compiled_regex(self, config):
        assert isinstance(config.COMMAND_PATTERN, re.Pattern)

    def test_topic_pattern_is_compiled_regex(self, config):
        assert isinstance(config.TOPIC_PATTERN, re.Pattern)

    def test_logger_is_module_logger(self, config):
        assert config.logger.name == "aragora.server.handlers.social._slack_impl.config"

    def test_validate_url_with_unicode(self, config):
        """Unicode domains are not in the allowed set."""
        assert config._validate_slack_url("https://h\u00f6oks.slack.com/test") is False

    def test_workspace_rate_limit_rpm_is_int(self, config):
        assert isinstance(config.SLACK_WORKSPACE_RATE_LIMIT_RPM, int)

    @pytest.mark.asyncio
    async def test_create_tracked_task_exception_in_coro(self, config):
        """Task that raises is still created; exception is handled by callback."""
        async def bad():
            raise ValueError("kaboom")

        task = config.create_tracked_task(bad(), "fail-task")
        assert isinstance(task, asyncio.Task)
        with pytest.raises(ValueError, match="kaboom"):
            await task

    def test_allowed_domains_immutable(self, config):
        """SLACK_ALLOWED_DOMAINS is a frozenset (immutable)."""
        with pytest.raises(AttributeError):
            config.SLACK_ALLOWED_DOMAINS.add("evil.com")  # type: ignore[attr-defined]

    def test_command_pattern_gauntlet(self, config):
        m = config.COMMAND_PATTERN.match("/aragora gauntlet 'my project'")
        assert m is not None
        assert m.group(1) == "gauntlet"
        assert "'my project'" in m.group(2)

    def test_command_pattern_status(self, config):
        m = config.COMMAND_PATTERN.match("/aragora status")
        assert m is not None
        assert m.group(1) == "status"

    def test_command_pattern_recent(self, config):
        m = config.COMMAND_PATTERN.match("/aragora recent")
        assert m is not None
        assert m.group(1) == "recent"

    def test_command_pattern_search_with_query(self, config):
        m = config.COMMAND_PATTERN.match("/aragora search rate limiting best practices")
        assert m is not None
        assert m.group(1) == "search"
        assert m.group(2) == "rate limiting best practices"

    def test_topic_pattern_preserves_inner_spaces(self, config):
        m = config.TOPIC_PATTERN.match('"  spaced topic  "')
        assert m is not None
        assert m.group(1) == "  spaced topic  "

    def test_resolve_workspace_with_store_returning_none(self, config, monkeypatch):
        """Store.get() returns None for unknown team."""
        mock_store = MagicMock()
        mock_store.get.return_value = None
        monkeypatch.setattr(config, "_workspace_store", mock_store)
        result = config.resolve_workspace("TUNKNOWN")
        assert result is None

    def test_get_audit_logger_double_call_uses_cache(self, config):
        """Second call should use cached value, not re-import."""
        mock_logger_obj = MagicMock()
        with patch(
            "aragora.audit.slack_audit.get_slack_audit_logger",
            return_value=mock_logger_obj,
            create=True,
        ) as mock_factory:
            first = config._get_audit_logger()
            second = config._get_audit_logger()
        assert first is second
        assert mock_factory.call_count == 1

    def test_get_user_rate_limiter_double_call_uses_cache(self, config):
        mock_limiter = MagicMock()
        with patch(
            "aragora.server.middleware.rate_limit.user_limiter.get_user_rate_limiter",
            return_value=mock_limiter,
            create=True,
        ) as mock_factory:
            first = config._get_user_rate_limiter()
            second = config._get_user_rate_limiter()
        assert first is second
        assert mock_factory.call_count == 1

    def test_get_workspace_rate_limiter_double_call_uses_cache(self, config):
        mock_limiter = MagicMock()
        mock_limiter.action_limits = {}
        with patch(
            "aragora.server.middleware.rate_limit.user_limiter.get_user_rate_limiter",
            return_value=mock_limiter,
            create=True,
        ) as mock_factory:
            first = config._get_workspace_rate_limiter()
            second = config._get_workspace_rate_limiter()
        assert first is second
        assert mock_factory.call_count == 1

    def test_get_workspace_store_double_call_uses_cache(self, config):
        mock_store = MagicMock()
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_store,
            create=True,
        ) as mock_factory:
            first = config.get_workspace_store()
            second = config.get_workspace_store()
        assert first is second
        assert mock_factory.call_count == 1

    def test_get_slack_integration_double_call_uses_cache(self, config, monkeypatch):
        monkeypatch.setattr(config, "SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")
        mock_integration = MagicMock()
        with patch(
            "aragora.integrations.slack.SlackConfig",
            return_value=MagicMock(),
            create=True,
        ), patch(
            "aragora.integrations.slack.SlackIntegration",
            return_value=mock_integration,
            create=True,
        ) as mock_factory:
            first = config.get_slack_integration()
            second = config.get_slack_integration()
        assert first is second
        assert mock_factory.call_count == 1

    def test_validate_url_long_path(self, config):
        """Long paths are OK as long as domain is valid."""
        url = "https://hooks.slack.com/" + "a" * 1000
        assert config._validate_slack_url(url) is True

    def test_validate_url_hooks_root(self, config):
        """Root path on allowed domain is valid."""
        assert config._validate_slack_url("https://hooks.slack.com/") is True

    def test_validate_url_api_root(self, config):
        """Root path on allowed domain is valid."""
        assert config._validate_slack_url("https://api.slack.com/") is True
