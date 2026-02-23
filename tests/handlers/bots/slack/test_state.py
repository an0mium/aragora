"""
Tests for Slack Bot Handler State Management.

Covers all public functions and module-level state of
aragora.server.handlers.bots.slack.state:

- _active_debates: module-level dict for active debate sessions
- _user_votes: module-level dict for user votes per debate
- _slack_integration: module-level singleton (lazy-initialized)

- get_active_debates(): returns reference to _active_debates
- get_user_votes(): returns reference to _user_votes
- get_slack_integration():
  - Returns cached singleton when already set
  - Returns module-level override from parent package
  - Returns None when SLACK_WEBHOOK_URL is empty/unset
  - Creates SlackConnector when webhook URL is set
  - Returns None when SlackConnector import fails
  - Handles exceptions in module override resolution
- get_debate_vote_counts(debate_id):
  - Empty votes for unknown debate
  - Correct tallying for single and multiple voters
  - Tie counting
  - Single voter
- __all__ exports
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_MODULE = "aragora.server.handlers.bots.slack.state"
CONNECTOR_IMPORT = "aragora.connectors.slack.SlackConnector"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def state_module():
    """Import the state module lazily (after conftest patches)."""
    import aragora.server.handlers.bots.slack.state as mod

    return mod


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset module-level state before and after each test."""
    import aragora.server.handlers.bots.slack.state as mod

    mod._active_debates.clear()
    mod._user_votes.clear()
    mod._slack_integration = None
    yield
    mod._active_debates.clear()
    mod._user_votes.clear()
    mod._slack_integration = None


# ===========================================================================
# get_active_debates()
# ===========================================================================


class TestGetActiveDebates:
    """Tests for get_active_debates()."""

    def test_returns_dict(self, state_module):
        result = state_module.get_active_debates()
        assert isinstance(result, dict)

    def test_returns_empty_initially(self, state_module):
        result = state_module.get_active_debates()
        assert result == {}

    def test_returns_same_reference(self, state_module):
        """get_active_debates returns the module-level dict reference."""
        ref1 = state_module.get_active_debates()
        ref2 = state_module.get_active_debates()
        assert ref1 is ref2

    def test_reflects_mutations(self, state_module):
        """Mutations to the returned dict are visible on next call."""
        debates = state_module.get_active_debates()
        debates["debate-1"] = {"topic": "test", "status": "active"}
        assert state_module.get_active_debates()["debate-1"]["topic"] == "test"

    def test_reflects_direct_module_mutation(self, state_module):
        """Mutations via module attribute are visible through get_active_debates."""
        state_module._active_debates["d-99"] = {"x": 1}
        assert state_module.get_active_debates()["d-99"] == {"x": 1}


# ===========================================================================
# get_user_votes()
# ===========================================================================


class TestGetUserVotes:
    """Tests for get_user_votes()."""

    def test_returns_dict(self, state_module):
        result = state_module.get_user_votes()
        assert isinstance(result, dict)

    def test_returns_empty_initially(self, state_module):
        result = state_module.get_user_votes()
        assert result == {}

    def test_returns_same_reference(self, state_module):
        ref1 = state_module.get_user_votes()
        ref2 = state_module.get_user_votes()
        assert ref1 is ref2

    def test_reflects_mutations(self, state_module):
        votes = state_module.get_user_votes()
        votes["debate-1"] = {"user-a": "agent-1"}
        assert state_module.get_user_votes()["debate-1"]["user-a"] == "agent-1"

    def test_reflects_direct_module_mutation(self, state_module):
        state_module._user_votes["d-5"] = {"u1": "a1"}
        assert state_module.get_user_votes()["d-5"] == {"u1": "a1"}


# ===========================================================================
# get_slack_integration()
# ===========================================================================


class TestGetSlackIntegration:
    """Tests for get_slack_integration()."""

    # ---- cached singleton ----

    def test_returns_cached_singleton(self, state_module):
        """When _slack_integration is already set, return it directly."""
        sentinel = MagicMock(name="cached_connector")
        state_module._slack_integration = sentinel
        result = state_module.get_slack_integration()
        assert result is sentinel

    def test_cached_singleton_no_env_check(self, state_module):
        """Cached singleton skips environment variable check."""
        sentinel = MagicMock(name="cached")
        state_module._slack_integration = sentinel
        with patch.dict("os.environ", {}, clear=True):
            result = state_module.get_slack_integration()
        assert result is sentinel

    # ---- module-level override from parent package ----

    def test_module_override_from_parent_package(self, state_module):
        """When parent package has _slack_integration set, use that override."""
        mock_connector = MagicMock(name="override_connector")
        mock_package = types.ModuleType("aragora.server.handlers.bots.slack")
        mock_package._slack_integration = mock_connector

        with patch.dict(sys.modules, {"aragora.server.handlers.bots.slack": mock_package}):
            result = state_module.get_slack_integration()

        assert result is mock_connector

    def test_module_override_none_falls_through(self, state_module):
        """When parent package _slack_integration is None, skip override."""
        mock_package = types.ModuleType("aragora.server.handlers.bots.slack")
        mock_package._slack_integration = None

        with patch.dict(sys.modules, {"aragora.server.handlers.bots.slack": mock_package}):
            with patch.dict("os.environ", {}, clear=True):
                result = state_module.get_slack_integration()

        assert result is None

    def test_module_override_missing_attribute(self, state_module):
        """When parent package lacks _slack_integration attr, skip override."""
        mock_package = types.ModuleType("aragora.server.handlers.bots.slack")
        # Intentionally do NOT set _slack_integration

        with patch.dict(sys.modules, {"aragora.server.handlers.bots.slack": mock_package}):
            with patch.dict("os.environ", {}, clear=True):
                result = state_module.get_slack_integration()

        assert result is None

    def test_module_override_no_parent_in_sys_modules(self, state_module):
        """When parent package is not in sys.modules, skip override gracefully."""
        saved = sys.modules.get("aragora.server.handlers.bots.slack")
        try:
            sys.modules.pop("aragora.server.handlers.bots.slack", None)
            with patch.dict("os.environ", {}, clear=True):
                result = state_module.get_slack_integration()
            assert result is None
        finally:
            if saved is not None:
                sys.modules["aragora.server.handlers.bots.slack"] = saved

    def test_module_override_exception_caught(self, state_module):
        """Exceptions during override resolution are caught and logged."""

        class ExplodingModule:
            """Fake module that raises TypeError when _slack_integration is accessed."""

            def __init__(self):
                self._has_attr = True

            def __hasattr__(self, name):
                if name == "_slack_integration":
                    return True
                return False

            def __getattr__(self, name):
                if name == "_slack_integration":
                    raise TypeError("boom")
                raise AttributeError(name)

        mock_package = ExplodingModule()

        with patch.dict(sys.modules, {"aragora.server.handlers.bots.slack": mock_package}):
            with patch.dict("os.environ", {}, clear=True):
                # Should not raise, should fall through gracefully
                result = state_module.get_slack_integration()

        assert result is None

    # ---- no webhook URL ----

    def test_no_webhook_url_returns_none(self, state_module):
        """When SLACK_WEBHOOK_URL is not set, returns None."""
        with patch.dict("os.environ", {}, clear=True):
            result = state_module.get_slack_integration()
        assert result is None

    def test_empty_webhook_url_returns_none(self, state_module):
        """When SLACK_WEBHOOK_URL is empty string, returns None."""
        with patch.dict("os.environ", {"SLACK_WEBHOOK_URL": ""}):
            result = state_module.get_slack_integration()
        assert result is None

    # ---- creates SlackConnector ----

    def test_creates_connector_with_webhook_url(self, state_module):
        """Creates SlackConnector when SLACK_WEBHOOK_URL is configured."""
        mock_connector_cls = MagicMock(name="SlackConnector")
        mock_instance = MagicMock(name="connector_instance")
        mock_connector_cls.return_value = mock_instance

        with patch.dict("os.environ", {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/test"}):
            with patch(f"{STATE_MODULE}.SlackConnector", mock_connector_cls, create=True):
                # We need to patch the import itself
                with patch.dict(
                    sys.modules,
                    {"aragora.connectors.slack": types.ModuleType("aragora.connectors.slack")},
                ):
                    sys.modules["aragora.connectors.slack"].SlackConnector = mock_connector_cls
                    result = state_module.get_slack_integration()

        assert result is mock_instance
        mock_connector_cls.assert_called_once_with(webhook_url="https://hooks.slack.com/test")

    def test_caches_connector_after_creation(self, state_module):
        """After creating a connector, it is cached in module state."""
        mock_connector_cls = MagicMock(name="SlackConnector")
        mock_instance = MagicMock(name="connector_instance")
        mock_connector_cls.return_value = mock_instance

        with patch.dict("os.environ", {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/x"}):
            with patch.dict(
                sys.modules,
                {"aragora.connectors.slack": types.ModuleType("aragora.connectors.slack")},
            ):
                sys.modules["aragora.connectors.slack"].SlackConnector = mock_connector_cls
                state_module.get_slack_integration()

        # Now check cached
        assert state_module._slack_integration is mock_instance

    def test_import_error_returns_none(self, state_module):
        """When SlackConnector cannot be imported, returns None."""
        with patch.dict("os.environ", {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/x"}):
            # Remove the module to force ImportError
            saved = sys.modules.pop("aragora.connectors.slack", None)
            saved2 = sys.modules.pop("aragora.connectors", None)
            try:
                with patch(
                    "builtins.__import__", side_effect=_import_blocker("aragora.connectors.slack")
                ):
                    result = state_module.get_slack_integration()
                assert result is None
            finally:
                if saved is not None:
                    sys.modules["aragora.connectors.slack"] = saved
                if saved2 is not None:
                    sys.modules["aragora.connectors"] = saved2

    # ---- override sets _slack_integration on module ----

    def test_override_sets_module_attribute(self, state_module):
        """Module override also sets _slack_integration on the state module."""
        mock_connector = MagicMock(name="override_conn")
        mock_package = types.ModuleType("aragora.server.handlers.bots.slack")
        mock_package._slack_integration = mock_connector

        with patch.dict(sys.modules, {"aragora.server.handlers.bots.slack": mock_package}):
            state_module.get_slack_integration()

        assert state_module._slack_integration is mock_connector


# ===========================================================================
# get_debate_vote_counts()
# ===========================================================================


class TestGetDebateVoteCounts:
    """Tests for get_debate_vote_counts()."""

    def test_unknown_debate_returns_empty(self, state_module):
        result = state_module.get_debate_vote_counts("nonexistent")
        assert result == {}

    def test_debate_with_no_votes_returns_empty(self, state_module):
        state_module._user_votes["d1"] = {}
        result = state_module.get_debate_vote_counts("d1")
        assert result == {}

    def test_single_vote(self, state_module):
        state_module._user_votes["d1"] = {"user-a": "agent-1"}
        result = state_module.get_debate_vote_counts("d1")
        assert result == {"agent-1": 1}

    def test_multiple_votes_same_agent(self, state_module):
        state_module._user_votes["d1"] = {
            "user-a": "agent-1",
            "user-b": "agent-1",
            "user-c": "agent-1",
        }
        result = state_module.get_debate_vote_counts("d1")
        assert result == {"agent-1": 3}

    def test_multiple_agents(self, state_module):
        state_module._user_votes["d1"] = {
            "user-a": "agent-1",
            "user-b": "agent-2",
            "user-c": "agent-1",
            "user-d": "agent-3",
        }
        result = state_module.get_debate_vote_counts("d1")
        assert result == {"agent-1": 2, "agent-2": 1, "agent-3": 1}

    def test_tie(self, state_module):
        state_module._user_votes["d1"] = {
            "user-a": "agent-1",
            "user-b": "agent-2",
        }
        result = state_module.get_debate_vote_counts("d1")
        assert result == {"agent-1": 1, "agent-2": 1}

    def test_many_voters(self, state_module):
        """Stress test with many voters."""
        state_module._user_votes["d1"] = {f"user-{i}": f"agent-{i % 5}" for i in range(100)}
        result = state_module.get_debate_vote_counts("d1")
        assert sum(result.values()) == 100
        assert len(result) == 5
        for agent_id, count in result.items():
            assert count == 20

    def test_does_not_mutate_votes(self, state_module):
        """get_debate_vote_counts does not modify the underlying _user_votes."""
        state_module._user_votes["d1"] = {"u1": "a1", "u2": "a2"}
        state_module.get_debate_vote_counts("d1")
        assert state_module._user_votes["d1"] == {"u1": "a1", "u2": "a2"}

    def test_different_debates_independent(self, state_module):
        state_module._user_votes["d1"] = {"u1": "a1"}
        state_module._user_votes["d2"] = {"u2": "a2", "u3": "a2"}
        r1 = state_module.get_debate_vote_counts("d1")
        r2 = state_module.get_debate_vote_counts("d2")
        assert r1 == {"a1": 1}
        assert r2 == {"a2": 2}


# ===========================================================================
# __all__ exports
# ===========================================================================


class TestModuleExports:
    """Tests for __all__ and module-level exports."""

    def test_all_contains_get_active_debates(self, state_module):
        assert "get_active_debates" in state_module.__all__

    def test_all_contains_get_user_votes(self, state_module):
        assert "get_user_votes" in state_module.__all__

    def test_all_contains_get_slack_integration(self, state_module):
        assert "get_slack_integration" in state_module.__all__

    def test_all_contains_get_debate_vote_counts(self, state_module):
        assert "get_debate_vote_counts" in state_module.__all__

    def test_all_contains_active_debates(self, state_module):
        assert "_active_debates" in state_module.__all__

    def test_all_contains_user_votes(self, state_module):
        assert "_user_votes" in state_module.__all__

    def test_all_length(self, state_module):
        """__all__ has exactly 6 entries."""
        assert len(state_module.__all__) == 6

    def test_all_entries_are_accessible(self, state_module):
        """Every name in __all__ is accessible on the module."""
        for name in state_module.__all__:
            assert hasattr(state_module, name), f"{name} not found on module"


# ===========================================================================
# Module-level state
# ===========================================================================


class TestModuleLevelState:
    """Tests for module-level state variables."""

    def test_active_debates_is_dict(self, state_module):
        assert isinstance(state_module._active_debates, dict)

    def test_user_votes_is_dict(self, state_module):
        assert isinstance(state_module._user_votes, dict)

    def test_slack_integration_initially_none(self, state_module):
        assert state_module._slack_integration is None

    def test_active_debates_supports_nested_dict(self, state_module):
        state_module._active_debates["d1"] = {
            "topic": "Test debate",
            "channel": "C123",
            "agents": ["a1", "a2"],
            "started_at": 1234567890,
        }
        assert state_module._active_debates["d1"]["topic"] == "Test debate"

    def test_user_votes_supports_nested_mapping(self, state_module):
        state_module._user_votes["d1"] = {"u1": "a1", "u2": "a2"}
        assert state_module._user_votes["d1"]["u1"] == "a1"


# ===========================================================================
# Helpers (used by tests above)
# ===========================================================================


def _import_blocker(blocked_module: str):
    """Create an __import__ replacement that blocks a specific module."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _blocker(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"Blocked: {name}")
        return real_import(name, *args, **kwargs)

    return _blocker
