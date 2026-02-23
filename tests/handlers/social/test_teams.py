"""Comprehensive tests for the Microsoft Teams integration handler.

Covers all routes and behavior of TeamsIntegrationHandler:
- can_handle() routing for all 4 defined ROUTES
- GET /api/v1/integrations/teams/status
- POST /api/v1/integrations/teams/commands (all 10 commands)
- POST /api/v1/integrations/teams/interactive (vote, cancel, view_receipt, unknown)
- POST /api/v1/integrations/teams/notify
- Path parameter extraction
- Error handling (invalid JSON, missing fields, connector unavailable)
- Adaptive Card block building helpers
- Active debate tracking lifecycle
- Connector singleton behavior
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.social.teams import (
    COMMAND_PATTERN,
    TOPIC_PATTERN,
    TeamsIntegrationHandler,
    get_teams_connector,
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


class MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to TeamsIntegrationHandler methods."""

    def __init__(
        self,
        method: str = "GET",
        path: str = "/api/v1/integrations/teams/status",
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.command = method
        self.path = path
        self.client_address = ("127.0.0.1", 12345)
        self.rfile = MagicMock()

        self.headers: dict[str, str] = headers or {"User-Agent": "test-agent"}

        if body is not None:
            raw = json.dumps(body).encode()
            self.rfile.read.return_value = raw
            self.headers["Content-Length"] = str(len(raw))
            self.headers["Content-Type"] = "application/json"
        else:
            self.rfile.read.return_value = b""
            self.headers["Content-Length"] = "0"

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like access for compatibility."""
        return self.headers.get(key, default)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a TeamsIntegrationHandler with a mock server context."""
    ctx: dict[str, Any] = {
        "user_store": MagicMock(),
        "nomic_dir": "/tmp/test",
        "stream_emitter": MagicMock(),
    }
    return TeamsIntegrationHandler(ctx)


@pytest.fixture
def mock_connector():
    """Create a mock Teams connector with common methods."""
    connector = AsyncMock()
    send_result = MagicMock()
    send_result.success = True
    send_result.message_id = "msg_001"
    send_result.error = None
    connector.send_message.return_value = send_result
    return connector


@pytest.fixture(autouse=True)
def _patch_rate_limit(monkeypatch):
    """Bypass rate limiting for tests."""
    monkeypatch.setenv("ARAGORA_USE_DISTRIBUTED_RATE_LIMIT", "false")


@pytest.fixture(autouse=True)
def _reset_connector():
    """Reset the Teams connector singleton between tests."""
    import aragora.server.handlers.social.teams as teams_mod

    teams_mod._teams_connector = None
    yield
    teams_mod._teams_connector = None


# ===========================================================================
# can_handle() routing tests
# ===========================================================================


class TestCanHandle:
    """Tests for handler routing via can_handle()."""

    def test_handle_commands_route(self, handler):
        assert handler.can_handle("/api/v1/integrations/teams/commands") is True

    def test_handle_interactive_route(self, handler):
        assert handler.can_handle("/api/v1/integrations/teams/interactive") is True

    def test_handle_status_route(self, handler):
        assert handler.can_handle("/api/v1/integrations/teams/status") is True

    def test_handle_notify_route(self, handler):
        assert handler.can_handle("/api/v1/integrations/teams/notify") is True

    def test_rejects_unknown_path(self, handler):
        assert handler.can_handle("/api/v1/other") is False

    def test_rejects_partial_prefix(self, handler):
        assert handler.can_handle("/api/v1/integrations/teams") is False

    def test_rejects_extra_segment(self, handler):
        assert handler.can_handle("/api/v1/integrations/teams/status/extra") is False

    def test_rejects_different_integration(self, handler):
        assert handler.can_handle("/api/v1/integrations/slack/commands") is False

    def test_routes_list_has_four_entries(self, handler):
        assert len(handler.ROUTES) == 4

    def test_can_handle_with_any_method(self, handler):
        """can_handle only checks path, not method."""
        assert handler.can_handle("/api/v1/integrations/teams/status", method="POST") is True
        assert handler.can_handle("/api/v1/integrations/teams/commands", method="GET") is True


# ===========================================================================
# Command pattern regex tests
# ===========================================================================


class TestCommandPattern:
    """Tests for the COMMAND_PATTERN regex used to parse bot commands."""

    def test_simple_command(self):
        m = COMMAND_PATTERN.match("help")
        assert m and m.group(1) == "help"

    def test_command_with_args(self):
        m = COMMAND_PATTERN.match("debate Should we use microservices?")
        assert m and m.group(1) == "debate"
        assert m.group(2) == "Should we use microservices?"

    def test_command_with_mention(self):
        m = COMMAND_PATTERN.match("@aragora debate Topic here")
        assert m and m.group(1) == "debate"
        assert m.group(2) == "Topic here"

    def test_case_insensitive(self):
        m = COMMAND_PATTERN.match("DEBATE test topic")
        assert m and m.group(1) == "DEBATE"

    def test_no_match_empty(self):
        m = COMMAND_PATTERN.match("")
        assert m is None

    def test_args_are_none_when_absent(self):
        m = COMMAND_PATTERN.match("status")
        assert m and m.group(2) is None


class TestTopicPattern:
    """Tests for the TOPIC_PATTERN regex used to strip quotes from topics."""

    def test_unquoted_topic(self):
        m = TOPIC_PATTERN.match("My topic")
        assert m and m.group(1) == "My topic"

    def test_single_quoted_topic(self):
        m = TOPIC_PATTERN.match("'My topic'")
        assert m and m.group(1) == "My topic"

    def test_double_quoted_topic(self):
        m = TOPIC_PATTERN.match('"My topic"')
        assert m and m.group(1) == "My topic"


# ===========================================================================
# GET /api/v1/integrations/teams/status
# ===========================================================================


class TestStatusEndpoint:
    """Tests for the status GET endpoint."""

    def test_status_returns_200(self, handler):
        h = MockHTTPHandler()
        result = handler.handle("/api/v1/integrations/teams/status", {}, h)
        assert _status(result) == 200

    def test_status_has_required_keys(self, handler):
        h = MockHTTPHandler()
        result = handler.handle("/api/v1/integrations/teams/status", {}, h)
        data = _body(result)
        assert "enabled" in data
        assert "app_id_configured" in data
        assert "password_configured" in data
        assert "tenant_id_configured" in data
        assert "connector_ready" in data

    def test_status_disabled_without_credentials(self, handler):
        with patch(
            "aragora.server.handlers.social.teams.get_teams_connector",
            return_value=None,
        ):
            result = handler._get_status()
            data = _body(result)
            assert data["connector_ready"] is False
            assert data["enabled"] is False

    def test_status_enabled_with_connector(self, handler, mock_connector):
        with patch(
            "aragora.server.handlers.social.teams.get_teams_connector",
            return_value=mock_connector,
        ):
            result = handler._get_status()
            data = _body(result)
            assert data["connector_ready"] is True
            assert data["enabled"] is True

    def test_handle_returns_none_for_non_status_get(self, handler):
        """handle() only processes the status path; other paths return None."""
        h = MockHTTPHandler()
        result = handler.handle("/api/v1/integrations/teams/commands", {}, h)
        assert result is None


# ===========================================================================
# POST /api/v1/integrations/teams/commands — debate command
# ===========================================================================


class TestDebateCommand:
    """Tests for the 'debate' command."""

    @pytest.mark.asyncio
    async def test_debate_starts_successfully(self, handler, mock_connector):
        body = {
            "text": "debate Should we adopt K8s?",
            "conversation": {"id": "conv_1"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "u1", "name": "Alice"},
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch(
                "aragora.server.handlers.social.teams.get_teams_connector",
                return_value=mock_connector,
            ):
                with patch("aragora.server.handlers.social.teams.create_tracked_task"):
                    result = await handler._handle_command(h)
                    data = _body(result)
                    assert data["success"] is True
                    assert data["topic"] == "Should we adopt K8s?"
                    assert data["conversation_id"] == "conv_1"

    @pytest.mark.asyncio
    async def test_debate_empty_topic_sends_error(self, handler, mock_connector):
        body = {
            "text": "debate",
            "conversation": {"id": "conv_2"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "u2"},
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch(
                "aragora.server.handlers.social.teams.get_teams_connector",
                return_value=mock_connector,
            ):
                result = await handler._handle_command(h)
                data = _body(result)
                assert data["status"] == "error"

    @pytest.mark.asyncio
    async def test_debate_already_running_sends_error(self, handler, mock_connector):
        handler._active_debates["conv_dup"] = {"topic": "Old", "status": "running"}
        body = {
            "text": "debate New topic",
            "conversation": {"id": "conv_dup"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "u3"},
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch(
                "aragora.server.handlers.social.teams.get_teams_connector",
                return_value=mock_connector,
            ):
                result = await handler._handle_command(h)
                data = _body(result)
                assert data["status"] == "error"
                assert "already running" in data["message"]

    @pytest.mark.asyncio
    async def test_debate_no_connector_returns_503(self, handler):
        body = {
            "text": "debate My topic",
            "conversation": {"id": "conv_nc"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "u4"},
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch(
                "aragora.server.handlers.social.teams.get_teams_connector",
                return_value=None,
            ):
                result = await handler._handle_command(h)
                assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_debate_ack_failure_returns_500(self, handler):
        bad_connector = AsyncMock()
        bad_result = MagicMock(success=False, message_id=None, error="send failed")
        bad_connector.send_message.return_value = bad_result

        body = {
            "text": "debate Ack fail topic",
            "conversation": {"id": "conv_af"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "u5", "name": "Bob"},
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch(
                "aragora.server.handlers.social.teams.get_teams_connector",
                return_value=bad_connector,
            ):
                result = await handler._handle_command(h)
                assert _status(result) == 500


# ===========================================================================
# POST /api/v1/integrations/teams/commands — plan command
# ===========================================================================


class TestPlanCommand:
    """Tests for the 'plan' command."""

    @pytest.mark.asyncio
    async def test_plan_command_includes_plan_flags(self, handler, mock_connector):
        body = {
            "text": "plan Build a CI pipeline",
            "conversation": {"id": "conv_plan"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "u6", "name": "Carol"},
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch.object(handler, "_start_debate") as mock_start:
                mock_start.return_value = MagicMock(body=b'{"success": true}', status_code=200)
                await handler._handle_command(h)
                _, kwargs = mock_start.call_args
                di = kwargs["decision_integrity"]
                assert di["include_plan"] is True
                assert di["include_context"] is False
                assert kwargs["mode_label"] == "plan"


# ===========================================================================
# POST /api/v1/integrations/teams/commands — implement command
# ===========================================================================


class TestImplementCommand:
    """Tests for the 'implement' command."""

    @pytest.mark.asyncio
    async def test_implement_command_includes_execution(self, handler, mock_connector):
        body = {
            "text": "implement Add rate limiter",
            "conversation": {"id": "conv_impl"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "u7", "name": "Dave"},
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch.object(handler, "_start_debate") as mock_start:
                mock_start.return_value = MagicMock(body=b'{"success": true}', status_code=200)
                await handler._handle_command(h)
                _, kwargs = mock_start.call_args
                di = kwargs["decision_integrity"]
                assert di["include_plan"] is True
                assert di["include_context"] is True
                assert di["execution_mode"] == "execute"
                assert di["execution_engine"] == "hybrid"
                assert kwargs["mode_label"] == "implementation plan"


# ===========================================================================
# POST /api/v1/integrations/teams/commands — status command
# ===========================================================================


class TestStatusCommand:
    """Tests for the 'status' command."""

    @pytest.mark.asyncio
    async def test_status_no_active_debate(self, handler):
        body = {
            "text": "status",
            "conversation": {"id": "conv_stat"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "u8"},
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            result = await handler._handle_command(h)
            data = _body(result)
            assert data["active"] is False

    @pytest.mark.asyncio
    async def test_status_with_active_debate(self, handler):
        handler._active_debates["conv_active"] = {
            "topic": "Active topic",
            "status": "running",
            "receipt_id": "rcpt_123",
        }
        body = {
            "text": "status",
            "conversation": {"id": "conv_active"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "u9"},
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            result = await handler._handle_command(h)
            data = _body(result)
            assert data["active"] is True
            assert data["topic"] == "Active topic"
            assert data["status"] == "running"
            assert data["receipt_id"] == "rcpt_123"


# ===========================================================================
# POST /api/v1/integrations/teams/commands — cancel command
# ===========================================================================


class TestCancelCommand:
    """Tests for the 'cancel' command."""

    @pytest.mark.asyncio
    async def test_cancel_existing_debate(self, handler):
        handler._active_debates["conv_can"] = {"topic": "To cancel"}
        body = {
            "text": "cancel",
            "conversation": {"id": "conv_can"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "u10"},
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            result = await handler._handle_command(h)
            data = _body(result)
            assert data["cancelled"] is True
            assert data["topic"] == "To cancel"
            assert "conv_can" not in handler._active_debates

    @pytest.mark.asyncio
    async def test_cancel_no_debate(self, handler):
        body = {
            "text": "cancel",
            "conversation": {"id": "conv_empty"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "u11"},
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            result = await handler._handle_command(h)
            data = _body(result)
            assert data["cancelled"] is False


# ===========================================================================
# POST /api/v1/integrations/teams/commands — help command
# ===========================================================================


class TestHelpCommand:
    """Tests for the 'help' command."""

    @pytest.mark.asyncio
    async def test_help_command(self, handler, mock_connector):
        body = {
            "text": "help",
            "conversation": {"id": "conv_help"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "u12"},
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch(
                "aragora.server.handlers.social.teams.get_teams_connector",
                return_value=mock_connector,
            ):
                result = await handler._handle_command(h)
                data = _body(result)
                assert data["status"] == "help_sent"

    @pytest.mark.asyncio
    async def test_help_without_connector(self, handler):
        body = {
            "text": "help",
            "conversation": {"id": "conv_help2"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "u13"},
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch(
                "aragora.server.handlers.social.teams.get_teams_connector",
                return_value=None,
            ):
                result = await handler._handle_command(h)
                data = _body(result)
                assert data["status"] == "help_sent"

    @pytest.mark.asyncio
    async def test_unmatched_text_triggers_help(self, handler, mock_connector):
        """When text does not match COMMAND_PATTERN, help is shown."""
        body = {
            "text": "",
            "conversation": {"id": "conv_nomatch"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "u14"},
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch(
                "aragora.server.handlers.social.teams.get_teams_connector",
                return_value=mock_connector,
            ):
                result = await handler._handle_command(h)
                data = _body(result)
                assert data["status"] == "help_sent"


# ===========================================================================
# POST /api/v1/integrations/teams/commands — leaderboard command
# ===========================================================================


class TestLeaderboardCommand:
    """Tests for the 'leaderboard' command."""

    @pytest.mark.asyncio
    async def test_leaderboard_with_store(self, handler, mock_connector):
        mock_ranking = MagicMock(agent_name="claude", elo=1700, wins=50, losses=10)
        with patch(
            "aragora.server.handlers.social.teams.get_teams_connector",
            return_value=mock_connector,
        ):
            body = {
                "text": "leaderboard",
                "conversation": {"id": "conv_lb"},
                "serviceUrl": "https://smba.trafficmanager.net/teams/",
                "from": {"id": "u15"},
            }
            h = MockHTTPHandler(method="POST", body=body)
            with patch.object(handler, "_read_json_body", return_value=body):
                with patch("aragora.ranking.elo.get_elo_store") as mock_store_fn:
                    mock_store = MagicMock()
                    mock_store.get_leaderboard.return_value = [mock_ranking]
                    mock_store_fn.return_value = mock_store
                    result = await handler._handle_command(h)
                    data = _body(result)
                    assert data["success"] is True
                    assert len(data["rankings"]) == 1
                    assert data["rankings"][0]["agent"] == "claude"

    @pytest.mark.asyncio
    async def test_leaderboard_fallback_rankings(self, handler, mock_connector):
        with patch(
            "aragora.server.handlers.social.teams.get_teams_connector",
            return_value=mock_connector,
        ):
            body = {
                "text": "leaderboard",
                "conversation": {"id": "conv_lb2"},
                "serviceUrl": "https://smba.trafficmanager.net/teams/",
                "from": {"id": "u16"},
            }
            h = MockHTTPHandler(method="POST", body=body)
            with patch.object(handler, "_read_json_body", return_value=body):
                with patch("aragora.ranking.elo.get_elo_store") as mock_store_fn:
                    mock_store = MagicMock()
                    mock_store.get_leaderboard.return_value = []
                    mock_store_fn.return_value = mock_store
                    result = await handler._handle_command(h)
                    data = _body(result)
                    assert data["success"] is True
                    # Falls back to sample rankings
                    assert len(data["rankings"]) == 3

    @pytest.mark.asyncio
    async def test_leaderboard_import_error(self, handler):
        body = {
            "text": "leaderboard",
            "conversation": {"id": "conv_lb3"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "u17"},
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch(
                "aragora.ranking.elo.get_elo_store",
                side_effect=ImportError("no elo"),
            ):
                # _get_leaderboard's internal try/except catches ImportError
                result = await handler._handle_command(h)
                assert _status(result) == 500


# ===========================================================================
# POST /api/v1/integrations/teams/commands — agents command
# ===========================================================================


class TestAgentsCommand:
    """Tests for the 'agents' command."""

    @pytest.mark.asyncio
    async def test_agents_with_available(self, handler, mock_connector):
        with patch(
            "aragora.server.handlers.social.teams.get_teams_connector",
            return_value=mock_connector,
        ):
            body = {
                "text": "agents",
                "conversation": {"id": "conv_ag"},
                "serviceUrl": "https://smba.trafficmanager.net/teams/",
                "from": {"id": "u18"},
            }
            h = MockHTTPHandler(method="POST", body=body)
            with patch.object(handler, "_read_json_body", return_value=body):
                with patch(
                    "aragora.agents.list_available_agents",
                    return_value={"claude": {"model": "claude-4"}},
                ):
                    result = await handler._handle_command(h)
                    data = _body(result)
                    assert data["success"] is True
                    assert len(data["agents"]) == 1
                    assert data["agents"][0]["name"] == "claude"

    @pytest.mark.asyncio
    async def test_agents_fallback_when_empty(self, handler, mock_connector):
        with patch(
            "aragora.server.handlers.social.teams.get_teams_connector",
            return_value=mock_connector,
        ):
            body = {
                "text": "agents",
                "conversation": {"id": "conv_ag2"},
                "serviceUrl": "https://smba.trafficmanager.net/teams/",
                "from": {"id": "u19"},
            }
            h = MockHTTPHandler(method="POST", body=body)
            with patch.object(handler, "_read_json_body", return_value=body):
                with patch(
                    "aragora.agents.list_available_agents",
                    return_value={},
                ):
                    result = await handler._handle_command(h)
                    data = _body(result)
                    assert data["success"] is True
                    # Falls back to default agent list
                    assert len(data["agents"]) == 3

    @pytest.mark.asyncio
    async def test_agents_import_error(self, handler):
        body = {
            "text": "agents",
            "conversation": {"id": "conv_ag3"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "u20"},
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch(
                "aragora.agents.list_available_agents",
                side_effect=ImportError("no agents module"),
            ):
                # _list_agents's internal try/except catches ImportError
                result = await handler._handle_command(h)
                assert _status(result) == 500


# ===========================================================================
# POST /api/v1/integrations/teams/commands — recent command
# ===========================================================================


class TestRecentCommand:
    """Tests for the 'recent' command."""

    @pytest.mark.asyncio
    async def test_recent_with_debates(self, handler, mock_connector):
        mock_debate = MagicMock(id="d1", topic="AI Safety", status="completed")
        with patch(
            "aragora.server.handlers.social.teams.get_teams_connector",
            return_value=mock_connector,
        ):
            body = {
                "text": "recent",
                "conversation": {"id": "conv_rec"},
                "serviceUrl": "https://smba.trafficmanager.net/teams/",
                "from": {"id": "u21"},
            }
            h = MockHTTPHandler(method="POST", body=body)
            with patch.object(handler, "_read_json_body", return_value=body):
                with patch("aragora.server.storage.get_debates_db") as mock_db_fn:
                    mock_db = MagicMock()
                    mock_db.list_recent.return_value = [mock_debate]
                    mock_db_fn.return_value = mock_db
                    result = await handler._handle_command(h)
                    data = _body(result)
                    assert data["success"] is True
                    assert len(data["debates"]) == 1
                    assert data["debates"][0]["topic"] == "AI Safety"

    @pytest.mark.asyncio
    async def test_recent_fallback_when_empty(self, handler, mock_connector):
        with patch(
            "aragora.server.handlers.social.teams.get_teams_connector",
            return_value=mock_connector,
        ):
            body = {
                "text": "recent",
                "conversation": {"id": "conv_rec2"},
                "serviceUrl": "https://smba.trafficmanager.net/teams/",
                "from": {"id": "u22"},
            }
            h = MockHTTPHandler(method="POST", body=body)
            with patch.object(handler, "_read_json_body", return_value=body):
                with patch("aragora.server.storage.get_debates_db") as mock_db_fn:
                    mock_db = MagicMock()
                    mock_db.list_recent.return_value = []
                    mock_db_fn.return_value = mock_db
                    result = await handler._handle_command(h)
                    data = _body(result)
                    assert data["success"] is True
                    assert data["debates"][0]["id"] == "none"


# ===========================================================================
# POST /api/v1/integrations/teams/commands — search command
# ===========================================================================


class TestSearchCommand:
    """Tests for the 'search' command."""

    @pytest.mark.asyncio
    async def test_search_with_results(self, handler, mock_connector):
        mock_result = MagicMock(id="d2", topic="Found debate", status="completed")
        with patch(
            "aragora.server.handlers.social.teams.get_teams_connector",
            return_value=mock_connector,
        ):
            body = {
                "text": "search microservices",
                "conversation": {"id": "conv_search"},
                "serviceUrl": "https://smba.trafficmanager.net/teams/",
                "from": {"id": "u23"},
            }
            h = MockHTTPHandler(method="POST", body=body)
            with patch.object(handler, "_read_json_body", return_value=body):
                with patch("aragora.server.storage.get_debates_db") as mock_db_fn:
                    mock_db = MagicMock()
                    mock_db.search.return_value = [mock_result]
                    mock_db_fn.return_value = mock_db
                    result = await handler._handle_command(h)
                    data = _body(result)
                    assert data["success"] is True
                    assert data["query"] == "microservices"
                    assert len(data["results"]) == 1

    @pytest.mark.asyncio
    async def test_search_empty_query(self, handler, mock_connector):
        with patch(
            "aragora.server.handlers.social.teams.get_teams_connector",
            return_value=mock_connector,
        ):
            body = {
                "text": "search",
                "conversation": {"id": "conv_search2"},
                "serviceUrl": "https://smba.trafficmanager.net/teams/",
                "from": {"id": "u24"},
            }
            h = MockHTTPHandler(method="POST", body=body)
            with patch.object(handler, "_read_json_body", return_value=body):
                result = await handler._handle_command(h)
                data = _body(result)
                # Empty search shows error message
                assert data["status"] == "error"

    @pytest.mark.asyncio
    async def test_search_no_results(self, handler, mock_connector):
        with patch(
            "aragora.server.handlers.social.teams.get_teams_connector",
            return_value=mock_connector,
        ):
            body = {
                "text": "search nonexistent_topic_xyz",
                "conversation": {"id": "conv_search3"},
                "serviceUrl": "https://smba.trafficmanager.net/teams/",
                "from": {"id": "u25"},
            }
            h = MockHTTPHandler(method="POST", body=body)
            with patch.object(handler, "_read_json_body", return_value=body):
                with patch("aragora.server.storage.get_debates_db") as mock_db_fn:
                    mock_db = MagicMock()
                    mock_db.search.return_value = []
                    mock_db_fn.return_value = mock_db
                    result = await handler._handle_command(h)
                    data = _body(result)
                    assert data["success"] is True
                    assert data["results"] == []


# ===========================================================================
# POST /api/v1/integrations/teams/commands — unknown command
# ===========================================================================


class TestUnknownCommand:
    """Tests for unknown commands."""

    @pytest.mark.asyncio
    async def test_unknown_command_returns_error(self, handler, mock_connector):
        body = {
            "text": "foobarbaz",
            "conversation": {"id": "conv_unk"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "u26"},
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch(
                "aragora.server.handlers.social.teams.get_teams_connector",
                return_value=mock_connector,
            ):
                result = await handler._handle_command(h)
                data = _body(result)
                assert data["status"] == "error"
                assert "foobarbaz" in data["message"]


# ===========================================================================
# POST /api/v1/integrations/teams/commands — bot mention stripping
# ===========================================================================


class TestBotMentionStripping:
    """Tests for stripping <at>...</at> mentions from incoming text."""

    @pytest.mark.asyncio
    async def test_strip_at_mention(self, handler):
        body = {
            "text": "<at>Aragora</at> status",
            "conversation": {"id": "conv_at"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "u27"},
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch.object(handler, "_get_debate_status") as mock_status:
                mock_status.return_value = MagicMock(body=b'{"active": false}', status_code=200)
                await handler._handle_command(h)
                mock_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_strip_nested_at_mention(self, handler, mock_connector):
        body = {
            "text": "<at>My Bot</at> help",
            "conversation": {"id": "conv_at2"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "u28"},
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch(
                "aragora.server.handlers.social.teams.get_teams_connector",
                return_value=mock_connector,
            ):
                result = await handler._handle_command(h)
                data = _body(result)
                assert data["status"] == "help_sent"


# ===========================================================================
# POST /api/v1/integrations/teams/commands — error handling
# ===========================================================================


class TestCommandErrorHandling:
    """Tests for error handling in command processing."""

    @pytest.mark.asyncio
    async def test_invalid_body_returns_400(self, handler):
        h = MockHTTPHandler(method="POST")
        with patch.object(handler, "_read_json_body", return_value=None):
            result = await handler._handle_command(h)
            assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_runtime_error_returns_500(self, handler):
        body = {
            "text": "debate Topic",
            "conversation": {"id": "conv_err"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "u29"},
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch.object(
                handler,
                "_start_debate",
                side_effect=RuntimeError("boom"),
            ):
                result = await handler._handle_command(h)
                assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_os_error_returns_500(self, handler):
        body = {
            "text": "debate Topic",
            "conversation": {"id": "conv_oserr"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "u30"},
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch.object(
                handler,
                "_start_debate",
                side_effect=OSError("disk full"),
            ):
                result = await handler._handle_command(h)
                assert _status(result) == 500


# ===========================================================================
# POST /api/v1/integrations/teams/interactive — vote action
# ===========================================================================


class TestInteractiveVoteAction:
    """Tests for vote handling via the interactive endpoint."""

    def test_vote_returns_recorded(self, handler):
        body = {
            "value": {"action": "vote", "vote": "agree", "debate_id": "d_int"},
            "conversation": {"id": "conv_iv"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "u31", "name": "Eve"},
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch("aragora.server.storage.get_debates_db", return_value=None):
                result = handler._handle_interactive(h)
                data = _body(result)
                assert data["status"] == "vote_recorded"
                assert data["vote"] == "agree"

    def test_vote_disagree(self, handler):
        body = {
            "value": {"action": "vote", "vote": "disagree", "debate_id": "d_dis"},
            "conversation": {"id": "conv_iv2"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "u32"},
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch("aragora.server.storage.get_debates_db", return_value=None):
                result = handler._handle_interactive(h)
                data = _body(result)
                assert data["vote"] == "disagree"

    def test_vote_records_in_database(self, handler):
        body = {
            "value": {"action": "vote", "vote": "agree", "debate_id": "d_db"},
            "conversation": {"id": "conv_iv3"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "u33", "name": "Frank"},
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch("aragora.server.storage.get_debates_db") as mock_db_fn:
                mock_db = MagicMock()
                mock_db_fn.return_value = mock_db
                handler._handle_interactive(h)
                mock_db.record_vote.assert_called_once_with(
                    debate_id="d_db",
                    voter_id="teams:u33",
                    vote="agree",
                    source="teams",
                )

    def test_vote_handles_missing_user(self, handler):
        body = {
            "value": {"action": "vote", "vote": "agree", "debate_id": "d_nu"},
            "conversation": {"id": "conv_iv4"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch("aragora.server.storage.get_debates_db", return_value=None):
                # "from" key is missing so from_user defaults to {}
                # _handle_interactive extracts from_user as body.get("from", {})
                result = handler._handle_interactive(h)
                data = _body(result)
                assert data["status"] == "vote_recorded"

    def test_vote_handles_db_error(self, handler):
        body = {
            "value": {"action": "vote", "vote": "agree", "debate_id": "d_dbe"},
            "conversation": {"id": "conv_iv5"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "u34"},
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch("aragora.server.storage.get_debates_db") as mock_db_fn:
                mock_db = MagicMock()
                mock_db.record_vote.side_effect = OSError("connection lost")
                mock_db_fn.return_value = mock_db
                # Should not raise, just log
                result = handler._handle_interactive(h)
                data = _body(result)
                assert data["status"] == "vote_recorded"


# ===========================================================================
# POST /api/v1/integrations/teams/interactive — cancel action
# ===========================================================================


class TestInteractiveCancelAction:
    """Tests for cancel_debate action via interactive endpoint."""

    def test_cancel_via_interactive(self, handler):
        handler._active_debates["conv_ic"] = {"topic": "Cancel me"}
        body = {
            "value": {"action": "cancel_debate"},
            "conversation": {"id": "conv_ic"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            result = handler._handle_interactive(h)
            data = _body(result)
            assert data["cancelled"] is True


# ===========================================================================
# POST /api/v1/integrations/teams/interactive — view_receipt action
# ===========================================================================


class TestInteractiveViewReceipt:
    """Tests for view_receipt action via interactive endpoint."""

    def test_view_receipt(self, handler):
        body = {
            "value": {"action": "view_receipt", "receipt_id": "rcpt_xyz"},
            "conversation": {"id": "conv_vr"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            result = handler._handle_interactive(h)
            data = _body(result)
            assert data["status"] == "ok"
            assert data["receipt_id"] == "rcpt_xyz"


# ===========================================================================
# POST /api/v1/integrations/teams/interactive — unknown action
# ===========================================================================


class TestInteractiveUnknownAction:
    """Tests for unknown actions via the interactive endpoint."""

    def test_unknown_action(self, handler):
        body = {
            "value": {"action": "some_random_action"},
            "conversation": {"id": "conv_ua"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            result = handler._handle_interactive(h)
            data = _body(result)
            assert data["status"] == "unknown_action"


# ===========================================================================
# POST /api/v1/integrations/teams/interactive — error handling
# ===========================================================================


class TestInteractiveErrorHandling:
    """Tests for error handling in the interactive endpoint."""

    def test_invalid_body_returns_400(self, handler):
        h = MockHTTPHandler(method="POST")
        with patch.object(handler, "_read_json_body", return_value=None):
            result = handler._handle_interactive(h)
            assert _status(result) == 400

    def test_runtime_error_returns_500(self, handler):
        body = {
            "value": {"action": "vote", "vote": "agree"},
            "conversation": {"id": "conv_ie"},
            "serviceUrl": "https://smba.trafficmanager.net/teams/",
            "from": {"id": "u35"},
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch.object(handler, "_handle_vote", side_effect=RuntimeError("crash")):
                result = handler._handle_interactive(h)
                assert _status(result) == 500


# ===========================================================================
# POST /api/v1/integrations/teams/notify
# ===========================================================================


class TestNotifyEndpoint:
    """Tests for the notification endpoint."""

    @pytest.mark.asyncio
    async def test_notify_success(self, handler, mock_connector):
        body = {
            "conversation_id": "conv_n1",
            "service_url": "https://smba.trafficmanager.net/teams/",
            "message": "Hello from Aragora",
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch(
                "aragora.server.handlers.social.teams.get_teams_connector",
                return_value=mock_connector,
            ):
                result = await handler._handle_notify(h)
                data = _body(result)
                assert data["success"] is True
                assert data["message_id"] == "msg_001"

    @pytest.mark.asyncio
    async def test_notify_with_blocks(self, handler, mock_connector):
        body = {
            "conversation_id": "conv_n2",
            "service_url": "https://smba.trafficmanager.net/teams/",
            "message": "Debate complete",
            "blocks": [{"type": "TextBlock", "text": "Done"}],
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch(
                "aragora.server.handlers.social.teams.get_teams_connector",
                return_value=mock_connector,
            ):
                result = await handler._handle_notify(h)
                mock_connector.send_message.assert_called_once()
                call_kwargs = mock_connector.send_message.call_args.kwargs
                assert call_kwargs["blocks"] == [{"type": "TextBlock", "text": "Done"}]

    @pytest.mark.asyncio
    async def test_notify_invalid_body(self, handler):
        h = MockHTTPHandler(method="POST")
        with patch.object(handler, "_read_json_body", return_value=None):
            result = await handler._handle_notify(h)
            assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_notify_missing_conversation_id(self, handler):
        body = {
            "service_url": "https://smba.trafficmanager.net/teams/",
            "message": "Test",
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            result = await handler._handle_notify(h)
            assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_notify_missing_service_url(self, handler):
        body = {
            "conversation_id": "conv_n3",
            "message": "Test",
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            result = await handler._handle_notify(h)
            assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_notify_no_connector(self, handler):
        body = {
            "conversation_id": "conv_n4",
            "service_url": "https://smba.trafficmanager.net/teams/",
            "message": "Test",
        }
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch(
                "aragora.server.handlers.social.teams.get_teams_connector",
                return_value=None,
            ):
                result = await handler._handle_notify(h)
                assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_notify_connection_error(self, handler):
        body = {
            "conversation_id": "conv_n5",
            "service_url": "https://smba.trafficmanager.net/teams/",
            "message": "Test",
        }
        bad_connector = AsyncMock()
        bad_connector.send_message.side_effect = ConnectionError("timeout")
        h = MockHTTPHandler(method="POST", body=body)
        with patch.object(handler, "_read_json_body", return_value=body):
            with patch(
                "aragora.server.handlers.social.teams.get_teams_connector",
                return_value=bad_connector,
            ):
                result = await handler._handle_notify(h)
                assert _status(result) == 500


# ===========================================================================
# handle_post() dispatch
# ===========================================================================


class TestHandlePostDispatch:
    """Tests for the top-level handle_post() method dispatching."""

    @pytest.mark.asyncio
    async def test_dispatch_to_commands(self, handler):
        h = MockHTTPHandler(method="POST")
        with patch.object(handler, "_handle_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = MagicMock(body=b'{"ok": true}', status_code=200)
            result = await handler.handle_post("/api/v1/integrations/teams/commands", {}, h)
            mock_cmd.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_to_interactive(self, handler):
        h = MockHTTPHandler(method="POST")
        with patch.object(handler, "_handle_interactive") as mock_int:
            mock_int.return_value = MagicMock(body=b'{"ok": true}', status_code=200)
            result = await handler.handle_post("/api/v1/integrations/teams/interactive", {}, h)
            mock_int.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_to_notify(self, handler):
        h = MockHTTPHandler(method="POST")
        with patch.object(handler, "_handle_notify", new_callable=AsyncMock) as mock_n:
            mock_n.return_value = MagicMock(body=b'{"ok": true}', status_code=200)
            result = await handler.handle_post("/api/v1/integrations/teams/notify", {}, h)
            mock_n.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_unknown_path_returns_404(self, handler):
        h = MockHTTPHandler(method="POST")
        result = await handler.handle_post("/api/v1/integrations/teams/unknown", {}, h)
        assert _status(result) == 404


# ===========================================================================
# Active debates tracking
# ===========================================================================


class TestActiveDebatesTracking:
    """Tests for the in-memory active debates dictionary."""

    def test_empty_on_init(self, handler):
        assert handler._active_debates == {}

    def test_debate_stored_by_conv_id(self, handler):
        handler._active_debates["c1"] = {"topic": "T1", "status": "running"}
        assert handler._active_debates["c1"]["topic"] == "T1"

    def test_cancel_removes_entry(self, handler):
        handler._active_debates["c2"] = {"topic": "T2"}
        conv = {"id": "c2"}
        handler._cancel_debate(conv)
        assert "c2" not in handler._active_debates

    def test_status_returns_active_true(self, handler):
        handler._active_debates["c3"] = {
            "topic": "T3",
            "status": "running",
            "receipt_id": None,
        }
        result = handler._get_debate_status({"id": "c3"})
        data = _body(result)
        assert data["active"] is True

    def test_status_returns_active_false(self, handler):
        result = handler._get_debate_status({"id": "nonexistent"})
        data = _body(result)
        assert data["active"] is False


# ===========================================================================
# Connector singleton
# ===========================================================================


class TestConnectorSingleton:
    """Tests for the get_teams_connector() singleton factory."""

    def test_returns_none_without_app_id(self):
        import aragora.server.handlers.social.teams as teams_mod

        teams_mod._teams_connector = None
        with patch.object(teams_mod, "TEAMS_APP_ID", ""):
            with patch.object(teams_mod, "TEAMS_APP_PASSWORD", "secret"):
                result = get_teams_connector()
                assert result is None

    def test_returns_none_without_password(self):
        import aragora.server.handlers.social.teams as teams_mod

        teams_mod._teams_connector = None
        with patch.object(teams_mod, "TEAMS_APP_ID", "app_id"):
            with patch.object(teams_mod, "TEAMS_APP_PASSWORD", ""):
                result = get_teams_connector()
                assert result is None

    def test_returns_none_on_import_error(self):
        import aragora.server.handlers.social.teams as teams_mod

        teams_mod._teams_connector = None
        with patch.object(teams_mod, "TEAMS_APP_ID", "app_id"):
            with patch.object(teams_mod, "TEAMS_APP_PASSWORD", "pwd"):
                with patch.dict(
                    "sys.modules",
                    {"aragora.connectors.chat.teams": None},
                ):
                    result = get_teams_connector()
                    assert result is None

    def test_returns_cached_singleton(self):
        import aragora.server.handlers.social.teams as teams_mod

        sentinel = MagicMock()
        teams_mod._teams_connector = sentinel
        result = get_teams_connector()
        assert result is sentinel


# ===========================================================================
# Adaptive Card block builders
# ===========================================================================


class TestBlockBuilders:
    """Tests for Adaptive Card block builder methods."""

    def test_build_starting_blocks_fallback(self, handler):
        """When TeamsAdaptiveCards is unavailable, fallback blocks are returned."""
        with patch.dict("sys.modules", {"aragora.connectors.chat.teams_adaptive_cards": None}):
            blocks = handler._build_starting_blocks("My Topic", "Alice")
            assert len(blocks) >= 3
            assert blocks[0]["type"] == "TextBlock"
            assert blocks[0]["text"] == "Debate Starting"
            # Verify topic and user are in the blocks
            topic_block = blocks[1]
            assert "My Topic" in topic_block["text"]
            user_block = blocks[2]
            assert "Alice" in user_block["text"]

    def test_build_result_blocks_fallback(self, handler):
        """When TeamsAdaptiveCards is unavailable, fallback result blocks are returned."""
        mock_result = MagicMock()
        mock_result.consensus = "Use microservices"
        mock_result.final_answer = "Use microservices"
        mock_result.confidence = 0.85
        mock_result.rounds_completed = 3
        mock_result.receipt_id = None
        mock_result.agent_votes = {}
        mock_result.agent_summaries = {}
        mock_result.rounds = []

        with patch.dict("sys.modules", {"aragora.connectors.chat.teams_adaptive_cards": None}):
            blocks = handler._build_result_blocks("Test Topic", mock_result)
            assert blocks[0]["text"] == "Debate Complete"
            # Find the decision block
            decision_texts = [b["text"] for b in blocks if "Decision" in b.get("text", "")]
            assert len(decision_texts) >= 1

    def test_build_result_blocks_with_receipt(self, handler):
        """Result blocks include receipt link when receipt_id is present."""
        mock_result = MagicMock()
        mock_result.consensus = "Adopt"
        mock_result.final_answer = "Adopt"
        mock_result.confidence = 0.9
        mock_result.rounds_completed = 2
        mock_result.receipt_id = "rcpt_abc"
        mock_result.agent_votes = {}
        mock_result.agent_summaries = {}
        mock_result.rounds = []

        with patch.dict("sys.modules", {"aragora.connectors.chat.teams_adaptive_cards": None}):
            blocks = handler._build_result_blocks("Topic", mock_result)
            action_blocks = [b for b in blocks if b.get("type") == "ActionSet"]
            assert len(action_blocks) == 1
            assert "rcpt_abc" in action_blocks[0]["actions"][0]["url"]

    def test_build_leaderboard_blocks(self, handler):
        rankings = [
            {"agent": "claude", "elo": 1700},
            {"agent": "gpt4", "elo": 1650},
            {"agent": "gemini", "elo": 1600},
        ]
        blocks = handler._build_leaderboard_blocks(rankings)
        assert len(blocks) == 1
        card = blocks[0]
        assert card["type"] == "AdaptiveCard"
        # Header + 3 ranking rows
        assert len(card["body"]) == 4

    def test_build_agents_blocks(self, handler):
        agents = [
            {"name": "claude", "model": "claude-4"},
            {"name": "gpt4", "model": "gpt-4"},
        ]
        blocks = handler._build_agents_blocks(agents)
        assert len(blocks) == 1
        card = blocks[0]
        assert card["type"] == "AdaptiveCard"
        # Header + 2 agent rows
        assert len(card["body"]) == 3

    def test_build_recent_blocks(self, handler):
        debates = [
            {"topic": "AI Safety", "status": "completed"},
        ]
        blocks = handler._build_recent_blocks(debates)
        card = blocks[0]
        assert card["type"] == "AdaptiveCard"
        assert len(card["body"]) == 2  # Header + 1 debate row

    def test_build_search_results_blocks(self, handler):
        results = [
            {"topic": "Found Topic", "status": "completed"},
        ]
        blocks = handler._build_search_results_blocks("my query", results)
        card = blocks[0]
        assert "my query" in card["body"][0]["text"]

    def test_build_voting_card_fallback(self, handler):
        with patch.dict("sys.modules", {"aragora.connectors.chat.teams_adaptive_cards": None}):
            blocks = handler._build_voting_card("Topic", "Yes we should", "d_vote")
            assert len(blocks) == 3
            action_set = blocks[2]
            assert action_set["type"] == "ActionSet"
            assert len(action_set["actions"]) == 2
            assert action_set["actions"][0]["data"]["vote"] == "agree"
            assert action_set["actions"][1]["data"]["vote"] == "disagree"

    def test_build_error_card_fallback(self, handler):
        with patch.dict("sys.modules", {"aragora.connectors.chat.teams_adaptive_cards": None}):
            blocks = handler._build_error_card("Oops", "Something went wrong")
            assert blocks[0]["text"] == "Oops"
            assert blocks[0]["color"] == "Attention"
            assert blocks[1]["text"] == "Something went wrong"


# ===========================================================================
# _read_json_body
# ===========================================================================


class TestReadJsonBody:
    """Tests for the _read_json_body helper."""

    def test_reads_valid_json(self, handler):
        h = MockHTTPHandler(body={"key": "value"})
        result = handler._read_json_body(h)
        assert result == {"key": "value"}

    def test_returns_none_for_zero_length(self, handler):
        h = MockHTTPHandler()  # Content-Length is 0 (no body)
        result = handler._read_json_body(h)
        assert result is None

    def test_returns_none_for_oversized_body(self, handler):
        h = MockHTTPHandler()
        h.headers["Content-Length"] = str(20 * 1024 * 1024)  # 20 MB
        result = handler._read_json_body(h)
        assert result is None

    def test_returns_none_for_invalid_json(self, handler):
        h = MagicMock()
        h.headers = {"Content-Length": "5"}
        h.rfile.read.return_value = b"nope!"
        result = handler._read_json_body(h)
        assert result is None


# ===========================================================================
# _send_error and _send_unknown_command helpers
# ===========================================================================


class TestSendHelpers:
    """Tests for _send_error and _send_unknown_command."""

    @pytest.mark.asyncio
    async def test_send_error_with_connector(self, handler, mock_connector):
        with patch(
            "aragora.server.handlers.social.teams.get_teams_connector",
            return_value=mock_connector,
        ):
            result = await handler._send_error(
                "Bad request", {"id": "conv_se"}, "https://smba.trafficmanager.net/teams/"
            )
            data = _body(result)
            assert data["status"] == "error"
            assert data["message"] == "Bad request"
            mock_connector.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_error_without_connector(self, handler):
        with patch(
            "aragora.server.handlers.social.teams.get_teams_connector",
            return_value=None,
        ):
            result = await handler._send_error(
                "No conn", {"id": "conv_se2"}, "https://smba.trafficmanager.net/teams/"
            )
            data = _body(result)
            assert data["status"] == "error"

    @pytest.mark.asyncio
    async def test_send_unknown_command(self, handler, mock_connector):
        with patch(
            "aragora.server.handlers.social.teams.get_teams_connector",
            return_value=mock_connector,
        ):
            result = await handler._send_unknown_command(
                "bazqux", {"id": "conv_uc"}, "https://smba.trafficmanager.net/teams/"
            )
            data = _body(result)
            assert "bazqux" in data["message"]


# ===========================================================================
# _handle_vote directly
# ===========================================================================


class TestHandleVoteDirect:
    """Tests for the _handle_vote method called directly."""

    def test_vote_records_user_id(self, handler):
        with patch("aragora.server.storage.get_debates_db") as mock_db_fn:
            mock_db = MagicMock()
            mock_db_fn.return_value = mock_db
            handler._handle_vote(
                {"vote": "agree", "debate_id": "d_hv"},
                {"id": "c_hv"},
                "https://smba.trafficmanager.net/teams/",
                {"id": "voter_42"},
            )
            mock_db.record_vote.assert_called_once_with(
                debate_id="d_hv",
                voter_id="teams:voter_42",
                vote="agree",
                source="teams",
            )

    def test_vote_null_from_user(self, handler):
        with patch("aragora.server.storage.get_debates_db", return_value=None):
            result = handler._handle_vote(
                {"vote": "disagree", "debate_id": "d_null"},
                {"id": "c_null"},
                "https://smba.trafficmanager.net/teams/",
                None,
            )
            data = _body(result)
            assert data["debate_id"] == "d_null"

    def test_vote_aggregator_import_error(self, handler):
        """If VoteAggregator import fails, vote still succeeds."""
        with patch("aragora.server.storage.get_debates_db", return_value=None):
            with patch.dict("sys.modules", {"aragora.debate.vote_aggregator": None}):
                result = handler._handle_vote(
                    {"vote": "agree", "debate_id": "d_agg"},
                    {"id": "c_agg"},
                    "https://smba.trafficmanager.net/teams/",
                    {"id": "u_agg"},
                )
                data = _body(result)
                assert data["status"] == "vote_recorded"


# ===========================================================================
# _handle_view_receipt directly
# ===========================================================================


class TestHandleViewReceipt:
    """Tests for the _handle_view_receipt method."""

    def test_returns_receipt_id(self, handler):
        result = handler._handle_view_receipt(
            {"receipt_id": "rcpt_42"},
            {"id": "c_vr"},
            "https://smba.trafficmanager.net/teams/",
        )
        data = _body(result)
        assert data["status"] == "ok"
        assert data["receipt_id"] == "rcpt_42"

    def test_returns_none_receipt_id(self, handler):
        result = handler._handle_view_receipt(
            {},
            {"id": "c_vr2"},
            "https://smba.trafficmanager.net/teams/",
        )
        data = _body(result)
        assert data["receipt_id"] is None


# ===========================================================================
# _get_debate_status and _cancel_debate directly
# ===========================================================================


class TestDebateStatusAndCancel:
    """Tests for _get_debate_status and _cancel_debate called directly."""

    def test_get_status_active(self, handler):
        handler._active_debates["c_s1"] = {
            "topic": "Active",
            "status": "running",
            "receipt_id": "r1",
        }
        result = handler._get_debate_status({"id": "c_s1"})
        data = _body(result)
        assert data["active"] is True
        assert data["topic"] == "Active"
        assert data["receipt_id"] == "r1"

    def test_get_status_inactive(self, handler):
        result = handler._get_debate_status({"id": "c_s_missing"})
        data = _body(result)
        assert data["active"] is False
        assert "No active debate" in data["message"]

    def test_get_status_empty_conversation_id(self, handler):
        result = handler._get_debate_status({})
        data = _body(result)
        assert data["active"] is False

    def test_cancel_returns_topic(self, handler):
        handler._active_debates["c_c1"] = {"topic": "Cancelled one"}
        result = handler._cancel_debate({"id": "c_c1"})
        data = _body(result)
        assert data["cancelled"] is True
        assert data["topic"] == "Cancelled one"

    def test_cancel_missing_debate(self, handler):
        result = handler._cancel_debate({"id": "c_c_nope"})
        data = _body(result)
        assert data["cancelled"] is False
        assert "No active debate" in data["message"]


# ===========================================================================
# _send_help_response
# ===========================================================================


class TestSendHelpResponse:
    """Tests for the help response builder."""

    @pytest.mark.asyncio
    async def test_help_response_structure(self, handler, mock_connector):
        with patch(
            "aragora.server.handlers.social.teams.get_teams_connector",
            return_value=mock_connector,
        ):
            result = await handler._send_help_response(
                {"id": "c_help"}, "https://smba.trafficmanager.net/teams/"
            )
            data = _body(result)
            assert data["status"] == "help_sent"
            # Connector should have been called with help blocks
            mock_connector.send_message.assert_called_once()
            call_kwargs = mock_connector.send_message.call_args.kwargs
            assert call_kwargs["text"] == "Aragora Help"
            # Verify blocks contain FactSet with commands
            blocks = call_kwargs["blocks"]
            fact_sets = [b for b in blocks if b.get("type") == "FactSet"]
            assert len(fact_sets) == 1
            facts = fact_sets[0]["facts"]
            command_titles = [f["title"] for f in facts]
            assert "@aragora debate <topic>" in command_titles
            assert "@aragora help" in command_titles


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
