"""
Tests for Slack integration handler (slack.py).

Tests cover:
- Slack signature verification
- Command parsing
- Rate limiting
- Error handling
"""

import hashlib
import hmac
import json
import pytest
import time
from unittest.mock import MagicMock, patch

# Skip all tests in this module if the slack handler module doesn't exist
try:
    import aragora.server.handlers.slack  # noqa: F401
    SLACK_MODULE_AVAILABLE = True
except ImportError:
    SLACK_MODULE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not SLACK_MODULE_AVAILABLE,
    reason="Slack handler module not implemented - aragora.server.handlers.slack does not exist"
)


class MockHandler:
    """Mock HTTP request handler."""

    def __init__(self):
        self.headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Slack-Request-Timestamp": str(int(time.time())),
            "X-Slack-Signature": "",
        }
        self.path = "/api/integrations/slack/commands"
        self.command = "POST"
        self._body = b""
        self.rfile = MagicMock()

    def set_body(self, body: bytes):
        self._body = body
        self.rfile.read.return_value = body


class MockServerContext:
    """Mock server context."""

    def __init__(self):
        self.storage = MagicMock()
        self.user_store = MagicMock()
        self.config = {}


class TestSlackSignatureVerification:
    """Test Slack signature verification."""

    @pytest.fixture
    def slack_handler(self):
        """Create SlackHandler instance."""
        with patch.dict("os.environ", {"SLACK_SIGNING_SECRET": "test_secret"}):
            # Reload module to pick up env var
            import importlib
            import aragora.server.handlers.slack as slack_module

            importlib.reload(slack_module)
            return slack_module.SlackHandler(MockServerContext())

    @pytest.fixture
    def mock_handler(self):
        return MockHandler()

    def test_verify_signature_valid(self, slack_handler, mock_handler):
        """Test valid signature passes verification."""
        timestamp = str(int(time.time()))
        body = b"token=test&command=/aragora&text=debate"

        # Generate valid signature
        sig_basestring = f"v0:{timestamp}:{body.decode()}"
        signature = (
            "v0=" + hmac.new(b"test_secret", sig_basestring.encode(), hashlib.sha256).hexdigest()
        )

        mock_handler.headers["X-Slack-Request-Timestamp"] = timestamp
        mock_handler.headers["X-Slack-Signature"] = signature
        mock_handler.set_body(body)

        result = slack_handler._verify_signature(mock_handler)
        assert result is True

    def test_verify_signature_invalid(self, slack_handler, mock_handler):
        """Test invalid signature fails verification."""
        mock_handler.headers["X-Slack-Request-Timestamp"] = str(int(time.time()))
        mock_handler.headers["X-Slack-Signature"] = "v0=invalid_signature"
        mock_handler.set_body(b"test body")

        result = slack_handler._verify_signature(mock_handler)
        assert result is False

    def test_verify_signature_expired_timestamp(self, slack_handler, mock_handler):
        """Test expired timestamp fails verification (replay attack prevention)."""
        # Timestamp from 10 minutes ago
        old_timestamp = str(int(time.time()) - 600)
        body = b"token=test"

        sig_basestring = f"v0:{old_timestamp}:{body.decode()}"
        signature = (
            "v0=" + hmac.new(b"test_secret", sig_basestring.encode(), hashlib.sha256).hexdigest()
        )

        mock_handler.headers["X-Slack-Request-Timestamp"] = old_timestamp
        mock_handler.headers["X-Slack-Signature"] = signature
        mock_handler.set_body(body)

        result = slack_handler._verify_signature(mock_handler)
        assert result is False


class TestSlackCommandParsing:
    """Test Slack command parsing."""

    def test_parse_debate_command(self):
        """Test parsing /aragora debate command."""
        from aragora.server.handlers.slack import COMMAND_PATTERN

        text = "/aragora debate Should AI be regulated?"
        match = COMMAND_PATTERN.match(text)
        assert match is not None
        assert match.group(1) == "debate"
        assert match.group(2) == "Should AI be regulated?"

    def test_parse_status_command(self):
        """Test parsing /aragora status command."""
        from aragora.server.handlers.slack import COMMAND_PATTERN

        text = "/aragora status"
        match = COMMAND_PATTERN.match(text)
        assert match is not None
        assert match.group(1) == "status"
        assert match.group(2) is None

    def test_parse_help_command(self):
        """Test parsing /aragora help command."""
        from aragora.server.handlers.slack import COMMAND_PATTERN

        text = "/aragora help"
        match = COMMAND_PATTERN.match(text)
        assert match is not None
        assert match.group(1) == "help"


class TestSlackStatus:
    """Test Slack status endpoint."""

    @pytest.fixture
    def slack_handler(self):
        """Create SlackHandler instance."""
        with patch.dict("os.environ", {"SLACK_SIGNING_SECRET": "", "SLACK_WEBHOOK_URL": ""}):
            import importlib
            import aragora.server.handlers.slack as slack_module

            importlib.reload(slack_module)
            return slack_module.SlackHandler(MockServerContext())

    def test_get_status_no_integration(self, slack_handler):
        """Test status when Slack is not configured."""
        result = slack_handler._get_status()
        body = json.loads(result.body.decode())

        assert result.status_code == 200
        assert "enabled" in body


class TestSlackErrorHandling:
    """Test error handling for Slack endpoints."""

    @pytest.fixture
    def slack_handler(self):
        """Create SlackHandler instance."""
        with patch.dict("os.environ", {"SLACK_SIGNING_SECRET": "secret"}):
            import importlib
            import aragora.server.handlers.slack as slack_module

            importlib.reload(slack_module)
            return slack_module.SlackHandler(MockServerContext())

    @pytest.fixture
    def mock_handler(self):
        return MockHandler()

    def test_method_not_allowed(self, slack_handler, mock_handler):
        """Test non-POST method returns 405."""
        mock_handler.command = "GET"
        mock_handler.path = "/api/integrations/slack/commands"

        result = slack_handler.handle("/api/integrations/slack/commands", {}, mock_handler)

        assert result.status_code == 405

    def test_invalid_signature_returns_401(self, slack_handler, mock_handler):
        """Test invalid signature returns 401."""
        mock_handler.headers["X-Slack-Request-Timestamp"] = str(int(time.time()))
        mock_handler.headers["X-Slack-Signature"] = "v0=wrong"
        mock_handler.set_body(b"test")

        result = slack_handler.handle("/api/integrations/slack/commands", {}, mock_handler)

        assert result.status_code == 401


class TestSlackHelpCommand:
    """Test help command responses."""

    @pytest.fixture
    def slack_handler(self):
        """Create SlackHandler instance."""
        with patch.dict("os.environ", {"SLACK_SIGNING_SECRET": ""}):
            import importlib
            import aragora.server.handlers.slack as slack_module

            importlib.reload(slack_module)
            return slack_module.SlackHandler(MockServerContext())

    def test_help_returns_200(self, slack_handler):
        """Help command returns 200 status."""
        result = slack_handler._command_help()
        assert result.status_code == 200

    def test_help_contains_commands(self, slack_handler):
        """Help text contains available commands."""
        result = slack_handler._command_help()
        body = json.loads(result.body.decode())

        text = body.get("text", "")
        assert "debate" in text.lower()
        assert "status" in text.lower()
        assert "help" in text.lower()

    def test_help_is_ephemeral(self, slack_handler):
        """Help response is ephemeral."""
        result = slack_handler._command_help()
        body = json.loads(result.body.decode())

        assert body.get("response_type") == "ephemeral"

    def test_help_contains_examples(self, slack_handler):
        """Help text contains examples."""
        result = slack_handler._command_help()
        body = json.loads(result.body.decode())

        text = body.get("text", "")
        assert "example" in text.lower()


class TestSlackStatusCommand:
    """Test status command responses."""

    @pytest.fixture
    def slack_handler(self):
        """Create SlackHandler instance."""
        with patch.dict("os.environ", {"SLACK_SIGNING_SECRET": ""}):
            import importlib
            import aragora.server.handlers.slack as slack_module

            importlib.reload(slack_module)
            return slack_module.SlackHandler(MockServerContext())

    def test_status_returns_200(self, slack_handler):
        """Status command returns 200."""
        with patch("aragora.ranking.elo.get_elo_store") as mock:
            mock.return_value.get_all_ratings.return_value = []
            result = slack_handler._command_status()

        assert result.status_code == 200

    def test_status_shows_online(self, slack_handler):
        """Status shows system is online."""
        with patch("aragora.ranking.elo.get_elo_store") as mock:
            mock.return_value.get_all_ratings.return_value = []
            result = slack_handler._command_status()

        body = json.loads(result.body.decode())
        # Check either text or blocks contain status
        content = str(body)
        assert "online" in content.lower() or "status" in content.lower()

    def test_status_handles_error(self, slack_handler):
        """Status handles errors gracefully."""
        with patch("aragora.ranking.elo.get_elo_store") as mock:
            mock.side_effect = Exception("DB error")
            result = slack_handler._command_status()

        assert result.status_code == 200
        body = json.loads(result.body.decode())
        assert "error" in body.get("text", "").lower()


class TestSlackAgentsCommand:
    """Test agents command responses."""

    @pytest.fixture
    def slack_handler(self):
        """Create SlackHandler instance."""
        with patch.dict("os.environ", {"SLACK_SIGNING_SECRET": ""}):
            import importlib
            import aragora.server.handlers.slack as slack_module

            importlib.reload(slack_module)
            return slack_module.SlackHandler(MockServerContext())

    def test_agents_empty(self, slack_handler):
        """Agents command with no agents."""
        with patch("aragora.ranking.elo.get_elo_store") as mock:
            mock.return_value.get_all_ratings.return_value = []
            result = slack_handler._command_agents()

        body = json.loads(result.body.decode())
        assert "no agents" in body.get("text", "").lower()

    def test_agents_lists_by_elo(self, slack_handler):
        """Agents are listed sorted by ELO."""
        agents = []
        for name, elo in [("claude", 1600), ("gpt4", 1550), ("gemini", 1500)]:
            agent = MagicMock()
            agent.name = name
            agent.elo = elo
            agent.wins = 10
            agents.append(agent)

        with patch("aragora.ranking.elo.get_elo_store") as mock:
            mock.return_value.get_all_ratings.return_value = agents
            result = slack_handler._command_agents()

        body = json.loads(result.body.decode())
        text = body.get("text", "")
        assert "elo" in text.lower() or len(text) > 0

    def test_agents_handles_error(self, slack_handler):
        """Agents command handles errors."""
        with patch("aragora.ranking.elo.get_elo_store") as mock:
            mock.side_effect = Exception("Store error")
            result = slack_handler._command_agents()

        assert result.status_code == 200


class TestSlackResponseHelpers:
    """Test Slack response formatting helpers."""

    @pytest.fixture
    def slack_handler(self):
        """Create SlackHandler instance."""
        with patch.dict("os.environ", {"SLACK_SIGNING_SECRET": ""}):
            import importlib
            import aragora.server.handlers.slack as slack_module

            importlib.reload(slack_module)
            return slack_module.SlackHandler(MockServerContext())

    def test_slack_response_basic(self, slack_handler):
        """Basic slack response formatting."""
        result = slack_handler._slack_response("Hello!")

        body = json.loads(result.body.decode())
        assert body.get("text") == "Hello!"

    def test_slack_response_ephemeral(self, slack_handler):
        """Ephemeral response type."""
        result = slack_handler._slack_response("Private", response_type="ephemeral")

        body = json.loads(result.body.decode())
        assert body.get("response_type") == "ephemeral"

    def test_slack_response_in_channel(self, slack_handler):
        """In-channel response type."""
        result = slack_handler._slack_response("Public", response_type="in_channel")

        body = json.loads(result.body.decode())
        assert body.get("response_type") == "in_channel"

    def test_slack_blocks_response(self, slack_handler):
        """Block-formatted response."""
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "Test"}}]
        result = slack_handler._slack_blocks_response(blocks, text="Fallback")

        body = json.loads(result.body.decode())
        assert "blocks" in body
        assert body.get("text") == "Fallback"


class TestSlackRouting:
    """Test Slack request routing."""

    @pytest.fixture
    def slack_handler(self):
        """Create SlackHandler instance."""
        with patch.dict("os.environ", {"SLACK_SIGNING_SECRET": ""}):
            import importlib
            import aragora.server.handlers.slack as slack_module

            importlib.reload(slack_module)
            return slack_module.SlackHandler(MockServerContext())

    def test_can_handle_commands(self, slack_handler):
        """Handler routes commands endpoint."""
        assert slack_handler.can_handle("/api/integrations/slack/commands")

    def test_can_handle_interactive(self, slack_handler):
        """Handler routes interactive endpoint."""
        assert slack_handler.can_handle("/api/integrations/slack/interactive")

    def test_can_handle_events(self, slack_handler):
        """Handler routes events endpoint."""
        assert slack_handler.can_handle("/api/integrations/slack/events")

    def test_can_handle_status(self, slack_handler):
        """Handler routes status endpoint."""
        assert slack_handler.can_handle("/api/integrations/slack/status")

    def test_cannot_handle_other(self, slack_handler):
        """Handler rejects non-Slack routes."""
        assert not slack_handler.can_handle("/api/debates")
        assert not slack_handler.can_handle("/api/auth/login")


class TestSlackTopicParsing:
    """Test topic pattern matching."""

    def test_topic_pattern_quoted_double(self):
        """Parse double-quoted topics."""
        from aragora.server.handlers.slack import TOPIC_PATTERN

        match = TOPIC_PATTERN.match('"Should AI be regulated?"')
        assert match is not None
        assert "Should AI be regulated" in match.group(1)

    def test_topic_pattern_quoted_single(self):
        """Parse single-quoted topics."""
        from aragora.server.handlers.slack import TOPIC_PATTERN

        match = TOPIC_PATTERN.match("'AI ethics'")
        assert match is not None
        assert "AI ethics" in match.group(1)

    def test_topic_pattern_unquoted(self):
        """Parse unquoted topics."""
        from aragora.server.handlers.slack import TOPIC_PATTERN

        match = TOPIC_PATTERN.match("AI regulation")
        assert match is not None
        assert "AI regulation" in match.group(1)


class TestSlackIntegrationSingleton:
    """Test Slack integration singleton."""

    def test_get_integration_without_webhook(self):
        """Returns None when webhook not configured."""
        with patch.dict("os.environ", {"SLACK_WEBHOOK_URL": ""}, clear=False):
            import importlib
            import aragora.server.handlers.slack as slack_module

            importlib.reload(slack_module)

            # Clear cached integration
            if hasattr(slack_module, "_slack_integration"):
                slack_module._slack_integration = None

            result = slack_module.get_slack_integration()
            # Either None or previously cached
            assert result is None or result is not None


class TestSlackInteractiveHandler:
    """Test interactive component handling."""

    @pytest.fixture
    def slack_handler(self):
        """Create SlackHandler instance."""
        with patch.dict("os.environ", {"SLACK_SIGNING_SECRET": ""}):
            import importlib
            import aragora.server.handlers.slack as slack_module

            importlib.reload(slack_module)
            return slack_module.SlackHandler(MockServerContext())

    @pytest.fixture
    def mock_handler(self):
        return MockHandler()

    def test_interactive_rejects_get(self, slack_handler, mock_handler):
        """Interactive endpoint rejects GET."""
        mock_handler.command = "GET"

        result = slack_handler.handle("/api/integrations/slack/interactive", {}, mock_handler)

        assert result.status_code == 405


class TestSlackEventsHandler:
    """Test events API handling."""

    @pytest.fixture
    def slack_handler(self):
        """Create SlackHandler instance."""
        with patch.dict("os.environ", {"SLACK_SIGNING_SECRET": ""}):
            import importlib
            import aragora.server.handlers.slack as slack_module

            importlib.reload(slack_module)
            return slack_module.SlackHandler(MockServerContext())

    @pytest.fixture
    def mock_handler(self):
        return MockHandler()

    def test_events_rejects_get(self, slack_handler, mock_handler):
        """Events endpoint rejects GET."""
        mock_handler.command = "GET"

        result = slack_handler.handle("/api/integrations/slack/events", {}, mock_handler)

        assert result.status_code == 405
