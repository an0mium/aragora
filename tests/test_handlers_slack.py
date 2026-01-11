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
        with patch.dict('os.environ', {'SLACK_SIGNING_SECRET': 'test_secret'}):
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
        signature = "v0=" + hmac.new(
            b"test_secret",
            sig_basestring.encode(),
            hashlib.sha256
        ).hexdigest()
        
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
        signature = "v0=" + hmac.new(
            b"test_secret",
            sig_basestring.encode(),
            hashlib.sha256
        ).hexdigest()
        
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
        with patch.dict('os.environ', {'SLACK_SIGNING_SECRET': '', 'SLACK_WEBHOOK_URL': ''}):
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
        with patch.dict('os.environ', {'SLACK_SIGNING_SECRET': 'secret'}):
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
        
        result = slack_handler.handle(
            "/api/integrations/slack/commands",
            {},
            mock_handler
        )
        
        assert result.status_code == 405

    def test_invalid_signature_returns_401(self, slack_handler, mock_handler):
        """Test invalid signature returns 401."""
        mock_handler.headers["X-Slack-Request-Timestamp"] = str(int(time.time()))
        mock_handler.headers["X-Slack-Signature"] = "v0=wrong"
        mock_handler.set_body(b"test")
        
        result = slack_handler.handle(
            "/api/integrations/slack/commands",
            {},
            mock_handler
        )
        
        assert result.status_code == 401
