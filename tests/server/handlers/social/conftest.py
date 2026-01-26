"""
Shared fixtures for social handler tests.

Provides reusable mock handlers, fixtures, and helper functions for testing
Slack, Teams, notifications, and other social integration handlers.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.base import HandlerResult


# ===========================================================================
# Mock HTTP Handler
# ===========================================================================


class MockHandler:
    """Reusable mock HTTP handler for all social handler tests.

    Provides a simulated HTTP request handler with configurable headers,
    body content, path, and method. Supports both JSON and form-encoded bodies.

    Examples:
        # JSON body
        handler = MockHandler.with_json_body({"key": "value"}, path="/api/test")

        # Form-encoded body (for Slack commands)
        handler = MockHandler.with_form_body(
            {"command": "/aragora", "text": "help"},
            path="/api/integrations/slack/commands"
        )
    """

    def __init__(
        self,
        headers: Optional[Dict[str, str]] = None,
        body: bytes = b"",
        path: str = "/",
        method: str = "GET",
        client_address: Optional[Tuple[str, int]] = None,
    ):
        self.headers = headers or {}
        self._body = body
        self.path = path
        self.command = method
        self.rfile = BytesIO(body)
        self.client_address = client_address or ("127.0.0.1", 12345)
        self.response_code: Optional[int] = None
        self._response_headers: Dict[str, str] = {}

    def send_response(self, code: int) -> None:
        self.response_code = code

    def send_header(self, key: str, value: str) -> None:
        self._response_headers[key] = value

    def end_headers(self) -> None:
        pass

    @classmethod
    def with_json_body(
        cls,
        data: Dict[str, Any],
        path: str = "/",
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None,
        client_address: Optional[Tuple[str, int]] = None,
    ) -> "MockHandler":
        """Create a MockHandler with a JSON body.

        Args:
            data: Dictionary to serialize as JSON body
            path: Request path
            method: HTTP method
            headers: Additional headers (Content-Type and Content-Length auto-added)
            client_address: Client IP and port tuple

        Returns:
            Configured MockHandler instance
        """
        body = json.dumps(data).encode("utf-8")
        all_headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }
        if headers:
            all_headers.update(headers)
        return cls(
            headers=all_headers,
            body=body,
            path=path,
            method=method,
            client_address=client_address,
        )

    @classmethod
    def with_form_body(
        cls,
        data: Dict[str, str],
        path: str = "/",
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None,
        client_address: Optional[Tuple[str, int]] = None,
    ) -> "MockHandler":
        """Create a MockHandler with form-encoded body (for Slack commands).

        Args:
            data: Dictionary to encode as form data
            path: Request path
            method: HTTP method
            headers: Additional headers (Content-Type and Content-Length auto-added)
            client_address: Client IP and port tuple

        Returns:
            Configured MockHandler instance
        """
        from urllib.parse import urlencode

        body = urlencode(data).encode("utf-8")
        all_headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Content-Length": str(len(body)),
        }
        if headers:
            all_headers.update(headers)
        return cls(
            headers=all_headers,
            body=body,
            path=path,
            method=method,
            client_address=client_address,
        )


# ===========================================================================
# Result Parsing Helpers
# ===========================================================================


def parse_result(result: HandlerResult) -> Tuple[int, Dict[str, Any]]:
    """Parse HandlerResult into (status_code, body_dict).

    Args:
        result: HandlerResult from handler method

    Returns:
        Tuple of (status_code, parsed JSON body as dict)
    """
    body = json.loads(result.body.decode("utf-8"))
    return result.status_code, body


def get_body(result: HandlerResult) -> bytes:
    """Get raw body bytes from HandlerResult."""
    return result.body


def get_status_code(result: HandlerResult) -> int:
    """Get status code from HandlerResult."""
    return result.status_code


def get_json(result: HandlerResult) -> Dict[str, Any]:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


def get_data(result: HandlerResult) -> Dict[str, Any]:
    """Get 'data' field from success response."""
    body = get_json(result)
    return body.get("data", body)


def get_error(result: HandlerResult) -> str:
    """Get error message from error response."""
    body = get_json(result)
    error = body.get("error", "")
    if isinstance(error, dict):
        return error.get("message", "")
    return error


# ===========================================================================
# Slack-Specific Helpers
# ===========================================================================


def generate_slack_signature(
    body: str,
    timestamp: str,
    signing_secret: str,
) -> str:
    """Generate a valid Slack signature for testing.

    See: https://api.slack.com/authentication/verifying-requests-from-slack

    Args:
        body: Request body string
        timestamp: Unix timestamp string
        signing_secret: Slack signing secret

    Returns:
        Signature in format "v0=<hex_digest>"
    """
    sig_basestring = f"v0:{timestamp}:{body}"
    signature = hmac.new(
        signing_secret.encode("utf-8"),
        sig_basestring.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"v0={signature}"


def create_slack_command_handler(
    command: str = "/aragora",
    text: str = "help",
    user_id: str = "U12345",
    channel_id: str = "C12345",
    team_id: str = "T12345",
    response_url: str = "https://hooks.slack.com/commands/T12345/12345/token",
    signing_secret: str = "test_signing_secret",
) -> MockHandler:
    """Create a MockHandler configured for Slack slash command testing.

    Args:
        command: Slack command (e.g., "/aragora")
        text: Command text/arguments
        user_id: Slack user ID
        channel_id: Slack channel ID
        team_id: Slack team/workspace ID
        response_url: URL for async responses
        signing_secret: Secret for signature generation

    Returns:
        MockHandler with valid Slack headers and body
    """
    from urllib.parse import urlencode

    data = {
        "command": command,
        "text": text,
        "user_id": user_id,
        "channel_id": channel_id,
        "team_id": team_id,
        "response_url": response_url,
    }
    body = urlencode(data)
    timestamp = str(int(time.time()))
    signature = generate_slack_signature(body, timestamp, signing_secret)

    return MockHandler(
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Content-Length": str(len(body)),
            "X-Slack-Request-Timestamp": timestamp,
            "X-Slack-Signature": signature,
        },
        body=body.encode("utf-8"),
        path="/api/v1/integrations/slack/commands",
        method="POST",
    )


def create_slack_interactive_handler(
    action_id: str = "vote_for",
    user_id: str = "U12345",
    team_id: str = "T12345",
    response_url: str = "https://hooks.slack.com/actions/T12345/12345/token",
    signing_secret: str = "test_signing_secret",
    payload_extras: Optional[Dict[str, Any]] = None,
) -> MockHandler:
    """Create a MockHandler for Slack interactive component testing.

    Args:
        action_id: Action identifier
        user_id: Slack user ID
        team_id: Slack team/workspace ID
        response_url: URL for async responses
        signing_secret: Secret for signature generation
        payload_extras: Additional fields to merge into payload

    Returns:
        MockHandler with valid Slack interactive payload
    """
    from urllib.parse import urlencode

    payload = {
        "type": "block_actions",
        "user": {"id": user_id, "name": "testuser"},
        "team": {"id": team_id},
        "actions": [{"action_id": action_id, "value": "test_value"}],
        "response_url": response_url,
    }
    if payload_extras:
        payload.update(payload_extras)

    data = {"payload": json.dumps(payload)}
    body = urlencode(data)
    timestamp = str(int(time.time()))
    signature = generate_slack_signature(body, timestamp, signing_secret)

    return MockHandler(
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Content-Length": str(len(body)),
            "X-Slack-Request-Timestamp": timestamp,
            "X-Slack-Signature": signature,
        },
        body=body.encode("utf-8"),
        path="/api/v1/integrations/slack/interactive",
        method="POST",
    )


def create_slack_event_handler(
    event_type: str = "message",
    team_id: str = "T12345",
    event_data: Optional[Dict[str, Any]] = None,
    signing_secret: str = "test_signing_secret",
) -> MockHandler:
    """Create a MockHandler for Slack Events API testing.

    Args:
        event_type: Type of event (e.g., "message", "app_mention")
        team_id: Slack team/workspace ID
        event_data: Additional event data
        signing_secret: Secret for signature generation

    Returns:
        MockHandler with valid Slack event payload
    """
    payload = {
        "type": "event_callback",
        "team_id": team_id,
        "event": {"type": event_type, **(event_data or {})},
    }
    body = json.dumps(payload)
    timestamp = str(int(time.time()))
    signature = generate_slack_signature(body, timestamp, signing_secret)

    return MockHandler(
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            "X-Slack-Request-Timestamp": timestamp,
            "X-Slack-Signature": signature,
        },
        body=body.encode("utf-8"),
        path="/api/v1/integrations/slack/events",
        method="POST",
    )


# ===========================================================================
# Common Fixtures
# ===========================================================================


@pytest.fixture
def mock_server_context():
    """Create a mock server context for handler initialization."""
    return {
        "storage": MagicMock(),
        "elo_system": MagicMock(),
        "arena": MagicMock(),
    }


@pytest.fixture
def mock_http_handler():
    """Create a basic mock HTTP handler."""
    return MockHandler(
        headers={"Content-Type": "application/json", "Content-Length": "0"},
        path="/",
    )


# ===========================================================================
# Mock User and Auth Fixtures
# ===========================================================================


@dataclass
class MockUser:
    """Mock authenticated user for testing secure handlers."""

    user_id: str = "test-user-id"
    org_id: str = "test-org-id"
    email: str = "test@example.com"
    name: str = "Test User"
    roles: List[str] = field(default_factory=lambda: ["member"])
    permissions: List[str] = field(default_factory=list)


@pytest.fixture
def mock_user():
    """Create a mock authenticated user."""
    return MockUser()


@pytest.fixture
def mock_admin_user():
    """Create a mock admin user."""
    return MockUser(
        user_id="admin-user-id",
        roles=["admin"],
        permissions=["notifications.read", "notifications.write", "admin.system"],
    )


# ===========================================================================
# Integration Cache Reset Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_slack_globals():
    """Reset Slack global state before each test."""
    # Import and reset Slack module globals
    try:
        from aragora.server.handlers.social import slack

        # Reset singleton instances
        slack._slack_integration = None
        slack._slack_handler = None
        slack._workspace_store = None
        slack._slack_user_limiter = None
        slack._slack_workspace_limiter = None
        slack._slack_audit = None
    except ImportError:
        pass  # Module may not be available in all test contexts
    yield


@pytest.fixture(autouse=True)
def reset_teams_globals():
    """Reset Teams global state before each test."""
    try:
        from aragora.server.handlers.social import teams

        teams._teams_connector = None
    except ImportError:
        pass
    yield


@pytest.fixture(autouse=True)
def reset_notifications_globals():
    """Reset notifications global state before each test."""
    try:
        from aragora.server.handlers.social import notifications

        notifications._org_email_integrations.clear()
        notifications._org_telegram_integrations.clear()
        notifications._system_email_integration = None
        notifications._system_telegram_integration = None
    except ImportError:
        pass
    yield


# ===========================================================================
# Rate Limiter Fixtures
# ===========================================================================


@pytest.fixture
def mock_rate_limiter_allowed():
    """Mock rate limiter that always allows requests."""
    limiter = MagicMock()
    limiter.is_allowed.return_value = True
    limiter.allow.return_value = MagicMock(allowed=True, retry_after=0)
    return limiter


@pytest.fixture
def mock_rate_limiter_blocked():
    """Mock rate limiter that blocks requests."""
    limiter = MagicMock()
    limiter.is_allowed.return_value = False
    limiter.allow.return_value = MagicMock(allowed=False, retry_after=30)
    return limiter


# ===========================================================================
# Webhook Security Fixtures
# ===========================================================================


@pytest.fixture
def mock_verify_slack_signature_success():
    """Mock successful Slack signature verification."""
    with patch("aragora.connectors.chat.webhook_security.verify_slack_signature") as mock:
        mock.return_value = MagicMock(verified=True, error=None)
        yield mock


@pytest.fixture
def mock_verify_slack_signature_failure():
    """Mock failed Slack signature verification."""
    with patch("aragora.connectors.chat.webhook_security.verify_slack_signature") as mock:
        mock.return_value = MagicMock(verified=False, error="Invalid signature")
        yield mock
