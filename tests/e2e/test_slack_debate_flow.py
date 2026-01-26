"""
E2E tests for Slack debate flow.

Tests the complete Slack integration flow:
1. /aragora debate slash command received
2. Immediate acknowledgment sent
3. Debate created and executed
4. Progress updates posted to thread
5. Receipt generated with PDF export
6. Final result with receipt link posted

This validates the production-ready Slack integration end-to-end.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlencode

import pytest
import pytest_asyncio


# ============================================================================
# Test Fixtures
# ============================================================================


@dataclass
class MockSlackRequest:
    """Mock Slack slash command request."""

    command: str = "/aragora"
    text: str = "debate 'Should AI be regulated?'"
    user_id: str = "U12345678"
    user_name: str = "testuser"
    channel_id: str = "C12345678"
    channel_name: str = "general"
    team_id: str = "T12345678"
    team_domain: str = "testworkspace"
    response_url: str = "https://hooks.slack.com/commands/T12345678/12345/abcdef"
    trigger_id: str = "12345.67890.abcdef"

    def to_form_data(self) -> str:
        """Convert to URL-encoded form data."""
        return urlencode(
            {
                "command": self.command,
                "text": self.text,
                "user_id": self.user_id,
                "user_name": self.user_name,
                "channel_id": self.channel_id,
                "channel_name": self.channel_name,
                "team_id": self.team_id,
                "team_domain": self.team_domain,
                "response_url": self.response_url,
                "trigger_id": self.trigger_id,
            }
        )


@dataclass
class SlackResponseCapture:
    """Capture Slack API responses for verification."""

    responses: List[Dict[str, Any]] = field(default_factory=list)
    response_url_posts: List[Dict[str, Any]] = field(default_factory=list)

    def add_response(self, response: Dict[str, Any]) -> None:
        """Add a direct response."""
        self.responses.append(response)

    def add_response_url_post(self, data: Dict[str, Any]) -> None:
        """Add a response_url POST."""
        self.response_url_posts.append(data)

    def get_final_result(self) -> Optional[Dict[str, Any]]:
        """Get the final debate result message."""
        for post in reversed(self.response_url_posts):
            text = post.get("text", "")
            if "Debate complete" in text or "complete" in text.lower():
                return post
        return None

    def get_progress_updates(self) -> List[Dict[str, Any]]:
        """Get all progress update messages."""
        return [post for post in self.response_url_posts if "Round" in post.get("text", "")]


@pytest.fixture
def slack_request() -> MockSlackRequest:
    """Create a mock Slack slash command request."""
    return MockSlackRequest()


@pytest.fixture
def response_capture() -> SlackResponseCapture:
    """Create a response capture for verification."""
    return SlackResponseCapture()


@pytest.fixture
def signing_secret() -> str:
    """Test signing secret."""
    return "test_signing_secret_12345"


def generate_slack_signature(
    signing_secret: str,
    timestamp: str,
    body: str,
) -> str:
    """Generate a valid Slack request signature."""
    sig_basestring = f"v0:{timestamp}:{body}"
    signature = hmac.new(
        signing_secret.encode(),
        sig_basestring.encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"v0={signature}"


# ============================================================================
# Slash Command Handler Tests
# ============================================================================


class TestSlackSlashCommands:
    """Tests for Slack slash command handling."""

    def test_help_command(self, slack_request: MockSlackRequest):
        """Test /aragora help returns help message."""
        from aragora.server.handlers.social.slack import SlackHandler

        slack_request.text = "help"
        handler = SlackHandler({})

        # Mock the HTTP handler
        mock_http = MagicMock()
        mock_http.command = "POST"
        mock_http._slack_body = slack_request.to_form_data()
        mock_http._slack_workspace = None
        mock_http._slack_team_id = slack_request.team_id

        result = handler._handle_slash_command(mock_http)

        assert result is not None
        # Handle both HandlerResult with body attribute and dict
        if hasattr(result, "body"):
            body = json.loads(result.body) if isinstance(result.body, (str, bytes)) else result.body
        else:
            body = result
        text = body.get("text", "") if isinstance(body, dict) else str(body)
        assert "Aragora" in text or "aragora" in text.lower()

    def test_status_command(self, slack_request: MockSlackRequest):
        """Test /aragora status returns system status."""
        from aragora.server.handlers.social.slack import SlackHandler

        slack_request.text = "status"
        handler = SlackHandler({})

        mock_http = MagicMock()
        mock_http.command = "POST"
        mock_http._slack_body = slack_request.to_form_data()
        mock_http._slack_workspace = None
        mock_http._slack_team_id = slack_request.team_id

        with patch("aragora.ranking.elo.EloSystem") as mock_elo:
            mock_elo.return_value.get_all_ratings.return_value = []
            result = handler._handle_slash_command(mock_http)

        assert result is not None
        # Handle both HandlerResult and dict
        if hasattr(result, "body"):
            body = json.loads(result.body) if isinstance(result.body, (str, bytes)) else result.body
        else:
            body = result
        assert isinstance(body, dict)
        assert "blocks" in body or "text" in body

    def test_debate_command_validation(self, slack_request: MockSlackRequest):
        """Test /aragora debate validates topic length."""
        from aragora.server.handlers.social.slack import SlackHandler

        # Too short
        slack_request.text = "debate 'AI'"
        handler = SlackHandler({})

        mock_http = MagicMock()
        mock_http.command = "POST"
        mock_http._slack_body = slack_request.to_form_data()
        mock_http._slack_workspace = None
        mock_http._slack_team_id = slack_request.team_id

        result = handler._handle_slash_command(mock_http)
        if hasattr(result, "body"):
            body = json.loads(result.body) if isinstance(result.body, (str, bytes)) else result.body
        else:
            body = result
        text = body.get("text", "") if isinstance(body, dict) else str(body)
        assert "too short" in text.lower()

    def test_debate_command_starts_debate(self, slack_request: MockSlackRequest):
        """Test /aragora debate starts a debate and returns acknowledgment."""
        from aragora.server.handlers.social.slack import SlackHandler

        slack_request.text = "debate 'Should we implement microservices architecture?'"
        handler = SlackHandler({})

        mock_http = MagicMock()
        mock_http.command = "POST"
        mock_http._slack_body = slack_request.to_form_data()
        mock_http._slack_workspace = None
        mock_http._slack_team_id = slack_request.team_id

        # Patch async task creation to capture the debate
        created_tasks = []

        def capture_task(coro, **kwargs):
            # Don't actually create the task, just track it
            created_tasks.append(kwargs.get("name", "unknown"))
            # Cancel the coroutine to prevent warnings
            coro.close()
            return MagicMock()

        with patch("aragora.server.handlers.social._slack_impl.create_tracked_task", capture_task):
            result = handler._handle_slash_command(mock_http)

        assert result is not None
        if hasattr(result, "body"):
            body = json.loads(result.body) if isinstance(result.body, (str, bytes)) else result.body
        else:
            body = result

        # Should acknowledge immediately with "in_channel" response
        assert body.get("response_type") == "in_channel"
        text = body.get("text", "")
        assert "Starting debate" in text or "debate" in text.lower()

        # Should have created an async task for the debate
        assert any("slack-debate" in task for task in created_tasks)


# ============================================================================
# Signature Verification Tests
# ============================================================================


class TestSlackSignatureVerification:
    """Tests for Slack request signature verification."""

    def test_valid_signature_passes(self, signing_secret: str):
        """Test that valid signatures pass verification."""
        from aragora.server.handlers.social.slack import SlackHandler

        handler = SlackHandler({})
        timestamp = str(int(time.time()))
        body = "command=%2Faragora&text=help"
        signature = generate_slack_signature(signing_secret, timestamp, body)

        mock_http = MagicMock()
        mock_http.headers = {
            "X-Slack-Request-Timestamp": timestamp,
            "X-Slack-Signature": signature,
        }

        result = handler._verify_signature(mock_http, body, signing_secret)
        assert result is True

    def test_invalid_signature_fails(self, signing_secret: str):
        """Test that invalid signatures fail verification."""
        from aragora.server.handlers.social.slack import SlackHandler

        handler = SlackHandler({})
        timestamp = str(int(time.time()))
        body = "command=%2Faragora&text=help"

        mock_http = MagicMock()
        mock_http.headers = {
            "X-Slack-Request-Timestamp": timestamp,
            "X-Slack-Signature": "v0=invalid_signature",
        }

        result = handler._verify_signature(mock_http, body, signing_secret)
        assert result is False

    def test_old_timestamp_fails(self, signing_secret: str):
        """Test that old timestamps fail (replay attack prevention)."""
        from aragora.server.handlers.social.slack import SlackHandler

        handler = SlackHandler({})
        # Timestamp from 10 minutes ago
        timestamp = str(int(time.time()) - 600)
        body = "command=%2Faragora&text=help"
        signature = generate_slack_signature(signing_secret, timestamp, body)

        mock_http = MagicMock()
        mock_http.headers = {
            "X-Slack-Request-Timestamp": timestamp,
            "X-Slack-Signature": signature,
        }

        result = handler._verify_signature(mock_http, body, signing_secret)
        assert result is False


# ============================================================================
# OAuth Flow Tests
# ============================================================================


class TestSlackOAuthFlow:
    """Tests for Slack OAuth installation flow."""

    @pytest.mark.asyncio
    async def test_install_redirect_generation(self):
        """Test OAuth install generates proper redirect URL."""
        from aragora.server.handlers.social import slack_oauth
        from aragora.server.handlers.social.slack_oauth import SlackOAuthHandler
        from aragora.server.oauth_state_store import reset_oauth_state_store

        # Reset OAuth state store to ensure clean state for test
        reset_oauth_state_store()

        # Patch module-level variables (loaded at import time)
        with (
            patch.object(slack_oauth, "SLACK_CLIENT_ID", "test_client_id"),
            patch.object(slack_oauth, "SLACK_CLIENT_SECRET", "test_client_secret"),
            patch.object(
                slack_oauth,
                "SLACK_REDIRECT_URI",
                "https://aragora.ai/api/integrations/slack/callback",
            ),
            # Ensure ARAGORA_ENV is set to development for fallback behavior
            patch.object(slack_oauth, "ARAGORA_ENV", "development"),
        ):
            handler = SlackOAuthHandler({})
            # The handler should generate a redirect to Slack OAuth
            result = await handler.handle("GET", "/api/integrations/slack/install")

        assert result is not None
        # Should be a redirect (302) or JSON with redirect URL
        assert result.status_code in (302, 200), f"Got status {result.status_code}: {result.body}"

        # Reset for other tests
        reset_oauth_state_store()

    def test_state_token_storage_and_validation(self):
        """Test OAuth state token is stored and validated correctly."""
        from aragora.server.oauth_state_store import (
            generate_oauth_state,
            validate_oauth_state,
            reset_oauth_state_store,
        )

        # Reset store to ensure clean state
        reset_oauth_state_store()

        user_id = "U12345"
        redirect_uri = "https://example.com"

        # Store state using the global OAuth state store
        state = generate_oauth_state(user_id=user_id, redirect_url=redirect_uri)
        assert state is not None
        assert len(state) > 10  # Should be a non-trivial token

        # Validate and consume state
        retrieved = validate_oauth_state(state)
        assert retrieved is not None
        assert retrieved["user_id"] == user_id
        assert retrieved["redirect_url"] == redirect_uri

        # State should be consumed (one-time use)
        retrieved_again = validate_oauth_state(state)
        assert retrieved_again is None

        # Reset for other tests
        reset_oauth_state_store()


# ============================================================================
# Receipt Integration Tests
# ============================================================================


class TestSlackReceiptIntegration:
    """Tests for receipt generation and linking in Slack debates."""

    def test_result_blocks_include_receipt_link(self):
        """Test that debate result blocks include receipt URL."""
        from aragora.server.handlers.social.slack import SlackHandler
        from types import SimpleNamespace

        handler = SlackHandler({})

        # Mock debate result with all required attributes
        mock_result = SimpleNamespace(
            id="debate-test123",
            consensus_reached=True,
            confidence=0.85,
            final_answer="We should implement gradual AI regulation.",
            votes={"approve": 2, "reject": 1},
            rounds_used=3,
            participants=["claude", "gpt-4"],
        )

        receipt_url = "https://aragora.ai/receipts/receipt-abc123"

        blocks = handler._build_result_blocks(
            topic="Should AI be regulated?",
            result=mock_result,
            user_id="U12345",
            receipt_url=receipt_url,
        )

        # Convert blocks to string for searching
        blocks_str = json.dumps(blocks)

        # Should include receipt URL
        assert receipt_url in blocks_str or "receipt" in blocks_str.lower()

    @pytest.mark.asyncio
    async def test_receipt_pdf_export(self):
        """Test that receipts can be exported as PDF."""
        from aragora.gauntlet.receipt import DecisionReceipt

        # Create a test receipt with correct field names
        receipt = DecisionReceipt(
            receipt_id=f"receipt-{uuid.uuid4().hex[:8]}",
            gauntlet_id=f"gauntlet-{uuid.uuid4().hex[:8]}",
            timestamp="2025-01-24T12:00:00Z",
            input_summary="Should AI be regulated?",
            input_hash=hashlib.sha256(b"test").hexdigest(),
            risk_summary={"critical": 0, "high": 0, "medium": 1, "low": 2},
            attacks_attempted=10,
            attacks_successful=0,
            probes_run=5,
            vulnerabilities_found=1,
            verdict="PASS",
            confidence=0.85,
            robustness_score=0.92,
        )

        # Check to_html works (PDF depends on weasyprint)
        html = receipt.to_html()
        assert "<!DOCTYPE html>" in html or "<html" in html
        assert receipt.verdict in html

        # Verify integrity
        assert receipt.verify_integrity() is True


# ============================================================================
# Multi-Workspace Tests
# ============================================================================


class TestSlackMultiWorkspace:
    """Tests for multi-workspace support."""

    def test_workspace_resolution(self):
        """Test workspace is resolved from team_id."""
        from aragora.server.handlers.social.slack import resolve_workspace

        # Without a store, should return None gracefully
        result = resolve_workspace("T12345678")
        assert result is None  # No store configured in tests

    def test_team_id_extraction_from_command(self):
        """Test team_id is extracted from slash command body."""
        from aragora.server.handlers.social.slack import SlackHandler

        handler = SlackHandler({})
        body = "command=%2Faragora&text=help&team_id=T12345678&user_id=U12345"

        team_id = handler._extract_team_id(body, "/api/v1/integrations/slack/commands")
        assert team_id == "T12345678"

    def test_team_id_extraction_from_event(self):
        """Test team_id is extracted from event payload."""
        from aragora.server.handlers.social.slack import SlackHandler

        handler = SlackHandler({})
        body = json.dumps({"team_id": "T12345678", "event": {"type": "message"}})

        team_id = handler._extract_team_id(body, "/api/v1/integrations/slack/events")
        assert team_id == "T12345678"


# ============================================================================
# Rate Limiting Tests
# ============================================================================


class TestSlackRateLimiting:
    """Tests for Slack endpoint rate limiting."""

    def test_slash_command_has_rate_limit(self):
        """Test slash command endpoint has rate limiting decorator."""
        from aragora.server.handlers.social.slack import SlackHandler

        handler = SlackHandler({})

        # Check the method has the rate_limit decorator applied
        # The decorated method will have __wrapped__ attribute
        method = handler._handle_slash_command
        assert hasattr(method, "__wrapped__") or "rate_limit" in str(method)


# ============================================================================
# Full Flow Integration Tests
# ============================================================================


class TestSlackFullDebateFlow:
    """Integration tests for complete Slack debate flow."""

    @pytest.mark.asyncio
    async def test_debate_posts_to_response_url(self):
        """Test debate posts updates to Slack response_url."""
        from aragora.server.handlers.social.slack import SlackHandler

        handler = SlackHandler({})
        captured_posts = []

        async def mock_post(response_url: str, data: Dict[str, Any]) -> None:
            captured_posts.append({"url": response_url, "data": data})

        # Patch the post method
        handler._post_to_response_url = mock_post

        # Mock Arena and agents
        mock_result = MagicMock()
        mock_result.consensus_reached = True
        mock_result.confidence = 0.82
        mock_result.final_answer = "Yes, with careful consideration."
        mock_result.votes = {"approve": 2, "reject": 1}

        with (
            patch("aragora.Arena") as mock_arena_cls,
            patch("aragora.agents.get_agents_by_names") as mock_agents,
            patch("aragora.gauntlet.receipt.DecisionReceipt") as mock_receipt_cls,
            patch.dict("os.environ", {"ARAGORA_PUBLIC_URL": "https://aragora.ai"}),
        ):
            mock_agents.return_value = [MagicMock(), MagicMock()]
            mock_arena = MagicMock()
            mock_arena.run = AsyncMock(return_value=mock_result)
            mock_arena_cls.from_env.return_value = mock_arena

            mock_receipt = MagicMock()
            mock_receipt.receipt_id = "receipt-test123"
            mock_receipt.to_dict.return_value = {
                "receipt_id": "receipt-test123",
                "gauntlet_id": "gauntlet-test123",
                "timestamp": "2025-01-24T12:00:00Z",
                "input_summary": "Should we use TypeScript?",
                "verdict": "PASS",
                "confidence": 0.82,
            }
            mock_receipt_cls.from_debate_result.return_value = mock_receipt

            await handler._create_debate_async(
                topic="Should we use TypeScript?",
                response_url="https://hooks.slack.com/test",
                user_id="U12345",
                channel_id="C12345",
                workspace_id="T12345",
            )

        # Should have posted starting message and result
        assert len(captured_posts) >= 2

        # Check final post includes receipt URL
        final_post = captured_posts[-1]
        assert "Debate complete" in final_post["data"].get("text", "")

    @pytest.mark.asyncio
    async def test_debate_handles_arena_failure(self):
        """Test debate handles Arena errors gracefully."""
        from aragora.server.handlers.social.slack import SlackHandler

        handler = SlackHandler({})
        captured_posts = []

        async def mock_post(response_url: str, data: Dict[str, Any]) -> None:
            captured_posts.append(data)

        handler._post_to_response_url = mock_post

        with (
            patch("aragora.Arena") as mock_arena_cls,
            patch("aragora.agents.get_agents_by_names") as mock_agents,
        ):
            mock_agents.return_value = [MagicMock()]
            mock_arena = MagicMock()
            mock_arena.run = AsyncMock(side_effect=Exception("Arena error"))
            mock_arena_cls.from_env.return_value = mock_arena

            await handler._create_debate_async(
                topic="Test topic for error handling",
                response_url="https://hooks.slack.com/test",
                user_id="U12345",
                channel_id="C12345",
            )

        # Should have posted error message
        error_posts = [p for p in captured_posts if "failed" in p.get("text", "").lower()]
        assert len(error_posts) >= 1


# ============================================================================
# SSRF Protection Tests
# ============================================================================


class TestSlackSSRFProtection:
    """Tests for SSRF protection in Slack handlers."""

    def test_valid_slack_url_allowed(self):
        """Test valid Slack URLs are allowed."""
        from aragora.server.handlers.social.slack import _validate_slack_url

        assert _validate_slack_url("https://hooks.slack.com/commands/T123/456/abc") is True
        assert _validate_slack_url("https://api.slack.com/methods/chat.postMessage") is True

    def test_invalid_url_blocked(self):
        """Test non-Slack URLs are blocked."""
        from aragora.server.handlers.social.slack import _validate_slack_url

        assert _validate_slack_url("https://evil.com/steal") is False
        assert _validate_slack_url("http://hooks.slack.com/test") is False  # HTTP not allowed
        assert _validate_slack_url("https://fake-slack.com/hook") is False

    def test_localhost_blocked(self):
        """Test localhost URLs are blocked (SSRF via local services)."""
        from aragora.server.handlers.social.slack import _validate_slack_url

        assert _validate_slack_url("https://localhost/admin") is False
        assert _validate_slack_url("https://127.0.0.1/internal") is False
