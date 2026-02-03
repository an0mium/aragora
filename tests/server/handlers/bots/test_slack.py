"""Comprehensive tests for Slack bot handler.

Tests cover:
- Signature verification (valid, invalid, old timestamp)
- Block Kit message builders
- Event handlers (URL verification, app_mention, message)
- Slash commands (ask, status, help, leaderboard, vote, unknown)
- Interactive components (vote buttons, shortcuts, modal submissions)
- SlackHandler class (routing, status, webhooks, authentication)
- Global state management (_active_debates, _user_votes)
- Error handling and edge cases
- Vote counting and debate session management
"""

import asyncio
import hashlib
import hmac
import json
import time
from urllib.parse import urlencode
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import os
import types

import pytest

from aragora.server.handlers.bots import slack
from aragora.server.handlers.bots.slack import (
    AGENT_DISPLAY_NAMES,
    SlackHandler,
    _active_debates,
    _user_votes,
    build_consensus_message_blocks,
    build_debate_message_blocks,
    build_debate_result_blocks,
    get_debate_vote_counts,
    get_slack_integration,
    handle_slack_commands,
    handle_slack_events,
    handle_slack_interactions,
    register_slack_routes,
    verify_slack_signature,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(
        self,
        headers: dict | None = None,
        body: bytes = b"",
        path: str = "/",
        method: str = "POST",
    ):
        self.headers = headers or {}
        self._body = body
        self.path = path
        self.command = method
        self.rfile = BytesIO(body)

    def get(self, key: str, default: str = "") -> str:
        return self.headers.get(key, default)


class MockRequest:
    """Mock async request for handler testing."""

    def __init__(self, body: bytes = b""):
        self._body = body

    async def body(self) -> bytes:
        return self._body


@pytest.fixture
def mock_server_context():
    """Create a mock server context for handler initialization."""
    return {
        "storage": MagicMock(),
        "user_store": MagicMock(),
        "elo_system": MagicMock(),
        "continuum_memory": MagicMock(),
        "critique_store": MagicMock(),
        "document_store": MagicMock(),
        "evidence_store": MagicMock(),
        "usage_tracker": MagicMock(),
    }


@pytest.fixture
def slack_handler(mock_server_context):
    """Create a SlackHandler instance with mocked signing secret."""
    with patch.dict(os.environ, {"SLACK_SIGNING_SECRET": "test_secret"}):
        handler = SlackHandler(mock_server_context)
    return handler


@pytest.fixture(autouse=True)
def clean_state():
    """Clean global state before and after each test."""
    _active_debates.clear()
    _user_votes.clear()
    yield
    _active_debates.clear()
    _user_votes.clear()


def compute_slack_signature(body: bytes, timestamp: str, signing_secret: str) -> str:
    """Helper to compute valid Slack signature."""
    sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    return (
        "v0="
        + hmac.new(
            signing_secret.encode(),
            sig_basestring.encode(),
            hashlib.sha256,
        ).hexdigest()
    )


# =============================================================================
# Test Signature Verification
# =============================================================================


class TestSlackSignatureVerification:
    """Tests for Slack request signature verification."""

    def test_verify_valid_signature(self):
        """Should verify a valid Slack signature."""
        body = b'{"test": "data"}'
        timestamp = str(int(time.time()))
        signing_secret = "test_secret_key"
        expected_sig = compute_slack_signature(body, timestamp, signing_secret)

        result = verify_slack_signature(body, timestamp, expected_sig, signing_secret)
        assert result is True

    def test_reject_invalid_signature(self):
        """Should reject an invalid signature."""
        body = b'{"test": "data"}'
        timestamp = str(int(time.time()))
        signing_secret = "test_secret_key"
        invalid_sig = "v0=invalid_signature_hash"

        result = verify_slack_signature(body, timestamp, invalid_sig, signing_secret)
        assert result is False

    def test_reject_old_timestamp(self):
        """Should reject requests with timestamps older than 5 minutes."""
        body = b'{"test": "data"}'
        old_timestamp = str(int(time.time()) - 400)  # 6+ minutes ago
        signing_secret = "test_secret_key"
        signature = compute_slack_signature(body, old_timestamp, signing_secret)

        result = verify_slack_signature(body, old_timestamp, signature, signing_secret)
        assert result is False

    def test_verify_signature_at_boundary(self):
        """Should accept request at exactly 5 minute boundary."""
        body = b'{"test": "data"}'
        timestamp = str(int(time.time()) - 299)  # Just under 5 minutes
        signing_secret = "test_secret_key"
        signature = compute_slack_signature(body, timestamp, signing_secret)

        result = verify_slack_signature(body, timestamp, signature, signing_secret)
        assert result is True

    def test_reject_future_timestamp(self):
        """Should handle future timestamps correctly."""
        body = b'{"test": "data"}'
        future_timestamp = str(int(time.time()) + 400)  # 6+ minutes in future
        signing_secret = "test_secret_key"
        signature = compute_slack_signature(body, future_timestamp, signing_secret)

        # abs() check means future timestamps also rejected if too far
        result = verify_slack_signature(body, future_timestamp, signature, signing_secret)
        assert result is False

    def test_empty_body_signature(self):
        """Should handle empty body correctly."""
        body = b""
        timestamp = str(int(time.time()))
        signing_secret = "test_secret_key"
        signature = compute_slack_signature(body, timestamp, signing_secret)

        result = verify_slack_signature(body, timestamp, signature, signing_secret)
        assert result is True


# =============================================================================
# Test Block Kit Builders
# =============================================================================


class TestSlackBlockBuilders:
    """Tests for Slack Block Kit message builders."""

    def test_build_debate_message_blocks(self):
        """Should build valid debate message blocks."""
        blocks = build_debate_message_blocks(
            debate_id="test-123",
            task="Should we adopt microservices?",
            agents=["Claude", "GPT-4"],
            current_round=2,
            total_rounds=5,
            include_vote_buttons=True,
        )

        assert isinstance(blocks, list)
        assert len(blocks) > 0

        # Check for header
        header_block = blocks[0]
        assert header_block["type"] == "header"

        # Check for task section
        task_block = blocks[1]
        assert task_block["type"] == "section"
        assert "microservices" in task_block["text"]["text"]

    def test_build_debate_message_without_vote_buttons(self):
        """Should build message without vote buttons when disabled."""
        blocks = build_debate_message_blocks(
            debate_id="test-123",
            task="Test task",
            agents=["Claude"],
            current_round=1,
            total_rounds=3,
            include_vote_buttons=False,
        )

        # Should not contain vote prompt text
        block_texts = [b.get("text", {}).get("text", "") for b in blocks if "text" in b]
        assert not any("Cast your vote" in t for t in block_texts)

    def test_build_debate_message_max_agents(self):
        """Should limit vote buttons to 5 agents maximum."""
        agents = ["Claude", "GPT-4", "Gemini", "Mistral", "DeepSeek", "Grok", "Qwen"]
        blocks = build_debate_message_blocks(
            debate_id="test-123",
            task="Test task",
            agents=agents,
            current_round=1,
            total_rounds=3,
            include_vote_buttons=True,
        )

        # Find actions block with vote buttons
        action_blocks = [b for b in blocks if b.get("type") == "actions"]
        # First actions block has vote buttons
        if action_blocks:
            vote_buttons = [
                elem
                for elem in action_blocks[0].get("elements", [])
                if elem.get("action_id", "").startswith("vote_")
            ]
            assert len(vote_buttons) <= 5

    def test_build_debate_message_fields_section(self):
        """Should include agents and progress fields."""
        blocks = build_debate_message_blocks(
            debate_id="test-123",
            task="Test task",
            agents=["Claude", "GPT-4"],
            current_round=3,
            total_rounds=5,
            include_vote_buttons=False,
        )

        # Find section with fields
        fields_section = None
        for block in blocks:
            if block.get("type") == "section" and "fields" in block:
                fields_section = block
                break

        assert fields_section is not None
        fields = fields_section["fields"]
        assert len(fields) >= 2

        # Check for agents field
        agents_field = fields[0]
        assert "Claude" in agents_field["text"]

        # Check for progress field
        progress_field = fields[1]
        assert "Round 3/5" in progress_field["text"]

    def test_build_consensus_message_blocks(self):
        """Should build valid consensus message blocks."""
        blocks = build_consensus_message_blocks(
            debate_id="test-123",
            task="Test debate",
            consensus_reached=True,
            confidence=0.85,
            winner="Claude",
            final_answer="The recommendation is to proceed with option A.",
            vote_counts={"Claude": 5, "GPT-4": 3},
        )

        assert isinstance(blocks, list)
        assert len(blocks) > 0

        # Check for header block
        header = blocks[0]
        assert header["type"] == "header"
        assert "Consensus" in header["text"]["text"]

    def test_build_consensus_message_blocks_no_consensus(self):
        """Should build consensus blocks when no consensus reached."""
        blocks = build_consensus_message_blocks(
            debate_id="test-456",
            task="Contentious topic",
            consensus_reached=False,
            confidence=0.45,
            winner=None,
            final_answer=None,
            vote_counts={},
        )

        assert isinstance(blocks, list)
        assert len(blocks) > 0

        # Check header shows no consensus
        header = blocks[0]
        assert "No Consensus" in header["text"]["text"]

    def test_build_consensus_truncates_long_answer(self):
        """Should truncate final answer over 500 characters."""
        long_answer = "A" * 600
        blocks = build_consensus_message_blocks(
            debate_id="test-123",
            task="Test",
            consensus_reached=True,
            confidence=0.9,
            winner="Claude",
            final_answer=long_answer,
            vote_counts={},
        )

        # Find the answer block
        for block in blocks:
            if block.get("type") == "section" and "Decision" in block.get("text", {}).get(
                "text", ""
            ):
                assert "..." in block["text"]["text"]
                break

    def test_build_consensus_vote_counts_sorted(self):
        """Should display vote counts sorted by count descending."""
        blocks = build_consensus_message_blocks(
            debate_id="test-123",
            task="Test",
            consensus_reached=True,
            confidence=0.8,
            winner="Claude",
            final_answer="Decision",
            vote_counts={"GPT-4": 2, "Claude": 5, "Gemini": 3},
        )

        # Find votes section
        for block in blocks:
            text = block.get("text", {}).get("text", "")
            if "User Votes" in text:
                # Claude should appear before Gemini which appears before GPT-4
                claude_pos = text.find("Claude")
                gemini_pos = text.find("Gemini")
                gpt_pos = text.find("GPT-4")
                assert claude_pos < gemini_pos < gpt_pos
                break

    def test_build_debate_result_blocks_alias(self):
        """Should use the backward compatibility alias correctly."""
        # build_debate_result_blocks should be an alias for build_consensus_message_blocks
        assert build_debate_result_blocks is build_consensus_message_blocks


# =============================================================================
# Test Event Handlers
# =============================================================================


class TestSlackEventHandler:
    """Tests for Slack Events API handler."""

    @pytest.mark.asyncio
    async def test_handle_url_verification(self):
        """Should respond to URL verification challenge."""
        request = MockRequest(
            body=json.dumps(
                {"type": "url_verification", "challenge": "test_challenge_123"}
            ).encode()
        )

        result = await handle_slack_events(request)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["challenge"] == "test_challenge_123"

    @pytest.mark.asyncio
    async def test_handle_app_mention(self):
        """Should handle app mention events."""
        request = MockRequest(
            body=json.dumps(
                {
                    "type": "event_callback",
                    "event": {
                        "type": "app_mention",
                        "text": "@aragora ask about microservices",
                        "channel": "C123456",
                        "user": "U123456",
                        "files": [
                            {
                                "id": "F123",
                                "name": "spec.txt",
                                "mimetype": "text/plain",
                                "preview_plain_text": "Spec details",
                            }
                        ],
                    },
                }
            ).encode()
        )

        result = await handle_slack_events(request)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "response_type" in body

    @pytest.mark.asyncio
    async def test_handle_app_mention_routes_attachments(self):
        """Should route Slack attachments into DecisionRequest."""
        request = MockRequest(
            body=json.dumps(
                {
                    "type": "event_callback",
                    "event": {
                        "type": "app_mention",
                        "text": "@aragora ask about microservices",
                        "channel": "C123456",
                        "user": "U123456",
                        "files": [
                            {
                                "id": "F123",
                                "name": "spec.txt",
                                "mimetype": "text/plain",
                                "preview_plain_text": "Spec details",
                            }
                        ],
                    },
                }
            ).encode()
        )

        captured = {}

        async def fake_route(req):
            captured["request"] = req
            return MagicMock()

        with patch("aragora.core.get_decision_router") as mock_get:
            mock_get.return_value = MagicMock(route=fake_route)
            result = await handle_slack_events(request)
            await asyncio.sleep(0)

        assert result.status_code == 200
        routed = captured.get("request")
        assert routed is not None
        assert routed.attachments
        assert routed.attachments[0]["file_id"] == "F123"

    @pytest.mark.asyncio
    async def test_handle_app_mention_downloads_file(self):
        """Should hydrate Slack file attachments with downloaded data."""
        request = MockRequest(
            body=json.dumps(
                {
                    "type": "event_callback",
                    "event": {
                        "type": "app_mention",
                        "text": "@aragora ask about microservices",
                        "channel": "C123456",
                        "user": "U123456",
                        "files": [
                            {
                                "id": "F123",
                                "name": "spec.txt",
                                "mimetype": "text/plain",
                                "preview_plain_text": "Spec details",
                            }
                        ],
                    },
                }
            ).encode()
        )

        captured = {}

        async def fake_route(req):
            captured["request"] = req
            return MagicMock()

        class FakeSlackConnector:
            bot_token = "token"

            async def download_file(self, file_id: str):
                return types.SimpleNamespace(
                    content=b"hello",
                    filename="spec.txt",
                    content_type="text/plain",
                    size=5,
                )

        with patch("aragora.connectors.chat.registry.get_connector") as mock_get_connector:
            mock_get_connector.return_value = FakeSlackConnector()
            with patch("aragora.core.get_decision_router") as mock_get_router:
                mock_get_router.return_value = MagicMock(route=fake_route)
                result = await handle_slack_events(request)
                await asyncio.sleep(0)

        assert result.status_code == 200
        routed = captured.get("request")
        assert routed is not None
        assert routed.attachments
        assert routed.attachments[0]["data"] == b"hello"

    @pytest.mark.asyncio
    async def test_handle_message_event(self):
        """Should handle message events."""
        request = MockRequest(
            body=json.dumps(
                {
                    "type": "event_callback",
                    "event": {
                        "type": "message",
                        "text": "Hello bot",
                        "channel": "C123456",
                        "user": "U123456",
                    },
                }
            ).encode()
        )

        result = await handle_slack_events(request)

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self):
        """Should handle invalid JSON gracefully."""
        request = MockRequest(body=b"not valid json")

        result = await handle_slack_events(request)

        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_handle_empty_event(self):
        """Should handle empty event object."""
        request = MockRequest(body=json.dumps({"type": "event_callback"}).encode())

        result = await handle_slack_events(request)

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_unknown_event_type(self):
        """Should handle unknown event types gracefully."""
        request = MockRequest(
            body=json.dumps(
                {
                    "type": "event_callback",
                    "event": {
                        "type": "unknown_event",
                        "data": "some data",
                    },
                }
            ).encode()
        )

        result = await handle_slack_events(request)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body.get("ok") is True


# =============================================================================
# Test Slash Commands
# =============================================================================


class TestSlackCommandHandler:
    """Tests for Slack slash command handler."""

    @pytest.mark.asyncio
    async def test_handle_ask_command(self):
        """Should handle /aragora ask command."""
        body = (
            "command=/aragora&"
            "text=ask+Should+we+use+microservices&"
            "user_id=U123&"
            "user_name=testuser&"
            "channel_id=C123&"
            "response_url=https://hooks.slack.com/test"
        )
        request = MockRequest(body=body.encode())

        # Mock the debate starting to avoid timeout
        with patch.object(slack, "_start_slack_debate", new_callable=AsyncMock) as mock_start:
            mock_start.return_value = "test-debate-id-123"
            result = await handle_slack_commands(request)

        assert result.status_code == 200
        response_body = json.loads(result.body)
        assert response_body["response_type"] == "in_channel"
        assert "microservices" in response_body["text"]

    @pytest.mark.asyncio
    async def test_handle_ask_command_with_attachments(self):
        """Should pass attachments through to debate starter."""
        attachments = [{"filename": "spec.txt", "content": "hello"}]
        body = urlencode(
            {
                "command": "/aragora",
                "text": "ask Should we use microservices",
                "user_id": "U123",
                "user_name": "testuser",
                "channel_id": "C123",
                "response_url": "https://hooks.slack.com/test",
                "attachments": json.dumps(attachments),
            }
        )
        request = MockRequest(body=body.encode())

        with patch(
            "aragora.server.handlers.bots.slack.commands.start_slack_debate",
            new_callable=AsyncMock,
        ) as mock_start:
            mock_start.return_value = "test-debate-id-123"
            result = await handle_slack_commands(request)

        assert result.status_code == 200
        _, kwargs = mock_start.call_args
        assert kwargs.get("attachments") == attachments

    @pytest.mark.asyncio
    async def test_handle_ask_command_has_blocks(self):
        """Should return Block Kit blocks with ask command."""
        body = "command=/aragora&text=ask+What+is+the+meaning+of+life&user_id=U123&channel_id=C123"
        request = MockRequest(body=body.encode())

        # Mock the debate starting to avoid timeout
        with patch.object(slack, "_start_slack_debate", new_callable=AsyncMock) as mock_start:
            mock_start.return_value = "test-debate-id-123"
            result = await handle_slack_commands(request)

        assert result.status_code == 200
        response_body = json.loads(result.body)
        assert "blocks" in response_body
        assert isinstance(response_body["blocks"], list)

    @pytest.mark.asyncio
    async def test_handle_status_command(self):
        """Should handle /aragora status command."""
        body = "command=/aragora&text=status&user_id=U123"
        request = MockRequest(body=body.encode())

        result = await handle_slack_commands(request)

        assert result.status_code == 200
        response_body = json.loads(result.body)
        assert response_body["response_type"] == "ephemeral"
        assert "active debate" in response_body["text"]

    @pytest.mark.asyncio
    async def test_handle_help_command(self):
        """Should handle /aragora help command."""
        body = "command=/aragora&text=help&user_id=U123"
        request = MockRequest(body=body.encode())

        result = await handle_slack_commands(request)

        assert result.status_code == 200
        response_body = json.loads(result.body)
        assert "Aragora Commands" in response_body["text"]
        assert "/aragora ask" in response_body["text"]

    @pytest.mark.asyncio
    async def test_handle_leaderboard_command(self):
        """Should handle /aragora leaderboard command."""
        body = "command=/aragora&text=leaderboard&user_id=U123"
        request = MockRequest(body=body.encode())

        result = await handle_slack_commands(request)

        assert result.status_code == 200
        response_body = json.loads(result.body)
        assert response_body["response_type"] == "in_channel"
        assert "Leaderboard" in response_body["text"]

    @pytest.mark.asyncio
    async def test_handle_vote_command(self):
        """Should handle /aragora vote command."""
        body = "command=/aragora&text=vote&user_id=U123"
        request = MockRequest(body=body.encode())

        result = await handle_slack_commands(request)

        assert result.status_code == 200
        response_body = json.loads(result.body)
        assert response_body["response_type"] == "ephemeral"
        assert "vote buttons" in response_body["text"]

    @pytest.mark.asyncio
    async def test_handle_unknown_command(self):
        """Should show help for unknown commands."""
        body = "command=/aragora&text=unknowncommand&user_id=U123"
        request = MockRequest(body=body.encode())

        result = await handle_slack_commands(request)

        assert result.status_code == 200
        response_body = json.loads(result.body)
        assert "Aragora Commands" in response_body["text"]

    @pytest.mark.asyncio
    async def test_handle_empty_command(self):
        """Should show help for empty command text."""
        body = "command=/aragora&text=&user_id=U123"
        request = MockRequest(body=body.encode())

        result = await handle_slack_commands(request)

        assert result.status_code == 200
        response_body = json.loads(result.body)
        assert "Aragora Commands" in response_body["text"]

    @pytest.mark.asyncio
    async def test_handle_ask_without_question(self):
        """Should handle ask without a question."""
        body = "command=/aragora&text=ask&user_id=U123&channel_id=C123"
        request = MockRequest(body=body.encode())

        # Mock the debate starting to avoid timeout (empty args case)
        with patch.object(slack, "_start_slack_debate", new_callable=AsyncMock) as mock_start:
            mock_start.return_value = "test-debate-id-123"
            result = await handle_slack_commands(request)

        # Should show help when no question provided
        assert result.status_code == 200


# =============================================================================
# Test Interactive Components
# =============================================================================


class TestSlackInteractionHandler:
    """Tests for Slack interactive components handler."""

    @pytest.mark.asyncio
    async def test_handle_block_action_vote(self):
        """Should handle vote button clicks."""
        payload = {
            "type": "block_actions",
            "user": {"id": "U123", "name": "testuser"},
            "actions": [
                {
                    "action_id": "vote_debate123_claude",
                    "value": json.dumps({"debate_id": "debate123", "agent": "Claude"}),
                }
            ],
        }

        body = f"payload={json.dumps(payload)}"
        request = MockRequest(body=body.encode())

        with patch.object(slack, "audit_data"):
            result = await handle_slack_interactions(request)

        assert result.status_code == 200
        response_body = json.loads(result.body)
        assert response_body["response_type"] == "ephemeral"
        assert "Claude" in response_body["text"]

    @pytest.mark.asyncio
    async def test_handle_vote_records_in_state(self):
        """Should record vote in global state."""
        payload = {
            "type": "block_actions",
            "user": {"id": "U123", "name": "testuser"},
            "actions": [
                {
                    "action_id": "vote_test_claude",
                    "value": json.dumps({"debate_id": "debate-xyz", "agent": "Claude"}),
                }
            ],
        }

        body = f"payload={json.dumps(payload)}"
        request = MockRequest(body=body.encode())

        with patch.object(slack, "audit_data"):
            await handle_slack_interactions(request)

        assert "debate-xyz" in _user_votes
        assert _user_votes["debate-xyz"]["U123"] == "Claude"

    @pytest.mark.asyncio
    async def test_handle_summary_action(self):
        """Should handle summary button clicks."""
        payload = {
            "type": "block_actions",
            "user": {"id": "U123", "name": "testuser"},
            "actions": [
                {
                    "action_id": "summary_debate123",
                    "value": "debate123",
                }
            ],
        }

        body = f"payload={json.dumps(payload)}"
        request = MockRequest(body=body.encode())

        result = await handle_slack_interactions(request)

        assert result.status_code == 200
        response_body = json.loads(result.body)
        # The text truncates ID to 8 chars: "Fetching summary for debate `debate12...`"
        assert "debate12" in response_body["text"]
        assert "Fetching summary" in response_body["text"]

    @pytest.mark.asyncio
    async def test_handle_shortcut_start_debate(self):
        """Should handle start debate shortcut."""
        payload = {
            "type": "shortcut",
            "callback_id": "start_debate",
            "user": {"id": "U123"},
            "trigger_id": "trigger123",
        }

        body = f"payload={json.dumps(payload)}"
        request = MockRequest(body=body.encode())

        result = await handle_slack_interactions(request)

        assert result.status_code == 200
        response_body = json.loads(result.body)
        assert response_body.get("response_action") == "open_modal"

    @pytest.mark.asyncio
    async def test_handle_view_submission_with_task(self):
        """Should handle modal submission with valid task."""
        payload = {
            "type": "view_submission",
            "user": {"id": "U123"},
            "view": {
                "callback_id": "start_debate_modal",
                "state": {
                    "values": {
                        "task_block": {"task_input": {"value": "Test debate question"}},
                        "agents_block": {
                            "agents_select": {
                                "selected_options": [
                                    {"value": "claude"},
                                    {"value": "gpt4"},
                                ]
                            }
                        },
                        "rounds_block": {"rounds_select": {"selected_option": {"value": "5"}}},
                    }
                },
            },
        }

        body = f"payload={json.dumps(payload)}"
        request = MockRequest(body=body.encode())

        with patch.object(slack, "audit_data"):
            result = await handle_slack_interactions(request)

        assert result.status_code == 200
        response_body = json.loads(result.body)
        assert response_body.get("response_action") == "clear"

    @pytest.mark.asyncio
    async def test_handle_view_submission_missing_task(self):
        """Should return error when task is missing."""
        payload = {
            "type": "view_submission",
            "user": {"id": "U123"},
            "view": {
                "callback_id": "start_debate_modal",
                "state": {
                    "values": {
                        "task_block": {"task_input": {"value": ""}},
                        "agents_block": {
                            "agents_select": {"selected_options": [{"value": "claude"}]}
                        },
                    }
                },
            },
        }

        body = f"payload={json.dumps(payload)}"
        request = MockRequest(body=body.encode())

        result = await handle_slack_interactions(request)

        assert result.status_code == 200
        response_body = json.loads(result.body)
        assert response_body.get("response_action") == "errors"
        assert "task_block" in response_body.get("errors", {})

    @pytest.mark.asyncio
    async def test_handle_view_submission_missing_agents(self):
        """Should return error when agents are missing."""
        payload = {
            "type": "view_submission",
            "user": {"id": "U123"},
            "view": {
                "callback_id": "start_debate_modal",
                "state": {
                    "values": {
                        "task_block": {"task_input": {"value": "Test task"}},
                        "agents_block": {"agents_select": {"selected_options": []}},
                    }
                },
            },
        }

        body = f"payload={json.dumps(payload)}"
        request = MockRequest(body=body.encode())

        result = await handle_slack_interactions(request)

        assert result.status_code == 200
        response_body = json.loads(result.body)
        assert response_body.get("response_action") == "errors"
        assert "agents_block" in response_body.get("errors", {})

    @pytest.mark.asyncio
    async def test_handle_invalid_vote_json(self):
        """Should handle invalid JSON in vote value gracefully."""
        payload = {
            "type": "block_actions",
            "user": {"id": "U123", "name": "testuser"},
            "actions": [
                {
                    "action_id": "vote_test",
                    "value": "not valid json",
                }
            ],
        }

        body = f"payload={json.dumps(payload)}"
        request = MockRequest(body=body.encode())

        result = await handle_slack_interactions(request)

        # Should not crash, return ok
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_empty_payload(self):
        """Should handle empty payload gracefully."""
        body = "payload={}"
        request = MockRequest(body=body.encode())

        result = await handle_slack_interactions(request)

        assert result.status_code == 200


# =============================================================================
# Test SlackHandler Class
# =============================================================================


class TestSlackHandlerRouting:
    """Tests for SlackHandler route handling."""

    def test_can_handle_status(self, slack_handler):
        """Test handler recognizes status endpoint."""
        assert slack_handler.can_handle("/api/v1/bots/slack/status") is True

    def test_can_handle_events(self, slack_handler):
        """Test handler recognizes events endpoint."""
        assert slack_handler.can_handle("/api/v1/bots/slack/events") is True

    def test_can_handle_interactions(self, slack_handler):
        """Test handler recognizes interactions endpoint."""
        assert slack_handler.can_handle("/api/v1/bots/slack/interactions") is True

    def test_can_handle_commands(self, slack_handler):
        """Test handler recognizes commands endpoint."""
        assert slack_handler.can_handle("/api/v1/bots/slack/commands") is True

    def test_can_handle_integrations_prefix(self, slack_handler):
        """Test handler recognizes integrations prefix."""
        assert slack_handler.can_handle("/api/v1/integrations/slack/status") is True

    def test_cannot_handle_unknown(self, slack_handler):
        """Test handler rejects unknown endpoints."""
        assert slack_handler.can_handle("/api/v1/bots/discord/status") is False
        assert slack_handler.can_handle("/api/v1/other/endpoint") is False

    def test_bot_platform_set(self, slack_handler):
        """Test bot_platform is correctly set."""
        assert slack_handler.bot_platform == "slack"


class TestSlackHandlerStatus:
    """Tests for SlackHandler status endpoint."""

    def test_is_bot_enabled_with_token(self, mock_server_context):
        """Test _is_bot_enabled returns True when token is set."""
        # Need to patch the module-level constants
        original_token = slack.SLACK_BOT_TOKEN
        try:
            slack.SLACK_BOT_TOKEN = "xoxb-test-token"
            handler = SlackHandler(mock_server_context)
            assert handler._is_bot_enabled() is True
        finally:
            slack.SLACK_BOT_TOKEN = original_token

    def test_is_bot_enabled_with_signing_secret(self, mock_server_context):
        """Test _is_bot_enabled returns True when signing secret is set."""
        # Need to patch the module-level constants
        original_secret = slack.SLACK_SIGNING_SECRET
        try:
            slack.SLACK_SIGNING_SECRET = "test-secret"
            handler = SlackHandler(mock_server_context)
            assert handler._is_bot_enabled() is True
        finally:
            slack.SLACK_SIGNING_SECRET = original_secret

    def test_is_bot_disabled_without_config(self, mock_server_context):
        """Test _is_bot_enabled returns False when not configured."""
        original_token = slack.SLACK_BOT_TOKEN
        original_secret = slack.SLACK_SIGNING_SECRET
        try:
            slack.SLACK_BOT_TOKEN = None
            slack.SLACK_SIGNING_SECRET = None
            handler = SlackHandler(mock_server_context)
            assert handler._is_bot_enabled() is False
        finally:
            slack.SLACK_BOT_TOKEN = original_token
            slack.SLACK_SIGNING_SECRET = original_secret


class TestSlackHandlerWebhooks:
    """Tests for SlackHandler webhook verification."""

    def test_verify_signature_with_valid_signature(self, slack_handler):
        """Test signature verification succeeds with valid signature."""
        body = b'{"test": "data"}'
        timestamp = str(int(time.time()))
        signing_secret = "test_secret"
        signature = compute_slack_signature(body, timestamp, signing_secret)

        mock_http = MockHandler(
            headers={
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": signature,
            },
            body=body,
        )

        result = slack_handler._verify_signature(mock_http)
        assert result is True

    def test_verify_signature_with_invalid_signature(self, slack_handler):
        """Test signature verification fails with invalid signature."""
        body = b'{"test": "data"}'
        timestamp = str(int(time.time()))

        mock_http = MockHandler(
            headers={
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": "v0=invalid",
            },
            body=body,
        )

        result = slack_handler._verify_signature(mock_http)
        assert result is False

    def test_handle_webhook_without_signing_secret(self, mock_server_context):
        """Test webhook handling returns 503 without signing secret."""
        with patch.dict(os.environ, {"SLACK_SIGNING_SECRET": ""}, clear=False):
            handler = SlackHandler(mock_server_context)
            handler._signing_secret = None

            mock_http = MockHandler(
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            result = handler.handle("/api/v1/bots/slack/commands", {}, mock_http)

            assert result is not None
            assert result.status_code == 503


# =============================================================================
# Test Global State Management
# =============================================================================


class TestSlackGlobalState:
    """Tests for Slack handler global state management."""

    def test_active_debates_storage(self):
        """Should store and retrieve active debates."""
        _active_debates["test-debate-1"] = {
            "task": "Test task",
            "agents": ["Claude", "GPT-4"],
            "rounds": 5,
            "current_round": 1,
            "status": "running",
        }

        assert "test-debate-1" in _active_debates
        assert _active_debates["test-debate-1"]["task"] == "Test task"

    def test_user_votes_storage(self):
        """Should store user votes correctly."""
        _user_votes["debate-1"] = {"user-1": "claude", "user-2": "gpt4"}

        assert _user_votes["debate-1"]["user-1"] == "claude"
        assert _user_votes["debate-1"]["user-2"] == "gpt4"

    def test_get_debate_vote_counts_empty(self):
        """Should return empty counts for unknown debate."""
        counts = get_debate_vote_counts("unknown-debate")
        assert counts == {}

    def test_get_debate_vote_counts(self):
        """Should calculate vote counts correctly."""
        _user_votes["debate-123"] = {
            "user-1": "Claude",
            "user-2": "Claude",
            "user-3": "GPT-4",
        }

        counts = get_debate_vote_counts("debate-123")

        assert counts["Claude"] == 2
        assert counts["GPT-4"] == 1

    def test_vote_override(self):
        """Should allow vote override (last vote wins)."""
        _user_votes["debate-1"] = {"user-1": "Claude"}
        _user_votes["debate-1"]["user-1"] = "GPT-4"

        assert _user_votes["debate-1"]["user-1"] == "GPT-4"


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestSlackHelperFunctions:
    """Tests for helper functions in slack module."""

    def test_agent_display_names_mapping(self):
        """Should map agent IDs to display names."""
        assert AGENT_DISPLAY_NAMES["claude"] == "Claude"
        assert AGENT_DISPLAY_NAMES["gpt4"] == "GPT-4"
        assert AGENT_DISPLAY_NAMES["gemini"] == "Gemini"
        assert AGENT_DISPLAY_NAMES["anthropic-api"] == "Claude"
        assert AGENT_DISPLAY_NAMES["openai-api"] == "GPT-4"

    def test_get_slack_integration_not_configured(self):
        """Should return None when Slack is not configured."""
        with patch.dict(os.environ, {"SLACK_WEBHOOK_URL": ""}, clear=False):
            # Reset cached integration
            slack._slack_integration = None
            result = get_slack_integration()
            assert result is None

    def test_get_slack_integration_cached(self):
        """Should return cached integration on second call."""
        mock_integration = MagicMock()
        slack._slack_integration = mock_integration

        result = get_slack_integration()
        assert result is mock_integration

        # Clean up
        slack._slack_integration = None

    def test_register_slack_routes(self):
        """Should register routes with router."""
        mock_router = MagicMock()

        register_slack_routes(mock_router)

        assert mock_router.add_route.call_count == 3

        # Check routes were registered
        call_args = [call[0] for call in mock_router.add_route.call_args_list]
        paths = [args[1] for args in call_args]
        assert "/api/bots/slack/events" in paths
        assert "/api/bots/slack/interactions" in paths
        assert "/api/bots/slack/commands" in paths


# =============================================================================
# Test SlackHandler Methods
# =============================================================================


class TestSlackHandlerMethods:
    """Tests for SlackHandler helper methods."""

    def test_command_help(self, slack_handler):
        """Should return help text."""
        result = slack_handler._command_help()

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["response_type"] == "ephemeral"
        assert "/aragora debate" in body["text"]
        assert "/aragora status" in body["text"]
        assert "/aragora help" in body["text"]

    def test_command_status(self, slack_handler):
        """Should return status information."""
        result = slack_handler._command_status()

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["response_type"] == "ephemeral"
        assert "Status" in body["text"]

    def test_slack_response(self, slack_handler):
        """Should create proper Slack response."""
        result = slack_handler._slack_response("Test message", "in_channel")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["text"] == "Test message"
        assert body["response_type"] == "in_channel"

    def test_slack_response_ephemeral(self, slack_handler):
        """Should create ephemeral Slack response."""
        result = slack_handler._slack_response("Private message", "ephemeral")

        body = json.loads(result.body)
        assert body["response_type"] == "ephemeral"

    def test_slack_blocks_response(self, slack_handler):
        """Should create Slack response with blocks."""
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "Test"}}]
        result = slack_handler._slack_blocks_response(blocks, "Fallback text")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["blocks"] == blocks
        assert body["text"] == "Fallback text"

    def test_slack_blocks_response_without_fallback(self, slack_handler):
        """Should create Slack response with blocks without fallback text."""
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "Test"}}]
        result = slack_handler._slack_blocks_response(blocks)

        body = json.loads(result.body)
        assert body["blocks"] == blocks
        assert "text" not in body

    def test_get_status(self, slack_handler):
        """Should return integration status."""
        result = slack_handler._get_status()

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "enabled" in body
        assert "configured" in body
        assert "active_debates" in body
        assert "features" in body

    def test_get_status_includes_features(self, slack_handler):
        """Should include feature flags in status."""
        result = slack_handler._get_status()

        body = json.loads(result.body)
        features = body["features"]
        assert features["slash_commands"] is True
        assert features["events_api"] is True
        assert features["interactive_components"] is True
        assert features["block_kit"] is True


# =============================================================================
# Test Edge Cases and Error Handling
# =============================================================================


class TestSlackEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_commands_handler_decode_error(self):
        """Should handle unicode decode errors gracefully."""
        # Invalid UTF-8 bytes
        request = MockRequest(body=b"\xff\xfe invalid")

        result = await handle_slack_commands(request)

        # Should return error response
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "Error" in body.get("text", "")

    @pytest.mark.asyncio
    async def test_interactions_invalid_payload_format(self):
        """Should handle malformed payload parameter."""
        body = "payload=not%20valid%20json"
        request = MockRequest(body=body.encode())

        result = await handle_slack_interactions(request)

        assert result.status_code == 500

    def test_build_modal_structure(self):
        """Should build correct modal structure."""
        modal = slack._build_start_debate_modal()

        assert modal["type"] == "modal"
        assert modal["callback_id"] == "start_debate_modal"
        assert "title" in modal
        assert "submit" in modal
        assert "close" in modal
        assert "blocks" in modal

        # Check blocks have required input elements
        blocks = modal["blocks"]
        block_ids = [b.get("block_id") for b in blocks]
        assert "task_block" in block_ids
        assert "agents_block" in block_ids
        assert "rounds_block" in block_ids

    def test_handle_method_not_allowed(self, slack_handler):
        """Should return 405 for non-POST webhook requests."""
        mock_http = MockHandler(
            headers={"Content-Type": "application/json"},
            method="GET",
        )

        result = slack_handler.handle("/api/v1/bots/slack/commands", {}, mock_http)

        assert result is not None
        assert result.status_code == 405


# =============================================================================
# Test ROUTES constant
# =============================================================================


class TestSlackRoutes:
    """Tests for SlackHandler ROUTES constant."""

    def test_routes_constant_defined(self):
        """Should have ROUTES constant defined."""
        assert hasattr(SlackHandler, "ROUTES")
        assert isinstance(SlackHandler.ROUTES, list)

    def test_routes_includes_all_endpoints(self):
        """Should include all expected endpoints."""
        routes = SlackHandler.ROUTES

        assert "/api/v1/bots/slack/status" in routes
        assert "/api/v1/bots/slack/events" in routes
        assert "/api/v1/bots/slack/interactions" in routes
        assert "/api/v1/bots/slack/commands" in routes
