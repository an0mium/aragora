"""Tests for Slack bot handler."""

import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.bots import slack


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

        # Compute expected signature
        sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
        expected_sig = (
            "v0="
            + hmac.new(
                signing_secret.encode(),
                sig_basestring.encode(),
                hashlib.sha256,
            ).hexdigest()
        )

        result = slack.verify_slack_signature(body, timestamp, expected_sig, signing_secret)
        assert result is True

    def test_reject_invalid_signature(self):
        """Should reject an invalid signature."""
        body = b'{"test": "data"}'
        timestamp = str(int(time.time()))
        signing_secret = "test_secret_key"
        invalid_sig = "v0=invalid_signature_hash"

        result = slack.verify_slack_signature(body, timestamp, invalid_sig, signing_secret)
        assert result is False

    def test_reject_old_timestamp(self):
        """Should reject requests with timestamps older than 5 minutes."""
        body = b'{"test": "data"}'
        old_timestamp = str(int(time.time()) - 400)  # 6+ minutes ago
        signing_secret = "test_secret_key"

        # Even with a valid signature, old timestamp should fail
        sig_basestring = f"v0:{old_timestamp}:{body.decode('utf-8')}"
        signature = (
            "v0="
            + hmac.new(
                signing_secret.encode(),
                sig_basestring.encode(),
                hashlib.sha256,
            ).hexdigest()
        )

        result = slack.verify_slack_signature(body, old_timestamp, signature, signing_secret)
        assert result is False


# =============================================================================
# Test Block Kit Builders
# =============================================================================


class TestSlackBlockBuilders:
    """Tests for Slack Block Kit message builders."""

    def test_build_debate_message_blocks(self):
        """Should build valid debate message blocks."""
        blocks = slack.build_debate_message_blocks(
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
        blocks = slack.build_debate_message_blocks(
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

    def test_build_consensus_message_blocks(self):
        """Should build valid consensus message blocks."""
        blocks = slack.build_consensus_message_blocks(
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

    def test_build_consensus_message_blocks_no_consensus(self):
        """Should build consensus blocks when no consensus reached."""
        blocks = slack.build_consensus_message_blocks(
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


# =============================================================================
# Test Event Handlers
# =============================================================================


class TestSlackEventHandler:
    """Tests for Slack Events API handler."""

    @pytest.mark.asyncio
    async def test_handle_url_verification(self):
        """Should respond to URL verification challenge."""
        request = MagicMock()
        request.body = AsyncMock(
            return_value=json.dumps(
                {"type": "url_verification", "challenge": "test_challenge_123"}
            ).encode()
        )

        result = await slack.handle_slack_events(request)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["challenge"] == "test_challenge_123"

    @pytest.mark.asyncio
    async def test_handle_app_mention(self):
        """Should handle app mention events."""
        request = MagicMock()
        request.body = AsyncMock(
            return_value=json.dumps(
                {
                    "type": "event_callback",
                    "event": {
                        "type": "app_mention",
                        "text": "@aragora ask about microservices",
                        "channel": "C123456",
                        "user": "U123456",
                    },
                }
            ).encode()
        )

        result = await slack.handle_slack_events(request)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "response_type" in body

    @pytest.mark.asyncio
    async def test_handle_message_event(self):
        """Should handle message events."""
        request = MagicMock()
        request.body = AsyncMock(
            return_value=json.dumps(
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

        result = await slack.handle_slack_events(request)

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self):
        """Should handle invalid JSON gracefully."""
        request = MagicMock()
        request.body = AsyncMock(return_value=b"not valid json")

        result = await slack.handle_slack_events(request)

        assert result.status_code == 500


# =============================================================================
# Test Slash Commands
# =============================================================================


class TestSlackCommandHandler:
    """Tests for Slack slash command handler."""

    @pytest.mark.asyncio
    async def test_handle_ask_command(self):
        """Should handle /aragora ask command."""
        request = MagicMock()
        body = (
            "command=/aragora&"
            "text=ask+Should+we+use+microservices&"
            "user_id=U123&"
            "user_name=testuser&"
            "channel_id=C123&"
            "response_url=https://hooks.slack.com/test"
        )
        request.body = AsyncMock(return_value=body.encode())

        result = await slack.handle_slack_commands(request)

        assert result.status_code == 200
        response_body = json.loads(result.body)
        assert response_body["response_type"] == "in_channel"
        assert "microservices" in response_body["text"]

    @pytest.mark.asyncio
    async def test_handle_status_command(self):
        """Should handle /aragora status command."""
        request = MagicMock()
        body = "command=/aragora&text=status&user_id=U123"
        request.body = AsyncMock(return_value=body.encode())

        result = await slack.handle_slack_commands(request)

        assert result.status_code == 200
        response_body = json.loads(result.body)
        assert response_body["response_type"] == "ephemeral"
        assert "active debate" in response_body["text"]

    @pytest.mark.asyncio
    async def test_handle_help_command(self):
        """Should handle /aragora help command."""
        request = MagicMock()
        body = "command=/aragora&text=help&user_id=U123"
        request.body = AsyncMock(return_value=body.encode())

        result = await slack.handle_slack_commands(request)

        assert result.status_code == 200
        response_body = json.loads(result.body)
        assert "Aragora Commands" in response_body["text"]
        assert "/aragora ask" in response_body["text"]

    @pytest.mark.asyncio
    async def test_handle_leaderboard_command(self):
        """Should handle /aragora leaderboard command."""
        request = MagicMock()
        body = "command=/aragora&text=leaderboard&user_id=U123"
        request.body = AsyncMock(return_value=body.encode())

        result = await slack.handle_slack_commands(request)

        assert result.status_code == 200
        response_body = json.loads(result.body)
        assert response_body["response_type"] == "in_channel"
        assert "Leaderboard" in response_body["text"]

    @pytest.mark.asyncio
    async def test_handle_unknown_command(self):
        """Should show help for unknown commands."""
        request = MagicMock()
        body = "command=/aragora&text=unknowncommand&user_id=U123"
        request.body = AsyncMock(return_value=body.encode())

        result = await slack.handle_slack_commands(request)

        assert result.status_code == 200
        response_body = json.loads(result.body)
        assert "Aragora Commands" in response_body["text"]


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
                    "action_id": "vote_claude",
                    "block_id": "vote_block",
                    "value": "debate-123:claude",
                }
            ],
        }

        request = MagicMock()
        body = f"payload={json.dumps(payload)}"
        request.body = AsyncMock(return_value=body.encode())

        result = await slack.handle_slack_interactions(request)

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_shortcut_start_debate(self):
        """Should handle start debate shortcut."""
        payload = {
            "type": "shortcut",
            "callback_id": "start_debate",
            "user": {"id": "U123"},
            "trigger_id": "trigger123",
        }

        request = MagicMock()
        body = f"payload={json.dumps(payload)}"
        request.body = AsyncMock(return_value=body.encode())

        result = await slack.handle_slack_interactions(request)

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

        request = MagicMock()
        body = f"payload={json.dumps(payload)}"
        request.body = AsyncMock(return_value=body.encode())

        with patch.object(slack, "audit_data"):
            result = await slack.handle_slack_interactions(request)

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

        request = MagicMock()
        body = f"payload={json.dumps(payload)}"
        request.body = AsyncMock(return_value=body.encode())

        result = await slack.handle_slack_interactions(request)

        assert result.status_code == 200
        response_body = json.loads(result.body)
        assert response_body.get("response_action") == "errors"
        assert "task_block" in response_body.get("errors", {})


# =============================================================================
# Test Global State
# =============================================================================


class TestSlackGlobalState:
    """Tests for Slack handler global state management."""

    def test_active_debates_storage(self):
        """Should store and retrieve active debates."""
        # Clean state
        slack._active_debates.clear()
        slack._user_votes.clear()

        # Add a debate
        slack._active_debates["test-debate-1"] = {
            "task": "Test task",
            "agents": ["Claude", "GPT-4"],
            "rounds": 5,
            "current_round": 1,
            "status": "running",
        }

        assert "test-debate-1" in slack._active_debates
        assert slack._active_debates["test-debate-1"]["task"] == "Test task"

        # Clean up
        slack._active_debates.clear()

    def test_user_votes_storage(self):
        """Should store user votes correctly."""
        slack._user_votes.clear()

        slack._user_votes["debate-1"] = {"user-1": "claude", "user-2": "gpt4"}

        assert slack._user_votes["debate-1"]["user-1"] == "claude"
        assert slack._user_votes["debate-1"]["user-2"] == "gpt4"

        # Clean up
        slack._user_votes.clear()
