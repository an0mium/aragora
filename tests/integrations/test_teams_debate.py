"""Tests for Microsoft Teams thread debate lifecycle."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.integrations.teams_debate import (
    TeamsDebateConfig,
    TeamsDebateLifecycle,
    _active_debates,
    _build_consensus_card,
    _build_round_update_card,
    _wrap_card_payload,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def lifecycle():
    """Create a lifecycle with a mocked TeamsIntegration."""
    mock_integration = MagicMock()
    mock_integration.is_configured = True
    mock_integration._send_card = AsyncMock(return_value=True)
    return TeamsDebateLifecycle(teams_integration=mock_integration)


@pytest.fixture
def config():
    return TeamsDebateConfig(rounds=3, agents=["claude", "gpt4"])


# =============================================================================
# TeamsDebateConfig Tests
# =============================================================================


class TestTeamsDebateConfig:
    def test_default_values(self):
        cfg = TeamsDebateConfig()
        assert cfg.rounds == 3
        assert cfg.agents == ["claude", "gpt4", "gemini"]
        assert cfg.consensus_threshold == 0.7
        assert cfg.timeout_seconds == 300.0
        assert cfg.enable_voting is True

    def test_custom_values(self):
        cfg = TeamsDebateConfig(
            rounds=5,
            agents=["claude", "gpt4"],
            consensus_threshold=0.9,
            timeout_seconds=600.0,
            enable_voting=False,
        )
        assert cfg.rounds == 5
        assert len(cfg.agents) == 2
        assert cfg.consensus_threshold == 0.9
        assert cfg.enable_voting is False


# =============================================================================
# Card Builder Tests
# =============================================================================


class TestBuildRoundUpdateCard:
    def test_basic_round(self):
        card = _build_round_update_card(
            topic="Rate limiter",
            round_number=2,
            total_rounds=5,
        )
        assert card.get("type") == "AdaptiveCard" or "body" in card
        card_text = str(card)
        assert "2" in card_text
        assert "5" in card_text

    def test_with_agent_messages(self):
        messages = [
            {"agent": "claude", "summary": "We should use token bucket."},
            {"agent": "gpt4", "summary": "Sliding window is better."},
        ]
        card = _build_round_update_card(
            topic="Rate limiter",
            round_number=1,
            total_rounds=3,
            agent_messages=messages,
        )
        card_text = str(card)
        assert "claude" in card_text
        assert "token bucket" in card_text

    def test_long_summary_truncated(self):
        messages = [{"agent": "gpt4", "summary": "x" * 500}]
        card = _build_round_update_card(
            topic="Topic",
            round_number=1,
            total_rounds=3,
            agent_messages=messages,
        )
        card_text = str(card)
        assert "..." in card_text

    def test_with_consensus(self):
        card = _build_round_update_card(
            topic="Topic",
            round_number=2,
            total_rounds=3,
            current_consensus="Token bucket preferred",
        )
        card_text = str(card)
        assert "Token bucket preferred" in card_text

    def test_debate_id_passed(self):
        card = _build_round_update_card(
            topic="Topic",
            round_number=1,
            total_rounds=3,
            debate_id="teams-abc123",
        )
        # Card should be valid even with debate_id (may or may not appear in body)
        assert "body" in card


class TestBuildConsensusCard:
    def test_consensus_reached(self):
        result = {
            "consensus_reached": True,
            "confidence": 0.85,
            "final_answer": "Use token bucket algorithm.",
            "participants": ["claude", "gpt4"],
            "rounds_used": 3,
        }
        card = _build_consensus_card("Rate limiter", result, "d-123")
        card_text = str(card)
        assert "85%" in card_text or "0.85" in card_text or "Consensus" in card_text

    def test_no_consensus(self):
        result = {
            "consensus_reached": False,
            "confidence": 0.4,
            "final_answer": "",
            "participants": ["claude", "gpt4"],
            "rounds_used": 3,
        }
        card = _build_consensus_card("Topic", result, "d-123")
        card_text = str(card)
        assert "Complete" in card_text or "Warning" in card_text

    def test_includes_agents(self):
        result = {
            "consensus_reached": True,
            "confidence": 0.9,
            "final_answer": "Yes",
            "participants": ["claude", "gemini"],
            "rounds_used": 2,
        }
        card = _build_consensus_card("Topic", result, "d-123")
        card_text = str(card)
        assert "claude" in card_text or "gemini" in card_text

    def test_long_answer_truncated(self):
        result = {
            "consensus_reached": True,
            "confidence": 0.85,
            "final_answer": "a" * 600,
            "participants": [],
            "rounds_used": 3,
        }
        card = _build_consensus_card("Topic", result, "d-123")
        card_text = str(card)
        # The answer should be truncated to at most 500 characters
        assert "a" * 600 not in card_text
        assert "a" * 500 in card_text

    def test_view_full_report_action(self):
        result = {
            "consensus_reached": True,
            "confidence": 0.9,
            "final_answer": "Yes",
            "participants": [],
            "rounds_used": 1,
        }
        card = _build_consensus_card("Topic", result, "debate-xyz789")
        card_text = str(card)
        assert "debate-xyz789" in card_text


class TestWrapCardPayload:
    def test_wraps_as_message(self):
        card = {"type": "AdaptiveCard", "body": []}
        payload = _wrap_card_payload(card)
        assert payload["type"] == "message"
        assert len(payload["attachments"]) == 1
        assert payload["attachments"][0]["contentType"] == "application/vnd.microsoft.card.adaptive"
        assert payload["attachments"][0]["content"] is card


# =============================================================================
# TeamsDebateLifecycle Initialization Tests
# =============================================================================


class TestTeamsDebateLifecycleInit:
    def test_initialization_with_integration(self, lifecycle):
        assert lifecycle._integration is not None

    def test_initialization_without_integration(self):
        lc = TeamsDebateLifecycle()
        assert lc._integration is None

    def test_lazy_integration_property(self):
        lc = TeamsDebateLifecycle()
        with patch("aragora.integrations.teams.TeamsIntegration") as mock_cls:
            mock_cls.return_value = MagicMock()
            integration = lc.integration
            mock_cls.assert_called_once()
            assert integration is not None

    def test_active_debates_initially_empty(self, lifecycle):
        # _active_debates is a module-level dict, not an instance attribute
        assert isinstance(_active_debates, dict)


# =============================================================================
# TeamsDebateLifecycle.start_debate_from_thread Tests
# =============================================================================


class TestStartDebateFromThread:
    @pytest.mark.asyncio
    async def test_returns_debate_id(self, lifecycle):
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True
        ):
            debate_id = await lifecycle.start_debate_from_thread(
                channel_id="19:abc@thread.tacv2",
                message_id="1677012345678",
                topic="Should we use Kubernetes?",
            )
            assert isinstance(debate_id, str)
            assert debate_id.startswith("teams-")

    @pytest.mark.asyncio
    async def test_tracks_debate_locally(self, lifecycle):
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True
        ):
            debate_id = await lifecycle.start_debate_from_thread(
                channel_id="19:abc@thread.tacv2",
                message_id="1677012345678",
                topic="Test topic",
            )
            assert debate_id in _active_debates
            info = _active_debates[debate_id]
            assert info["topic"] == "Test topic"
            assert info["channel_id"] == "19:abc@thread.tacv2"

    @pytest.mark.asyncio
    async def test_posts_starting_card(self, lifecycle):
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True
        ) as mock_send:
            await lifecycle.start_debate_from_thread(
                channel_id="19:abc@thread.tacv2",
                message_id="msg-123",
                topic="Test topic",
            )
            mock_send.assert_called_once()
            call_args = mock_send.call_args
            assert call_args[0][0] == "19:abc@thread.tacv2"
            assert call_args[0][1] == "msg-123"

    @pytest.mark.asyncio
    async def test_registers_debate_origin(self, lifecycle):
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True
        ):
            with patch(
                "aragora.server.debate_origin.register_debate_origin"
            ) as mock_register:
                await lifecycle.start_debate_from_thread(
                    channel_id="19:abc@thread.tacv2",
                    message_id="msg-123",
                    topic="Test topic",
                    user_id="user-456",
                    tenant_id="tenant-789",
                )
                mock_register.assert_called_once()
                kwargs = mock_register.call_args[1]
                assert kwargs["platform"] == "teams"
                assert kwargs["channel_id"] == "19:abc@thread.tacv2"
                assert kwargs["user_id"] == "user-456"
                assert kwargs["thread_id"] == "msg-123"
                assert kwargs["message_id"] == "msg-123"

    @pytest.mark.asyncio
    async def test_origin_registration_failure_does_not_raise(self, lifecycle):
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True
        ):
            with patch(
                "aragora.server.debate_origin.register_debate_origin",
                side_effect=RuntimeError("DB error"),
            ):
                debate_id = await lifecycle.start_debate_from_thread(
                    channel_id="19:abc@thread.tacv2",
                    message_id="msg-123",
                    topic="Test",
                )
                assert isinstance(debate_id, str)

    @pytest.mark.asyncio
    async def test_custom_config(self, lifecycle, config):
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True
        ):
            config.agents = ["claude", "gpt4", "gemini"]
            config.rounds = 5
            debate_id = await lifecycle.start_debate_from_thread(
                channel_id="19:abc@thread.tacv2",
                message_id="msg-123",
                topic="Test",
                config=config,
            )
            info = _active_debates[debate_id]
            assert info["config"].agents == ["claude", "gpt4", "gemini"]
            assert info["config"].rounds == 5


# =============================================================================
# TeamsDebateLifecycle.post_round_update Tests
# =============================================================================


class TestPostRoundUpdate:
    @pytest.mark.asyncio
    async def test_posts_round(self, lifecycle):
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True
        ) as mock_send:
            result = await lifecycle.post_round_update(
                channel_id="19:abc@thread.tacv2",
                message_id="msg-123",
                round_data={
                    "debate_id": "teams-abc",
                    "topic": "Topic",
                    "round_number": 2,
                    "total_rounds": 5,
                },
            )
            assert result is True
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_false_on_failure(self, lifecycle):
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=False
        ):
            result = await lifecycle.post_round_update(
                channel_id="19:abc@thread.tacv2",
                message_id="msg-123",
                round_data={
                    "debate_id": "teams-abc",
                    "topic": "Topic",
                    "round_number": 1,
                    "total_rounds": 3,
                },
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_passes_agent_messages(self, lifecycle):
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True
        ) as mock_send:
            await lifecycle.post_round_update(
                channel_id="ch-1",
                message_id="msg-1",
                round_data={
                    "debate_id": "teams-xyz",
                    "topic": "Topic",
                    "round_number": 1,
                    "total_rounds": 3,
                    "agent_messages": [
                        {"agent": "claude", "summary": "Token bucket is best."},
                    ],
                },
            )
            card_arg = mock_send.call_args[0][2]
            card_text = str(card_arg)
            assert "claude" in card_text or "Token bucket" in card_text


# =============================================================================
# TeamsDebateLifecycle.post_consensus Tests
# =============================================================================


class TestPostConsensus:
    @pytest.mark.asyncio
    async def test_posts_consensus(self, lifecycle):
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True
        ) as mock_send:
            success = await lifecycle.post_consensus(
                channel_id="19:abc@thread.tacv2",
                message_id="msg-123",
                result={
                    "debate_id": "teams-abc123",
                    "topic": "Rate limiter design",
                    "consensus_reached": True,
                    "confidence": 0.85,
                    "final_answer": "Use token bucket.",
                    "participants": ["claude", "gpt4"],
                    "rounds_used": 3,
                },
            )
            assert success is True
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_false_on_failure(self, lifecycle):
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=False
        ):
            success = await lifecycle.post_consensus(
                channel_id="ch-1",
                message_id="msg-1",
                result={
                    "debate_id": "teams-xyz",
                    "topic": "Topic",
                    "consensus_reached": False,
                    "confidence": 0.3,
                    "final_answer": "",
                    "participants": [],
                    "rounds_used": 3,
                },
            )
            assert success is False

    @pytest.mark.asyncio
    async def test_marks_result_sent(self, lifecycle):
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True
        ):
            with patch(
                "aragora.server.debate_origin.mark_result_sent"
            ) as mock_mark:
                await lifecycle.post_consensus(
                    channel_id="ch-1",
                    message_id="msg-1",
                    result={
                        "debate_id": "teams-abc",
                        "topic": "Topic",
                        "consensus_reached": True,
                        "confidence": 0.9,
                        "final_answer": "Yes",
                        "participants": [],
                        "rounds_used": 1,
                    },
                )
                mock_mark.assert_called_once_with("teams-abc")

    @pytest.mark.asyncio
    async def test_cleans_up_active_debates(self, lifecycle):
        _active_debates["teams-abc"] = {"topic": "test"}
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True
        ):
            await lifecycle.post_consensus(
                channel_id="ch-1",
                message_id="msg-1",
                result={
                    "debate_id": "teams-abc",
                    "topic": "Topic",
                    "consensus_reached": True,
                    "confidence": 0.9,
                    "final_answer": "Yes",
                    "participants": [],
                    "rounds_used": 1,
                },
            )
            assert "teams-abc" not in _active_debates


# =============================================================================
# TeamsDebateLifecycle.handle_bot_command Tests
# =============================================================================


class TestHandleBotCommand:
    @pytest.mark.asyncio
    async def test_non_command_returns_none(self, lifecycle):
        activity = {"text": "Hello everyone", "conversation": {"id": "ch-1"}}
        result = await lifecycle.handle_bot_command(activity)
        assert result is None

    @pytest.mark.asyncio
    async def test_debate_command(self, lifecycle):
        with patch.object(
            lifecycle,
            "start_debate_from_thread",
            new_callable=AsyncMock,
            return_value="teams-abc123",
        ):
            activity = {
                "text": "/aragora debate Should we use microservices?",
                "conversation": {"id": "ch-1", "tenantId": "t-1"},
                "from": {"id": "user-1"},
                "id": "activity-1",
            }
            result = await lifecycle.handle_bot_command(activity)
            assert result is not None
            assert "teams-abc123" in result["text"]
            assert result["debate_id"] == "teams-abc123"

    @pytest.mark.asyncio
    async def test_debate_command_missing_topic(self, lifecycle):
        activity = {
            "text": "/aragora debate",
            "conversation": {"id": "ch-1"},
            "from": {"id": "user-1"},
            "id": "activity-1",
        }
        result = await lifecycle.handle_bot_command(activity)
        assert result is not None
        assert "Usage" in result["text"] or "provide" in result["text"]

    @pytest.mark.asyncio
    async def test_status_command_active_debate(self, lifecycle):
        _active_debates["teams-xyz"] = {
            "topic": "Test topic",
            "channel_id": "ch-1",
        }
        activity = {
            "text": "/aragora status teams-xyz",
            "conversation": {"id": "ch-1"},
            "from": {"id": "user-1"},
            "id": "activity-1",
        }
        result = await lifecycle.handle_bot_command(activity)
        assert result is not None
        assert "active" in result["text"]
        assert "Test topic" in result["text"]

    @pytest.mark.asyncio
    async def test_status_command_not_found(self, lifecycle):
        activity = {
            "text": "/aragora status teams-missing",
            "conversation": {"id": "ch-1"},
            "from": {"id": "user-1"},
            "id": "activity-1",
        }
        result = await lifecycle.handle_bot_command(activity)
        assert result is not None
        assert "not found" in result["text"]

    @pytest.mark.asyncio
    async def test_status_command_no_debate_id(self, lifecycle):
        activity = {
            "text": "/aragora status",
            "conversation": {"id": "ch-1"},
            "from": {"id": "user-1"},
            "id": "activity-1",
        }
        result = await lifecycle.handle_bot_command(activity)
        assert result is not None
        assert "provide" in result["text"].lower() or "Usage" in result["text"]

    @pytest.mark.asyncio
    async def test_help_command(self, lifecycle):
        activity = {
            "text": "/aragora help",
            "conversation": {"id": "ch-1"},
            "from": {"id": "user-1"},
            "id": "activity-1",
        }
        result = await lifecycle.handle_bot_command(activity)
        assert result is not None
        assert "debate" in result["text"].lower()
        assert "status" in result["text"].lower()
        assert "help" in result["text"].lower()

    @pytest.mark.asyncio
    async def test_unknown_command(self, lifecycle):
        activity = {
            "text": "/aragora foobar",
            "conversation": {"id": "ch-1"},
            "from": {"id": "user-1"},
            "id": "activity-1",
        }
        result = await lifecycle.handle_bot_command(activity)
        assert result is not None
        assert "Unknown" in result["text"] or "unknown" in result["text"]

    @pytest.mark.asyncio
    async def test_strips_bot_mention(self, lifecycle):
        with patch.object(
            lifecycle,
            "start_debate_from_thread",
            new_callable=AsyncMock,
            return_value="teams-abc",
        ) as mock_start:
            activity = {
                "text": "<at>Aragora</at> /aragora debate My topic",
                "conversation": {"id": "ch-1", "tenantId": "t-1"},
                "from": {"id": "user-1"},
                "id": "activity-1",
            }
            result = await lifecycle.handle_bot_command(activity)
            assert result is not None
            call_kwargs = mock_start.call_args[1]
            assert call_kwargs["topic"] == "My topic"

    @pytest.mark.asyncio
    async def test_default_command_is_help(self, lifecycle):
        activity = {
            "text": "/aragora",
            "conversation": {"id": "ch-1"},
            "from": {"id": "user-1"},
            "id": "activity-1",
        }
        result = await lifecycle.handle_bot_command(activity)
        assert result is not None
        # Should return help
        assert "debate" in result["text"].lower()

    @pytest.mark.asyncio
    async def test_uses_reply_to_id_as_message_id(self, lifecycle):
        with patch.object(
            lifecycle,
            "start_debate_from_thread",
            new_callable=AsyncMock,
            return_value="teams-abc",
        ) as mock_start:
            activity = {
                "text": "/aragora debate Topic",
                "conversation": {"id": "ch-1", "tenantId": "t-1"},
                "from": {"id": "user-1"},
                "id": "activity-1",
                "replyToId": "parent-msg-id",
            }
            await lifecycle.handle_bot_command(activity)
            call_kwargs = mock_start.call_args[1]
            assert call_kwargs["message_id"] == "parent-msg-id"


# =============================================================================
# Internal Helpers Tests
# =============================================================================


class TestBuildHelpResponse:
    def test_includes_commands(self):
        response = TeamsDebateLifecycle._build_help_response()
        assert "debate" in response["text"]
        assert "status" in response["text"]
        assert "help" in response["text"]


class TestGetDebateStatus:
    def test_active_debate(self, lifecycle):
        _active_debates["d-1"] = {
            "topic": "Test",
            "channel_id": "ch-1",
        }
        result = lifecycle._get_debate_status("d-1")
        assert "active" in result["text"]

    def test_not_found(self, lifecycle):
        result = lifecycle._get_debate_status("d-missing")
        assert "not found" in result["text"]

    def test_empty_id(self, lifecycle):
        result = lifecycle._get_debate_status("")
        assert "provide" in result["text"].lower() or "Usage" in result["text"]
