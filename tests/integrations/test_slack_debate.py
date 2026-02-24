"""Tests for Slack thread debate lifecycle."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.integrations.slack_debate import (
    SlackDebateConfig,
    SlackDebateLifecycle,
    _build_consensus_blocks,
    _build_debate_started_blocks,
    _build_round_update_blocks,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def lifecycle():
    return SlackDebateLifecycle(bot_token="xoxb-test-token")


@pytest.fixture
def config():
    return SlackDebateConfig(rounds=3, agents=["claude", "gpt4"])


def _make_debate_result(**kwargs):
    """Create a mock DebateResult with sensible defaults."""
    result = MagicMock()
    result.debate_id = kwargs.get("debate_id", "debate-abc123")
    result.task = kwargs.get("task", "Should we adopt microservices?")
    result.final_answer = kwargs.get(
        "final_answer", "Yes, adopt a gradual migration strategy."
    )
    result.consensus_reached = kwargs.get("consensus_reached", True)
    result.confidence = kwargs.get("confidence", 0.85)
    result.rounds_used = kwargs.get("rounds_used", 3)
    result.winner = kwargs.get("winner", "claude")
    result.participants = kwargs.get("participants", ["claude", "gpt4"])
    return result


# =============================================================================
# SlackDebateConfig Tests
# =============================================================================


class TestSlackDebateConfig:
    def test_default_values(self):
        cfg = SlackDebateConfig()
        assert cfg.rounds == 3
        assert cfg.agents == ["claude", "gpt4"]
        assert cfg.consensus_threshold == 0.7
        assert cfg.timeout_seconds == 300.0
        assert cfg.metadata == {}

    def test_custom_values(self):
        cfg = SlackDebateConfig(
            rounds=5,
            agents=["claude", "gpt4", "gemini"],
            consensus_threshold=0.9,
            timeout_seconds=600.0,
            metadata={"team": "engineering"},
        )
        assert cfg.rounds == 5
        assert len(cfg.agents) == 3
        assert cfg.consensus_threshold == 0.9
        assert cfg.metadata["team"] == "engineering"


# =============================================================================
# SlackDebateLifecycle Initialization Tests
# =============================================================================


class TestSlackDebateLifecycleInit:
    def test_requires_bot_token(self):
        with pytest.raises(ValueError, match="Slack bot token is required"):
            SlackDebateLifecycle(bot_token="")

    def test_initialization(self, lifecycle):
        assert lifecycle._bot_token == "xoxb-test-token"
        assert lifecycle._session is None


# =============================================================================
# Block Kit Builder Tests
# =============================================================================


class TestBuildDebateStartedBlocks:
    def test_returns_blocks(self, config):
        blocks = _build_debate_started_blocks("d-123", "Test topic", config)
        assert len(blocks) > 0

    def test_header_block(self, config):
        blocks = _build_debate_started_blocks("d-123", "Test topic", config)
        assert blocks[0]["type"] == "header"
        assert "Debate Started" in blocks[0]["text"]["text"]

    def test_topic_in_blocks(self, config):
        blocks = _build_debate_started_blocks("d-123", "My debate topic", config)
        block_text = str(blocks)
        assert "My debate topic" in block_text

    def test_agents_in_blocks(self, config):
        blocks = _build_debate_started_blocks("d-123", "Topic", config)
        block_text = str(blocks)
        assert "claude" in block_text
        assert "gpt4" in block_text

    def test_debate_id_in_blocks(self, config):
        blocks = _build_debate_started_blocks("debate-abcdef123456", "Topic", config)
        block_text = str(blocks)
        assert "debate-abcde" in block_text

    def test_context_footer(self, config):
        blocks = _build_debate_started_blocks("d-123", "Topic", config)
        assert blocks[-1]["type"] == "context"
        assert "Aragora" in str(blocks[-1])


class TestBuildRoundUpdateBlocks:
    def test_basic_round(self):
        data = {"round": 2, "total_rounds": 5}
        blocks = _build_round_update_blocks(data)
        assert len(blocks) >= 1
        assert "Round 2/5" in str(blocks)

    def test_with_agent_proposal(self):
        data = {
            "round": 1,
            "total_rounds": 3,
            "agent": "claude",
            "proposal": "We should use token bucket algorithm for rate limiting.",
        }
        blocks = _build_round_update_blocks(data)
        block_text = str(blocks)
        assert "claude" in block_text
        assert "token bucket" in block_text

    def test_long_proposal_truncated(self):
        data = {
            "round": 1,
            "total_rounds": 3,
            "agent": "gpt4",
            "proposal": "x" * 500,
        }
        blocks = _build_round_update_blocks(data)
        block_text = str(blocks)
        assert "..." in block_text

    def test_phase_emoji_proposal(self):
        data = {"round": 1, "total_rounds": 3, "phase": "proposal"}
        blocks = _build_round_update_blocks(data)
        assert ":pencil:" in str(blocks)

    def test_phase_emoji_critique(self):
        data = {"round": 1, "total_rounds": 3, "phase": "critique"}
        blocks = _build_round_update_blocks(data)
        assert ":mag:" in str(blocks)

    def test_phase_emoji_revision(self):
        data = {"round": 1, "total_rounds": 3, "phase": "revision"}
        blocks = _build_round_update_blocks(data)
        assert ":arrows_counterclockwise:" in str(blocks)

    def test_phase_emoji_vote(self):
        data = {"round": 1, "total_rounds": 3, "phase": "vote"}
        blocks = _build_round_update_blocks(data)
        assert ":ballot_box:" in str(blocks)


class TestBuildConsensusBlocks:
    def test_consensus_reached(self):
        result = _make_debate_result(consensus_reached=True)
        blocks = _build_consensus_blocks(result)
        block_text = str(blocks)
        assert "Consensus Reached" in block_text
        assert ":white_check_mark:" in block_text

    def test_no_consensus(self):
        result = _make_debate_result(consensus_reached=False)
        blocks = _build_consensus_blocks(result)
        block_text = str(blocks)
        assert "No Consensus" in block_text
        assert ":x:" in block_text

    def test_includes_task(self):
        result = _make_debate_result(task="Rate limiter design")
        blocks = _build_consensus_blocks(result)
        assert "Rate limiter design" in str(blocks)

    def test_includes_confidence(self):
        result = _make_debate_result(confidence=0.92)
        blocks = _build_consensus_blocks(result)
        assert "92%" in str(blocks)

    def test_includes_winner(self):
        result = _make_debate_result(winner="gemini")
        blocks = _build_consensus_blocks(result)
        assert "gemini" in str(blocks)

    def test_includes_final_answer_when_consensus(self):
        result = _make_debate_result(
            consensus_reached=True, final_answer="Use caching."
        )
        blocks = _build_consensus_blocks(result)
        assert "Use caching." in str(blocks)
        assert "Final Decision" in str(blocks)

    def test_no_final_answer_block_without_consensus(self):
        result = _make_debate_result(consensus_reached=False, final_answer="Something")
        blocks = _build_consensus_blocks(result)
        assert "Final Decision" not in str(blocks)

    def test_long_final_answer_truncated(self):
        result = _make_debate_result(
            consensus_reached=True, final_answer="a" * 600
        )
        blocks = _build_consensus_blocks(result)
        block_text = str(blocks)
        assert "..." in block_text

    def test_footer_with_debate_id(self):
        result = _make_debate_result(debate_id="debate-xyz789")
        blocks = _build_consensus_blocks(result)
        assert "debate-x" in str(blocks)

    def test_none_confidence_handled(self):
        result = _make_debate_result(confidence=None)
        blocks = _build_consensus_blocks(result)
        assert "0%" in str(blocks)


# =============================================================================
# SlackDebateLifecycle._post_to_thread Tests
# =============================================================================


class TestPostToThread:
    @pytest.mark.asyncio
    async def test_successful_post(self, lifecycle):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"ok": True, "ts": "12345.6789"})

        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_session.closed = False

        lifecycle._session = mock_session

        result = await lifecycle._post_to_thread(
            channel_id="C01ABC",
            thread_ts="1234567890.123456",
            text="Test message",
            blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": "Hi"}}],
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_http_error(self, lifecycle):
        mock_resp = AsyncMock()
        mock_resp.status = 500

        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_session.closed = False

        lifecycle._session = mock_session

        result = await lifecycle._post_to_thread(
            channel_id="C01ABC", thread_ts="123.456", text="fail"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_slack_api_error(self, lifecycle):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(
            return_value={"ok": False, "error": "channel_not_found"}
        )

        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_session.closed = False

        lifecycle._session = mock_session

        result = await lifecycle._post_to_thread(
            channel_id="C_INVALID", thread_ts="123.456", text="test"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_network_error(self, lifecycle):
        mock_session = AsyncMock()
        mock_session.post.side_effect = OSError("Connection refused")
        mock_session.closed = False

        lifecycle._session = mock_session

        result = await lifecycle._post_to_thread(
            channel_id="C01ABC", thread_ts="123.456", text="test"
        )
        assert result is False


# =============================================================================
# SlackDebateLifecycle.start_debate_from_thread Tests
# =============================================================================


class TestStartDebateFromThread:
    @pytest.mark.asyncio
    async def test_returns_debate_id(self, lifecycle):
        with patch.object(
            lifecycle, "_post_to_thread", new_callable=AsyncMock, return_value=True
        ):
            with patch(
                "aragora.integrations.slack_debate.register_debate_origin",
                side_effect=ImportError("not available"),
            ):
                pass
            debate_id = await lifecycle.start_debate_from_thread(
                channel_id="C01ABC",
                thread_ts="1234567890.123456",
                topic="Should we use Kubernetes?",
            )
            assert isinstance(debate_id, str)
            assert len(debate_id) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_posts_announcement(self, lifecycle):
        with patch.object(
            lifecycle, "_post_to_thread", new_callable=AsyncMock, return_value=True
        ) as mock_post:
            await lifecycle.start_debate_from_thread(
                channel_id="C01ABC",
                thread_ts="123.456",
                topic="Test topic",
            )
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]["channel_id"] == "C01ABC" or call_args[0][0] == "C01ABC"
            assert "Test topic" in str(call_args)

    @pytest.mark.asyncio
    async def test_registers_origin(self, lifecycle):
        with patch.object(
            lifecycle, "_post_to_thread", new_callable=AsyncMock, return_value=True
        ):
            with patch(
                "aragora.server.debate_origin.register_debate_origin"
            ) as mock_register:
                await lifecycle.start_debate_from_thread(
                    channel_id="C01ABC",
                    thread_ts="123.456",
                    topic="Test topic",
                    user_id="U_USER",
                )
                mock_register.assert_called_once()
                kwargs = mock_register.call_args[1]
                assert kwargs["platform"] == "slack"
                assert kwargs["channel_id"] == "C01ABC"
                assert kwargs["user_id"] == "U_USER"
                assert kwargs["thread_id"] == "123.456"

    @pytest.mark.asyncio
    async def test_origin_registration_failure_does_not_raise(self, lifecycle):
        with patch.object(
            lifecycle, "_post_to_thread", new_callable=AsyncMock, return_value=True
        ):
            with patch(
                "aragora.server.debate_origin.register_debate_origin",
                side_effect=RuntimeError("DB error"),
            ):
                # Should not raise
                debate_id = await lifecycle.start_debate_from_thread(
                    channel_id="C01ABC",
                    thread_ts="123.456",
                    topic="Test",
                )
                assert isinstance(debate_id, str)

    @pytest.mark.asyncio
    async def test_custom_config(self, lifecycle, config):
        with patch.object(
            lifecycle, "_post_to_thread", new_callable=AsyncMock, return_value=True
        ) as mock_post:
            config.agents = ["claude", "gpt4", "gemini"]
            config.rounds = 5
            await lifecycle.start_debate_from_thread(
                channel_id="C01ABC",
                thread_ts="123.456",
                topic="Test",
                config=config,
            )
            # Blocks should include the custom agent list
            call_blocks = mock_post.call_args[0][3] if len(mock_post.call_args[0]) > 3 else mock_post.call_args[1].get("blocks")
            block_text = str(call_blocks)
            assert "gemini" in block_text


# =============================================================================
# SlackDebateLifecycle.post_round_update Tests
# =============================================================================


class TestPostRoundUpdate:
    @pytest.mark.asyncio
    async def test_posts_round(self, lifecycle):
        with patch.object(
            lifecycle, "_post_to_thread", new_callable=AsyncMock, return_value=True
        ) as mock_post:
            result = await lifecycle.post_round_update(
                channel_id="C01ABC",
                thread_ts="123.456",
                round_data={"round": 2, "total_rounds": 5},
            )
            assert result is True
            mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_false_on_failure(self, lifecycle):
        with patch.object(
            lifecycle, "_post_to_thread", new_callable=AsyncMock, return_value=False
        ):
            result = await lifecycle.post_round_update(
                channel_id="C01ABC",
                thread_ts="123.456",
                round_data={"round": 1, "total_rounds": 3},
            )
            assert result is False


# =============================================================================
# SlackDebateLifecycle.post_consensus Tests
# =============================================================================


class TestPostConsensus:
    @pytest.mark.asyncio
    async def test_posts_consensus_reached(self, lifecycle):
        with patch.object(
            lifecycle, "_post_to_thread", new_callable=AsyncMock, return_value=True
        ) as mock_post:
            result_obj = _make_debate_result(consensus_reached=True)
            success = await lifecycle.post_consensus(
                channel_id="C01ABC",
                thread_ts="123.456",
                result=result_obj,
            )
            assert success is True
            call_text = str(mock_post.call_args)
            assert "Consensus reached" in call_text

    @pytest.mark.asyncio
    async def test_posts_no_consensus(self, lifecycle):
        with patch.object(
            lifecycle, "_post_to_thread", new_callable=AsyncMock, return_value=True
        ) as mock_post:
            result_obj = _make_debate_result(consensus_reached=False)
            success = await lifecycle.post_consensus(
                channel_id="C01ABC",
                thread_ts="123.456",
                result=result_obj,
            )
            assert success is True
            call_text = str(mock_post.call_args)
            assert "not reached" in call_text

    @pytest.mark.asyncio
    async def test_returns_false_on_failure(self, lifecycle):
        with patch.object(
            lifecycle, "_post_to_thread", new_callable=AsyncMock, return_value=False
        ):
            result_obj = _make_debate_result()
            success = await lifecycle.post_consensus(
                channel_id="C01ABC",
                thread_ts="123.456",
                result=result_obj,
            )
            assert success is False


# =============================================================================
# SlackDebateLifecycle.handle_slash_command Tests
# =============================================================================


class TestHandleSlashCommand:
    @pytest.mark.asyncio
    async def test_missing_topic(self, lifecycle):
        response = await lifecycle.handle_slash_command(
            {"text": "", "channel_id": "C01ABC", "user_id": "U01"}
        )
        assert response["response_type"] == "ephemeral"
        assert "Usage" in response["text"]

    @pytest.mark.asyncio
    async def test_missing_channel(self, lifecycle):
        response = await lifecycle.handle_slash_command(
            {"text": "Test topic", "channel_id": "", "user_id": "U01"}
        )
        assert response["response_type"] == "ephemeral"
        assert "channel" in response["text"].lower()

    @pytest.mark.asyncio
    async def test_successful_command(self, lifecycle):
        with patch.object(
            lifecycle,
            "start_debate_from_thread",
            new_callable=AsyncMock,
            return_value="debate-123456789012",
        ):
            response = await lifecycle.handle_slash_command(
                {
                    "text": "Should we use microservices?",
                    "channel_id": "C01ABC",
                    "user_id": "U01",
                }
            )
            assert response["response_type"] == "in_channel"
            assert "debate-12345" in response["text"]
            assert "microservices" in response["text"]

    @pytest.mark.asyncio
    async def test_list_payload_values(self, lifecycle):
        """Slash commands from Slack may have list values from parse_qs."""
        with patch.object(
            lifecycle,
            "start_debate_from_thread",
            new_callable=AsyncMock,
            return_value="debate-abc",
        ) as mock_start:
            await lifecycle.handle_slash_command(
                {
                    "text": ["My topic"],
                    "channel_id": ["C01ABC"],
                    "user_id": ["U01"],
                }
            )
            call_kwargs = mock_start.call_args[1]
            assert call_kwargs["topic"] == "My topic"
            assert call_kwargs["channel_id"] == "C01ABC"
            assert call_kwargs["user_id"] == "U01"

    @pytest.mark.asyncio
    async def test_error_returns_ephemeral(self, lifecycle):
        with patch.object(
            lifecycle,
            "start_debate_from_thread",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            response = await lifecycle.handle_slash_command(
                {
                    "text": "Topic",
                    "channel_id": "C01ABC",
                    "user_id": "U01",
                }
            )
            assert response["response_type"] == "ephemeral"
            assert "Failed" in response["text"]

    @pytest.mark.asyncio
    async def test_thread_ts_forwarded(self, lifecycle):
        with patch.object(
            lifecycle,
            "start_debate_from_thread",
            new_callable=AsyncMock,
            return_value="d-123",
        ) as mock_start:
            await lifecycle.handle_slash_command(
                {
                    "text": "Topic",
                    "channel_id": "C01ABC",
                    "user_id": "U01",
                    "thread_ts": "1234567890.123",
                }
            )
            call_kwargs = mock_start.call_args[1]
            assert call_kwargs["thread_ts"] == "1234567890.123"


# =============================================================================
# Session Management Tests
# =============================================================================


class TestSessionManagement:
    @pytest.mark.asyncio
    async def test_close_with_no_session(self, lifecycle):
        # Should not raise
        await lifecycle.close()

    @pytest.mark.asyncio
    async def test_close_with_active_session(self, lifecycle):
        mock_session = AsyncMock()
        mock_session.closed = False
        lifecycle._session = mock_session
        await lifecycle.close()
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_with_already_closed_session(self, lifecycle):
        mock_session = AsyncMock()
        mock_session.closed = True
        lifecycle._session = mock_session
        await lifecycle.close()
        mock_session.close.assert_not_called()
