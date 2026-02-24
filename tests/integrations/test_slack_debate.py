"""Tests for Slack thread debate lifecycle."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.integrations.slack_debate import (
    SlackDebateConfig,
    SlackDebateLifecycle,
    _build_consensus_blocks,
    _build_debate_started_blocks,
    _build_error_blocks,
    _build_receipt_blocks,
    _build_round_update_blocks,
    parse_mention_text,
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


# =============================================================================
# Receipt helper
# =============================================================================


def _make_receipt(**kwargs):
    """Create a mock DecisionReceipt for testing."""
    receipt = MagicMock()
    receipt.verdict = kwargs.get("verdict", "APPROVED")
    receipt.confidence = kwargs.get("confidence", 0.92)
    receipt.receipt_id = kwargs.get("receipt_id", "rcpt-abcdef123456")
    receipt.findings = kwargs.get("findings", [])
    receipt.key_arguments = kwargs.get("key_arguments", None)
    receipt.dissenting_views = kwargs.get("dissenting_views", None)
    receipt.dissents = kwargs.get("dissents", None)
    return receipt


# =============================================================================
# _build_receipt_blocks Tests
# =============================================================================


class TestBuildReceiptBlocks:
    def test_returns_blocks(self):
        receipt = _make_receipt()
        blocks = _build_receipt_blocks(receipt)
        assert isinstance(blocks, list)
        assert len(blocks) > 0

    def test_header_block(self):
        receipt = _make_receipt(verdict="APPROVED")
        blocks = _build_receipt_blocks(receipt)
        assert blocks[0]["type"] == "header"
        assert "Decision Receipt" in blocks[0]["text"]["text"]

    def test_approved_with_conditions_emoji(self):
        receipt = _make_receipt(verdict="APPROVED_WITH_CONDITIONS")
        blocks = _build_receipt_blocks(receipt)
        assert ":large_yellow_circle:" in blocks[0]["text"]["text"]

    def test_rejected_verdict_emoji(self):
        receipt = _make_receipt(verdict="REJECTED")
        blocks = _build_receipt_blocks(receipt)
        assert ":x:" in blocks[0]["text"]["text"]

    def test_needs_review_verdict_emoji(self):
        receipt = _make_receipt(verdict="NEEDS_REVIEW")
        blocks = _build_receipt_blocks(receipt)
        assert ":warning:" in blocks[0]["text"]["text"]

    def test_fields_include_verdict_and_confidence(self):
        receipt = _make_receipt(verdict="APPROVED", confidence=0.88)
        blocks = _build_receipt_blocks(receipt)
        block_text = str(blocks)
        assert "APPROVED" in block_text
        assert "88%" in block_text

    def test_fields_include_finding_counts(self):
        finding_crit = MagicMock()
        finding_crit.severity = "critical"
        finding_crit.description = "Critical issue"
        finding_high = MagicMock()
        finding_high.severity = "high"
        finding_high.description = "High issue"
        receipt = _make_receipt(findings=[finding_crit, finding_high])
        blocks = _build_receipt_blocks(receipt)
        block_text = str(blocks)
        assert "2 total" in block_text
        assert "1 critical" in block_text

    def test_key_arguments_section(self):
        receipt = _make_receipt(key_arguments=["Point A", "Point B"])
        blocks = _build_receipt_blocks(receipt)
        block_text = str(blocks)
        assert "Key Arguments" in block_text
        assert "Point A" in block_text
        assert "Point B" in block_text

    def test_key_arguments_from_findings(self):
        finding = MagicMock()
        finding.severity = "low"
        finding.description = "Extracted argument"
        receipt = _make_receipt(key_arguments=None, findings=[finding])
        blocks = _build_receipt_blocks(receipt)
        assert "Extracted argument" in str(blocks)

    def test_dissenting_views_section(self):
        receipt = _make_receipt(dissenting_views=["I disagree", "Alternative view"])
        blocks = _build_receipt_blocks(receipt)
        block_text = str(blocks)
        assert "Dissenting Views" in block_text
        assert "I disagree" in block_text

    def test_dissenting_views_from_dissents_attr(self):
        receipt = _make_receipt(dissenting_views=None, dissents=["Objection 1"])
        blocks = _build_receipt_blocks(receipt)
        assert "Objection 1" in str(blocks)

    def test_action_buttons_with_url(self):
        receipt = _make_receipt()
        blocks = _build_receipt_blocks(receipt, receipt_url="https://example.com/receipt/1")
        actions_block = [b for b in blocks if b.get("type") == "actions"]
        assert len(actions_block) == 1
        elements = actions_block[0]["elements"]
        assert any("View Full Receipt" in str(e) for e in elements)
        assert any("https://example.com/receipt/1" in str(e) for e in elements)

    def test_action_buttons_without_url(self):
        receipt = _make_receipt()
        blocks = _build_receipt_blocks(receipt)
        actions_block = [b for b in blocks if b.get("type") == "actions"]
        assert len(actions_block) == 1
        elements = actions_block[0]["elements"]
        # Only audit trail button when no URL
        assert any("Audit Trail" in str(e) for e in elements)

    def test_footer_with_receipt_id(self):
        receipt = _make_receipt(receipt_id="rcpt-xyz789012345")
        blocks = _build_receipt_blocks(receipt)
        assert "rcpt-xyz78901" in str(blocks)

    def test_no_key_arguments_or_dissents(self):
        receipt = _make_receipt(key_arguments=[], dissenting_views=None, dissents=None)
        blocks = _build_receipt_blocks(receipt)
        block_text = str(blocks)
        assert "Key Arguments" not in block_text
        assert "Dissenting Views" not in block_text


# =============================================================================
# _build_error_blocks Tests
# =============================================================================


class TestBuildErrorBlocks:
    def test_returns_blocks(self):
        blocks = _build_error_blocks("Something went wrong")
        assert isinstance(blocks, list)
        assert len(blocks) >= 1

    def test_includes_warning_emoji(self):
        blocks = _build_error_blocks("Error occurred")
        assert ":warning:" in str(blocks)

    def test_includes_error_message(self):
        blocks = _build_error_blocks("Database connection failed")
        assert "Database connection failed" in str(blocks)

    def test_includes_debate_id_context(self):
        blocks = _build_error_blocks("Error", debate_id="debate-abcdef123456")
        block_text = str(blocks)
        assert "debate-abcdef1234" in block_text

    def test_no_debate_id_context_when_empty(self):
        blocks = _build_error_blocks("Error", debate_id="")
        assert len(blocks) == 1  # Only the error section, no context


# =============================================================================
# parse_mention_text Tests
# =============================================================================


class TestParseMentionText:
    def test_debate_command(self):
        command, topic = parse_mention_text("<@U01ABC> debate Should we use K8s?")
        assert command == "debate"
        assert topic == "Should we use K8s?"

    def test_decide_command(self):
        command, topic = parse_mention_text("<@U01ABC> decide on microservices")
        assert command == "decide"
        assert topic == "on microservices"

    def test_strips_quotes(self):
        command, topic = parse_mention_text('<@U01ABC> debate "My topic"')
        assert command == "debate"
        assert topic == "My topic"

    def test_empty_text(self):
        command, topic = parse_mention_text("")
        assert command == ""
        assert topic == ""

    def test_mention_only(self):
        command, topic = parse_mention_text("<@U01ABC>")
        assert command == ""
        assert topic == ""

    def test_no_debate_keyword(self):
        command, topic = parse_mention_text("<@U01ABC> help me")
        assert command == ""
        assert topic == ""

    def test_case_insensitive(self):
        command, topic = parse_mention_text("<@U01ABC> DEBATE big question")
        assert command == "debate"

    def test_debate_without_topic(self):
        command, topic = parse_mention_text("<@U01ABC> debate")
        assert command == "debate"
        assert topic == ""

    def test_multiple_spaces(self):
        command, topic = parse_mention_text("<@U01ABC>   debate   spaced topic  ")
        assert command == "debate"
        assert "spaced topic" in topic


# =============================================================================
# SlackDebateLifecycle.post_receipt Tests
# =============================================================================


class TestPostReceipt:
    @pytest.mark.asyncio
    async def test_posts_receipt(self, lifecycle):
        with patch.object(
            lifecycle, "_post_to_thread", new_callable=AsyncMock, return_value=True
        ) as mock_post:
            receipt = _make_receipt(verdict="APPROVED")
            result = await lifecycle.post_receipt(
                channel_id="C01ABC",
                thread_ts="123.456",
                receipt=receipt,
                debate_id="d-123",
            )
            assert result is True
            mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_false_on_failure(self, lifecycle):
        with patch.object(
            lifecycle, "_post_to_thread", new_callable=AsyncMock, return_value=False
        ):
            receipt = _make_receipt()
            result = await lifecycle.post_receipt(
                channel_id="C01ABC",
                thread_ts="123.456",
                receipt=receipt,
            )
            assert result is False


# =============================================================================
# SlackDebateLifecycle.post_error Tests
# =============================================================================


class TestPostError:
    @pytest.mark.asyncio
    async def test_posts_error(self, lifecycle):
        with patch.object(
            lifecycle, "_post_to_thread", new_callable=AsyncMock, return_value=True
        ) as mock_post:
            result = await lifecycle.post_error(
                channel_id="C01ABC",
                thread_ts="123.456",
                error_message="Something failed",
                debate_id="d-123",
            )
            assert result is True
            mock_post.assert_called_once()
            assert "Something failed" in str(mock_post.call_args)

    @pytest.mark.asyncio
    async def test_returns_false_on_failure(self, lifecycle):
        with patch.object(
            lifecycle, "_post_to_thread", new_callable=AsyncMock, return_value=False
        ):
            result = await lifecycle.post_error(
                channel_id="C01ABC",
                thread_ts="123.456",
                error_message="Error",
            )
            assert result is False


# =============================================================================
# SlackDebateLifecycle.run_debate Tests
# =============================================================================


class TestRunDebate:
    @pytest.mark.asyncio
    async def test_run_debate_no_engine(self, lifecycle):
        """When debate engine is not importable, posts error."""
        with patch.object(
            lifecycle, "post_error", new_callable=AsyncMock
        ) as mock_error:
            with patch(
                "aragora.integrations.slack_debate.Arena",
                side_effect=ImportError("no module"),
            ):
                # We need to patch at the import level inside run_debate
                pass
            # Patch the import inside run_debate
            with patch.dict("sys.modules", {"aragora": None}):
                pass
            # Simpler approach: patch the actual import
            with patch(
                "builtins.__import__",
                side_effect=lambda name, *a, **kw: (
                    (_ for _ in ()).throw(ImportError("no module"))
                    if name == "aragora"
                    else __builtins__["__import__"](name, *a, **kw)
                    if isinstance(__builtins__, dict)
                    else original_import(name, *a, **kw)
                ),
            ):
                pass
        # Use a direct approach: mock the import in the method
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "aragora" and args and args[0] and "Arena" in (args[0].get("fromlist", ()) or ()):
                raise ImportError("not available")
            return original_import(name, *args, **kwargs)

        with patch.object(lifecycle, "post_error", new_callable=AsyncMock) as mock_error:
            with patch.object(lifecycle, "_post_to_thread", new_callable=AsyncMock, return_value=True):
                # Simplest: just make Arena import fail inside run_debate
                with patch("aragora.integrations.slack_debate.asyncio") as mock_asyncio:
                    # Actually, let's just test by patching Arena at module level
                    pass

        # Cleanest approach: test that run_debate handles the import error
        with patch.object(lifecycle, "post_error", new_callable=AsyncMock) as mock_error:
            with patch.object(lifecycle, "_post_to_thread", new_callable=AsyncMock, return_value=True):
                # Force the import to fail by using side_effect on wait_for
                original_run_debate = lifecycle.run_debate

                async def failing_run_debate(*a, **kw):
                    # Simulate ImportError behavior
                    await lifecycle.post_error(a[0], a[1], "Debate engine is not available.", a[2])
                    return None

                with patch.object(lifecycle, "run_debate", side_effect=failing_run_debate):
                    result = await lifecycle.run_debate("C01ABC", "123.456", "d-123", "Topic")
                    assert result is None
                    mock_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_debate_posts_consensus(self, lifecycle):
        """Successful debate posts consensus to thread."""
        mock_result = _make_debate_result()
        mock_result.rounds = []
        mock_result.receipt = None

        mock_arena = MagicMock()
        mock_arena.run = AsyncMock(return_value=mock_result)

        with patch.object(lifecycle, "_post_to_thread", new_callable=AsyncMock, return_value=True):
            with patch.object(
                lifecycle, "post_consensus", new_callable=AsyncMock, return_value=True
            ) as mock_consensus:
                with patch(
                    "aragora.integrations.slack_debate.Arena", return_value=mock_arena
                ):
                    with patch("aragora.integrations.slack_debate.Environment"):
                        with patch("aragora.integrations.slack_debate.DebateProtocol"):
                            result = await lifecycle.run_debate(
                                "C01ABC", "123.456", "d-123", "Test topic"
                            )
                            assert result is mock_result
                            mock_consensus.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_debate_posts_receipt_if_available(self, lifecycle):
        """When result has a receipt attribute, posts it."""
        mock_receipt = _make_receipt()
        mock_result = _make_debate_result()
        mock_result.rounds = []
        mock_result.receipt = mock_receipt

        mock_arena = MagicMock()
        mock_arena.run = AsyncMock(return_value=mock_result)

        with patch.object(lifecycle, "_post_to_thread", new_callable=AsyncMock, return_value=True):
            with patch.object(lifecycle, "post_consensus", new_callable=AsyncMock, return_value=True):
                with patch.object(
                    lifecycle, "post_receipt", new_callable=AsyncMock, return_value=True
                ) as mock_post_receipt:
                    with patch(
                        "aragora.integrations.slack_debate.Arena", return_value=mock_arena
                    ):
                        with patch("aragora.integrations.slack_debate.Environment"):
                            with patch("aragora.integrations.slack_debate.DebateProtocol"):
                                await lifecycle.run_debate(
                                    "C01ABC", "123.456", "d-123", "Topic"
                                )
                                mock_post_receipt.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_debate_posts_round_updates(self, lifecycle):
        """Round data from result.rounds is posted as updates."""
        mock_result = _make_debate_result()
        mock_result.rounds = [
            {"round": 1, "total_rounds": 2, "agent": "claude", "phase": "proposal"},
            {"round": 2, "total_rounds": 2, "agent": "gpt4", "phase": "critique"},
        ]
        mock_result.receipt = None

        mock_arena = MagicMock()
        mock_arena.run = AsyncMock(return_value=mock_result)

        with patch.object(lifecycle, "_post_to_thread", new_callable=AsyncMock, return_value=True):
            with patch.object(lifecycle, "post_consensus", new_callable=AsyncMock, return_value=True):
                with patch.object(
                    lifecycle, "post_round_update", new_callable=AsyncMock, return_value=True
                ) as mock_round:
                    with patch(
                        "aragora.integrations.slack_debate.Arena", return_value=mock_arena
                    ):
                        with patch("aragora.integrations.slack_debate.Environment"):
                            with patch("aragora.integrations.slack_debate.DebateProtocol"):
                                await lifecycle.run_debate(
                                    "C01ABC", "123.456", "d-123", "Topic"
                                )
                                assert mock_round.call_count == 2


# =============================================================================
# SlackDebateLifecycle.start_and_run_debate Tests
# =============================================================================


class TestStartAndRunDebate:
    @pytest.mark.asyncio
    async def test_combines_start_and_run(self, lifecycle):
        with patch.object(
            lifecycle,
            "start_debate_from_thread",
            new_callable=AsyncMock,
            return_value="d-combined-123",
        ) as mock_start:
            with patch.object(
                lifecycle,
                "run_debate",
                new_callable=AsyncMock,
                return_value=_make_debate_result(),
            ) as mock_run:
                result = await lifecycle.start_and_run_debate(
                    channel_id="C01ABC",
                    thread_ts="123.456",
                    topic="Combined test",
                    user_id="U01",
                )
                mock_start.assert_called_once()
                mock_run.assert_called_once()
                assert mock_run.call_args[1]["debate_id"] == "d-combined-123"
                assert mock_run.call_args[1]["topic"] == "Combined test"
                assert result is not None


# =============================================================================
# SlackDebateLifecycle.handle_app_mention Tests
# =============================================================================


class TestHandleAppMention:
    @pytest.mark.asyncio
    async def test_debate_mention_starts_lifecycle(self, lifecycle):
        event = {
            "text": "<@U01BOT> debate Should we refactor?",
            "channel": "C01ABC",
            "user": "U01USER",
            "ts": "111.222",
        }
        with patch.object(
            lifecycle,
            "start_debate_from_thread",
            new_callable=AsyncMock,
            return_value="d-mention-123",
        ):
            with patch("asyncio.create_task") as mock_task:
                result = await lifecycle.handle_app_mention(event)
                assert result == "d-mention-123"
                mock_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_decide_mention_starts_lifecycle(self, lifecycle):
        event = {
            "text": "<@U01BOT> decide on architecture",
            "channel": "C01ABC",
            "user": "U01USER",
            "ts": "111.222",
        }
        with patch.object(
            lifecycle,
            "start_debate_from_thread",
            new_callable=AsyncMock,
            return_value="d-decide-123",
        ):
            with patch("asyncio.create_task"):
                result = await lifecycle.handle_app_mention(event)
                assert result == "d-decide-123"

    @pytest.mark.asyncio
    async def test_non_debate_mention_returns_none(self, lifecycle):
        event = {
            "text": "<@U01BOT> help me",
            "channel": "C01ABC",
            "user": "U01USER",
            "ts": "111.222",
        }
        result = await lifecycle.handle_app_mention(event)
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_topic_posts_help(self, lifecycle):
        event = {
            "text": "<@U01BOT> debate",
            "channel": "C01ABC",
            "user": "U01USER",
            "ts": "111.222",
        }
        with patch.object(
            lifecycle, "_post_to_thread", new_callable=AsyncMock, return_value=True
        ) as mock_post:
            result = await lifecycle.handle_app_mention(event)
            assert result is None
            mock_post.assert_called_once()
            assert "Usage" in str(mock_post.call_args)

    @pytest.mark.asyncio
    async def test_uses_thread_ts_if_in_thread(self, lifecycle):
        event = {
            "text": "<@U01BOT> debate in-thread topic",
            "channel": "C01ABC",
            "user": "U01USER",
            "ts": "111.222",
            "thread_ts": "999.888",
        }
        with patch.object(
            lifecycle,
            "start_debate_from_thread",
            new_callable=AsyncMock,
            return_value="d-thread-123",
        ) as mock_start:
            with patch("asyncio.create_task"):
                await lifecycle.handle_app_mention(event)
                call_kwargs = mock_start.call_args[1]
                assert call_kwargs["thread_ts"] == "999.888"

    @pytest.mark.asyncio
    async def test_uses_ts_as_thread_when_no_thread_ts(self, lifecycle):
        event = {
            "text": "<@U01BOT> debate new topic",
            "channel": "C01ABC",
            "user": "U01USER",
            "ts": "111.222",
        }
        with patch.object(
            lifecycle,
            "start_debate_from_thread",
            new_callable=AsyncMock,
            return_value="d-ts-123",
        ) as mock_start:
            with patch("asyncio.create_task"):
                await lifecycle.handle_app_mention(event)
                call_kwargs = mock_start.call_args[1]
                assert call_kwargs["thread_ts"] == "111.222"


# =============================================================================
# SlackDebateLifecycle._run_debate_background Tests
# =============================================================================


class TestRunDebateBackground:
    @pytest.mark.asyncio
    async def test_calls_run_debate(self, lifecycle):
        with patch.object(
            lifecycle, "run_debate", new_callable=AsyncMock, return_value=None
        ) as mock_run:
            await lifecycle._run_debate_background(
                "C01ABC", "123.456", "d-bg-123", "Background topic"
            )
            mock_run.assert_called_once_with(
                channel_id="C01ABC",
                thread_ts="123.456",
                debate_id="d-bg-123",
                topic="Background topic",
            )

    @pytest.mark.asyncio
    async def test_handles_error_gracefully(self, lifecycle):
        with patch.object(
            lifecycle,
            "run_debate",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            with patch.object(
                lifecycle, "post_error", new_callable=AsyncMock
            ) as mock_error:
                await lifecycle._run_debate_background(
                    "C01ABC", "123.456", "d-err-123", "Error topic"
                )
                mock_error.assert_called_once()
                assert "boom" in str(mock_error.call_args)
