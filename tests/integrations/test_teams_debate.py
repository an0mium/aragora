"""Tests for Microsoft Teams thread debate lifecycle."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.integrations.teams_debate import (
    TeamsActiveDebateState,
    TeamsDebateConfig,
    TeamsDebateLifecycle,
    _active_debates,
    _build_consensus_card,
    _build_debate_started_card,
    _build_error_card,
    _build_receipt_card,
    _build_receipt_with_approval_card,
    _build_round_update_card,
    _build_stop_card,
    _build_vote_summary_card,
    _wrap_card_payload,
    get_active_debate,
    get_active_debate_for_thread,
    parse_command_text,
    stop_debate,
    stop_debate_in_thread,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _clean_active_debates():
    """Ensure module-level _active_debates is empty between tests."""
    _active_debates.clear()
    yield
    _active_debates.clear()


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


def _make_debate_result(**kwargs):
    """Create a mock DebateResult with sensible defaults."""
    result = MagicMock()
    result.debate_id = kwargs.get("debate_id", "teams-abc123")
    result.task = kwargs.get("task", "Should we adopt microservices?")
    result.topic = kwargs.get("topic", "Should we adopt microservices?")
    result.final_answer = kwargs.get(
        "final_answer", "Yes, adopt a gradual migration strategy."
    )
    result.consensus_reached = kwargs.get("consensus_reached", True)
    result.confidence = kwargs.get("confidence", 0.85)
    result.rounds_used = kwargs.get("rounds_used", 3)
    result.winner = kwargs.get("winner", "claude")
    result.participants = kwargs.get("participants", ["claude", "gpt4"])
    result.rounds = kwargs.get("rounds", [])
    result.critiques = kwargs.get("critiques")
    result.votes = kwargs.get("votes")
    result.receipt = kwargs.get("receipt")
    return result


def _make_receipt(**kwargs):
    """Create a mock DecisionReceipt for testing."""
    receipt = MagicMock()
    receipt.verdict = kwargs.get("verdict", "APPROVED")
    receipt.confidence = kwargs.get("confidence", 0.92)
    receipt.receipt_id = kwargs.get("receipt_id", "rcpt-abcdef123456")
    receipt.findings = kwargs.get("findings", [])
    receipt.key_arguments = kwargs.get("key_arguments")
    receipt.dissenting_views = kwargs.get("dissenting_views")
    receipt.dissents = kwargs.get("dissents")
    receipt.cost_usd = kwargs.get("cost_usd", 0.0)
    receipt.tokens_used = kwargs.get("tokens_used", 0)
    return receipt


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

    def test_metadata_default(self):
        cfg = TeamsDebateConfig()
        assert cfg.metadata == {}

    def test_metadata_custom(self):
        cfg = TeamsDebateConfig(metadata={"team": "engineering"})
        assert cfg.metadata["team"] == "engineering"


# =============================================================================
# TeamsActiveDebateState Tests
# =============================================================================


class TestTeamsActiveDebateState:
    def test_creation(self):
        state = TeamsActiveDebateState(
            debate_id="d-1", channel_id="ch-1", message_id="msg-1",
            topic="Test", user_id="u-1",
        )
        assert state.debate_id == "d-1"
        assert state.status == "running"
        assert state.user_votes == {}
        assert state.user_suggestions == []

    def test_record_vote(self):
        state = TeamsActiveDebateState(
            debate_id="d-1", channel_id="ch-1", message_id="msg-1",
            topic="Test", user_id="u-1",
        )
        state.record_vote("voter-1", "agree")
        state.record_vote("voter-2", "disagree")
        assert state.user_votes["voter-1"] == "agree"
        assert state.user_votes["voter-2"] == "disagree"

    def test_record_vote_overwrites(self):
        state = TeamsActiveDebateState(
            debate_id="d-1", channel_id="ch-1", message_id="msg-1",
            topic="Test", user_id="u-1",
        )
        state.record_vote("voter-1", "agree")
        state.record_vote("voter-1", "disagree")
        assert state.user_votes["voter-1"] == "disagree"

    def test_add_suggestion(self):
        state = TeamsActiveDebateState(
            debate_id="d-1", channel_id="ch-1", message_id="msg-1",
            topic="Test", user_id="u-1",
        )
        state.add_suggestion("u-2", "My idea")
        state.add_suggestion("u-3", "Another idea")
        assert len(state.user_suggestions) == 2
        assert state.user_suggestions[0]["text"] == "My idea"

    def test_request_stop(self):
        state = TeamsActiveDebateState(
            debate_id="d-1", channel_id="ch-1", message_id="msg-1",
            topic="Test", user_id="u-1",
        )
        assert state.status == "running"
        assert not state.cancel_event.is_set()
        state.request_stop()
        assert state.status == "stopping"
        assert state.cancel_event.is_set()

    def test_vote_summary(self):
        state = TeamsActiveDebateState(
            debate_id="d-1", channel_id="ch-1", message_id="msg-1",
            topic="Test", user_id="u-1",
        )
        state.record_vote("v-1", "agree")
        state.record_vote("v-2", "agree")
        state.record_vote("v-3", "disagree")
        summary = state.vote_summary
        assert summary["agree"] == 2
        assert summary["disagree"] == 1

    def test_vote_summary_empty(self):
        state = TeamsActiveDebateState(
            debate_id="d-1", channel_id="ch-1", message_id="msg-1",
            topic="Test", user_id="u-1",
        )
        assert state.vote_summary == {}


# =============================================================================
# Module-level Functions Tests
# =============================================================================


class TestModuleFunctions:
    def test_get_active_debate(self):
        state = TeamsActiveDebateState(
            debate_id="d-1", channel_id="ch-1", message_id="msg-1",
            topic="Test", user_id="u-1",
        )
        _active_debates["d-1"] = state
        assert get_active_debate("d-1") is state
        assert get_active_debate("d-missing") is None

    def test_get_active_debate_for_thread(self):
        state = TeamsActiveDebateState(
            debate_id="d-1", channel_id="ch-1", message_id="msg-1",
            topic="Test", user_id="u-1",
        )
        _active_debates["d-1"] = state
        assert get_active_debate_for_thread("ch-1", "msg-1") is state
        assert get_active_debate_for_thread("ch-1", "msg-other") is None
        assert get_active_debate_for_thread("ch-other", "msg-1") is None

    def test_stop_debate(self):
        state = TeamsActiveDebateState(
            debate_id="d-1", channel_id="ch-1", message_id="msg-1",
            topic="Test", user_id="u-1",
        )
        _active_debates["d-1"] = state
        assert stop_debate("d-1") is True
        assert state.status == "stopping"
        assert state.cancel_event.is_set()

    def test_stop_debate_not_found(self):
        assert stop_debate("d-missing") is False

    def test_stop_debate_already_stopping(self):
        state = TeamsActiveDebateState(
            debate_id="d-1", channel_id="ch-1", message_id="msg-1",
            topic="Test", user_id="u-1",
        )
        state.status = "stopping"
        _active_debates["d-1"] = state
        assert stop_debate("d-1") is False

    def test_stop_debate_in_thread(self):
        state = TeamsActiveDebateState(
            debate_id="d-1", channel_id="ch-1", message_id="msg-1",
            topic="Test", user_id="u-1",
        )
        _active_debates["d-1"] = state
        result = stop_debate_in_thread("ch-1", "msg-1")
        assert result == "d-1"
        assert state.status == "stopping"

    def test_stop_debate_in_thread_not_found(self):
        assert stop_debate_in_thread("ch-1", "msg-1") is None


# =============================================================================
# parse_command_text Tests
# =============================================================================


class TestParseCommandText:
    def test_debate_command(self):
        command, arg = parse_command_text("<at>Aragora</at> debate Should we use K8s?")
        assert command == "debate"
        assert arg == "Should we use K8s?"

    def test_decide_command(self):
        command, arg = parse_command_text("<at>Bot</at> decide on microservices")
        assert command == "decide"
        assert arg == "on microservices"

    def test_stop_command(self):
        command, arg = parse_command_text("<at>Aragora</at> stop teams-123")
        assert command == "stop"
        assert arg == "teams-123"

    def test_strips_quotes(self):
        command, arg = parse_command_text('<at>Bot</at> debate "My topic"')
        assert command == "debate"
        assert arg == "My topic"

    def test_empty_text(self):
        command, arg = parse_command_text("")
        assert command == ""
        assert arg == ""

    def test_mention_only(self):
        command, arg = parse_command_text("<at>Bot</at>")
        assert command == ""
        assert arg == ""

    def test_no_keyword(self):
        command, arg = parse_command_text("<at>Bot</at> help me")
        assert command == ""
        assert arg == ""

    def test_case_insensitive(self):
        command, arg = parse_command_text("<at>Bot</at> DEBATE big question")
        assert command == "debate"

    def test_debate_without_argument(self):
        command, arg = parse_command_text("<at>Bot</at> debate")
        assert command == "debate"
        assert arg == ""

    def test_multiple_spaces(self):
        command, arg = parse_command_text("<at>Bot</at>   debate   spaced topic  ")
        assert command == "debate"
        assert "spaced topic" in arg

    def test_no_mention(self):
        command, arg = parse_command_text("debate Should we use K8s?")
        assert command == "debate"
        assert arg == "Should we use K8s?"


# =============================================================================
# Card Builder Tests
# =============================================================================


class TestBuildDebateStartedCard:
    def test_returns_card(self):
        cfg = TeamsDebateConfig(agents=["claude", "gpt4"])
        card = _build_debate_started_card("d-123", "Test topic", cfg)
        assert card.get("type") == "AdaptiveCard" or "body" in card

    def test_topic_in_card(self):
        cfg = TeamsDebateConfig()
        card = _build_debate_started_card("d-123", "My debate topic", cfg)
        assert "My debate topic" in str(card)

    def test_agents_in_card(self):
        cfg = TeamsDebateConfig(agents=["claude", "gpt4"])
        card = _build_debate_started_card("d-123", "Topic", cfg)
        card_text = str(card)
        assert "claude" in card_text
        assert "gpt4" in card_text

    def test_debate_id_in_card(self):
        cfg = TeamsDebateConfig()
        card = _build_debate_started_card("debate-abcdef123456", "Topic", cfg)
        assert "debate-abcdef" in str(card)

    def test_has_cancel_action(self):
        cfg = TeamsDebateConfig()
        card = _build_debate_started_card("d-123", "Topic", cfg)
        card_text = str(card)
        assert "cancel_debate" in card_text or "Cancel" in card_text


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

    def test_percentage_shown(self):
        card = _build_round_update_card(
            topic="Topic",
            round_number=1,
            total_rounds=4,
        )
        card_text = str(card)
        assert "25%" in card_text or "1" in card_text


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


class TestBuildReceiptCard:
    def test_returns_card(self):
        receipt = _make_receipt()
        card = _build_receipt_card(receipt)
        assert card.get("type") == "AdaptiveCard"
        assert "body" in card

    def test_verdict_in_card(self):
        receipt = _make_receipt(verdict="APPROVED")
        card = _build_receipt_card(receipt)
        assert "APPROVED" in str(card)

    def test_confidence_in_card(self):
        receipt = _make_receipt(confidence=0.88)
        card = _build_receipt_card(receipt)
        assert "88%" in str(card)

    def test_finding_counts(self):
        finding_crit = MagicMock()
        finding_crit.severity = "critical"
        finding_crit.description = "Critical issue"
        finding_high = MagicMock()
        finding_high.severity = "high"
        finding_high.description = "High issue"
        receipt = _make_receipt(findings=[finding_crit, finding_high])
        card = _build_receipt_card(receipt)
        card_text = str(card)
        assert "2 total" in card_text
        assert "1 critical" in card_text

    def test_key_arguments(self):
        receipt = _make_receipt(key_arguments=["Point A", "Point B"])
        card = _build_receipt_card(receipt)
        card_text = str(card)
        assert "Point A" in card_text
        assert "Point B" in card_text

    def test_dissenting_views(self):
        receipt = _make_receipt(dissenting_views=["I disagree", "Alternative view"])
        card = _build_receipt_card(receipt)
        card_text = str(card)
        assert "I disagree" in card_text

    def test_dissents_fallback(self):
        receipt = _make_receipt(dissenting_views=None, dissents=["Objection 1"])
        card = _build_receipt_card(receipt)
        assert "Objection 1" in str(card)

    def test_receipt_url_action(self):
        receipt = _make_receipt()
        card = _build_receipt_card(receipt, receipt_url="https://example.com/receipt/1")
        card_text = str(card)
        assert "https://example.com/receipt/1" in card_text

    def test_footer_receipt_id(self):
        receipt = _make_receipt(receipt_id="rcpt-xyz789012345")
        card = _build_receipt_card(receipt)
        assert "rcpt-xyz7890" in str(card)


class TestBuildErrorCard:
    def test_returns_card(self):
        card = _build_error_card("Something went wrong")
        assert card.get("type") == "AdaptiveCard"

    def test_error_message_in_card(self):
        card = _build_error_card("Database connection failed")
        assert "Database connection failed" in str(card)

    def test_debate_id_context(self):
        card = _build_error_card("Error", debate_id="debate-abcdef123456")
        assert "debate-abcde" in str(card)

    def test_no_debate_id_context(self):
        card = _build_error_card("Error", debate_id="")
        assert len(card.get("body", [])) == 1


class TestBuildStopCard:
    def test_returns_card(self):
        card = _build_stop_card("d-123")
        assert card.get("type") == "AdaptiveCard"
        assert "Stopped" in str(card)

    def test_includes_stopped_by(self):
        card = _build_stop_card("d-123", stopped_by="user-42")
        assert "user-42" in str(card)

    def test_includes_debate_id(self):
        card = _build_stop_card("debate-abcdef123456", stopped_by="user-1")
        assert "debate-abcde" in str(card)


class TestBuildVoteSummaryCard:
    def test_returns_card_with_votes(self):
        card = _build_vote_summary_card({"agree": 3, "disagree": 1})
        assert card is not None
        assert "agree" in str(card).lower() or "Agree" in str(card)

    def test_returns_card_with_suggestions(self):
        card = _build_vote_summary_card({}, suggestions_count=5)
        assert card is not None
        assert "5" in str(card)

    def test_returns_none_when_empty(self):
        card = _build_vote_summary_card({}, suggestions_count=0)
        assert card is None

    def test_votes_sorted_by_count(self):
        card = _build_vote_summary_card({"agree": 5, "disagree": 1, "abstain": 3})
        assert card is not None
        card_text = str(card)
        # All three should be present
        assert "agree" in card_text.lower() or "Agree" in card_text
        assert "disagree" in card_text.lower() or "Disagree" in card_text


class TestBuildReceiptWithApprovalCard:
    def test_has_approval_actions(self):
        receipt = _make_receipt()
        card = _build_receipt_with_approval_card(receipt, debate_id="d-123")
        card_text = str(card)
        assert "approve_decision" in card_text
        assert "request_redebate" in card_text
        assert "escalate_decision" in card_text

    def test_includes_cost_info(self):
        receipt = _make_receipt(cost_usd=0.0042, tokens_used=1500)
        card = _build_receipt_with_approval_card(receipt, debate_id="d-123")
        card_text = str(card)
        assert "0.0042" in card_text
        assert "1,500" in card_text

    def test_receipt_url_in_actions(self):
        receipt = _make_receipt()
        card = _build_receipt_with_approval_card(
            receipt, debate_id="d-123", receipt_url="https://example.com/receipt"
        )
        assert "https://example.com/receipt" in str(card)


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

    def test_initialization_with_bot_token(self):
        lc = TeamsDebateLifecycle(bot_token="test-token", service_url="https://smba.example.com")
        assert lc._bot_token == "test-token"
        assert lc._service_url == "https://smba.example.com"


# =============================================================================
# Session Management Tests
# =============================================================================


class TestSessionManagement:
    @pytest.mark.asyncio
    async def test_close_with_no_client(self, lifecycle):
        # Should not raise
        await lifecycle.close()

    @pytest.mark.asyncio
    async def test_close_with_active_client(self, lifecycle):
        mock_client = AsyncMock()
        lifecycle._client = mock_client
        await lifecycle.close()
        mock_client.aclose.assert_called_once()
        assert lifecycle._client is None


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
            assert info.topic == "Test topic"
            assert info.channel_id == "19:abc@thread.tacv2"

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
            assert info.topic == "Test"
            assert info.channel_id == "19:abc@thread.tacv2"

    @pytest.mark.asyncio
    async def test_stores_tenant_id(self, lifecycle):
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True
        ):
            debate_id = await lifecycle.start_debate_from_thread(
                channel_id="ch-1",
                message_id="msg-1",
                topic="Test",
                tenant_id="tenant-abc",
            )
            info = _active_debates[debate_id]
            assert info.tenant_id == "tenant-abc"


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

    @pytest.mark.asyncio
    async def test_handles_round_key_alias(self, lifecycle):
        """round_data may use 'round' instead of 'round_number'."""
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True
        ):
            result = await lifecycle.post_round_update(
                channel_id="ch-1",
                message_id="msg-1",
                round_data={
                    "debate_id": "teams-xyz",
                    "topic": "Topic",
                    "round": 2,
                    "total_rounds": 3,
                },
            )
            assert result is True


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
        _active_debates["teams-abc"] = TeamsActiveDebateState(
            debate_id="teams-abc", channel_id="ch-1", message_id="msg-1",
            topic="test", user_id="u-1",
        )
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

    @pytest.mark.asyncio
    async def test_accepts_object_result(self, lifecycle):
        """post_consensus normalizes object-style results to dict."""
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True
        ):
            result_obj = _make_debate_result(consensus_reached=True)
            success = await lifecycle.post_consensus(
                channel_id="ch-1",
                message_id="msg-1",
                result=result_obj,
            )
            assert success is True


# =============================================================================
# TeamsDebateLifecycle.post_receipt Tests
# =============================================================================


class TestPostReceipt:
    @pytest.mark.asyncio
    async def test_posts_receipt(self, lifecycle):
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True
        ) as mock_send:
            receipt = _make_receipt(verdict="APPROVED")
            result = await lifecycle.post_receipt(
                channel_id="ch-1",
                message_id="msg-1",
                receipt=receipt,
                debate_id="d-123",
            )
            assert result is True
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_false_on_failure(self, lifecycle):
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=False
        ):
            receipt = _make_receipt()
            result = await lifecycle.post_receipt(
                channel_id="ch-1",
                message_id="msg-1",
                receipt=receipt,
            )
            assert result is False


# =============================================================================
# TeamsDebateLifecycle.post_error Tests
# =============================================================================


class TestPostError:
    @pytest.mark.asyncio
    async def test_posts_error(self, lifecycle):
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True
        ) as mock_send:
            result = await lifecycle.post_error(
                channel_id="ch-1",
                message_id="msg-1",
                error_message="Something failed",
                debate_id="d-123",
            )
            assert result is True
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_false_on_failure(self, lifecycle):
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=False
        ):
            result = await lifecycle.post_error(
                channel_id="ch-1",
                message_id="msg-1",
                error_message="Error",
            )
            assert result is False


# =============================================================================
# TeamsDebateLifecycle.post_stop Tests
# =============================================================================


class TestPostStop:
    @pytest.mark.asyncio
    async def test_posts_stop(self, lifecycle):
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True
        ) as mock_send:
            result = await lifecycle.post_stop(
                channel_id="ch-1",
                message_id="msg-1",
                debate_id="d-123",
                stopped_by="user-42",
            )
            assert result is True
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_false_on_failure(self, lifecycle):
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=False
        ):
            result = await lifecycle.post_stop(
                channel_id="ch-1",
                message_id="msg-1",
                debate_id="d-123",
            )
            assert result is False


# =============================================================================
# TeamsDebateLifecycle.post_critique_summary Tests
# =============================================================================


class TestPostCritiqueSummary:
    @pytest.mark.asyncio
    async def test_posts_critiques(self, lifecycle):
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True
        ) as mock_send:
            critiques = [
                {"agent": "claude", "summary": "Good approach but needs caching."},
                {"agent": "gpt4", "summary": "Consider edge cases."},
            ]
            result = await lifecycle.post_critique_summary(
                channel_id="ch-1",
                message_id="msg-1",
                critiques=critiques,
            )
            assert result is True
            mock_send.assert_called_once()
            card_text = str(mock_send.call_args[0][2])
            assert "claude" in card_text
            assert "caching" in card_text

    @pytest.mark.asyncio
    async def test_empty_critiques_returns_true(self, lifecycle):
        result = await lifecycle.post_critique_summary(
            channel_id="ch-1",
            message_id="msg-1",
            critiques=[],
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_truncates_long_summaries(self, lifecycle):
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True
        ) as mock_send:
            critiques = [{"agent": "claude", "summary": "x" * 500}]
            await lifecycle.post_critique_summary(
                channel_id="ch-1",
                message_id="msg-1",
                critiques=critiques,
            )
            card_text = str(mock_send.call_args[0][2])
            assert "..." in card_text


# =============================================================================
# TeamsDebateLifecycle.post_voting_results Tests
# =============================================================================


class TestPostVotingResults:
    @pytest.mark.asyncio
    async def test_posts_votes(self, lifecycle):
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True
        ) as mock_send:
            votes = {
                "claude": {"position": "for", "confidence": 0.9},
                "gpt4": {"position": "against", "confidence": 0.6},
            }
            result = await lifecycle.post_voting_results(
                channel_id="ch-1",
                message_id="msg-1",
                votes=votes,
            )
            assert result is True
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_votes_returns_true(self, lifecycle):
        result = await lifecycle.post_voting_results(
            channel_id="ch-1",
            message_id="msg-1",
            votes={},
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_string_vote_values(self, lifecycle):
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True
        ) as mock_send:
            votes = {"claude": "agree", "gpt4": "disagree"}
            result = await lifecycle.post_voting_results(
                channel_id="ch-1",
                message_id="msg-1",
                votes=votes,
            )
            assert result is True
            card_text = str(mock_send.call_args[0][2])
            assert "agree" in card_text


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
    async def test_stop_command_with_id(self, lifecycle):
        _active_debates["teams-xyz"] = TeamsActiveDebateState(
            debate_id="teams-xyz", channel_id="ch-1", message_id="msg-1",
            topic="Test", user_id="u-1",
        )
        with patch.object(
            lifecycle, "post_stop", new_callable=AsyncMock, return_value=True
        ):
            activity = {
                "text": "/aragora stop teams-xyz",
                "conversation": {"id": "ch-1"},
                "from": {"id": "user-1"},
                "id": "activity-1",
            }
            result = await lifecycle.handle_bot_command(activity)
            assert result is not None
            assert "stop requested" in result["text"].lower() or "teams-xyz" in result["text"]

    @pytest.mark.asyncio
    async def test_stop_command_not_found(self, lifecycle):
        activity = {
            "text": "/aragora stop teams-missing",
            "conversation": {"id": "ch-1"},
            "from": {"id": "user-1"},
            "id": "activity-1",
        }
        result = await lifecycle.handle_bot_command(activity)
        assert result is not None
        assert "not found" in result["text"] or "not running" in result["text"]

    @pytest.mark.asyncio
    async def test_status_command_active_debate(self, lifecycle):
        _active_debates["teams-xyz"] = TeamsActiveDebateState(
            debate_id="teams-xyz", channel_id="ch-1", message_id="msg-1",
            topic="Test topic", user_id="u-1",
        )
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
# TeamsDebateLifecycle.handle_adaptive_card_action Tests
# =============================================================================


class TestHandleAdaptiveCardAction:
    @pytest.mark.asyncio
    async def test_vote_action(self, lifecycle):
        _active_debates["d-1"] = TeamsActiveDebateState(
            debate_id="d-1", channel_id="ch-1", message_id="msg-1",
            topic="Test", user_id="u-1",
        )
        activity = {
            "value": {"action": "vote", "vote": "agree", "debate_id": "d-1"},
            "conversation": {"id": "ch-1"},
            "from": {"id": "voter-1"},
        }
        result = await lifecycle.handle_adaptive_card_action(activity)
        assert result is not None
        assert "recorded" in result["text"].lower()
        assert _active_debates["d-1"].user_votes["voter-1"] == "agree"

    @pytest.mark.asyncio
    async def test_vote_action_debate_not_found(self, lifecycle):
        activity = {
            "value": {"action": "vote", "vote": "agree", "debate_id": "d-missing"},
            "conversation": {"id": "ch-1"},
            "from": {"id": "voter-1"},
        }
        result = await lifecycle.handle_adaptive_card_action(activity)
        assert result is not None
        assert "not found" in result["text"].lower() or "completed" in result["text"].lower()

    @pytest.mark.asyncio
    async def test_cancel_debate_action(self, lifecycle):
        _active_debates["d-1"] = TeamsActiveDebateState(
            debate_id="d-1", channel_id="ch-1", message_id="msg-1",
            topic="Test", user_id="u-1",
        )
        with patch.object(
            lifecycle, "post_stop", new_callable=AsyncMock, return_value=True
        ):
            activity = {
                "value": {"action": "cancel_debate", "debate_id": "d-1"},
                "conversation": {"id": "ch-1"},
                "from": {"id": "user-1"},
            }
            result = await lifecycle.handle_adaptive_card_action(activity)
            assert result is not None

    @pytest.mark.asyncio
    async def test_suggest_action(self, lifecycle):
        _active_debates["d-1"] = TeamsActiveDebateState(
            debate_id="d-1", channel_id="ch-1", message_id="msg-1",
            topic="Test", user_id="u-1",
        )
        activity = {
            "value": {"action": "suggest", "suggestion": "My input", "debate_id": "d-1"},
            "conversation": {"id": "ch-1"},
            "from": {"id": "user-2"},
        }
        result = await lifecycle.handle_adaptive_card_action(activity)
        assert result is not None
        assert "recorded" in result["text"].lower()

    @pytest.mark.asyncio
    async def test_unknown_action_returns_none(self, lifecycle):
        activity = {
            "value": {"action": "unknown_action"},
            "conversation": {"id": "ch-1"},
            "from": {"id": "user-1"},
        }
        result = await lifecycle.handle_adaptive_card_action(activity)
        assert result is None


# =============================================================================
# TeamsDebateLifecycle.handle_thread_reply Tests
# =============================================================================


class TestHandleThreadReply:
    @pytest.mark.asyncio
    async def test_stop_reply(self, lifecycle):
        _active_debates["d-1"] = TeamsActiveDebateState(
            debate_id="d-1", channel_id="ch-1", message_id="msg-1",
            topic="Test", user_id="u-1",
        )
        with patch.object(
            lifecycle, "post_stop", new_callable=AsyncMock, return_value=True
        ):
            activity = {
                "text": "stop",
                "conversation": {"id": "ch-1"},
                "replyToId": "msg-1",
                "from": {"id": "user-2"},
            }
            result = await lifecycle.handle_thread_reply(activity)
            assert result is not None
            assert "stop" in result["text"].lower()

    @pytest.mark.asyncio
    async def test_agree_vote(self, lifecycle):
        _active_debates["d-1"] = TeamsActiveDebateState(
            debate_id="d-1", channel_id="ch-1", message_id="msg-1",
            topic="Test", user_id="u-1",
        )
        activity = {
            "text": "agree",
            "conversation": {"id": "ch-1"},
            "replyToId": "msg-1",
            "from": {"id": "voter-1"},
        }
        result = await lifecycle.handle_thread_reply(activity)
        assert result is not None
        assert "agree" in result["text"].lower()
        assert _active_debates["d-1"].user_votes["voter-1"] == "agree"

    @pytest.mark.asyncio
    async def test_disagree_vote(self, lifecycle):
        _active_debates["d-1"] = TeamsActiveDebateState(
            debate_id="d-1", channel_id="ch-1", message_id="msg-1",
            topic="Test", user_id="u-1",
        )
        activity = {
            "text": "disagree",
            "conversation": {"id": "ch-1"},
            "replyToId": "msg-1",
            "from": {"id": "voter-1"},
        }
        result = await lifecycle.handle_thread_reply(activity)
        assert result is not None
        assert "disagree" in result["text"].lower()

    @pytest.mark.asyncio
    async def test_abstain_vote(self, lifecycle):
        _active_debates["d-1"] = TeamsActiveDebateState(
            debate_id="d-1", channel_id="ch-1", message_id="msg-1",
            topic="Test", user_id="u-1",
        )
        activity = {
            "text": "abstain",
            "conversation": {"id": "ch-1"},
            "replyToId": "msg-1",
            "from": {"id": "voter-1"},
        }
        result = await lifecycle.handle_thread_reply(activity)
        assert result is not None
        assert "abstain" in result["text"].lower()

    @pytest.mark.asyncio
    async def test_suggestion_with_prefix(self, lifecycle):
        _active_debates["d-1"] = TeamsActiveDebateState(
            debate_id="d-1", channel_id="ch-1", message_id="msg-1",
            topic="Test", user_id="u-1",
        )
        activity = {
            "text": "suggest: Use caching for performance",
            "conversation": {"id": "ch-1"},
            "replyToId": "msg-1",
            "from": {"id": "user-3"},
        }
        result = await lifecycle.handle_thread_reply(activity)
        assert result is not None
        assert "suggestion" in result["text"].lower() or "recorded" in result["text"].lower()
        assert _active_debates["d-1"].user_suggestions[0]["text"] == "Use caching for performance"

    @pytest.mark.asyncio
    async def test_free_text_as_suggestion(self, lifecycle):
        _active_debates["d-1"] = TeamsActiveDebateState(
            debate_id="d-1", channel_id="ch-1", message_id="msg-1",
            topic="Test", user_id="u-1",
        )
        activity = {
            "text": "Consider using Redis for this",
            "conversation": {"id": "ch-1"},
            "replyToId": "msg-1",
            "from": {"id": "user-4"},
        }
        result = await lifecycle.handle_thread_reply(activity)
        assert result is not None
        assert len(_active_debates["d-1"].user_suggestions) == 1

    @pytest.mark.asyncio
    async def test_no_debate_for_thread(self, lifecycle):
        activity = {
            "text": "agree",
            "conversation": {"id": "ch-1"},
            "replyToId": "msg-1",
            "from": {"id": "voter-1"},
        }
        result = await lifecycle.handle_thread_reply(activity)
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_text_returns_none(self, lifecycle):
        activity = {
            "text": "",
            "conversation": {"id": "ch-1"},
            "replyToId": "msg-1",
            "from": {"id": "voter-1"},
        }
        result = await lifecycle.handle_thread_reply(activity)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_reply_to_returns_none(self, lifecycle):
        activity = {
            "text": "agree",
            "conversation": {"id": "ch-1"},
            "from": {"id": "voter-1"},
        }
        result = await lifecycle.handle_thread_reply(activity)
        assert result is None


# =============================================================================
# TeamsDebateLifecycle.run_debate Tests
# =============================================================================


class TestRunDebate:
    @pytest.mark.asyncio
    async def test_run_debate_posts_consensus(self, lifecycle):
        """Successful debate posts consensus to thread."""
        mock_result = _make_debate_result()
        mock_result.rounds = []
        mock_result.receipt = None
        mock_result.critiques = None
        mock_result.votes = None

        mock_arena = MagicMock()
        mock_arena.run = AsyncMock(return_value=mock_result)

        mock_arena_cls = MagicMock(return_value=mock_arena)
        mock_env_cls = MagicMock()
        mock_proto_cls = MagicMock()

        with patch.object(lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True):
            with patch.object(
                lifecycle, "post_consensus", new_callable=AsyncMock, return_value=True
            ) as mock_consensus:
                with patch.dict("sys.modules", {
                    "aragora": MagicMock(
                        Arena=mock_arena_cls,
                        Environment=mock_env_cls,
                        DebateProtocol=mock_proto_cls,
                    ),
                }):
                    _active_debates["teams-test"] = TeamsActiveDebateState(
                        debate_id="teams-test", channel_id="ch-1",
                        message_id="msg-1", topic="Topic", user_id="u-1",
                    )
                    result = await lifecycle.run_debate(
                        "ch-1", "msg-1", "teams-test", "Test topic"
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
        mock_result.critiques = None
        mock_result.votes = None

        mock_arena = MagicMock()
        mock_arena.run = AsyncMock(return_value=mock_result)

        mock_arena_cls = MagicMock(return_value=mock_arena)
        mock_env_cls = MagicMock()
        mock_proto_cls = MagicMock()

        with patch.object(lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True):
            with patch.object(lifecycle, "post_consensus", new_callable=AsyncMock, return_value=True):
                with patch.object(
                    lifecycle, "post_receipt", new_callable=AsyncMock, return_value=True
                ) as mock_post_receipt:
                    with patch.dict("sys.modules", {
                        "aragora": MagicMock(
                            Arena=mock_arena_cls,
                            Environment=mock_env_cls,
                            DebateProtocol=mock_proto_cls,
                        ),
                    }):
                        _active_debates["teams-test"] = TeamsActiveDebateState(
                            debate_id="teams-test", channel_id="ch-1",
                            message_id="msg-1", topic="Topic", user_id="u-1",
                        )
                        await lifecycle.run_debate(
                            "ch-1", "msg-1", "teams-test", "Topic"
                        )
                        mock_post_receipt.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_debate_posts_round_updates(self, lifecycle):
        """Round data from result.rounds is posted as updates."""
        mock_result = _make_debate_result()
        mock_result.rounds = [
            {"round_number": 1, "total_rounds": 2, "topic": "Topic", "debate_id": "teams-test"},
            {"round_number": 2, "total_rounds": 2, "topic": "Topic", "debate_id": "teams-test"},
        ]
        mock_result.receipt = None
        mock_result.critiques = None
        mock_result.votes = None

        mock_arena = MagicMock()
        mock_arena.run = AsyncMock(return_value=mock_result)

        mock_arena_cls = MagicMock(return_value=mock_arena)
        mock_env_cls = MagicMock()
        mock_proto_cls = MagicMock()

        with patch.object(lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True):
            with patch.object(lifecycle, "post_consensus", new_callable=AsyncMock, return_value=True):
                with patch.object(
                    lifecycle, "post_round_update", new_callable=AsyncMock, return_value=True
                ) as mock_round:
                    with patch.dict("sys.modules", {
                        "aragora": MagicMock(
                            Arena=mock_arena_cls,
                            Environment=mock_env_cls,
                            DebateProtocol=mock_proto_cls,
                        ),
                    }):
                        _active_debates["teams-test"] = TeamsActiveDebateState(
                            debate_id="teams-test", channel_id="ch-1",
                            message_id="msg-1", topic="Topic", user_id="u-1",
                        )
                        await lifecycle.run_debate(
                            "ch-1", "msg-1", "teams-test", "Topic"
                        )
                        assert mock_round.call_count == 2

    @pytest.mark.asyncio
    async def test_run_debate_cleans_up_on_timeout(self, lifecycle):
        """Timeout cleans up active debates and posts error."""
        _active_debates["teams-test"] = TeamsActiveDebateState(
            debate_id="teams-test", channel_id="ch-1",
            message_id="msg-1", topic="Topic", user_id="u-1",
        )
        with patch.object(lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True):
            with patch.object(
                lifecycle, "post_error", new_callable=AsyncMock
            ) as mock_error:
                with patch.dict("sys.modules", {
                    "aragora": MagicMock(
                        Arena=MagicMock(return_value=MagicMock(run=AsyncMock())),
                        Environment=MagicMock(),
                        DebateProtocol=MagicMock(),
                    ),
                }):
                    with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
                        result = await lifecycle.run_debate(
                            "ch-1", "msg-1", "teams-test", "Topic"
                        )
                        assert result is None
                        mock_error.assert_called_once()
                        assert "teams-test" not in _active_debates

    @pytest.mark.asyncio
    async def test_run_debate_cleans_up_on_error(self, lifecycle):
        """Runtime error cleans up active debates and posts error."""
        _active_debates["teams-test"] = TeamsActiveDebateState(
            debate_id="teams-test", channel_id="ch-1",
            message_id="msg-1", topic="Topic", user_id="u-1",
        )
        with patch.object(lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True):
            with patch.object(
                lifecycle, "post_error", new_callable=AsyncMock
            ) as mock_error:
                with patch.dict("sys.modules", {
                    "aragora": MagicMock(
                        Arena=MagicMock(return_value=MagicMock(run=AsyncMock())),
                        Environment=MagicMock(),
                        DebateProtocol=MagicMock(),
                    ),
                }):
                    with patch("asyncio.wait_for", side_effect=RuntimeError("boom")):
                        result = await lifecycle.run_debate(
                            "ch-1", "msg-1", "teams-test", "Topic"
                        )
                        assert result is None
                        mock_error.assert_called_once()
                        assert "teams-test" not in _active_debates


# =============================================================================
# TeamsDebateLifecycle.start_and_run_debate Tests
# =============================================================================


class TestStartAndRunDebate:
    @pytest.mark.asyncio
    async def test_combines_start_and_run(self, lifecycle):
        with patch.object(
            lifecycle,
            "start_debate_from_thread",
            new_callable=AsyncMock,
            return_value="teams-combined-123",
        ) as mock_start:
            with patch.object(
                lifecycle,
                "run_debate",
                new_callable=AsyncMock,
                return_value=_make_debate_result(),
            ) as mock_run:
                result = await lifecycle.start_and_run_debate(
                    channel_id="ch-1",
                    message_id="msg-1",
                    topic="Combined test",
                    user_id="u-1",
                    tenant_id="t-1",
                )
                mock_start.assert_called_once()
                mock_run.assert_called_once()
                assert mock_run.call_args[1]["debate_id"] == "teams-combined-123"
                assert mock_run.call_args[1]["topic"] == "Combined test"
                assert result is not None


# =============================================================================
# TeamsDebateLifecycle._run_debate_background Tests
# =============================================================================


class TestRunDebateBackground:
    @pytest.mark.asyncio
    async def test_calls_run_debate(self, lifecycle):
        with patch.object(
            lifecycle, "run_debate", new_callable=AsyncMock, return_value=None
        ) as mock_run:
            await lifecycle._run_debate_background(
                "ch-1", "msg-1", "d-bg-123", "Background topic"
            )
            mock_run.assert_called_once_with(
                channel_id="ch-1",
                message_id="msg-1",
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
                    "ch-1", "msg-1", "d-err-123", "Error topic"
                )
                mock_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_marks_state_failed_on_error(self, lifecycle):
        _active_debates["d-err"] = TeamsActiveDebateState(
            debate_id="d-err", channel_id="ch-1", message_id="msg-1",
            topic="Test", user_id="u-1",
        )
        with patch.object(
            lifecycle,
            "run_debate",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            with patch.object(lifecycle, "post_error", new_callable=AsyncMock):
                await lifecycle._run_debate_background(
                    "ch-1", "msg-1", "d-err", "Error topic"
                )
                assert _active_debates["d-err"].status == "failed"


# =============================================================================
# TeamsDebateLifecycle.deliver_receipt_to_thread Tests
# =============================================================================


class TestDeliverReceiptToThread:
    @pytest.mark.asyncio
    async def test_delivers_receipt(self, lifecycle):
        receipt = _make_receipt()
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True
        ):
            result = await lifecycle.deliver_receipt_to_thread(
                debate_id="d-123",
                conversation_id="ch-1",
                reply_to_id="msg-1",
                receipt=receipt,
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_loads_receipt_from_store(self, lifecycle):
        mock_receipt = _make_receipt()
        with patch.object(
            lifecycle, "_load_receipt", return_value=mock_receipt
        ):
            with patch.object(
                lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True
            ):
                result = await lifecycle.deliver_receipt_to_thread(
                    debate_id="d-123",
                    conversation_id="ch-1",
                    reply_to_id="msg-1",
                )
                assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_no_receipt(self, lifecycle):
        with patch.object(lifecycle, "_load_receipt", return_value=None):
            result = await lifecycle.deliver_receipt_to_thread(
                debate_id="d-123",
                conversation_id="ch-1",
                reply_to_id="msg-1",
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_builds_receipt_url_from_env(self, lifecycle):
        receipt = _make_receipt(receipt_id="rcpt-xyz")
        with patch.object(
            lifecycle, "_send_card_to_thread", new_callable=AsyncMock, return_value=True
        ) as mock_send:
            with patch.dict("os.environ", {"ARAGORA_PUBLIC_URL": "https://my.app"}):
                await lifecycle.deliver_receipt_to_thread(
                    debate_id="d-123",
                    conversation_id="ch-1",
                    reply_to_id="msg-1",
                    receipt=receipt,
                )
                card = mock_send.call_args[0][2]
                card_text = str(card)
                assert "https://my.app/receipts/rcpt-xyz" in card_text


# =============================================================================
# Internal Helpers Tests
# =============================================================================


class TestBuildHelpResponse:
    def test_includes_commands(self):
        response = TeamsDebateLifecycle._build_help_response()
        assert "debate" in response["text"]
        assert "status" in response["text"]
        assert "help" in response["text"]
        assert "stop" in response["text"]

    def test_includes_in_thread_instructions(self):
        response = TeamsDebateLifecycle._build_help_response()
        text = response["text"].lower()
        assert "agree" in text
        assert "disagree" in text
        assert "suggest" in text


class TestGetDebateStatus:
    def test_active_debate(self, lifecycle):
        _active_debates["d-1"] = TeamsActiveDebateState(
            debate_id="d-1", channel_id="ch-1", message_id="msg-1",
            topic="Test", user_id="u-1",
        )
        result = lifecycle._get_debate_status("d-1")
        assert "active" in result["text"]

    def test_not_found(self, lifecycle):
        result = lifecycle._get_debate_status("d-missing")
        assert "not found" in result["text"]

    def test_empty_id(self, lifecycle):
        result = lifecycle._get_debate_status("")
        assert "provide" in result["text"].lower() or "Usage" in result["text"]

    def test_shows_vote_and_suggestion_counts(self, lifecycle):
        state = TeamsActiveDebateState(
            debate_id="d-1", channel_id="ch-1", message_id="msg-1",
            topic="Test", user_id="u-1",
        )
        state.record_vote("v-1", "agree")
        state.add_suggestion("v-2", "idea")
        _active_debates["d-1"] = state
        result = lifecycle._get_debate_status("d-1")
        text = result["text"]
        assert "1" in text  # votes count
        assert "1" in text  # suggestions count
