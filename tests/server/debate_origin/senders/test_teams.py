"""Tests for Microsoft Teams sender for debate origin result routing.

Tests cover:
1. Result card building (rich and fallback)
2. Message wrapping for Bot Framework
3. Proactive messaging via Bot Framework
4. Webhook fallback delivery
5. Result, receipt, and error message sending
6. Error handling and retries
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.debate_origin.models import DebateOrigin
from aragora.server.debate_origin.senders.teams import (
    _build_result_card,
    _wrap_card_as_message,
    _send_via_proactive,
    _send_via_webhook,
    _send_teams_result,
    _send_teams_receipt,
    _send_teams_error,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_origin() -> DebateOrigin:
    """Create a sample Teams debate origin for testing."""
    return DebateOrigin(
        debate_id="debate-teams-123",
        platform="teams",
        channel_id="19:abc123@thread.tacv2",
        user_id="user-teams-456",
        metadata={
            "topic": "Teams Integration Test",
            "webhook_url": "https://outlook.office.com/webhook/test",
        },
    )


@pytest.fixture
def sample_result() -> dict[str, Any]:
    """Create a sample debate result for testing."""
    return {
        "consensus_reached": True,
        "final_answer": "The team reached agreement on the approach.",
        "confidence": 0.85,
        "participants": ["claude", "gpt-4", "gemini"],
        "task": "Evaluate the proposal",
        "key_points": ["Point 1", "Point 2"],
        "dissenting_agents": [],
        "vote_summary": {"agree": 3, "disagree": 0},
    }


# =============================================================================
# Test: Result Card Building
# =============================================================================


class TestBuildResultCard:
    """Tests for _build_result_card function."""

    def test_builds_fallback_card_structure(self, sample_origin, sample_result):
        """_build_result_card returns valid Adaptive Card structure."""
        # Patch to force fallback card (no teams_cards import)
        with patch.dict("sys.modules", {"aragora.server.handlers.bots.teams_cards": None}):
            card = _build_result_card(sample_result, sample_origin)

        assert card["type"] == "AdaptiveCard"
        assert card["version"] == "1.4"
        assert "$schema" in card
        assert "body" in card
        assert "actions" in card

    def test_includes_consensus_status_in_card(self, sample_origin, sample_result):
        """_build_result_card shows consensus reached status."""
        with patch.dict("sys.modules", {"aragora.server.handlers.bots.teams_cards": None}):
            card = _build_result_card(sample_result, sample_origin)

        body = card["body"]
        # First TextBlock should indicate consensus
        assert any(
            "Consensus Reached" in str(item.get("text", ""))
            for item in body
            if isinstance(item, dict)
        )

    def test_shows_no_consensus_when_not_reached(self, sample_origin, sample_result):
        """_build_result_card shows 'Debate Complete' when no consensus."""
        sample_result["consensus_reached"] = False
        with patch.dict("sys.modules", {"aragora.server.handlers.bots.teams_cards": None}):
            card = _build_result_card(sample_result, sample_origin)

        body = card["body"]
        assert any(
            "Debate Complete" in str(item.get("text", ""))
            for item in body
            if isinstance(item, dict)
        )

    def test_includes_confidence_in_facts(self, sample_origin, sample_result):
        """_build_result_card includes confidence percentage in facts."""
        with patch.dict("sys.modules", {"aragora.server.handlers.bots.teams_cards": None}):
            card = _build_result_card(sample_result, sample_origin)

        # Find FactSet in body
        fact_sets = [item for item in card["body"] if item.get("type") == "FactSet"]
        assert len(fact_sets) >= 1
        facts = fact_sets[0]["facts"]
        confidence_fact = next((f for f in facts if f.get("title") == "Confidence"), None)
        assert confidence_fact is not None
        assert "85%" in confidence_fact["value"]

    def test_includes_participants_in_facts(self, sample_origin, sample_result):
        """_build_result_card lists participants in facts."""
        with patch.dict("sys.modules", {"aragora.server.handlers.bots.teams_cards": None}):
            card = _build_result_card(sample_result, sample_origin)

        fact_sets = [item for item in card["body"] if item.get("type") == "FactSet"]
        facts = fact_sets[0]["facts"]
        agents_fact = next((f for f in facts if f.get("title") == "Agents"), None)
        assert agents_fact is not None
        assert "claude" in agents_fact["value"]
        assert "gpt-4" in agents_fact["value"]

    def test_truncates_long_final_answer(self, sample_origin, sample_result):
        """_build_result_card truncates answers over 500 chars."""
        sample_result["final_answer"] = "X" * 700
        with patch.dict("sys.modules", {"aragora.server.handlers.bots.teams_cards": None}):
            card = _build_result_card(sample_result, sample_origin)

        # Find the decision container
        body_str = str(card["body"])
        assert "X" * 500 in body_str
        assert "..." in body_str
        assert "X" * 501 not in body_str

    def test_truncates_long_topic(self, sample_origin, sample_result):
        """_build_result_card truncates topics over 200 chars."""
        sample_result["task"] = "T" * 300
        with patch.dict("sys.modules", {"aragora.server.handlers.bots.teams_cards": None}):
            card = _build_result_card(sample_result, sample_origin)

        body_str = str(card["body"])
        # Topic should be truncated to 200 chars
        assert "T" * 200 in body_str
        assert "T" * 201 not in body_str

    def test_includes_action_buttons(self, sample_origin, sample_result):
        """_build_result_card includes View Full Report and Share actions."""
        with patch.dict("sys.modules", {"aragora.server.handlers.bots.teams_cards": None}):
            card = _build_result_card(sample_result, sample_origin)

        actions = card["actions"]
        assert len(actions) >= 2

        # Check for View Full Report
        view_action = next((a for a in actions if "View Full Report" in a.get("title", "")), None)
        assert view_action is not None
        assert f"debate/{sample_origin.debate_id}" in view_action["url"]

        # Check for Share action
        share_action = next((a for a in actions if "Share" in a.get("title", "")), None)
        assert share_action is not None
        assert share_action["type"] == "Action.Submit"

    def test_uses_origin_topic_as_fallback(self, sample_origin, sample_result):
        """_build_result_card uses origin metadata topic when task missing."""
        del sample_result["task"]
        with patch.dict("sys.modules", {"aragora.server.handlers.bots.teams_cards": None}):
            card = _build_result_card(sample_result, sample_origin)

        body_str = str(card["body"])
        assert "Teams Integration Test" in body_str

    def test_handles_empty_participants(self, sample_origin, sample_result):
        """_build_result_card handles empty participants list."""
        sample_result["participants"] = []
        with patch.dict("sys.modules", {"aragora.server.handlers.bots.teams_cards": None}):
            card = _build_result_card(sample_result, sample_origin)

        # Should still produce valid card
        assert card["type"] == "AdaptiveCard"

    def test_uses_rich_card_when_teams_cards_available(self, sample_origin, sample_result):
        """_build_result_card uses teams_cards template when available."""
        mock_card = {
            "type": "AdaptiveCard",
            "version": "1.5",
            "body": [{"type": "TextBlock", "text": "Rich Card"}],
        }
        mock_create_consensus_card = MagicMock(return_value=mock_card)
        mock_teams_cards = MagicMock(create_consensus_card=mock_create_consensus_card)

        with patch.dict(
            "sys.modules", {"aragora.server.handlers.bots.teams_cards": mock_teams_cards}
        ):
            card = _build_result_card(sample_result, sample_origin)

        assert card == mock_card
        mock_create_consensus_card.assert_called_once()

    def test_extracts_key_points_from_bullet_answer(self, sample_origin, sample_result):
        """_build_result_card extracts bullet points from answer when no key_points."""
        sample_result["key_points"] = []
        sample_result["final_answer"] = (
            "Summary:\n- First point here\n- Second point here\n* Third point here"
        )

        mock_card = {"type": "AdaptiveCard", "body": []}
        mock_create_consensus_card = MagicMock(return_value=mock_card)
        mock_teams_cards = MagicMock(create_consensus_card=mock_create_consensus_card)

        with patch.dict(
            "sys.modules", {"aragora.server.handlers.bots.teams_cards": mock_teams_cards}
        ):
            _build_result_card(sample_result, sample_origin)

        call_kwargs = mock_create_consensus_card.call_args[1]
        # Should have extracted key points
        assert call_kwargs.get("key_points") is not None
        assert len(call_kwargs["key_points"]) == 3


# =============================================================================
# Test: Message Wrapping
# =============================================================================


class TestWrapCardAsMessage:
    """Tests for _wrap_card_as_message function."""

    def test_wraps_card_correctly(self):
        """_wrap_card_as_message wraps card in Bot Framework message format."""
        card = {"type": "AdaptiveCard", "body": []}
        message = _wrap_card_as_message(card)

        assert message["type"] == "message"
        assert "attachments" in message
        assert len(message["attachments"]) == 1
        assert message["attachments"][0]["contentType"] == "application/vnd.microsoft.card.adaptive"
        assert message["attachments"][0]["content"] == card


# =============================================================================
# Test: Proactive Messaging
# =============================================================================


class TestSendViaProactive:
    """Tests for _send_via_proactive function."""

    @pytest.mark.asyncio
    async def test_sends_message_via_bot_framework(self, sample_origin):
        """_send_via_proactive sends through TeamsBot when available."""
        mock_bot = MagicMock()
        mock_bot.send_proactive_message = AsyncMock(return_value=True)
        mock_ref = {"conversation": {"id": sample_origin.channel_id}}

        mock_teams_module = MagicMock(
            TeamsBot=MagicMock(return_value=mock_bot),
            get_conversation_reference=MagicMock(return_value=mock_ref),
        )

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.teams": mock_teams_module}):
            result = await _send_via_proactive(sample_origin, text="Test message")

        assert result is True
        mock_bot.send_proactive_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_no_conversation_reference(self, sample_origin):
        """_send_via_proactive returns None when no conversation reference stored."""
        mock_teams_module = MagicMock(
            TeamsBot=MagicMock(),
            get_conversation_reference=MagicMock(return_value=None),
        )

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.teams": mock_teams_module}):
            result = await _send_via_proactive(sample_origin, text="Test")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_teams_module_unavailable(self, sample_origin):
        """_send_via_proactive returns None when TeamsBot module not available."""
        # Force ImportError by setting module to None
        with patch.dict("sys.modules", {"aragora.server.handlers.bots.teams": None}):
            # Need to actually trigger the ImportError
            import builtins

            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if "aragora.server.handlers.bots.teams" in name:
                    raise ImportError("No teams module")
                return original_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", mock_import):
                result = await _send_via_proactive(sample_origin, text="Test")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_runtime_error(self, sample_origin):
        """_send_via_proactive returns None on runtime errors."""
        mock_teams_module = MagicMock(
            get_conversation_reference=MagicMock(side_effect=RuntimeError("Connection failed")),
        )

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.teams": mock_teams_module}):
            result = await _send_via_proactive(sample_origin, text="Test")

        assert result is None

    @pytest.mark.asyncio
    async def test_sends_card_with_fallback_text(self, sample_origin):
        """_send_via_proactive passes card and fallback text to bot."""
        mock_bot = MagicMock()
        mock_bot.send_proactive_message = AsyncMock(return_value=True)
        mock_ref = {"conversation": {"id": sample_origin.channel_id}}

        mock_teams_module = MagicMock(
            TeamsBot=MagicMock(return_value=mock_bot),
            get_conversation_reference=MagicMock(return_value=mock_ref),
        )

        test_card = {"type": "AdaptiveCard", "body": []}

        with patch.dict("sys.modules", {"aragora.server.handlers.bots.teams": mock_teams_module}):
            await _send_via_proactive(sample_origin, card=test_card, fallback_text="Fallback")

        call_kwargs = mock_bot.send_proactive_message.call_args[1]
        assert call_kwargs["card"] == test_card
        assert call_kwargs["fallback_text"] == "Fallback"


# =============================================================================
# Test: Webhook Delivery
# =============================================================================


class TestSendViaWebhook:
    """Tests for _send_via_webhook function."""

    @pytest.mark.asyncio
    async def test_posts_to_webhook_successfully(self):
        """_send_via_webhook posts payload to webhook URL."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await _send_via_webhook(
                "https://outlook.office.com/webhook/test",
                {"type": "message", "text": "Test"},
                "result",
            )

        assert result is True
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_false_on_http_error(self):
        """_send_via_webhook returns False on non-success HTTP status."""
        mock_response = MagicMock()
        mock_response.is_success = False
        mock_response.status_code = 400

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await _send_via_webhook(
                "https://outlook.office.com/webhook/test",
                {"type": "message"},
                "result",
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_connection_error(self):
        """_send_via_webhook returns False on connection errors."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=OSError("Connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await _send_via_webhook(
                "https://outlook.office.com/webhook/test",
                {"type": "message"},
                "result",
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_httpx_unavailable(self):
        """_send_via_webhook returns False when httpx not available."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "httpx":
                raise ImportError("No httpx")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", mock_import):
            result = await _send_via_webhook(
                "https://outlook.office.com/webhook/test",
                {"type": "message"},
                "result",
            )

        assert result is False


# =============================================================================
# Test: Send Teams Result
# =============================================================================


class TestSendTeamsResult:
    """Tests for _send_teams_result function."""

    @pytest.mark.asyncio
    async def test_sends_via_proactive_first(self, sample_origin, sample_result):
        """_send_teams_result tries proactive messaging first."""
        with patch(
            "aragora.server.debate_origin.senders.teams._send_via_proactive",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_proactive:
            with patch(
                "aragora.server.debate_origin.senders.teams._send_via_webhook",
                new_callable=AsyncMock,
            ) as mock_webhook:
                result = await _send_teams_result(sample_origin, sample_result)

        assert result is True
        mock_proactive.assert_called_once()
        mock_webhook.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_to_webhook(self, sample_origin, sample_result):
        """_send_teams_result falls back to webhook when proactive unavailable."""
        with patch(
            "aragora.server.debate_origin.senders.teams._send_via_proactive",
            new_callable=AsyncMock,
            return_value=None,  # Proactive not available
        ):
            with patch(
                "aragora.server.debate_origin.senders.teams._send_via_webhook",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_webhook:
                result = await _send_teams_result(sample_origin, sample_result)

        assert result is True
        mock_webhook.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_false_when_no_webhook_url(self, sample_origin, sample_result):
        """_send_teams_result returns False when no webhook and proactive unavailable."""
        sample_origin.metadata = {}  # No webhook_url

        with patch(
            "aragora.server.debate_origin.senders.teams._send_via_proactive",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await _send_teams_result(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_tries_webhook_when_proactive_returns_false(self, sample_origin, sample_result):
        """_send_teams_result tries webhook when proactive explicitly fails."""
        with patch(
            "aragora.server.debate_origin.senders.teams._send_via_proactive",
            new_callable=AsyncMock,
            return_value=False,  # Proactive failed
        ):
            with patch(
                "aragora.server.debate_origin.senders.teams._send_via_webhook",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_webhook:
                result = await _send_teams_result(sample_origin, sample_result)

        assert result is True
        mock_webhook.assert_called_once()


# =============================================================================
# Test: Send Teams Receipt
# =============================================================================


class TestSendTeamsReceipt:
    """Tests for _send_teams_receipt function."""

    @pytest.mark.asyncio
    async def test_sends_receipt_card(self, sample_origin):
        """_send_teams_receipt sends receipt with link button."""
        with patch(
            "aragora.server.debate_origin.senders.teams._send_via_proactive",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_proactive:
            result = await _send_teams_receipt(
                sample_origin,
                "Decision approved with high confidence",
                "https://aragora.ai/receipt/123",
            )

        assert result is True
        mock_proactive.assert_called_once()

        # Verify card structure
        call_kwargs = mock_proactive.call_args[1]
        card = call_kwargs["card"]
        assert card["type"] == "AdaptiveCard"

        # Check for receipt title
        body_str = str(card["body"])
        assert "Decision Receipt" in body_str

        # Check for View Full Receipt action
        actions = card["actions"]
        assert any("View Full Receipt" in a.get("title", "") for a in actions)
        assert any("https://aragora.ai/receipt/123" in a.get("url", "") for a in actions)

    @pytest.mark.asyncio
    async def test_falls_back_to_webhook_for_receipt(self, sample_origin):
        """_send_teams_receipt falls back to webhook delivery."""
        with patch(
            "aragora.server.debate_origin.senders.teams._send_via_proactive",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with patch(
                "aragora.server.debate_origin.senders.teams._send_via_webhook",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_webhook:
                result = await _send_teams_receipt(
                    sample_origin,
                    "Receipt summary",
                    "https://aragora.ai/receipt/456",
                )

        assert result is True
        mock_webhook.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_false_when_no_delivery_option(self, sample_origin):
        """_send_teams_receipt returns False when no delivery method available."""
        sample_origin.metadata = {}  # No webhook

        with patch(
            "aragora.server.debate_origin.senders.teams._send_via_proactive",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await _send_teams_receipt(
                sample_origin,
                "Receipt",
                "https://aragora.ai/receipt/789",
            )

        assert result is False


# =============================================================================
# Test: Send Teams Error
# =============================================================================


class TestSendTeamsError:
    """Tests for _send_teams_error function."""

    @pytest.mark.asyncio
    async def test_sends_error_card(self, sample_origin):
        """_send_teams_error sends error with attention styling."""
        with patch(
            "aragora.server.debate_origin.senders.teams._send_via_proactive",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_proactive:
            # Force fallback card by making teams_cards unavailable
            with patch.dict("sys.modules", {"aragora.server.handlers.bots.teams_cards": None}):
                import builtins

                original_import = builtins.__import__

                def mock_import(name, *args, **kwargs):
                    if "teams_cards" in name:
                        raise ImportError("No teams_cards")
                    return original_import(name, *args, **kwargs)

                with patch.object(builtins, "__import__", mock_import):
                    result = await _send_teams_error(sample_origin, "Something went wrong")

        assert result is True
        mock_proactive.assert_called_once()

        # Verify card has error styling
        call_kwargs = mock_proactive.call_args[1]
        card = call_kwargs["card"]
        body_str = str(card["body"])
        assert "Aragora Notice" in body_str
        assert "Something went wrong" in body_str

    @pytest.mark.asyncio
    async def test_uses_error_card_template_when_available(self, sample_origin):
        """_send_teams_error uses teams_cards error template when available."""
        mock_error_card = {
            "type": "AdaptiveCard",
            "body": [{"type": "TextBlock", "text": "Error Template"}],
        }
        mock_create_error_card = MagicMock(return_value=mock_error_card)
        mock_teams_cards = MagicMock(create_error_card=mock_create_error_card)

        with patch.dict(
            "sys.modules", {"aragora.server.handlers.bots.teams_cards": mock_teams_cards}
        ):
            with patch(
                "aragora.server.debate_origin.senders.teams._send_via_proactive",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_proactive:
                result = await _send_teams_error(sample_origin, "Test error")

        assert result is True
        mock_create_error_card.assert_called_once()

        # Verify the error card was sent
        call_kwargs = mock_proactive.call_args[1]
        assert call_kwargs["card"] == mock_error_card

    @pytest.mark.asyncio
    async def test_falls_back_to_webhook_for_error(self, sample_origin):
        """_send_teams_error falls back to webhook delivery."""
        with patch(
            "aragora.server.debate_origin.senders.teams._send_via_proactive",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with patch(
                "aragora.server.debate_origin.senders.teams._send_via_webhook",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_webhook:
                with patch.dict("sys.modules", {"aragora.server.handlers.bots.teams_cards": None}):
                    import builtins

                    original_import = builtins.__import__

                    def mock_import(name, *args, **kwargs):
                        if "teams_cards" in name:
                            raise ImportError("No teams_cards")
                        return original_import(name, *args, **kwargs)

                    with patch.object(builtins, "__import__", mock_import):
                        result = await _send_teams_error(sample_origin, "Error message")

        assert result is True
        mock_webhook.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_false_when_no_delivery_option(self, sample_origin):
        """_send_teams_error returns False when no delivery method available."""
        sample_origin.metadata = {}  # No webhook

        with patch(
            "aragora.server.debate_origin.senders.teams._send_via_proactive",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with patch.dict("sys.modules", {"aragora.server.handlers.bots.teams_cards": None}):
                import builtins

                original_import = builtins.__import__

                def mock_import(name, *args, **kwargs):
                    if "teams_cards" in name:
                        raise ImportError("No teams_cards")
                    return original_import(name, *args, **kwargs)

                with patch.object(builtins, "__import__", mock_import):
                    result = await _send_teams_error(sample_origin, "Error")

        assert result is False
