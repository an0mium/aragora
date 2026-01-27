"""
Tests for Channel Router.

Tests the high-level dock-based message routing for debate results,
receipts, and error messages.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from aragora.channels.router import ChannelRouter, get_channel_router
from aragora.channels.dock import ChannelCapability, SendResult


# =============================================================================
# ChannelRouter Tests
# =============================================================================


class TestChannelRouter:
    """Tests for ChannelRouter."""

    def test_platform_normalization(self):
        """Test platform name normalization."""
        router = ChannelRouter()

        assert router._normalize_platform("gchat") == "google_chat"
        assert router._normalize_platform("SLACK") == "slack"
        assert router._normalize_platform(" Teams ") == "teams"
        assert router._normalize_platform("msteams") == "teams"

    def test_get_channel_router_singleton(self):
        """Test that get_channel_router returns singleton."""
        router1 = get_channel_router()
        router2 = get_channel_router()
        assert router1 is router2

    @pytest.mark.asyncio
    async def test_route_result_unknown_platform(self):
        """Test routing to unknown platform fails gracefully."""
        router = ChannelRouter()

        result = await router.route_result(
            platform="unknown_platform",
            channel_id="123",
            result={"consensus_reached": True, "final_answer": "Test"},
        )

        assert result.success is False
        assert "not available" in result.error

    @pytest.mark.asyncio
    async def test_route_result_with_mock_dock(self):
        """Test routing result through mocked dock."""
        from aragora.channels.registry import DockRegistry
        from aragora.channels.dock import ChannelDock

        # Create mock dock
        mock_dock = MagicMock(spec=ChannelDock)
        mock_dock.is_initialized = True
        mock_dock.send_result = AsyncMock(
            return_value=SendResult.ok(
                platform="test",
                channel_id="123",
                message_id="msg-1",
            )
        )
        mock_dock.supports.return_value = False

        # Create registry with mock dock
        registry = DockRegistry()
        registry._dock_classes["test"] = lambda config: mock_dock
        registry._dock_instances["test"] = mock_dock

        router = ChannelRouter(registry=registry)

        result = await router.route_result(
            platform="test",
            channel_id="123",
            result={"consensus_reached": True, "final_answer": "Answer"},
        )

        assert result.success is True
        mock_dock.send_result.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_error_with_mock_dock(self):
        """Test routing error through mocked dock."""
        from aragora.channels.registry import DockRegistry
        from aragora.channels.dock import ChannelDock

        mock_dock = MagicMock(spec=ChannelDock)
        mock_dock.is_initialized = True
        mock_dock.send_error = AsyncMock(
            return_value=SendResult.ok(
                platform="test",
                channel_id="123",
            )
        )

        registry = DockRegistry()
        registry._dock_instances["test"] = mock_dock

        router = ChannelRouter(registry=registry)

        result = await router.route_error(
            platform="test",
            channel_id="123",
            error_message="Something went wrong",
            debate_id="debate-1",
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_route_voice_unsupported(self):
        """Test voice routing to dock without voice support."""
        from aragora.channels.registry import DockRegistry
        from aragora.channels.dock import ChannelDock

        mock_dock = MagicMock(spec=ChannelDock)
        mock_dock.is_initialized = True
        mock_dock.supports.return_value = False  # No voice support

        registry = DockRegistry()
        registry._dock_instances["test"] = mock_dock

        router = ChannelRouter(registry=registry)

        result = await router.route_voice(
            platform="test",
            channel_id="123",
            audio_data=b"audio",
        )

        assert result.success is False
        assert "voice" in result.error.lower()


# =============================================================================
# Result Message Building Tests
# =============================================================================


class TestResultMessageBuilding:
    """Tests for result message formatting."""

    def test_build_result_message_with_consensus(self):
        """Test building result message with consensus."""
        router = ChannelRouter()

        result = {
            "consensus_reached": True,
            "confidence": 0.85,
            "final_answer": "The answer is yes.",
            "task": "Test question",
        }

        message = router._build_result_message(result)

        assert "Consensus Reached" in message.content
        assert "85%" in message.content
        assert "The answer is yes." in message.content

    def test_build_result_message_without_consensus(self):
        """Test building result message without consensus."""
        router = ChannelRouter()

        result = {
            "consensus_reached": False,
            "confidence": 0.45,
            "final_answer": "Unable to reach agreement.",
        }

        message = router._build_result_message(result)

        assert "No Consensus" in message.content
        assert "❌" in message.content

    def test_build_result_message_truncates_long_answers(self):
        """Test that long answers are truncated."""
        router = ChannelRouter()

        result = {
            "consensus_reached": True,
            "confidence": 0.9,
            "final_answer": "x" * 3000,
        }

        message = router._build_result_message(result)

        assert len(message.content) < 2500
        assert "..." in message.content


# =============================================================================
# Receipt Message Building Tests
# =============================================================================


class TestReceiptMessageBuilding:
    """Tests for receipt message formatting."""

    def test_format_receipt_summary_approved(self):
        """Test receipt summary for approved verdict."""
        router = ChannelRouter()

        class MockReceipt:
            verdict = "APPROVED"
            confidence = 0.9
            critical_count = 0
            high_count = 2
            cost_usd = 0.05
            budget_limit_usd = 1.0

        receipt = MockReceipt()
        summary = router._format_receipt_summary(receipt, "https://example.com/receipt")

        assert "✅" in summary
        assert "APPROVED" in summary
        assert "90%" in summary
        assert "0 critical" in summary
        assert "$0.0500" in summary
        assert "5%" in summary  # Budget percentage

    def test_format_receipt_summary_rejected(self):
        """Test receipt summary for rejected verdict."""
        router = ChannelRouter()

        class MockReceipt:
            verdict = "REJECTED"
            confidence = 0.7
            critical_count = 3
            high_count = 5

        receipt = MockReceipt()
        summary = router._format_receipt_summary(receipt, "https://example.com")

        assert "❌" in summary
        assert "REJECTED" in summary
        assert "3 critical" in summary


# =============================================================================
# Error Formatting Tests
# =============================================================================


class TestErrorFormatting:
    """Tests for error message formatting."""

    def test_format_known_error(self):
        """Test formatting a known error pattern."""
        router = ChannelRouter()

        friendly = router._format_error_for_chat("Rate limit exceeded", "debate-123")

        assert "processed" in friendly.lower() or "shortly" in friendly.lower()
        assert "debate-123" in friendly

    def test_format_unknown_error(self):
        """Test formatting an unknown error."""
        router = ChannelRouter()

        friendly = router._format_error_for_chat("Some unknown error xyz", "debate-456")

        assert "issue" in friendly.lower() or "try again" in friendly.lower()
        assert "debate-456" in friendly

    def test_format_error_without_debate_id(self):
        """Test formatting error without debate ID."""
        router = ChannelRouter()

        friendly = router._format_error_for_chat("timeout", None)

        assert "delay" in friendly.lower() or "wait" in friendly.lower()
        assert "Debate ID" not in friendly


# =============================================================================
# Integration Tests
# =============================================================================


class TestRouterIntegration:
    """Integration tests for router with real docks."""

    @pytest.mark.asyncio
    async def test_router_with_telegram_dock(self):
        """Test router integration with Telegram dock."""
        from aragora.channels.registry import DockRegistry
        from aragora.channels.docks.telegram import TelegramDock

        registry = DockRegistry()
        registry.register(TelegramDock)

        router = ChannelRouter(registry=registry)

        # Without token, dock should fail to initialize
        with patch.dict("os.environ", {}, clear=True):
            result = await router.route_result(
                platform="telegram",
                channel_id="123456",
                result={"consensus_reached": True, "final_answer": "Test"},
            )

            # Will fail because no token configured
            assert result.success is False

    @pytest.mark.asyncio
    async def test_router_with_teams_dock(self):
        """Test router integration with Teams dock."""
        from aragora.channels.registry import DockRegistry
        from aragora.channels.docks.teams import TeamsDock

        registry = DockRegistry()
        registry.register(TeamsDock)

        router = ChannelRouter(registry=registry)

        # Teams requires webhook URL
        result = await router.route_result(
            platform="teams",
            channel_id="channel123",
            result={"consensus_reached": True, "final_answer": "Test"},
            # No webhook_url provided
        )

        assert result.success is False
        assert "webhook" in result.error.lower()

    @pytest.mark.asyncio
    async def test_router_capability_check(self):
        """Test router checks capabilities before voice send."""
        from aragora.channels.registry import DockRegistry
        from aragora.channels.docks.slack import SlackDock

        registry = DockRegistry()
        registry.register(SlackDock)

        router = ChannelRouter(registry=registry)

        # Slack doesn't support voice
        with patch.dict("os.environ", {"SLACK_BOT_TOKEN": "test-token"}):
            result = await router.route_voice(
                platform="slack",
                channel_id="C123",
                audio_data=b"audio",
            )

            assert result.success is False
            assert "voice" in result.error.lower()
