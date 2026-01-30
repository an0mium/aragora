"""Comprehensive tests for debate origin routing.

Tests cover:
1. Platform sender dispatch
2. Error handling when senders fail
3. Multi-session routing
4. Receipt and error posting
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.debate_origin import DebateOrigin
from aragora.server.debate_origin.router import (
    route_debate_result,
    post_receipt_to_channel,
    send_error_to_channel,
    route_result_to_all_sessions,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_result():
    """Create a sample debate result."""
    return {
        "consensus_reached": True,
        "final_answer": "The answer is 42.",
        "confidence": 0.85,
        "participants": ["claude", "gpt4", "gemini"],
        "task": "What is the meaning of life?",
    }


@pytest.fixture
def telegram_origin():
    """Create a Telegram origin."""
    return DebateOrigin(
        debate_id="telegram-debate-001",
        platform="telegram",
        channel_id="123456789",
        user_id="987654321",
        message_id="msg-100",
    )


@pytest.fixture
def slack_origin():
    """Create a Slack origin."""
    return DebateOrigin(
        debate_id="slack-debate-001",
        platform="slack",
        channel_id="C12345678",
        user_id="U87654321",
        thread_id="1234567890.123456",
    )


@pytest.fixture
def discord_origin():
    """Create a Discord origin."""
    return DebateOrigin(
        debate_id="discord-debate-001",
        platform="discord",
        channel_id="123456789012345678",
        user_id="876543210987654321",
        message_id="111222333444555666",
    )


@pytest.fixture
def teams_origin():
    """Create a Teams origin."""
    return DebateOrigin(
        debate_id="teams-debate-001",
        platform="teams",
        channel_id="teams-channel-123",
        user_id="teams-user-456",
        metadata={"webhook_url": "https://webhook.teams.com/abc"},
    )


@pytest.fixture
def sample_receipt():
    """Create a sample receipt."""
    receipt = MagicMock()
    receipt.verdict = "APPROVED"
    receipt.confidence = 0.92
    receipt.critical_count = 0
    receipt.high_count = 1
    return receipt


# =============================================================================
# Test: Route Debate Result
# =============================================================================


class TestRouteDebateResult:
    """Tests for route_debate_result function."""

    @pytest.mark.asyncio
    async def test_returns_false_when_no_origin(self, sample_result):
        """route_debate_result returns False when origin not found."""
        with patch(
            "aragora.server.debate_origin.router.get_debate_origin",
            return_value=None,
        ):
            result = await route_debate_result("nonexistent", sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_when_already_sent(self, sample_result):
        """route_debate_result returns True when result already sent."""
        origin = DebateOrigin(
            debate_id="already-sent",
            platform="slack",
            channel_id="C123",
            user_id="U456",
            result_sent=True,
        )

        with patch(
            "aragora.server.debate_origin.router.get_debate_origin",
            return_value=origin,
        ):
            result = await route_debate_result("already-sent", sample_result)

        assert result is True

    @pytest.mark.asyncio
    async def test_routes_to_telegram(self, telegram_origin, sample_result):
        """route_debate_result calls telegram sender."""
        with patch(
            "aragora.server.debate_origin.router.get_debate_origin",
            return_value=telegram_origin,
        ):
            with patch(
                "aragora.server.debate_origin.router.USE_DOCK_ROUTING",
                False,
            ):
                with patch(
                    "aragora.server.debate_origin.router._send_telegram_result",
                    new_callable=AsyncMock,
                    return_value=True,
                ) as mock_send:
                    with patch(
                        "aragora.server.debate_origin.router.mark_result_sent",
                    ):
                        result = await route_debate_result(telegram_origin.debate_id, sample_result)

        assert result is True
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_to_slack(self, slack_origin, sample_result):
        """route_debate_result calls slack sender."""
        with patch(
            "aragora.server.debate_origin.router.get_debate_origin",
            return_value=slack_origin,
        ):
            with patch(
                "aragora.server.debate_origin.router.USE_DOCK_ROUTING",
                False,
            ):
                with patch(
                    "aragora.server.debate_origin.router._send_slack_result",
                    new_callable=AsyncMock,
                    return_value=True,
                ) as mock_send:
                    with patch(
                        "aragora.server.debate_origin.router.mark_result_sent",
                    ):
                        result = await route_debate_result(slack_origin.debate_id, sample_result)

        assert result is True
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_to_discord(self, discord_origin, sample_result):
        """route_debate_result calls discord sender."""
        with patch(
            "aragora.server.debate_origin.router.get_debate_origin",
            return_value=discord_origin,
        ):
            with patch(
                "aragora.server.debate_origin.router.USE_DOCK_ROUTING",
                False,
            ):
                with patch(
                    "aragora.server.debate_origin.router._send_discord_result",
                    new_callable=AsyncMock,
                    return_value=True,
                ) as mock_send:
                    with patch(
                        "aragora.server.debate_origin.router.mark_result_sent",
                    ):
                        result = await route_debate_result(discord_origin.debate_id, sample_result)

        assert result is True
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_to_teams(self, teams_origin, sample_result):
        """route_debate_result calls teams sender."""
        with patch(
            "aragora.server.debate_origin.router.get_debate_origin",
            return_value=teams_origin,
        ):
            with patch(
                "aragora.server.debate_origin.router.USE_DOCK_ROUTING",
                False,
            ):
                with patch(
                    "aragora.server.debate_origin.router._send_teams_result",
                    new_callable=AsyncMock,
                    return_value=True,
                ) as mock_send:
                    with patch(
                        "aragora.server.debate_origin.router.mark_result_sent",
                    ):
                        result = await route_debate_result(teams_origin.debate_id, sample_result)

        assert result is True
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_to_whatsapp(self, sample_result):
        """route_debate_result calls whatsapp sender."""
        origin = DebateOrigin(
            debate_id="wa-debate",
            platform="whatsapp",
            channel_id="+1234567890",
            user_id="wa-user",
        )

        with patch(
            "aragora.server.debate_origin.router.get_debate_origin",
            return_value=origin,
        ):
            with patch(
                "aragora.server.debate_origin.router.USE_DOCK_ROUTING",
                False,
            ):
                with patch(
                    "aragora.server.debate_origin.router._send_whatsapp_result",
                    new_callable=AsyncMock,
                    return_value=True,
                ) as mock_send:
                    with patch(
                        "aragora.server.debate_origin.router.mark_result_sent",
                    ):
                        result = await route_debate_result(origin.debate_id, sample_result)

        assert result is True
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_to_email(self, sample_result):
        """route_debate_result calls email sender."""
        origin = DebateOrigin(
            debate_id="email-debate",
            platform="email",
            channel_id="inbox",
            user_id="user@example.com",
        )

        with patch(
            "aragora.server.debate_origin.router.get_debate_origin",
            return_value=origin,
        ):
            with patch(
                "aragora.server.debate_origin.router.USE_DOCK_ROUTING",
                False,
            ):
                with patch(
                    "aragora.server.debate_origin.router._send_email_result",
                    new_callable=AsyncMock,
                    return_value=True,
                ) as mock_send:
                    with patch(
                        "aragora.server.debate_origin.router.mark_result_sent",
                    ):
                        result = await route_debate_result(origin.debate_id, sample_result)

        assert result is True
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_to_google_chat(self, sample_result):
        """route_debate_result calls google chat sender."""
        origin = DebateOrigin(
            debate_id="gchat-debate",
            platform="google_chat",
            channel_id="spaces/abc123",
            user_id="gchat-user",
        )

        with patch(
            "aragora.server.debate_origin.router.get_debate_origin",
            return_value=origin,
        ):
            with patch(
                "aragora.server.debate_origin.router.USE_DOCK_ROUTING",
                False,
            ):
                with patch(
                    "aragora.server.debate_origin.router._send_google_chat_result",
                    new_callable=AsyncMock,
                    return_value=True,
                ) as mock_send:
                    with patch(
                        "aragora.server.debate_origin.router.mark_result_sent",
                    ):
                        result = await route_debate_result(origin.debate_id, sample_result)

        assert result is True
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_false_for_unknown_platform(self, sample_result):
        """route_debate_result returns False for unknown platform."""
        origin = DebateOrigin(
            debate_id="unknown-platform",
            platform="foobar_chat",
            channel_id="123",
            user_id="456",
        )

        with patch(
            "aragora.server.debate_origin.router.get_debate_origin",
            return_value=origin,
        ):
            with patch(
                "aragora.server.debate_origin.router.USE_DOCK_ROUTING",
                False,
            ):
                result = await route_debate_result(origin.debate_id, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_marks_result_sent_on_success(self, telegram_origin, sample_result):
        """route_debate_result marks result as sent on success."""
        with patch(
            "aragora.server.debate_origin.router.get_debate_origin",
            return_value=telegram_origin,
        ):
            with patch(
                "aragora.server.debate_origin.router.USE_DOCK_ROUTING",
                False,
            ):
                with patch(
                    "aragora.server.debate_origin.router._send_telegram_result",
                    new_callable=AsyncMock,
                    return_value=True,
                ):
                    with patch(
                        "aragora.server.debate_origin.router.mark_result_sent",
                    ) as mock_mark:
                        await route_debate_result(telegram_origin.debate_id, sample_result)

        mock_mark.assert_called_once_with(telegram_origin.debate_id)

    @pytest.mark.asyncio
    async def test_does_not_mark_sent_on_failure(self, telegram_origin, sample_result):
        """route_debate_result does not mark sent on failure."""
        with patch(
            "aragora.server.debate_origin.router.get_debate_origin",
            return_value=telegram_origin,
        ):
            with patch(
                "aragora.server.debate_origin.router.USE_DOCK_ROUTING",
                False,
            ):
                with patch(
                    "aragora.server.debate_origin.router._send_telegram_result",
                    new_callable=AsyncMock,
                    return_value=False,
                ):
                    with patch(
                        "aragora.server.debate_origin.router.mark_result_sent",
                    ) as mock_mark:
                        await route_debate_result(telegram_origin.debate_id, sample_result)

        mock_mark.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_sender_exception(self, telegram_origin, sample_result):
        """route_debate_result handles sender exceptions."""
        with patch(
            "aragora.server.debate_origin.router.get_debate_origin",
            return_value=telegram_origin,
        ):
            with patch(
                "aragora.server.debate_origin.router.USE_DOCK_ROUTING",
                False,
            ):
                with patch(
                    "aragora.server.debate_origin.router._send_telegram_result",
                    new_callable=AsyncMock,
                    side_effect=OSError("Network error"),
                ):
                    result = await route_debate_result(telegram_origin.debate_id, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_includes_voice_when_requested(self, telegram_origin, sample_result):
        """route_debate_result sends voice when include_voice=True."""
        with patch(
            "aragora.server.debate_origin.router.get_debate_origin",
            return_value=telegram_origin,
        ):
            with patch(
                "aragora.server.debate_origin.router.USE_DOCK_ROUTING",
                False,
            ):
                with patch(
                    "aragora.server.debate_origin.router._send_telegram_result",
                    new_callable=AsyncMock,
                    return_value=True,
                ):
                    with patch(
                        "aragora.server.debate_origin.router._send_telegram_voice",
                        new_callable=AsyncMock,
                        return_value=True,
                    ) as mock_voice:
                        with patch(
                            "aragora.server.debate_origin.router.mark_result_sent",
                        ):
                            await route_debate_result(
                                telegram_origin.debate_id,
                                sample_result,
                                include_voice=True,
                            )

        mock_voice.assert_called_once()

    @pytest.mark.asyncio
    async def test_posts_receipt_when_provided(
        self, telegram_origin, sample_result, sample_receipt
    ):
        """route_debate_result posts receipt when provided."""
        with patch(
            "aragora.server.debate_origin.router.get_debate_origin",
            return_value=telegram_origin,
        ):
            with patch(
                "aragora.server.debate_origin.router.USE_DOCK_ROUTING",
                False,
            ):
                with patch(
                    "aragora.server.debate_origin.router._send_telegram_result",
                    new_callable=AsyncMock,
                    return_value=True,
                ):
                    with patch(
                        "aragora.server.debate_origin.router.mark_result_sent",
                    ):
                        with patch(
                            "aragora.server.debate_origin.router.post_receipt_to_channel",
                            new_callable=AsyncMock,
                        ) as mock_receipt:
                            await route_debate_result(
                                telegram_origin.debate_id,
                                sample_result,
                                receipt=sample_receipt,
                                receipt_url="https://example.com/receipt/123",
                            )

        mock_receipt.assert_called_once()


# =============================================================================
# Test: Post Receipt to Channel
# =============================================================================


class TestPostReceiptToChannel:
    """Tests for post_receipt_to_channel function."""

    @pytest.mark.asyncio
    async def test_posts_to_slack(self, slack_origin, sample_receipt):
        """post_receipt_to_channel calls slack receipt sender."""
        with patch(
            "aragora.server.debate_origin.router.USE_DOCK_ROUTING",
            False,
        ):
            with patch(
                "aragora.server.debate_origin.router._send_slack_receipt",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_send:
                result = await post_receipt_to_channel(
                    slack_origin, sample_receipt, "https://example.com/r"
                )

        assert result is True
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_posts_to_teams(self, teams_origin, sample_receipt):
        """post_receipt_to_channel calls teams receipt sender."""
        with patch(
            "aragora.server.debate_origin.router.USE_DOCK_ROUTING",
            False,
        ):
            with patch(
                "aragora.server.debate_origin.router._send_teams_receipt",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_send:
                result = await post_receipt_to_channel(
                    teams_origin, sample_receipt, "https://example.com/r"
                )

        assert result is True
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_posts_to_telegram(self, telegram_origin, sample_receipt):
        """post_receipt_to_channel calls telegram receipt sender."""
        with patch(
            "aragora.server.debate_origin.router.USE_DOCK_ROUTING",
            False,
        ):
            with patch(
                "aragora.server.debate_origin.router._send_telegram_receipt",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_send:
                result = await post_receipt_to_channel(
                    telegram_origin, sample_receipt, "https://example.com/r"
                )

        assert result is True
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_posts_to_discord(self, discord_origin, sample_receipt):
        """post_receipt_to_channel calls discord receipt sender."""
        with patch(
            "aragora.server.debate_origin.router.USE_DOCK_ROUTING",
            False,
        ):
            with patch(
                "aragora.server.debate_origin.router._send_discord_receipt",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_send:
                result = await post_receipt_to_channel(
                    discord_origin, sample_receipt, "https://example.com/r"
                )

        assert result is True
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_posts_to_google_chat(self, sample_receipt):
        """post_receipt_to_channel calls google chat receipt sender."""
        origin = DebateOrigin(
            debate_id="gchat-receipt",
            platform="google_chat",
            channel_id="spaces/xyz",
            user_id="gchat-user",
        )

        with patch(
            "aragora.server.debate_origin.router.USE_DOCK_ROUTING",
            False,
        ):
            with patch(
                "aragora.server.debate_origin.router._send_google_chat_receipt",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_send:
                result = await post_receipt_to_channel(
                    origin, sample_receipt, "https://example.com/r"
                )

        assert result is True
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_false_for_unsupported_platform(self, sample_receipt):
        """post_receipt_to_channel returns False for unsupported platform."""
        origin = DebateOrigin(
            debate_id="unsupported",
            platform="sms",
            channel_id="+1234567890",
            user_id="sms-user",
        )

        with patch(
            "aragora.server.debate_origin.router.USE_DOCK_ROUTING",
            False,
        ):
            result = await post_receipt_to_channel(origin, sample_receipt, "https://example.com/r")

        assert result is False

    @pytest.mark.asyncio
    async def test_handles_sender_error(self, slack_origin, sample_receipt):
        """post_receipt_to_channel handles sender errors."""
        with patch(
            "aragora.server.debate_origin.router.USE_DOCK_ROUTING",
            False,
        ):
            with patch(
                "aragora.server.debate_origin.router._send_slack_receipt",
                new_callable=AsyncMock,
                side_effect=OSError("Network error"),
            ):
                result = await post_receipt_to_channel(
                    slack_origin, sample_receipt, "https://example.com/r"
                )

        assert result is False


# =============================================================================
# Test: Send Error to Channel
# =============================================================================


class TestSendErrorToChannel:
    """Tests for send_error_to_channel function."""

    @pytest.mark.asyncio
    async def test_sends_to_slack(self, slack_origin):
        """send_error_to_channel calls slack error sender."""
        with patch(
            "aragora.server.debate_origin.router.USE_DOCK_ROUTING",
            False,
        ):
            with patch(
                "aragora.server.debate_origin.router._send_slack_error",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_send:
                result = await send_error_to_channel(
                    slack_origin, "Rate limit exceeded", "debate-123"
                )

        assert result is True
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_sends_to_teams(self, teams_origin):
        """send_error_to_channel calls teams error sender."""
        with patch(
            "aragora.server.debate_origin.router.USE_DOCK_ROUTING",
            False,
        ):
            with patch(
                "aragora.server.debate_origin.router._send_teams_error",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_send:
                result = await send_error_to_channel(teams_origin, "Timeout error", "debate-456")

        assert result is True
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_sends_to_telegram(self, telegram_origin):
        """send_error_to_channel calls telegram error sender."""
        with patch(
            "aragora.server.debate_origin.router.USE_DOCK_ROUTING",
            False,
        ):
            with patch(
                "aragora.server.debate_origin.router._send_telegram_error",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_send:
                result = await send_error_to_channel(telegram_origin, "Server error", "debate-789")

        assert result is True
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_sends_to_discord(self, discord_origin):
        """send_error_to_channel calls discord error sender."""
        with patch(
            "aragora.server.debate_origin.router.USE_DOCK_ROUTING",
            False,
        ):
            with patch(
                "aragora.server.debate_origin.router._send_discord_error",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_send:
                result = await send_error_to_channel(
                    discord_origin, "Budget exceeded", "debate-abc"
                )

        assert result is True
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_false_for_unsupported_platform(self):
        """send_error_to_channel returns False for unsupported platform."""
        origin = DebateOrigin(
            debate_id="unsupported-error",
            platform="fax",
            channel_id="+1234567890",
            user_id="fax-user",
        )

        with patch(
            "aragora.server.debate_origin.router.USE_DOCK_ROUTING",
            False,
        ):
            result = await send_error_to_channel(origin, "Some error", "debate-fax")

        assert result is False

    @pytest.mark.asyncio
    async def test_handles_sender_error(self, slack_origin):
        """send_error_to_channel handles sender errors."""
        with patch(
            "aragora.server.debate_origin.router.USE_DOCK_ROUTING",
            False,
        ):
            with patch(
                "aragora.server.debate_origin.router._send_slack_error",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Send failed"),
            ):
                result = await send_error_to_channel(slack_origin, "Error msg", "debate-err")

        assert result is False


# =============================================================================
# Test: Route Result to All Sessions
# =============================================================================


class TestRouteResultToAllSessions:
    """Tests for route_result_to_all_sessions function."""

    @pytest.mark.asyncio
    async def test_routes_to_primary_origin(self, sample_result):
        """route_result_to_all_sessions routes to primary origin."""
        with patch(
            "aragora.server.debate_origin.router.route_debate_result",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_route:
            with patch(
                "aragora.server.debate_origin.router.get_sessions_for_debate",
                new_callable=AsyncMock,
                return_value=[],
            ):
                with patch(
                    "aragora.server.debate_origin.router.get_debate_origin",
                    return_value=None,
                ):
                    count = await route_result_to_all_sessions("debate-123", sample_result)

        assert count == 1
        mock_route.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_to_additional_sessions(self, slack_origin, sample_result):
        """route_result_to_all_sessions routes to additional sessions."""
        additional_session = MagicMock()
        additional_session.session_id = "additional-session"
        additional_session.channel = "telegram"
        additional_session.user_id = "tg-user"
        additional_session.context = {"channel_id": "tg-123"}

        with patch(
            "aragora.server.debate_origin.router.route_debate_result",
            new_callable=AsyncMock,
            return_value=True,
        ):
            with patch(
                "aragora.server.debate_origin.router.get_sessions_for_debate",
                new_callable=AsyncMock,
                return_value=[additional_session],
            ):
                with patch(
                    "aragora.server.debate_origin.router.get_debate_origin",
                    return_value=slack_origin,
                ):
                    with patch(
                        "aragora.server.debate_origin.router._send_telegram_result",
                        new_callable=AsyncMock,
                        return_value=True,
                    ) as mock_tg_send:
                        count = await route_result_to_all_sessions(
                            slack_origin.debate_id, sample_result
                        )

        assert count == 2
        mock_tg_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_duplicate_session(self, telegram_origin, sample_result):
        """route_result_to_all_sessions skips duplicate session."""
        telegram_origin.session_id = "primary-session"

        same_session = MagicMock()
        same_session.session_id = "primary-session"  # Same as origin
        same_session.channel = "telegram"
        same_session.user_id = telegram_origin.user_id
        same_session.context = {"channel_id": telegram_origin.channel_id}

        with patch(
            "aragora.server.debate_origin.router.route_debate_result",
            new_callable=AsyncMock,
            return_value=True,
        ):
            with patch(
                "aragora.server.debate_origin.router.get_sessions_for_debate",
                new_callable=AsyncMock,
                return_value=[same_session],
            ):
                with patch(
                    "aragora.server.debate_origin.router.get_debate_origin",
                    return_value=telegram_origin,
                ):
                    count = await route_result_to_all_sessions(
                        telegram_origin.debate_id, sample_result
                    )

        # Only 1 - the primary, not the duplicate
        assert count == 1

    @pytest.mark.asyncio
    async def test_handles_session_send_failure(self, slack_origin, sample_result):
        """route_result_to_all_sessions handles session send failures."""
        failing_session = MagicMock()
        failing_session.session_id = "failing-session"
        failing_session.channel = "discord"
        failing_session.user_id = "discord-user"
        failing_session.context = {"channel_id": "discord-123"}

        with patch(
            "aragora.server.debate_origin.router.route_debate_result",
            new_callable=AsyncMock,
            return_value=True,
        ):
            with patch(
                "aragora.server.debate_origin.router.get_sessions_for_debate",
                new_callable=AsyncMock,
                return_value=[failing_session],
            ):
                with patch(
                    "aragora.server.debate_origin.router.get_debate_origin",
                    return_value=slack_origin,
                ):
                    with patch(
                        "aragora.server.debate_origin.router._send_discord_result",
                        new_callable=AsyncMock,
                        side_effect=OSError("Network error"),
                    ):
                        count = await route_result_to_all_sessions(
                            slack_origin.debate_id, sample_result
                        )

        # Only primary succeeded
        assert count == 1

    @pytest.mark.asyncio
    async def test_handles_session_lookup_failure(self, sample_result):
        """route_result_to_all_sessions handles session lookup failures."""
        with patch(
            "aragora.server.debate_origin.router.route_debate_result",
            new_callable=AsyncMock,
            return_value=True,
        ):
            with patch(
                "aragora.server.debate_origin.router.get_sessions_for_debate",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Session manager unavailable"),
            ):
                with patch(
                    "aragora.server.debate_origin.router.get_debate_origin",
                    return_value=None,
                ):
                    count = await route_result_to_all_sessions("debate-lookup-fail", sample_result)

        # Primary succeeded, session lookup failed gracefully
        assert count == 1

    @pytest.mark.asyncio
    async def test_routes_to_multiple_platforms(self, slack_origin, sample_result):
        """route_result_to_all_sessions routes to multiple different platforms."""
        telegram_session = MagicMock()
        telegram_session.session_id = "tg-session"
        telegram_session.channel = "telegram"
        telegram_session.user_id = "tg-user"
        telegram_session.context = {"channel_id": "tg-123"}

        discord_session = MagicMock()
        discord_session.session_id = "discord-session"
        discord_session.channel = "discord"
        discord_session.user_id = "discord-user"
        discord_session.context = {"channel_id": "discord-123"}

        with patch(
            "aragora.server.debate_origin.router.route_debate_result",
            new_callable=AsyncMock,
            return_value=True,
        ):
            with patch(
                "aragora.server.debate_origin.router.get_sessions_for_debate",
                new_callable=AsyncMock,
                return_value=[telegram_session, discord_session],
            ):
                with patch(
                    "aragora.server.debate_origin.router.get_debate_origin",
                    return_value=slack_origin,
                ):
                    with patch(
                        "aragora.server.debate_origin.router._send_telegram_result",
                        new_callable=AsyncMock,
                        return_value=True,
                    ):
                        with patch(
                            "aragora.server.debate_origin.router._send_discord_result",
                            new_callable=AsyncMock,
                            return_value=True,
                        ):
                            count = await route_result_to_all_sessions(
                                slack_origin.debate_id, sample_result
                            )

        # Primary + telegram + discord
        assert count == 3
