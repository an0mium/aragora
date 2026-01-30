"""Tests for Matrix/Element integration."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.integrations.matrix import (
    MatrixConfig,
    MatrixIntegration,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    return MatrixConfig(
        homeserver_url="https://matrix.org",
        access_token="syt_test_token",
        user_id="@aragora-bot:matrix.org",
        room_id="!abc123:matrix.org",
    )


@pytest.fixture
def integration(config):
    return MatrixIntegration(config)


@pytest.fixture
def unconfigured_integration():
    return MatrixIntegration(MatrixConfig())


def _make_debate_result(**kwargs):
    result = MagicMock()
    result.task = kwargs.get("task", "Design a rate limiter")
    result.final_answer = kwargs.get("final_answer", "Use token bucket algorithm")
    result.consensus_reached = kwargs.get("consensus_reached", True)
    result.confidence = kwargs.get("confidence", 0.85)
    result.rounds_used = kwargs.get("rounds_used", 3)
    result.winner = kwargs.get("winner", "claude")
    result.debate_id = kwargs.get("debate_id", "debate-abc123")
    result.participants = kwargs.get("participants", ["claude", "gpt4"])
    return result


# =============================================================================
# MatrixConfig Tests
# =============================================================================


class TestMatrixConfig:
    def test_default_values(self):
        cfg = MatrixConfig(
            homeserver_url="https://matrix.org",
            access_token="tok",
            room_id="!room:matrix.org",
        )
        assert cfg.notify_on_consensus is True
        assert cfg.notify_on_debate_end is True
        assert cfg.notify_on_error is True
        assert cfg.notify_on_leaderboard is False
        assert cfg.enable_commands is True
        assert cfg.use_html is True
        assert cfg.max_messages_per_minute == 10

    def test_env_fallback(self):
        with patch.dict(
            "os.environ",
            {
                "MATRIX_HOMESERVER_URL": "https://env.matrix.org",
                "MATRIX_ACCESS_TOKEN": "env_token",
                "MATRIX_USER_ID": "@bot:env.matrix.org",
                "MATRIX_ROOM_ID": "!env_room:matrix.org",
            },
        ):
            cfg = MatrixConfig()
            assert cfg.homeserver_url == "https://env.matrix.org"
            assert cfg.access_token == "env_token"
            assert cfg.user_id == "@bot:env.matrix.org"
            assert cfg.room_id == "!env_room:matrix.org"

    def test_strips_trailing_slash(self):
        cfg = MatrixConfig(
            homeserver_url="https://matrix.org/",
            access_token="tok",
            room_id="!r:m.org",
        )
        assert cfg.homeserver_url == "https://matrix.org"


# =============================================================================
# MatrixIntegration Tests
# =============================================================================


class TestMatrixIntegration:
    def test_initialization(self, integration):
        assert integration._session is None
        assert integration._message_count == 0
        assert integration._sync_token is None

    def test_is_configured(self, integration):
        assert integration.is_configured is True

    def test_not_configured(self, unconfigured_integration):
        assert unconfigured_integration.is_configured is False

    def test_api_url(self, integration):
        url = integration._api_url("/rooms/!abc123:matrix.org/send/m.room.message/txn1")
        assert url.startswith("https://matrix.org/_matrix/client/v3/")

    def test_check_rate_limit_allows(self, integration):
        assert integration._check_rate_limit() is True
        assert integration._message_count == 1

    def test_check_rate_limit_blocks(self, integration):
        integration._message_count = 10
        integration._last_reset = datetime.now()
        assert integration._check_rate_limit() is False

    def test_check_rate_limit_resets(self, integration):
        integration._message_count = 10
        integration._last_reset = datetime.now() - timedelta(seconds=61)
        assert integration._check_rate_limit() is True

    def test_get_headers(self, integration):
        headers = integration._get_headers()
        assert headers["Authorization"] == "Bearer syt_test_token"
        assert headers["Content-Type"] == "application/json"

    def test_escape_html(self, integration):
        assert (
            integration._escape_html('<b>"test"&</b>') == "&lt;b&gt;&quot;test&quot;&amp;&lt;/b&gt;"
        )

    @pytest.mark.asyncio
    async def test_send_message_not_configured(self, unconfigured_integration):
        result = await unconfigured_integration.send_message("Hello")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_message_rate_limited(self, integration):
        integration._message_count = 10
        integration._last_reset = datetime.now()
        result = await integration.send_message("Hello")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_message_text_only(self, integration):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_session = AsyncMock()
        mock_session.put = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=False),
            )
        )
        with patch.object(integration, "_get_session", return_value=mock_session):
            result = await integration.send_message("Hello plain text")
            assert result is True

    @pytest.mark.asyncio
    async def test_send_message_with_html(self, integration):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_session = AsyncMock()
        mock_session.put = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=False),
            )
        )
        with patch.object(integration, "_get_session", return_value=mock_session):
            result = await integration.send_message("Hello", html="<b>Hello</b>")
            assert result is True

    @pytest.mark.asyncio
    async def test_post_debate_summary_disabled(self, integration):
        integration.config.notify_on_debate_end = False
        result = _make_debate_result()
        success = await integration.post_debate_summary(result)
        assert success is False

    @pytest.mark.asyncio
    async def test_post_debate_summary(self, integration):
        with patch.object(integration, "send_message", new_callable=AsyncMock, return_value=True):
            result = _make_debate_result()
            success = await integration.post_debate_summary(result)
            assert success is True

    @pytest.mark.asyncio
    async def test_post_debate_summary_no_answer(self, integration):
        with patch.object(integration, "send_message", new_callable=AsyncMock, return_value=True):
            result = _make_debate_result(final_answer=None)
            success = await integration.post_debate_summary(result)
            assert success is True

    @pytest.mark.asyncio
    async def test_send_consensus_alert_disabled(self, integration):
        integration.config.notify_on_consensus = False
        result = await integration.send_consensus_alert("d-1", "Answer", 0.9)
        assert result is False

    @pytest.mark.asyncio
    async def test_send_consensus_alert_below_threshold(self, integration):
        result = await integration.send_consensus_alert("d-1", "Answer", 0.5)
        assert result is False

    @pytest.mark.asyncio
    async def test_send_consensus_alert(self, integration):
        with patch.object(integration, "send_message", new_callable=AsyncMock, return_value=True):
            result = await integration.send_consensus_alert(
                debate_id="d-1",
                answer="Token bucket is best",
                confidence=0.9,
                agents=["claude", "gpt4"],
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_send_error_alert_disabled(self, integration):
        integration.config.notify_on_error = False
        result = await integration.send_error_alert("d-1", "Error")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_error_alert(self, integration):
        with patch.object(integration, "send_message", new_callable=AsyncMock, return_value=True):
            result = await integration.send_error_alert(
                debate_id="d-1", error="Timeout", phase="proposal"
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_send_error_alert_no_phase(self, integration):
        with patch.object(integration, "send_message", new_callable=AsyncMock, return_value=True):
            result = await integration.send_error_alert(debate_id="d-1", error="Err")
            assert result is True

    @pytest.mark.asyncio
    async def test_send_leaderboard_update_disabled(self, integration):
        result = await integration.send_leaderboard_update([])
        assert result is False

    @pytest.mark.asyncio
    async def test_send_leaderboard_update(self, integration):
        integration.config.notify_on_leaderboard = True
        with patch.object(integration, "send_message", new_callable=AsyncMock, return_value=True):
            rankings = [
                {"name": "claude", "elo": 1800, "wins": 10, "losses": 2},
                {"name": "gpt4", "elo": 1750, "wins": 8, "losses": 4},
            ]
            result = await integration.send_leaderboard_update(rankings, domain="coding")
            assert result is True

    @pytest.mark.asyncio
    async def test_verify_connection_not_configured(self, unconfigured_integration):
        result = await unconfigured_integration.verify_connection()
        assert result is False

    @pytest.mark.asyncio
    async def test_join_room_not_configured(self, unconfigured_integration):
        result = await unconfigured_integration.join_room("!room:test")
        assert result is False

    @pytest.mark.asyncio
    async def test_join_room(self, integration):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=False),
            )
        )
        with patch.object(integration, "_get_session", return_value=mock_session):
            result = await integration.join_room("!room:matrix.org")
            assert result is True

    @pytest.mark.asyncio
    async def test_context_manager(self, config):
        async with MatrixIntegration(config) as matrix:
            assert isinstance(matrix, MatrixIntegration)

    @pytest.mark.asyncio
    async def test_close(self, integration):
        mock_session = AsyncMock()
        mock_session.closed = False
        integration._session = mock_session
        await integration.close()
        mock_session.close.assert_called_once()
