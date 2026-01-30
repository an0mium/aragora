"""Tests for Zoom integration."""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.integrations.zoom import (
    ZoomConfig,
    ZoomIntegration,
    ZoomMeetingInfo,
    ZoomWebhookEvent,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    return ZoomConfig(
        client_id="test_client_id",
        client_secret="test_client_secret",
        account_id="test_account_id",
        webhook_secret="test_webhook_secret",
        bot_jid="test_bot_jid@xmpp.zoom.us",
    )


@pytest.fixture
def integration(config):
    return ZoomIntegration(config)


@pytest.fixture
def unconfigured_integration():
    return ZoomIntegration(ZoomConfig())


def _make_debate_result(**kwargs):
    result = MagicMock()
    result.task = kwargs.get("task", "Design a rate limiter")
    result.final_answer = kwargs.get("final_answer", "Use token bucket")
    result.consensus_reached = kwargs.get("consensus_reached", True)
    result.confidence = kwargs.get("confidence", 0.85)
    result.rounds_used = kwargs.get("rounds_used", 3)
    result.winner = kwargs.get("winner", "claude")
    result.debate_id = kwargs.get("debate_id", "debate-abc123")
    result.participants = kwargs.get("participants", ["claude", "gpt4"])
    return result


# =============================================================================
# ZoomConfig Tests
# =============================================================================


class TestZoomConfig:
    def test_default_values(self):
        cfg = ZoomConfig()
        assert cfg.api_base_url == "https://api.zoom.us/v2"
        assert cfg.oauth_url == "https://zoom.us/oauth/token"
        assert cfg.max_requests_per_minute == 30
        assert cfg.max_requests_per_day == 1000

    def test_from_env(self):
        with patch.dict(
            "os.environ",
            {
                "ZOOM_CLIENT_ID": "env_id",
                "ZOOM_CLIENT_SECRET": "env_secret",
                "ZOOM_ACCOUNT_ID": "env_account",
            },
        ):
            cfg = ZoomConfig.from_env()
            assert cfg.client_id == "env_id"
            assert cfg.client_secret == "env_secret"
            assert cfg.account_id == "env_account"

    def test_is_configured(self, config):
        assert config.is_configured is True

    def test_is_not_configured(self):
        cfg = ZoomConfig()
        # When env vars are not set, should be empty
        with patch.dict("os.environ", {}, clear=True):
            cfg2 = ZoomConfig(client_id="", client_secret="", account_id="")
            assert cfg2.is_configured is False


# =============================================================================
# ZoomMeetingInfo Tests
# =============================================================================


class TestZoomMeetingInfo:
    def test_from_api_response(self):
        data = {
            "id": "12345678901",
            "topic": "Debate Discussion",
            "start_time": "2024-01-15T10:00:00Z",
            "duration": 60,
            "host_id": "host123",
            "join_url": "https://zoom.us/j/12345678901",
            "password": "abc123",
        }
        info = ZoomMeetingInfo.from_api_response(data)
        assert info.meeting_id == "12345678901"
        assert info.topic == "Debate Discussion"
        assert info.duration == 60
        assert info.host_id == "host123"
        assert info.join_url == "https://zoom.us/j/12345678901"

    def test_from_api_response_no_start_time(self):
        data = {"id": "123", "topic": "Test"}
        info = ZoomMeetingInfo.from_api_response(data)
        assert info.start_time is None

    def test_from_api_response_invalid_start_time(self):
        data = {"id": "123", "topic": "Test", "start_time": "invalid"}
        info = ZoomMeetingInfo.from_api_response(data)
        assert info.start_time is None


# =============================================================================
# ZoomWebhookEvent Tests
# =============================================================================


class TestZoomWebhookEvent:
    def test_from_request(self):
        data = {
            "event": "meeting.started",
            "payload": {
                "account_id": "acc123",
                "object": {"id": "123"},
            },
            "event_ts": 1700000000,
        }
        event = ZoomWebhookEvent.from_request(data)
        assert event.event_type == "meeting.started"
        assert event.account_id == "acc123"
        assert event.event_ts == 1700000000

    def test_from_request_empty(self):
        event = ZoomWebhookEvent.from_request({})
        assert event.event_type == ""
        assert event.payload == {}


# =============================================================================
# ZoomIntegration Tests
# =============================================================================


class TestZoomIntegration:
    def test_initialization(self, integration):
        assert integration._session is None
        assert integration._access_token is None

    def test_is_configured(self, integration):
        assert integration.is_configured is True

    def test_not_configured(self, unconfigured_integration):
        assert unconfigured_integration.is_configured is False

    def test_check_rate_limit_allows(self, integration):
        assert integration._check_rate_limit() is True
        assert integration._request_count_minute == 1

    def test_check_rate_limit_per_minute(self, integration):
        integration._request_count_minute = 30
        integration._last_minute_reset = datetime.now()
        assert integration._check_rate_limit() is False

    def test_check_rate_limit_per_day(self, integration):
        integration._request_count_day = 1000
        integration._last_day_reset = datetime.now()
        assert integration._check_rate_limit() is False

    def test_check_rate_limit_resets_minute(self, integration):
        integration._request_count_minute = 30
        integration._last_minute_reset = datetime.now() - timedelta(seconds=61)
        assert integration._check_rate_limit() is True

    def test_check_rate_limit_resets_day(self, integration):
        integration._request_count_day = 1000
        integration._last_day_reset = datetime.now() - timedelta(days=2)
        assert integration._check_rate_limit() is True

    def test_verify_webhook_valid(self, integration):
        body = b'{"event":"test"}'
        timestamp = "1234567890"
        message = f"v0:{timestamp}:{body.decode()}"
        expected = hmac.new(b"test_webhook_secret", message.encode(), hashlib.sha256).hexdigest()
        signature = f"v0={expected}"
        assert integration.verify_webhook(body, signature, timestamp) is True

    def test_verify_webhook_invalid(self, integration):
        result = integration.verify_webhook(b"body", "invalid_sig", "timestamp")
        assert result is False

    def test_verify_webhook_no_secret(self, unconfigured_integration):
        result = unconfigured_integration.verify_webhook(b"body", "sig", "ts")
        assert result is False

    @pytest.mark.asyncio
    async def test_handle_webhook_invalid_signature(self, integration):
        result = await integration.handle_webhook(b"body", "bad_sig", "ts")
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_webhook_valid(self, integration):
        body = json.dumps(
            {"event": "meeting.started", "payload": {"account_id": "a1"}, "event_ts": 123}
        ).encode()
        # Create valid signature
        timestamp = "123"
        message = f"v0:{timestamp}:{body.decode()}"
        expected = hmac.new(b"test_webhook_secret", message.encode(), hashlib.sha256).hexdigest()
        signature = f"v0={expected}"

        result = await integration.handle_webhook(body, signature, timestamp)
        assert result is not None
        assert result.event_type == "meeting.started"

    @pytest.mark.asyncio
    async def test_handle_webhook_invalid_json(self, integration):
        body = b"not json"
        timestamp = "123"
        message = f"v0:{timestamp}:{body.decode()}"
        expected = hmac.new(b"test_webhook_secret", message.encode(), hashlib.sha256).hexdigest()
        signature = f"v0={expected}"

        result = await integration.handle_webhook(body, signature, timestamp)
        assert result is None

    @pytest.mark.asyncio
    async def test_send_chat_message_no_bot_jid(self, unconfigured_integration):
        result = await unconfigured_integration.send_chat_message("jid", "msg")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_chat_message(self, integration):
        with patch.object(integration, "_api_request", new_callable=AsyncMock, return_value={}):
            result = await integration.send_chat_message("recipient@zoom", "Hello!")
            assert result is True

    @pytest.mark.asyncio
    async def test_send_chat_message_to_channel(self, integration):
        with patch.object(
            integration, "_api_request", new_callable=AsyncMock, return_value={}
        ) as mock_api:
            result = await integration.send_chat_message("channel123", "Hello!", is_channel=True)
            assert result is True
            call_data = mock_api.call_args[1]["data"]
            assert "to_channel" in call_data

    @pytest.mark.asyncio
    async def test_send_debate_summary_disabled(self, integration):
        integration.config.notify_on_debate_end = False
        result = _make_debate_result()
        success = await integration.send_debate_summary("jid", result)
        assert success is False

    @pytest.mark.asyncio
    async def test_send_debate_summary(self, integration):
        with patch.object(
            integration, "send_chat_message", new_callable=AsyncMock, return_value=True
        ):
            result = _make_debate_result()
            success = await integration.send_debate_summary("jid", result)
            assert success is True

    @pytest.mark.asyncio
    async def test_create_meeting(self, integration):
        mock_response = {
            "id": "123",
            "topic": "Debate",
            "join_url": "https://zoom.us/j/123",
        }
        with patch.object(
            integration, "_api_request", new_callable=AsyncMock, return_value=mock_response
        ):
            meeting = await integration.create_meeting("Debate Discussion", duration=30)
            assert meeting.topic == "Debate"
            assert meeting.meeting_id == "123"

    @pytest.mark.asyncio
    async def test_get_meeting(self, integration):
        mock_response = {"id": "456", "topic": "Test Meeting"}
        with patch.object(
            integration, "_api_request", new_callable=AsyncMock, return_value=mock_response
        ):
            meeting = await integration.get_meeting("456")
            assert meeting.meeting_id == "456"

    @pytest.mark.asyncio
    async def test_list_recordings(self, integration):
        mock_response = {"meetings": [{"id": "m1"}, {"id": "m2"}]}
        with patch.object(
            integration, "_api_request", new_callable=AsyncMock, return_value=mock_response
        ):
            recordings = await integration.list_recordings()
            assert len(recordings) == 2

    @pytest.mark.asyncio
    async def test_context_manager(self, config):
        async with ZoomIntegration(config) as zoom:
            assert isinstance(zoom, ZoomIntegration)

    @pytest.mark.asyncio
    async def test_close(self, integration):
        mock_session = AsyncMock()
        mock_session.closed = False
        integration._session = mock_session
        await integration.close()
        mock_session.close.assert_called_once()
