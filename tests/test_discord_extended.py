"""Extended tests for Discord integration.

Focuses on gaps in existing coverage:
- Rate limiting logic (30/min)
- Retry logic with exponential backoff
- HTTP error handling (4xx, 5xx)
- Field truncation limits
- Session management
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.integrations.discord import (
    DiscordConfig,
    DiscordEmbed,
    DiscordIntegration,
    DiscordWebhookManager,
    create_discord_integration,
    discord_manager,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def discord_config():
    """Create a basic Discord config."""
    return DiscordConfig(
        webhook_url="https://discord.com/api/webhooks/123/abc",
        username="Test Bot",
        rate_limit_per_minute=30,
        retry_count=3,
        retry_delay=0.1,  # Fast for tests
    )


@pytest.fixture
def discord_integration(discord_config):
    """Create Discord integration with config."""
    return DiscordIntegration(discord_config)


@pytest.fixture
def mock_response():
    """Factory for mock aiohttp responses."""
    def _make(status=204, headers=None, text="", json_data=None):
        response = AsyncMock()
        response.status = status
        response.headers = headers or {}
        response.text = AsyncMock(return_value=text)
        if json_data:
            response.json = AsyncMock(return_value=json_data)
        return response
    return _make


class MockContextManager:
    """Helper for creating async context manager mocks."""

    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


@pytest.fixture
def mock_session(mock_response):
    """Create a mock aiohttp session."""
    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()
    # Default to successful response
    session.post = MagicMock(return_value=MockContextManager(mock_response(204)))
    return session


def create_response_sequence(responses):
    """Create a function that returns responses in sequence."""
    index = [0]

    def get_next(*args, **kwargs):
        response = responses[min(index[0], len(responses) - 1)]
        index[0] += 1
        return MockContextManager(response)

    return get_next


# =============================================================================
# Rate Limiting Tests
# =============================================================================


class TestDiscordRateLimiting:
    """Tests for rate limiting logic."""

    @pytest.mark.asyncio
    async def test_rate_limit_check_allows_first_request(self, discord_integration):
        """First request should be allowed immediately."""
        # Should not raise or sleep
        await discord_integration._check_rate_limit()
        assert len(discord_integration._request_times) == 1

    @pytest.mark.asyncio
    async def test_rate_limit_tracks_request_times(self, discord_integration):
        """Request times are tracked."""
        for _ in range(5):
            await discord_integration._check_rate_limit()

        assert len(discord_integration._request_times) == 5

    @pytest.mark.asyncio
    async def test_rate_limit_cleanup_old_requests(self, discord_integration):
        """Requests older than 60 seconds are cleaned up."""
        # Add old timestamps manually
        loop = asyncio.get_event_loop()
        old_time = loop.time() - 70  # 70 seconds ago
        discord_integration._request_times = [old_time] * 10

        await discord_integration._check_rate_limit()

        # Old requests should be removed, only new one remains
        assert len(discord_integration._request_times) == 1

    @pytest.mark.asyncio
    async def test_rate_limit_waits_when_exceeded(self, discord_config):
        """Rate limit waits when limit is exceeded."""
        discord_config.rate_limit_per_minute = 2  # Low limit for test
        integration = DiscordIntegration(discord_config)

        # Fill up rate limit
        loop = asyncio.get_event_loop()
        now = loop.time()
        integration._request_times = [now, now]  # At limit

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await integration._check_rate_limit()
            # Should have waited
            mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_limit_window_is_60_seconds(self, discord_config):
        """Rate limit window is exactly 60 seconds."""
        discord_config.rate_limit_per_minute = 2
        integration = DiscordIntegration(discord_config)

        loop = asyncio.get_event_loop()
        # Request 59 seconds ago (should still count)
        integration._request_times = [loop.time() - 59, loop.time() - 59]

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await integration._check_rate_limit()
            mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_limit_respects_config_value(self, discord_config):
        """Rate limit uses configured value."""
        discord_config.rate_limit_per_minute = 5
        integration = DiscordIntegration(discord_config)

        # Add 4 requests (under limit)
        loop = asyncio.get_event_loop()
        now = loop.time()
        integration._request_times = [now] * 4

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await integration._check_rate_limit()
            # Should not wait, still under limit
            mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_discord_429_retry_after_header(self, discord_integration, mock_response):
        """Discord 429 response uses Retry-After header."""
        responses = [
            mock_response(429, headers={"Retry-After": "2"}),
            mock_response(204),
        ]

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=create_response_sequence(responses))
        discord_integration._session = mock_session

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            embed = DiscordEmbed(title="Test")
            result = await discord_integration._send_webhook([embed])

            assert result is True
            # Should have slept for 2 seconds (from Retry-After header)
            mock_sleep.assert_any_call(2.0)

    @pytest.mark.asyncio
    async def test_discord_429_default_retry_after(self, discord_integration, mock_response):
        """Discord 429 without Retry-After uses default 5 seconds."""
        responses = [
            mock_response(429, headers={}),  # No Retry-After
            mock_response(204),
        ]

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=create_response_sequence(responses))
        discord_integration._session = mock_session

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            embed = DiscordEmbed(title="Test")
            await discord_integration._send_webhook([embed])

            # Default is 5 seconds
            mock_sleep.assert_any_call(5.0)


# =============================================================================
# Retry Logic Tests
# =============================================================================


class TestDiscordRetryLogic:
    """Tests for retry logic with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, discord_integration, mock_session):
        """Timeout triggers retry."""
        discord_integration._session = mock_session
        discord_integration.config.retry_count = 3

        # All attempts timeout
        async def timeout_effect(*args, **kwargs):
            raise asyncio.TimeoutError()

        mock_session.post = timeout_effect

        with patch("asyncio.sleep", new_callable=AsyncMock):
            embed = DiscordEmbed(title="Test")
            result = await discord_integration._send_webhook([embed])

            assert result is False

    @pytest.mark.asyncio
    async def test_retry_exponential_backoff(self, discord_config, mock_session, mock_response):
        """Retries use exponential backoff."""
        discord_config.retry_delay = 1.0
        discord_config.retry_count = 3
        integration = DiscordIntegration(discord_config)
        integration._session = mock_session

        # All attempts fail with 500
        async def fail_effect(*args, **kwargs):
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=mock_response(500, text="Server error"))
            ctx.__aexit__ = AsyncMock(return_value=None)
            return ctx

        mock_session.post = fail_effect

        sleep_times = []
        async def track_sleep(duration):
            sleep_times.append(duration)

        with patch("asyncio.sleep", side_effect=track_sleep):
            embed = DiscordEmbed(title="Test")
            await integration._send_webhook([embed])

        # Backoff: retry_delay * (attempt + 1)
        # Attempt 0 fails -> sleep(1.0 * 1) = 1.0
        # Attempt 1 fails -> sleep(1.0 * 2) = 2.0
        # Attempt 2 fails -> no sleep (last attempt)
        assert sleep_times == [1.0, 2.0]

    @pytest.mark.asyncio
    async def test_retry_success_on_second_attempt(self, discord_integration, mock_response):
        """Success on second attempt stops retrying."""
        responses = [
            mock_response(500, text="Error"),
            mock_response(204),  # Success
        ]

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=create_response_sequence(responses))
        discord_integration._session = mock_session

        with patch("asyncio.sleep", new_callable=AsyncMock):
            embed = DiscordEmbed(title="Test")
            result = await discord_integration._send_webhook([embed])

            assert result is True
            assert mock_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exhausted_returns_false(self, discord_config, mock_response):
        """Max retries exhausted returns False."""
        discord_config.retry_count = 2
        integration = DiscordIntegration(discord_config)

        mock_session = MagicMock()
        mock_session.closed = False
        # Always return 500
        mock_session.post = MagicMock(return_value=MockContextManager(mock_response(500, text="Error")))
        integration._session = mock_session

        with patch("asyncio.sleep", new_callable=AsyncMock):
            embed = DiscordEmbed(title="Test")
            result = await integration._send_webhook([embed])

            assert result is False
            assert mock_session.post.call_count == 2  # Exactly retry_count attempts

    @pytest.mark.asyncio
    async def test_retry_on_generic_exception(self, discord_integration):
        """Generic exception triggers retry."""
        mock_session = MagicMock()
        mock_session.closed = False

        def error_effect(*args, **kwargs):
            raise ConnectionError("Connection failed")

        mock_session.post = MagicMock(side_effect=error_effect)
        discord_integration._session = mock_session

        with patch("asyncio.sleep", new_callable=AsyncMock):
            embed = DiscordEmbed(title="Test")
            result = await discord_integration._send_webhook([embed])

            assert result is False
            assert mock_session.post.call_count == 3  # All retries attempted

    @pytest.mark.asyncio
    async def test_no_sleep_after_last_retry(self, discord_config, mock_session, mock_response):
        """No sleep after last retry attempt."""
        discord_config.retry_count = 2
        integration = DiscordIntegration(discord_config)
        integration._session = mock_session

        async def fail_effect(*args, **kwargs):
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=mock_response(500, text="Error"))
            ctx.__aexit__ = AsyncMock(return_value=None)
            return ctx

        mock_session.post = fail_effect

        sleep_calls = []
        async def track_sleep(duration):
            sleep_calls.append(duration)

        with patch("asyncio.sleep", side_effect=track_sleep):
            embed = DiscordEmbed(title="Test")
            await integration._send_webhook([embed])

        # Only 1 sleep (after attempt 0, before attempt 1)
        # No sleep after attempt 1 (last attempt)
        assert len(sleep_calls) == 1

    @pytest.mark.asyncio
    async def test_429_retries_within_limit(self, discord_integration, mock_response):
        """429 rate limit retries and succeeds within retry limit."""
        discord_integration.config.retry_count = 3

        # First: 429, then success (within retry limit)
        responses = [
            mock_response(429, headers={"Retry-After": "0.1"}),
            mock_response(204),
        ]

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=create_response_sequence(responses))
        discord_integration._session = mock_session

        with patch("asyncio.sleep", new_callable=AsyncMock):
            embed = DiscordEmbed(title="Test")
            result = await discord_integration._send_webhook([embed])

            assert result is True
            # 429 uses continue and retries within limit
            assert mock_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_delay_from_config(self, discord_config, mock_session, mock_response):
        """Retry delay uses configured value."""
        discord_config.retry_delay = 2.5
        discord_config.retry_count = 2
        integration = DiscordIntegration(discord_config)
        integration._session = mock_session

        async def fail_effect(*args, **kwargs):
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(return_value=mock_response(500, text="Error"))
            ctx.__aexit__ = AsyncMock(return_value=None)
            return ctx

        mock_session.post = fail_effect

        sleep_times = []
        async def track_sleep(duration):
            sleep_times.append(duration)

        with patch("asyncio.sleep", side_effect=track_sleep):
            embed = DiscordEmbed(title="Test")
            await integration._send_webhook([embed])

        # Should be retry_delay * (attempt + 1) = 2.5 * 1 = 2.5
        assert sleep_times == [2.5]


# =============================================================================
# HTTP Error Handling Tests
# =============================================================================


class TestDiscordHttpErrors:
    """Tests for HTTP error handling."""

    @pytest.mark.asyncio
    async def test_400_bad_request_logged(self, discord_integration, mock_response, caplog):
        """400 Bad Request is logged as error."""
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(
            return_value=MockContextManager(mock_response(400, text="Bad Request: invalid json"))
        )
        discord_integration._session = mock_session

        with patch("asyncio.sleep", new_callable=AsyncMock):
            embed = DiscordEmbed(title="Test")
            result = await discord_integration._send_webhook([embed])

        assert result is False
        assert "400" in caplog.text

    @pytest.mark.asyncio
    async def test_401_unauthorized_logged(self, discord_integration, mock_response, caplog):
        """401 Unauthorized is logged."""
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(
            return_value=MockContextManager(mock_response(401, text="Unauthorized"))
        )
        discord_integration._session = mock_session

        with patch("asyncio.sleep", new_callable=AsyncMock):
            embed = DiscordEmbed(title="Test")
            result = await discord_integration._send_webhook([embed])

        assert result is False
        assert "401" in caplog.text

    @pytest.mark.asyncio
    async def test_403_forbidden_logged(self, discord_integration, mock_response, caplog):
        """403 Forbidden is logged."""
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(
            return_value=MockContextManager(mock_response(403, text="Forbidden"))
        )
        discord_integration._session = mock_session

        with patch("asyncio.sleep", new_callable=AsyncMock):
            embed = DiscordEmbed(title="Test")
            result = await discord_integration._send_webhook([embed])

        assert result is False
        assert "403" in caplog.text

    @pytest.mark.asyncio
    async def test_404_not_found_logged(self, discord_integration, mock_response, caplog):
        """404 Not Found is logged (invalid webhook URL)."""
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(
            return_value=MockContextManager(mock_response(404, text="Unknown Webhook"))
        )
        discord_integration._session = mock_session

        with patch("asyncio.sleep", new_callable=AsyncMock):
            embed = DiscordEmbed(title="Test")
            result = await discord_integration._send_webhook([embed])

        assert result is False
        assert "404" in caplog.text

    @pytest.mark.asyncio
    async def test_500_internal_server_error(self, discord_integration, mock_response, caplog):
        """500 Internal Server Error triggers retry."""
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(
            return_value=MockContextManager(mock_response(500, text="Internal Server Error"))
        )
        discord_integration._session = mock_session

        with patch("asyncio.sleep", new_callable=AsyncMock):
            embed = DiscordEmbed(title="Test")
            result = await discord_integration._send_webhook([embed])

        assert result is False
        assert mock_session.post.call_count == 3  # Retried
        assert "500" in caplog.text

    @pytest.mark.asyncio
    async def test_503_service_unavailable(self, discord_integration, mock_response, caplog):
        """503 Service Unavailable triggers retry."""
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(
            return_value=MockContextManager(mock_response(503, text="Service Unavailable"))
        )
        discord_integration._session = mock_session

        with patch("asyncio.sleep", new_callable=AsyncMock):
            embed = DiscordEmbed(title="Test")
            result = await discord_integration._send_webhook([embed])

        assert result is False
        assert mock_session.post.call_count == 3  # Retried
        assert "503" in caplog.text

    @pytest.mark.asyncio
    async def test_204_no_content_success(self, discord_integration, mock_response):
        """204 No Content is success."""
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(return_value=MockContextManager(mock_response(204)))
        discord_integration._session = mock_session

        embed = DiscordEmbed(title="Test")
        result = await discord_integration._send_webhook([embed])

        assert result is True


# =============================================================================
# Field Truncation Tests
# =============================================================================


class TestDiscordFieldTruncation:
    """Tests for Discord field truncation limits."""

    def test_embed_description_truncated_at_2048(self):
        """Description is truncated at 2048 characters."""
        long_description = "x" * 3000
        embed = DiscordEmbed(title="Test", description=long_description)

        result = embed.to_dict()

        assert len(result["description"]) == 2048

    def test_embed_fields_limited_to_25(self):
        """Fields array is limited to 25 items."""
        fields = [{"name": f"Field {i}", "value": f"Value {i}", "inline": True} for i in range(30)]
        embed = DiscordEmbed(title="Test", fields=fields)

        result = embed.to_dict()

        assert len(result["fields"]) == 25

    def test_truncate_method_short_text(self, discord_integration):
        """Truncate method returns short text unchanged."""
        text = "Short text"

        result = discord_integration._truncate(text, 100)

        assert result == "Short text"

    def test_truncate_method_exact_length(self, discord_integration):
        """Truncate method returns exact length text unchanged."""
        text = "x" * 100

        result = discord_integration._truncate(text, 100)

        assert result == text
        assert len(result) == 100

    def test_truncate_method_adds_ellipsis(self, discord_integration):
        """Truncate method adds ellipsis for long text."""
        text = "x" * 200

        result = discord_integration._truncate(text, 100)

        assert len(result) == 100
        assert result.endswith("...")

    @pytest.mark.asyncio
    async def test_content_truncated_at_2000(self, discord_integration, mock_response):
        """Message content is truncated at 2000 characters."""
        captured_payload = {}

        class CapturingContextManager:
            def __init__(self, payload_ref, response):
                self.payload_ref = payload_ref
                self.response = response

            async def __aenter__(self):
                return self.response

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        def capture_post(*args, **kwargs):
            captured_payload.update(kwargs.get("json", {}))
            return CapturingContextManager(captured_payload, mock_response(204))

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(side_effect=capture_post)
        discord_integration._session = mock_session

        embed = DiscordEmbed(title="Test")
        long_content = "x" * 3000
        await discord_integration._send_webhook([embed], content=long_content)

        assert len(captured_payload.get("content", "")) == 2000

    def test_embed_fields_individual_truncation(self, discord_integration):
        """Individual field values use truncation."""
        long_text = "y" * 2000
        truncated = discord_integration._truncate(long_text, 1024)

        assert len(truncated) == 1024
        assert truncated.endswith("...")


# =============================================================================
# Session Management Tests
# =============================================================================


class TestDiscordSessionManagement:
    """Tests for aiohttp session management."""

    @pytest.mark.asyncio
    async def test_session_lazy_creation(self, discord_integration):
        """Session is created lazily on first use."""
        assert discord_integration._session is None

        with patch("aiohttp.ClientSession") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.closed = False
            mock_cls.return_value = mock_instance

            session = await discord_integration._get_session()

            mock_cls.assert_called_once()
            assert discord_integration._session is mock_instance

    @pytest.mark.asyncio
    async def test_session_reused(self, discord_integration):
        """Existing session is reused."""
        mock_session = MagicMock()
        mock_session.closed = False
        discord_integration._session = mock_session

        with patch("aiohttp.ClientSession") as mock_cls:
            session = await discord_integration._get_session()

            mock_cls.assert_not_called()
            assert session is mock_session

    @pytest.mark.asyncio
    async def test_closed_session_recreated(self, discord_integration):
        """Closed session is recreated."""
        old_session = MagicMock()
        old_session.closed = True
        discord_integration._session = old_session

        with patch("aiohttp.ClientSession") as mock_cls:
            new_session = MagicMock()
            new_session.closed = False
            mock_cls.return_value = new_session

            session = await discord_integration._get_session()

            mock_cls.assert_called_once()
            assert session is new_session

    @pytest.mark.asyncio
    async def test_close_method(self, discord_integration):
        """Close method closes session."""
        mock_session = AsyncMock()
        mock_session.closed = False
        discord_integration._session = mock_session

        await discord_integration.close()

        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_no_session(self, discord_integration):
        """Close handles no session gracefully."""
        assert discord_integration._session is None

        # Should not raise
        await discord_integration.close()


# =============================================================================
# Webhook Manager Tests
# =============================================================================


class TestDiscordWebhookManager:
    """Tests for DiscordWebhookManager."""

    def test_register_integration(self):
        """Register creates integration."""
        manager = DiscordWebhookManager()
        config = DiscordConfig(webhook_url="https://discord.com/api/webhooks/test")

        manager.register("test", config)

        assert "test" in manager._integrations
        assert isinstance(manager._integrations["test"], DiscordIntegration)

    def test_unregister_integration(self):
        """Unregister removes integration."""
        manager = DiscordWebhookManager()
        config = DiscordConfig(webhook_url="https://discord.com/api/webhooks/test")
        manager.register("test", config)

        manager.unregister("test")

        assert "test" not in manager._integrations

    def test_unregister_nonexistent(self):
        """Unregister handles nonexistent integration."""
        manager = DiscordWebhookManager()

        # Should not raise
        manager.unregister("nonexistent")

    @pytest.mark.asyncio
    async def test_broadcast_calls_all_integrations(self):
        """Broadcast calls method on all integrations."""
        manager = DiscordWebhookManager()

        # Create mock integrations
        integration1 = AsyncMock()
        integration1.send_debate_start = AsyncMock(return_value=True)
        integration2 = AsyncMock()
        integration2.send_debate_start = AsyncMock(return_value=True)

        manager._integrations = {"int1": integration1, "int2": integration2}

        results = await manager.broadcast(
            "send_debate_start",
            debate_id="123",
            topic="Test topic",
            agents=["agent1"],
            config={},
        )

        assert results == {"int1": True, "int2": True}
        integration1.send_debate_start.assert_called_once()
        integration2.send_debate_start.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_handles_failures(self):
        """Broadcast handles individual failures gracefully."""
        manager = DiscordWebhookManager()

        integration1 = AsyncMock()
        integration1.send_debate_start = AsyncMock(return_value=True)
        integration2 = AsyncMock()
        integration2.send_debate_start = AsyncMock(side_effect=Exception("Failed"))

        manager._integrations = {"int1": integration1, "int2": integration2}

        results = await manager.broadcast(
            "send_debate_start",
            debate_id="123",
            topic="Test",
            agents=[],
            config={},
        )

        assert results["int1"] is True
        assert results["int2"] is False

    @pytest.mark.asyncio
    async def test_broadcast_unknown_method(self):
        """Broadcast skips integrations without method."""
        manager = DiscordWebhookManager()

        integration = MagicMock()
        # No send_custom method
        del integration.send_custom

        manager._integrations = {"int1": integration}

        results = await manager.broadcast("send_custom")

        assert results == {}

    @pytest.mark.asyncio
    async def test_close_all(self):
        """Close all closes all integrations."""
        manager = DiscordWebhookManager()

        integration1 = AsyncMock()
        integration2 = AsyncMock()

        manager._integrations = {"int1": integration1, "int2": integration2}

        await manager.close_all()

        integration1.close.assert_called_once()
        integration2.close.assert_called_once()


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateDiscordIntegration:
    """Tests for create_discord_integration factory."""

    def test_creates_integration_with_url(self):
        """Factory creates integration with URL."""
        integration = create_discord_integration("https://discord.com/api/webhooks/test")

        assert isinstance(integration, DiscordIntegration)
        assert integration.config.webhook_url == "https://discord.com/api/webhooks/test"

    def test_passes_kwargs_to_config(self):
        """Factory passes kwargs to config."""
        integration = create_discord_integration(
            "https://discord.com/api/webhooks/test",
            username="Custom Bot",
            retry_count=5,
        )

        assert integration.config.username == "Custom Bot"
        assert integration.config.retry_count == 5

    def test_default_config_values(self):
        """Factory uses default config values."""
        integration = create_discord_integration("https://discord.com/api/webhooks/test")

        assert integration.config.enabled is True
        assert integration.config.rate_limit_per_minute == 30
        assert integration.config.retry_delay == 1.0


# =============================================================================
# Disabled Integration Tests
# =============================================================================


class TestDiscordDisabled:
    """Tests for disabled Discord integration."""

    @pytest.mark.asyncio
    async def test_disabled_returns_false(self):
        """Disabled integration returns False without sending."""
        config = DiscordConfig(
            webhook_url="https://discord.com/api/webhooks/test",
            enabled=False,
        )
        integration = DiscordIntegration(config)

        embed = DiscordEmbed(title="Test")
        result = await integration._send_webhook([embed])

        assert result is False

    @pytest.mark.asyncio
    async def test_disabled_skips_rate_limit_check(self):
        """Disabled integration skips rate limit check."""
        config = DiscordConfig(
            webhook_url="https://discord.com/api/webhooks/test",
            enabled=False,
        )
        integration = DiscordIntegration(config)

        embed = DiscordEmbed(title="Test")
        await integration._send_webhook([embed])

        # No request times recorded since we returned early
        assert len(integration._request_times) == 0


# =============================================================================
# Concurrency Tests
# =============================================================================


class TestDiscordConcurrency:
    """Tests for concurrent webhook operations."""

    @pytest.mark.asyncio
    async def test_concurrent_webhook_sends(self, discord_config, mock_response):
        """Multiple webhooks can send concurrently."""
        integration = DiscordIntegration(discord_config)

        # Create mock session
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(
            return_value=MockContextManager(mock_response(204))
        )
        integration._session = mock_session

        # Send multiple webhooks concurrently
        embeds = [DiscordEmbed(title=f"Test {i}") for i in range(5)]
        tasks = [integration._send_webhook([embed]) for embed in embeds]

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(results)
        # All requests should have been tracked for rate limiting
        assert len(integration._request_times) == 5

    @pytest.mark.asyncio
    async def test_session_reuse_under_load(self, discord_config, mock_response):
        """Session is reused across concurrent requests."""
        integration = DiscordIntegration(discord_config)

        session_created = [0]
        original_get_session = integration._get_session

        async def tracked_get_session():
            session_created[0] += 1
            return await original_get_session()

        # Set up mock session
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.post = MagicMock(
            return_value=MockContextManager(mock_response(204))
        )

        with patch.object(integration, '_get_session', side_effect=tracked_get_session):
            integration._session = mock_session

            # Send multiple webhooks concurrently
            embeds = [DiscordEmbed(title=f"Test {i}") for i in range(10)]
            tasks = [integration._send_webhook([embed]) for embed in embeds]

            await asyncio.gather(*tasks)

        # All requests completed
        assert mock_session.post.call_count == 10

    @pytest.mark.asyncio
    async def test_session_recreation_on_close(self, discord_config, mock_response):
        """Closed session is recreated on next request."""
        integration = DiscordIntegration(discord_config)

        # First session (will be closed)
        closed_session = MagicMock()
        closed_session.closed = True
        integration._session = closed_session

        # New session to be created
        with patch('aiohttp.ClientSession') as mock_cls:
            new_session = MagicMock()
            new_session.closed = False
            new_session.post = MagicMock(
                return_value=MockContextManager(mock_response(204))
            )
            mock_cls.return_value = new_session

            embed = DiscordEmbed(title="Test")
            result = await integration._send_webhook([embed])

            # Should have created new session
            mock_cls.assert_called_once()
            assert result is True

    @pytest.mark.asyncio
    async def test_multiple_integrations_parallel(self, mock_response):
        """Multiple integration instances work in parallel."""
        configs = [
            DiscordConfig(
                webhook_url=f"https://discord.com/api/webhooks/{i}/abc",
                rate_limit_per_minute=30,
            )
            for i in range(3)
        ]
        integrations = [DiscordIntegration(config) for config in configs]

        # Set up mock sessions for each integration
        for integration in integrations:
            mock_session = MagicMock()
            mock_session.closed = False
            mock_session.post = MagicMock(
                return_value=MockContextManager(mock_response(204))
            )
            integration._session = mock_session

        # Send from all integrations concurrently
        async def send_from_integration(idx):
            embed = DiscordEmbed(title=f"Integration {idx}")
            return await integrations[idx]._send_webhook([embed])

        results = await asyncio.gather(*[
            send_from_integration(i) for i in range(3)
        ])

        # All should succeed
        assert all(results)
        # Each integration should have its own request tracking
        for integration in integrations:
            assert len(integration._request_times) == 1
