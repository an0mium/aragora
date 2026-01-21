"""
Tests for async webhook dispatcher.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestAsyncWebhookDispatcher:
    """Tests for AsyncWebhookDispatcher."""

    @pytest.fixture
    def mock_webhook(self):
        """Create a mock webhook config."""
        webhook = MagicMock()
        webhook.id = "test-webhook-123"
        webhook.url = "https://example.com/webhook"
        webhook.secret = "test-secret"
        return webhook

    @pytest.mark.asyncio
    async def test_dispatcher_context_manager(self):
        """Test dispatcher works as context manager."""
        from aragora.events.async_dispatcher import AsyncWebhookDispatcher

        async with AsyncWebhookDispatcher() as dispatcher:
            assert dispatcher._client is not None

        # Client should be closed after context
        assert dispatcher._client is None

    @pytest.mark.asyncio
    async def test_dispatch_success(self, mock_webhook):
        """Test successful webhook dispatch."""
        from aragora.events.async_dispatcher import AsyncWebhookDispatcher

        async with AsyncWebhookDispatcher() as dispatcher:
            # Mock the httpx client response
            mock_response = MagicMock()
            mock_response.status_code = 200

            with patch.object(dispatcher._client, "post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = mock_response

                success, status, error = await dispatcher.dispatch(
                    mock_webhook, {"event": "test", "data": {}}
                )

                assert success is True
                assert status == 200
                assert error is None
                mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_failure(self, mock_webhook):
        """Test failed webhook dispatch."""
        from aragora.events.async_dispatcher import AsyncWebhookDispatcher

        async with AsyncWebhookDispatcher() as dispatcher:
            mock_response = MagicMock()
            mock_response.status_code = 500

            with patch.object(dispatcher._client, "post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = mock_response

                success, status, error = await dispatcher.dispatch(
                    mock_webhook, {"event": "test"}
                )

                assert success is False
                assert status == 500
                assert "500" in error

    @pytest.mark.asyncio
    async def test_dispatch_connection_error(self, mock_webhook):
        """Test webhook dispatch with connection error."""
        from aragora.events.async_dispatcher import AsyncWebhookDispatcher

        async with AsyncWebhookDispatcher() as dispatcher:
            with patch.object(dispatcher._client, "post", new_callable=AsyncMock) as mock_post:
                mock_post.side_effect = Exception("Connection refused")

                success, status, error = await dispatcher.dispatch(
                    mock_webhook, {"event": "test"}
                )

                assert success is False
                assert status == 0
                assert "Connection refused" in error

    @pytest.mark.asyncio
    async def test_dispatch_includes_signature(self, mock_webhook):
        """Test that dispatch includes signature header."""
        from aragora.events.async_dispatcher import AsyncWebhookDispatcher

        async with AsyncWebhookDispatcher() as dispatcher:
            mock_response = MagicMock()
            mock_response.status_code = 200

            with patch.object(dispatcher._client, "post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = mock_response

                await dispatcher.dispatch(mock_webhook, {"event": "test"})

                # Check headers
                call_kwargs = mock_post.call_args.kwargs
                headers = call_kwargs.get("headers", {})
                assert "X-Aragora-Signature" in headers
                assert headers["X-Aragora-Signature"].startswith("sha256=")
                assert headers["X-Aragora-Event"] == "test"

    @pytest.mark.asyncio
    async def test_dispatch_includes_correlation_id(self, mock_webhook):
        """Test that dispatch includes correlation ID when present."""
        from aragora.events.async_dispatcher import AsyncWebhookDispatcher

        async with AsyncWebhookDispatcher() as dispatcher:
            mock_response = MagicMock()
            mock_response.status_code = 200

            with patch.object(dispatcher._client, "post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = mock_response

                await dispatcher.dispatch(
                    mock_webhook,
                    {"event": "test", "correlation_id": "trace-123"},
                )

                headers = mock_post.call_args.kwargs.get("headers", {})
                assert headers.get("X-Aragora-Correlation-ID") == "trace-123"


class TestAsyncDeliveryWithRetry:
    """Tests for dispatch_with_retry."""

    @pytest.fixture
    def mock_webhook(self):
        """Create a mock webhook config."""
        webhook = MagicMock()
        webhook.id = "test-webhook-456"
        webhook.url = "https://example.com/webhook"
        webhook.secret = "test-secret"
        return webhook

    @pytest.mark.asyncio
    async def test_retry_on_5xx(self, mock_webhook):
        """Test that 5xx errors trigger retry."""
        from aragora.events.async_dispatcher import AsyncWebhookDispatcher

        async with AsyncWebhookDispatcher() as dispatcher:
            responses = [
                MagicMock(status_code=503),
                MagicMock(status_code=200),
            ]

            with patch.object(dispatcher._client, "post", new_callable=AsyncMock) as mock_post:
                mock_post.side_effect = responses

                result = await dispatcher.dispatch_with_retry(
                    mock_webhook,
                    {"event": "test"},
                    max_retries=2,
                    initial_delay=0.01,
                )

                assert result.success is True
                assert result.retry_count == 1
                assert mock_post.call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_4xx(self, mock_webhook):
        """Test that 4xx errors don't trigger retry."""
        from aragora.events.async_dispatcher import AsyncWebhookDispatcher

        async with AsyncWebhookDispatcher() as dispatcher:
            mock_response = MagicMock()
            mock_response.status_code = 400

            with patch.object(dispatcher._client, "post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = mock_response

                result = await dispatcher.dispatch_with_retry(
                    mock_webhook,
                    {"event": "test"},
                    max_retries=3,
                    initial_delay=0.01,
                )

                assert result.success is False
                assert result.status_code == 400
                assert result.retry_count == 0
                assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self, mock_webhook):
        """Test behavior when all retries fail."""
        from aragora.events.async_dispatcher import AsyncWebhookDispatcher

        async with AsyncWebhookDispatcher() as dispatcher:
            mock_response = MagicMock()
            mock_response.status_code = 503

            with patch.object(dispatcher._client, "post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = mock_response

                result = await dispatcher.dispatch_with_retry(
                    mock_webhook,
                    {"event": "test"},
                    max_retries=2,
                    initial_delay=0.01,
                )

                assert result.success is False
                assert result.retry_count == 2
                assert mock_post.call_count == 3  # Initial + 2 retries


class TestGlobalAsyncDispatcher:
    """Tests for global async dispatcher functions."""

    @pytest.fixture(autouse=True)
    async def reset_global(self):
        """Reset global dispatcher before each test."""
        from aragora.events import async_dispatcher

        if async_dispatcher._async_dispatcher is not None:
            await async_dispatcher._async_dispatcher.close()
            async_dispatcher._async_dispatcher = None
        yield
        if async_dispatcher._async_dispatcher is not None:
            await async_dispatcher._async_dispatcher.close()
            async_dispatcher._async_dispatcher = None

    @pytest.mark.asyncio
    async def test_get_async_dispatcher(self):
        """Test getting global dispatcher."""
        from aragora.events.async_dispatcher import get_async_dispatcher

        d1 = await get_async_dispatcher()
        d2 = await get_async_dispatcher()

        assert d1 is d2

    @pytest.mark.asyncio
    async def test_shutdown_async_dispatcher(self):
        """Test shutting down global dispatcher."""
        from aragora.events.async_dispatcher import (
            get_async_dispatcher,
            shutdown_async_dispatcher,
        )
        from aragora.events import async_dispatcher

        await get_async_dispatcher()
        assert async_dispatcher._async_dispatcher is not None

        await shutdown_async_dispatcher()
        assert async_dispatcher._async_dispatcher is None


class TestAsyncDeliveryResult:
    """Tests for AsyncDeliveryResult dataclass."""

    def test_result_creation(self):
        """Test creating delivery result."""
        from aragora.events.async_dispatcher import AsyncDeliveryResult

        result = AsyncDeliveryResult(
            success=True,
            status_code=200,
            retry_count=0,
            duration_ms=150.5,
        )

        assert result.success is True
        assert result.status_code == 200
        assert result.error is None
        assert result.duration_ms == 150.5

    def test_result_with_error(self):
        """Test creating failure result."""
        from aragora.events.async_dispatcher import AsyncDeliveryResult

        result = AsyncDeliveryResult(
            success=False,
            status_code=503,
            error="Service unavailable",
            retry_count=3,
            duration_ms=5000.0,
        )

        assert result.success is False
        assert result.error == "Service unavailable"
        assert result.retry_count == 3
