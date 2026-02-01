"""Tests for request retries and timeouts in the Aragora SDK."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from aragora.client import AragoraAsyncClient, AragoraClient
from aragora.exceptions import AragoraError


class TestSyncRetries:
    """Tests for synchronous client retry behavior."""

    def test_successful_request_no_retry(self) -> None:
        """Successful request does not trigger retry."""
        with patch("httpx.Client.request") as mock_request:
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_response.content = b'{"status": "ok"}'
            mock_response.json.return_value = {"status": "ok"}
            mock_request.return_value = mock_response

            client = AragoraClient(base_url="https://api.aragora.ai", max_retries=3)
            result = client.request("GET", "/api/v1/health")

            assert mock_request.call_count == 1
            assert result["status"] == "ok"
            client.close()

    def test_retry_on_timeout(self) -> None:
        """Client retries on timeout exception."""
        with patch("httpx.Client.request") as mock_request:
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_response.content = b'{"status": "ok"}'
            mock_response.json.return_value = {"status": "ok"}

            # First two calls timeout, third succeeds
            mock_request.side_effect = [
                httpx.TimeoutException("Timeout"),
                httpx.TimeoutException("Timeout"),
                mock_response,
            ]

            with patch("time.sleep") as mock_sleep:
                client = AragoraClient(
                    base_url="https://api.aragora.ai",
                    max_retries=3,
                    retry_delay=1.0,
                )
                result = client.request("GET", "/api/v1/health")

                assert mock_request.call_count == 3
                assert result["status"] == "ok"
                # Check exponential backoff: 1*2^0=1, 1*2^1=2
                assert mock_sleep.call_count == 2
                client.close()

    def test_retry_on_connection_error(self) -> None:
        """Client retries on connection error."""
        with patch("httpx.Client.request") as mock_request:
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_response.content = b'{"status": "ok"}'
            mock_response.json.return_value = {"status": "ok"}

            # First call fails, second succeeds
            mock_request.side_effect = [
                httpx.ConnectError("Connection refused"),
                mock_response,
            ]

            with patch("time.sleep"):
                client = AragoraClient(base_url="https://api.aragora.ai", max_retries=3)
                result = client.request("GET", "/api/v1/health")

                assert mock_request.call_count == 2
                assert result["status"] == "ok"
                client.close()

    def test_max_retries_exceeded_timeout(self) -> None:
        """Client raises error after max retries on timeout."""
        with patch("httpx.Client.request") as mock_request:
            mock_request.side_effect = httpx.TimeoutException("Timeout")

            with patch("time.sleep"):
                client = AragoraClient(base_url="https://api.aragora.ai", max_retries=3)

                with pytest.raises(AragoraError) as exc_info:
                    client.request("GET", "/api/v1/health")

                assert "timed out" in str(exc_info.value).lower()
                assert mock_request.call_count == 3
                client.close()

    def test_max_retries_exceeded_connection(self) -> None:
        """Client raises error after max retries on connection error."""
        with patch("httpx.Client.request") as mock_request:
            mock_request.side_effect = httpx.ConnectError("Connection refused")

            with patch("time.sleep"):
                client = AragoraClient(base_url="https://api.aragora.ai", max_retries=3)

                with pytest.raises(AragoraError) as exc_info:
                    client.request("GET", "/api/v1/health")

                assert "connection" in str(exc_info.value).lower()
                assert mock_request.call_count == 3
                client.close()

    def test_exponential_backoff(self) -> None:
        """Retry delay uses exponential backoff."""
        with patch("httpx.Client.request") as mock_request:
            mock_request.side_effect = httpx.TimeoutException("Timeout")

            with patch("time.sleep") as mock_sleep:
                client = AragoraClient(
                    base_url="https://api.aragora.ai",
                    max_retries=4,
                    retry_delay=1.0,
                )

                with pytest.raises(AragoraError):
                    client.request("GET", "/api/v1/health")

                # Exponential backoff: 1*2^0=1, 1*2^1=2, 1*2^2=4
                expected_delays = [1.0, 2.0, 4.0]
                actual_delays = [c[0][0] for c in mock_sleep.call_args_list]
                assert actual_delays == expected_delays
                client.close()

    def test_custom_retry_delay(self) -> None:
        """Custom retry delay is respected."""
        with patch("httpx.Client.request") as mock_request:
            mock_request.side_effect = httpx.TimeoutException("Timeout")

            with patch("time.sleep") as mock_sleep:
                client = AragoraClient(
                    base_url="https://api.aragora.ai",
                    max_retries=3,
                    retry_delay=0.5,
                )

                with pytest.raises(AragoraError):
                    client.request("GET", "/api/v1/health")

                # With retry_delay=0.5: 0.5*2^0=0.5, 0.5*2^1=1.0
                expected_delays = [0.5, 1.0]
                actual_delays = [c[0][0] for c in mock_sleep.call_args_list]
                assert actual_delays == expected_delays
                client.close()

    def test_no_retry_on_http_error(self) -> None:
        """HTTP errors (4xx, 5xx) are not retried by default."""
        from aragora.exceptions import ValidationError

        with patch("httpx.Client.request") as mock_request:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.is_success = False
            mock_response.status_code = 400
            mock_response.text = "Bad request"
            mock_response.json.return_value = {"error": "Bad request"}
            mock_response.headers = {}
            mock_request.return_value = mock_response

            client = AragoraClient(base_url="https://api.aragora.ai", max_retries=3)

            with pytest.raises(ValidationError):
                client.request("GET", "/api/v1/debates")

            # Should not retry on HTTP errors
            assert mock_request.call_count == 1
            client.close()


class TestAsyncRetries:
    """Tests for async client retry behavior."""

    @pytest.mark.asyncio
    async def test_async_successful_request_no_retry(self) -> None:
        """Async successful request does not trigger retry."""
        with patch("httpx.AsyncClient.request") as mock_request:
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_response.content = b'{"status": "ok"}'
            mock_response.json.return_value = {"status": "ok"}
            mock_request.return_value = mock_response

            async with AragoraAsyncClient(
                base_url="https://api.aragora.ai", max_retries=3
            ) as client:
                result = await client.request("GET", "/api/v1/health")

                assert mock_request.call_count == 1
                assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_async_retry_on_timeout(self) -> None:
        """Async client retries on timeout."""
        with patch("httpx.AsyncClient.request") as mock_request:
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_response.content = b'{"status": "ok"}'
            mock_response.json.return_value = {"status": "ok"}

            # First call times out, second succeeds
            mock_request.side_effect = [
                httpx.TimeoutException("Timeout"),
                mock_response,
            ]

            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                async with AragoraAsyncClient(
                    base_url="https://api.aragora.ai",
                    max_retries=3,
                    retry_delay=1.0,
                ) as client:
                    result = await client.request("GET", "/api/v1/health")

                    assert mock_request.call_count == 2
                    assert result["status"] == "ok"
                    mock_sleep.assert_called_once_with(1.0)

    @pytest.mark.asyncio
    async def test_async_retry_on_connection_error(self) -> None:
        """Async client retries on connection error."""
        with patch("httpx.AsyncClient.request") as mock_request:
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_response.content = b'{"status": "ok"}'
            mock_response.json.return_value = {"status": "ok"}

            mock_request.side_effect = [
                httpx.ConnectError("Connection refused"),
                mock_response,
            ]

            with patch("asyncio.sleep", new_callable=AsyncMock):
                async with AragoraAsyncClient(
                    base_url="https://api.aragora.ai", max_retries=3
                ) as client:
                    result = await client.request("GET", "/api/v1/health")

                    assert mock_request.call_count == 2
                    assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_async_max_retries_exceeded(self) -> None:
        """Async client raises error after max retries."""
        with patch("httpx.AsyncClient.request") as mock_request:
            mock_request.side_effect = httpx.TimeoutException("Timeout")

            with patch("asyncio.sleep", new_callable=AsyncMock):
                async with AragoraAsyncClient(
                    base_url="https://api.aragora.ai", max_retries=3
                ) as client:
                    with pytest.raises(AragoraError) as exc_info:
                        await client.request("GET", "/api/v1/health")

                    assert "timed out" in str(exc_info.value).lower()
                    assert mock_request.call_count == 3

    @pytest.mark.asyncio
    async def test_async_exponential_backoff(self) -> None:
        """Async retry uses exponential backoff."""
        with patch("httpx.AsyncClient.request") as mock_request:
            mock_request.side_effect = httpx.TimeoutException("Timeout")

            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                async with AragoraAsyncClient(
                    base_url="https://api.aragora.ai",
                    max_retries=4,
                    retry_delay=0.5,
                ) as client:
                    with pytest.raises(AragoraError):
                        await client.request("GET", "/api/v1/health")

                    # Check exponential backoff calls
                    expected_delays = [0.5, 1.0, 2.0]
                    actual_delays = [c[0][0] for c in mock_sleep.call_args_list]
                    assert actual_delays == expected_delays


class TestTimeoutConfiguration:
    """Tests for timeout configuration."""

    def test_default_timeout(self) -> None:
        """Default timeout is 30 seconds."""
        client = AragoraClient(base_url="https://api.aragora.ai")
        assert client.timeout == 30.0
        client.close()

    def test_custom_timeout(self) -> None:
        """Custom timeout is applied."""
        client = AragoraClient(base_url="https://api.aragora.ai", timeout=60.0)
        assert client.timeout == 60.0
        client.close()

    def test_timeout_passed_to_httpx_client(self) -> None:
        """Timeout is passed to httpx client."""
        with patch("httpx.Client") as mock_httpx_client:
            client = AragoraClient(base_url="https://api.aragora.ai", timeout=45.0)

            # Check that timeout was passed to httpx.Client
            mock_httpx_client.assert_called_once()
            call_kwargs = mock_httpx_client.call_args[1]
            assert call_kwargs["timeout"] == 45.0
            client.close()

    @pytest.mark.asyncio
    async def test_async_timeout_passed_to_httpx_client(self) -> None:
        """Async timeout is passed to httpx AsyncClient."""
        with patch("httpx.AsyncClient") as mock_httpx_client:
            # Create a proper async mock for aclose
            mock_instance = MagicMock()
            mock_instance.aclose = AsyncMock()
            mock_httpx_client.return_value = mock_instance

            async with AragoraAsyncClient(base_url="https://api.aragora.ai", timeout=90.0):
                # Check timeout was passed
                mock_httpx_client.assert_called_once()
                call_kwargs = mock_httpx_client.call_args[1]
                assert call_kwargs["timeout"] == 90.0


class TestRetryConfiguration:
    """Tests for retry configuration."""

    def test_default_max_retries(self) -> None:
        """Default max_retries is 3."""
        client = AragoraClient(base_url="https://api.aragora.ai")
        assert client.max_retries == 3
        client.close()

    def test_custom_max_retries(self) -> None:
        """Custom max_retries is applied."""
        client = AragoraClient(base_url="https://api.aragora.ai", max_retries=5)
        assert client.max_retries == 5
        client.close()

    def test_default_retry_delay(self) -> None:
        """Default retry_delay is 1.0 seconds."""
        client = AragoraClient(base_url="https://api.aragora.ai")
        assert client.retry_delay == 1.0
        client.close()

    def test_custom_retry_delay(self) -> None:
        """Custom retry_delay is applied."""
        client = AragoraClient(base_url="https://api.aragora.ai", retry_delay=2.5)
        assert client.retry_delay == 2.5
        client.close()

    def test_max_retries_one(self) -> None:
        """With max_retries=1, only one attempt is made."""
        with patch("httpx.Client.request") as mock_request:
            mock_request.side_effect = httpx.TimeoutException("Timeout")

            client = AragoraClient(base_url="https://api.aragora.ai", max_retries=1)

            with pytest.raises(AragoraError):
                client.request("GET", "/api/v1/health")

            assert mock_request.call_count == 1
            client.close()


class TestEmptyResponseHandling:
    """Tests for handling empty responses."""

    def test_empty_response_returns_empty_dict(self) -> None:
        """Empty response content returns empty dict."""
        with patch("httpx.Client.request") as mock_request:
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_response.content = b""  # Empty content
            mock_request.return_value = mock_response

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.request("DELETE", "/api/v1/debates/deb_123")

            assert result == {}
            client.close()

    @pytest.mark.asyncio
    async def test_async_empty_response_returns_empty_dict(self) -> None:
        """Async empty response returns empty dict."""
        with patch("httpx.AsyncClient.request") as mock_request:
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_response.content = b""
            mock_request.return_value = mock_response

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.request("DELETE", "/api/v1/debates/deb_123")

                assert result == {}
