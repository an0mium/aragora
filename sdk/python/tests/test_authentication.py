"""Tests for authentication handling in the Aragora SDK."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from aragora.client import AragoraAsyncClient, AragoraClient
from aragora.exceptions import AuthenticationError, AuthorizationError


class TestAPIKeyAuthentication:
    """Tests for API key authentication."""

    def test_api_key_included_in_headers(self) -> None:
        """API key is included in Authorization header as Bearer token."""
        client = AragoraClient(base_url="https://api.aragora.ai", api_key="sk-test-key")
        headers = client._build_headers()
        assert headers["Authorization"] == "Bearer sk-test-key"
        client.close()

    def test_no_authorization_header_without_api_key(self) -> None:
        """No Authorization header when API key is not provided."""
        client = AragoraClient(base_url="https://api.aragora.ai")
        headers = client._build_headers()
        assert "Authorization" not in headers
        client.close()

    def test_api_key_can_be_jwt_token(self) -> None:
        """API key can be a JWT token."""
        jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        client = AragoraClient(base_url="https://api.aragora.ai", api_key=jwt_token)
        headers = client._build_headers()
        assert headers["Authorization"] == f"Bearer {jwt_token}"
        client.close()

    @pytest.mark.asyncio
    async def test_async_client_api_key_in_headers(self) -> None:
        """Async client includes API key in Authorization header."""
        async with AragoraAsyncClient(
            base_url="https://api.aragora.ai", api_key="sk-test-key"
        ) as client:
            headers = client._build_headers()
            assert headers["Authorization"] == "Bearer sk-test-key"


class TestAuthenticationErrors:
    """Tests for authentication error handling."""

    def test_401_raises_authentication_error(self) -> None:
        """401 response raises AuthenticationError."""
        client = AragoraClient(base_url="https://api.aragora.ai", api_key="invalid-key")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 401
        mock_response.is_success = False
        mock_response.text = "Invalid API key"
        mock_response.json.return_value = {"error": "Invalid API key"}
        mock_response.headers = {}

        with pytest.raises(AuthenticationError) as exc_info:
            client._handle_error_response(mock_response)

        assert "Invalid API key" in str(exc_info.value)
        assert exc_info.value.status_code == 401
        client.close()

    def test_403_raises_authorization_error(self) -> None:
        """403 response raises AuthorizationError."""
        client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 403
        mock_response.is_success = False
        mock_response.text = "Access denied"
        mock_response.json.return_value = {"error": "Access denied"}
        mock_response.headers = {}

        with pytest.raises(AuthorizationError) as exc_info:
            client._handle_error_response(mock_response)

        assert "Access denied" in str(exc_info.value)
        assert exc_info.value.status_code == 403
        client.close()

    def test_authentication_error_includes_response_body(self) -> None:
        """AuthenticationError includes response body for debugging."""
        client = AragoraClient(base_url="https://api.aragora.ai")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 401
        mock_response.is_success = False
        mock_response.text = "Unauthorized"
        mock_response.json.return_value = {
            "error": "Token expired",
            "code": "TOKEN_EXPIRED",
        }
        mock_response.headers = {}

        with pytest.raises(AuthenticationError) as exc_info:
            client._handle_error_response(mock_response)

        assert exc_info.value.response_body["code"] == "TOKEN_EXPIRED"
        client.close()

    @pytest.mark.asyncio
    async def test_async_401_raises_authentication_error(self) -> None:
        """Async client raises AuthenticationError on 401."""
        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 401
            mock_response.is_success = False
            mock_response.text = "Invalid token"
            mock_response.json.return_value = {"error": "Invalid token"}
            mock_response.headers = {}

            with pytest.raises(AuthenticationError):
                client._handle_error_response(mock_response)


class TestJWTAuthentication:
    """Tests for JWT authentication scenarios."""

    def test_jwt_token_format_accepted(self) -> None:
        """JWT tokens are accepted as API keys."""
        # Example JWT structure
        jwt = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyMTIzIiwiZXhwIjoxNzA0MDY3MjAwfQ.signature"
        client = AragoraClient(base_url="https://api.aragora.ai", api_key=jwt)
        assert client.api_key == jwt
        client.close()

    def test_empty_api_key_treated_as_none(self) -> None:
        """Empty string API key should not add Authorization header."""
        client = AragoraClient(base_url="https://api.aragora.ai", api_key="")
        headers = client._build_headers()
        # Empty string is falsy, so no Authorization header
        assert "Authorization" not in headers
        client.close()


class TestAuthenticationWithRequest:
    """Tests for authentication during actual requests."""

    def test_request_includes_auth_header(self) -> None:
        """HTTP requests include the Authorization header."""
        with patch("httpx.Client.request") as mock_request:
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_response.content = b'{"status": "ok"}'
            mock_response.json.return_value = {"status": "ok"}
            mock_request.return_value = mock_response

            client = AragoraClient(base_url="https://api.aragora.ai", api_key="sk-test-key")
            client.request("GET", "/api/v1/health")

            # Verify request was called with auth header
            call_kwargs = mock_request.call_args[1]
            assert call_kwargs["headers"]["Authorization"] == "Bearer sk-test-key"
            client.close()

    @pytest.mark.asyncio
    async def test_async_request_includes_auth_header(self) -> None:
        """Async HTTP requests include the Authorization header."""
        with patch("httpx.AsyncClient.request") as mock_request:
            mock_response = MagicMock()
            mock_response.is_success = True
            mock_response.content = b'{"status": "ok"}'
            mock_response.json.return_value = {"status": "ok"}

            # Create async mock
            mock_request.return_value = mock_response

            async with AragoraAsyncClient(
                base_url="https://api.aragora.ai", api_key="sk-test-key"
            ) as client:
                await client.request("GET", "/api/v1/health")

                # Verify request was called with auth header
                call_kwargs = mock_request.call_args[1]
                assert call_kwargs["headers"]["Authorization"] == "Bearer sk-test-key"
