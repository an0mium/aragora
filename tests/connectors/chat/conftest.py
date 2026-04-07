"""Shared fixtures for chat connector tests."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from aragora.connectors.chat.base import ChatPlatformConnector
from aragora.connectors.chat.models import (
    FileAttachment,
    SendMessageResponse,
    WebhookEvent,
)


class StubConnector(ChatPlatformConnector):
    """Minimal concrete implementation for testing base connector behaviour."""

    @property
    def platform_name(self) -> str:
        return "stub"

    @property
    def platform_display_name(self) -> str:
        return "Stub Platform"

    async def send_message(self, channel_id, text, blocks=None, thread_id=None, **kw):
        return SendMessageResponse(success=True, message_id="msg-1")

    async def update_message(self, channel_id, message_id, text, blocks=None, **kw):
        return SendMessageResponse(success=True, message_id=message_id)

    async def delete_message(self, channel_id, message_id, **kw):
        return True

    async def respond_to_command(self, command, text, blocks=None, ephemeral=True, **kw):
        return SendMessageResponse(success=True)

    async def respond_to_interaction(
        self, interaction, text, blocks=None, replace_original=False, **kw
    ):
        return SendMessageResponse(success=True)

    async def upload_file(
        self,
        channel_id,
        content,
        filename,
        content_type="application/octet-stream",
        title=None,
        thread_id=None,
        **kw,
    ):
        return FileAttachment(
            id="file-1", filename=filename, content_type=content_type, size=len(content)
        )

    async def download_file(self, file_id, **kw):
        return FileAttachment(id=file_id, filename="test.txt", content_type="text/plain", size=0)

    def format_blocks(self, title=None, body=None, fields=None, actions=None, **kw):
        return [{"type": "section", "text": body}]

    def format_button(self, text, action_id, value=None, style="default", url=None):
        return {"type": "button", "text": text, "action_id": action_id}

    def verify_webhook(self, headers, body):
        return True

    def parse_webhook_event(self, headers, body):
        return WebhookEvent(platform="stub", event_type="message")


@pytest.fixture(autouse=True)
def reset_circuit_breakers():
    """Clear circuit breakers before and after each chat connector test."""
    from aragora.resilience import _circuit_breakers, _circuit_breakers_lock

    with _circuit_breakers_lock:
        _circuit_breakers.clear()
    yield
    with _circuit_breakers_lock:
        _circuit_breakers.clear()


@pytest.fixture
def connector():
    """Create a default stub connector with circuit breaker enabled."""
    return StubConnector(bot_token="tok-123", signing_secret="sec-456")


@pytest.fixture
def connector_no_cb():
    """Create a stub connector with circuit breaker disabled."""
    return StubConnector(bot_token="tok-123", enable_circuit_breaker=False)


@pytest.fixture
def rsa_key_pair():
    """Generate an RSA key pair for JWT verification tests."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )
    public_key = private_key.public_key()

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    return private_key, public_key, private_pem, public_pem


@pytest.fixture
def mock_jwks_client():
    """Create a controllable mock PyJWKClient pair."""
    mock_client = MagicMock()
    mock_signing_key = MagicMock()
    mock_client.get_signing_key_from_jwt.return_value = mock_signing_key
    return mock_client, mock_signing_key


@pytest.fixture
def mock_httpx_response():
    """Create a reusable mock httpx response factory."""

    def _create_response(json_data: dict, status_code: int = 200, text: str | None = None):
        mock = MagicMock()
        mock.json.return_value = json_data
        mock.status_code = status_code
        payload_text = text if text is not None else json.dumps(json_data)
        mock.text = payload_text
        mock.content = payload_text.encode()
        return mock

    return _create_response
