"""Tests for OpenRouter API key rotation handler."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest

from aragora.security.openrouter_rotator import (
    _get_current_key,
    _update_local_env,
    rotate_openrouter_key,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeServiceConfig:
    """Minimal ServiceKeyConfig stand-in."""

    secret_manager_key = "OPENROUTER_API_KEY"
    secret_id = "aragora/production"
    standalone_secret_id = "aragora/api/openrouter"


class FakeProxyConfig:
    """Minimal ProxyConfig stand-in."""

    services = {"openrouter": FakeServiceConfig()}
    aws_region = "us-east-2"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_openrouter_handler_registered():
    """Import the module and verify it auto-registers."""
    from aragora.security.api_key_proxy import _rotation_handlers

    import aragora.security.openrouter_rotator  # noqa: F401

    assert "openrouter" in _rotation_handlers


# ---------------------------------------------------------------------------
# _get_current_key
# ---------------------------------------------------------------------------


def test_get_current_key_from_env(monkeypatch):
    """Falls back to env when secrets module unavailable."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key-123")
    key = _get_current_key(FakeServiceConfig())
    assert key == "test-openrouter-key-123"


def test_get_current_key_none_when_missing(monkeypatch):
    """Returns None when no key is available."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    key = _get_current_key(FakeServiceConfig())
    assert key is None


# ---------------------------------------------------------------------------
# _update_local_env
# ---------------------------------------------------------------------------


def test_update_local_env(tmp_path, monkeypatch):
    """Updates .env file with new key value."""
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER_KEY=foo\nOPENROUTER_API_KEY=old-key-value\n")

    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))

    _update_local_env("OPENROUTER_API_KEY", "new-key-value")

    content = env_file.read_text()
    assert "OPENROUTER_API_KEY=new-key-value" in content
    assert "OTHER_KEY=foo" in content
    assert "old-key-value" not in content


def test_update_local_env_no_file(tmp_path, monkeypatch):
    """Gracefully handles missing .env file."""
    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))
    # Should not raise
    _update_local_env("OPENROUTER_API_KEY", "new-key-value")
    assert os.environ.get("OPENROUTER_API_KEY") == "new-key-value"


# ---------------------------------------------------------------------------
# ProxyConfig includes openrouter
# ---------------------------------------------------------------------------


def test_proxy_config_includes_openrouter():
    """Default ProxyConfig includes openrouter service."""
    from aragora.security.api_key_proxy import ProxyConfig

    config = ProxyConfig.default()
    assert "openrouter" in config.services
    assert config.services["openrouter"].secret_manager_key == "OPENROUTER_API_KEY"
    assert config.services["openrouter"].rotation_interval_hours == 12.0


# ---------------------------------------------------------------------------
# Full rotation flow (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rotate_openrouter_key_no_service_in_config():
    """Rotation returns None when service not found in config."""
    config = FakeProxyConfig()
    config.services = {}
    result = await rotate_openrouter_key("openrouter", config)
    assert result is None


@pytest.mark.asyncio
async def test_rotate_openrouter_key_no_current_key(monkeypatch):
    """Rotation fails when no current key exists."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    result = await rotate_openrouter_key("openrouter", FakeProxyConfig())
    assert result is None


@pytest.mark.asyncio
async def test_rotate_openrouter_key_create_fails(monkeypatch):
    """Rotation fails when key creation fails."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "current-key")

    with patch(
        "aragora.security.openrouter_rotator._create_openrouter_key",
        new_callable=AsyncMock,
        return_value=None,
    ):
        result = await rotate_openrouter_key("openrouter", FakeProxyConfig())
    assert result is None


@pytest.mark.asyncio
async def test_rotate_openrouter_key_success(monkeypatch):
    """Full successful rotation flow."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "old-openrouter-key-abc123")

    new_key = "new-openrouter-key-xyz789"

    with (
        patch(
            "aragora.security.openrouter_rotator._create_openrouter_key",
            new_callable=AsyncMock,
            return_value={
                "key": new_key,
                "key_id": "key-123",
                "name": "aragora-prod-20260224-0100",
            },
        ),
        patch(
            "aragora.security.openrouter_rotator._update_secrets_manager",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "aragora.security.openrouter_rotator._update_standalone_secret",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch("aragora.security.openrouter_rotator._update_local_env"),
        patch(
            "aragora.security.openrouter_rotator._get_key_id",
            new_callable=AsyncMock,
            return_value="old-key-456",
        ),
        patch(
            "aragora.security.openrouter_rotator._delete_openrouter_key",
            new_callable=AsyncMock,
            return_value=True,
        ),
    ):
        result = await rotate_openrouter_key("openrouter", FakeProxyConfig())

    assert result == new_key


@pytest.mark.asyncio
async def test_rotate_aborts_on_secrets_manager_failure(monkeypatch):
    """If secrets manager update fails, abort rotation."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "old-key")

    with (
        patch(
            "aragora.security.openrouter_rotator._create_openrouter_key",
            new_callable=AsyncMock,
            return_value={
                "key": "new-key-value",
                "key_id": "new-key-id",
                "name": "test",
            },
        ),
        patch(
            "aragora.security.openrouter_rotator._update_secrets_manager",
            new_callable=AsyncMock,
            return_value=False,
        ),
    ):
        result = await rotate_openrouter_key("openrouter", FakeProxyConfig())

    assert result is None


@pytest.mark.asyncio
async def test_rotate_continues_without_old_key_deletion(monkeypatch):
    """Rotation succeeds even if old key ID can't be found for deletion."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "old-key")

    with (
        patch(
            "aragora.security.openrouter_rotator._create_openrouter_key",
            new_callable=AsyncMock,
            return_value={
                "key": "new-key-value",
                "key_id": "new-key-id",
                "name": "test",
            },
        ),
        patch(
            "aragora.security.openrouter_rotator._update_secrets_manager",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "aragora.security.openrouter_rotator._update_standalone_secret",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch("aragora.security.openrouter_rotator._update_local_env"),
        patch(
            "aragora.security.openrouter_rotator._get_key_id",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        result = await rotate_openrouter_key("openrouter", FakeProxyConfig())

    # Should still succeed — old key deletion is best-effort
    assert result == "new-key-value"


# ---------------------------------------------------------------------------
# API key proxy startup integration
# ---------------------------------------------------------------------------


def test_proxy_config_has_openrouter_with_all_services():
    """Default config includes openrouter alongside other services."""
    from aragora.security.api_key_proxy import ProxyConfig

    config = ProxyConfig.default()
    assert "elevenlabs" in config.services
    assert "gemini" in config.services
    assert "fal" in config.services
    assert "openrouter" in config.services
