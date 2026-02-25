"""Tests for fal.ai API key rotation handler."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest

from aragora.security.fal_rotator import (
    _get_current_key,
    _update_local_env,
    rotate_fal_key,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeServiceConfig:
    """Minimal ServiceKeyConfig stand-in."""

    secret_manager_key = "FAL_KEY"
    secret_id = "aragora/production"
    standalone_secret_id = "aragora/api/fal"


class FakeProxyConfig:
    """Minimal ProxyConfig stand-in."""

    services = {"fal": FakeServiceConfig()}
    aws_region = "us-east-2"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_fal_handler_registered():
    """Import the module and verify it auto-registers."""
    from aragora.security.api_key_proxy import _rotation_handlers

    import aragora.security.fal_rotator  # noqa: F401

    assert "fal" in _rotation_handlers


# ---------------------------------------------------------------------------
# _get_current_key
# ---------------------------------------------------------------------------


def test_get_current_key_from_env(monkeypatch):
    """Falls back to env when secrets module unavailable."""
    monkeypatch.setenv("FAL_KEY", "test-fal-key-123")
    key = _get_current_key(FakeServiceConfig())
    assert key == "test-fal-key-123"


def test_get_current_key_fal_key_fallback(monkeypatch):
    """Falls back to FAL_KEY env var."""
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.setenv("FAL_KEY", "fal-key-456")
    key = _get_current_key(FakeServiceConfig())
    assert key == "fal-key-456"


def test_get_current_key_none_when_missing(monkeypatch):
    """Returns None when no key is available."""
    monkeypatch.delenv("FAL_KEY", raising=False)
    key = _get_current_key(FakeServiceConfig())
    assert key is None


# ---------------------------------------------------------------------------
# _update_local_env
# ---------------------------------------------------------------------------


def test_update_local_env(tmp_path, monkeypatch):
    """Updates .env file with new key value."""
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER_KEY=foo\nFAL_KEY=old-key-value\n")

    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))

    _update_local_env("FAL_KEY", "new-key-value")

    content = env_file.read_text()
    assert "FAL_KEY=new-key-value" in content
    assert "OTHER_KEY=foo" in content
    assert "old-key-value" not in content


def test_update_local_env_no_file(tmp_path, monkeypatch):
    """Gracefully handles missing .env file."""
    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))
    # Should not raise
    _update_local_env("FAL_KEY", "new-key-value")
    assert os.environ.get("FAL_KEY") == "new-key-value"


# ---------------------------------------------------------------------------
# ProxyConfig includes fal
# ---------------------------------------------------------------------------


def test_proxy_config_includes_fal():
    """Default ProxyConfig includes fal service."""
    from aragora.security.api_key_proxy import ProxyConfig

    config = ProxyConfig.default()
    assert "fal" in config.services
    assert config.services["fal"].secret_manager_key == "FAL_KEY"
    assert config.services["fal"].rotation_interval_hours == 6.0


# ---------------------------------------------------------------------------
# Full rotation flow (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rotate_fal_key_no_service_in_config():
    """Rotation returns None when service not found in config."""
    config = FakeProxyConfig()
    config.services = {}
    result = await rotate_fal_key("fal", config)
    assert result is None


@pytest.mark.asyncio
async def test_rotate_fal_key_no_current_key(monkeypatch):
    """Rotation fails when no current key exists."""
    monkeypatch.delenv("FAL_KEY", raising=False)
    result = await rotate_fal_key("fal", FakeProxyConfig())
    assert result is None


@pytest.mark.asyncio
async def test_rotate_fal_key_create_fails(monkeypatch):
    """Rotation fails when key creation fails."""
    monkeypatch.setenv("FAL_KEY", "current-key")

    with patch(
        "aragora.security.fal_rotator._create_fal_key",
        new_callable=AsyncMock,
        return_value=None,
    ):
        result = await rotate_fal_key("fal", FakeProxyConfig())
    assert result is None


@pytest.mark.asyncio
async def test_rotate_fal_key_success(monkeypatch):
    """Full successful rotation flow."""
    monkeypatch.setenv("FAL_KEY", "old-fal-key-abc123")

    new_key = "new-fal-key-xyz789"

    with (
        patch(
            "aragora.security.fal_rotator._create_fal_key",
            new_callable=AsyncMock,
            return_value={
                "key": new_key,
                "key_id": "key-123",
                "name": "aragora-prod-20260224-0100",
            },
        ),
        patch(
            "aragora.security.fal_rotator._update_secrets_manager",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "aragora.security.fal_rotator._update_standalone_secret",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch("aragora.security.fal_rotator._update_local_env"),
        patch(
            "aragora.security.fal_rotator._get_key_id",
            new_callable=AsyncMock,
            return_value="old-key-456",
        ),
        patch(
            "aragora.security.fal_rotator._delete_fal_key",
            new_callable=AsyncMock,
            return_value=True,
        ),
    ):
        result = await rotate_fal_key("fal", FakeProxyConfig())

    assert result == new_key


@pytest.mark.asyncio
async def test_rotate_aborts_on_secrets_manager_failure(monkeypatch):
    """If secrets manager update fails, abort rotation."""
    monkeypatch.setenv("FAL_KEY", "old-key")

    with (
        patch(
            "aragora.security.fal_rotator._create_fal_key",
            new_callable=AsyncMock,
            return_value={
                "key": "new-key-value",
                "key_id": "new-key-id",
                "name": "test",
            },
        ),
        patch(
            "aragora.security.fal_rotator._update_secrets_manager",
            new_callable=AsyncMock,
            return_value=False,
        ),
    ):
        result = await rotate_fal_key("fal", FakeProxyConfig())

    assert result is None


@pytest.mark.asyncio
async def test_rotate_continues_without_old_key_deletion(monkeypatch):
    """Rotation succeeds even if old key ID can't be found for deletion."""
    monkeypatch.setenv("FAL_KEY", "old-key")

    with (
        patch(
            "aragora.security.fal_rotator._create_fal_key",
            new_callable=AsyncMock,
            return_value={
                "key": "new-key-value",
                "key_id": "new-key-id",
                "name": "test",
            },
        ),
        patch(
            "aragora.security.fal_rotator._update_secrets_manager",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "aragora.security.fal_rotator._update_standalone_secret",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch("aragora.security.fal_rotator._update_local_env"),
        patch(
            "aragora.security.fal_rotator._get_key_id",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        result = await rotate_fal_key("fal", FakeProxyConfig())

    # Should still succeed — old key deletion is best-effort
    assert result == "new-key-value"


# ---------------------------------------------------------------------------
# API key proxy startup integration
# ---------------------------------------------------------------------------


def test_proxy_config_has_all_three_services():
    """Default config includes elevenlabs, gemini, and fal."""
    from aragora.security.api_key_proxy import ProxyConfig

    config = ProxyConfig.default()
    assert "elevenlabs" in config.services
    assert "gemini" in config.services
    assert "fal" in config.services
