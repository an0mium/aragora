"""Tests for DeepSeek API key rotation handler."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest

from aragora.security.deepseek_rotator import (
    _get_current_key,
    _update_local_env,
    rotate_deepseek_key,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeServiceConfig:
    """Minimal ServiceKeyConfig stand-in."""

    secret_manager_key = "DEEPSEEK_API_KEY"
    secret_id = "aragora/production"
    standalone_secret_id = "aragora/api/deepseek"


class FakeProxyConfig:
    """Minimal ProxyConfig stand-in."""

    services = {"deepseek": FakeServiceConfig()}
    aws_region = "us-east-2"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_deepseek_handler_registered():
    """Import the module and verify it auto-registers."""
    from aragora.security.api_key_proxy import _rotation_handlers

    import aragora.security.deepseek_rotator  # noqa: F401

    assert "deepseek" in _rotation_handlers


# ---------------------------------------------------------------------------
# _get_current_key
# ---------------------------------------------------------------------------


def test_get_current_key_from_env(monkeypatch):
    """Falls back to env when secrets module unavailable."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key-123")
    key = _get_current_key(FakeServiceConfig())
    assert key == "test-deepseek-key-123"


def test_get_current_key_none_when_missing(monkeypatch):
    """Returns None when no key is available."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    key = _get_current_key(FakeServiceConfig())
    assert key is None


# ---------------------------------------------------------------------------
# _update_local_env
# ---------------------------------------------------------------------------


def test_update_local_env(tmp_path, monkeypatch):
    """Updates .env file with new key value."""
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER_KEY=foo\nDEEPSEEK_API_KEY=old-key-value\n")

    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))

    _update_local_env("DEEPSEEK_API_KEY", "new-key-value")

    content = env_file.read_text()
    assert "DEEPSEEK_API_KEY=new-key-value" in content
    assert "OTHER_KEY=foo" in content
    assert "old-key-value" not in content


def test_update_local_env_no_file(tmp_path, monkeypatch):
    """Gracefully handles missing .env file."""
    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))
    # Should not raise
    _update_local_env("DEEPSEEK_API_KEY", "new-key-value")
    assert os.environ.get("DEEPSEEK_API_KEY") == "new-key-value"


# ---------------------------------------------------------------------------
# ProxyConfig includes deepseek
# ---------------------------------------------------------------------------


def test_proxy_config_includes_deepseek():
    """Default ProxyConfig includes deepseek service."""
    from aragora.security.api_key_proxy import ProxyConfig

    config = ProxyConfig.default()
    assert "deepseek" in config.services
    assert config.services["deepseek"].secret_manager_key == "DEEPSEEK_API_KEY"
    assert config.services["deepseek"].rotation_interval_hours == 8.0


# ---------------------------------------------------------------------------
# Full rotation flow (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rotate_deepseek_key_no_service_in_config():
    """Rotation returns None when service not found in config."""
    config = FakeProxyConfig()
    config.services = {}
    result = await rotate_deepseek_key("deepseek", config)
    assert result is None


@pytest.mark.asyncio
async def test_rotate_deepseek_key_no_current_key(monkeypatch):
    """Rotation fails when no current key exists."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    result = await rotate_deepseek_key("deepseek", FakeProxyConfig())
    assert result is None


@pytest.mark.asyncio
async def test_rotate_deepseek_key_create_fails(monkeypatch):
    """Rotation fails when key creation fails."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "current-key")

    with patch(
        "aragora.security.deepseek_rotator._create_deepseek_key",
        new_callable=AsyncMock,
        return_value=None,
    ):
        result = await rotate_deepseek_key("deepseek", FakeProxyConfig())
    assert result is None


@pytest.mark.asyncio
async def test_rotate_deepseek_key_success(monkeypatch):
    """Full successful rotation flow."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "old-deepseek-key-abc123")

    new_key = "new-deepseek-key-xyz789"

    with (
        patch(
            "aragora.security.deepseek_rotator._create_deepseek_key",
            new_callable=AsyncMock,
            return_value={
                "key": new_key,
                "key_id": "key-123",
                "name": "aragora-prod-20260224-0100",
            },
        ),
        patch(
            "aragora.security.deepseek_rotator._update_secrets_manager",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "aragora.security.deepseek_rotator._update_standalone_secret",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch("aragora.security.deepseek_rotator._update_local_env"),
        patch(
            "aragora.security.deepseek_rotator._get_key_id",
            new_callable=AsyncMock,
            return_value="old-key-456",
        ),
        patch(
            "aragora.security.deepseek_rotator._delete_deepseek_key",
            new_callable=AsyncMock,
            return_value=True,
        ),
    ):
        result = await rotate_deepseek_key("deepseek", FakeProxyConfig())

    assert result == new_key


@pytest.mark.asyncio
async def test_rotate_aborts_on_secrets_manager_failure(monkeypatch):
    """If secrets manager update fails, abort rotation."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "old-key")

    with (
        patch(
            "aragora.security.deepseek_rotator._create_deepseek_key",
            new_callable=AsyncMock,
            return_value={
                "key": "new-key-value",
                "key_id": "new-key-id",
                "name": "test",
            },
        ),
        patch(
            "aragora.security.deepseek_rotator._update_secrets_manager",
            new_callable=AsyncMock,
            return_value=False,
        ),
    ):
        result = await rotate_deepseek_key("deepseek", FakeProxyConfig())

    assert result is None


@pytest.mark.asyncio
async def test_rotate_continues_without_old_key_deletion(monkeypatch):
    """Rotation succeeds even if old key ID can't be found for deletion."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "old-key")

    with (
        patch(
            "aragora.security.deepseek_rotator._create_deepseek_key",
            new_callable=AsyncMock,
            return_value={
                "key": "new-key-value",
                "key_id": "new-key-id",
                "name": "test",
            },
        ),
        patch(
            "aragora.security.deepseek_rotator._update_secrets_manager",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "aragora.security.deepseek_rotator._update_standalone_secret",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch("aragora.security.deepseek_rotator._update_local_env"),
        patch(
            "aragora.security.deepseek_rotator._get_key_id",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        result = await rotate_deepseek_key("deepseek", FakeProxyConfig())

    # Should still succeed — old key deletion is best-effort
    assert result == "new-key-value"
