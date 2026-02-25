"""Tests for Gemini API key rotation handler."""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.security.gemini_rotator import (
    _get_current_key,
    _get_project_id,
    _update_local_env,
    rotate_gemini_key,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeServiceConfig:
    """Minimal ServiceKeyConfig stand-in."""

    secret_manager_key = "GEMINI_API_KEY"
    secret_id = "aragora/production"
    standalone_secret_id = "aragora/api/gemini"


class FakeProxyConfig:
    """Minimal ProxyConfig stand-in."""

    services = {"gemini": FakeServiceConfig()}
    aws_region = "us-east-2"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_gemini_handler_registered():
    """Import the module and verify it auto-registers."""
    from aragora.security.api_key_proxy import _rotation_handlers

    # Importing the module triggers _register()
    import aragora.security.gemini_rotator  # noqa: F401

    assert "gemini" in _rotation_handlers


# ---------------------------------------------------------------------------
# _get_current_key
# ---------------------------------------------------------------------------


def test_get_current_key_from_env(monkeypatch):
    """Falls back to env when secrets module unavailable."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key-123")
    with patch("aragora.config.secrets.get_secret", return_value=None):
        key = _get_current_key(FakeServiceConfig())
    assert key == "test-gemini-key-123"


def test_get_current_key_google_api_key_fallback(monkeypatch):
    """Falls back to GOOGLE_API_KEY when GEMINI_API_KEY missing."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key-456")
    with patch("aragora.config.secrets.get_secret", return_value=None):
        key = _get_current_key(FakeServiceConfig())
    assert key == "google-key-456"


def test_get_current_key_none_when_missing(monkeypatch):
    """Returns None when no key is available."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with patch("aragora.config.secrets.get_secret", return_value=None):
        key = _get_current_key(FakeServiceConfig())
    assert key is None


# ---------------------------------------------------------------------------
# _get_project_id
# ---------------------------------------------------------------------------


def test_get_project_id_from_env(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "my-project-123")
    assert _get_project_id() == "my-project-123"


def test_get_project_id_from_google_cloud_project(monkeypatch):
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-cloud-project")
    assert _get_project_id() == "my-cloud-project"


# ---------------------------------------------------------------------------
# _update_local_env
# ---------------------------------------------------------------------------


def test_update_local_env(tmp_path, monkeypatch):
    """Updates .env file with new key value."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OTHER_KEY=foo\nGEMINI_API_KEY=old-key-value\nGOOGLE_API_KEY=old-key-value\n"
    )

    # Patch to find our test .env
    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))

    _update_local_env("GEMINI_API_KEY", "new-key-value")

    content = env_file.read_text()
    assert "GEMINI_API_KEY=new-key-value" in content
    assert "GOOGLE_API_KEY=new-key-value" in content
    assert "OTHER_KEY=foo" in content
    assert "old-key-value" not in content


# ---------------------------------------------------------------------------
# ProxyConfig includes gemini
# ---------------------------------------------------------------------------


def test_proxy_config_includes_gemini():
    """Default ProxyConfig includes gemini service."""
    from aragora.security.api_key_proxy import ProxyConfig

    config = ProxyConfig.default()
    assert "gemini" in config.services
    assert config.services["gemini"].secret_manager_key == "GEMINI_API_KEY"
    assert config.services["gemini"].rotation_interval_hours == 4.0


# ---------------------------------------------------------------------------
# Full rotation flow (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rotate_gemini_key_no_service_in_config():
    """Rotation returns None when service not found in config."""
    config = FakeProxyConfig()
    config.services = {}  # Empty services
    result = await rotate_gemini_key("gemini", config)
    assert result is None


@pytest.mark.asyncio
async def test_rotate_gemini_key_no_service_config():
    """Rotation fails when service not in config."""

    class EmptyConfig:
        services = {}

    with patch("aragora.security.gemini_rotator._get_access_token", return_value=None):
        result = await rotate_gemini_key("gemini", EmptyConfig())
    assert result is None


@pytest.mark.asyncio
async def test_rotate_gemini_key_no_credentials(monkeypatch):
    """Rotation fails when no Google Cloud credentials."""
    monkeypatch.setenv("GEMINI_API_KEY", "current-key")

    with patch("aragora.security.gemini_rotator._get_access_token", return_value=None):
        result = await rotate_gemini_key("gemini", FakeProxyConfig())
    assert result is None


@pytest.mark.asyncio
async def test_rotate_gemini_key_no_project_id(monkeypatch):
    """Rotation fails when project ID not available."""
    monkeypatch.setenv("GEMINI_API_KEY", "current-key")

    with (
        patch("aragora.security.gemini_rotator._get_access_token", return_value="token-123"),
        patch("aragora.security.gemini_rotator._get_project_id", return_value=None),
    ):
        result = await rotate_gemini_key("gemini", FakeProxyConfig())
    assert result is None


@pytest.mark.asyncio
async def test_rotate_gemini_key_success(monkeypatch):
    """Full successful rotation flow."""
    monkeypatch.setenv("GEMINI_API_KEY", "old-key-abc123")

    new_key = "new-gemini-key-xyz789"

    with (
        patch("aragora.security.gemini_rotator._get_access_token", return_value="access-token"),
        patch("aragora.security.gemini_rotator._get_project_id", return_value="my-project"),
        patch(
            "aragora.security.gemini_rotator._create_api_key",
            new_callable=AsyncMock,
            return_value={
                "name": "projects/my-project/locations/global/keys/key-123",
                "key_string": new_key,
                "display_name": "aragora-gemini-20260224-0100",
            },
        ),
        patch(
            "aragora.security.gemini_rotator._restrict_key",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "aragora.security.gemini_rotator._update_secrets_manager",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "aragora.security.gemini_rotator._update_standalone_secret",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch("aragora.security.gemini_rotator._update_local_env"),
        patch(
            "aragora.security.gemini_rotator._find_key_by_string",
            new_callable=AsyncMock,
            return_value="projects/my-project/locations/global/keys/old-key-456",
        ),
        patch(
            "aragora.security.gemini_rotator._delete_api_key",
            new_callable=AsyncMock,
            return_value=True,
        ),
    ):
        result = await rotate_gemini_key("gemini", FakeProxyConfig())

    assert result == new_key


@pytest.mark.asyncio
async def test_rotate_aborts_on_secrets_manager_failure(monkeypatch):
    """If secrets manager update fails, abort and clean up new key."""
    monkeypatch.setenv("GEMINI_API_KEY", "old-key")

    delete_mock = AsyncMock(return_value=True)

    with (
        patch("aragora.security.gemini_rotator._get_access_token", return_value="access-token"),
        patch("aragora.security.gemini_rotator._get_project_id", return_value="proj"),
        patch(
            "aragora.security.gemini_rotator._create_api_key",
            new_callable=AsyncMock,
            return_value={
                "name": "projects/proj/locations/global/keys/new-key",
                "key_string": "new-key-value",
                "display_name": "test",
            },
        ),
        patch(
            "aragora.security.gemini_rotator._restrict_key",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "aragora.security.gemini_rotator._update_secrets_manager",
            new_callable=AsyncMock,
            return_value=False,  # Failure!
        ),
        patch(
            "aragora.security.gemini_rotator._delete_api_key",
            delete_mock,
        ),
    ):
        result = await rotate_gemini_key("gemini", FakeProxyConfig())

    # Should abort and return None
    assert result is None
    # Should attempt to clean up the newly created key
    delete_mock.assert_called_once_with(
        "access-token", "projects/proj/locations/global/keys/new-key"
    )
