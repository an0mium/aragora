"""
Tests for email configuration handlers.

Covers:
- handle_get_config (read config with cache + store fallback)
- handle_update_config (update config, persist, reset prioritizer)
- RBAC permission checks
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

import aragora.server.handlers.email.storage as storage_mod
from aragora.server.handlers.email.config import (
    handle_get_config,
    handle_update_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    saved_store = storage_mod._email_store
    storage_mod._prioritizer = None
    storage_mod._email_store = None
    with storage_mod._user_configs_lock:
        storage_mod._user_configs.clear()
    yield
    storage_mod._prioritizer = None
    storage_mod._email_store = saved_store
    with storage_mod._user_configs_lock:
        storage_mod._user_configs.clear()


@pytest.fixture(autouse=True)
def _bypass_rbac():
    with patch(
        "aragora.server.handlers.email.config._check_email_permission",
        return_value=None,
    ):
        yield


@pytest.fixture(autouse=True)
def _bypass_require_permission():
    """Bypass @require_permission decorator on handle_update_config."""
    with patch(
        "aragora.server.handlers.email.config.require_permission",
        lambda perm: (lambda fn: fn),
    ):
        yield


@pytest.fixture(autouse=True)
def _bypass_store():
    with (
        patch(
            "aragora.server.handlers.email.config._load_config_from_store",
            return_value={},
        ),
        patch(
            "aragora.server.handlers.email.config._save_config_to_store",
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# handle_get_config
# ---------------------------------------------------------------------------


class TestHandleGetConfig:
    @pytest.mark.asyncio
    async def test_returns_default_config_for_new_user(self):
        result = await handle_get_config("new_user")
        assert result["success"] is True
        cfg = result["config"]
        assert cfg["vip_domains"] == []
        assert cfg["vip_addresses"] == []
        assert cfg["internal_domains"] == []
        assert cfg["auto_archive_senders"] == []
        assert cfg["tier_1_confidence_threshold"] == 0.7
        assert cfg["tier_2_confidence_threshold"] == 0.6
        assert cfg["enable_slack_signals"] is True
        assert cfg["enable_calendar_signals"] is True
        assert cfg["enable_drive_signals"] is True

    @pytest.mark.asyncio
    async def test_returns_cached_config(self):
        with storage_mod._user_configs_lock:
            storage_mod._user_configs["u1"] = {
                "vip_domains": ["test.com"],
                "vip_addresses": ["boss@test.com"],
            }
        result = await handle_get_config("u1")
        assert result["success"] is True
        assert result["config"]["vip_domains"] == ["test.com"]
        assert result["config"]["vip_addresses"] == ["boss@test.com"]

    @pytest.mark.asyncio
    async def test_loads_from_store_when_not_cached(self):
        store_config = {"vip_domains": ["stored.com"], "internal_domains": ["internal.com"]}
        with patch(
            "aragora.server.handlers.email.config._load_config_from_store",
            return_value=store_config,
        ):
            result = await handle_get_config("u1")
        assert result["config"]["vip_domains"] == ["stored.com"]
        assert result["config"]["internal_domains"] == ["internal.com"]
        # Verify it was cached
        with storage_mod._user_configs_lock:
            assert "u1" in storage_mod._user_configs

    @pytest.mark.asyncio
    async def test_rbac_denied(self):
        with patch(
            "aragora.server.handlers.email.config._check_email_permission",
            return_value={"success": False, "error": "denied"},
        ):
            result = await handle_get_config("u1", auth_context=MagicMock())
        assert result["success"] is False


# ---------------------------------------------------------------------------
# handle_update_config
# ---------------------------------------------------------------------------


class TestHandleUpdateConfig:
    @pytest.mark.asyncio
    async def test_update_vip_domains(self):
        result = await handle_update_config("u1", {"vip_domains": ["newclient.com"]})
        assert result["success"] is True
        assert "newclient.com" in result["config"]["vip_domains"]

    @pytest.mark.asyncio
    async def test_update_multiple_fields(self):
        result = await handle_update_config(
            "u1",
            {
                "vip_addresses": ["ceo@corp.com"],
                "internal_domains": ["corp.com"],
                "enable_slack_signals": False,
                "tier_1_confidence_threshold": 0.8,
            },
        )
        assert result["success"] is True
        cfg = result["config"]
        assert "ceo@corp.com" in cfg["vip_addresses"]
        assert "corp.com" in cfg["internal_domains"]
        assert cfg["enable_slack_signals"] is False
        assert cfg["tier_1_confidence_threshold"] == 0.8

    @pytest.mark.asyncio
    async def test_preserves_existing_fields(self):
        """Updating one field doesn't erase others."""
        with storage_mod._user_configs_lock:
            storage_mod._user_configs["u1"] = {"vip_domains": ["existing.com"]}
        result = await handle_update_config("u1", {"vip_addresses": ["new@addr.com"]})
        assert result["success"] is True
        assert "existing.com" in result["config"]["vip_domains"]
        assert "new@addr.com" in result["config"]["vip_addresses"]

    @pytest.mark.asyncio
    async def test_persists_to_store(self):
        with patch("aragora.server.handlers.email.config._save_config_to_store") as mock_save:
            await handle_update_config("u1", {"vip_domains": ["x.com"]})
        mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_resets_prioritizer(self):
        storage_mod._prioritizer = MagicMock()
        await handle_update_config("u1", {"vip_domains": ["x.com"]})
        assert storage_mod._prioritizer is None

    @pytest.mark.asyncio
    async def test_empty_updates_is_noop(self):
        result = await handle_update_config("u1", {})
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_none_config_updates(self):
        result = await handle_update_config("u1")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_exception_returns_error(self):
        with patch(
            "aragora.server.handlers.email.config._save_config_to_store",
            side_effect=RuntimeError("db error"),
        ):
            result = await handle_update_config("u1", {"vip_domains": ["x.com"]})
        assert result["success"] is False
        assert "db error" in result["error"]
