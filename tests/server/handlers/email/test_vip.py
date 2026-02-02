"""
Tests for VIP management handlers.

Covers:
- handle_add_vip (add email/domain)
- handle_remove_vip (remove email/domain)
- Persistent store integration
- Prioritizer reset on VIP change
- RBAC permission checks
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import aragora.server.handlers.email.storage as storage_mod
from aragora.server.handlers.email.vip import handle_add_vip, handle_remove_vip


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    storage_mod._prioritizer = None
    storage_mod._gmail_connector = None
    storage_mod._email_store = None
    with storage_mod._user_configs_lock:
        storage_mod._user_configs.clear()
    yield
    storage_mod._prioritizer = None
    storage_mod._gmail_connector = None
    storage_mod._email_store = None
    with storage_mod._user_configs_lock:
        storage_mod._user_configs.clear()


@pytest.fixture(autouse=True)
def _bypass_rbac():
    with patch(
        "aragora.server.handlers.email.vip._check_email_permission",
        return_value=None,
    ):
        yield


@pytest.fixture(autouse=True)
def _bypass_store():
    """Prevent real persistent store access."""
    with (
        patch(
            "aragora.server.handlers.email.vip.get_email_store",
            return_value=None,
        ),
        patch(
            "aragora.server.handlers.email.vip._load_config_from_store",
            return_value={},
        ),
        patch(
            "aragora.server.handlers.email.vip._save_config_to_store",
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# handle_add_vip
# ---------------------------------------------------------------------------


class TestHandleAddVip:
    @pytest.mark.asyncio
    async def test_add_email(self):
        result = await handle_add_vip(user_id="u1", email="ceo@corp.com")
        assert result["success"] is True
        assert result["added"]["email"] == "ceo@corp.com"
        assert "ceo@corp.com" in result["vip_addresses"]

    @pytest.mark.asyncio
    async def test_add_domain(self):
        result = await handle_add_vip(user_id="u1", domain="important.com")
        assert result["success"] is True
        assert result["added"]["domain"] == "important.com"
        assert "important.com" in result["vip_domains"]

    @pytest.mark.asyncio
    async def test_add_both(self):
        result = await handle_add_vip(user_id="u1", email="ceo@corp.com", domain="corp.com")
        assert result["success"] is True
        assert "ceo@corp.com" in result["vip_addresses"]
        assert "corp.com" in result["vip_domains"]

    @pytest.mark.asyncio
    async def test_no_duplicates(self):
        """Adding the same email twice doesn't duplicate."""
        await handle_add_vip(user_id="u1", email="ceo@corp.com")
        result = await handle_add_vip(user_id="u1", email="ceo@corp.com")
        assert result["vip_addresses"].count("ceo@corp.com") == 1

    @pytest.mark.asyncio
    async def test_resets_prioritizer(self):
        """Adding a VIP should reset the prioritizer singleton."""
        storage_mod._prioritizer = MagicMock()  # simulate existing
        await handle_add_vip(user_id="u1", email="ceo@corp.com")
        assert storage_mod._prioritizer is None

    @pytest.mark.asyncio
    async def test_persists_to_store(self):
        """Config is saved to persistent store."""
        with patch("aragora.server.handlers.email.vip._save_config_to_store") as mock_save:
            await handle_add_vip(user_id="u1", email="ceo@corp.com")
        mock_save.assert_called_once()
        call_args = mock_save.call_args[0]
        assert call_args[0] == "u1"

    @pytest.mark.asyncio
    async def test_add_to_vip_store_table(self):
        """VIP email is also added to dedicated store table."""
        mock_store = MagicMock()
        with patch(
            "aragora.server.handlers.email.vip.get_email_store",
            return_value=mock_store,
        ):
            await handle_add_vip(user_id="u1", email="ceo@corp.com", workspace_id="ws1")
        mock_store.add_vip_sender.assert_called_once_with("u1", "ws1", "ceo@corp.com")

    @pytest.mark.asyncio
    async def test_rbac_denied(self):
        with patch(
            "aragora.server.handlers.email.vip._check_email_permission",
            return_value={"success": False, "error": "denied"},
        ):
            result = await handle_add_vip(user_id="u1", email="x@y.com", auth_context=MagicMock())
        assert result["success"] is False


# ---------------------------------------------------------------------------
# handle_remove_vip
# ---------------------------------------------------------------------------


class TestHandleRemoveVip:
    @pytest.mark.asyncio
    async def test_remove_email(self):
        # Seed the config first
        with storage_mod._user_configs_lock:
            storage_mod._user_configs["u1"] = {
                "vip_addresses": ["ceo@corp.com", "cfo@corp.com"],
                "vip_domains": [],
            }
        result = await handle_remove_vip(user_id="u1", email="ceo@corp.com")
        assert result["success"] is True
        assert result["removed"]["email"] == "ceo@corp.com"
        assert "ceo@corp.com" not in result["vip_addresses"]
        assert "cfo@corp.com" in result["vip_addresses"]

    @pytest.mark.asyncio
    async def test_remove_domain(self):
        with storage_mod._user_configs_lock:
            storage_mod._user_configs["u1"] = {
                "vip_addresses": [],
                "vip_domains": ["important.com"],
            }
        result = await handle_remove_vip(user_id="u1", domain="important.com")
        assert result["success"] is True
        assert result["removed"]["domain"] == "important.com"
        assert "important.com" not in result["vip_domains"]

    @pytest.mark.asyncio
    async def test_remove_nonexistent_is_noop(self):
        with storage_mod._user_configs_lock:
            storage_mod._user_configs["u1"] = {
                "vip_addresses": ["other@corp.com"],
                "vip_domains": [],
            }
        result = await handle_remove_vip(user_id="u1", email="nonexistent@corp.com")
        assert result["success"] is True
        assert result["removed"]["email"] is None

    @pytest.mark.asyncio
    async def test_resets_prioritizer(self):
        storage_mod._prioritizer = MagicMock()
        with storage_mod._user_configs_lock:
            storage_mod._user_configs["u1"] = {"vip_addresses": ["x@y.com"], "vip_domains": []}
        await handle_remove_vip(user_id="u1", email="x@y.com")
        assert storage_mod._prioritizer is None

    @pytest.mark.asyncio
    async def test_remove_from_vip_store_table(self):
        mock_store = MagicMock()
        with storage_mod._user_configs_lock:
            storage_mod._user_configs["u1"] = {
                "vip_addresses": ["ceo@corp.com"],
                "vip_domains": [],
            }
        with patch(
            "aragora.server.handlers.email.vip.get_email_store",
            return_value=mock_store,
        ):
            await handle_remove_vip(user_id="u1", email="ceo@corp.com", workspace_id="ws1")
        mock_store.remove_vip_sender.assert_called_once_with("u1", "ws1", "ceo@corp.com")

    @pytest.mark.asyncio
    async def test_rbac_denied(self):
        with patch(
            "aragora.server.handlers.email.vip._check_email_permission",
            return_value={"success": False, "error": "denied"},
        ):
            result = await handle_remove_vip(
                user_id="u1", email="x@y.com", auth_context=MagicMock()
            )
        assert result["success"] is False
