"""Tests for VIP management handlers (aragora/server/handlers/email/vip.py).

Covers all public functions, success/error paths, edge cases:
- handle_add_vip: add email, add domain, add both, duplicates, store errors,
  no email/domain, cache miss loads from store, store add_vip_sender failure,
  _save_config_to_store integration, prioritizer reset, exception paths
- handle_remove_vip: remove email, remove domain, remove both, non-existent,
  no vip_addresses/vip_domains keys, store errors, cache miss loads from store,
  store remove_vip_sender failure, prioritizer reset, exception paths
- Auth context: _AUTH_CONTEXT_UNSET bypass, permission check delegation,
  permission denied propagation
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import aragora.server.handlers.email.storage as storage_module
from aragora.server.handlers.email.vip import (
    _AUTH_CONTEXT_UNSET,
    PERM_EMAIL_READ,
    PERM_EMAIL_UPDATE,
    handle_add_vip,
    handle_remove_vip,
)


# ============================================================================
# Helpers
# ============================================================================

STORAGE = "aragora.server.handlers.email.vip"


def _mock_store() -> MagicMock:
    """Build a mock email store with VIP methods."""
    store = MagicMock()
    store.add_vip_sender = MagicMock()
    store.remove_vip_sender = MagicMock()
    return store


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def _reset_module_globals():
    """Reset module-level mutable state between tests."""
    orig_prioritizer = storage_module._prioritizer
    orig_user_configs = storage_module._user_configs.copy()

    yield

    storage_module._prioritizer = orig_prioritizer
    storage_module._user_configs.clear()
    storage_module._user_configs.update(orig_user_configs)


@pytest.fixture
def mock_store():
    """Provide a mock email store."""
    store = _mock_store()
    with patch(f"{STORAGE}.get_email_store", return_value=store):
        yield store


@pytest.fixture
def no_store():
    """Provide None as email store (store unavailable)."""
    with patch(f"{STORAGE}.get_email_store", return_value=None):
        yield


@pytest.fixture
def empty_config():
    """Patch _load_config_from_store to return empty dict and _save_config_to_store to no-op."""
    with (
        patch(f"{STORAGE}._load_config_from_store", return_value={}),
        patch(f"{STORAGE}._save_config_to_store") as save_mock,
    ):
        yield save_mock


# ============================================================================
# handle_add_vip - Basic Success Paths
# ============================================================================


class TestAddVipEmail:
    """Tests for adding a VIP email address."""

    @pytest.mark.asyncio
    async def test_add_email_success(self, mock_store, empty_config):
        result = await handle_add_vip(user_id="u1", email="vip@example.com", workspace_id="ws1")
        assert result["success"] is True
        assert result["added"]["email"] == "vip@example.com"
        assert result["added"]["domain"] is None
        assert "vip@example.com" in result["vip_addresses"]
        assert result["vip_domains"] == []

    @pytest.mark.asyncio
    async def test_add_email_calls_store(self, mock_store, empty_config):
        await handle_add_vip(user_id="u1", email="vip@example.com", workspace_id="ws1")
        mock_store.add_vip_sender.assert_called_once_with("u1", "ws1", "vip@example.com")

    @pytest.mark.asyncio
    async def test_add_email_persists_config(self, mock_store, empty_config):
        await handle_add_vip(user_id="u1", email="vip@example.com", workspace_id="ws1")
        empty_config.assert_called_once()
        args = empty_config.call_args
        assert args[0][0] == "u1"
        assert "vip@example.com" in args[0][1].get("vip_addresses", [])

    @pytest.mark.asyncio
    async def test_add_email_resets_prioritizer(self, mock_store, empty_config):
        storage_module._prioritizer = MagicMock()
        await handle_add_vip(user_id="u1", email="vip@test.com")
        assert storage_module._prioritizer is None

    @pytest.mark.asyncio
    async def test_add_duplicate_email_no_duplicate(self, mock_store, empty_config):
        """Adding an email already in the list should not create a duplicate."""
        storage_module._user_configs["u1"] = {
            "vip_addresses": ["vip@example.com"],
        }
        with (
            patch(f"{STORAGE}._load_config_from_store") as load_mock,
            patch(f"{STORAGE}._save_config_to_store"),
        ):
            result = await handle_add_vip(user_id="u1", email="vip@example.com")
        assert result["success"] is True
        assert result["vip_addresses"].count("vip@example.com") == 1


class TestAddVipDomain:
    """Tests for adding a VIP domain."""

    @pytest.mark.asyncio
    async def test_add_domain_success(self, mock_store, empty_config):
        result = await handle_add_vip(user_id="u1", domain="important.com", workspace_id="ws1")
        assert result["success"] is True
        assert result["added"]["domain"] == "important.com"
        assert result["added"]["email"] is None
        assert "important.com" in result["vip_domains"]
        assert result["vip_addresses"] == []

    @pytest.mark.asyncio
    async def test_add_duplicate_domain_no_duplicate(self, mock_store, empty_config):
        storage_module._user_configs["u1"] = {
            "vip_domains": ["important.com"],
        }
        with (
            patch(f"{STORAGE}._load_config_from_store"),
            patch(f"{STORAGE}._save_config_to_store"),
        ):
            result = await handle_add_vip(user_id="u1", domain="important.com")
        assert result["success"] is True
        assert result["vip_domains"].count("important.com") == 1

    @pytest.mark.asyncio
    async def test_add_domain_does_not_call_add_vip_sender(self, mock_store, empty_config):
        """Domain additions should not call store.add_vip_sender (only emails do)."""
        await handle_add_vip(user_id="u1", domain="important.com")
        mock_store.add_vip_sender.assert_not_called()


class TestAddVipBoth:
    """Tests for adding both email and domain at once."""

    @pytest.mark.asyncio
    async def test_add_both_email_and_domain(self, mock_store, empty_config):
        result = await handle_add_vip(
            user_id="u1",
            email="ceo@corp.com",
            domain="corp.com",
        )
        assert result["success"] is True
        assert result["added"]["email"] == "ceo@corp.com"
        assert result["added"]["domain"] == "corp.com"
        assert "ceo@corp.com" in result["vip_addresses"]
        assert "corp.com" in result["vip_domains"]


class TestAddVipNoEmailNoDomain:
    """Tests when neither email nor domain is provided."""

    @pytest.mark.asyncio
    async def test_add_nothing(self, mock_store, empty_config):
        result = await handle_add_vip(user_id="u1")
        assert result["success"] is True
        assert result["added"]["email"] is None
        assert result["added"]["domain"] is None
        assert result["vip_addresses"] == []
        assert result["vip_domains"] == []


# ============================================================================
# handle_add_vip - Config Loading / Cache Miss
# ============================================================================


class TestAddVipCacheMiss:
    """Tests that config is loaded from store on cache miss."""

    @pytest.mark.asyncio
    async def test_cache_miss_loads_from_store(self, mock_store):
        with (
            patch(
                f"{STORAGE}._load_config_from_store",
                return_value={"vip_addresses": ["existing@test.com"]},
            ),
            patch(f"{STORAGE}._save_config_to_store"),
        ):
            result = await handle_add_vip(
                user_id="new_user", email="new@test.com", workspace_id="ws1"
            )
        assert result["success"] is True
        assert "existing@test.com" in result["vip_addresses"]
        assert "new@test.com" in result["vip_addresses"]

    @pytest.mark.asyncio
    async def test_cache_hit_skips_load(self, mock_store):
        """When config is already cached, _load_config_from_store is NOT called."""
        storage_module._user_configs["u1"] = {"vip_addresses": ["old@test.com"]}
        with (
            patch(f"{STORAGE}._load_config_from_store") as load_mock,
            patch(f"{STORAGE}._save_config_to_store"),
        ):
            result = await handle_add_vip(user_id="u1", email="new@test.com")
        load_mock.assert_not_called()
        assert result["success"] is True


# ============================================================================
# handle_add_vip - Store Failures
# ============================================================================


class TestAddVipStoreFailures:
    """Tests for store-level failures during add operations."""

    @pytest.mark.asyncio
    async def test_store_add_vip_sender_keyerror_graceful(self, empty_config):
        store = _mock_store()
        store.add_vip_sender.side_effect = KeyError("missing key")
        with patch(f"{STORAGE}.get_email_store", return_value=store):
            result = await handle_add_vip(user_id="u1", email="vip@test.com")
        # Should succeed despite store failure (graceful degradation)
        assert result["success"] is True
        assert "vip@test.com" in result["vip_addresses"]

    @pytest.mark.asyncio
    async def test_store_add_vip_sender_oserror_graceful(self, empty_config):
        store = _mock_store()
        store.add_vip_sender.side_effect = OSError("disk full")
        with patch(f"{STORAGE}.get_email_store", return_value=store):
            result = await handle_add_vip(user_id="u1", email="vip@test.com")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_store_add_vip_sender_valueerror_graceful(self, empty_config):
        store = _mock_store()
        store.add_vip_sender.side_effect = ValueError("bad value")
        with patch(f"{STORAGE}.get_email_store", return_value=store):
            result = await handle_add_vip(user_id="u1", email="vip@test.com")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_store_add_vip_sender_typeerror_graceful(self, empty_config):
        store = _mock_store()
        store.add_vip_sender.side_effect = TypeError("type mismatch")
        with patch(f"{STORAGE}.get_email_store", return_value=store):
            result = await handle_add_vip(user_id="u1", email="vip@test.com")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_store_unavailable_still_succeeds(self, no_store, empty_config):
        result = await handle_add_vip(user_id="u1", email="vip@test.com")
        assert result["success"] is True
        assert "vip@test.com" in result["vip_addresses"]


# ============================================================================
# handle_add_vip - Outer Exception Path
# ============================================================================


class TestAddVipOuterException:
    """Tests for the outer try/except that catches config-level errors."""

    @pytest.mark.asyncio
    async def test_save_config_raises_keyerror(self):
        with (
            patch(f"{STORAGE}._load_config_from_store", return_value={}),
            patch(f"{STORAGE}._save_config_to_store", side_effect=KeyError("boom")),
            patch(f"{STORAGE}.get_email_store", return_value=None),
        ):
            result = await handle_add_vip(user_id="u1", email="test@test.com")
        assert result["success"] is False
        assert "Failed to add VIP" in result["error"]

    @pytest.mark.asyncio
    async def test_save_config_raises_oserror(self):
        with (
            patch(f"{STORAGE}._load_config_from_store", return_value={}),
            patch(f"{STORAGE}._save_config_to_store", side_effect=OSError("io error")),
            patch(f"{STORAGE}.get_email_store", return_value=None),
        ):
            result = await handle_add_vip(user_id="u1", email="test@test.com")
        assert result["success"] is False
        assert "Failed to add VIP" in result["error"]

    @pytest.mark.asyncio
    async def test_save_config_raises_valueerror(self):
        with (
            patch(f"{STORAGE}._load_config_from_store", return_value={}),
            patch(f"{STORAGE}._save_config_to_store", side_effect=ValueError("bad")),
            patch(f"{STORAGE}.get_email_store", return_value=None),
        ):
            result = await handle_add_vip(user_id="u1", email="test@test.com")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_save_config_raises_typeerror(self):
        with (
            patch(f"{STORAGE}._load_config_from_store", return_value={}),
            patch(f"{STORAGE}._save_config_to_store", side_effect=TypeError("bad type")),
            patch(f"{STORAGE}.get_email_store", return_value=None),
        ):
            result = await handle_add_vip(user_id="u1", email="test@test.com")
        assert result["success"] is False


# ============================================================================
# handle_add_vip - Auth Context
# ============================================================================


class TestAddVipAuthContext:
    """Tests for auth context handling in handle_add_vip."""

    @pytest.mark.asyncio
    async def test_auth_context_unset_skips_permission_check(self, mock_store, empty_config):
        """When auth_context is _AUTH_CONTEXT_UNSET, permission check is skipped."""
        with patch(f"{STORAGE}._check_email_permission") as check_mock:
            result = await handle_add_vip(
                user_id="u1",
                email="vip@test.com",
                auth_context=_AUTH_CONTEXT_UNSET,
            )
        check_mock.assert_not_called()
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_auth_context_provided_calls_check(self, mock_store, empty_config):
        """When auth_context is provided, permission check is called."""
        ctx = MagicMock()
        with patch(f"{STORAGE}._check_email_permission", return_value=None) as check_mock:
            result = await handle_add_vip(
                user_id="u1",
                email="vip@test.com",
                auth_context=ctx,
            )
        check_mock.assert_called_once_with(ctx, PERM_EMAIL_UPDATE)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_auth_context_denied_returns_error(self, mock_store, empty_config):
        """When permission check returns an error, the error is propagated."""
        ctx = MagicMock()
        perm_error = {"success": False, "error": "Permission denied"}
        with patch(f"{STORAGE}._check_email_permission", return_value=perm_error):
            result = await handle_add_vip(
                user_id="u1",
                email="vip@test.com",
                auth_context=ctx,
            )
        assert result["success"] is False
        assert result["error"] == "Permission denied"

    @pytest.mark.asyncio
    async def test_auth_context_none_calls_check(self, mock_store, empty_config):
        """When auth_context is explicitly None (not UNSET), still runs permission check."""
        with patch(f"{STORAGE}._check_email_permission", return_value=None) as check_mock:
            result = await handle_add_vip(
                user_id="u1",
                email="vip@test.com",
                auth_context=None,
            )
        check_mock.assert_called_once_with(None, PERM_EMAIL_UPDATE)
        assert result["success"] is True


# ============================================================================
# handle_add_vip - Defaults
# ============================================================================


class TestAddVipDefaults:
    """Tests for default parameter values."""

    @pytest.mark.asyncio
    async def test_default_user_id(self, mock_store, empty_config):
        result = await handle_add_vip(email="vip@test.com")
        assert result["success"] is True
        # Check saved with default user_id
        args = empty_config.call_args
        assert args[0][0] == "default"

    @pytest.mark.asyncio
    async def test_default_workspace_id(self, mock_store, empty_config):
        result = await handle_add_vip(user_id="u1", email="vip@test.com")
        assert result["success"] is True
        args = empty_config.call_args
        assert args[0][2] == "default"


# ============================================================================
# handle_remove_vip - Basic Success Paths
# ============================================================================


class TestRemoveVipEmail:
    """Tests for removing a VIP email address."""

    @pytest.mark.asyncio
    async def test_remove_email_success(self, mock_store):
        storage_module._user_configs["u1"] = {
            "vip_addresses": ["vip@test.com", "other@test.com"],
        }
        with patch(f"{STORAGE}._save_config_to_store"):
            result = await handle_remove_vip(user_id="u1", email="vip@test.com", workspace_id="ws1")
        assert result["success"] is True
        assert result["removed"]["email"] == "vip@test.com"
        assert result["removed"]["domain"] is None
        assert "vip@test.com" not in result["vip_addresses"]
        assert "other@test.com" in result["vip_addresses"]

    @pytest.mark.asyncio
    async def test_remove_email_calls_store(self, mock_store):
        storage_module._user_configs["u1"] = {
            "vip_addresses": ["vip@test.com"],
        }
        with patch(f"{STORAGE}._save_config_to_store"):
            await handle_remove_vip(user_id="u1", email="vip@test.com", workspace_id="ws1")
        mock_store.remove_vip_sender.assert_called_once_with("u1", "ws1", "vip@test.com")

    @pytest.mark.asyncio
    async def test_remove_email_persists_config(self, mock_store):
        storage_module._user_configs["u1"] = {
            "vip_addresses": ["vip@test.com"],
        }
        with patch(f"{STORAGE}._save_config_to_store") as save_mock:
            await handle_remove_vip(user_id="u1", email="vip@test.com", workspace_id="ws1")
        save_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_email_resets_prioritizer(self, mock_store):
        storage_module._user_configs["u1"] = {
            "vip_addresses": ["vip@test.com"],
        }
        storage_module._prioritizer = MagicMock()
        with patch(f"{STORAGE}._save_config_to_store"):
            await handle_remove_vip(user_id="u1", email="vip@test.com")
        assert storage_module._prioritizer is None

    @pytest.mark.asyncio
    async def test_remove_nonexistent_email(self, mock_store):
        storage_module._user_configs["u1"] = {
            "vip_addresses": ["other@test.com"],
        }
        with patch(f"{STORAGE}._save_config_to_store"):
            result = await handle_remove_vip(user_id="u1", email="nonexistent@test.com")
        assert result["success"] is True
        assert result["removed"]["email"] is None
        mock_store.remove_vip_sender.assert_not_called()

    @pytest.mark.asyncio
    async def test_remove_email_no_vip_addresses_key(self, mock_store):
        """When config has no vip_addresses key at all."""
        storage_module._user_configs["u1"] = {}
        with patch(f"{STORAGE}._save_config_to_store"):
            result = await handle_remove_vip(user_id="u1", email="test@test.com")
        assert result["success"] is True
        assert result["removed"]["email"] is None


class TestRemoveVipDomain:
    """Tests for removing a VIP domain."""

    @pytest.mark.asyncio
    async def test_remove_domain_success(self, mock_store):
        storage_module._user_configs["u1"] = {
            "vip_domains": ["corp.com", "other.com"],
        }
        with patch(f"{STORAGE}._save_config_to_store"):
            result = await handle_remove_vip(user_id="u1", domain="corp.com", workspace_id="ws1")
        assert result["success"] is True
        assert result["removed"]["domain"] == "corp.com"
        assert result["removed"]["email"] is None
        assert "corp.com" not in result["vip_domains"]
        assert "other.com" in result["vip_domains"]

    @pytest.mark.asyncio
    async def test_remove_nonexistent_domain(self, mock_store):
        storage_module._user_configs["u1"] = {
            "vip_domains": ["other.com"],
        }
        with patch(f"{STORAGE}._save_config_to_store"):
            result = await handle_remove_vip(user_id="u1", domain="nonexistent.com")
        assert result["success"] is True
        assert result["removed"]["domain"] is None

    @pytest.mark.asyncio
    async def test_remove_domain_no_vip_domains_key(self, mock_store):
        """When config has no vip_domains key at all."""
        storage_module._user_configs["u1"] = {}
        with patch(f"{STORAGE}._save_config_to_store"):
            result = await handle_remove_vip(user_id="u1", domain="test.com")
        assert result["success"] is True
        assert result["removed"]["domain"] is None

    @pytest.mark.asyncio
    async def test_remove_domain_does_not_call_store_remove(self, mock_store):
        """Domain removals should not call store.remove_vip_sender."""
        storage_module._user_configs["u1"] = {
            "vip_domains": ["corp.com"],
        }
        with patch(f"{STORAGE}._save_config_to_store"):
            await handle_remove_vip(user_id="u1", domain="corp.com")
        mock_store.remove_vip_sender.assert_not_called()


class TestRemoveVipBoth:
    """Tests for removing both email and domain at once."""

    @pytest.mark.asyncio
    async def test_remove_both_email_and_domain(self, mock_store):
        storage_module._user_configs["u1"] = {
            "vip_addresses": ["ceo@corp.com"],
            "vip_domains": ["corp.com"],
        }
        with patch(f"{STORAGE}._save_config_to_store"):
            result = await handle_remove_vip(
                user_id="u1",
                email="ceo@corp.com",
                domain="corp.com",
            )
        assert result["success"] is True
        assert result["removed"]["email"] == "ceo@corp.com"
        assert result["removed"]["domain"] == "corp.com"
        assert result["vip_addresses"] == []
        assert result["vip_domains"] == []


class TestRemoveVipNoEmailNoDomain:
    """Tests when neither email nor domain is provided to remove."""

    @pytest.mark.asyncio
    async def test_remove_nothing(self, mock_store):
        storage_module._user_configs["u1"] = {
            "vip_addresses": ["keep@test.com"],
            "vip_domains": ["keep.com"],
        }
        with patch(f"{STORAGE}._save_config_to_store"):
            result = await handle_remove_vip(user_id="u1")
        assert result["success"] is True
        assert result["removed"]["email"] is None
        assert result["removed"]["domain"] is None
        assert "keep@test.com" in result["vip_addresses"]
        assert "keep.com" in result["vip_domains"]


# ============================================================================
# handle_remove_vip - Config Loading / Cache Miss
# ============================================================================


class TestRemoveVipCacheMiss:
    """Tests that config is loaded from store on cache miss."""

    @pytest.mark.asyncio
    async def test_cache_miss_loads_from_store(self, mock_store):
        with (
            patch(
                f"{STORAGE}._load_config_from_store",
                return_value={"vip_addresses": ["vip@test.com"]},
            ),
            patch(f"{STORAGE}._save_config_to_store"),
        ):
            result = await handle_remove_vip(user_id="new_user", email="vip@test.com")
        assert result["success"] is True
        assert result["removed"]["email"] == "vip@test.com"
        assert result["vip_addresses"] == []


# ============================================================================
# handle_remove_vip - Store Failures
# ============================================================================


class TestRemoveVipStoreFailures:
    """Tests for store-level failures during remove operations."""

    @pytest.mark.asyncio
    async def test_store_remove_vip_sender_keyerror_graceful(self):
        store = _mock_store()
        store.remove_vip_sender.side_effect = KeyError("missing")
        storage_module._user_configs["u1"] = {
            "vip_addresses": ["vip@test.com"],
        }
        with (
            patch(f"{STORAGE}.get_email_store", return_value=store),
            patch(f"{STORAGE}._save_config_to_store"),
        ):
            result = await handle_remove_vip(user_id="u1", email="vip@test.com")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_store_remove_vip_sender_oserror_graceful(self):
        store = _mock_store()
        store.remove_vip_sender.side_effect = OSError("disk")
        storage_module._user_configs["u1"] = {
            "vip_addresses": ["vip@test.com"],
        }
        with (
            patch(f"{STORAGE}.get_email_store", return_value=store),
            patch(f"{STORAGE}._save_config_to_store"),
        ):
            result = await handle_remove_vip(user_id="u1", email="vip@test.com")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_store_unavailable_remove_still_succeeds(self):
        storage_module._user_configs["u1"] = {
            "vip_addresses": ["vip@test.com"],
        }
        with (
            patch(f"{STORAGE}.get_email_store", return_value=None),
            patch(f"{STORAGE}._save_config_to_store"),
        ):
            result = await handle_remove_vip(user_id="u1", email="vip@test.com")
        assert result["success"] is True
        assert result["removed"]["email"] == "vip@test.com"


# ============================================================================
# handle_remove_vip - Outer Exception Path
# ============================================================================


class TestRemoveVipOuterException:
    """Tests for the outer try/except in handle_remove_vip."""

    @pytest.mark.asyncio
    async def test_save_config_raises_keyerror(self):
        storage_module._user_configs["u1"] = {
            "vip_addresses": ["vip@test.com"],
        }
        with (
            patch(f"{STORAGE}._save_config_to_store", side_effect=KeyError("boom")),
            patch(f"{STORAGE}.get_email_store", return_value=None),
        ):
            result = await handle_remove_vip(user_id="u1", email="vip@test.com")
        assert result["success"] is False
        assert "Failed to remove VIP" in result["error"]

    @pytest.mark.asyncio
    async def test_save_config_raises_oserror(self):
        storage_module._user_configs["u1"] = {}
        with (
            patch(f"{STORAGE}._save_config_to_store", side_effect=OSError("io")),
            patch(f"{STORAGE}.get_email_store", return_value=None),
        ):
            result = await handle_remove_vip(user_id="u1", email="test@test.com")
        assert result["success"] is False
        assert "Failed to remove VIP" in result["error"]

    @pytest.mark.asyncio
    async def test_save_config_raises_typeerror(self):
        storage_module._user_configs["u1"] = {}
        with (
            patch(f"{STORAGE}._save_config_to_store", side_effect=TypeError("nope")),
            patch(f"{STORAGE}.get_email_store", return_value=None),
        ):
            result = await handle_remove_vip(user_id="u1")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_save_config_raises_valueerror(self):
        storage_module._user_configs["u1"] = {}
        with (
            patch(f"{STORAGE}._save_config_to_store", side_effect=ValueError("bad")),
            patch(f"{STORAGE}.get_email_store", return_value=None),
        ):
            result = await handle_remove_vip(user_id="u1")
        assert result["success"] is False


# ============================================================================
# handle_remove_vip - Auth Context
# ============================================================================


class TestRemoveVipAuthContext:
    """Tests for auth context handling in handle_remove_vip."""

    @pytest.mark.asyncio
    async def test_auth_context_unset_skips_permission_check(self, mock_store):
        storage_module._user_configs["u1"] = {}
        with (
            patch(f"{STORAGE}._check_email_permission") as check_mock,
            patch(f"{STORAGE}._save_config_to_store"),
        ):
            result = await handle_remove_vip(
                user_id="u1",
                auth_context=_AUTH_CONTEXT_UNSET,
            )
        check_mock.assert_not_called()
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_auth_context_provided_calls_check(self, mock_store):
        ctx = MagicMock()
        storage_module._user_configs["u1"] = {}
        with (
            patch(f"{STORAGE}._check_email_permission", return_value=None) as check_mock,
            patch(f"{STORAGE}._save_config_to_store"),
        ):
            result = await handle_remove_vip(
                user_id="u1",
                auth_context=ctx,
            )
        check_mock.assert_called_once_with(ctx, PERM_EMAIL_UPDATE)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_auth_context_denied_returns_error(self, mock_store):
        ctx = MagicMock()
        perm_error = {"success": False, "error": "Permission denied"}
        with patch(f"{STORAGE}._check_email_permission", return_value=perm_error):
            result = await handle_remove_vip(
                user_id="u1",
                email="vip@test.com",
                auth_context=ctx,
            )
        assert result["success"] is False
        assert result["error"] == "Permission denied"


# ============================================================================
# Constants and Module-Level
# ============================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_perm_email_read_value(self):
        assert PERM_EMAIL_READ == "email:read"

    def test_perm_email_update_value(self):
        assert PERM_EMAIL_UPDATE == "email:update"

    def test_auth_context_unset_is_unique_object(self):
        """_AUTH_CONTEXT_UNSET should be a unique sentinel, not None or other falsy."""
        assert _AUTH_CONTEXT_UNSET is not None
        assert _AUTH_CONTEXT_UNSET is not False
        assert _AUTH_CONTEXT_UNSET != 0

    def test_auth_context_unset_identity(self):
        """The sentinel should be a specific object type (used for `is` comparison)."""
        assert type(_AUTH_CONTEXT_UNSET) is object


# ============================================================================
# Integration-style: sequential add+remove
# ============================================================================


class TestAddThenRemove:
    """Integration-style tests combining add and remove operations."""

    @pytest.mark.asyncio
    async def test_add_then_remove_email(self, mock_store, empty_config):
        # Add
        result = await handle_add_vip(user_id="u1", email="vip@test.com")
        assert result["success"] is True
        assert "vip@test.com" in result["vip_addresses"]

        # Remove (u1 is now cached)
        with patch(f"{STORAGE}._save_config_to_store"):
            result = await handle_remove_vip(user_id="u1", email="vip@test.com")
        assert result["success"] is True
        assert result["removed"]["email"] == "vip@test.com"
        assert result["vip_addresses"] == []

    @pytest.mark.asyncio
    async def test_add_then_remove_domain(self, mock_store, empty_config):
        result = await handle_add_vip(user_id="u1", domain="corp.com")
        assert "corp.com" in result["vip_domains"]

        with patch(f"{STORAGE}._save_config_to_store"):
            result = await handle_remove_vip(user_id="u1", domain="corp.com")
        assert result["removed"]["domain"] == "corp.com"
        assert result["vip_domains"] == []


# ============================================================================
# Return shape validation
# ============================================================================


class TestReturnShape:
    """Validate that all return dicts have the expected keys."""

    @pytest.mark.asyncio
    async def test_add_success_shape(self, mock_store, empty_config):
        result = await handle_add_vip(user_id="u1", email="a@b.com")
        assert set(result.keys()) == {"success", "added", "vip_addresses", "vip_domains"}
        assert set(result["added"].keys()) == {"email", "domain"}

    @pytest.mark.asyncio
    async def test_add_error_shape(self):
        with (
            patch(f"{STORAGE}._load_config_from_store", side_effect=KeyError("x")),
            patch(f"{STORAGE}.get_email_store", return_value=None),
        ):
            result = await handle_add_vip(user_id="u1", email="a@b.com")
        assert set(result.keys()) == {"success", "error"}

    @pytest.mark.asyncio
    async def test_remove_success_shape(self, mock_store):
        storage_module._user_configs["u1"] = {}
        with patch(f"{STORAGE}._save_config_to_store"):
            result = await handle_remove_vip(user_id="u1")
        assert set(result.keys()) == {"success", "removed", "vip_addresses", "vip_domains"}
        assert set(result["removed"].keys()) == {"email", "domain"}

    @pytest.mark.asyncio
    async def test_remove_error_shape(self):
        storage_module._user_configs["u1"] = {}
        with (
            patch(f"{STORAGE}._save_config_to_store", side_effect=ValueError("x")),
            patch(f"{STORAGE}.get_email_store", return_value=None),
        ):
            result = await handle_remove_vip(user_id="u1")
        assert set(result.keys()) == {"success", "error"}
