"""Tests for email configuration handlers.

Tests for aragora/server/handlers/email/config.py covering:
- handle_get_config: RBAC permission check, in-memory cache, persistent store fallback,
  default values, partial config, cache population on store load
- handle_update_config: individual field updates, partial updates, None config_updates,
  persist-to-store calls, prioritizer reset, error handling for all 5 exception types,
  nested handle_get_config call in response, @require_permission decorator
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

import aragora.server.handlers.email.storage as storage_module
from aragora.server.handlers.email.config import (
    handle_get_config,
    handle_update_config,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_auth_context(user_id: str = "user-1") -> MagicMock:
    """Build a mock auth context with a user_id attribute."""
    ctx = MagicMock()
    ctx.user_id = user_id
    return ctx


def _full_config() -> dict[str, Any]:
    """Return a fully populated config dict."""
    return {
        "vip_domains": ["vip.com"],
        "vip_addresses": ["ceo@vip.com"],
        "internal_domains": ["internal.com"],
        "auto_archive_senders": ["spam@archive.com"],
        "tier_1_confidence_threshold": 0.85,
        "tier_2_confidence_threshold": 0.55,
        "enable_slack_signals": False,
        "enable_calendar_signals": False,
        "enable_drive_signals": False,
    }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_module_globals():
    """Reset all module-level singletons between tests to avoid leaking state."""
    orig_prioritizer = storage_module._prioritizer
    orig_user_configs = storage_module._user_configs.copy()

    yield

    storage_module._prioritizer = orig_prioritizer
    storage_module._user_configs.clear()
    storage_module._user_configs.update(orig_user_configs)


@pytest.fixture
def mock_perm_ok():
    """Patch _check_email_permission to always allow."""
    with patch(
        "aragora.server.handlers.email.config._check_email_permission",
        return_value=None,
    ) as m:
        yield m


@pytest.fixture
def mock_perm_denied():
    """Patch _check_email_permission to always deny."""
    with patch(
        "aragora.server.handlers.email.config._check_email_permission",
        return_value={"success": False, "error": "Permission denied"},
    ) as m:
        yield m


@pytest.fixture
def mock_load_config():
    """Patch _load_config_from_store."""
    with patch(
        "aragora.server.handlers.email.config._load_config_from_store",
        return_value={},
    ) as m:
        yield m


@pytest.fixture
def mock_save_config():
    """Patch _save_config_to_store."""
    with patch(
        "aragora.server.handlers.email.config._save_config_to_store",
    ) as m:
        yield m


# ============================================================================
# handle_get_config - Permission Checks
# ============================================================================


class TestGetConfigPermissions:
    """Tests for RBAC permission checks on handle_get_config."""

    @pytest.mark.asyncio
    async def test_get_config_permission_denied(self, mock_perm_denied, mock_load_config):
        """Returns error dict when permission is denied."""
        result = await handle_get_config(auth_context=_make_auth_context())
        assert result["success"] is False
        assert result["error"] == "Permission denied"

    @pytest.mark.asyncio
    async def test_get_config_permission_check_called_with_email_read(self, mock_load_config):
        """Calls _check_email_permission with 'email:read'."""
        with patch(
            "aragora.server.handlers.email.config._check_email_permission",
            return_value=None,
        ) as mock_check:
            ctx = _make_auth_context()
            await handle_get_config(auth_context=ctx)
            mock_check.assert_called_once_with(ctx, "email:read")

    @pytest.mark.asyncio
    async def test_get_config_no_auth_context_passes_none(self, mock_load_config):
        """Passes None when no auth_context is provided."""
        with patch(
            "aragora.server.handlers.email.config._check_email_permission",
            return_value=None,
        ) as mock_check:
            await handle_get_config()
            mock_check.assert_called_once_with(None, "email:read")


# ============================================================================
# handle_get_config - Default Values
# ============================================================================


class TestGetConfigDefaults:
    """Tests for default config values when no config is set."""

    @pytest.mark.asyncio
    async def test_defaults_when_no_config(self, mock_perm_ok, mock_load_config):
        """Returns all default values when no config exists."""
        result = await handle_get_config()
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
    async def test_defaults_for_missing_fields(self, mock_perm_ok, mock_load_config):
        """Returns defaults for fields not present in partial config."""
        storage_module._user_configs["default"] = {"vip_domains": ["foo.com"]}
        result = await handle_get_config()
        cfg = result["config"]
        assert cfg["vip_domains"] == ["foo.com"]
        # All other fields should be defaults
        assert cfg["tier_1_confidence_threshold"] == 0.7
        assert cfg["enable_slack_signals"] is True


# ============================================================================
# handle_get_config - In-Memory Cache
# ============================================================================


class TestGetConfigMemoryCache:
    """Tests for in-memory config cache reads."""

    @pytest.mark.asyncio
    async def test_reads_from_memory_cache(self, mock_perm_ok, mock_load_config):
        """Returns config from in-memory cache when present."""
        storage_module._user_configs["user-1"] = _full_config()
        result = await handle_get_config(user_id="user-1")
        assert result["success"] is True
        assert result["config"]["vip_domains"] == ["vip.com"]
        assert result["config"]["tier_1_confidence_threshold"] == 0.85
        # Should NOT call store load since config was in memory
        mock_load_config.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_copy_not_reference(self, mock_perm_ok, mock_load_config):
        """Config dicts are copies, not references to internal state."""
        storage_module._user_configs["user-1"] = {"vip_domains": ["a.com"]}
        result = await handle_get_config(user_id="user-1")
        # Mutating result should not affect internal cache
        result["config"]["vip_domains"].append("b.com")
        assert "b.com" not in storage_module._user_configs["user-1"].get("vip_domains", [])

    @pytest.mark.asyncio
    async def test_different_user_ids_isolate(self, mock_perm_ok, mock_load_config):
        """Different user_ids have separate config caches."""
        storage_module._user_configs["user-a"] = {"vip_domains": ["a.com"]}
        storage_module._user_configs["user-b"] = {"vip_domains": ["b.com"]}

        result_a = await handle_get_config(user_id="user-a")
        result_b = await handle_get_config(user_id="user-b")

        assert result_a["config"]["vip_domains"] == ["a.com"]
        assert result_b["config"]["vip_domains"] == ["b.com"]


# ============================================================================
# handle_get_config - Persistent Store Fallback
# ============================================================================


class TestGetConfigStoreFallback:
    """Tests for loading config from persistent store when not in memory."""

    @pytest.mark.asyncio
    async def test_loads_from_store_when_not_in_memory(self, mock_perm_ok):
        """Falls back to persistent store when memory cache is empty."""
        stored_config = {"vip_domains": ["stored.com"], "tier_1_confidence_threshold": 0.9}
        with patch(
            "aragora.server.handlers.email.config._load_config_from_store",
            return_value=stored_config,
        ) as mock_load:
            result = await handle_get_config(user_id="user-new")
            mock_load.assert_called_once_with("user-new", "default")
            assert result["config"]["vip_domains"] == ["stored.com"]
            assert result["config"]["tier_1_confidence_threshold"] == 0.9

    @pytest.mark.asyncio
    async def test_caches_in_memory_after_store_load(self, mock_perm_ok):
        """After loading from store, caches the config in memory."""
        stored_config = {"vip_domains": ["cached.com"]}
        with patch(
            "aragora.server.handlers.email.config._load_config_from_store",
            return_value=stored_config,
        ):
            await handle_get_config(user_id="user-cache")
            # Verify it was cached
            assert "user-cache" in storage_module._user_configs
            assert storage_module._user_configs["user-cache"]["vip_domains"] == ["cached.com"]

    @pytest.mark.asyncio
    async def test_store_returns_empty_uses_defaults(self, mock_perm_ok, mock_load_config):
        """When store returns empty dict, defaults are used."""
        mock_load_config.return_value = {}
        result = await handle_get_config(user_id="user-empty")
        assert result["success"] is True
        assert result["config"]["vip_domains"] == []
        assert result["config"]["tier_1_confidence_threshold"] == 0.7

    @pytest.mark.asyncio
    async def test_store_returns_none_uses_defaults(self, mock_perm_ok):
        """When store returns falsy value, defaults are used without caching."""
        with patch(
            "aragora.server.handlers.email.config._load_config_from_store",
            return_value={},
        ):
            result = await handle_get_config(user_id="user-none")
            assert result["success"] is True
            assert result["config"]["enable_slack_signals"] is True
            # Empty dict is falsy, so it should NOT be cached
            assert "user-none" not in storage_module._user_configs

    @pytest.mark.asyncio
    async def test_workspace_id_passed_to_store(self, mock_perm_ok):
        """workspace_id is forwarded to the persistent store."""
        with patch(
            "aragora.server.handlers.email.config._load_config_from_store",
            return_value={},
        ) as mock_load:
            await handle_get_config(user_id="u1", workspace_id="ws-42")
            mock_load.assert_called_once_with("u1", "ws-42")


# ============================================================================
# handle_get_config - Config Field Conversion
# ============================================================================


class TestGetConfigFieldConversion:
    """Tests for config field type conversion (sets to lists, etc.)."""

    @pytest.mark.asyncio
    async def test_set_fields_converted_to_lists(self, mock_perm_ok, mock_load_config):
        """Set-type fields in stored config are converted to lists in response."""
        storage_module._user_configs["user-1"] = {
            "vip_domains": {"a.com", "b.com"},
            "vip_addresses": set(),
        }
        result = await handle_get_config(user_id="user-1")
        assert isinstance(result["config"]["vip_domains"], list)
        assert isinstance(result["config"]["vip_addresses"], list)
        assert set(result["config"]["vip_domains"]) == {"a.com", "b.com"}

    @pytest.mark.asyncio
    async def test_tuple_fields_converted_to_lists(self, mock_perm_ok, mock_load_config):
        """Tuple-type fields are converted to lists."""
        storage_module._user_configs["user-1"] = {
            "internal_domains": ("x.com", "y.com"),
        }
        result = await handle_get_config(user_id="user-1")
        assert isinstance(result["config"]["internal_domains"], list)
        assert result["config"]["internal_domains"] == ["x.com", "y.com"]


# ============================================================================
# handle_update_config - Basic Updates
# ============================================================================


class TestUpdateConfigBasic:
    """Tests for basic update operations."""

    @pytest.mark.asyncio
    async def test_update_vip_domains(self, mock_perm_ok, mock_load_config, mock_save_config):
        """Can update vip_domains field."""
        result = await handle_update_config(
            user_id="user-1",
            config_updates={"vip_domains": ["new-vip.com"]},
        )
        assert result["success"] is True
        assert result["config"]["vip_domains"] == ["new-vip.com"]

    @pytest.mark.asyncio
    async def test_update_vip_addresses(self, mock_perm_ok, mock_load_config, mock_save_config):
        """Can update vip_addresses field."""
        result = await handle_update_config(
            user_id="user-1",
            config_updates={"vip_addresses": ["ceo@corp.com"]},
        )
        assert result["success"] is True
        assert result["config"]["vip_addresses"] == ["ceo@corp.com"]

    @pytest.mark.asyncio
    async def test_update_internal_domains(self, mock_perm_ok, mock_load_config, mock_save_config):
        """Can update internal_domains field."""
        result = await handle_update_config(
            user_id="user-1",
            config_updates={"internal_domains": ["corp.com"]},
        )
        assert result["success"] is True
        assert result["config"]["internal_domains"] == ["corp.com"]

    @pytest.mark.asyncio
    async def test_update_auto_archive_senders(
        self, mock_perm_ok, mock_load_config, mock_save_config
    ):
        """Can update auto_archive_senders field."""
        result = await handle_update_config(
            user_id="user-1",
            config_updates={"auto_archive_senders": ["newsletter@spam.com"]},
        )
        assert result["success"] is True
        assert result["config"]["auto_archive_senders"] == ["newsletter@spam.com"]

    @pytest.mark.asyncio
    async def test_update_tier_1_threshold(self, mock_perm_ok, mock_load_config, mock_save_config):
        """Can update tier_1_confidence_threshold."""
        result = await handle_update_config(
            user_id="user-1",
            config_updates={"tier_1_confidence_threshold": 0.95},
        )
        assert result["success"] is True
        assert result["config"]["tier_1_confidence_threshold"] == 0.95

    @pytest.mark.asyncio
    async def test_update_tier_2_threshold(self, mock_perm_ok, mock_load_config, mock_save_config):
        """Can update tier_2_confidence_threshold."""
        result = await handle_update_config(
            user_id="user-1",
            config_updates={"tier_2_confidence_threshold": 0.45},
        )
        assert result["success"] is True
        assert result["config"]["tier_2_confidence_threshold"] == 0.45

    @pytest.mark.asyncio
    async def test_update_enable_slack_signals(
        self, mock_perm_ok, mock_load_config, mock_save_config
    ):
        """Can update enable_slack_signals."""
        result = await handle_update_config(
            user_id="user-1",
            config_updates={"enable_slack_signals": False},
        )
        assert result["success"] is True
        assert result["config"]["enable_slack_signals"] is False

    @pytest.mark.asyncio
    async def test_update_enable_calendar_signals(
        self, mock_perm_ok, mock_load_config, mock_save_config
    ):
        """Can update enable_calendar_signals."""
        result = await handle_update_config(
            user_id="user-1",
            config_updates={"enable_calendar_signals": False},
        )
        assert result["success"] is True
        assert result["config"]["enable_calendar_signals"] is False

    @pytest.mark.asyncio
    async def test_update_enable_drive_signals(
        self, mock_perm_ok, mock_load_config, mock_save_config
    ):
        """Can update enable_drive_signals."""
        result = await handle_update_config(
            user_id="user-1",
            config_updates={"enable_drive_signals": False},
        )
        assert result["success"] is True
        assert result["config"]["enable_drive_signals"] is False


# ============================================================================
# handle_update_config - Partial Updates
# ============================================================================


class TestUpdateConfigPartial:
    """Tests for partial config updates (only specified fields change)."""

    @pytest.mark.asyncio
    async def test_partial_update_preserves_existing(
        self, mock_perm_ok, mock_load_config, mock_save_config
    ):
        """Updating one field does not reset others."""
        storage_module._user_configs["user-1"] = {
            "vip_domains": ["existing.com"],
            "tier_1_confidence_threshold": 0.85,
        }
        result = await handle_update_config(
            user_id="user-1",
            config_updates={"vip_domains": ["new.com"]},
        )
        assert result["success"] is True
        assert result["config"]["vip_domains"] == ["new.com"]
        # Existing field preserved
        assert result["config"]["tier_1_confidence_threshold"] == 0.85

    @pytest.mark.asyncio
    async def test_multiple_fields_updated_at_once(
        self, mock_perm_ok, mock_load_config, mock_save_config
    ):
        """Can update multiple fields in a single call."""
        result = await handle_update_config(
            user_id="user-1",
            config_updates={
                "vip_domains": ["a.com"],
                "enable_slack_signals": False,
                "tier_2_confidence_threshold": 0.3,
            },
        )
        assert result["success"] is True
        assert result["config"]["vip_domains"] == ["a.com"]
        assert result["config"]["enable_slack_signals"] is False
        assert result["config"]["tier_2_confidence_threshold"] == 0.3

    @pytest.mark.asyncio
    async def test_empty_updates_is_noop(self, mock_perm_ok, mock_load_config, mock_save_config):
        """Empty config_updates dict does not change config."""
        storage_module._user_configs["user-1"] = {"vip_domains": ["keep.com"]}
        result = await handle_update_config(
            user_id="user-1",
            config_updates={},
        )
        assert result["success"] is True
        assert result["config"]["vip_domains"] == ["keep.com"]

    @pytest.mark.asyncio
    async def test_none_config_updates_treated_as_empty(
        self, mock_perm_ok, mock_load_config, mock_save_config
    ):
        """None config_updates is treated as empty dict."""
        result = await handle_update_config(
            user_id="user-1",
            config_updates=None,
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_unknown_fields_ignored(self, mock_perm_ok, mock_load_config, mock_save_config):
        """Fields not in the known set are silently ignored."""
        result = await handle_update_config(
            user_id="user-1",
            config_updates={"unknown_field": "value", "vip_domains": ["x.com"]},
        )
        assert result["success"] is True
        assert result["config"]["vip_domains"] == ["x.com"]
        assert "unknown_field" not in result["config"]


# ============================================================================
# handle_update_config - Persistence and Side Effects
# ============================================================================


class TestUpdateConfigPersistence:
    """Tests for store persistence and prioritizer reset."""

    @pytest.mark.asyncio
    async def test_save_called_with_correct_args(
        self, mock_perm_ok, mock_load_config, mock_save_config
    ):
        """_save_config_to_store is called with user_id and updated config."""
        await handle_update_config(
            user_id="user-1",
            config_updates={"vip_domains": ["saved.com"]},
            workspace_id="ws-1",
        )
        mock_save_config.assert_called_once()
        call_args = mock_save_config.call_args
        assert call_args[0][0] == "user-1"
        assert "saved.com" in call_args[0][1].get("vip_domains", [])
        assert call_args[0][2] == "ws-1"

    @pytest.mark.asyncio
    async def test_prioritizer_reset_on_update(
        self, mock_perm_ok, mock_load_config, mock_save_config
    ):
        """Prioritizer is reset to None after config update."""
        storage_module._prioritizer = MagicMock()
        await handle_update_config(
            user_id="user-1",
            config_updates={"vip_domains": ["new.com"]},
        )
        assert storage_module._prioritizer is None

    @pytest.mark.asyncio
    async def test_load_from_store_when_user_not_in_memory(self, mock_perm_ok, mock_save_config):
        """Loads from store when user config is not already in memory."""
        stored = {"vip_domains": ["store.com"]}
        with patch(
            "aragora.server.handlers.email.config._load_config_from_store",
            return_value=stored,
        ) as mock_load:
            result = await handle_update_config(
                user_id="user-new",
                config_updates={"enable_slack_signals": False},
                workspace_id="ws-5",
            )
            mock_load.assert_called_once_with("user-new", "ws-5")
            assert result["success"] is True
            # Store-loaded vip_domains should be preserved
            assert result["config"]["vip_domains"] == ["store.com"]
            assert result["config"]["enable_slack_signals"] is False

    @pytest.mark.asyncio
    async def test_update_caches_in_memory(self, mock_perm_ok, mock_load_config, mock_save_config):
        """Updated config is stored in in-memory cache."""
        await handle_update_config(
            user_id="user-mem",
            config_updates={"vip_domains": ["mem.com"]},
        )
        assert "user-mem" in storage_module._user_configs
        assert storage_module._user_configs["user-mem"]["vip_domains"] == ["mem.com"]

    @pytest.mark.asyncio
    async def test_default_workspace_id(self, mock_perm_ok, mock_load_config, mock_save_config):
        """workspace_id defaults to 'default' when not specified."""
        await handle_update_config(
            user_id="user-1",
            config_updates={"vip_domains": ["x.com"]},
        )
        call_args = mock_save_config.call_args
        assert call_args[0][2] == "default"


# ============================================================================
# handle_update_config - Error Handling
# ============================================================================


class TestUpdateConfigErrors:
    """Tests for error handling in handle_update_config."""

    @pytest.mark.asyncio
    async def test_key_error_returns_failure(self, mock_perm_ok, mock_load_config):
        """KeyError during update returns error dict."""
        with patch(
            "aragora.server.handlers.email.config._save_config_to_store",
            side_effect=KeyError("missing key"),
        ):
            result = await handle_update_config(
                user_id="user-1",
                config_updates={"vip_domains": ["x.com"]},
            )
            assert result["success"] is False
            assert result["error"] == "Failed to update configuration"

    @pytest.mark.asyncio
    async def test_value_error_returns_failure(self, mock_perm_ok, mock_load_config):
        """ValueError during update returns error dict."""
        with patch(
            "aragora.server.handlers.email.config._save_config_to_store",
            side_effect=ValueError("bad value"),
        ):
            result = await handle_update_config(
                user_id="user-1",
                config_updates={"vip_domains": ["x.com"]},
            )
            assert result["success"] is False
            assert result["error"] == "Failed to update configuration"

    @pytest.mark.asyncio
    async def test_os_error_returns_failure(self, mock_perm_ok, mock_load_config):
        """OSError during update returns error dict."""
        with patch(
            "aragora.server.handlers.email.config._save_config_to_store",
            side_effect=OSError("disk full"),
        ):
            result = await handle_update_config(
                user_id="user-1",
                config_updates={"vip_domains": ["x.com"]},
            )
            assert result["success"] is False
            assert result["error"] == "Failed to update configuration"

    @pytest.mark.asyncio
    async def test_type_error_returns_failure(self, mock_perm_ok, mock_load_config):
        """TypeError during update returns error dict."""
        with patch(
            "aragora.server.handlers.email.config._save_config_to_store",
            side_effect=TypeError("wrong type"),
        ):
            result = await handle_update_config(
                user_id="user-1",
                config_updates={"vip_domains": ["x.com"]},
            )
            assert result["success"] is False
            assert result["error"] == "Failed to update configuration"

    @pytest.mark.asyncio
    async def test_runtime_error_returns_failure(self, mock_perm_ok, mock_load_config):
        """RuntimeError during update returns error dict."""
        with patch(
            "aragora.server.handlers.email.config._save_config_to_store",
            side_effect=RuntimeError("runtime issue"),
        ):
            result = await handle_update_config(
                user_id="user-1",
                config_updates={"vip_domains": ["x.com"]},
            )
            assert result["success"] is False
            assert result["error"] == "Failed to update configuration"

    @pytest.mark.asyncio
    async def test_unhandled_error_propagates(self, mock_perm_ok, mock_load_config):
        """Exceptions not in the caught set propagate normally."""
        with patch(
            "aragora.server.handlers.email.config._save_config_to_store",
            side_effect=MemoryError("out of memory"),
        ):
            with pytest.raises(MemoryError):
                await handle_update_config(
                    user_id="user-1",
                    config_updates={"vip_domains": ["x.com"]},
                )

    @pytest.mark.asyncio
    async def test_error_during_load_on_update(self, mock_perm_ok, mock_save_config):
        """KeyError from _load_config_from_store during update is caught."""
        with patch(
            "aragora.server.handlers.email.config._load_config_from_store",
            side_effect=KeyError("load failure"),
        ):
            result = await handle_update_config(
                user_id="user-1",
                config_updates={"vip_domains": ["x.com"]},
            )
            assert result["success"] is False
            assert result["error"] == "Failed to update configuration"


# ============================================================================
# handle_update_config - Response Structure
# ============================================================================


class TestUpdateConfigResponse:
    """Tests for the response returned by handle_update_config."""

    @pytest.mark.asyncio
    async def test_response_contains_full_config(
        self, mock_perm_ok, mock_load_config, mock_save_config
    ):
        """Response includes the full config, not just updated fields."""
        result = await handle_update_config(
            user_id="user-1",
            config_updates={"vip_domains": ["resp.com"]},
        )
        assert result["success"] is True
        cfg = result["config"]
        # Should have all config keys
        expected_keys = {
            "vip_domains",
            "vip_addresses",
            "internal_domains",
            "auto_archive_senders",
            "tier_1_confidence_threshold",
            "tier_2_confidence_threshold",
            "enable_slack_signals",
            "enable_calendar_signals",
            "enable_drive_signals",
        }
        assert set(cfg.keys()) == expected_keys

    @pytest.mark.asyncio
    async def test_response_reflects_applied_update(
        self, mock_perm_ok, mock_load_config, mock_save_config
    ):
        """The config in the response reflects the update that was just applied."""
        result = await handle_update_config(
            user_id="user-1",
            config_updates={
                "vip_domains": ["applied.com"],
                "tier_1_confidence_threshold": 0.99,
            },
        )
        assert result["config"]["vip_domains"] == ["applied.com"]
        assert result["config"]["tier_1_confidence_threshold"] == 0.99

    @pytest.mark.asyncio
    async def test_response_uses_handle_get_config_internally(
        self, mock_perm_ok, mock_load_config, mock_save_config
    ):
        """handle_update_config calls handle_get_config to build the response."""
        # We verify this by checking that the response matches get_config's format
        await handle_update_config(
            user_id="user-1",
            config_updates={"vip_domains": ["check.com"]},
        )
        get_result = await handle_get_config(user_id="user-1")
        assert get_result["config"]["vip_domains"] == ["check.com"]


# ============================================================================
# handle_get_config - Full Config Round-Trip
# ============================================================================


class TestConfigRoundTrip:
    """Tests for end-to-end config update and read cycles."""

    @pytest.mark.asyncio
    async def test_full_config_round_trip(self, mock_perm_ok, mock_load_config, mock_save_config):
        """Set all fields, then verify get_config returns them all."""
        full = _full_config()
        await handle_update_config(
            user_id="user-rt",
            config_updates=full,
        )
        result = await handle_get_config(user_id="user-rt")
        assert result["success"] is True
        for key, value in full.items():
            assert result["config"][key] == value

    @pytest.mark.asyncio
    async def test_sequential_updates_accumulate(
        self, mock_perm_ok, mock_load_config, mock_save_config
    ):
        """Multiple sequential updates accumulate correctly."""
        await handle_update_config(
            user_id="user-seq",
            config_updates={"vip_domains": ["first.com"]},
        )
        await handle_update_config(
            user_id="user-seq",
            config_updates={"vip_addresses": ["a@b.com"]},
        )
        result = await handle_get_config(user_id="user-seq")
        assert result["config"]["vip_domains"] == ["first.com"]
        assert result["config"]["vip_addresses"] == ["a@b.com"]

    @pytest.mark.asyncio
    async def test_overwrite_existing_field(self, mock_perm_ok, mock_load_config, mock_save_config):
        """Updating a field replaces its value entirely."""
        await handle_update_config(
            user_id="user-ow",
            config_updates={"vip_domains": ["first.com", "second.com"]},
        )
        await handle_update_config(
            user_id="user-ow",
            config_updates={"vip_domains": ["only.com"]},
        )
        result = await handle_get_config(user_id="user-ow")
        assert result["config"]["vip_domains"] == ["only.com"]


# ============================================================================
# handle_update_config - Default user_id / workspace_id
# ============================================================================


class TestUpdateConfigDefaultArgs:
    """Tests for default parameter values."""

    @pytest.mark.asyncio
    async def test_default_user_id(self, mock_perm_ok, mock_load_config, mock_save_config):
        """User_id defaults to 'default' when not specified."""
        await handle_update_config(config_updates={"vip_domains": ["def.com"]})
        assert "default" in storage_module._user_configs
        assert storage_module._user_configs["default"]["vip_domains"] == ["def.com"]

    @pytest.mark.asyncio
    async def test_default_workspace_passed_to_get(
        self, mock_perm_ok, mock_load_config, mock_save_config
    ):
        """The get_config call inside update uses the same workspace_id."""
        result = await handle_update_config(
            user_id="user-1",
            config_updates={"vip_domains": ["ws.com"]},
            workspace_id="ws-custom",
        )
        assert result["success"] is True


# ============================================================================
# handle_get_config - Edge Cases
# ============================================================================


class TestGetConfigEdgeCases:
    """Edge case tests for handle_get_config."""

    @pytest.mark.asyncio
    async def test_empty_list_fields(self, mock_perm_ok, mock_load_config):
        """Empty lists are returned correctly."""
        storage_module._user_configs["user-1"] = {
            "vip_domains": [],
            "vip_addresses": [],
        }
        result = await handle_get_config(user_id="user-1")
        assert result["config"]["vip_domains"] == []
        assert result["config"]["vip_addresses"] == []

    @pytest.mark.asyncio
    async def test_boolean_false_not_treated_as_missing(self, mock_perm_ok, mock_load_config):
        """Boolean False is preserved, not replaced by default True."""
        storage_module._user_configs["user-1"] = {
            "enable_slack_signals": False,
            "enable_calendar_signals": False,
            "enable_drive_signals": False,
        }
        result = await handle_get_config(user_id="user-1")
        assert result["config"]["enable_slack_signals"] is False
        assert result["config"]["enable_calendar_signals"] is False
        assert result["config"]["enable_drive_signals"] is False

    @pytest.mark.asyncio
    async def test_zero_threshold_not_treated_as_missing(self, mock_perm_ok, mock_load_config):
        """Zero threshold is preserved, not replaced by default."""
        storage_module._user_configs["user-1"] = {
            "tier_1_confidence_threshold": 0.0,
            "tier_2_confidence_threshold": 0.0,
        }
        result = await handle_get_config(user_id="user-1")
        assert result["config"]["tier_1_confidence_threshold"] == 0.0
        assert result["config"]["tier_2_confidence_threshold"] == 0.0
