"""
Email configuration handlers.

Provides handlers for:
- Getting email prioritization config
- Updating email prioritization config
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.server.handlers.base import require_permission

from .storage import (
    _check_email_permission,
    _load_config_from_store,
    _save_config_to_store,
    _user_configs,
    _user_configs_lock,
    _prioritizer_lock,
)

logger = logging.getLogger(__name__)

# Import the module-level _prioritizer for reset
import aragora.server.handlers.email.storage as storage_module


async def handle_get_config(
    user_id: str = "default",
    workspace_id: str = "default",
    auth_context: Any | None = None,
) -> dict[str, Any]:
    """
    Get email prioritization configuration.

    GET /api/email/config

    Now loads from persistent store with in-memory cache fallback.
    """
    # Check RBAC permission
    perm_error = _check_email_permission(auth_context, "email:read")
    if perm_error:
        return perm_error

    # Thread-safe read with snapshot
    with _user_configs_lock:
        config = _user_configs.get(user_id, {}).copy()

    # If not in memory, try loading from persistent store
    if not config:
        config = _load_config_from_store(user_id, workspace_id)
        if config:
            # Cache in memory
            with _user_configs_lock:
                _user_configs[user_id] = config.copy()

    return {
        "success": True,
        "config": {
            "vip_domains": list(config.get("vip_domains", [])),
            "vip_addresses": list(config.get("vip_addresses", [])),
            "internal_domains": list(config.get("internal_domains", [])),
            "auto_archive_senders": list(config.get("auto_archive_senders", [])),
            "tier_1_confidence_threshold": config.get("tier_1_confidence_threshold", 0.7),
            "tier_2_confidence_threshold": config.get("tier_2_confidence_threshold", 0.6),
            "enable_slack_signals": config.get("enable_slack_signals", True),
            "enable_calendar_signals": config.get("enable_calendar_signals", True),
            "enable_drive_signals": config.get("enable_drive_signals", True),
        },
    }


@require_permission("admin:system")
async def handle_update_config(
    user_id: str = "default",
    config_updates: dict[str, Any] = None,
    workspace_id: str = "default",
) -> dict[str, Any]:
    """
    Update email prioritization configuration.

    PUT /api/email/config
    {
        "vip_domains": ["importantclient.com"],
        "vip_addresses": ["ceo@company.com"],
        "internal_domains": ["mycompany.com"],
        "auto_archive_senders": ["newsletter@example.com"]
    }

    Now persists to SQLite for durability across restarts.
    """
    try:
        if config_updates is None:
            config_updates = {}

        # Thread-safe config update
        with _user_configs_lock:
            # Get or create user config (load from store if not in memory)
            if user_id not in _user_configs:
                _user_configs[user_id] = _load_config_from_store(user_id, workspace_id)

            # Update config
            user_config = _user_configs[user_id]

            if "vip_domains" in config_updates:
                user_config["vip_domains"] = config_updates["vip_domains"]
            if "vip_addresses" in config_updates:
                user_config["vip_addresses"] = config_updates["vip_addresses"]
            if "internal_domains" in config_updates:
                user_config["internal_domains"] = config_updates["internal_domains"]
            if "auto_archive_senders" in config_updates:
                user_config["auto_archive_senders"] = config_updates["auto_archive_senders"]
            if "tier_1_confidence_threshold" in config_updates:
                user_config["tier_1_confidence_threshold"] = config_updates[
                    "tier_1_confidence_threshold"
                ]
            if "tier_2_confidence_threshold" in config_updates:
                user_config["tier_2_confidence_threshold"] = config_updates[
                    "tier_2_confidence_threshold"
                ]
            if "enable_slack_signals" in config_updates:
                user_config["enable_slack_signals"] = config_updates["enable_slack_signals"]
            if "enable_calendar_signals" in config_updates:
                user_config["enable_calendar_signals"] = config_updates["enable_calendar_signals"]
            if "enable_drive_signals" in config_updates:
                user_config["enable_drive_signals"] = config_updates["enable_drive_signals"]

            # Persist to store
            _save_config_to_store(user_id, user_config, workspace_id)

        # Reset prioritizer to pick up new config (thread-safe)
        with _prioritizer_lock:
            storage_module._prioritizer = None

        return {
            "success": True,
            "config": (await handle_get_config(user_id, workspace_id))["config"],
        }

    except (KeyError, ValueError, OSError, TypeError, RuntimeError) as e:
        logger.exception(f"Failed to update config: {e}")
        return {
            "success": False,
            "error": "Failed to update configuration",
        }
