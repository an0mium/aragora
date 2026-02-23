"""
VIP management handlers.

Provides handlers for:
- Adding VIP emails/domains
- Removing VIP emails/domains
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.rbac.decorators import require_permission

from .storage import (
    _check_email_permission,
    _load_config_from_store,
    _save_config_to_store,
    _user_configs,
    _user_configs_lock,
    _prioritizer_lock,
    get_email_store,
)

logger = logging.getLogger(__name__)

# RBAC permission constants
PERM_EMAIL_READ = "email:read"
PERM_EMAIL_UPDATE = "email:update"

_AUTH_CONTEXT_UNSET = object()

# Import the module-level _prioritizer for reset
import aragora.server.handlers.email.storage as storage_module


@require_permission(PERM_EMAIL_UPDATE, context_param="auth_context")
async def handle_add_vip(
    user_id: str = "default",
    email: str | None = None,
    domain: str | None = None,
    workspace_id: str = "default",
    auth_context: Any | None = _AUTH_CONTEXT_UNSET,
) -> dict[str, Any]:
    """
    Add a VIP email or domain.

    POST /api/email/vip
    {
        "email": "important@example.com"
    }
    or
    {
        "domain": "importantcompany.com"
    }

    Now persists to SQLite for durability.
    """
    if auth_context is not _AUTH_CONTEXT_UNSET:
        perm_error = _check_email_permission(auth_context, PERM_EMAIL_UPDATE)
        if perm_error:
            return perm_error

    try:
        # Thread-safe config update
        with _user_configs_lock:
            if user_id not in _user_configs:
                _user_configs[user_id] = _load_config_from_store(user_id, workspace_id)

            config = _user_configs[user_id]

            if email:
                if "vip_addresses" not in config:
                    config["vip_addresses"] = []
                if email not in config["vip_addresses"]:
                    config["vip_addresses"].append(email)
                # Also add to dedicated VIP table for fast lookups
                store = get_email_store()
                if store:
                    try:
                        store.add_vip_sender(user_id, workspace_id, email)
                    except (KeyError, ValueError, OSError, TypeError) as e:
                        logger.debug("Failed to add VIP sender to store: %s", e)

            if domain:
                if "vip_domains" not in config:
                    config["vip_domains"] = []
                if domain not in config["vip_domains"]:
                    config["vip_domains"].append(domain)

            # Persist to store
            _save_config_to_store(user_id, config, workspace_id)

            result_addresses = list(config.get("vip_addresses", []))
            result_domains = list(config.get("vip_domains", []))

        # Reset prioritizer (thread-safe)
        with _prioritizer_lock:
            storage_module._prioritizer = None

        return {
            "success": True,
            "added": {"email": email, "domain": domain},
            "vip_addresses": result_addresses,
            "vip_domains": result_domains,
        }

    except (KeyError, ValueError, OSError, TypeError) as e:
        logger.exception("Failed to add VIP: %s", e)
        return {
            "success": False,
            "error": "Failed to add VIP",
        }


@require_permission(PERM_EMAIL_UPDATE, context_param="auth_context")
async def handle_remove_vip(
    user_id: str = "default",
    email: str | None = None,
    domain: str | None = None,
    workspace_id: str = "default",
    auth_context: Any | None = _AUTH_CONTEXT_UNSET,
) -> dict[str, Any]:
    """
    Remove a VIP email or domain.

    DELETE /api/email/vip
    {
        "email": "notimportant@example.com"
    }

    Now persists removal to SQLite.
    """
    if auth_context is not _AUTH_CONTEXT_UNSET:
        perm_error = _check_email_permission(auth_context, PERM_EMAIL_UPDATE)
        if perm_error:
            return perm_error

    try:
        # Thread-safe config update
        with _user_configs_lock:
            if user_id not in _user_configs:
                # Load from store first
                _user_configs[user_id] = _load_config_from_store(user_id, workspace_id)

            config = _user_configs[user_id]
            removed: dict[str, str | None] = {"email": None, "domain": None}

            if email and "vip_addresses" in config:
                if email in config["vip_addresses"]:
                    config["vip_addresses"].remove(email)
                    removed["email"] = email
                    # Also remove from dedicated VIP table
                    store = get_email_store()
                    if store:
                        try:
                            store.remove_vip_sender(user_id, workspace_id, email)
                        except (KeyError, ValueError, OSError, TypeError) as e:
                            logger.debug("Failed to remove VIP sender from store: %s", e)

            if domain and "vip_domains" in config:
                if domain in config["vip_domains"]:
                    config["vip_domains"].remove(domain)
                    removed["domain"] = domain

            # Persist to store
            _save_config_to_store(user_id, config, workspace_id)

            result_addresses = list(config.get("vip_addresses", []))
            result_domains = list(config.get("vip_domains", []))

        # Reset prioritizer (thread-safe)
        with _prioritizer_lock:
            storage_module._prioritizer = None

        return {
            "success": True,
            "removed": removed,
            "vip_addresses": result_addresses,
            "vip_domains": result_domains,
        }

    except (KeyError, ValueError, OSError, TypeError) as e:
        logger.exception("Failed to remove VIP: %s", e)
        return {
            "success": False,
            "error": "Failed to remove VIP",
        }
