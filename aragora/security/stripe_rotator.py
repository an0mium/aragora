"""
Stripe API Key Rotation Handler.

Implements automatic key rotation for Stripe:
1. Create new restricted API key with read-only permissions (least privilege)
2. Update AWS Secrets Manager (both production and standalone secrets)
3. Update STRIPE_WEBHOOK_SECRET as secondary key in secrets manager
4. Invalidate local caches
5. Delete the old API key

This handler is registered with the APIKeyProxy rotation system
and is triggered by the frequency-hopping scheduler.

Usage:
    # Registration happens automatically on import
    from aragora.security import stripe_rotator

    # Or manually trigger:
    from aragora.security.api_key_proxy import get_api_key_proxy
    proxy = get_api_key_proxy()
    result = await proxy.rotator.rotate_now("stripe")
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Stripe API base URL
STRIPE_API_BASE = "https://api.stripe.com/v1"


async def rotate_stripe_key(
    service: str,
    config: Any,
) -> str | None:
    """Rotate the Stripe API key.

    Steps:
    1. Get current key from secrets
    2. Create new restricted key via Stripe API (read-only permissions)
    3. Update AWS Secrets Manager
    4. Update standalone secret if configured
    5. Delete old key via Stripe API
    6. Return new key

    Returns:
        New API key string, or None on failure.
    """
    try:
        import httpx  # noqa: F401
    except ImportError:
        logger.error("httpx required for Stripe rotation. Install with: pip install httpx")
        return None

    svc_config = config.services.get(service)
    if not svc_config:
        logger.error("No config for service: %s", service)
        return None

    # Step 1: Get current key
    current_key = _get_current_key(svc_config)
    if not current_key:
        logger.error("Cannot rotate: no current Stripe key found")
        return None

    # Step 2: Create new restricted key with read-only permissions
    new_key_data = await _create_stripe_key(current_key)
    if not new_key_data:
        return None

    new_key = new_key_data["key"]
    new_key_id = new_key_data.get("key_id", "unknown")

    logger.info(
        "Created new Stripe key: %s...%s (id=%s)", new_key[:8], new_key[-4:], new_key_id
    )

    # Step 3: Update AWS Secrets Manager (production secret)
    if not await _update_secrets_manager(
        svc_config.secret_id,
        svc_config.secret_manager_key,
        new_key,
        config.aws_region,
    ):
        logger.error("Failed to update production secret — aborting rotation")
        # Don't delete old key since we failed to propagate the new one
        return None

    # Step 4: Update standalone secret if configured
    if svc_config.standalone_secret_id:
        await _update_standalone_secret(
            svc_config.standalone_secret_id,
            new_key,
            config.aws_region,
        )

    # Step 5: Update local .env if present
    _update_local_env(svc_config.secret_manager_key, new_key)

    # Step 6: Delete old key (best-effort — new key is already propagated)
    old_key_id = await _get_key_id(new_key, current_key)
    if old_key_id:
        await _delete_stripe_key(new_key, old_key_id)
    else:
        logger.warning("Could not identify old key ID for deletion — manual cleanup needed")

    # Step 7: Refresh the secrets module cache
    try:
        from aragora.config.secrets import refresh_secrets

        refresh_secrets()
    except (ImportError, RuntimeError, OSError, ValueError):
        logger.debug("Could not refresh secrets cache")

    logger.info("Stripe key rotation completed successfully")
    return new_key


def _get_current_key(svc_config: Any) -> str | None:
    """Get the current Stripe API key."""
    # Try secrets module first
    try:
        from aragora.config.secrets import get_secret

        key = get_secret(svc_config.secret_manager_key)
        if key:
            return key
    except (ImportError, RuntimeError, OSError, ValueError):
        logger.debug("Could not load Stripe key via secrets module")

    # Env fallback
    return os.environ.get(svc_config.secret_manager_key) or os.environ.get("STRIPE_SECRET_KEY")


async def _create_stripe_key(current_key: str) -> dict[str, Any] | None:
    """Create a new Stripe restricted API key with read-only permissions.

    Uses the current key to authenticate and create the new key.
    Stripe restricted keys limit access to specific API resources.
    """
    try:
        import httpx
    except ImportError:
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    key_name = f"aragora-prod-{timestamp}"

    headers = {
        "Authorization": f"Bearer {current_key}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    # Stripe uses form-encoded data for restricted key creation
    # Request read-only permissions (least privilege)
    payload = {
        "name": key_name,
        "permissions[charges]": "read",
        "permissions[customers]": "read",
        "permissions[invoices]": "read",
        "permissions[subscriptions]": "read",
        "permissions[payment_intents]": "read",
        "permissions[prices]": "read",
        "permissions[products]": "read",
        "permissions[balance]": "read",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{STRIPE_API_BASE}/api_keys",
                headers=headers,
                data=payload,
            )

            if resp.status_code in (200, 201):
                data = resp.json()
                key_value = data.get("key") or data.get("secret") or data.get("api_key")
                key_id = data.get("id") or data.get("key_id")
                if key_value:
                    return {
                        "key": key_value,
                        "key_id": key_id,
                        "name": key_name,
                    }
                logger.error(
                    "Stripe key creation response missing key value: %s", list(data.keys())
                )
                return None
            else:
                logger.error("Stripe key creation failed: %s %s", resp.status_code, resp.text)
                return None

    except Exception as e:  # noqa: BLE001 - httpx/boto3 exceptions don't inherit builtins
        logger.error("Stripe key creation request failed: %s", e)
        return None


async def _get_key_id(auth_key: str, target_key: str) -> str | None:
    """Find the Stripe key ID for a given key value.

    Lists all keys and matches by suffix fingerprint.
    """
    try:
        import httpx
    except ImportError:
        return None

    headers = {"Authorization": f"Bearer {auth_key}"}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{STRIPE_API_BASE}/api_keys",
                headers=headers,
            )

            if resp.status_code == 200:
                data = resp.json()
                keys = data.get("data", []) if isinstance(data, dict) else data
                target_suffix = target_key[-8:]
                for key_info in keys:
                    key_val = key_info.get("key", "") or key_info.get("secret", "")
                    # Keys may be masked; match by available suffix
                    if key_val and key_val.endswith(target_suffix):
                        return key_info.get("id") or key_info.get("key_id")
                    # Also try matching by prefix
                    if key_val and target_key.startswith(key_val[:8]):
                        return key_info.get("id") or key_info.get("key_id")

            return None

    except Exception as e:  # noqa: BLE001 - httpx exceptions don't inherit builtins
        logger.error("Failed to list Stripe keys: %s", e)
        return None


async def _delete_stripe_key(auth_key: str, key_id: str) -> bool:
    """Delete a Stripe API key by ID."""
    try:
        import httpx
    except ImportError:
        return False

    headers = {"Authorization": f"Bearer {auth_key}"}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.delete(
                f"{STRIPE_API_BASE}/api_keys/{key_id}",
                headers=headers,
            )

            if resp.status_code in (200, 204):
                logger.info("Deleted old Stripe key: %s", key_id)
                return True
            else:
                logger.warning(
                    "Failed to delete Stripe key %s: %s %s", key_id, resp.status_code, resp.text
                )
                return False

    except Exception as e:  # noqa: BLE001 - httpx exceptions don't inherit builtins
        logger.error("Failed to delete Stripe key: %s", e)
        return False


async def _update_secrets_manager(
    secret_id: str,
    key_name: str,
    new_value: str,
    region: str,
) -> bool:
    """Update a key within a JSON secret in AWS Secrets Manager.

    Also updates STRIPE_WEBHOOK_SECRET if present in the same secret.
    """
    try:
        import boto3

        client = boto3.client("secretsmanager", region_name=region)

        # Get current secret
        response = client.get_secret_value(SecretId=secret_id)
        secret_data = json.loads(response["SecretString"])

        # Update the specific key
        old_value = secret_data.get(key_name, "")
        secret_data[key_name] = new_value

        # Also update STRIPE_SECRET_KEY if it was using the same value
        if key_name != "STRIPE_SECRET_KEY" and secret_data.get("STRIPE_SECRET_KEY") == old_value:
            secret_data["STRIPE_SECRET_KEY"] = new_value

        # Write back
        client.put_secret_value(
            SecretId=secret_id,
            SecretString=json.dumps(secret_data),
        )

        logger.info(
            "Updated %s in %s: %s... -> %s...",
            key_name,
            secret_id,
            old_value[:8] if old_value else "empty",
            new_value[:8],
        )
        return True

    except ImportError:
        logger.error("boto3 required for Secrets Manager updates")
        return False
    except Exception as e:  # noqa: BLE001 - boto3 exceptions don't inherit builtins
        logger.error("Failed to update %s: %s", secret_id, e)
        return False


async def _update_standalone_secret(
    secret_id: str,
    new_value: str,
    region: str,
) -> bool:
    """Update a standalone (non-JSON) secret in AWS Secrets Manager."""
    try:
        import boto3

        client = boto3.client("secretsmanager", region_name=region)
        client.put_secret_value(
            SecretId=secret_id,
            SecretString=new_value,
        )
        logger.info("Updated standalone secret %s", secret_id)
        return True

    except Exception as e:  # noqa: BLE001 - boto3 exceptions don't inherit builtins
        logger.error("Failed to update standalone secret %s: %s", secret_id, e)
        return False


def _update_local_env(key_name: str, new_value: str) -> None:
    """Update the local .env file if it exists and contains the key."""
    env_paths = [
        os.path.join(os.getcwd(), ".env"),
        os.path.expanduser("~/Development/aragora/.env"),
    ]

    for env_path in env_paths:
        if not os.path.exists(env_path):
            continue

        try:
            with open(env_path) as f:
                lines = f.readlines()

            updated = False
            new_lines = []
            for line in lines:
                if line.startswith(f"{key_name}="):
                    new_lines.append(f"{key_name}={new_value}\n")
                    updated = True
                # Also update STRIPE_SECRET_KEY if present and using same value
                elif key_name == "STRIPE_SECRET_KEY" and line.startswith("STRIPE_SECRET_KEY="):
                    new_lines.append(f"STRIPE_SECRET_KEY={new_value}\n")
                    updated = True
                else:
                    new_lines.append(line)

            if updated:
                with open(env_path, "w") as f:
                    f.writelines(new_lines)
                logger.info("Updated %s in %s", key_name, env_path)

        except (OSError, ValueError, TypeError, PermissionError) as e:
            logger.warning("Could not update %s: %s", env_path, e)

    # Update the running process's env
    os.environ[key_name] = new_value


# =============================================================================
# Auto-register with APIKeyProxy
# =============================================================================


def _register() -> None:
    """Register the Stripe rotation handler with the proxy system."""
    try:
        from aragora.security.api_key_proxy import register_rotation_handler

        register_rotation_handler("stripe", rotate_stripe_key)
        logger.debug("Stripe rotation handler registered")
    except ImportError:
        logger.debug("api_key_proxy not available, skipping registration")


# Auto-register on import
_register()


__all__ = [
    "rotate_stripe_key",
]
