"""
ElevenLabs API Key Rotation Handler.

Implements automatic key rotation for ElevenLabs:
1. Create new API key with TTS-only permissions (least privilege)
2. Update AWS Secrets Manager (both production and standalone secrets)
3. Invalidate local caches
4. Delete the old API key

This handler is registered with the APIKeyProxy rotation system
and is triggered by the frequency-hopping scheduler.

Usage:
    # Registration happens automatically on import
    from aragora.security import elevenlabs_rotator

    # Or manually trigger:
    from aragora.security.api_key_proxy import get_api_key_proxy
    proxy = get_api_key_proxy()
    result = await proxy.rotator.rotate_now("elevenlabs")
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ElevenLabs API base URL
ELEVENLABS_API_BASE = "https://api.elevenlabs.io"


async def rotate_elevenlabs_key(
    service: str,
    config: Any,
) -> str | None:
    """Rotate the ElevenLabs API key.

    Steps:
    1. Get current key from secrets
    2. Create new key via ElevenLabs API (TTS-only permissions)
    3. Update AWS Secrets Manager
    4. Update standalone secret if configured
    5. Delete old key via ElevenLabs API
    6. Return new key

    Returns:
        New API key string, or None on failure.
    """
    try:
        import httpx  # noqa: F401
    except ImportError:
        logger.error("httpx required for ElevenLabs rotation. Install with: pip install httpx")
        return None

    svc_config = config.services.get(service)
    if not svc_config:
        logger.error("No config for service: %s", service)
        return None

    # Step 1: Get current key
    current_key = _get_current_key(svc_config)
    if not current_key:
        logger.error("Cannot rotate: no current ElevenLabs key found")
        return None

    # Step 2: Create new key with TTS-only permissions
    new_key_data = await _create_elevenlabs_key(current_key)
    if not new_key_data:
        return None

    new_key = new_key_data["api_key"]
    new_key_id = new_key_data.get("xi_api_key_id", "unknown")

    logger.info(
        "Created new ElevenLabs key: %s...%s (id=%s)", new_key[:8], new_key[-4:], new_key_id
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
    # We use the NEW key to delete the old one
    old_key_id = await _get_key_id_by_value(new_key, current_key)
    if old_key_id:
        await _delete_elevenlabs_key(new_key, old_key_id)
    else:
        logger.warning("Could not identify old key ID for deletion — manual cleanup needed")

    # Step 7: Refresh the secrets module cache
    try:
        from aragora.config.secrets import refresh_secrets

        refresh_secrets()
    except (ImportError, RuntimeError, OSError, ValueError):
        logger.debug("Could not refresh secrets cache")

    logger.info("ElevenLabs key rotation completed successfully")
    return new_key


def _get_current_key(svc_config: Any) -> str | None:
    """Get the current ElevenLabs API key."""
    # Try secrets module first
    try:
        from aragora.config.secrets import get_secret

        key = get_secret(svc_config.secret_manager_key)
        if key:
            return key
    except (ImportError, RuntimeError, OSError, ValueError):
        logger.debug("Could not load ElevenLabs key via secrets module")

    # Env fallback
    return os.environ.get(svc_config.secret_manager_key)


async def _create_elevenlabs_key(current_key: str) -> dict[str, Any] | None:
    """Create a new ElevenLabs API key with TTS-only permissions.

    Uses the current key to authenticate and create the new key.
    """
    try:
        import httpx
    except ImportError:
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    key_name = f"aragora-prod-{timestamp}"

    # ElevenLabs API key creation endpoint
    # Note: This uses the undocumented /v1/user/api-keys endpoint
    # If this changes, fall back to the dashboard UI approach
    headers = {
        "xi-api-key": current_key,
        "Content-Type": "application/json",
    }

    payload = {
        "name": key_name,
        # Restrict to TTS only (least privilege)
        "permissions": {
            "text_to_speech": True,
            "speech_to_speech": False,
            "voice_design": False,
            "audio_isolation": False,
            "projects": False,
            "voice_library": False,
            "dubbing": False,
            "sound_generation": False,
            "pronunciation_dictionaries": False,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{ELEVENLABS_API_BASE}/v1/user/api-keys",
                headers=headers,
                json=payload,
            )

            if resp.status_code in (200, 201):
                data = resp.json()
                return {
                    "api_key": data.get("api_key_value") or data.get("xi_api_key"),
                    "xi_api_key_id": data.get("xi_api_key_id") or data.get("id"),
                    "name": key_name,
                }
            else:
                logger.error("ElevenLabs key creation failed: %s %s", resp.status_code, resp.text)
                return None

    except Exception as e:  # noqa: BLE001 - httpx/boto3 exceptions don't inherit builtins
        logger.error("ElevenLabs key creation request failed: %s", e)
        return None


async def _get_key_id_by_value(auth_key: str, target_key: str) -> str | None:
    """Find the ElevenLabs key ID for a given key value.

    Lists all keys and matches by prefix/suffix fingerprint.
    """
    try:
        import httpx
    except ImportError:
        return None

    headers = {"xi-api-key": auth_key}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{ELEVENLABS_API_BASE}/v1/user/api-keys",
                headers=headers,
            )

            if resp.status_code == 200:
                keys = resp.json()
                # Match by last 4 characters (keys are usually partially masked)
                target_suffix = target_key[-4:]
                for key_info in keys:
                    masked = key_info.get("api_key_value", "") or key_info.get("xi_api_key", "")
                    if masked.endswith(target_suffix):
                        return key_info.get("xi_api_key_id") or key_info.get("id")

            return None

    except Exception as e:  # noqa: BLE001 - httpx/boto3 exceptions don't inherit builtins
        logger.error("Failed to list ElevenLabs keys: %s", e)
        return None


async def _delete_elevenlabs_key(auth_key: str, key_id: str) -> bool:
    """Delete an ElevenLabs API key by ID."""
    try:
        import httpx
    except ImportError:
        return False

    headers = {"xi-api-key": auth_key}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.delete(
                f"{ELEVENLABS_API_BASE}/v1/user/api-keys/{key_id}",
                headers=headers,
            )

            if resp.status_code in (200, 204):
                logger.info("Deleted old ElevenLabs key: %s", key_id)
                return True
            else:
                logger.warning(
                    "Failed to delete ElevenLabs key %s: %s %s", key_id, resp.status_code, resp.text
                )
                return False

    except Exception as e:  # noqa: BLE001 - httpx/boto3 exceptions don't inherit builtins
        logger.error("Failed to delete ElevenLabs key: %s", e)
        return False


async def _update_secrets_manager(
    secret_id: str,
    key_name: str,
    new_value: str,
    region: str,
) -> bool:
    """Update a key within a JSON secret in AWS Secrets Manager."""
    try:
        import boto3

        client = boto3.client("secretsmanager", region_name=region)

        # Get current secret
        response = client.get_secret_value(SecretId=secret_id)
        secret_data = json.loads(response["SecretString"])

        # Update the specific key
        old_value = secret_data.get(key_name, "")
        secret_data[key_name] = new_value

        # Write back
        client.put_secret_value(
            SecretId=secret_id,
            SecretString=json.dumps(secret_data),
        )

        logger.info(
            "Updated %s in %s: %s... -> %s...", key_name, secret_id, old_value[:8], new_value[:8]
        )
        return True

    except ImportError:
        logger.error("boto3 required for Secrets Manager updates")
        return False
    except Exception as e:  # noqa: BLE001 - httpx/boto3 exceptions don't inherit builtins
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

    except Exception as e:  # noqa: BLE001 - httpx/boto3 exceptions don't inherit builtins
        logger.error("Failed to update standalone secret %s: %s", secret_id, e)
        return False


def _update_local_env(key_name: str, new_value: str) -> None:
    """Update the local .env file if it exists and contains the key."""
    # Find .env in common locations
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
                else:
                    new_lines.append(line)

            if updated:
                with open(env_path, "w") as f:
                    f.writelines(new_lines)
                logger.info("Updated %s in %s", key_name, env_path)

        except (OSError, ValueError, TypeError, PermissionError) as e:
            logger.warning("Could not update %s: %s", env_path, e)

    # Also update the running process's env
    os.environ[key_name] = new_value


# =============================================================================
# Auto-register with APIKeyProxy
# =============================================================================


def _register() -> None:
    """Register the ElevenLabs rotation handler with the proxy system."""
    try:
        from aragora.security.api_key_proxy import register_rotation_handler

        register_rotation_handler("elevenlabs", rotate_elevenlabs_key)
        logger.debug("ElevenLabs rotation handler registered")
    except ImportError:
        logger.debug("api_key_proxy not available, skipping registration")


# Auto-register on import
_register()


__all__ = [
    "rotate_elevenlabs_key",
]
