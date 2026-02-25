"""
Gemini (Google Generative Language) API Key Rotation Handler.

Implements automatic key rotation for Google Gemini API keys:
1. Create new API key via Google Cloud API Keys service
2. Restrict to Generative Language API only (least privilege)
3. Update AWS Secrets Manager (both production and standalone secrets)
4. Invalidate local caches
5. Delete the old API key

This handler is registered with the APIKeyProxy rotation system
and is triggered by the frequency-hopping scheduler.

Requirements:
    - Google Cloud project with API Keys API enabled
    - Service account credentials (GOOGLE_APPLICATION_CREDENTIALS) or
      Application Default Credentials for key management
    - The Generative Language API key itself does NOT have permission
      to manage API keys — OAuth2/service account is required

Usage:
    # Registration happens automatically on import
    from aragora.security import gemini_rotator

    # Or manually trigger:
    from aragora.security.api_key_proxy import get_api_key_proxy
    proxy = get_api_key_proxy()
    result = await proxy.rotator.rotate_now("gemini")
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Google Cloud API Keys REST endpoint
GCP_API_KEYS_BASE = "https://apikeys.googleapis.com/v2"


async def rotate_gemini_key(
    service: str,
    config: Any,
) -> str | None:
    """Rotate the Gemini API key.

    Steps:
    1. Obtain Google Cloud OAuth2 token (via service account / ADC)
    2. Get current key from secrets
    3. Create new API key via Google Cloud API Keys service
    4. Restrict new key to Generative Language API only
    5. Update AWS Secrets Manager
    6. Update standalone secret if configured
    7. Delete old key
    8. Return new key

    Returns:
        New API key string, or None on failure.
    """
    try:
        import httpx  # noqa: F401
    except ImportError:
        logger.error("httpx required for Gemini key rotation. Install with: pip install httpx")
        return None

    svc_config = config.services.get(service)
    if not svc_config:
        logger.error("No config for service: %s", service)
        return None

    # Step 1: Get OAuth2 access token for key management
    access_token = _get_access_token()
    if not access_token:
        logger.error(
            "Cannot rotate Gemini key: no Google Cloud credentials found. "
            "Set GOOGLE_APPLICATION_CREDENTIALS or configure Application Default Credentials."
        )
        return None

    # Get the GCP project ID
    project_id = _get_project_id()
    if not project_id:
        logger.error("Cannot rotate: GCP project ID not found. Set GCP_PROJECT_ID env var.")
        return None

    # Step 2: Get current key value (for later deletion)
    current_key = _get_current_key(svc_config)
    if not current_key:
        logger.error("Cannot rotate: no current Gemini key found")
        return None

    # Step 3: Create new API key
    new_key_data = await _create_api_key(access_token, project_id)
    if not new_key_data:
        return None

    new_key = new_key_data["key_string"]
    new_key_name = new_key_data["name"]

    logger.info(
        "Created new Gemini API key: %s...%s (name=%s)",
        new_key[:8],
        new_key[-4:],
        new_key_name,
    )

    # Step 4: Restrict new key to Generative Language API
    if not await _restrict_key(access_token, new_key_name):
        logger.warning("Failed to restrict new key — proceeding with unrestricted key")

    # Step 5: Update AWS Secrets Manager (production secret)
    if not await _update_secrets_manager(
        svc_config.secret_id,
        svc_config.secret_manager_key,
        new_key,
        config.aws_region,
    ):
        logger.error("Failed to update production secret — aborting rotation")
        # Try to clean up the new key since we couldn't propagate it
        await _delete_api_key(access_token, new_key_name)
        return None

    # Step 6: Update standalone secret if configured
    if svc_config.standalone_secret_id:
        await _update_standalone_secret(
            svc_config.standalone_secret_id,
            new_key,
            config.aws_region,
        )

    # Step 7: Update local .env if present
    _update_local_env(svc_config.secret_manager_key, new_key)

    # Step 8: Delete old key (best-effort — new key is already propagated)
    old_key_name = await _find_key_by_string(access_token, project_id, current_key)
    if old_key_name:
        await _delete_api_key(access_token, old_key_name)
    else:
        logger.warning("Could not identify old key for deletion — manual cleanup needed")

    # Step 9: Refresh the secrets module cache
    try:
        from aragora.config.secrets import refresh_secrets

        refresh_secrets()
    except (ImportError, RuntimeError, OSError, ValueError):
        logger.debug("Could not refresh secrets cache")

    logger.info("Gemini key rotation completed successfully")
    return new_key


def _get_access_token() -> str | None:
    """Get a Google Cloud OAuth2 access token for API key management.

    Tries:
    1. google.auth.default() (Application Default Credentials / service account)
    2. GOOGLE_ACCESS_TOKEN env var (manual / CI)
    """
    # Try google-auth library (preferred)
    try:
        import google.auth
        import google.auth.transport.requests

        credentials, _project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
        return credentials.token
    except ImportError:
        logger.debug("google-auth not installed, trying fallback methods")
    except Exception as e:  # noqa: BLE001 - google-auth can raise various exceptions
        logger.debug("google.auth.default() failed: %s", e)

    # Manual token fallback
    token = os.environ.get("GOOGLE_ACCESS_TOKEN")
    if token:
        return token

    return None


def _get_project_id() -> str | None:
    """Get the GCP project ID."""
    # Explicit env var
    project_id = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project_id:
        return project_id

    # Try google-auth
    try:
        import google.auth

        _, project = google.auth.default()
        if project:
            return project
    except (ImportError, ValueError, RuntimeError, OSError):
        pass

    # Try gcloud config
    try:
        import subprocess

        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],  # noqa: S607 -- fixed command
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    return None


def _get_current_key(svc_config: Any) -> str | None:
    """Get the current Gemini API key."""
    # Try secrets module first
    try:
        from aragora.config.secrets import get_secret

        key = get_secret(svc_config.secret_manager_key)
        if key:
            return key
    except (ImportError, RuntimeError, OSError, ValueError):
        logger.debug("Could not load Gemini key via secrets module")

    # Env fallback — check both GEMINI_API_KEY and GOOGLE_API_KEY
    return os.environ.get(svc_config.secret_manager_key) or os.environ.get("GOOGLE_API_KEY")


async def _create_api_key(
    access_token: str,
    project_id: str,
) -> dict[str, Any] | None:
    """Create a new Google Cloud API key.

    Uses the API Keys v2 REST API to create a key.
    """
    try:
        import httpx
    except ImportError:
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    display_name = f"aragora-gemini-{timestamp}"

    url = f"{GCP_API_KEYS_BASE}/projects/{project_id}/locations/global/keys"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "displayName": display_name,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, headers=headers, json=payload)

            if resp.status_code in (200, 201):
                # API Keys creation returns a long-running operation
                operation = resp.json()
                key_data = await _wait_for_operation(client, access_token, operation)
                if key_data:
                    # Now get the key string
                    key_name = key_data.get("name", "")
                    key_string = await _get_key_string(client, access_token, key_name)
                    if key_string:
                        return {
                            "name": key_name,
                            "key_string": key_string,
                            "display_name": display_name,
                        }
                return None
            else:
                logger.error(
                    "Google API key creation failed: %s %s",
                    resp.status_code,
                    resp.text,
                )
                return None

    except Exception as e:  # noqa: BLE001 - httpx/google exceptions
        logger.error("Google API key creation request failed: %s", e)
        return None


async def _wait_for_operation(
    client: Any,
    access_token: str,
    operation: dict[str, Any],
    max_polls: int = 30,
    poll_interval: float = 2.0,
) -> dict[str, Any] | None:
    """Wait for a Google Cloud long-running operation to complete."""
    import asyncio

    # If the operation is already done (synchronous creation)
    if operation.get("done"):
        return operation.get("response", operation.get("metadata", {}))

    op_name = operation.get("name", "")
    if not op_name:
        logger.error("Operation has no name: %s", operation)
        return None

    headers = {"Authorization": f"Bearer {access_token}"}

    for _i in range(max_polls):
        await asyncio.sleep(poll_interval)

        resp = await client.get(
            f"https://apikeys.googleapis.com/v2/{op_name}",
            headers=headers,
        )

        if resp.status_code != 200:
            logger.warning("Operation poll failed: %s", resp.status_code)
            continue

        op_data = resp.json()
        if op_data.get("done"):
            if "error" in op_data:
                logger.error("Operation failed: %s", op_data["error"])
                return None
            return op_data.get("response", op_data.get("metadata", {}))

    logger.error("Operation timed out after %s polls", max_polls)
    return None


async def _get_key_string(
    client: Any,
    access_token: str,
    key_name: str,
) -> str | None:
    """Get the key string for an API key (requires separate call)."""
    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        resp = await client.get(
            f"{GCP_API_KEYS_BASE}/{key_name}/keyString",
            headers=headers,
        )

        if resp.status_code == 200:
            return resp.json().get("keyString")
        else:
            logger.error("Failed to get key string: %s %s", resp.status_code, resp.text)
            return None

    except Exception as e:  # noqa: BLE001 - httpx exceptions
        logger.error("Failed to get key string: %s", e)
        return None


async def _restrict_key(
    access_token: str,
    key_name: str,
) -> bool:
    """Restrict an API key to only the Generative Language API.

    Applies API restrictions so the key can ONLY call:
    - generativelanguage.googleapis.com (Gemini)
    """
    try:
        import httpx
    except ImportError:
        return False

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    # Patch the key with API restrictions
    payload = {
        "restrictions": {
            "apiTargets": [
                {
                    "service": "generativelanguage.googleapis.com",
                }
            ]
        }
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.patch(
                f"{GCP_API_KEYS_BASE}/{key_name}",
                headers=headers,
                json=payload,
                params={"updateMask": "restrictions"},
            )

            if resp.status_code == 200:
                # This also returns an operation — wait for it
                operation = resp.json()
                if operation.get("done", True):
                    logger.info("API key restricted to Generative Language API")
                    return True
                result = await _wait_for_operation(client, access_token, operation, max_polls=10)
                if result is not None:
                    logger.info("API key restricted to Generative Language API")
                    return True
                return False
            else:
                logger.warning(
                    "Failed to restrict API key: %s %s",
                    resp.status_code,
                    resp.text,
                )
                return False

    except Exception as e:  # noqa: BLE001 - httpx exceptions
        logger.error("Failed to restrict API key: %s", e)
        return False


async def _find_key_by_string(
    access_token: str,
    project_id: str,
    target_key: str,
) -> str | None:
    """Find the resource name of an API key by its key string.

    Uses the lookupKey API to find which key resource has a given key string.
    """
    try:
        import httpx
    except ImportError:
        return None

    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Use the lookupKey endpoint
            resp = await client.get(
                f"{GCP_API_KEYS_BASE}/keys:lookupKey",
                headers=headers,
                params={"keyString": target_key},
            )

            if resp.status_code == 200:
                data = resp.json()
                return data.get("name")
            else:
                logger.warning(
                    "Could not look up key: %s %s",
                    resp.status_code,
                    resp.text,
                )
                return None

    except Exception as e:  # noqa: BLE001 - httpx exceptions
        logger.error("Failed to look up API key: %s", e)
        return None


async def _delete_api_key(
    access_token: str,
    key_name: str,
) -> bool:
    """Delete a Google Cloud API key by resource name."""
    try:
        import httpx
    except ImportError:
        return False

    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.delete(
                f"{GCP_API_KEYS_BASE}/{key_name}",
                headers=headers,
            )

            if resp.status_code in (200, 202):
                logger.info("Deleted old Gemini API key: %s", key_name)
                return True
            else:
                logger.warning(
                    "Failed to delete API key %s: %s %s",
                    key_name,
                    resp.status_code,
                    resp.text,
                )
                return False

    except Exception as e:  # noqa: BLE001 - httpx exceptions
        logger.error("Failed to delete API key: %s", e)
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

        # Also update GOOGLE_API_KEY if it was using the same value
        if secret_data.get("GOOGLE_API_KEY") == old_value:
            secret_data["GOOGLE_API_KEY"] = new_value

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
    except Exception as e:  # noqa: BLE001 - boto3 exceptions
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

    except Exception as e:  # noqa: BLE001 - boto3 exceptions
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
                # Also update GOOGLE_API_KEY if present
                elif key_name == "GEMINI_API_KEY" and line.startswith("GOOGLE_API_KEY="):
                    new_lines.append(f"GOOGLE_API_KEY={new_value}\n")
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
    if key_name == "GEMINI_API_KEY":
        os.environ["GOOGLE_API_KEY"] = new_value


# =============================================================================
# Auto-register with APIKeyProxy
# =============================================================================


def _register() -> None:
    """Register the Gemini rotation handler with the proxy system."""
    try:
        from aragora.security.api_key_proxy import register_rotation_handler

        register_rotation_handler("gemini", rotate_gemini_key)
        logger.debug("Gemini rotation handler registered")
    except ImportError:
        logger.debug("api_key_proxy not available, skipping registration")


# Auto-register on import
_register()


__all__ = [
    "rotate_gemini_key",
]
