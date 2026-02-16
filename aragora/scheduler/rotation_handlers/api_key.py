"""
API key rotation handler.

Handles rotation for LLM providers and other API keys.

Providers with programmatic key rotation:
- OpenAI (via Admin API)
- Anthropic (via Admin API)
- Stripe (via rolling key API)
- Cloudflare (via API token management)

Providers requiring manual rotation:
- Google (Gemini), Mistral, xAI, OpenRouter, DeepSeek, ElevenLabs, npm
"""

from datetime import datetime, timezone
from typing import Any
import logging

from .base import RotationHandler, RotationError, RotationResult

logger = logging.getLogger(__name__)


class APIKeyRotationHandler(RotationHandler):
    """
    Handler for API key rotation.

    Providers with programmatic support:
    - OpenAI (via Admin API, requires org admin)
    - Anthropic (via Admin API, requires admin key)
    - Stripe (via rolling key API)
    - Cloudflare (via API token management)

    Providers requiring manual rotation:
    - Google (Gemini), Mistral, xAI, OpenRouter, DeepSeek, ElevenLabs, npm

    For manual rotation, this handler:
    1. Sends notification that rotation is needed
    2. Validates new key when provided
    3. Updates secret storage
    """

    @property
    def secret_type(self) -> str:
        return "api_key"

    def __init__(
        self,
        grace_period_hours: int = 48,  # Longer for API keys
        max_retries: int = 3,
        notification_callback: Any | None = None,
    ):
        """
        Initialize API key rotation handler.

        Args:
            grace_period_hours: Hours old key remains valid
            max_retries: Maximum retry attempts
            notification_callback: Async callback for manual rotation notifications
        """
        super().__init__(grace_period_hours, max_retries)
        self.notification_callback = notification_callback

    async def generate_new_credentials(
        self, secret_id: str, metadata: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """
        Generate or request new API key.

        Args:
            secret_id: API key identifier (e.g., "ANTHROPIC_API_KEY")
            metadata: Should contain 'provider' and optionally 'new_key' for manual rotation

        Returns:
            Tuple of (new_key, updated_metadata)
        """
        provider = metadata.get("provider", "unknown")

        # Check if a new key was provided manually
        if "new_key" in metadata:
            new_key = metadata["new_key"]
            logger.info(f"Using manually provided new key for {secret_id}")
            return new_key, {
                **metadata,
                "version": f"v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                "rotated_at": datetime.now(timezone.utc).isoformat(),
                "rotation_method": "manual",
            }

        # Try programmatic rotation for supported providers
        if provider == "openai":
            return await self._rotate_openai(secret_id, metadata)
        elif provider == "anthropic":
            return await self._rotate_anthropic(secret_id, metadata)
        elif provider == "stripe":
            return await self._rotate_stripe(secret_id, metadata)
        elif provider == "cloudflare":
            return await self._rotate_cloudflare(secret_id, metadata)
        elif provider == "github":
            return await self._rotate_github(secret_id, metadata)
        else:
            # For providers without programmatic rotation, send notification
            await self._notify_manual_rotation(secret_id, provider, metadata)
            raise RotationError(
                f"Provider {provider} requires manual key rotation. "
                f"Notification sent to configured channels.",
                secret_id,
            )

    async def _rotate_openai(
        self, secret_id: str, metadata: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """
        Rotate OpenAI API key programmatically.

        Requires organization admin API key with key management permissions.
        """
        admin_key = metadata.get("admin_key")
        if not admin_key:
            try:
                from aragora.config.secrets import get_secret

                admin_key = get_secret("OPENAI_ADMIN_KEY")
            except (ImportError, KeyError, OSError) as e:
                logger.debug(f"Could not load OpenAI admin key from secrets: {e}")

        if not admin_key:
            await self._notify_manual_rotation(secret_id, "openai", metadata)
            raise RotationError(
                "OpenAI admin key required for programmatic rotation. "
                "Notification sent for manual rotation.",
                secret_id,
            )

        try:
            from aragora.server.http_client_pool import get_http_pool

            pool = get_http_pool()
            async with pool.get_session("openai") as client:
                # Create new API key
                response = await client.post(
                    "https://api.openai.com/v1/organization/api_keys",
                    headers={
                        "Authorization": f"Bearer {admin_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "name": f"aragora-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
                        "scopes": metadata.get("scopes", ["model.read", "model.request"]),
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                key_data = response.json()

            new_key = key_data.get("key")
            if not new_key:
                raise RotationError("No key in OpenAI response", secret_id)

            logger.info(f"Created new OpenAI API key for {secret_id}")

            return new_key, {
                **metadata,
                "key_id": key_data.get("id"),
                "version": f"v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                "rotated_at": datetime.now(timezone.utc).isoformat(),
                "rotation_method": "programmatic",
            }

        except ImportError as e:
            raise RotationError(f"Required module not installed: {e}", secret_id)
        except (OSError, ConnectionError, TimeoutError, ValueError, KeyError) as e:
            logger.error(f"OpenAI key rotation failed: {e}")
            await self._notify_manual_rotation(secret_id, "openai", metadata)
            raise RotationError(
                f"OpenAI programmatic rotation failed: {e}. Notification sent for manual rotation.",
                secret_id,
            )

    async def _rotate_github(
        self, secret_id: str, metadata: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """
        Rotate GitHub personal access token or fine-grained token.

        Note: GitHub doesn't support creating tokens via API.
        This handler manages token expiration notifications.
        """
        await self._notify_manual_rotation(secret_id, "github", metadata)
        raise RotationError(
            "GitHub tokens must be rotated manually. Notification sent.",
            secret_id,
        )

    async def _rotate_anthropic(
        self, secret_id: str, metadata: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """
        Rotate Anthropic API key via Admin API.

        Requires an admin key with key management permissions.
        Creates a new key, then schedules the old one for deletion.
        """
        admin_key = metadata.get("admin_key")
        if not admin_key:
            try:
                from aragora.config.secrets import get_secret

                admin_key = get_secret("ANTHROPIC_ADMIN_KEY")
            except (ImportError, KeyError, OSError) as e:
                logger.debug(f"Could not load Anthropic admin key: {e}")

        if not admin_key:
            await self._notify_manual_rotation(secret_id, "anthropic", metadata)
            raise RotationError(
                "Anthropic admin key required for programmatic rotation. "
                "Notification sent for manual rotation.",
                secret_id,
            )

        org_id = metadata.get("org_id")
        if not org_id:
            try:
                from aragora.config.secrets import get_secret

                org_id = get_secret("ANTHROPIC_ORG_ID")
            except (ImportError, KeyError, OSError):
                pass

        if not org_id:
            await self._notify_manual_rotation(secret_id, "anthropic", metadata)
            raise RotationError(
                "Anthropic org_id required for programmatic rotation. "
                "Set ANTHROPIC_ORG_ID in secrets.",
                secret_id,
            )

        try:
            from aragora.server.http_client_pool import get_http_pool

            pool = get_http_pool()
            async with pool.get_session("anthropic") as client:
                # Create new API key
                response = await client.post(
                    f"https://api.anthropic.com/v1/organizations/{org_id}/api_keys",
                    headers={
                        "x-api-key": admin_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "name": f"aragora-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                key_data = response.json()

            new_key = key_data.get("api_key")
            if not new_key:
                raise RotationError("No key in Anthropic response", secret_id)

            logger.info(f"Created new Anthropic API key for {secret_id}")

            return new_key, {
                **metadata,
                "key_id": key_data.get("id"),
                "version": f"v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                "rotated_at": datetime.now(timezone.utc).isoformat(),
                "rotation_method": "programmatic",
            }

        except ImportError as e:
            raise RotationError(f"Required module not installed: {e}", secret_id)
        except (OSError, ConnectionError, TimeoutError, ValueError, KeyError) as e:
            logger.error(f"Anthropic key rotation failed: {e}")
            await self._notify_manual_rotation(secret_id, "anthropic", metadata)
            raise RotationError(
                f"Anthropic programmatic rotation failed: {e}. "
                f"Notification sent for manual rotation.",
                secret_id,
            )

    async def _rotate_stripe(
        self, secret_id: str, metadata: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """
        Rotate Stripe API key via the rolling key API.

        After rolling, the old key remains valid for 24 hours.
        """
        current_key = metadata.get("current_key")
        if not current_key:
            try:
                from aragora.config.secrets import get_secret

                current_key = get_secret(secret_id)
            except (ImportError, KeyError, OSError) as e:
                logger.debug(f"Could not load current Stripe key: {e}")

        if not current_key:
            await self._notify_manual_rotation(secret_id, "stripe", metadata)
            raise RotationError(
                "Current Stripe key required for rolling rotation. "
                "Notification sent for manual rotation.",
                secret_id,
            )

        try:
            from aragora.server.http_client_pool import get_http_pool

            pool = get_http_pool()
            async with pool.get_session("stripe") as client:
                response = await client.post(
                    "https://api.stripe.com/v1/api_keys/roll",
                    headers={
                        "Authorization": f"Bearer {current_key}",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                key_data = response.json()

            new_key = key_data.get("secret")
            if not new_key:
                raise RotationError("No key in Stripe roll response", secret_id)

            logger.info(
                f"Rolled Stripe API key for {secret_id}. "
                f"Old key valid for 24 hours."
            )

            return new_key, {
                **metadata,
                "key_id": key_data.get("id"),
                "old_key_expires": key_data.get("expires"),
                "version": f"v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                "rotated_at": datetime.now(timezone.utc).isoformat(),
                "rotation_method": "programmatic",
                "grace_note": "Old key valid for 24 hours after rolling",
            }

        except ImportError as e:
            raise RotationError(f"Required module not installed: {e}", secret_id)
        except (OSError, ConnectionError, TimeoutError, ValueError, KeyError) as e:
            logger.error(f"Stripe key rotation failed: {e}")
            await self._notify_manual_rotation(secret_id, "stripe", metadata)
            raise RotationError(
                f"Stripe programmatic rotation failed: {e}. "
                f"Notification sent for manual rotation.",
                secret_id,
            )

    async def _rotate_cloudflare(
        self, secret_id: str, metadata: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """
        Rotate Cloudflare API token.

        Creates a new token with the same policies, validates it,
        then deletes the old token.
        """
        admin_token = metadata.get("admin_token")
        if not admin_token:
            try:
                from aragora.config.secrets import get_secret

                admin_token = get_secret("CLOUDFLARE_ADMIN_TOKEN")
            except (ImportError, KeyError, OSError) as e:
                logger.debug(f"Could not load Cloudflare admin token: {e}")

        if not admin_token:
            await self._notify_manual_rotation(secret_id, "cloudflare", metadata)
            raise RotationError(
                "Cloudflare admin token required for programmatic rotation. "
                "Notification sent for manual rotation.",
                secret_id,
            )

        token_name = metadata.get(
            "token_name",
            f"aragora-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        )
        policies = metadata.get("policies", [])

        try:
            from aragora.server.http_client_pool import get_http_pool

            pool = get_http_pool()
            async with pool.get_session("cloudflare") as client:
                # Create new token
                create_payload: dict[str, Any] = {
                    "name": token_name,
                }
                if policies:
                    create_payload["policies"] = policies

                response = await client.post(
                    "https://api.cloudflare.com/client/v4/user/tokens",
                    headers={
                        "Authorization": f"Bearer {admin_token}",
                        "Content-Type": "application/json",
                    },
                    json=create_payload,
                    timeout=30.0,
                )
                response.raise_for_status()
                result = response.json()

            if not result.get("success"):
                errors = result.get("errors", [])
                raise RotationError(
                    f"Cloudflare token creation failed: {errors}",
                    secret_id,
                )

            new_token = result["result"]["value"]
            new_token_id = result["result"]["id"]

            logger.info(f"Created new Cloudflare API token for {secret_id}")

            # Schedule old token deletion if we have the old token ID
            old_token_id = metadata.get("old_token_id")
            if old_token_id:
                try:
                    async with pool.get_session("cloudflare") as client:
                        await client.delete(
                            f"https://api.cloudflare.com/client/v4/user/tokens/{old_token_id}",
                            headers={
                                "Authorization": f"Bearer {admin_token}",
                            },
                            timeout=10.0,
                        )
                    logger.info(f"Deleted old Cloudflare token {old_token_id}")
                except (OSError, ConnectionError, TimeoutError) as e:
                    logger.warning(f"Failed to delete old Cloudflare token: {e}")

            return new_token, {
                **metadata,
                "token_id": new_token_id,
                "old_token_id": new_token_id,  # For next rotation
                "version": f"v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                "rotated_at": datetime.now(timezone.utc).isoformat(),
                "rotation_method": "programmatic",
            }

        except ImportError as e:
            raise RotationError(f"Required module not installed: {e}", secret_id)
        except (OSError, ConnectionError, TimeoutError, ValueError, KeyError) as e:
            logger.error(f"Cloudflare token rotation failed: {e}")
            await self._notify_manual_rotation(secret_id, "cloudflare", metadata)
            raise RotationError(
                f"Cloudflare programmatic rotation failed: {e}. "
                f"Notification sent for manual rotation.",
                secret_id,
            )

    async def _notify_manual_rotation(
        self, secret_id: str, provider: str, metadata: dict[str, Any]
    ) -> None:
        """Send notification that manual rotation is required."""
        if self.notification_callback:
            try:
                await self.notification_callback(
                    secret_id=secret_id,
                    provider=provider,
                    message=f"API key rotation required for {secret_id} ({provider}). "
                    f"Please generate a new key and update the secret.",
                    metadata=metadata,
                )
                logger.info(f"Sent rotation notification for {secret_id}")
            except (OSError, ConnectionError, TimeoutError, RuntimeError) as e:
                logger.error(f"Failed to send notification for {secret_id}: {e}")
        else:
            logger.warning(
                f"Manual rotation required for {secret_id} ({provider}) "
                f"but no notification callback configured"
            )

    async def validate_credentials(
        self, secret_id: str, secret_value: str, metadata: dict[str, Any]
    ) -> bool:
        """
        Validate API key by making a test request.

        Args:
            secret_id: API key identifier
            secret_value: Key to test
            metadata: Provider details

        Returns:
            True if key is valid
        """
        provider = metadata.get("provider", "unknown")

        validators = {
            "anthropic": self._validate_anthropic,
            "openai": self._validate_openai,
            "google": self._validate_google,
            "mistral": self._validate_mistral,
            "openrouter": self._validate_openrouter,
            "xai": self._validate_xai,
            "deepseek": self._validate_deepseek,
            "stripe": self._validate_stripe,
            "cloudflare": self._validate_cloudflare,
        }

        validator = validators.get(provider)
        if validator:
            return await validator(secret_id, secret_value, metadata)

        logger.warning(f"No validator for provider {provider}")
        return True

    async def _validate_anthropic(self, secret_id: str, key: str, metadata: dict[str, Any]) -> bool:
        """Validate Anthropic API key."""
        try:
            from aragora.server.http_client_pool import get_http_pool

            pool = get_http_pool()
            async with pool.get_session("anthropic") as client:
                response = await client.get(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": key,
                        "anthropic-version": "2023-06-01",
                    },
                    timeout=10.0,
                )
                # 401 = invalid key, 400 = valid key but bad request
                return response.status_code != 401

        except (ImportError, OSError, ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"Anthropic validation error: {e}")
            return False

    async def _validate_openai(self, secret_id: str, key: str, metadata: dict[str, Any]) -> bool:
        """Validate OpenAI API key."""
        try:
            from aragora.server.http_client_pool import get_http_pool

            pool = get_http_pool()
            async with pool.get_session("openai") as client:
                response = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=10.0,
                )
                return response.status_code == 200

        except (ImportError, OSError, ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"OpenAI validation error: {e}")
            return False

    async def _validate_google(self, secret_id: str, key: str, metadata: dict[str, Any]) -> bool:
        """Validate Google/Gemini API key."""
        try:
            from aragora.server.http_client_pool import get_http_pool

            pool = get_http_pool()
            async with pool.get_session("google") as client:
                response = await client.get(
                    f"https://generativelanguage.googleapis.com/v1/models?key={key}",
                    timeout=10.0,
                )
                return response.status_code == 200

        except (ImportError, OSError, ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"Google validation error: {e}")
            return False

    async def _validate_mistral(self, secret_id: str, key: str, metadata: dict[str, Any]) -> bool:
        """Validate Mistral API key."""
        try:
            from aragora.server.http_client_pool import get_http_pool

            pool = get_http_pool()
            async with pool.get_session("mistral") as client:
                response = await client.get(
                    "https://api.mistral.ai/v1/models",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=10.0,
                )
                return response.status_code == 200

        except (ImportError, OSError, ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"Mistral validation error: {e}")
            return False

    async def _validate_openrouter(
        self, secret_id: str, key: str, metadata: dict[str, Any]
    ) -> bool:
        """Validate OpenRouter API key."""
        try:
            from aragora.server.http_client_pool import get_http_pool

            pool = get_http_pool()
            async with pool.get_session("openrouter") as client:
                response = await client.get(
                    "https://openrouter.ai/api/v1/auth/key",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=10.0,
                )
                return response.status_code == 200

        except (ImportError, OSError, ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"OpenRouter validation error: {e}")
            return False

    async def _validate_xai(self, secret_id: str, key: str, metadata: dict[str, Any]) -> bool:
        """Validate xAI/Grok API key."""
        try:
            from aragora.server.http_client_pool import get_http_pool

            pool = get_http_pool()
            async with pool.get_session("xai") as client:
                response = await client.get(
                    "https://api.x.ai/v1/models",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=10.0,
                )
                return response.status_code == 200

        except (ImportError, OSError, ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"xAI validation error: {e}")
            return False

    async def _validate_deepseek(self, secret_id: str, key: str, metadata: dict[str, Any]) -> bool:
        """Validate DeepSeek API key."""
        try:
            from aragora.server.http_client_pool import get_http_pool

            pool = get_http_pool()
            async with pool.get_session("deepseek") as client:
                response = await client.get(
                    "https://api.deepseek.com/v1/models",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=10.0,
                )
                return response.status_code == 200

        except (ImportError, OSError, ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"DeepSeek validation error: {e}")
            return False

    async def _validate_stripe(self, secret_id: str, key: str, metadata: dict[str, Any]) -> bool:
        """Validate Stripe API key by checking balance endpoint."""
        try:
            from aragora.server.http_client_pool import get_http_pool

            pool = get_http_pool()
            async with pool.get_session("stripe") as client:
                response = await client.get(
                    "https://api.stripe.com/v1/balance",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=10.0,
                )
                return response.status_code == 200

        except (ImportError, OSError, ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"Stripe validation error: {e}")
            return False

    async def _validate_cloudflare(
        self, secret_id: str, key: str, metadata: dict[str, Any]
    ) -> bool:
        """Validate Cloudflare API token via verify endpoint."""
        try:
            from aragora.server.http_client_pool import get_http_pool

            pool = get_http_pool()
            async with pool.get_session("cloudflare") as client:
                response = await client.get(
                    "https://api.cloudflare.com/client/v4/user/tokens/verify",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=10.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("success", False)
                return False

        except (ImportError, OSError, ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"Cloudflare validation error: {e}")
            return False

    async def revoke_old_credentials(
        self, secret_id: str, old_value: str, metadata: dict[str, Any]
    ) -> bool:
        """
        Revoke old API key if provider supports it.

        Args:
            secret_id: API key identifier
            old_value: Old key to revoke
            metadata: Provider details

        Returns:
            True if revocation succeeded or not supported
        """
        provider = metadata.get("provider", "unknown")
        key_id = metadata.get("old_key_id")

        if provider == "openai" and key_id:
            return await self._revoke_openai(secret_id, key_id, metadata)
        elif provider == "anthropic" and key_id:
            return await self._revoke_anthropic(secret_id, key_id, metadata)

        # Most providers don't support programmatic revocation
        logger.info(
            f"Provider {provider} doesn't support programmatic key revocation. "
            f"Old key for {secret_id} should be manually revoked."
        )
        return True

    async def _revoke_openai(self, secret_id: str, key_id: str, metadata: dict[str, Any]) -> bool:
        """Revoke OpenAI API key."""
        admin_key = metadata.get("admin_key")
        if not admin_key:
            try:
                from aragora.config.secrets import get_secret

                admin_key = get_secret("OPENAI_ADMIN_KEY")
            except (ImportError, KeyError, OSError) as e:
                logger.debug(f"Could not load OpenAI admin key from secrets for revocation: {e}")

        if not admin_key:
            logger.warning(f"No admin key for OpenAI revocation of {secret_id}")
            return False

        try:
            from aragora.server.http_client_pool import get_http_pool

            pool = get_http_pool()
            async with pool.get_session("openai") as client:
                response = await client.delete(
                    f"https://api.openai.com/v1/organization/api_keys/{key_id}",
                    headers={"Authorization": f"Bearer {admin_key}"},
                    timeout=30.0,
                )
                if response.status_code < 400:
                    logger.info(f"Revoked old OpenAI key {key_id}")
                    return True
                else:
                    logger.warning(f"OpenAI revocation returned {response.status_code}")
                    return False

        except (ImportError, OSError, ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"OpenAI revocation error: {e}")
            return False

    async def _revoke_anthropic(
        self, secret_id: str, key_id: str, metadata: dict[str, Any]
    ) -> bool:
        """Revoke Anthropic API key."""
        admin_key = metadata.get("admin_key")
        if not admin_key:
            try:
                from aragora.config.secrets import get_secret

                admin_key = get_secret("ANTHROPIC_ADMIN_KEY")
            except (ImportError, KeyError, OSError) as e:
                logger.debug(f"Could not load Anthropic admin key for revocation: {e}")

        org_id = metadata.get("org_id")
        if not org_id:
            try:
                from aragora.config.secrets import get_secret

                org_id = get_secret("ANTHROPIC_ORG_ID")
            except (ImportError, KeyError, OSError):
                pass

        if not admin_key or not org_id:
            logger.warning(f"No admin key/org_id for Anthropic revocation of {secret_id}")
            return False

        try:
            from aragora.server.http_client_pool import get_http_pool

            pool = get_http_pool()
            async with pool.get_session("anthropic") as client:
                response = await client.delete(
                    f"https://api.anthropic.com/v1/organizations/{org_id}/api_keys/{key_id}",
                    headers={
                        "x-api-key": admin_key,
                        "anthropic-version": "2023-06-01",
                    },
                    timeout=30.0,
                )
                if response.status_code < 400:
                    logger.info(f"Revoked old Anthropic key {key_id}")
                    return True
                else:
                    logger.warning(f"Anthropic revocation returned {response.status_code}")
                    return False

        except (ImportError, OSError, ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"Anthropic revocation error: {e}")
            return False

    async def rotate_with_new_key(
        self,
        secret_id: str,
        new_key: str,
        current_value: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RotationResult:
        """
        Convenience method for manual rotation with a provided new key.

        Args:
            secret_id: API key identifier
            new_key: The new key provided manually
            current_value: Current key (for revocation)
            metadata: Provider details

        Returns:
            RotationResult with status
        """
        metadata = metadata or {}
        metadata["new_key"] = new_key
        return await self.rotate(secret_id, current_value, metadata)
