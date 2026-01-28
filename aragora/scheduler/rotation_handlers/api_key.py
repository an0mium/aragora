"""
API key rotation handler.

Handles rotation for LLM providers and other API keys.
Note: Most LLM providers don't support programmatic key rotation,
so this handler primarily manages manual rotation workflows.
"""

from datetime import datetime
from typing import Any
import logging

from .base import RotationHandler, RotationError, RotationResult

logger = logging.getLogger(__name__)


class APIKeyRotationHandler(RotationHandler):
    """
    Handler for API key rotation.

    Providers with programmatic support:
    - OpenAI (via API, requires org admin)
    - GitHub (via API)

    Providers requiring manual rotation:
    - Anthropic
    - Google (Gemini)
    - Mistral
    - Grok/xAI
    - OpenRouter

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
                "version": f"v{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                "rotated_at": datetime.utcnow().isoformat(),
                "rotation_method": "manual",
            }

        # Try programmatic rotation for supported providers
        if provider == "openai":
            return await self._rotate_openai(secret_id, metadata)
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
            except Exception:
                pass

        if not admin_key:
            await self._notify_manual_rotation(secret_id, "openai", metadata)
            raise RotationError(
                "OpenAI admin key required for programmatic rotation. "
                "Notification sent for manual rotation.",
                secret_id,
            )

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                # Create new API key
                response = await client.post(
                    "https://api.openai.com/v1/organization/api_keys",
                    headers={
                        "Authorization": f"Bearer {admin_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "name": f"aragora-{datetime.utcnow().strftime('%Y%m%d')}",
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
                "version": f"v{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                "rotated_at": datetime.utcnow().isoformat(),
                "rotation_method": "programmatic",
            }

        except ImportError:
            raise RotationError("httpx not installed. Install with: pip install httpx", secret_id)
        except Exception as e:
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
            except Exception as e:
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
        }

        validator = validators.get(provider)
        if validator:
            return await validator(secret_id, secret_value, metadata)

        logger.warning(f"No validator for provider {provider}")
        return True

    async def _validate_anthropic(self, secret_id: str, key: str, metadata: dict[str, Any]) -> bool:
        """Validate Anthropic API key."""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
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

        except Exception as e:
            logger.error(f"Anthropic validation error: {e}")
            return False

    async def _validate_openai(self, secret_id: str, key: str, metadata: dict[str, Any]) -> bool:
        """Validate OpenAI API key."""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=10.0,
                )
                return response.status_code == 200

        except Exception as e:
            logger.error(f"OpenAI validation error: {e}")
            return False

    async def _validate_google(self, secret_id: str, key: str, metadata: dict[str, Any]) -> bool:
        """Validate Google/Gemini API key."""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://generativelanguage.googleapis.com/v1/models?key={key}",
                    timeout=10.0,
                )
                return response.status_code == 200

        except Exception as e:
            logger.error(f"Google validation error: {e}")
            return False

    async def _validate_mistral(self, secret_id: str, key: str, metadata: dict[str, Any]) -> bool:
        """Validate Mistral API key."""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.mistral.ai/v1/models",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=10.0,
                )
                return response.status_code == 200

        except Exception as e:
            logger.error(f"Mistral validation error: {e}")
            return False

    async def _validate_openrouter(
        self, secret_id: str, key: str, metadata: dict[str, Any]
    ) -> bool:
        """Validate OpenRouter API key."""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://openrouter.ai/api/v1/auth/key",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=10.0,
                )
                return response.status_code == 200

        except Exception as e:
            logger.error(f"OpenRouter validation error: {e}")
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
            except Exception:
                pass

        if not admin_key:
            logger.warning(f"No admin key for OpenAI revocation of {secret_id}")
            return False

        try:
            import httpx

            async with httpx.AsyncClient() as client:
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

        except Exception as e:
            logger.error(f"OpenAI revocation error: {e}")
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
