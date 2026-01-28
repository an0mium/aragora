"""
OAuth token rotation handler.

Handles refresh token rotation for OAuth 2.0 integrations.
"""

from datetime import datetime, timedelta
from typing import Any
import logging

from .base import RotationHandler, RotationError

logger = logging.getLogger(__name__)


class OAuthRotationHandler(RotationHandler):
    """
    Handler for OAuth token rotation.

    Supports:
    - Google OAuth (Gmail, Calendar, Drive)
    - Microsoft OAuth (Graph API, Office 365)
    - GitHub OAuth
    - Slack OAuth
    - Generic OAuth 2.0
    """

    @property
    def secret_type(self) -> str:
        return "oauth"

    def __init__(
        self,
        grace_period_hours: int = 1,  # Shorter for tokens
        max_retries: int = 3,
    ):
        """
        Initialize OAuth rotation handler.

        Args:
            grace_period_hours: Hours old token remains valid (usually 1 for OAuth)
            max_retries: Maximum retry attempts
        """
        super().__init__(grace_period_hours, max_retries)

    async def generate_new_credentials(
        self, secret_id: str, metadata: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """
        Refresh OAuth tokens.

        Args:
            secret_id: OAuth credential identifier
            metadata: Should contain 'provider', 'refresh_token', 'client_id', 'client_secret'

        Returns:
            Tuple of (new_access_token, updated_metadata with new refresh_token if issued)
        """
        provider = metadata.get("provider", "generic")
        refresh_token = metadata.get("refresh_token")
        client_id = metadata.get("client_id")
        client_secret = metadata.get("client_secret")

        if not refresh_token:
            raise RotationError("Refresh token required for OAuth rotation", secret_id)

        if not client_id or not client_secret:
            # Try to get from secrets manager
            try:
                from aragora.config.secrets import get_secret

                client_id = client_id or get_secret(f"{provider.upper()}_CLIENT_ID")
                client_secret = client_secret or get_secret(f"{provider.upper()}_CLIENT_SECRET")
            except Exception:
                pass

        if not client_id or not client_secret:
            raise RotationError(
                f"Client credentials required for {provider} OAuth rotation", secret_id
            )

        # Get token endpoint based on provider
        token_url = self._get_token_url(provider, metadata)

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    token_url,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                        "client_id": client_id,
                        "client_secret": client_secret,
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                token_data = response.json()

            new_access_token = token_data.get("access_token")
            new_refresh_token = token_data.get("refresh_token", refresh_token)
            expires_in = token_data.get("expires_in", 3600)

            if not new_access_token:
                raise RotationError("No access token in response", secret_id)

            logger.info(f"Refreshed OAuth tokens for {secret_id} ({provider})")

            return new_access_token, {
                **metadata,
                "refresh_token": new_refresh_token,
                "expires_at": (datetime.utcnow() + timedelta(seconds=expires_in)).isoformat(),
                "version": f"v{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                "rotated_at": datetime.utcnow().isoformat(),
            }

        except ImportError:
            raise RotationError("httpx not installed. Install with: pip install httpx", secret_id)
        except Exception as e:
            raise RotationError(f"OAuth refresh failed: {e}", secret_id)

    def _get_token_url(self, provider: str, metadata: dict[str, Any]) -> str:
        """Get the token endpoint URL for a provider."""
        # Allow custom token URL override
        if "token_url" in metadata:
            return metadata["token_url"]

        provider_urls = {
            "google": "https://oauth2.googleapis.com/token",
            "microsoft": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
            "github": "https://github.com/login/oauth/access_token",
            "slack": "https://slack.com/api/oauth.v2.access",
            "discord": "https://discord.com/api/oauth2/token",
            "salesforce": "https://login.salesforce.com/services/oauth2/token",
            "hubspot": "https://api.hubapi.com/oauth/v1/token",
            "jira": "https://auth.atlassian.com/oauth/token",
        }

        # Handle tenant-specific Microsoft URLs
        if provider == "microsoft":
            tenant = metadata.get("tenant", "common")
            return f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"

        return provider_urls.get(
            provider, metadata.get("token_url", "https://oauth.example.com/token")
        )

    async def validate_credentials(
        self, secret_id: str, secret_value: str, metadata: dict[str, Any]
    ) -> bool:
        """
        Validate OAuth token by making a test API call.

        Args:
            secret_id: OAuth credential identifier
            secret_value: Access token to test
            metadata: Provider details

        Returns:
            True if token is valid
        """
        provider = metadata.get("provider", "generic")
        validation_url = self._get_validation_url(provider, metadata)

        if not validation_url:
            logger.warning(f"No validation URL for provider {provider}")
            return True

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    validation_url,
                    headers={"Authorization": f"Bearer {secret_value}"},
                    timeout=10.0,
                )

                if response.status_code == 200:
                    logger.info(f"Validated OAuth token for {secret_id}")
                    return True
                elif response.status_code == 401:
                    logger.error(f"OAuth token invalid for {secret_id}")
                    return False
                else:
                    logger.warning(
                        f"Unexpected status {response.status_code} validating {secret_id}"
                    )
                    return response.status_code < 400

        except ImportError:
            logger.warning("httpx not installed, assuming token valid")
            return True
        except Exception as e:
            logger.error(f"OAuth validation error for {secret_id}: {e}")
            return False

    def _get_validation_url(self, provider: str, metadata: dict[str, Any]) -> str | None:
        """Get the validation endpoint URL for a provider."""
        if "validation_url" in metadata:
            return metadata["validation_url"]

        validation_urls = {
            "google": "https://www.googleapis.com/oauth2/v1/tokeninfo",
            "microsoft": "https://graph.microsoft.com/v1.0/me",
            "github": "https://api.github.com/user",
            "slack": "https://slack.com/api/auth.test",
            "discord": "https://discord.com/api/users/@me",
        }

        return validation_urls.get(provider)

    async def revoke_old_credentials(
        self, secret_id: str, old_value: str, metadata: dict[str, Any]
    ) -> bool:
        """
        Revoke old OAuth token if provider supports it.

        Args:
            secret_id: OAuth credential identifier
            old_value: Old access token to revoke
            metadata: Provider details

        Returns:
            True if revocation succeeded or not supported
        """
        provider = metadata.get("provider", "generic")
        revoke_url = self._get_revoke_url(provider, metadata)

        if not revoke_url:
            logger.info(f"Token revocation not supported for {provider}")
            return True

        client_id = metadata.get("client_id")
        client_secret = metadata.get("client_secret")

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    revoke_url,
                    data={
                        "token": old_value,
                        "client_id": client_id,
                        "client_secret": client_secret,
                    },
                    timeout=10.0,
                )

                if response.status_code < 400:
                    logger.info(f"Revoked old OAuth token for {secret_id}")
                    return True
                else:
                    logger.warning(
                        f"Token revocation returned {response.status_code} for {secret_id}"
                    )
                    return False

        except ImportError:
            logger.warning("httpx not installed, skipping revocation")
            return True
        except Exception as e:
            logger.error(f"OAuth revocation error for {secret_id}: {e}")
            return False

    def _get_revoke_url(self, provider: str, metadata: dict[str, Any]) -> str | None:
        """Get the revocation endpoint URL for a provider."""
        if "revoke_url" in metadata:
            return metadata["revoke_url"]

        revoke_urls = {
            "google": "https://oauth2.googleapis.com/revoke",
            "github": None,  # GitHub doesn't support token revocation
            "slack": "https://slack.com/api/auth.revoke",
            "discord": "https://discord.com/api/oauth2/token/revoke",
        }

        return revoke_urls.get(provider)
