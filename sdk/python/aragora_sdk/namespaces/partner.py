"""
Partner API Namespace.

Provides methods for partner management:
- Partner registration and profile
- API key management
- Usage statistics
- Webhook configuration
- Rate limit information

Endpoints:
    POST   /api/v1/partners/register     - Register as partner
    GET    /api/v1/partners/me           - Get partner profile
    POST   /api/v1/partners/keys         - Create API key
    GET    /api/v1/partners/keys         - List API keys
    DELETE /api/v1/partners/keys/{id}    - Revoke API key
    GET    /api/v1/partners/usage        - Get usage statistics
    POST   /api/v1/partners/webhooks     - Configure webhook
    GET    /api/v1/partners/limits       - Get rate limits
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

class PartnerAPI:
    """
    Synchronous Partner API.

    Provides methods for partner registration, API key management,
    and usage tracking.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> # Register as a partner
        >>> result = client.partner.register(
        ...     name="My Company",
        ...     email="partner@example.com",
        ...     company="My Company Inc",
        ... )
        >>> # Create an API key
        >>> key = client.partner.create_api_key(
        ...     name="Production Key",
        ...     scopes=["debates:read", "debates:write"],
        ... )
        >>> print(f"API Key: {key['key']}")  # Only shown once!
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Registration

class AsyncPartnerAPI:
    """Asynchronous Partner API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # =========================================================================
    # Registration

