"""
WooCommerce webhook handling.

Provides webhook CRUD operations and signature verification for
WooCommerce webhook payloads (HMAC-SHA256, base64-encoded).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from aragora.connectors.exceptions import (
    ConnectorAPIError,
    ConnectorCircuitOpenError,
    ConnectorTimeoutError,
)

from aragora.connectors.ecommerce.woocommerce.models import validate_id

logger = logging.getLogger(__name__)


async def get_webhooks(connector) -> list[dict[str, Any]]:
    """Get all registered webhooks.

    Args:
        connector: A WooCommerceConnector instance

    Returns:
        List of webhook data
    """
    try:
        data = await connector._request("GET", "webhooks")
        return data if isinstance(data, list) else []
    except (ConnectorAPIError, OSError) as e:
        logger.error(f"Failed to get webhooks: {e}")
        return []


async def create_webhook(
    connector,
    name: str,
    topic: str,
    delivery_url: str,
    secret: str = "",
    status: str = "active",
) -> Optional[dict[str, Any]]:
    """Register a new webhook.

    Args:
        connector: A WooCommerceConnector instance
        name: Webhook name
        topic: Event topic (e.g., order.created, product.updated)
        delivery_url: URL to receive webhook payloads
        secret: Secret for payload signing
        status: Webhook status (active, paused, disabled)

    Returns:
        Created webhook data or None on failure
    """
    try:
        webhook_data = {
            "name": name,
            "topic": topic,
            "delivery_url": delivery_url,
            "secret": secret,
            "status": status,
        }
        data = await connector._request("POST", "webhooks", json_data=webhook_data)
        return data
    except (ConnectorAPIError, OSError) as e:
        logger.error(f"Failed to create webhook {name}: {e}")
        return None


async def delete_webhook(connector, webhook_id: int, force: bool = True) -> bool:
    """Delete a webhook.

    Args:
        connector: A WooCommerceConnector instance
        webhook_id: Webhook ID (must be alphanumeric)
        force: Permanently delete

    Returns:
        True if successful

    Raises:
        ConnectorValidationError: If webhook_id contains invalid characters
    """
    validate_id(webhook_id, "webhook_id")
    try:
        await connector._request(
            "DELETE",
            f"webhooks/{webhook_id}",
            params={"force": force},
        )
        return True
    except (ConnectorAPIError, ConnectorCircuitOpenError, ConnectorTimeoutError, OSError) as e:
        logger.error(f"Failed to delete webhook {webhook_id}: {e}")
        return False


def verify_webhook_signature(
    payload: bytes,
    signature: str,
    secret: str | None = None,
    get_webhook_secret=None,
) -> bool:
    """Verify webhook payload signature.

    WooCommerce uses base64-encoded HMAC-SHA256 signatures.

    Args:
        payload: Raw webhook payload bytes
        signature: X-WC-Webhook-Signature header value
        secret: Webhook secret (optional)
        get_webhook_secret: Callable that returns the webhook secret if secret is None

    Returns:
        True if signature is valid
    """
    effective_secret = secret
    if effective_secret is None and get_webhook_secret is not None:
        effective_secret = get_webhook_secret()

    if not effective_secret:
        import os

        env = os.environ.get("ARAGORA_ENV", "production").lower()
        is_production = env not in ("development", "dev", "local", "test")
        if is_production:
            logger.error(
                "SECURITY: WooCommerce webhook secret not configured in production. "
                "Rejecting webhook to prevent signature bypass."
            )
            return False
        logger.warning(
            "WooCommerce webhook secret not configured - skipping verification. "
            "This is only acceptable in development!"
        )
        return True

    import base64
    import hashlib
    import hmac

    expected = base64.b64encode(
        hmac.new(effective_secret.encode(), payload, hashlib.sha256).digest()
    ).decode()
    return hmac.compare_digest(expected, signature)


__all__ = [
    "create_webhook",
    "delete_webhook",
    "get_webhooks",
    "verify_webhook_signature",
]
