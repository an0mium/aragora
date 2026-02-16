"""PagerDuty connector instance management."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_connector_instances: dict[str, Any] = {}  # tenant_id -> PagerDutyConnector
_active_contexts: dict[str, Any] = {}  # tenant_id -> context manager


async def get_pagerduty_connector(tenant_id: str):
    """Get or create PagerDuty connector for tenant."""
    if tenant_id not in _connector_instances:
        try:
            import os

            from aragora.connectors.devops.pagerduty import (
                PagerDutyConnector,
                PagerDutyCredentials,
            )

            api_key = os.getenv("PAGERDUTY_API_KEY")
            email = os.getenv("PAGERDUTY_EMAIL")
            webhook_secret = os.getenv("PAGERDUTY_WEBHOOK_SECRET")

            if not api_key or not email:
                return None

            credentials = PagerDutyCredentials(
                api_key=api_key,
                email=email,
                webhook_secret=webhook_secret,
            )

            connector = PagerDutyConnector(credentials)
            # Enter context to initialize client
            await connector.__aenter__()
            _connector_instances[tenant_id] = connector
            _active_contexts[tenant_id] = connector

        except ImportError:
            return None
        except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
            logger.error(f"Failed to initialize PagerDuty connector: {e}")
            return None

    return _connector_instances.get(tenant_id)


def clear_connector_instances() -> None:
    """Clear all connector instances (for testing)."""
    _connector_instances.clear()
    _active_contexts.clear()
