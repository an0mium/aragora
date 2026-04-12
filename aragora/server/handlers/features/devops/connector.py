"""PagerDuty connector instance management."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_connector_instances: dict[str, Any] = {}  # tenant_id -> PagerDutyConnector
_active_contexts: dict[str, Any] = {}  # tenant_id -> context manager


def _load_pagerduty_types():
    from aragora.connectors.devops.pagerduty import (
        PagerDutyConnector,
        PagerDutyCredentials,
    )

    return PagerDutyConnector, PagerDutyCredentials


def _get_pagerduty_env() -> tuple[str | None, str | None, str | None]:
    import os

    return (
        os.getenv("PAGERDUTY_API_KEY"),
        os.getenv("PAGERDUTY_EMAIL"),
        os.getenv("PAGERDUTY_WEBHOOK_SECRET"),
    )


async def get_pagerduty_connector(tenant_id: str):
    """Get or create PagerDuty connector for tenant."""
    if tenant_id not in _connector_instances:
        try:
            PagerDutyConnector, PagerDutyCredentials = _load_pagerduty_types()
            api_key, email, webhook_secret = _get_pagerduty_env()

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
            logger.error("Failed to initialize PagerDuty connector: %s", e)
            return None

    return _connector_instances.get(tenant_id)


def clear_connector_instances() -> None:
    """Clear all connector instances (for testing)."""
    _connector_instances.clear()
    _active_contexts.clear()
