"""Server composition for higher-layer capabilities consumed by billing."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def init_billing_edge_adapters(*, strict: bool = False) -> bool:
    """Register audit, chat, knowledge, and notification capabilities for billing."""
    try:
        from aragora.audit.billing_sink import register_billing_audit_sink
        from aragora.connectors.chat.budget_alert_sink import register_budget_alert_sinks
        from aragora.knowledge.mound.cost_adapter_registration import (
            register_billing_cost_adapter,
        )
        from aragora.notifications.billing_sink import register_billing_notification_sink

        register_billing_audit_sink()
        register_budget_alert_sinks()
        register_billing_cost_adapter()
        register_billing_notification_sink()
        logger.info("Billing edge adapters registered")
        return True
    except (ImportError, RuntimeError, OSError, ConnectionError, ValueError, TypeError) as exc:
        if strict:
            raise RuntimeError("Billing edge adapter registration failed") from exc
        logger.warning("Billing edge adapter registration failed: %s", exc)
        return False


__all__ = ["init_billing_edge_adapters"]
