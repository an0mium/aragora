"""Control-plane channel adapter for observability SLO alerts."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.observability.slo_alert_bridge import SLOAlertConfig

logger = logging.getLogger(__name__)


class ControlPlaneSLOAlertSink:
    """Translate primitive alert fields to control-plane notifications."""

    def __init__(self, config: SLOAlertConfig) -> None:
        self._config = config
        self._manager: Any | None = None

    def _get_manager(self) -> Any:
        if self._manager is None:
            from aragora.control_plane.channels import (
                ChannelConfig,
                NotificationChannel,
                NotificationManager,
            )

            manager = NotificationManager()
            manager.add_channel(
                ChannelConfig(
                    channel_type=NotificationChannel.SLACK,
                    slack_webhook_url=self._config.slack_webhook_url,
                    slack_channel=self._config.slack_channel,
                )
            )
            self._manager = manager
        return self._manager

    async def notify(
        self,
        *,
        event_type: str,
        title: str,
        body: str,
        priority: str,
        metadata: dict[str, Any],
    ) -> Any:
        """Deliver an observability alert through the channel manager."""
        from aragora.control_plane.channels import (
            NotificationEventType,
            NotificationPriority,
        )

        return await self._get_manager().notify(
            event_type=NotificationEventType(event_type),
            title=title,
            body=body,
            priority=NotificationPriority(priority),
            metadata=metadata,
        )


def register_slo_alert_sink() -> bool:
    """Register this adapter with the observability-owned sink contract."""
    try:
        from aragora.observability.slo_alert_bridge import register_channel_alert_sink
    except ImportError as e:
        logger.debug("Observability channel sink contract unavailable: %s", e)
        return False

    register_channel_alert_sink(ControlPlaneSLOAlertSink)
    return True


__all__ = ["ControlPlaneSLOAlertSink", "register_slo_alert_sink"]
