"""Control-plane channel adapter for observability SLO alerts."""

from __future__ import annotations

from typing import Any

from aragora.observability.slo_alert_bridge import (
    SLOAlertConfig,
    register_channel_alert_sink,
)


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


def register_slo_alert_sink() -> None:
    """Register this adapter with the observability-owned sink contract."""
    register_channel_alert_sink(ControlPlaneSLOAlertSink)


register_slo_alert_sink()


__all__ = ["ControlPlaneSLOAlertSink", "register_slo_alert_sink"]
