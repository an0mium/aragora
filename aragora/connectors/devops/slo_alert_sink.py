"""PagerDuty adapter for observability SLO alerts."""

from __future__ import annotations

import logging
from typing import Any

from aragora.connectors.devops.pagerduty import PagerDutyError
from aragora.observability.slo_alert_bridge import (
    SLOAlertConfig,
    register_pagerduty_alert_sink,
)

logger = logging.getLogger(__name__)


class PagerDutySLOAlertSink:
    """Translate the observability sink contract to PagerDuty requests."""

    def __init__(self, config: SLOAlertConfig) -> None:
        self._config = config
        self._connector: Any | None = None

    def _get_connector(self) -> Any:
        if self._connector is None:
            from aragora.connectors.devops.pagerduty import (
                PagerDutyConnector,
                PagerDutyCredentials,
            )

            credentials = PagerDutyCredentials(
                api_key=self._config.pagerduty_api_key or "",
                email=self._config.pagerduty_email,
            )
            self._connector = PagerDutyConnector(credentials)
        return self._connector

    async def create_incident(
        self,
        *,
        title: str,
        service_id: str,
        urgency: str,
        description: str,
        incident_key: str | None,
    ) -> str | None:
        """Create a PagerDuty incident and return its identifier."""
        try:
            from aragora.connectors.devops.pagerduty import (
                IncidentCreateRequest,
                IncidentUrgency,
            )

            request = IncidentCreateRequest(
                title=title,
                service_id=service_id,
                urgency=IncidentUrgency(urgency),
                description=description,
                incident_key=incident_key,
            )
            incident = await self._get_connector().create_incident(request)
            return incident.id
        except ImportError:
            logger.debug("PagerDuty connector not available")
        except (
            PagerDutyError,
            OSError,
            ConnectionError,
            TimeoutError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as e:
            logger.error("PagerDuty incident delivery failed: %s", e)
        return None

    async def resolve_incident(self, incident_id: str, resolution: str) -> bool:
        """Resolve a PagerDuty incident."""
        try:
            await self._get_connector().resolve_incident(
                incident_id,
                resolution=resolution,
            )
            return True
        except (
            PagerDutyError,
            OSError,
            ConnectionError,
            TimeoutError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as e:
            logger.error("PagerDuty incident resolution failed: %s", e)
            return False


def register_slo_alert_sink() -> None:
    """Register this adapter with the observability-owned sink contract."""
    register_pagerduty_alert_sink(PagerDutySLOAlertSink)


register_slo_alert_sink()


__all__ = ["PagerDutySLOAlertSink", "register_slo_alert_sink"]
