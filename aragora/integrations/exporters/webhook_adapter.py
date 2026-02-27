"""Generic webhook adapter for Decision Plan export.

Sends TicketData as JSON POSTs to any webhook URL (n8n, Zapier, Make,
or custom endpoints).  Supports HMAC-SHA256 signing and batch export.

Usage:
    adapter = WebhookAdapter(
        url="https://hooks.zapier.com/hooks/catch/123/abc",
        secret="optional-hmac-secret",
    )
    result = await adapter.export_ticket(ticket)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from aiohttp import ClientTimeout

from aragora.integrations.exporters.base import (
    ExportAdapter,
    ExportReceipt,
    TicketData,
)

logger = logging.getLogger(__name__)

# Default timeout for webhook calls
_DEFAULT_TIMEOUT = ClientTimeout(total=15, connect=5, sock_read=10)


def _sign_payload(secret: str, body: bytes) -> str:
    """Compute HMAC-SHA256 signature for the webhook payload."""
    return "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


@dataclass
class WebhookAdapterConfig:
    """Configuration for the webhook adapter."""

    url: str
    secret: str = ""
    timeout: ClientTimeout = field(default_factory=lambda: _DEFAULT_TIMEOUT)
    headers: dict[str, str] = field(default_factory=dict)
    # When True, sends all tickets in a single POST as a JSON array
    batch_mode: bool = False


class WebhookAdapter(ExportAdapter):
    """Export adapter that POSTs ticket data to a generic webhook URL.

    Compatible with n8n, Zapier, Make, and any HTTP endpoint that
    accepts JSON POST requests.
    """

    def __init__(
        self,
        url: str,
        secret: str = "",
        *,
        timeout: ClientTimeout | None = None,
        headers: dict[str, str] | None = None,
        batch_mode: bool = False,
    ) -> None:
        self._config = WebhookAdapterConfig(
            url=url,
            secret=secret,
            timeout=timeout or _DEFAULT_TIMEOUT,
            headers=headers or {},
            batch_mode=batch_mode,
        )
        self._session: aiohttp.ClientSession | None = None

    @property
    def name(self) -> str:
        return "webhook"

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._config.timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # -- Single ticket export ------------------------------------------------

    async def export_ticket(self, ticket: TicketData) -> dict[str, Any]:
        """POST a single ticket to the webhook URL."""
        payload = self._format_payload(ticket)
        return await self._post(payload)

    # -- Batch override ------------------------------------------------------

    async def export_tickets(self, tickets: list[TicketData]) -> ExportReceipt:
        """Export tickets, optionally as a single batch POST."""
        if self._config.batch_mode and tickets:
            return await self._export_batch(tickets)
        # Default: sequential per-ticket export from base class
        return await super().export_tickets(tickets)

    async def _export_batch(self, tickets: list[TicketData]) -> ExportReceipt:
        """Send all tickets in a single POST."""
        receipt = ExportReceipt(
            adapter_name=self.name,
            plan_id=tickets[0].plan_id if tickets else "",
            debate_id=tickets[0].debate_id if tickets else "",
        )
        payload = {
            "event": "decision_plan_export",
            "timestamp": time.time(),
            "plan_id": tickets[0].plan_id if tickets else "",
            "debate_id": tickets[0].debate_id if tickets else "",
            "ticket_count": len(tickets),
            "tickets": [self._format_payload(t) for t in tickets],
        }
        try:
            result = await self._post(payload)
            if result.get("success"):
                receipt.tickets_exported = len(tickets)
                receipt.mark_success()
            else:
                receipt.tickets_failed = len(tickets)
                receipt.mark_failed(result.get("error", "Batch export failed"))
        except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
            receipt.tickets_failed = len(tickets)
            receipt.mark_failed(str(exc))

        receipt.ticket_results = [
            {"task_id": t.task_id, "title": t.title, "success": receipt.tickets_exported > 0}
            for t in tickets
        ]
        return receipt

    # -- Payload formatting --------------------------------------------------

    @staticmethod
    def _format_payload(ticket: TicketData) -> dict[str, Any]:
        """Format a TicketData as the JSON payload for the webhook."""
        payload: dict[str, Any] = {
            "event": "decision_plan_ticket",
            "timestamp": time.time(),
            "ticket": ticket.to_dict(),
        }
        if ticket.acceptance_criteria:
            payload["ticket"]["acceptance_criteria"] = ticket.acceptance_criteria
        return payload

    # -- HTTP POST -----------------------------------------------------------

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the HTTP POST with optional HMAC signing."""
        body = json.dumps(payload, default=str).encode("utf-8")

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": "Aragora-DecisionExporter/1.0",
            "X-Aragora-Event": "decision_plan_export",
            **self._config.headers,
        }
        if self._config.secret:
            headers["X-Aragora-Signature"] = _sign_payload(self._config.secret, body)

        session = await self._get_session()
        try:
            async with session.post(
                self._config.url,
                data=body,
                headers=headers,
            ) as response:
                if 200 <= response.status < 300:
                    return {"success": True, "status_code": response.status}
                else:
                    return {
                        "success": False,
                        "status_code": response.status,
                        "error": f"HTTP {response.status}",
                    }
        except aiohttp.ClientError as exc:
            logger.warning("Webhook POST failed: %s", exc)
            return {"success": False, "error": str(exc)}
