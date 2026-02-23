"""SendGrid Connector.

Provides integration with SendGrid for email delivery and templates.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import httpx

from aragora.connectors.base import BaseConnector, Evidence
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)

CONFIG_ENV_VARS = ("SENDGRID_API_KEY",)

_SAFE_QUERY_RE = re.compile(r"[^\w\s@.\-+]")
MAX_QUERY_LENGTH = 500

_SG_API_BASE = "https://api.sendgrid.com/v3"


def _sanitize_query(query: str) -> str:
    """Sanitize query to prevent injection."""
    query = query[:MAX_QUERY_LENGTH]
    return _SAFE_QUERY_RE.sub("", query)


class SendGridConnector(BaseConnector):
    """SendGrid connector for email delivery and template management."""

    def __init__(self) -> None:
        super().__init__()
        self._configured = all(os.environ.get(v) for v in CONFIG_ENV_VARS)

    @property
    def name(self) -> str:
        return "sendgrid"

    @property
    def source_type(self) -> SourceType:
        return SourceType.EXTERNAL_API

    @property
    def is_configured(self) -> bool:
        return self._configured

    def _get_headers(self) -> dict[str, str]:
        api_key = os.environ.get("SENDGRID_API_KEY", "")
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[Evidence]:
        """Search SendGrid activity for relevant email data."""
        if not self._configured:
            logger.debug("SendGrid connector not configured")
            return []

        sanitized = _sanitize_query(query)
        if not sanitized.strip():
            return []

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_SG_API_BASE}/messages",
                    headers=self._get_headers(),
                    params={"query": sanitized, "limit": limit},
                )
                resp.raise_for_status()
                return resp.json()

        try:
            data = await self._request_with_retry(_do_request, "search")
        except Exception:
            logger.warning("SendGrid search failed", exc_info=True)
            return []

        results: list[Evidence] = []
        messages = data.get("messages", [])
        for msg in messages[:limit]:
            msg_id = msg.get("msg_id", "")
            subject = msg.get("subject", "")
            to_email = msg.get("to_email", "")
            status = msg.get("status", "")
            results.append(
                Evidence(
                    id=f"sg_msg_{msg_id}",
                    source_type=self.source_type,
                    source_id=f"sendgrid://messages/{msg_id}",
                    content=f"Email to {to_email}: {subject} (status: {status})",
                    title=subject or f"Message {msg_id}",
                    confidence=0.7,
                    freshness=1.0,
                    authority=0.6,
                    metadata={"msg_id": msg_id, "to_email": to_email, "status": status},
                )
            )
        return results

    async def fetch(self, evidence_id: str, **kwargs: Any) -> Evidence | None:
        """Fetch a specific email or template from SendGrid."""
        if not self._configured:
            return None

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_SG_API_BASE}/messages/{evidence_id}",
                    headers=self._get_headers(),
                )
                resp.raise_for_status()
                return resp.json()

        try:
            data = await self._request_with_retry(_do_request, "fetch")
        except Exception:
            logger.warning("SendGrid fetch failed", exc_info=True)
            return None

        msg_id = data.get("msg_id", evidence_id)
        subject = data.get("subject", "")
        to_email = data.get("to_email", "")
        status = data.get("status", "")
        return Evidence(
            id=f"sg_msg_{msg_id}",
            source_type=self.source_type,
            source_id=f"sendgrid://messages/{msg_id}",
            content=f"Email to {to_email}: {subject} (status: {status})",
            title=subject or f"Message {msg_id}",
            confidence=0.7,
            freshness=1.0,
            authority=0.6,
            metadata={"msg_id": msg_id, "to_email": to_email, "status": status},
        )
