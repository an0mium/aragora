"""Twilio Connector.

Provides integration with Twilio for SMS and voice communications.
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

CONFIG_ENV_VARS = ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN")

_SAFE_QUERY_RE = re.compile(r"[^\w\s@.\-+]")
MAX_QUERY_LENGTH = 500

_TWILIO_API_BASE = "https://api.twilio.com/2010-04-01"


def _sanitize_query(query: str) -> str:
    """Sanitize query to prevent injection."""
    query = query[:MAX_QUERY_LENGTH]
    return _SAFE_QUERY_RE.sub("", query)


class TwilioConnector(BaseConnector):
    """Twilio connector for SMS and voice communications."""

    def __init__(self) -> None:
        super().__init__()
        self._configured = all(os.environ.get(v) for v in CONFIG_ENV_VARS)

    @property
    def name(self) -> str:
        return "twilio"

    @property
    def source_type(self) -> SourceType:
        return SourceType.EXTERNAL_API

    @property
    def is_configured(self) -> bool:
        return self._configured

    def _get_auth(self) -> tuple[str, str]:
        account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
        return (account_sid, auth_token)

    def _get_account_sid(self) -> str:
        return os.environ.get("TWILIO_ACCOUNT_SID", "")

    async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[Evidence]:
        """Search Twilio message logs for relevant data."""
        if not self._configured:
            logger.debug("Twilio connector not configured")
            return []

        sanitized = _sanitize_query(query)
        if not sanitized.strip():
            return []

        sid = self._get_account_sid()
        auth = self._get_auth()

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_TWILIO_API_BASE}/Accounts/{sid}/Messages.json",
                    auth=auth,
                    params={"To": sanitized, "PageSize": limit},
                )
                resp.raise_for_status()
                return resp.json()

        try:
            data = await self._request_with_retry(_do_request, "search")
        except Exception:
            logger.warning("Twilio search failed", exc_info=True)
            return []

        results: list[Evidence] = []
        messages = data.get("messages", [])
        for msg in messages[:limit]:
            msg_sid = msg.get("sid", "")
            body = msg.get("body", "")
            from_num = msg.get("from", "")
            to_num = msg.get("to", "")
            status = msg.get("status", "")
            results.append(
                Evidence(
                    id=f"twilio_msg_{msg_sid}",
                    source_type=self.source_type,
                    source_id=f"twilio://messages/{msg_sid}",
                    content=f"SMS from {from_num} to {to_num}: {body}",
                    title=f"Message {msg_sid}",
                    confidence=0.7,
                    freshness=1.0,
                    authority=0.6,
                    metadata={
                        "sid": msg_sid,
                        "from": from_num,
                        "to": to_num,
                        "status": status,
                    },
                )
            )
        return results

    async def fetch(self, evidence_id: str, **kwargs: Any) -> Evidence | None:
        """Fetch a specific message from Twilio."""
        if not self._configured:
            return None

        sid = self._get_account_sid()
        auth = self._get_auth()

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_TWILIO_API_BASE}/Accounts/{sid}/Messages/{evidence_id}.json",
                    auth=auth,
                )
                resp.raise_for_status()
                return resp.json()

        try:
            data = await self._request_with_retry(_do_request, "fetch")
        except Exception:
            logger.warning("Twilio fetch failed", exc_info=True)
            return None

        msg_sid = data.get("sid", evidence_id)
        body = data.get("body", "")
        from_num = data.get("from", "")
        to_num = data.get("to", "")
        status = data.get("status", "")
        return Evidence(
            id=f"twilio_msg_{msg_sid}",
            source_type=self.source_type,
            source_id=f"twilio://messages/{msg_sid}",
            content=f"SMS from {from_num} to {to_num}: {body}",
            title=f"Message {msg_sid}",
            confidence=0.7,
            freshness=1.0,
            authority=0.6,
            metadata={
                "sid": msg_sid,
                "from": from_num,
                "to": to_num,
                "status": status,
            },
        )
