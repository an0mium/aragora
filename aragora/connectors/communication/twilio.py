"""Twilio Connector.

Provides integration with Twilio for SMS and voice message history.

Searches:
- SMS/MMS messages (inbound and outbound)
- Call records (voice call logs)

Environment Variables:
- ``TWILIO_ACCOUNT_SID`` -- Twilio Account SID.
- ``TWILIO_AUTH_TOKEN`` -- Twilio Auth Token.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import httpx

from aragora.connectors.base import BaseConnector, Evidence
from aragora.connectors.exceptions import ConnectorError
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
    """Twilio connector for SMS and voice message history."""

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
        """Search Twilio for SMS messages or call records.

        The query is matched against the ``To`` or ``From`` phone number
        fields. Pass ``search_type="calls"`` to search call logs instead
        of messages.

        Args:
            query: Phone number or fragment to filter by (e.g. "+1555").
            limit: Maximum number of results (capped at 100).
            **kwargs: Optional ``search_type`` ("messages" or "calls").
                      Defaults to "messages".

        Returns:
            List of Evidence objects from matching Twilio records.
        """
        if not self._configured:
            logger.debug("Twilio connector not configured")
            return []

        sanitized = _sanitize_query(query)
        if not sanitized.strip():
            return []

        search_type = kwargs.get("search_type", "messages")
        capped_limit = min(limit, 100)

        if search_type == "calls":
            return await self._search_calls(sanitized, capped_limit)
        return await self._search_messages(sanitized, capped_limit)

    async def _search_messages(self, query: str, limit: int) -> list[Evidence]:
        """Search SMS/MMS messages via the Twilio Messages API."""
        account_sid = self._get_account_sid()
        auth = self._get_auth()

        # Twilio's list filter accepts To or From as exact E.164 numbers.
        # If the query looks like a phone number, filter by To; otherwise
        # we fetch recent messages and filter client-side.
        params: dict[str, Any] = {"PageSize": limit}
        if query.startswith("+") or query.replace("-", "").replace(" ", "").isdigit():
            params["To"] = query

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_TWILIO_API_BASE}/Accounts/{account_sid}/Messages.json",
                    auth=auth,
                    params=params,
                )
                resp.raise_for_status()
                return resp.json()

        try:
            data = await self._request_with_retry(_do_request, "search_messages")
        except (ConnectorError, httpx.HTTPError, OSError, ValueError):
            logger.warning("Twilio message search failed", exc_info=True)
            return []

        results: list[Evidence] = []
        query_lower = query.lower()
        for msg in data.get("messages", [])[:limit]:
            msg_sid = msg.get("sid", "")
            body = msg.get("body", "")
            from_num = msg.get("from", "")
            to_num = msg.get("to", "")
            status = msg.get("status", "")
            direction = msg.get("direction", "")
            date_sent = msg.get("date_sent", "")

            # Client-side filter when we could not use API-level filtering
            if "To" not in params:
                combined = f"{body} {from_num} {to_num}".lower()
                if query_lower not in combined:
                    continue

            results.append(
                Evidence(
                    id=f"tw_msg_{msg_sid}",
                    source_type=self.source_type,
                    source_id=f"twilio://messages/{msg_sid}",
                    content=f"SMS from {from_num} to {to_num}: {body[:500]}",
                    title=f"SMS {direction}: {from_num} -> {to_num}",
                    url=f"https://console.twilio.com/us1/monitor/logs/sms/{msg_sid}",
                    author=from_num,
                    created_at=date_sent,
                    confidence=0.75,
                    freshness=self.calculate_freshness(date_sent) if date_sent else 1.0,
                    authority=0.6,
                    metadata={
                        "type": "sms",
                        "sid": msg_sid,
                        "from": from_num,
                        "to": to_num,
                        "status": status,
                        "direction": direction,
                        "date_sent": date_sent,
                    },
                )
            )
        return results

    async def _search_calls(self, query: str, limit: int) -> list[Evidence]:
        """Search call records via the Twilio Calls API."""
        account_sid = self._get_account_sid()
        auth = self._get_auth()

        params: dict[str, Any] = {"PageSize": limit}
        if query.startswith("+") or query.replace("-", "").replace(" ", "").isdigit():
            params["To"] = query

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_TWILIO_API_BASE}/Accounts/{account_sid}/Calls.json",
                    auth=auth,
                    params=params,
                )
                resp.raise_for_status()
                return resp.json()

        try:
            data = await self._request_with_retry(_do_request, "search_calls")
        except (ConnectorError, httpx.HTTPError, OSError, ValueError):
            logger.warning("Twilio call search failed", exc_info=True)
            return []

        results: list[Evidence] = []
        query_lower = query.lower()
        for call in data.get("calls", [])[:limit]:
            call_sid = call.get("sid", "")
            from_num = call.get("from", "")
            to_num = call.get("to", "")
            status = call.get("status", "")
            direction = call.get("direction", "")
            duration = call.get("duration", "0")
            start_time = call.get("start_time", "")

            if "To" not in params:
                combined = f"{from_num} {to_num}".lower()
                if query_lower not in combined:
                    continue

            results.append(
                Evidence(
                    id=f"tw_call_{call_sid}",
                    source_type=self.source_type,
                    source_id=f"twilio://calls/{call_sid}",
                    content=(
                        f"Call from {from_num} to {to_num} "
                        f"(duration: {duration}s, status: {status})"
                    ),
                    title=f"Call {direction}: {from_num} -> {to_num}",
                    url=f"https://console.twilio.com/us1/monitor/logs/calls/{call_sid}",
                    author=from_num,
                    created_at=start_time,
                    confidence=0.7,
                    freshness=self.calculate_freshness(start_time) if start_time else 1.0,
                    authority=0.6,
                    metadata={
                        "type": "call",
                        "sid": call_sid,
                        "from": from_num,
                        "to": to_num,
                        "status": status,
                        "direction": direction,
                        "duration": duration,
                        "start_time": start_time,
                    },
                )
            )
        return results

    async def fetch(self, evidence_id: str, **kwargs: Any) -> Evidence | None:
        """Fetch a specific message or call record from Twilio.

        The ``evidence_id`` should be in one of the following formats:
        - ``tw_msg_<SID>`` -- fetches an SMS/MMS message
        - ``tw_call_<SID>`` -- fetches a call record
        """
        if not self._configured:
            return None

        cached = self._cache_get(evidence_id)
        if cached is not None:
            return cached

        if evidence_id.startswith("tw_call_"):
            return await self._fetch_call(evidence_id[len("tw_call_"):], evidence_id)
        elif evidence_id.startswith("tw_msg_"):
            return await self._fetch_message(evidence_id[len("tw_msg_"):], evidence_id)

        return None

    async def _fetch_message(self, msg_sid: str, evidence_id: str) -> Evidence | None:
        """Fetch a single SMS/MMS message by SID."""
        account_sid = self._get_account_sid()
        auth = self._get_auth()

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_TWILIO_API_BASE}/Accounts/{account_sid}/Messages/{msg_sid}.json",
                    auth=auth,
                )
                resp.raise_for_status()
                return resp.json()

        try:
            msg = await self._request_with_retry(_do_request, "fetch_message")
        except (ConnectorError, httpx.HTTPError, OSError, ValueError):
            logger.warning("Twilio message fetch failed for %s", evidence_id, exc_info=True)
            return None

        body = msg.get("body", "")
        from_num = msg.get("from", "")
        to_num = msg.get("to", "")
        status = msg.get("status", "")
        direction = msg.get("direction", "")
        date_sent = msg.get("date_sent", "")
        price = msg.get("price", "")
        num_media = msg.get("num_media", "0")

        evidence = Evidence(
            id=evidence_id,
            source_type=self.source_type,
            source_id=f"twilio://messages/{msg_sid}",
            content=f"SMS from {from_num} to {to_num}: {body[:2000]}",
            title=f"SMS {direction}: {from_num} -> {to_num}",
            url=f"https://console.twilio.com/us1/monitor/logs/sms/{msg_sid}",
            author=from_num,
            created_at=date_sent,
            confidence=0.8,
            freshness=self.calculate_freshness(date_sent) if date_sent else 1.0,
            authority=0.6,
            metadata={
                "type": "sms",
                "sid": msg_sid,
                "from": from_num,
                "to": to_num,
                "status": status,
                "direction": direction,
                "date_sent": date_sent,
                "price": price,
                "num_media": num_media,
            },
        )
        self._cache_put(evidence.id, evidence)
        return evidence

    async def _fetch_call(self, call_sid: str, evidence_id: str) -> Evidence | None:
        """Fetch a single call record by SID."""
        account_sid = self._get_account_sid()
        auth = self._get_auth()

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_TWILIO_API_BASE}/Accounts/{account_sid}/Calls/{call_sid}.json",
                    auth=auth,
                )
                resp.raise_for_status()
                return resp.json()

        try:
            call = await self._request_with_retry(_do_request, "fetch_call")
        except (ConnectorError, httpx.HTTPError, OSError, ValueError):
            logger.warning("Twilio call fetch failed for %s", evidence_id, exc_info=True)
            return None

        from_num = call.get("from", "")
        to_num = call.get("to", "")
        status = call.get("status", "")
        direction = call.get("direction", "")
        duration = call.get("duration", "0")
        start_time = call.get("start_time", "")
        price = call.get("price", "")

        evidence = Evidence(
            id=evidence_id,
            source_type=self.source_type,
            source_id=f"twilio://calls/{call_sid}",
            content=(
                f"Call from {from_num} to {to_num} "
                f"(duration: {duration}s, status: {status}, price: {price})"
            ),
            title=f"Call {direction}: {from_num} -> {to_num}",
            url=f"https://console.twilio.com/us1/monitor/logs/calls/{call_sid}",
            author=from_num,
            created_at=start_time,
            confidence=0.8,
            freshness=self.calculate_freshness(start_time) if start_time else 1.0,
            authority=0.6,
            metadata={
                "type": "call",
                "sid": call_sid,
                "from": from_num,
                "to": to_num,
                "status": status,
                "direction": direction,
                "duration": duration,
                "start_time": start_time,
                "price": price,
            },
        )
        self._cache_put(evidence.id, evidence)
        return evidence
