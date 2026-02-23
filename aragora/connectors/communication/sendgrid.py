"""SendGrid Connector.

Provides integration with SendGrid for email activity and template management.

Searches:
- Email activity (messages sent, delivered, opened, etc.)
- Templates (dynamic and legacy transactional templates)

Environment Variables:
- ``SENDGRID_API_KEY`` -- SendGrid API key with Email Activity read scope.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any
from urllib.parse import quote

import httpx

from aragora.connectors.base import BaseConnector, Evidence
from aragora.connectors.exceptions import ConnectorError
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
    """SendGrid connector for email activity and template management."""

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
        """Search SendGrid for email activity or templates.

        Uses the Email Activity API to find messages matching a query
        (by subject, recipient, or status). Pass ``search_type="templates"``
        to search dynamic transactional templates instead.

        Args:
            query: Search term (matched against email subject/recipient).
            limit: Maximum number of results to return (capped at 50).
            **kwargs: Optional ``search_type`` ("activity" or "templates").
                      Defaults to "activity".

        Returns:
            List of Evidence objects from matching SendGrid records.
        """
        if not self._configured:
            logger.debug("SendGrid connector not configured")
            return []

        sanitized = _sanitize_query(query)
        if not sanitized.strip():
            return []

        search_type = kwargs.get("search_type", "activity")
        capped_limit = min(limit, 50)

        if search_type == "templates":
            return await self._search_templates(sanitized, capped_limit)
        return await self._search_activity(sanitized, capped_limit)

    async def _search_activity(self, query: str, limit: int) -> list[Evidence]:
        """Search email activity via the SendGrid Messages API."""
        # The Email Activity API accepts an RFC-5321-like query string.
        # We search by subject containing the query term.
        sg_query = f'subject="{query}"'

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_SG_API_BASE}/messages",
                    headers=self._get_headers(),
                    params={"query": sg_query, "limit": limit},
                )
                resp.raise_for_status()
                return resp.json()

        try:
            data = await self._request_with_retry(_do_request, "search_activity")
        except (ConnectorError, httpx.HTTPError, OSError, ValueError):
            logger.warning("SendGrid activity search failed", exc_info=True)
            return []

        results: list[Evidence] = []
        for msg in data.get("messages", [])[:limit]:
            msg_id = msg.get("msg_id", "")
            subject = msg.get("subject", "(no subject)")
            to_email = msg.get("to_email", "")
            from_email = msg.get("from_email", "")
            status = msg.get("status", "unknown")
            last_event_time = msg.get("last_event_time", "")

            results.append(
                Evidence(
                    id=f"sg_msg_{msg_id}",
                    source_type=self.source_type,
                    source_id=f"sendgrid://messages/{msg_id}",
                    content=(
                        f'Email: "{subject}" from {from_email} to {to_email} (status: {status})'
                    ),
                    title=f"Email: {subject}",
                    url=f"https://app.sendgrid.com/email_activity/{quote(msg_id, safe='')}",
                    author=from_email,
                    created_at=last_event_time,
                    confidence=0.7,
                    freshness=self.calculate_freshness(last_event_time) if last_event_time else 1.0,
                    authority=0.6,
                    metadata={
                        "type": "email_activity",
                        "msg_id": msg_id,
                        "subject": subject,
                        "to_email": to_email,
                        "from_email": from_email,
                        "status": status,
                        "last_event_time": last_event_time,
                    },
                )
            )
        return results

    async def _search_templates(self, query: str, limit: int) -> list[Evidence]:
        """Search transactional templates via the SendGrid Templates API."""

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_SG_API_BASE}/templates",
                    headers=self._get_headers(),
                    params={"generations": "dynamic", "page_size": limit},
                )
                resp.raise_for_status()
                return resp.json()

        try:
            data = await self._request_with_retry(_do_request, "search_templates")
        except (ConnectorError, httpx.HTTPError, OSError, ValueError):
            logger.warning("SendGrid template search failed", exc_info=True)
            return []

        results: list[Evidence] = []
        query_lower = query.lower()
        for tpl in data.get("result", data.get("templates", [])):
            tpl_id = tpl.get("id", "")
            tpl_name = tpl.get("name", "")
            updated_at = tpl.get("updated_at", "")

            # Client-side filter by name since the API has no search param
            if query_lower and query_lower not in tpl_name.lower():
                continue

            versions = tpl.get("versions", [])
            active_version = next(
                (v for v in versions if v.get("active")),
                versions[0] if versions else {},
            )
            subject = active_version.get("subject", "")

            results.append(
                Evidence(
                    id=f"sg_tpl_{tpl_id}",
                    source_type=self.source_type,
                    source_id=f"sendgrid://templates/{tpl_id}",
                    content=f'Template: "{tpl_name}" (subject: {subject})',
                    title=f"Template: {tpl_name}",
                    url=f"https://mc.sendgrid.com/dynamic-templates/{tpl_id}",
                    created_at=updated_at,
                    confidence=0.75,
                    freshness=self.calculate_freshness(updated_at) if updated_at else 1.0,
                    authority=0.7,
                    metadata={
                        "type": "template",
                        "template_id": tpl_id,
                        "name": tpl_name,
                        "subject": subject,
                        "updated_at": updated_at,
                        "version_count": len(versions),
                    },
                )
            )

            if len(results) >= limit:
                break

        return results

    async def fetch(self, evidence_id: str, **kwargs: Any) -> Evidence | None:
        """Fetch a specific email message or template from SendGrid.

        The ``evidence_id`` should be in one of the following formats:
        - ``sg_msg_<msg_id>`` -- fetches an email activity message
        - ``sg_tpl_<template_id>`` -- fetches a transactional template
        """
        if not self._configured:
            return None

        cached = self._cache_get(evidence_id)
        if cached is not None:
            return cached

        if evidence_id.startswith("sg_tpl_"):
            return await self._fetch_template(evidence_id[len("sg_tpl_") :], evidence_id)
        elif evidence_id.startswith("sg_msg_"):
            return await self._fetch_message(evidence_id[len("sg_msg_") :], evidence_id)

        return None

    async def _fetch_message(self, msg_id: str, evidence_id: str) -> Evidence | None:
        """Fetch a single email message from the Activity API."""

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_SG_API_BASE}/messages/{quote(msg_id, safe='')}",
                    headers=self._get_headers(),
                )
                resp.raise_for_status()
                return resp.json()

        try:
            msg = await self._request_with_retry(_do_request, "fetch_message")
        except (ConnectorError, httpx.HTTPError, OSError, ValueError):
            logger.warning("SendGrid message fetch failed for %s", evidence_id, exc_info=True)
            return None

        subject = msg.get("subject", "(no subject)")
        to_email = msg.get("to_email", "")
        from_email = msg.get("from_email", "")
        status = msg.get("status", "unknown")
        last_event_time = msg.get("last_event_time", "")
        events = msg.get("events", [])

        evidence = Evidence(
            id=evidence_id,
            source_type=self.source_type,
            source_id=f"sendgrid://messages/{msg_id}",
            content=(
                f'Email: "{subject}" from {from_email} to {to_email} '
                f"(status: {status}, events: {len(events)})"
            ),
            title=f"Email: {subject}",
            url=f"https://app.sendgrid.com/email_activity/{quote(msg_id, safe='')}",
            author=from_email,
            created_at=last_event_time,
            confidence=0.75,
            freshness=self.calculate_freshness(last_event_time) if last_event_time else 1.0,
            authority=0.6,
            metadata={
                "type": "email_activity",
                "msg_id": msg_id,
                "subject": subject,
                "to_email": to_email,
                "from_email": from_email,
                "status": status,
                "last_event_time": last_event_time,
                "event_count": len(events),
            },
        )
        self._cache_put(evidence.id, evidence)
        return evidence

    async def _fetch_template(self, template_id: str, evidence_id: str) -> Evidence | None:
        """Fetch a single transactional template."""

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{_SG_API_BASE}/templates/{quote(template_id, safe='')}",
                    headers=self._get_headers(),
                )
                resp.raise_for_status()
                return resp.json()

        try:
            tpl = await self._request_with_retry(_do_request, "fetch_template")
        except (ConnectorError, httpx.HTTPError, OSError, ValueError):
            logger.warning("SendGrid template fetch failed for %s", evidence_id, exc_info=True)
            return None

        tpl_name = tpl.get("name", "")
        updated_at = tpl.get("updated_at", "")
        versions = tpl.get("versions", [])
        active_version = next(
            (v for v in versions if v.get("active")),
            versions[0] if versions else {},
        )
        subject = active_version.get("subject", "")
        html_content = active_version.get("html_content", "")

        evidence = Evidence(
            id=evidence_id,
            source_type=self.source_type,
            source_id=f"sendgrid://templates/{template_id}",
            content=(f'Template: "{tpl_name}" (subject: {subject})\n\n{html_content[:2000]}'),
            title=f"Template: {tpl_name}",
            url=f"https://mc.sendgrid.com/dynamic-templates/{template_id}",
            created_at=updated_at,
            confidence=0.8,
            freshness=self.calculate_freshness(updated_at) if updated_at else 1.0,
            authority=0.7,
            metadata={
                "type": "template",
                "template_id": template_id,
                "name": tpl_name,
                "subject": subject,
                "updated_at": updated_at,
                "version_count": len(versions),
            },
        )
        self._cache_put(evidence.id, evidence)
        return evidence
