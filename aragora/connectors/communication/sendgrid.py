"""SendGrid Connector.

Provides production-quality integration with SendGrid for email sending,
email activity search, and template management.

Capabilities:
- Send transactional and marketing emails via the SendGrid v3 Mail Send API
- Search email activity (messages sent, delivered, opened, etc.)
- Search and retrieve dynamic transactional templates
- Health checks via the SendGrid Scopes API
- Rate limiting to stay within SendGrid plan limits
- Circuit breaker integration for failure protection
- Retry with exponential backoff on transient failures

Environment Variables:
- ``SENDGRID_API_KEY`` -- SendGrid API key with Mail Send + Email Activity read scopes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any
from urllib.parse import quote

import httpx

from aragora.connectors.base import (
    BaseConnector,
    ConnectorCapabilities,
    Evidence,
)
from aragora.connectors.exceptions import (
    ConnectorAPIError,
    ConnectorAuthError,
    ConnectorConfigError,
    ConnectorError,
    ConnectorValidationError,
)
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)

CONFIG_ENV_VARS = ("SENDGRID_API_KEY",)

_SAFE_QUERY_RE = re.compile(r"[^\w\s@.\-+]")
MAX_QUERY_LENGTH = 500

_SG_API_BASE = "https://api.sendgrid.com/v3"

# SendGrid free tier: 100 emails/day. Pro: 100k/month.
# API rate limit: ~600 requests/minute for most endpoints.
_DEFAULT_RATE_LIMIT_DELAY = 0.1  # 100ms between requests (safe for all tiers)
_MAX_SEND_BODY_BYTES = 30 * 1024 * 1024  # 30MB max per SendGrid docs


def _sanitize_query(query: str) -> str:
    """Sanitize query to prevent injection."""
    query = query[:MAX_QUERY_LENGTH]
    return _SAFE_QUERY_RE.sub("", query)


def _validate_email(email: str) -> bool:
    """Basic email format validation."""
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email))


class SendGridConnector(BaseConnector):
    """SendGrid connector for email sending, activity search, and template management.

    This is a production-quality connector that supports:
    - Sending transactional emails via the v3 Mail Send API
    - Searching email activity history
    - Searching and fetching dynamic transactional templates
    - Health checks, rate limiting, circuit breaker, and retry logic

    Example::

        connector = SendGridConnector()
        if connector.is_configured:
            # Send an email
            result = await connector.send({
                "to": "user@example.com",
                "from": "noreply@myapp.com",
                "subject": "Hello",
                "content": "Welcome to our platform!",
            })

            # Search email activity
            results = await connector.search("welcome email", limit=10)

            # Search templates
            templates = await connector.search("onboarding", search_type="templates")
    """

    def __init__(
        self,
        rate_limit_delay: float = _DEFAULT_RATE_LIMIT_DELAY,
        timeout: float = 30.0,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_retries=max_retries, **kwargs)
        self._configured = all(os.environ.get(v) for v in CONFIG_ENV_VARS)
        self._rate_limit_delay = rate_limit_delay
        self._timeout = timeout
        self._last_request_time: float = 0.0

    @property
    def name(self) -> str:
        return "sendgrid"

    @property
    def source_type(self) -> SourceType:
        return SourceType.EXTERNAL_API

    @property
    def is_configured(self) -> bool:
        return self._configured

    def capabilities(self) -> ConnectorCapabilities:
        """Report SendGrid connector capabilities."""
        return ConnectorCapabilities(
            can_send=True,
            can_receive=False,
            can_search=True,
            can_sync=False,
            can_stream=False,
            can_batch=True,
            is_stateful=False,
            requires_auth=True,
            supports_oauth=False,
            supports_webhooks=True,
            supports_files=True,  # Attachments
            supports_rich_text=True,  # HTML emails
            supports_reactions=False,
            supports_threads=False,
            supports_voice=False,
            supports_delivery_receipts=True,
            supports_retry=True,
            has_circuit_breaker=self._enable_circuit_breaker,
            max_requests_per_second=10.0,  # Conservative default
            max_message_size_bytes=_MAX_SEND_BODY_BYTES,
            platform_features=["templates", "email_activity", "suppressions"],
        )

    def _get_headers(self) -> dict[str, str]:
        api_key = os.environ.get("SENDGRID_API_KEY", "")
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._rate_limit_delay:
            await asyncio.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    async def _perform_health_check(self, timeout: float) -> bool:
        """Verify SendGrid connectivity by checking API key scopes."""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(
                    f"{_SG_API_BASE}/scopes",
                    headers=self._get_headers(),
                )
                if resp.status_code == 401:
                    logger.warning("SendGrid health check: invalid API key")
                    return False
                if resp.status_code == 403:
                    logger.warning("SendGrid health check: insufficient permissions")
                    return False
                resp.raise_for_status()
                return True
        except httpx.TimeoutException:
            logger.warning("SendGrid health check timed out")
            return False
        except httpx.HTTPError:
            logger.warning("SendGrid health check failed", exc_info=True)
            return False

    async def send(self, data: dict[str, Any]) -> dict[str, Any]:
        """Send an email via the SendGrid v3 Mail Send API.

        Args:
            data: Email payload with the following keys:
                - ``to`` (str or list[str]): Recipient email address(es). Required.
                - ``from`` or ``from_email`` (str): Sender email address. Required.
                - ``subject`` (str): Email subject line. Required.
                - ``content`` or ``text`` (str): Plain text body.
                - ``html`` (str): HTML body (optional, takes precedence over text).
                - ``template_id`` (str): SendGrid dynamic template ID (optional).
                - ``dynamic_template_data`` (dict): Template substitution data.
                - ``categories`` (list[str]): Tracking categories.
                - ``reply_to`` (str): Reply-to email address.
                - ``attachments`` (list[dict]): File attachments.

        Returns:
            Dict with ``status``, ``message_id``, and ``status_code`` keys.

        Raises:
            ConnectorConfigError: If connector is not configured.
            ConnectorValidationError: If required fields are missing.
            ConnectorAuthError: If API key is invalid.
            ConnectorRateLimitError: If rate limited by SendGrid.
            ConnectorAPIError: On other API failures.
        """
        if not self._configured:
            raise ConnectorConfigError(
                "SendGrid connector not configured (missing SENDGRID_API_KEY)",
                connector_name=self.name,
                config_key="SENDGRID_API_KEY",
            )

        # Validate required fields
        to_email = data.get("to")
        from_email = data.get("from") or data.get("from_email")
        subject = data.get("subject")

        if not to_email:
            raise ConnectorValidationError(
                "Recipient email ('to') is required",
                connector_name=self.name,
                field="to",
            )
        if not from_email:
            raise ConnectorValidationError(
                "Sender email ('from' or 'from_email') is required",
                connector_name=self.name,
                field="from",
            )

        # Normalize to_email to a list
        if isinstance(to_email, str):
            to_list = [to_email]
        else:
            to_list = list(to_email)

        # Validate email addresses
        if not _validate_email(from_email):
            raise ConnectorValidationError(
                f"Invalid sender email format: {from_email}",
                connector_name=self.name,
                field="from",
            )
        for addr in to_list:
            if not _validate_email(addr):
                raise ConnectorValidationError(
                    f"Invalid recipient email format: {addr}",
                    connector_name=self.name,
                    field="to",
                )

        # Build the Mail Send v3 payload
        personalizations = [{"to": [{"email": addr} for addr in to_list]}]
        dynamic_data = data.get("dynamic_template_data")
        if dynamic_data:
            personalizations[0]["dynamic_template_data"] = dynamic_data

        payload: dict[str, Any] = {
            "personalizations": personalizations,
            "from": {"email": from_email},
        }

        # Template-based sending vs content-based sending
        template_id = data.get("template_id")
        if template_id:
            payload["template_id"] = template_id
        else:
            if not subject:
                raise ConnectorValidationError(
                    "Subject is required when not using a template",
                    connector_name=self.name,
                    field="subject",
                )
            payload["subject"] = subject

            content_list = []
            text_body = data.get("content") or data.get("text")
            html_body = data.get("html")
            if text_body:
                content_list.append({"type": "text/plain", "value": text_body})
            if html_body:
                content_list.append({"type": "text/html", "value": html_body})
            if not content_list:
                raise ConnectorValidationError(
                    "Email body ('content', 'text', or 'html') is required",
                    connector_name=self.name,
                    field="content",
                )
            payload["content"] = content_list

        # Optional fields
        reply_to = data.get("reply_to")
        if reply_to:
            payload["reply_to"] = {"email": reply_to}

        categories = data.get("categories")
        if categories:
            payload["categories"] = categories[:10]  # SendGrid max 10

        attachments = data.get("attachments")
        if attachments:
            payload["attachments"] = attachments

        from_name = data.get("from_name")
        if from_name:
            payload["from"]["name"] = from_name

        await self._rate_limit()

        async def _do_send() -> httpx.Response:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{_SG_API_BASE}/mail/send",
                    headers=self._get_headers(),
                    json=payload,
                )
                # 401/403 should not be retried
                if resp.status_code == 401:
                    raise ConnectorAuthError(
                        "Invalid SendGrid API key",
                        connector_name=self.name,
                    )
                if resp.status_code == 403:
                    raise ConnectorAuthError(
                        "SendGrid API key lacks mail.send permission",
                        connector_name=self.name,
                    )
                resp.raise_for_status()
                return resp

        try:
            resp = await self._request_with_retry(_do_send, "send_email")
        except ConnectorAuthError:
            raise
        except ConnectorError:
            raise
        except (httpx.HTTPError, OSError) as exc:
            raise ConnectorAPIError(
                "Failed to send email via SendGrid",
                connector_name=self.name,
            ) from exc

        # SendGrid returns 202 Accepted on success with X-Message-Id header
        message_id = resp.headers.get("X-Message-Id", "")
        return {
            "status": "accepted",
            "message_id": message_id,
            "status_code": resp.status_code,
            "to": to_list,
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

        await self._rate_limit()

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{_SG_API_BASE}/messages",
                    headers=self._get_headers(),
                    params={"query": sg_query, "limit": limit},
                )
                if resp.status_code == 401:
                    raise ConnectorAuthError(
                        "Invalid SendGrid API key",
                        connector_name=self.name,
                    )
                if resp.status_code == 403:
                    raise ConnectorAuthError(
                        "SendGrid API key lacks email activity read permission",
                        connector_name=self.name,
                    )
                resp.raise_for_status()
                return resp.json()

        try:
            data = await self._request_with_retry(_do_request, "search_activity")
        except ConnectorAuthError:
            raise
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

        await self._rate_limit()

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{_SG_API_BASE}/templates",
                    headers=self._get_headers(),
                    params={"generations": "dynamic", "page_size": limit},
                )
                if resp.status_code == 401:
                    raise ConnectorAuthError(
                        "Invalid SendGrid API key",
                        connector_name=self.name,
                    )
                resp.raise_for_status()
                return resp.json()

        try:
            data = await self._request_with_retry(_do_request, "search_templates")
        except ConnectorAuthError:
            raise
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
            return await self._fetch_template(evidence_id[len("sg_tpl_"):], evidence_id)
        elif evidence_id.startswith("sg_msg_"):
            return await self._fetch_message(evidence_id[len("sg_msg_"):], evidence_id)

        return None

    async def _fetch_message(self, msg_id: str, evidence_id: str) -> Evidence | None:
        """Fetch a single email message from the Activity API."""

        await self._rate_limit()

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{_SG_API_BASE}/messages/{quote(msg_id, safe='')}",
                    headers=self._get_headers(),
                )
                if resp.status_code == 401:
                    raise ConnectorAuthError(
                        "Invalid SendGrid API key",
                        connector_name=self.name,
                    )
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                return resp.json()

        try:
            msg = await self._request_with_retry(_do_request, "fetch_message")
        except ConnectorAuthError:
            raise
        except (ConnectorError, httpx.HTTPError, OSError, ValueError):
            logger.warning("SendGrid message fetch failed for %s", evidence_id, exc_info=True)
            return None

        if msg is None:
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

        await self._rate_limit()

        async def _do_request() -> Any:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{_SG_API_BASE}/templates/{quote(template_id, safe='')}",
                    headers=self._get_headers(),
                )
                if resp.status_code == 401:
                    raise ConnectorAuthError(
                        "Invalid SendGrid API key",
                        connector_name=self.name,
                    )
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                return resp.json()

        try:
            tpl = await self._request_with_retry(_do_request, "fetch_template")
        except ConnectorAuthError:
            raise
        except (ConnectorError, httpx.HTTPError, OSError, ValueError):
            logger.warning("SendGrid template fetch failed for %s", evidence_id, exc_info=True)
            return None

        if tpl is None:
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
