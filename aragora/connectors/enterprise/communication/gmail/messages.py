"""
Gmail message operations.

Provides message listing, retrieval, parsing, sending, replying,
searching, syncing, and prioritization integration.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from email.utils import parseaddr
from typing import Any, Protocol, TYPE_CHECKING
from collections.abc import AsyncIterator

from aragora.connectors.enterprise.base import SyncItem, SyncState
from aragora.reasoning.provenance import SourceType

from ..models import (
    BatchFetchResult,
    EmailAttachment,
    EmailMessage,
    EmailThread,
    MessageFetchFailure,
)

if TYPE_CHECKING:
    import httpx

    from . import GmailConnector

logger = logging.getLogger(__name__)

_MAX_PAGES = 1000  # Safety cap for pagination loops


class GmailBaseMethods(Protocol):
    """Protocol defining expected methods from base classes for type checking."""

    user_id: str
    include_spam_trash: bool
    exclude_labels: set[str]
    labels: list[str] | None
    max_results: int

    @property
    def source_type(self) -> SourceType: ...
    async def _get_access_token(self) -> str: ...
    async def _api_request(
        self, endpoint: str, method: str = "GET", **kwargs: Any
    ) -> dict[str, Any]: ...
    @asynccontextmanager
    def _get_client(self) -> AsyncIterator[httpx.AsyncClient]: ...
    def check_circuit_breaker(self) -> bool: ...
    def get_circuit_breaker_status(self) -> dict[str, Any]: ...
    def record_success(self) -> None: ...
    def record_failure(self) -> None: ...
    async def get_user_info(self) -> dict[str, Any]: ...


class GmailMessagesMixin(GmailBaseMethods):
    """Mixin providing message operations for the Gmail connector."""

    async def list_messages(
        self,
        query: str = "",
        label_ids: list[str] | None = None,
        page_token: str | None = None,
        max_results: int = 100,
    ) -> tuple[list[str], str | None]:
        """
        List message IDs matching criteria.

        Args:
            query: Gmail search query
            label_ids: Filter by label IDs
            page_token: Pagination token
            max_results: Max messages to return

        Returns:
            Tuple of (message_ids, next_page_token)
        """
        params: dict[str, Any] = {
            "maxResults": min(max_results, 500),
            "includeSpamTrash": self.include_spam_trash,
        }

        if query:
            params["q"] = query
        if label_ids:
            params["labelIds"] = label_ids
        if page_token:
            params["pageToken"] = page_token

        data = await self._api_request("/messages", params=params)

        message_ids = [m["id"] for m in data.get("messages", [])]
        next_token = data.get("nextPageToken")

        return message_ids, next_token

    async def get_message(
        self,
        message_id: str,
        format: str = "full",
    ) -> EmailMessage:
        """
        Get a single message by ID.

        Args:
            message_id: Message ID
            format: "full", "metadata", or "minimal"

        Returns:
            EmailMessage object
        """
        params = {"format": format}
        data = await self._api_request(f"/messages/{message_id}", params=params)

        return self._parse_message(data)

    async def get_messages(
        self,
        message_ids: list[str],
        format: str = "full",
        max_concurrent: int = 10,
        strict: bool = False,
    ) -> list[EmailMessage]:
        """
        Get multiple messages by IDs in parallel.

        Fetches messages concurrently to avoid N+1 query patterns.
        For detailed failure information, use get_messages_batch() instead.

        Args:
            message_ids: List of message IDs to fetch
            format: "full", "metadata", or "minimal"
            max_concurrent: Maximum concurrent requests (default 10)
            strict: If True, raises an exception when any message fails to fetch.
                   If False (default), returns partial results and logs failures.

        Returns:
            List of EmailMessage objects (may be shorter than input if some fail
            and strict=False)

        Raises:
            RuntimeError: If strict=True and any message fails to fetch.
                         The exception message includes details about all failures.
        """
        result = await self.get_messages_batch(
            message_ids=message_ids,
            format=format,
            max_concurrent=max_concurrent,
        )

        if strict and result.failures:
            failed_summary = ", ".join(
                f"{f.message_id} ({f.error_type})" for f in result.failures[:5]
            )
            if len(result.failures) > 5:
                failed_summary += f" and {len(result.failures) - 5} more"
            raise RuntimeError(
                f"Failed to fetch {result.failure_count} of {result.total_requested} "
                f"messages: {failed_summary}"
            )

        return result.messages

    async def get_messages_batch(
        self,
        message_ids: list[str],
        format: str = "full",
        max_concurrent: int = 10,
    ) -> BatchFetchResult:
        """
        Get multiple messages by IDs in parallel with detailed failure tracking.

        Fetches messages concurrently to avoid N+1 query patterns.
        Returns a BatchFetchResult containing both successful messages and
        detailed information about any failures.

        Args:
            message_ids: List of message IDs to fetch
            format: "full", "metadata", or "minimal"
            max_concurrent: Maximum concurrent requests (default 10)

        Returns:
            BatchFetchResult containing:
            - messages: List of successfully fetched EmailMessage objects
            - failures: List of MessageFetchFailure with error details
            - Computed properties: is_complete, is_partial, is_total_failure,
              failed_ids, retryable_ids, success_count, failure_count
        """
        if not message_ids:
            return BatchFetchResult(
                messages=[],
                failures=[],
                total_requested=0,
            )

        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        messages: list[EmailMessage] = []
        failures: list[MessageFetchFailure] = []

        async def fetch_one(msg_id: str) -> tuple[EmailMessage | None, MessageFetchFailure | None]:
            async with semaphore:
                try:
                    msg = await self.get_message(msg_id, format=format)
                    return msg, None
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    TypeError,
                    RuntimeError,
                    ConnectionError,
                    TimeoutError,
                ) as e:
                    # Log with appropriate level based on error type
                    error_type = type(e).__name__
                    error_msg = str(e)

                    # Determine if this is a retryable error
                    is_retryable = any(
                        x in error_msg.lower()
                        for x in ["timeout", "connection", "429", "503", "502", "rate"]
                    )

                    if is_retryable:
                        logger.warning(
                            f"[Gmail] Retryable error fetching message {msg_id}: "
                            f"{error_type}: {error_msg}"
                        )
                    else:
                        logger.error(
                            f"[Gmail] Failed to fetch message {msg_id}: {error_type}: {error_msg}"
                        )

                    failure = MessageFetchFailure(
                        message_id=msg_id,
                        error=e,
                        error_type=error_type,
                        is_retryable=is_retryable,
                    )
                    return None, failure

        # Fetch all messages in parallel
        results = await asyncio.gather(
            *[fetch_one(msg_id) for msg_id in message_ids],
            return_exceptions=False,
        )

        # Separate successes and failures
        for msg, failure in results:
            if msg is not None:
                messages.append(msg)
            elif failure is not None:
                failures.append(failure)

        # Log summary if there were failures
        if failures:
            retryable_count = sum(1 for f in failures if f.is_retryable)
            logger.warning(
                f"[Gmail] Batch fetch completed with {len(failures)} failures "
                f"out of {len(message_ids)} requests "
                f"({retryable_count} retryable, {len(failures) - retryable_count} permanent)"
            )

        return BatchFetchResult(
            messages=messages,
            failures=failures,
            total_requested=len(message_ids),
        )

    def _parse_message(self, data: dict[str, Any]) -> EmailMessage:
        """Parse Gmail API message response into EmailMessage."""
        headers = {}
        payload = data.get("payload", {})

        # Extract headers
        for header in payload.get("headers", []):
            name = header.get("name", "").lower()
            value = header.get("value", "")
            headers[name] = value

        # Parse addresses
        from_addr = parseaddr(headers.get("from", ""))[1]
        to_addrs = [parseaddr(a)[1] for a in headers.get("to", "").split(",") if a.strip()]
        cc_addrs = [parseaddr(a)[1] for a in headers.get("cc", "").split(",") if a.strip()]
        bcc_addrs = [parseaddr(a)[1] for a in headers.get("bcc", "").split(",") if a.strip()]

        # Parse date
        date_str = headers.get("date", "")
        try:
            from email.utils import parsedate_to_datetime

            date = parsedate_to_datetime(date_str)
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to parse date string '{date_str}': {e}")
            date = datetime.now(timezone.utc)

        # Extract body
        body_text, body_html, attachments = self._extract_body_and_attachments(payload)

        # Labels and flags
        labels = data.get("labelIds", [])
        is_read = "UNREAD" not in labels
        is_starred = "STARRED" in labels
        is_important = "IMPORTANT" in labels

        return EmailMessage(
            id=data["id"],
            thread_id=data.get("threadId", data["id"]),
            subject=headers.get("subject", "(No Subject)"),
            from_address=from_addr,
            to_addresses=to_addrs,
            cc_addresses=cc_addrs,
            bcc_addresses=bcc_addrs,
            date=date,
            body_text=body_text,
            body_html=body_html,
            snippet=data.get("snippet", ""),
            labels=labels,
            headers=headers,
            attachments=attachments,
            is_read=is_read,
            is_starred=is_starred,
            is_important=is_important,
        )

    def _extract_body_and_attachments(
        self,
        payload: dict[str, Any],
    ) -> tuple[str, str, list[EmailAttachment]]:
        """Extract body text, HTML, and attachments from message payload."""
        body_text = ""
        body_html = ""
        attachments = []

        mime_type = payload.get("mimeType", "")
        body = payload.get("body", {})
        parts = payload.get("parts", [])

        # Single-part message
        if not parts and body.get("data"):
            content = base64.urlsafe_b64decode(body["data"]).decode("utf-8", errors="replace")
            if "text/plain" in mime_type:
                body_text = content
            elif "text/html" in mime_type:
                body_html = content

        # Multi-part message
        for part in parts:
            part_mime = part.get("mimeType", "")
            part_body = part.get("body", {})
            part_filename = part.get("filename", "")

            # Attachment
            if part_filename and part_body.get("attachmentId"):
                attachments.append(
                    EmailAttachment(
                        id=part_body["attachmentId"],
                        filename=part_filename,
                        mime_type=part_mime,
                        size=part_body.get("size", 0),
                    )
                )
                continue

            # Nested multipart
            if part.get("parts"):
                nested_text, nested_html, nested_attachments = self._extract_body_and_attachments(
                    part
                )
                if not body_text:
                    body_text = nested_text
                if not body_html:
                    body_html = nested_html
                attachments.extend(nested_attachments)
                continue

            # Body content
            if part_body.get("data"):
                content = base64.urlsafe_b64decode(part_body["data"]).decode(
                    "utf-8", errors="replace"
                )
                if "text/plain" in part_mime and not body_text:
                    body_text = content
                elif "text/html" in part_mime and not body_html:
                    body_html = content

        return body_text, body_html, attachments

    async def get_thread(self, thread_id: str) -> EmailThread:
        """Get a conversation thread with all messages."""
        data = await self._api_request(f"/threads/{thread_id}", params={"format": "full"})

        messages = [self._parse_message(m) for m in data.get("messages", [])]

        # Collect participants
        participants = set()
        for msg in messages:
            participants.add(msg.from_address)
            participants.update(msg.to_addresses)
            participants.update(msg.cc_addresses)

        # Get thread subject from first message
        subject = messages[0].subject if messages else ""

        return EmailThread(
            id=thread_id,
            subject=subject,
            messages=messages,
            participants=list(participants),
            labels=list(set(label for msg in messages for label in msg.labels)),
            last_message_date=messages[-1].date if messages else None,
            snippet=data.get("snippet", ""),
            message_count=len(messages),
        )

    async def get_history(
        self,
        start_history_id: str,
        label_id: str | None = None,
        page_token: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None, str]:
        """
        Get message history changes since a history ID.

        Returns:
            Tuple of (history_records, next_page_token, new_history_id)
        """
        params: dict[str, Any] = {
            "startHistoryId": start_history_id,
            "historyTypes": ["messageAdded", "labelAdded", "labelRemoved"],
        }

        if label_id:
            params["labelId"] = label_id
        if page_token:
            params["pageToken"] = page_token

        try:
            data = await self._api_request("/history", params=params)
        except (OSError, ValueError, KeyError, RuntimeError, ConnectionError, TimeoutError) as e:
            # History ID may be expired - need full sync
            if "404" in str(e) or "historyId" in str(e).lower():
                logger.warning("[Gmail] History ID expired, need full sync")
                return [], None, ""
            raise

        history = data.get("history", [])
        next_token = data.get("nextPageToken")
        new_history_id = data.get("historyId", start_history_id)

        return history, next_token, new_history_id

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[Any]:
        """Search Gmail messages."""
        from aragora.connectors.base import Evidence

        message_ids, _ = await self.list_messages(query=query, max_results=limit)

        # Batch fetch messages to avoid N+1 queries
        messages = await self.get_messages(message_ids[:limit], format="metadata")

        results = []
        for msg in messages:
            results.append(
                Evidence(
                    id=f"gmail-{msg.id}",
                    source_type=self.source_type,
                    source_id=msg.id,
                    content=msg.snippet,
                    title=msg.subject,
                    url=f"https://mail.google.com/mail/u/0/#inbox/{msg.id}",
                    author=msg.from_address,
                    confidence=0.8,
                    metadata={
                        "thread_id": msg.thread_id,
                        "date": msg.date.isoformat() if msg.date else None,
                        "labels": msg.labels,
                    },
                )
            )

        return results

    async def fetch(self, evidence_id: str) -> Any | None:
        """Fetch a specific email by ID."""
        from aragora.connectors.base import Evidence

        # Extract message ID
        if evidence_id.startswith("gmail-"):
            message_id = evidence_id[6:]
        else:
            message_id = evidence_id

        try:
            msg = await self.get_message(message_id)

            return Evidence(
                id=f"gmail-{msg.id}",
                source_type=self.source_type,
                source_id=msg.id,
                content=msg.body_text or msg.snippet,
                title=msg.subject,
                url=f"https://mail.google.com/mail/u/0/#inbox/{msg.id}",
                author=msg.from_address,
                confidence=0.85,
                metadata={
                    "thread_id": msg.thread_id,
                    "date": msg.date.isoformat() if msg.date else None,
                    "labels": msg.labels,
                    "to": msg.to_addresses,
                    "cc": msg.cc_addresses,
                    "is_read": msg.is_read,
                    "is_starred": msg.is_starred,
                },
            )

        except (OSError, ValueError, KeyError, RuntimeError) as e:
            logger.error(f"[Gmail] Fetch failed: {e}")
            return None

    async def sync_items(
        self,
        state: SyncState,
        batch_size: int = 100,
    ) -> AsyncIterator[SyncItem]:
        """
        Yield Gmail messages for syncing.

        Uses History API for incremental sync when cursor is available.
        """
        items_yielded = 0

        # Check for incremental sync
        if state.cursor:
            logger.info(f"[Gmail] Starting incremental sync from history {state.cursor[:20]}...")

            # Get changes since last sync
            history_id = state.cursor
            page_token = None
            new_message_ids = set()

            for _page in range(_MAX_PAGES):
                history, page_token, new_history_id = await self.get_history(
                    history_id,
                    page_token=page_token,
                )

                if not history and not page_token:
                    # History ID expired or no changes
                    if not new_history_id:
                        logger.info("[Gmail] History ID expired, falling back to full sync")
                        state.cursor = None
                        break
                    state.cursor = new_history_id
                    return

                # Extract new/changed message IDs
                for record in history:
                    for msg_added in record.get("messagesAdded", []):
                        new_message_ids.add(msg_added["message"]["id"])

                if not page_token:
                    state.cursor = new_history_id
                    break

            # Fetch and yield new messages
            for msg_id in new_message_ids:
                try:
                    msg = await self.get_message(msg_id)

                    # Skip excluded labels
                    if self.exclude_labels and any(
                        lbl in self.exclude_labels for lbl in msg.labels
                    ):
                        continue

                    yield self._message_to_sync_item(msg)
                    items_yielded += 1

                    if items_yielded >= batch_size:
                        await asyncio.sleep(0)

                except (OSError, ValueError, KeyError, RuntimeError) as e:
                    logger.warning(f"[Gmail] Failed to fetch message {msg_id}: {e}")

            return

        # Full sync
        logger.info("[Gmail] Starting full sync...")

        # Get current profile for history ID
        profile = await self.get_user_info()
        state.cursor = str(profile.get("historyId", ""))

        # Build query
        query_parts = []
        if self.labels:
            for label in self.labels:
                query_parts.append(f"label:{label}")

        query = " OR ".join(query_parts) if query_parts else ""

        # Iterate through messages
        page_token = None
        for _page in range(_MAX_PAGES):
            message_ids, page_token = await self.list_messages(
                query=query,
                max_results=min(self.max_results, batch_size),
                page_token=page_token,
            )

            for msg_id in message_ids:
                try:
                    msg = await self.get_message(msg_id)

                    # Skip excluded labels
                    if self.exclude_labels and any(
                        lbl in self.exclude_labels for lbl in msg.labels
                    ):
                        continue

                    yield self._message_to_sync_item(msg)
                    items_yielded += 1

                    if items_yielded >= batch_size:
                        await asyncio.sleep(0)

                    if items_yielded >= self.max_results:
                        return

                except (OSError, ValueError, KeyError, RuntimeError) as e:
                    logger.warning(f"[Gmail] Failed to fetch message {msg_id}: {e}")

            if not page_token:
                break

    async def send_message(
        self,
        to: list[str],
        subject: str,
        body: str,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
        reply_to: str | None = None,
        html_body: str | None = None,
    ) -> dict[str, Any]:
        """
        Send an email message.

        Requires gmail.send scope to be authorized.

        Args:
            to: List of recipient email addresses
            subject: Email subject line
            body: Plain text body
            cc: Optional CC recipients
            bcc: Optional BCC recipients
            reply_to: Optional reply-to address
            html_body: Optional HTML body (sent as alternative)

        Returns:
            Dict with message_id and thread_id of sent message
        """
        # Get fresh token (will refresh if expired)
        access_token = await self._get_access_token()

        from email.mime.base import MIMEBase
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        # Build MIME message
        message: MIMEBase
        if html_body:
            message = MIMEMultipart("alternative")
            message.attach(MIMEText(body, "plain"))
            message.attach(MIMEText(html_body, "html"))
        else:
            message = MIMEText(body, "plain")

        message["To"] = ", ".join(to)
        message["Subject"] = subject

        if cc:
            message["Cc"] = ", ".join(cc)
        if bcc:
            message["Bcc"] = ", ".join(bcc)
        if reply_to:
            message["Reply-To"] = reply_to

        # Encode message
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

        # Check circuit breaker first
        if not self.check_circuit_breaker():
            cb_status = self.get_circuit_breaker_status()
            raise ConnectionError(
                f"Circuit breaker open for Gmail. Cooldown: {cb_status.get('cooldown_seconds', 60)}s"
            )

        # Send via Gmail API
        try:
            async with self._get_client() as client:
                response = await client.post(
                    f"https://gmail.googleapis.com/gmail/v1/users/{self.user_id}/messages/send",
                    headers={"Authorization": f"Bearer {access_token}"},
                    json={"raw": raw_message},
                )

                if response.status_code != 200:
                    error = response.json().get("error", {})
                    if response.status_code >= 500 or response.status_code == 429:
                        self.record_failure()
                    raise RuntimeError(
                        f"Failed to send email: {error.get('message', response.text)}"
                    )

                self.record_success()
                result = response.json()
                logger.info(f"[Gmail] Sent message: {result.get('id')}")

                return {
                    "message_id": result.get("id"),
                    "thread_id": result.get("threadId"),
                    "success": True,
                }
        except (OSError, ConnectionError):
            self.record_failure()
            raise

    async def reply_to_message(
        self,
        original_message_id: str,
        body: str,
        cc: list[str] | None = None,
        html_body: str | None = None,
    ) -> dict[str, Any]:
        """
        Reply to an existing email message.

        Maintains thread context and proper In-Reply-To headers.

        Args:
            original_message_id: Gmail message ID to reply to
            body: Reply body text
            cc: Optional additional CC recipients
            html_body: Optional HTML body

        Returns:
            Dict with message_id and thread_id of sent reply
        """
        # Get fresh token (will refresh if expired)
        access_token = await self._get_access_token()

        # Fetch original message for context
        original = await self.get_message(original_message_id)
        if not original:
            raise ValueError(f"Original message not found: {original_message_id}")

        from email.mime.base import MIMEBase
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        # Build reply subject
        subject = original.subject
        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"

        # Determine reply recipients (reply to sender, CC original recipients)
        to = [original.from_address]
        reply_cc = list(
            set(original.to_addresses + original.cc_addresses) - {original.from_address}
        )
        if cc:
            reply_cc.extend(cc)

        # Build MIME message
        message: MIMEBase
        if html_body:
            message = MIMEMultipart("alternative")
            message.attach(MIMEText(body, "plain"))
            message.attach(MIMEText(html_body, "html"))
        else:
            message = MIMEText(body, "plain")

        message["To"] = ", ".join(to)
        message["Subject"] = subject

        if reply_cc:
            message["Cc"] = ", ".join(reply_cc)

        # Thread headers
        if original.message_id_header:
            message["In-Reply-To"] = original.message_id_header
            message["References"] = original.message_id_header

        # Encode message
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

        # Check circuit breaker first
        if not self.check_circuit_breaker():
            cb_status = self.get_circuit_breaker_status()
            raise ConnectionError(
                f"Circuit breaker open for Gmail. Cooldown: {cb_status.get('cooldown_seconds', 60)}s"
            )

        # Send via Gmail API (include threadId to maintain thread)
        try:
            async with self._get_client() as client:
                response = await client.post(
                    f"https://gmail.googleapis.com/gmail/v1/users/{self.user_id}/messages/send",
                    headers={"Authorization": f"Bearer {access_token}"},
                    json={
                        "raw": raw_message,
                        "threadId": original.thread_id,
                    },
                )

                if response.status_code != 200:
                    error = response.json().get("error", {})
                    if response.status_code >= 500 or response.status_code == 429:
                        self.record_failure()
                    raise RuntimeError(
                        f"Failed to send reply: {error.get('message', response.text)}"
                    )

                self.record_success()
                result = response.json()
                logger.info(
                    f"[Gmail] Sent reply: {result.get('id')} in thread {result.get('threadId')}"
                )

                return {
                    "message_id": result.get("id"),
                    "thread_id": result.get("threadId"),
                    "in_reply_to": original_message_id,
                    "success": True,
                }
        except (OSError, ConnectionError):
            self.record_failure()
            raise

    async def sync_with_prioritization(
        self,
        messages: list[EmailMessage],
        prioritizer: Any | None = None,
        timeout_seconds: float = 30.0,
    ) -> list[dict[str, Any]]:
        """
        Sync messages with email prioritization scoring.

        Combines Gmail sync with intelligent prioritization to rank
        emails by importance, urgency, and context relevance.

        Args:
            messages: List of EmailMessage objects to prioritize
            prioritizer: EmailPrioritizer instance (creates one if not provided)
            timeout_seconds: Timeout for each prioritization call

        Returns:
            List of dicts with message and priority result:
            [
                {
                    "message": EmailMessage,
                    "priority_result": EmailPriorityResult,
                    "priority": "HIGH",  # String for easy filtering
                    "confidence": 0.85,
                    "rationale": "VIP sender + urgent keywords",
                }
            ]
        """
        if not messages:
            return []

        # Create prioritizer if not provided
        if prioritizer is None:
            try:
                from aragora.services.email_prioritization import EmailPrioritizer

                # Cast self to GmailConnector for type safety
                # (self is a mixin that's part of the full connector)
                from typing import cast

                gmail_connector = cast("GmailConnector", self)
                prioritizer = EmailPrioritizer(gmail_connector=gmail_connector)
            except ImportError:
                logger.warning("[Gmail] EmailPrioritizer not available, skipping prioritization")
                # Return messages without prioritization
                return [
                    {
                        "message": msg,
                        "priority_result": None,
                        "priority": "MEDIUM",
                        "confidence": 0.0,
                        "rationale": "Prioritization not available",
                    }
                    for msg in messages
                ]

        results: list[dict[str, Any]] = []

        for msg in messages:
            try:
                priority_result = await asyncio.wait_for(
                    prioritizer.score_email(msg),
                    timeout=timeout_seconds,
                )

                results.append(
                    {
                        "message": msg,
                        "priority_result": priority_result,
                        "priority": priority_result.priority.name,
                        "confidence": priority_result.confidence,
                        "rationale": priority_result.rationale,
                        "suggested_labels": priority_result.suggested_labels,
                        "auto_archive": priority_result.auto_archive,
                    }
                )

                # Update message importance fields
                msg.importance_score = 1.0 - (priority_result.priority.value - 1) / 5.0
                msg.importance_reason = priority_result.rationale

            except asyncio.TimeoutError:
                logger.warning(f"[Gmail] Prioritization timeout for message {msg.id}")
                results.append(
                    {
                        "message": msg,
                        "priority_result": None,
                        "priority": "MEDIUM",
                        "confidence": 0.0,
                        "rationale": "Prioritization timed out",
                    }
                )

            except (OSError, ValueError, KeyError, RuntimeError) as e:
                logger.warning(f"[Gmail] Prioritization failed for message {msg.id}: {e}")
                results.append(
                    {
                        "message": msg,
                        "priority_result": None,
                        "priority": "MEDIUM",
                        "confidence": 0.0,
                        "rationale": "Prioritization failed",
                    }
                )

        # Sort by priority (lower value = higher priority)
        results.sort(
            key=lambda r: (
                r["priority_result"].priority.value if r["priority_result"] else 3,
                -r["confidence"],
            )
        )

        logger.info(f"[Gmail] Prioritized {len(results)} messages")
        return results

    async def rank_inbox(
        self,
        max_messages: int = 50,
        labels: list[str] | None = None,
        query: str = "",
    ) -> list[dict[str, Any]]:
        """
        Fetch and rank inbox messages by priority.

        Convenience method that combines fetching and prioritization.

        Args:
            max_messages: Maximum messages to fetch and rank
            labels: Label filters (default: connector's configured labels)
            query: Additional Gmail search query

        Returns:
            Prioritized list of messages with scores
        """
        # Use connector's configured labels if not specified
        filter_labels = labels or self.labels

        # Build query
        search_query = query
        if filter_labels:
            label_filter = " OR ".join(f"label:{lbl}" for lbl in filter_labels)
            search_query = f"({label_filter}) {query}".strip()

        # Fetch messages
        message_ids, _ = await self.list_messages(
            query=search_query,
            max_results=max_messages,
        )

        # Batch fetch messages to avoid N+1 queries
        messages = await self.get_messages(message_ids[:max_messages])

        # Prioritize and return
        return await self.sync_with_prioritization(messages)

    def _message_to_sync_item(self, msg: EmailMessage) -> SyncItem:
        """Convert EmailMessage to SyncItem for Knowledge Mound ingestion."""
        # Build content with context
        content_parts = [
            f"Subject: {msg.subject}",
            f"From: {msg.from_address}",
            f"To: {', '.join(msg.to_addresses)}",
            f"Date: {msg.date.isoformat() if msg.date else 'Unknown'}",
            "",
            msg.body_text or msg.snippet,
        ]

        if msg.cc_addresses:
            content_parts.insert(3, f"CC: {', '.join(msg.cc_addresses)}")

        return SyncItem(
            id=f"gmail-{msg.id}",
            content="\n".join(content_parts)[:50000],
            source_type="email",
            source_id=f"gmail/{msg.id}",
            title=msg.subject,
            url=f"https://mail.google.com/mail/u/0/#inbox/{msg.id}",
            author=msg.from_address,
            created_at=msg.date,
            updated_at=msg.date,
            domain="enterprise/gmail",
            confidence=0.85,
            metadata={
                "message_id": msg.id,
                "thread_id": msg.thread_id,
                "labels": msg.labels,
                "to_addresses": msg.to_addresses,
                "cc_addresses": msg.cc_addresses,
                "is_read": msg.is_read,
                "is_starred": msg.is_starred,
                "is_important": msg.is_important,
                "has_attachments": len(msg.attachments) > 0,
                "attachment_count": len(msg.attachments),
            },
        )
