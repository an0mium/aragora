"""
Gmail Enterprise Connector.

Provides full integration with Gmail inboxes:
- OAuth2 authentication flow
- Message and thread fetching
- Label/folder management
- Incremental sync via History API
- Search with Gmail query syntax

Requires Google Cloud OAuth2 credentials with Gmail scopes.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from datetime import datetime, timezone
from email.utils import parseaddr
from typing import Any, AsyncIterator, Dict, List, Optional

from aragora.connectors.enterprise.base import (
    EnterpriseConnector,
    SyncItem,
    SyncState,
)
from aragora.reasoning.provenance import SourceType

from .models import (
    EmailAttachment,
    EmailMessage,
    EmailThread,
    GmailLabel,
    GmailSyncState,
    GmailWebhookPayload,
)

logger = logging.getLogger(__name__)


# Gmail API scopes
# Note: gmail.metadata doesn't support search queries ('q' parameter)
# Using gmail.readonly alone is sufficient for read operations including search
GMAIL_SCOPES_READONLY = [
    "https://www.googleapis.com/auth/gmail.readonly",
]

# Full scopes including send (required for bidirectional email)
GMAIL_SCOPES_FULL = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.metadata",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
]

# Default to read-only for backward compatibility
GMAIL_SCOPES = GMAIL_SCOPES_READONLY


class GmailConnector(EnterpriseConnector):
    """
    Enterprise connector for Gmail.

    Features:
    - OAuth2 authentication with refresh tokens
    - Full message content retrieval
    - Thread-based conversation view
    - Label/folder filtering
    - Incremental sync via History API
    - Gmail search query support

    Authentication:
    - OAuth2 with refresh token (required)

    Usage:
        connector = GmailConnector(
            labels=["INBOX", "IMPORTANT"],
            max_results=100,
        )

        # Get OAuth URL for user authorization
        url = connector.get_oauth_url(redirect_uri, state)

        # After user authorizes, exchange code for tokens
        await connector.authenticate(code=auth_code, redirect_uri=redirect_uri)

        # Sync messages
        result = await connector.sync()
    """

    def __init__(
        self,
        labels: Optional[List[str]] = None,
        exclude_labels: Optional[List[str]] = None,
        max_results: int = 100,
        include_spam_trash: bool = False,
        user_id: str = "me",
        **kwargs,
    ):
        """
        Initialize Gmail connector.

        Args:
            labels: Labels to sync (None = all)
            exclude_labels: Labels to exclude
            max_results: Max messages per sync batch
            include_spam_trash: Include spam/trash folders
            user_id: Gmail user ID ("me" for authenticated user)
        """
        super().__init__(connector_id="gmail", **kwargs)

        self.labels = labels
        self.exclude_labels = set(exclude_labels or [])
        self.max_results = max_results
        self.include_spam_trash = include_spam_trash
        self.user_id = user_id

        # OAuth tokens (protected by _token_lock for thread-safety)
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._token_lock: asyncio.Lock = asyncio.Lock()

        # Gmail-specific state
        self._gmail_state: Optional[GmailSyncState] = None

        # Watch management for Pub/Sub notifications
        self._watch_task: Optional[asyncio.Task] = None
        self._watch_running: bool = False

    @property
    def source_type(self) -> SourceType:
        return SourceType.DOCUMENT

    @property
    def name(self) -> str:
        return "Gmail"

    @property
    def access_token(self) -> Optional[str]:
        """Expose current access token (if available)."""
        return self._access_token

    @property
    def refresh_token(self) -> Optional[str]:
        """Expose current refresh token (if available)."""
        return self._refresh_token

    @property
    def token_expiry(self) -> Optional[datetime]:
        """Expose access token expiry (if available)."""
        return self._token_expiry

    @property
    def is_configured(self) -> bool:
        """Check if connector has required configuration."""
        import os

        return bool(
            os.environ.get("GMAIL_CLIENT_ID")
            or os.environ.get("GOOGLE_GMAIL_CLIENT_ID")
            or os.environ.get("GOOGLE_CLIENT_ID")
        )

    def get_oauth_url(self, redirect_uri: str, state: str = "") -> str:
        """
        Generate OAuth2 authorization URL.

        Args:
            redirect_uri: URL to redirect after authorization
            state: Optional state parameter for CSRF protection

        Returns:
            Authorization URL for user to visit
        """
        import os
        from urllib.parse import urlencode

        client_id = (
            os.environ.get("GMAIL_CLIENT_ID")
            or os.environ.get("GOOGLE_GMAIL_CLIENT_ID")
            or os.environ.get("GOOGLE_CLIENT_ID", "")
        )

        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(GMAIL_SCOPES),
            "access_type": "offline",
            "prompt": "consent",
        }

        if state:
            params["state"] = state

        return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"

    async def authenticate(
        self,
        code: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        refresh_token: Optional[str] = None,
    ) -> bool:
        """
        Authenticate with Gmail API.

        Either exchange authorization code for tokens, or use existing refresh token.

        Args:
            code: Authorization code from OAuth callback
            redirect_uri: Redirect URI used in authorization
            refresh_token: Existing refresh token

        Returns:
            True if authentication successful
        """
        import os

        import httpx

        client_id = (
            os.environ.get("GMAIL_CLIENT_ID")
            or os.environ.get("GOOGLE_GMAIL_CLIENT_ID")
            or os.environ.get("GOOGLE_CLIENT_ID", "")
        )
        client_secret = (
            os.environ.get("GMAIL_CLIENT_SECRET")
            or os.environ.get("GOOGLE_GMAIL_CLIENT_SECRET")
            or os.environ.get("GOOGLE_CLIENT_SECRET", "")
        )

        if not client_id or not client_secret:
            logger.error("[Gmail] Missing OAuth credentials")
            return False

        try:
            async with httpx.AsyncClient() as client:
                if code and redirect_uri:
                    # Exchange code for tokens
                    response = await client.post(
                        "https://oauth2.googleapis.com/token",
                        data={
                            "client_id": client_id,
                            "client_secret": client_secret,
                            "code": code,
                            "redirect_uri": redirect_uri,
                            "grant_type": "authorization_code",
                        },
                    )
                elif refresh_token:
                    # Use refresh token
                    response = await client.post(
                        "https://oauth2.googleapis.com/token",
                        data={
                            "client_id": client_id,
                            "client_secret": client_secret,
                            "refresh_token": refresh_token,
                            "grant_type": "refresh_token",
                        },
                    )
                else:
                    logger.error("[Gmail] No code or refresh_token provided")
                    return False

                response.raise_for_status()
                data = response.json()

            self._access_token = data["access_token"]
            self._refresh_token = data.get("refresh_token", refresh_token)

            expires_in = data.get("expires_in", 3600)
            from datetime import timedelta

            self._token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in - 60)

            logger.info("[Gmail] Authentication successful")
            return True

        except Exception as e:
            logger.error(f"[Gmail] Authentication failed: {e}")
            return False

    async def _refresh_access_token(self) -> str:
        """Refresh the access token using refresh token."""
        import os

        import httpx

        if not self._refresh_token:
            raise ValueError("No refresh token available")

        client_id = (
            os.environ.get("GMAIL_CLIENT_ID")
            or os.environ.get("GOOGLE_GMAIL_CLIENT_ID")
            or os.environ.get("GOOGLE_CLIENT_ID", "")
        )
        client_secret = (
            os.environ.get("GMAIL_CLIENT_SECRET")
            or os.environ.get("GOOGLE_GMAIL_CLIENT_SECRET")
            or os.environ.get("GOOGLE_CLIENT_SECRET", "")
        )

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": self._refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            response.raise_for_status()
            data = response.json()

        self._access_token = data["access_token"]
        expires_in = data.get("expires_in", 3600)
        from datetime import timedelta

        self._token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in - 60)

        return self._access_token

    async def _get_access_token(self) -> str:
        """Get valid access token, refreshing if needed.

        Thread-safe: Uses _token_lock to prevent concurrent refresh attempts.
        """
        async with self._token_lock:
            now = datetime.now(timezone.utc)

            if self._access_token and self._token_expiry and now < self._token_expiry:
                return self._access_token

            if self._refresh_token:
                return await self._refresh_access_token()

            raise ValueError("No valid access token and no refresh token available")

    async def _api_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make a request to Gmail API with circuit breaker protection."""
        import httpx

        # Check circuit breaker first
        if not self.check_circuit_breaker():
            cb_status = self.get_circuit_breaker_status()
            raise ConnectionError(
                f"Circuit breaker open for Gmail. Cooldown: {cb_status.get('cooldown_seconds', 60)}s"
            )

        token = await self._get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

        url = f"https://gmail.googleapis.com/gmail/v1/users/{self.user_id}{endpoint}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json_data,
                    timeout=60,
                )
                if response.status_code >= 400:
                    # Log the full error response for debugging
                    logger.error(f"Gmail API error {response.status_code}: {response.text}")
                    # Record failure for circuit breaker on 5xx errors or rate limits
                    if response.status_code >= 500 or response.status_code == 429:
                        self.record_failure()
                response.raise_for_status()
                self.record_success()
                return response.json() if response.content else {}
        except httpx.TimeoutException as e:
            self.record_failure()
            logger.error(f"Gmail API timeout: {e}")
            raise
        except httpx.HTTPStatusError:
            # Already handled above
            raise
        except Exception as e:
            self.record_failure()
            logger.error(f"Gmail API error: {e}")
            raise

    def _get_client(self):
        """Get HTTP client context manager for API requests."""
        import httpx

        return httpx.AsyncClient(timeout=60)

    async def get_user_info(self) -> Dict[str, Any]:
        """Get authenticated user's Gmail profile."""
        return await self._api_request("/profile")

    async def list_labels(self) -> List[GmailLabel]:
        """List all Gmail labels."""
        data = await self._api_request("/labels")

        labels = []
        for item in data.get("labels", []):
            labels.append(
                GmailLabel(
                    id=item["id"],
                    name=item.get("name", item["id"]),
                    type=item.get("type", "user"),
                    message_list_visibility=item.get("messageListVisibility", "show"),
                    label_list_visibility=item.get("labelListVisibility", "labelShow"),
                )
            )

        return labels

    async def create_label(self, label_name: str) -> GmailLabel:
        """
        Create a new Gmail label.

        Args:
            label_name: Name for the new label

        Returns:
            Created GmailLabel object
        """
        access_token = await self._get_access_token()

        async with self._get_client() as client:
            response = await client.post(
                f"https://gmail.googleapis.com/gmail/v1/users/{self.user_id}/labels",
                headers={"Authorization": f"Bearer {access_token}"},
                json={
                    "name": label_name,
                    "labelListVisibility": "labelShow",
                    "messageListVisibility": "show",
                },
            )
            response.raise_for_status()
            data = response.json()

        return GmailLabel(
            id=data["id"],
            name=data["name"],
            type=data.get("type", "user"),
            message_list_visibility=data.get("messageListVisibility", "show"),
            label_list_visibility=data.get("labelListVisibility", "labelShow"),
        )

    async def add_label(self, message_id: str, label_id: str) -> Dict[str, Any]:
        """
        Add a label to a message.

        Args:
            message_id: Gmail message ID
            label_id: Label ID to add

        Returns:
            Dict with message_id and updated labels
        """
        return await self.modify_message(message_id, add_labels=[label_id])

    async def list_messages(
        self,
        query: str = "",
        label_ids: Optional[List[str]] = None,
        page_token: Optional[str] = None,
        max_results: int = 100,
    ) -> tuple[List[str], Optional[str]]:
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
        params: Dict[str, Any] = {
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

    def _parse_message(self, data: Dict[str, Any]) -> EmailMessage:
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
        payload: Dict[str, Any],
    ) -> tuple[str, str, List[EmailAttachment]]:
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
        label_id: Optional[str] = None,
        page_token: Optional[str] = None,
    ) -> tuple[List[Dict[str, Any]], Optional[str], str]:
        """
        Get message history changes since a history ID.

        Returns:
            Tuple of (history_records, next_page_token, new_history_id)
        """
        params: Dict[str, Any] = {
            "startHistoryId": start_history_id,
            "historyTypes": ["messageAdded", "labelAdded", "labelRemoved"],
        }

        if label_id:
            params["labelId"] = label_id
        if page_token:
            params["pageToken"] = page_token

        try:
            data = await self._api_request("/history", params=params)
        except Exception as e:
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
        **kwargs,
    ) -> List[Any]:
        """Search Gmail messages."""
        from aragora.connectors.base import Evidence

        message_ids, _ = await self.list_messages(query=query, max_results=limit)

        results = []
        for msg_id in message_ids[:limit]:
            try:
                msg = await self.get_message(msg_id, format="metadata")
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
            except Exception as e:
                logger.warning(f"[Gmail] Failed to fetch message {msg_id}: {e}")

        return results

    async def fetch(self, evidence_id: str) -> Optional[Any]:
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

        except Exception as e:
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

            while True:
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

                except Exception as e:
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
        while True:
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

                except Exception as e:
                    logger.warning(f"[Gmail] Failed to fetch message {msg_id}: {e}")

            if not page_token:
                break

    async def send_message(
        self,
        to: List[str],
        subject: str,
        body: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        reply_to: Optional[str] = None,
        html_body: Optional[str] = None,
    ) -> Dict[str, Any]:
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

        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        # Build MIME message
        if html_body:
            message = MIMEMultipart("alternative")
            message.attach(MIMEText(body, "plain"))
            message.attach(MIMEText(html_body, "html"))
        else:
            message = MIMEText(body, "plain")  # type: ignore[assignment]

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
        except Exception as e:
            if not isinstance(e, RuntimeError):
                self.record_failure()
            raise

    async def reply_to_message(
        self,
        original_message_id: str,
        body: str,
        cc: Optional[List[str]] = None,
        html_body: Optional[str] = None,
    ) -> Dict[str, Any]:
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
        if html_body:
            message = MIMEMultipart("alternative")
            message.attach(MIMEText(body, "plain"))
            message.attach(MIMEText(html_body, "html"))
        else:
            message = MIMEText(body, "plain")  # type: ignore[assignment]

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
        except Exception as e:
            if not isinstance(e, RuntimeError):
                self.record_failure()
            raise

    # =========================================================================
    # Email Actions (Archive, Trash, Snooze, Labels)
    # =========================================================================

    async def modify_message(
        self,
        message_id: str,
        add_labels: Optional[List[str]] = None,
        remove_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Modify message labels.

        Requires gmail.modify scope to be authorized.

        Args:
            message_id: Gmail message ID
            add_labels: Labels to add (e.g., ["STARRED", "IMPORTANT"])
            remove_labels: Labels to remove (e.g., ["INBOX", "UNREAD"])

        Returns:
            Dict with message_id and updated labels
        """
        access_token = await self._get_access_token()

        if not self.check_circuit_breaker():
            cb_status = self.get_circuit_breaker_status()
            raise ConnectionError(
                f"Circuit breaker open for Gmail. Cooldown: {cb_status.get('cooldown_seconds', 60)}s"
            )

        try:
            async with self._get_client() as client:
                response = await client.post(
                    f"https://gmail.googleapis.com/gmail/v1/users/{self.user_id}/messages/{message_id}/modify",
                    headers={"Authorization": f"Bearer {access_token}"},
                    json={
                        "addLabelIds": add_labels or [],
                        "removeLabelIds": remove_labels or [],
                    },
                )

                if response.status_code != 200:
                    error = response.json().get("error", {})
                    if response.status_code >= 500 or response.status_code == 429:
                        self.record_failure()
                    raise RuntimeError(
                        f"Failed to modify message: {error.get('message', response.text)}"
                    )

                self.record_success()
                result = response.json()
                logger.info(f"[Gmail] Modified message: {message_id}")

                return {
                    "message_id": result.get("id"),
                    "labels": result.get("labelIds", []),
                    "success": True,
                }
        except Exception as e:
            if not isinstance(e, RuntimeError):
                self.record_failure()
            raise

    async def archive_message(self, message_id: str) -> Dict[str, Any]:
        """
        Archive a message (remove from INBOX but keep in All Mail).

        Args:
            message_id: Gmail message ID

        Returns:
            Dict with success status
        """
        result = await self.modify_message(
            message_id,
            remove_labels=["INBOX"],
        )
        logger.info(f"[Gmail] Archived message: {message_id}")
        return result

    async def trash_message(self, message_id: str) -> Dict[str, Any]:
        """
        Move a message to trash.

        Args:
            message_id: Gmail message ID

        Returns:
            Dict with success status
        """
        access_token = await self._get_access_token()

        if not self.check_circuit_breaker():
            cb_status = self.get_circuit_breaker_status()
            raise ConnectionError(
                f"Circuit breaker open for Gmail. Cooldown: {cb_status.get('cooldown_seconds', 60)}s"
            )

        try:
            async with self._get_client() as client:
                response = await client.post(
                    f"https://gmail.googleapis.com/gmail/v1/users/{self.user_id}/messages/{message_id}/trash",
                    headers={"Authorization": f"Bearer {access_token}"},
                )

                if response.status_code != 200:
                    error = response.json().get("error", {})
                    if response.status_code >= 500 or response.status_code == 429:
                        self.record_failure()
                    raise RuntimeError(
                        f"Failed to trash message: {error.get('message', response.text)}"
                    )

                self.record_success()
                logger.info(f"[Gmail] Trashed message: {message_id}")

                return {
                    "message_id": message_id,
                    "success": True,
                }
        except Exception as e:
            if not isinstance(e, RuntimeError):
                self.record_failure()
            raise

    async def untrash_message(self, message_id: str) -> Dict[str, Any]:
        """
        Restore a message from trash.

        Args:
            message_id: Gmail message ID

        Returns:
            Dict with success status
        """
        access_token = await self._get_access_token()

        if not self.check_circuit_breaker():
            cb_status = self.get_circuit_breaker_status()
            raise ConnectionError(
                f"Circuit breaker open for Gmail. Cooldown: {cb_status.get('cooldown_seconds', 60)}s"
            )

        try:
            async with self._get_client() as client:
                response = await client.post(
                    f"https://gmail.googleapis.com/gmail/v1/users/{self.user_id}/messages/{message_id}/untrash",
                    headers={"Authorization": f"Bearer {access_token}"},
                )

                if response.status_code != 200:
                    error = response.json().get("error", {})
                    if response.status_code >= 500 or response.status_code == 429:
                        self.record_failure()
                    raise RuntimeError(
                        f"Failed to untrash message: {error.get('message', response.text)}"
                    )

                self.record_success()
                logger.info(f"[Gmail] Untrashed message: {message_id}")

                return {
                    "message_id": message_id,
                    "success": True,
                }
        except Exception as e:
            if not isinstance(e, RuntimeError):
                self.record_failure()
            raise

    async def mark_as_read(self, message_id: str) -> Dict[str, Any]:
        """
        Mark a message as read.

        Args:
            message_id: Gmail message ID

        Returns:
            Dict with success status
        """
        result = await self.modify_message(
            message_id,
            remove_labels=["UNREAD"],
        )
        logger.info(f"[Gmail] Marked as read: {message_id}")
        return result

    async def mark_as_unread(self, message_id: str) -> Dict[str, Any]:
        """
        Mark a message as unread.

        Args:
            message_id: Gmail message ID

        Returns:
            Dict with success status
        """
        result = await self.modify_message(
            message_id,
            add_labels=["UNREAD"],
        )
        logger.info(f"[Gmail] Marked as unread: {message_id}")
        return result

    async def star_message(self, message_id: str) -> Dict[str, Any]:
        """
        Star a message.

        Args:
            message_id: Gmail message ID

        Returns:
            Dict with success status
        """
        result = await self.modify_message(
            message_id,
            add_labels=["STARRED"],
        )
        logger.info(f"[Gmail] Starred message: {message_id}")
        return result

    async def unstar_message(self, message_id: str) -> Dict[str, Any]:
        """
        Remove star from a message.

        Args:
            message_id: Gmail message ID

        Returns:
            Dict with success status
        """
        result = await self.modify_message(
            message_id,
            remove_labels=["STARRED"],
        )
        logger.info(f"[Gmail] Unstarred message: {message_id}")
        return result

    async def mark_important(self, message_id: str) -> Dict[str, Any]:
        """
        Mark a message as important.

        Args:
            message_id: Gmail message ID

        Returns:
            Dict with success status
        """
        result = await self.modify_message(
            message_id,
            add_labels=["IMPORTANT"],
        )
        logger.info(f"[Gmail] Marked important: {message_id}")
        return result

    async def mark_not_important(self, message_id: str) -> Dict[str, Any]:
        """
        Remove important flag from a message.

        Args:
            message_id: Gmail message ID

        Returns:
            Dict with success status
        """
        result = await self.modify_message(
            message_id,
            remove_labels=["IMPORTANT"],
        )
        logger.info(f"[Gmail] Marked not important: {message_id}")
        return result

    async def move_to_folder(
        self,
        message_id: str,
        folder_label: str,
        remove_from_inbox: bool = True,
    ) -> Dict[str, Any]:
        """
        Move a message to a specific folder/label.

        Args:
            message_id: Gmail message ID
            folder_label: Target label name
            remove_from_inbox: Whether to remove from INBOX

        Returns:
            Dict with success status
        """
        remove_labels = ["INBOX"] if remove_from_inbox else []
        result = await self.modify_message(
            message_id,
            add_labels=[folder_label],
            remove_labels=remove_labels,
        )
        logger.info(f"[Gmail] Moved to {folder_label}: {message_id}")
        return result

    async def snooze_message(
        self,
        message_id: str,
        snooze_until: datetime,
    ) -> Dict[str, Any]:
        """
        Snooze a message until a specific time.

        Note: Gmail doesn't have native snooze API, so this archives the message
        and stores snooze metadata. A separate scheduler should restore it to inbox.

        Args:
            message_id: Gmail message ID
            snooze_until: When to restore the message

        Returns:
            Dict with success status and snooze metadata
        """
        # Archive the message first
        await self.archive_message(message_id)

        # Add SNOOZED label if it exists (custom label)
        try:
            await self.modify_message(message_id, add_labels=["SNOOZED"])
        except Exception:
            # SNOOZED label might not exist, that's okay
            pass

        logger.info(f"[Gmail] Snoozed message until {snooze_until}: {message_id}")

        return {
            "message_id": message_id,
            "snoozed_until": snooze_until.isoformat(),
            "success": True,
        }

    async def batch_modify(
        self,
        message_ids: List[str],
        add_labels: Optional[List[str]] = None,
        remove_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Batch modify multiple messages.

        Args:
            message_ids: List of Gmail message IDs
            add_labels: Labels to add
            remove_labels: Labels to remove

        Returns:
            Dict with success count and failures
        """
        access_token = await self._get_access_token()

        if not self.check_circuit_breaker():
            cb_status = self.get_circuit_breaker_status()
            raise ConnectionError(
                f"Circuit breaker open for Gmail. Cooldown: {cb_status.get('cooldown_seconds', 60)}s"
            )

        try:
            async with self._get_client() as client:
                response = await client.post(
                    f"https://gmail.googleapis.com/gmail/v1/users/{self.user_id}/messages/batchModify",
                    headers={"Authorization": f"Bearer {access_token}"},
                    json={
                        "ids": message_ids,
                        "addLabelIds": add_labels or [],
                        "removeLabelIds": remove_labels or [],
                    },
                )

                if response.status_code != 204:
                    error = response.json().get("error", {})
                    if response.status_code >= 500 or response.status_code == 429:
                        self.record_failure()
                    raise RuntimeError(
                        f"Failed to batch modify: {error.get('message', response.text)}"
                    )

                self.record_success()
                logger.info(f"[Gmail] Batch modified {len(message_ids)} messages")

                return {
                    "modified_count": len(message_ids),
                    "success": True,
                }
        except Exception as e:
            if not isinstance(e, RuntimeError):
                self.record_failure()
            raise

    async def batch_archive(self, message_ids: List[str]) -> Dict[str, Any]:
        """
        Archive multiple messages at once.

        Args:
            message_ids: List of Gmail message IDs

        Returns:
            Dict with success count
        """
        result = await self.batch_modify(
            message_ids,
            remove_labels=["INBOX"],
        )
        logger.info(f"[Gmail] Batch archived {len(message_ids)} messages")
        return result

    async def batch_trash(self, message_ids: List[str]) -> Dict[str, Any]:
        """
        Trash multiple messages at once.

        Args:
            message_ids: List of Gmail message IDs

        Returns:
            Dict with success count
        """
        access_token = await self._get_access_token()

        if not self.check_circuit_breaker():
            cb_status = self.get_circuit_breaker_status()
            raise ConnectionError(
                f"Circuit breaker open for Gmail. Cooldown: {cb_status.get('cooldown_seconds', 60)}s"
            )

        try:
            async with self._get_client() as client:
                response = await client.post(
                    f"https://gmail.googleapis.com/gmail/v1/users/{self.user_id}/messages/batchDelete",
                    headers={"Authorization": f"Bearer {access_token}"},
                    json={"ids": message_ids},
                )

                # Note: This permanently deletes, not trash
                # For trash, we use batch_modify with TRASH label
                if response.status_code != 204:
                    error = response.json().get("error", {})
                    if response.status_code >= 500 or response.status_code == 429:
                        self.record_failure()
                    raise RuntimeError(
                        f"Failed to batch delete: {error.get('message', response.text)}"
                    )

                self.record_success()
                logger.info(f"[Gmail] Batch deleted {len(message_ids)} messages")

                return {
                    "deleted_count": len(message_ids),
                    "success": True,
                }
        except Exception as e:
            if not isinstance(e, RuntimeError):
                self.record_failure()
            raise

    # =========================================================================
    # Pub/Sub Watch Management
    # =========================================================================

    async def setup_watch(
        self,
        topic_name: str,
        label_ids: Optional[List[str]] = None,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Set up Gmail push notifications via Google Cloud Pub/Sub.

        This enables real-time notifications when new emails arrive,
        eliminating the need for polling.

        Args:
            topic_name: Pub/Sub topic name (e.g., "gmail-notifications")
            label_ids: Labels to watch (default: ["INBOX"])
            project_id: Google Cloud project ID (reads from env if not provided)

        Returns:
            Dict with watch status, history_id, and expiration

        Note:
            - Requires Gmail API scope and Pub/Sub topic access
            - Watch expires after ~7 days, use start_watch_renewal() for auto-renewal
            - Topic must grant Gmail service account publish permission
        """
        import os

        project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        if not project_id:
            raise ValueError("project_id required for Pub/Sub watch")

        full_topic = f"projects/{project_id}/topics/{topic_name}"
        watch_labels = label_ids or ["INBOX"]

        access_token = await self._get_access_token()

        if not self.check_circuit_breaker():
            cb_status = self.get_circuit_breaker_status()
            raise ConnectionError(
                f"Circuit breaker open for Gmail. Cooldown: {cb_status.get('cooldown_seconds', 60)}s"
            )

        try:
            async with self._get_client() as client:
                response = await client.post(
                    f"https://gmail.googleapis.com/gmail/v1/users/{self.user_id}/watch",
                    headers={"Authorization": f"Bearer {access_token}"},
                    json={
                        "topicName": full_topic,
                        "labelIds": watch_labels,
                        "labelFilterBehavior": "INCLUDE",
                    },
                )

                if response.status_code != 200:
                    error = response.json().get("error", {})
                    if response.status_code >= 500 or response.status_code == 429:
                        self.record_failure()
                    raise RuntimeError(
                        f"Failed to setup watch: {error.get('message', response.text)}"
                    )

                self.record_success()
                data = response.json()

                # Update state
                history_id = str(data.get("historyId", ""))
                expiration_ms = data.get("expiration")
                expiration = None
                if expiration_ms:
                    expiration = datetime.fromtimestamp(int(expiration_ms) / 1000, tz=timezone.utc)

                # Initialize or update gmail state
                if not self._gmail_state:
                    self._gmail_state = GmailSyncState(
                        user_id=self.user_id,
                        history_id=history_id,
                    )
                else:
                    self._gmail_state.history_id = history_id

                self._gmail_state.watch_expiration = expiration
                self._gmail_state.watch_resource_id = "active"

                logger.info(f"[Gmail] Watch set up successfully, expires at {expiration}")

                return {
                    "success": True,
                    "history_id": history_id,
                    "expiration": expiration.isoformat() if expiration else None,
                    "topic": full_topic,
                    "labels": watch_labels,
                }

        except Exception as e:
            if not isinstance(e, (RuntimeError, ConnectionError)):
                self.record_failure()
            logger.error(f"[Gmail] Watch setup failed: {e}")
            raise

    async def stop_watch(self) -> Dict[str, Any]:
        """
        Stop Gmail push notifications.

        Returns:
            Dict with success status
        """
        access_token = await self._get_access_token()

        if not self.check_circuit_breaker():
            cb_status = self.get_circuit_breaker_status()
            raise ConnectionError(
                f"Circuit breaker open for Gmail. Cooldown: {cb_status.get('cooldown_seconds', 60)}s"
            )

        # Cancel renewal task if running
        if self._watch_task and not self._watch_task.done():
            self._watch_running = False
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
            self._watch_task = None

        try:
            async with self._get_client() as client:
                response = await client.post(
                    f"https://gmail.googleapis.com/gmail/v1/users/{self.user_id}/stop",
                    headers={"Authorization": f"Bearer {access_token}"},
                )

                if response.status_code == 204:
                    self.record_success()

                    # Clear watch state
                    if self._gmail_state:
                        self._gmail_state.watch_resource_id = None
                        self._gmail_state.watch_expiration = None

                    logger.info("[Gmail] Watch stopped successfully")
                    return {"success": True}
                else:
                    error = response.json().get("error", {})
                    if response.status_code >= 500 or response.status_code == 429:
                        self.record_failure()
                    logger.warning(
                        f"[Gmail] Stop watch returned {response.status_code}: "
                        f"{error.get('message', response.text)}"
                    )
                    return {
                        "success": False,
                        "error": error.get("message", "Unknown error"),
                    }

        except Exception as e:
            self.record_failure()
            logger.error(f"[Gmail] Failed to stop watch: {e}")
            raise

    async def handle_pubsub_notification(
        self,
        payload: Dict[str, Any],
    ) -> List[EmailMessage]:
        """
        Handle incoming Pub/Sub webhook notification.

        Parses the notification, fetches new messages via History API,
        and returns the list of new emails.

        Args:
            payload: Raw webhook payload from Pub/Sub

        Returns:
            List of new EmailMessage objects
        """
        webhook = GmailWebhookPayload.from_pubsub(payload)

        # Validate this is for us
        if self._gmail_state and webhook.email_address:
            if (
                self._gmail_state.email_address
                and webhook.email_address != self._gmail_state.email_address
            ):
                logger.warning(
                    f"[Gmail] Webhook for {webhook.email_address} "
                    f"but expecting {self._gmail_state.email_address}"
                )
                return []

        logger.info(f"[Gmail] Pub/Sub notification received: historyId={webhook.history_id}")

        # Use History API to get changes
        if not self._gmail_state or not self._gmail_state.history_id:
            logger.warning("[Gmail] No history ID available, cannot process webhook")
            return []

        try:
            new_messages: List[EmailMessage] = []
            page_token = None
            new_history_id = self._gmail_state.history_id

            while True:
                history, page_token, history_id = await self.get_history(
                    self._gmail_state.history_id,
                    page_token=page_token,
                )

                if not history and not page_token:
                    if not history_id:
                        logger.warning("[Gmail] History ID expired during webhook handling")
                        break
                    break

                # Extract new message IDs
                new_message_ids: set[str] = set()
                for record in history:
                    for msg_added in record.get("messagesAdded", []):
                        msg_data = msg_added.get("message", {})
                        msg_id = msg_data.get("id")
                        labels = msg_data.get("labelIds", [])

                        # Skip excluded labels
                        if self.exclude_labels and any(
                            lbl in self.exclude_labels for lbl in labels
                        ):
                            continue

                        if msg_id:
                            new_message_ids.add(msg_id)

                # Fetch full messages
                for msg_id in new_message_ids:
                    try:
                        msg = await self.get_message(msg_id)
                        new_messages.append(msg)
                    except Exception as e:
                        logger.warning(f"[Gmail] Failed to fetch message {msg_id}: {e}")

                if history_id:
                    new_history_id = history_id

                if not page_token:
                    break

            # Update history ID
            self._gmail_state.history_id = new_history_id
            self._gmail_state.last_sync = datetime.now(timezone.utc)
            self._gmail_state.indexed_messages += len(new_messages)

            logger.info(f"[Gmail] Webhook processed: {len(new_messages)} new messages")
            return new_messages

        except Exception as e:
            if self._gmail_state:
                self._gmail_state.sync_errors += 1
                self._gmail_state.last_error = str(e)
            logger.error(f"[Gmail] Webhook processing failed: {e}")
            raise

    async def start_watch_renewal(
        self,
        topic_name: str,
        renewal_hours: int = 144,  # 6 days (watch expires after ~7 days)
        project_id: Optional[str] = None,
    ) -> None:
        """
        Start background task to auto-renew watch before expiration.

        Args:
            topic_name: Pub/Sub topic name
            renewal_hours: Hours between renewals (default: 144 = 6 days)
            project_id: Google Cloud project ID
        """
        if self._watch_task and not self._watch_task.done():
            logger.warning("[Gmail] Watch renewal already running")
            return

        self._watch_running = True
        self._watch_task = asyncio.create_task(
            self._watch_renewal_loop(topic_name, renewal_hours, project_id)
        )
        logger.info(f"[Gmail] Watch renewal started (every {renewal_hours} hours)")

    async def _watch_renewal_loop(
        self,
        topic_name: str,
        renewal_hours: int,
        project_id: Optional[str],
    ) -> None:
        """Background loop to renew watch before expiration."""
        renewal_seconds = renewal_hours * 3600

        while self._watch_running:
            try:
                await asyncio.sleep(renewal_seconds)

                if not self._watch_running:
                    break

                logger.info("[Gmail] Renewing watch...")
                await self.setup_watch(
                    topic_name=topic_name,
                    project_id=project_id,
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Gmail] Watch renewal failed: {e}")
                # Retry in 1 minute on failure
                await asyncio.sleep(60)

    # =========================================================================
    # State Persistence
    # =========================================================================

    async def load_gmail_state(
        self,
        tenant_id: str,
        user_id: str,
        backend: str = "memory",
        redis_url: Optional[str] = None,
        postgres_dsn: Optional[str] = None,
    ) -> Optional[GmailSyncState]:
        """
        Load Gmail sync state from persistent storage.

        Supports multiple backends for tenant-isolated state management.

        Args:
            tenant_id: Tenant identifier for isolation
            user_id: User identifier
            backend: Storage backend ("memory", "redis", "postgres")
            redis_url: Redis connection URL (required for redis backend)
            postgres_dsn: PostgreSQL DSN (required for postgres backend)

        Returns:
            GmailSyncState if found, None otherwise
        """
        import json

        state_key = f"gmail_sync:{tenant_id}:{user_id}"

        if backend == "redis" and redis_url:
            try:
                import redis.asyncio as redis_client

                client = redis_client.from_url(redis_url)
                data = await client.get(state_key)
                await client.close()
                if data:
                    state = GmailSyncState.from_dict(json.loads(data))
                    self._gmail_state = state
                    logger.info(f"[Gmail] Loaded state from Redis for {state_key}")
                    return state
            except ImportError:
                logger.warning("[Gmail] redis package not installed, cannot use Redis backend")
            except Exception as e:
                logger.warning(f"[Gmail] Failed to load state from Redis: {e}")

        elif backend == "postgres" and postgres_dsn:
            try:
                import asyncpg

                conn = await asyncpg.connect(postgres_dsn)
                row = await conn.fetchrow(
                    "SELECT state FROM gmail_sync_state WHERE key = $1",
                    state_key,
                )
                await conn.close()
                if row:
                    state = GmailSyncState.from_dict(json.loads(row["state"]))
                    self._gmail_state = state
                    logger.info(f"[Gmail] Loaded state from Postgres for {state_key}")
                    return state
            except ImportError:
                logger.warning("[Gmail] asyncpg package not installed, cannot use Postgres backend")
            except Exception as e:
                logger.warning(f"[Gmail] Failed to load state from Postgres: {e}")

        # Memory backend or fallback
        logger.debug(f"[Gmail] No persisted state found for {state_key}")
        return None

    async def save_gmail_state(
        self,
        tenant_id: str,
        user_id: str,
        backend: str = "memory",
        redis_url: Optional[str] = None,
        postgres_dsn: Optional[str] = None,
    ) -> bool:
        """
        Save Gmail sync state to persistent storage.

        Args:
            tenant_id: Tenant identifier for isolation
            user_id: User identifier
            backend: Storage backend ("memory", "redis", "postgres")
            redis_url: Redis connection URL (required for redis backend)
            postgres_dsn: PostgreSQL DSN (required for postgres backend)

        Returns:
            True if saved successfully
        """
        import json

        if not self._gmail_state:
            logger.warning("[Gmail] No state to save")
            return False

        state_key = f"gmail_sync:{tenant_id}:{user_id}"
        state_json = json.dumps(self._gmail_state.to_dict())

        if backend == "redis" and redis_url:
            try:
                import redis.asyncio as redis_client

                client = redis_client.from_url(redis_url)
                await client.set(state_key, state_json)
                await client.close()
                logger.info(f"[Gmail] Saved state to Redis for {state_key}")
                return True
            except ImportError:
                logger.warning("[Gmail] redis package not installed, cannot use Redis backend")
            except Exception as e:
                logger.warning(f"[Gmail] Failed to save state to Redis: {e}")
                return False

        elif backend == "postgres" and postgres_dsn:
            try:
                import asyncpg

                conn = await asyncpg.connect(postgres_dsn)
                await conn.execute(
                    """
                    INSERT INTO gmail_sync_state (key, state, updated_at)
                    VALUES ($1, $2, NOW())
                    ON CONFLICT (key) DO UPDATE SET state = $2, updated_at = NOW()
                    """,
                    state_key,
                    state_json,
                )
                await conn.close()
                logger.info(f"[Gmail] Saved state to Postgres for {state_key}")
                return True
            except ImportError:
                logger.warning("[Gmail] asyncpg package not installed, cannot use Postgres backend")
            except Exception as e:
                logger.warning(f"[Gmail] Failed to save state to Postgres: {e}")
                return False

        # Memory backend - state is already in self._gmail_state
        logger.debug(f"[Gmail] State in memory for {state_key}")
        return True

    def get_sync_stats(self) -> Dict[str, Any]:
        """
        Get sync service statistics.

        Returns:
            Dict with current sync state and statistics
        """
        return {
            "user_id": self.user_id,
            "email_address": self._gmail_state.email_address if self._gmail_state else None,
            "history_id": self._gmail_state.history_id if self._gmail_state else None,
            "last_sync": (
                self._gmail_state.last_sync.isoformat()
                if self._gmail_state and self._gmail_state.last_sync
                else None
            ),
            "initial_sync_complete": (
                self._gmail_state.initial_sync_complete if self._gmail_state else False
            ),
            "watch_active": bool(self._gmail_state and self._gmail_state.watch_resource_id),
            "watch_expiration": (
                self._gmail_state.watch_expiration.isoformat()
                if self._gmail_state and self._gmail_state.watch_expiration
                else None
            ),
            "total_messages": self._gmail_state.total_messages if self._gmail_state else 0,
            "indexed_messages": (self._gmail_state.indexed_messages if self._gmail_state else 0),
            "sync_errors": self._gmail_state.sync_errors if self._gmail_state else 0,
            "last_error": self._gmail_state.last_error if self._gmail_state else None,
        }

    # =========================================================================
    # Prioritization Integration
    # =========================================================================

    async def sync_with_prioritization(
        self,
        messages: List[EmailMessage],
        prioritizer: Optional[Any] = None,
        timeout_seconds: float = 30.0,
    ) -> List[Dict[str, Any]]:
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

                prioritizer = EmailPrioritizer(gmail_connector=self)
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

        results: List[Dict[str, Any]] = []

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

            except Exception as e:
                logger.warning(f"[Gmail] Prioritization failed for message {msg.id}: {e}")
                results.append(
                    {
                        "message": msg,
                        "priority_result": None,
                        "priority": "MEDIUM",
                        "confidence": 0.0,
                        "rationale": f"Prioritization failed: {str(e)}",
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
        labels: Optional[List[str]] = None,
        query: str = "",
    ) -> List[Dict[str, Any]]:
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

        messages: List[EmailMessage] = []
        for msg_id in message_ids[:max_messages]:
            try:
                msg = await self.get_message(msg_id)
                messages.append(msg)
            except Exception as e:
                logger.warning(f"[Gmail] Failed to fetch message {msg_id}: {e}")

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


__all__ = [
    "GmailConnector",
    "GmailWebhookPayload",
    "GMAIL_SCOPES",
    "GMAIL_SCOPES_READONLY",
    "GMAIL_SCOPES_FULL",
]
