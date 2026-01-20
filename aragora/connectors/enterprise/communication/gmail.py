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
import email
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
)

logger = logging.getLogger(__name__)


# Gmail API scopes
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.metadata",
]


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

        # OAuth tokens
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

        # Gmail-specific state
        self._gmail_state: Optional[GmailSyncState] = None

    @property
    def source_type(self) -> SourceType:
        return SourceType.DOCUMENT

    @property
    def name(self) -> str:
        return "Gmail"

    def is_configured(self) -> bool:
        """Check if connector has required configuration."""
        import os
        return bool(
            os.environ.get("GMAIL_CLIENT_ID") or
            os.environ.get("GOOGLE_GMAIL_CLIENT_ID") or
            os.environ.get("GOOGLE_CLIENT_ID")
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
            os.environ.get("GMAIL_CLIENT_ID") or
            os.environ.get("GOOGLE_GMAIL_CLIENT_ID") or
            os.environ.get("GOOGLE_CLIENT_ID", "")
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
            os.environ.get("GMAIL_CLIENT_ID") or
            os.environ.get("GOOGLE_GMAIL_CLIENT_ID") or
            os.environ.get("GOOGLE_CLIENT_ID", "")
        )
        client_secret = (
            os.environ.get("GMAIL_CLIENT_SECRET") or
            os.environ.get("GOOGLE_GMAIL_CLIENT_SECRET") or
            os.environ.get("GOOGLE_CLIENT_SECRET", "")
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
            os.environ.get("GMAIL_CLIENT_ID") or
            os.environ.get("GOOGLE_GMAIL_CLIENT_ID") or
            os.environ.get("GOOGLE_CLIENT_ID", "")
        )
        client_secret = (
            os.environ.get("GMAIL_CLIENT_SECRET") or
            os.environ.get("GOOGLE_GMAIL_CLIENT_SECRET") or
            os.environ.get("GOOGLE_CLIENT_SECRET", "")
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
        """Get valid access token, refreshing if needed."""
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
        """Make a request to Gmail API."""
        import httpx

        token = await self._get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

        url = f"https://gmail.googleapis.com/gmail/v1/users/{self.user_id}{endpoint}"

        async with httpx.AsyncClient() as client:
            response = await client.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_data,
                timeout=60,
            )
            response.raise_for_status()
            return response.json() if response.content else {}

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
                nested_text, nested_html, nested_attachments = self._extract_body_and_attachments(part)
                if not body_text:
                    body_text = nested_text
                if not body_html:
                    body_html = nested_html
                attachments.extend(nested_attachments)
                continue

            # Body content
            if part_body.get("data"):
                content = base64.urlsafe_b64decode(part_body["data"]).decode("utf-8", errors="replace")
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
                    if self.exclude_labels and any(l in self.exclude_labels for l in msg.labels):
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
                    if self.exclude_labels and any(l in self.exclude_labels for l in msg.labels):
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


__all__ = ["GmailConnector", "GMAIL_SCOPES"]
