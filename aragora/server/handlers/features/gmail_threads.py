"""
Gmail Threads and Drafts Management.

Provides REST API endpoints for Gmail thread and draft operations:
- GET /api/v1/gmail/threads - List threads
- GET /api/v1/gmail/threads/:id - Get thread with all messages
- POST /api/v1/gmail/threads/:id/archive - Archive thread
- POST /api/v1/gmail/threads/:id/trash - Trash thread
- POST /api/v1/gmail/threads/:id/labels - Modify thread labels
- POST /api/v1/gmail/drafts - Create draft
- GET /api/v1/gmail/drafts - List drafts
- GET /api/v1/gmail/drafts/:id - Get draft
- PUT /api/v1/gmail/drafts/:id - Update draft
- DELETE /api/v1/gmail/drafts/:id - Delete draft
- POST /api/v1/gmail/drafts/:id/send - Send draft
- GET /api/v1/gmail/messages/:id/attachments/:attachment_id - Get attachment
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any, Dict, List, Optional, Union

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)
from .gmail_ingest import get_user_state

logger = logging.getLogger(__name__)


class GmailThreadsHandler(BaseHandler):
    """Handler for Gmail threads and drafts endpoints."""

    ROUTES = [
        "/api/v1/gmail/threads",
        "/api/v1/gmail/drafts",
    ]

    ROUTE_PREFIXES = [
        "/api/v1/gmail/threads/",
        "/api/v1/gmail/drafts/",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the path."""
        if path in self.ROUTES:
            return True
        for prefix in self.ROUTE_PREFIXES:
            if path.startswith(prefix):
                return True
        # Also handle attachment downloads
        if "/attachments/" in path and path.startswith("/api/v1/gmail/messages/"):
            return True
        return False

    def handle(
        self,
        path: str,
        query_params: Dict[str, Any],
        handler: Any,
    ) -> Optional[HandlerResult]:
        """Route GET requests."""
        user_id = query_params.get("user_id", "default")
        state = get_user_state(user_id)

        if not state or not getattr(state, "refresh_token", None):
            return error_response("Not connected - please authenticate first", 401)

        # Thread operations
        if path == "/api/v1/gmail/threads":
            return self._list_threads(state, query_params)

        if path.startswith("/api/v1/gmail/threads/"):
            parts = path.split("/")
            thread_id = parts[4] if len(parts) > 4 else None
            if thread_id and len(parts) == 5:
                return self._get_thread(state, thread_id)

        # Draft operations
        if path == "/api/v1/gmail/drafts":
            return self._list_drafts(state, query_params)

        if path.startswith("/api/v1/gmail/drafts/"):
            parts = path.split("/")
            draft_id = parts[4] if len(parts) > 4 else None
            if draft_id and len(parts) == 5:
                return self._get_draft(state, draft_id)

        # Attachment download
        if "/attachments/" in path and path.startswith("/api/v1/gmail/messages/"):
            parts = path.split("/")
            # /api/v1/gmail/messages/{message_id}/attachments/{attachment_id}
            if len(parts) >= 7:
                message_id = parts[4]
                attachment_id = parts[6]
                return self._get_attachment(state, message_id, attachment_id)

        return error_response("Not found", 404)

    def handle_post(
        self,
        path: str,
        body: Dict[str, Any],
        handler: Any,
    ) -> Optional[HandlerResult]:
        """Route POST requests."""
        user_id = body.get("user_id", "default")
        state = get_user_state(user_id)

        if not state or not getattr(state, "refresh_token", None):
            return error_response("Not connected - please authenticate first", 401)

        # Thread operations
        if path.startswith("/api/v1/gmail/threads/"):
            parts = path.split("/")
            if len(parts) >= 6:
                thread_id = parts[4]
                action = parts[5]

                if action == "archive":
                    return self._archive_thread(state, thread_id)
                elif action == "trash":
                    return self._trash_thread(state, thread_id, body)
                elif action == "labels":
                    return self._modify_thread_labels(state, thread_id, body)

        # Draft operations
        if path == "/api/v1/gmail/drafts":
            return self._create_draft(state, body)

        if path.startswith("/api/v1/gmail/drafts/"):
            parts = path.split("/")
            if len(parts) >= 6:
                draft_id = parts[4]
                action = parts[5]

                if action == "send":
                    return self._send_draft(state, draft_id)

        return error_response("Not found", 404)

    def handle_put(
        self,
        path: str,
        body: Dict[str, Any],
        handler: Any,
    ) -> Optional[HandlerResult]:
        """Route PUT requests."""
        user_id = body.get("user_id", "default")
        state = get_user_state(user_id)

        if not state or not getattr(state, "refresh_token", None):
            return error_response("Not connected", 401)

        # Draft update
        if path.startswith("/api/v1/gmail/drafts/"):
            parts = path.split("/")
            draft_id = parts[4] if len(parts) > 4 else None
            if draft_id and len(parts) == 5:
                return self._update_draft(state, draft_id, body)

        return error_response("Not found", 404)

    def handle_delete(
        self,
        path: str,
        query_params: Dict[str, Any],
        handler: Any,
    ) -> Optional[HandlerResult]:
        """Route DELETE requests."""
        user_id = query_params.get("user_id", "default")
        state = get_user_state(user_id)

        if not state or not getattr(state, "refresh_token", None):
            return error_response("Not connected", 401)

        # Draft deletion
        if path.startswith("/api/v1/gmail/drafts/"):
            parts = path.split("/")
            draft_id = parts[4] if len(parts) > 4 else None
            if draft_id and len(parts) == 5:
                return self._delete_draft(state, draft_id)

        return error_response("Not found", 404)

    # =========================================================================
    # Thread Operations
    # =========================================================================

    def _list_threads(self, state: Any, query_params: Dict[str, Any]) -> HandlerResult:
        """List Gmail threads."""
        query = query_params.get("q", query_params.get("query", ""))
        label_ids = (
            query_params.get("label_ids", "").split(",") if query_params.get("label_ids") else None
        )
        max_results = int(query_params.get("limit", 20))
        page_token = query_params.get("page_token")

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                threads, next_page = loop.run_until_complete(
                    self._api_list_threads(state, query, label_ids, max_results, page_token)
                )
                return json_response(
                    {
                        "threads": threads,
                        "count": len(threads),
                        "next_page_token": next_page,
                    }
                )
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"[GmailThreads] List threads failed: {e}")
            return error_response(f"Failed to list threads: {e}", 500)

    async def _api_list_threads(
        self,
        state: Any,
        query: str,
        label_ids: Optional[List[str]],
        max_results: int,
        page_token: Optional[str],
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """List threads via Gmail API."""
        import httpx

        token = state.access_token

        params: Dict[str, Any] = {"maxResults": min(max_results, 100)}
        if query:
            params["q"] = query
        if label_ids:
            params["labelIds"] = label_ids
        if page_token:
            params["pageToken"] = page_token

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://gmail.googleapis.com/gmail/v1/users/me/threads",
                headers={"Authorization": f"Bearer {token}"},
                params=params,
            )
            response.raise_for_status()
            data = response.json()

        threads = []
        for t in data.get("threads", []):
            threads.append(
                {
                    "id": t["id"],
                    "snippet": t.get("snippet", ""),
                    "history_id": t.get("historyId"),
                }
            )

        return threads, data.get("nextPageToken")

    def _get_thread(self, state: Any, thread_id: str) -> HandlerResult:
        """Get a thread with all messages."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                from aragora.connectors.enterprise.communication.gmail import GmailConnector

                connector = GmailConnector()
                connector._access_token = state.access_token
                connector._refresh_token = state.refresh_token
                connector._token_expiry = state.token_expiry

                thread = loop.run_until_complete(connector.get_thread(thread_id))

                return json_response(
                    {
                        "thread": {
                            "id": thread.id,
                            "subject": thread.subject,
                            "snippet": thread.snippet,
                            "message_count": thread.message_count,
                            "participants": thread.participants,
                            "labels": thread.labels,
                            "last_message_date": (
                                thread.last_message_date.isoformat()
                                if thread.last_message_date
                                else None
                            ),
                            "messages": [
                                {
                                    "id": msg.id,
                                    "subject": msg.subject,
                                    "from": msg.from_address,
                                    "to": msg.to_addresses,
                                    "cc": msg.cc_addresses,
                                    "date": msg.date.isoformat() if msg.date else None,
                                    "snippet": msg.snippet,
                                    "body_text": msg.body_text[:2000] if msg.body_text else None,
                                    "is_read": msg.is_read,
                                    "is_starred": msg.is_starred,
                                    "labels": msg.labels,
                                    "attachments": [
                                        {
                                            "id": a.id,
                                            "filename": a.filename,
                                            "mime_type": a.mime_type,
                                            "size": a.size,
                                        }
                                        for a in msg.attachments
                                    ],
                                }
                                for msg in thread.messages
                            ],
                        }
                    }
                )
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"[GmailThreads] Get thread failed: {e}")
            return error_response(f"Failed to get thread: {e}", 500)

    def _archive_thread(self, state: Any, thread_id: str) -> HandlerResult:
        """Archive a thread (remove INBOX label from all messages)."""
        return self._modify_thread_labels(state, thread_id, {"remove": ["INBOX"]})

    def _trash_thread(self, state: Any, thread_id: str, body: Dict[str, Any]) -> HandlerResult:
        """Move thread to trash or restore from trash."""
        to_trash = body.get("trash", True)

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                if to_trash:
                    loop.run_until_complete(self._api_trash_thread(state, thread_id))
                else:
                    loop.run_until_complete(self._api_untrash_thread(state, thread_id))

                return json_response(
                    {
                        "thread_id": thread_id,
                        "trashed": to_trash,
                        "success": True,
                    }
                )
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"[GmailThreads] Trash thread failed: {e}")
            return error_response(f"Failed to trash thread: {e}", 500)

    async def _api_trash_thread(self, state: Any, thread_id: str) -> None:
        """Trash thread via Gmail API."""
        import httpx

        token = state.access_token

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://gmail.googleapis.com/gmail/v1/users/me/threads/{thread_id}/trash",
                headers={"Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()

    async def _api_untrash_thread(self, state: Any, thread_id: str) -> None:
        """Untrash thread via Gmail API."""
        import httpx

        token = state.access_token

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://gmail.googleapis.com/gmail/v1/users/me/threads/{thread_id}/untrash",
                headers={"Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()

    def _modify_thread_labels(
        self,
        state: Any,
        thread_id: str,
        body: Dict[str, Any],
    ) -> HandlerResult:
        """Modify labels on all messages in a thread."""
        add_labels = body.get("add", [])
        remove_labels = body.get("remove", [])

        if not add_labels and not remove_labels:
            return error_response("Must specify labels to add or remove", 400)

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                loop.run_until_complete(
                    self._api_modify_thread_labels(state, thread_id, add_labels, remove_labels)
                )
                return json_response(
                    {
                        "thread_id": thread_id,
                        "success": True,
                    }
                )
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"[GmailThreads] Modify thread labels failed: {e}")
            return error_response(f"Failed to modify thread labels: {e}", 500)

    async def _api_modify_thread_labels(
        self,
        state: Any,
        thread_id: str,
        add_labels: List[str],
        remove_labels: List[str],
    ) -> Dict[str, Any]:
        """Modify thread labels via Gmail API."""
        import httpx

        token = state.access_token

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://gmail.googleapis.com/gmail/v1/users/me/threads/{thread_id}/modify",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "addLabelIds": add_labels,
                    "removeLabelIds": remove_labels,
                },
            )
            response.raise_for_status()
            return response.json()

    # =========================================================================
    # Draft Operations
    # =========================================================================

    def _list_drafts(self, state: Any, query_params: Dict[str, Any]) -> HandlerResult:
        """List Gmail drafts."""
        max_results = int(query_params.get("limit", 20))
        page_token = query_params.get("page_token")

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                drafts, next_page = loop.run_until_complete(
                    self._api_list_drafts(state, max_results, page_token)
                )
                return json_response(
                    {
                        "drafts": drafts,
                        "count": len(drafts),
                        "next_page_token": next_page,
                    }
                )
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"[GmailThreads] List drafts failed: {e}")
            return error_response(f"Failed to list drafts: {e}", 500)

    async def _api_list_drafts(
        self,
        state: Any,
        max_results: int,
        page_token: Optional[str],
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """List drafts via Gmail API."""
        import httpx

        token = state.access_token

        params: Dict[str, Any] = {"maxResults": min(max_results, 100)}
        if page_token:
            params["pageToken"] = page_token

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://gmail.googleapis.com/gmail/v1/users/me/drafts",
                headers={"Authorization": f"Bearer {token}"},
                params=params,
            )
            response.raise_for_status()
            data = response.json()

        drafts = []
        for d in data.get("drafts", []):
            drafts.append(
                {
                    "id": d["id"],
                    "message_id": d.get("message", {}).get("id"),
                    "thread_id": d.get("message", {}).get("threadId"),
                }
            )

        return drafts, data.get("nextPageToken")

    def _get_draft(self, state: Any, draft_id: str) -> HandlerResult:
        """Get a draft with message content."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                draft = loop.run_until_complete(self._api_get_draft(state, draft_id))
                return json_response({"draft": draft})
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"[GmailThreads] Get draft failed: {e}")
            return error_response(f"Failed to get draft: {e}", 500)

    async def _api_get_draft(self, state: Any, draft_id: str) -> Dict[str, Any]:
        """Get draft via Gmail API."""
        import httpx

        token = state.access_token

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://gmail.googleapis.com/gmail/v1/users/me/drafts/{draft_id}",
                headers={"Authorization": f"Bearer {token}"},
                params={"format": "full"},
            )
            response.raise_for_status()
            return response.json()

    def _create_draft(self, state: Any, body: Dict[str, Any]) -> HandlerResult:
        """Create a new draft."""
        to = body.get("to", [])
        subject = body.get("subject", "")
        body_text = body.get("body", "")
        html_body = body.get("html_body")
        reply_to_message_id = body.get("reply_to_message_id")
        thread_id = body.get("thread_id")

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                draft = loop.run_until_complete(
                    self._api_create_draft(
                        state, to, subject, body_text, html_body, reply_to_message_id, thread_id
                    )
                )
                return json_response({"draft": draft, "success": True})
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"[GmailThreads] Create draft failed: {e}")
            return error_response(f"Failed to create draft: {e}", 500)

    async def _api_create_draft(
        self,
        state: Any,
        to: List[str],
        subject: str,
        body_text: str,
        html_body: Optional[str],
        reply_to_message_id: Optional[str],
        thread_id: Optional[str],
    ) -> Dict[str, Any]:
        """Create draft via Gmail API."""
        import httpx
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        token = state.access_token

        # Build MIME message
        message: Union[MIMEMultipart, MIMEText]
        if html_body:
            message = MIMEMultipart("alternative")
            message.attach(MIMEText(body_text, "plain"))
            message.attach(MIMEText(html_body, "html"))
        else:
            message = MIMEText(body_text, "plain")

        if to:
            message["To"] = ", ".join(to)
        message["Subject"] = subject

        # Encode message
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

        draft_data: Dict[str, Any] = {"message": {"raw": raw_message}}
        if thread_id:
            draft_data["message"]["threadId"] = thread_id

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://gmail.googleapis.com/gmail/v1/users/me/drafts",
                headers={"Authorization": f"Bearer {token}"},
                json=draft_data,
            )
            response.raise_for_status()
            return response.json()

    def _update_draft(self, state: Any, draft_id: str, body: Dict[str, Any]) -> HandlerResult:
        """Update an existing draft."""
        to = body.get("to", [])
        subject = body.get("subject", "")
        body_text = body.get("body", "")
        html_body = body.get("html_body")

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                draft = loop.run_until_complete(
                    self._api_update_draft(state, draft_id, to, subject, body_text, html_body)
                )
                return json_response({"draft": draft, "success": True})
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"[GmailThreads] Update draft failed: {e}")
            return error_response(f"Failed to update draft: {e}", 500)

    async def _api_update_draft(
        self,
        state: Any,
        draft_id: str,
        to: List[str],
        subject: str,
        body_text: str,
        html_body: Optional[str],
    ) -> Dict[str, Any]:
        """Update draft via Gmail API."""
        import httpx
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        token = state.access_token

        # Build MIME message
        message: Union[MIMEMultipart, MIMEText]
        if html_body:
            message = MIMEMultipart("alternative")
            message.attach(MIMEText(body_text, "plain"))
            message.attach(MIMEText(html_body, "html"))
        else:
            message = MIMEText(body_text, "plain")

        if to:
            message["To"] = ", ".join(to)
        message["Subject"] = subject

        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"https://gmail.googleapis.com/gmail/v1/users/me/drafts/{draft_id}",
                headers={"Authorization": f"Bearer {token}"},
                json={"message": {"raw": raw_message}},
            )
            response.raise_for_status()
            return response.json()

    def _delete_draft(self, state: Any, draft_id: str) -> HandlerResult:
        """Delete a draft."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                loop.run_until_complete(self._api_delete_draft(state, draft_id))
                return json_response({"deleted": draft_id, "success": True})
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"[GmailThreads] Delete draft failed: {e}")
            return error_response(f"Failed to delete draft: {e}", 500)

    async def _api_delete_draft(self, state: Any, draft_id: str) -> None:
        """Delete draft via Gmail API."""
        import httpx

        token = state.access_token

        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"https://gmail.googleapis.com/gmail/v1/users/me/drafts/{draft_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()

    def _send_draft(self, state: Any, draft_id: str) -> HandlerResult:
        """Send a draft."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                result = loop.run_until_complete(self._api_send_draft(state, draft_id))
                return json_response(
                    {
                        "message_id": result.get("id"),
                        "thread_id": result.get("threadId"),
                        "success": True,
                    }
                )
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"[GmailThreads] Send draft failed: {e}")
            return error_response(f"Failed to send draft: {e}", 500)

    async def _api_send_draft(self, state: Any, draft_id: str) -> Dict[str, Any]:
        """Send draft via Gmail API."""
        import httpx

        token = state.access_token

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://gmail.googleapis.com/gmail/v1/users/me/drafts/send",
                headers={"Authorization": f"Bearer {token}"},
                json={"id": draft_id},
            )
            response.raise_for_status()
            return response.json()

    # =========================================================================
    # Attachment Operations
    # =========================================================================

    def _get_attachment(
        self,
        state: Any,
        message_id: str,
        attachment_id: str,
    ) -> HandlerResult:
        """Get attachment data."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                attachment = loop.run_until_complete(
                    self._api_get_attachment(state, message_id, attachment_id)
                )
                return json_response(
                    {
                        "attachment_id": attachment_id,
                        "message_id": message_id,
                        "data": attachment.get("data"),  # Base64 encoded
                        "size": attachment.get("size"),
                    }
                )
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"[GmailThreads] Get attachment failed: {e}")
            return error_response(f"Failed to get attachment: {e}", 500)

    async def _api_get_attachment(
        self,
        state: Any,
        message_id: str,
        attachment_id: str,
    ) -> Dict[str, Any]:
        """Get attachment via Gmail API."""
        import httpx

        token = state.access_token

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}/attachments/{attachment_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()
            return response.json()


# Export for handler registration
__all__ = ["GmailThreadsHandler"]
