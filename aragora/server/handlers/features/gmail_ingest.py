"""
Gmail inbox ingestion handler.

Endpoints:
- POST /api/gmail/connect - Start OAuth flow
- GET /api/gmail/auth/callback - Handle OAuth callback
- POST /api/gmail/sync - Start email sync
- GET /api/gmail/sync/status - Get sync progress
- GET /api/gmail/messages - List indexed emails
- GET /api/gmail/message/{id} - Get single email
- POST /api/gmail/search - Search emails
- DELETE /api/gmail/disconnect - Disconnect account
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
    safe_error_message,
)
from ..utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiter for Gmail endpoints (20 requests per minute - OAuth + sync operations)
_gmail_limiter = RateLimiter(requests_per_minute=20)

# In-memory token storage per user (use Redis/DB in production)
_user_tokens: Dict[str, Dict[str, Any]] = {}
_user_tokens_lock = threading.Lock()

# Sync jobs per user
_sync_jobs: Dict[str, Dict[str, Any]] = {}


@dataclass
class GmailUserState:
    """Per-user Gmail state."""

    user_id: str
    email_address: str = ""
    access_token: str = ""
    refresh_token: str = ""
    token_expiry: Optional[datetime] = None
    history_id: str = ""
    last_sync: Optional[datetime] = None
    indexed_count: int = 0
    total_count: int = 0
    connected_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "user_id": self.user_id,
            "email_address": self.email_address,
            "history_id": self.history_id,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "indexed_count": self.indexed_count,
            "total_count": self.total_count,
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "is_connected": bool(self.refresh_token),
        }


def get_user_state(user_id: str) -> Optional[GmailUserState]:
    """Get Gmail state for a user."""
    with _user_tokens_lock:
        data = _user_tokens.get(user_id)
        if not data:
            return None

        return GmailUserState(
            user_id=user_id,
            email_address=data.get("email_address", ""),
            access_token=data.get("access_token", ""),
            refresh_token=data.get("refresh_token", ""),
            token_expiry=data.get("token_expiry"),
            history_id=data.get("history_id", ""),
            last_sync=data.get("last_sync"),
            indexed_count=data.get("indexed_count", 0),
            total_count=data.get("total_count", 0),
            connected_at=data.get("connected_at"),
        )


def save_user_state(state: GmailUserState) -> None:
    """Save Gmail state for a user."""
    with _user_tokens_lock:
        _user_tokens[state.user_id] = {
            "email_address": state.email_address,
            "access_token": state.access_token,
            "refresh_token": state.refresh_token,
            "token_expiry": state.token_expiry,
            "history_id": state.history_id,
            "last_sync": state.last_sync,
            "indexed_count": state.indexed_count,
            "total_count": state.total_count,
            "connected_at": state.connected_at,
        }


def delete_user_state(user_id: str) -> bool:
    """Delete Gmail state for a user."""
    with _user_tokens_lock:
        if user_id in _user_tokens:
            del _user_tokens[user_id]
            return True
        return False


class GmailIngestHandler(BaseHandler):
    """Handler for Gmail inbox ingestion endpoints."""

    ROUTES = [
        "/api/gmail/connect",
        "/api/gmail/auth/url",
        "/api/gmail/auth/callback",
        "/api/gmail/sync",
        "/api/gmail/sync/status",
        "/api/gmail/messages",
        "/api/gmail/search",
        "/api/gmail/disconnect",
        "/api/gmail/status",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the path."""
        return path.startswith("/api/gmail/")

    def handle(
        self,
        path: str,
        query_params: Dict[str, Any],
        handler: Any,
    ) -> Optional[HandlerResult]:
        """Route GET requests."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _gmail_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for Gmail endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # Extract user_id from auth (simplified - use real auth in production)
        user_id = query_params.get("user_id", "default")

        if path == "/api/gmail/status":
            return self._get_status(user_id)

        if path == "/api/gmail/auth/url":
            return self._get_auth_url(query_params)

        if path == "/api/gmail/auth/callback":
            # Handle OAuth callback GET
            return self._handle_oauth_callback(query_params, user_id)

        if path == "/api/gmail/sync/status":
            return self._get_sync_status(user_id)

        if path == "/api/gmail/messages":
            return self._list_messages(user_id, query_params)

        if path.startswith("/api/gmail/message/"):
            message_id = path.split("/")[-1]
            return self._get_message(user_id, message_id)

        return error_response("Not found", 404)

    def handle_post(
        self,
        path: str,
        body: Dict[str, Any],
        handler: Any,
    ) -> Optional[HandlerResult]:
        """Route POST requests."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _gmail_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for Gmail endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # Extract user_id from body or auth
        user_id = body.get("user_id", "default")

        if path == "/api/gmail/connect":
            return self._start_connect(body, user_id)

        if path == "/api/gmail/auth/callback":
            return self._handle_oauth_callback_post(body, user_id)

        if path == "/api/gmail/sync":
            return self._start_sync(body, user_id)

        if path == "/api/gmail/search":
            return self._search(user_id, body)

        if path == "/api/gmail/disconnect":
            return self._disconnect(user_id)

        return error_response("Not found", 404)

    def _get_status(self, user_id: str) -> HandlerResult:
        """Get connection status for user."""
        state = get_user_state(user_id)

        if not state:
            return json_response({
                "connected": False,
                "configured": self._is_configured(),
            })

        return json_response({
            "connected": bool(state.refresh_token),
            "configured": self._is_configured(),
            "email_address": state.email_address,
            "indexed_count": state.indexed_count,
            "last_sync": state.last_sync.isoformat() if state.last_sync else None,
        })

    def _is_configured(self) -> bool:
        """Check if Gmail OAuth is configured."""
        return bool(
            os.environ.get("GMAIL_CLIENT_ID") or
            os.environ.get("GOOGLE_GMAIL_CLIENT_ID") or
            os.environ.get("GOOGLE_CLIENT_ID")
        )

    def _get_auth_url(self, query_params: Dict[str, Any]) -> HandlerResult:
        """Generate OAuth authorization URL."""
        redirect_uri = query_params.get("redirect_uri", "http://localhost:3000/inbox/callback")
        state = query_params.get("state", "")

        try:
            from aragora.connectors.enterprise.communication.gmail import GmailConnector

            connector = GmailConnector()
            url = connector.get_oauth_url(redirect_uri, state)

            return json_response({"url": url})

        except Exception as e:
            logger.error(f"[Gmail] Failed to generate auth URL: {e}")
            return error_response("Failed to generate authorization URL", 500)

    def _start_connect(self, body: Dict[str, Any], user_id: str) -> HandlerResult:
        """Start OAuth connection flow."""
        redirect_uri = body.get("redirect_uri", "http://localhost:3000/inbox/callback")
        state = body.get("state", user_id)

        try:
            from aragora.connectors.enterprise.communication.gmail import GmailConnector

            connector = GmailConnector()
            url = connector.get_oauth_url(redirect_uri, state)

            return json_response({
                "url": url,
                "state": state,
            })

        except Exception as e:
            logger.error(f"[Gmail] Failed to start connect: {e}")
            return error_response("Failed to start connection", 500)

    def _handle_oauth_callback(
        self,
        query_params: Dict[str, Any],
        user_id: str,
    ) -> HandlerResult:
        """Handle OAuth callback (GET)."""
        code = query_params.get("code")
        state = query_params.get("state", user_id)
        error = query_params.get("error")

        if error:
            return error_response(f"OAuth error: {error}", 400)

        if not code:
            return error_response("Missing authorization code", 400)

        # Use state as user_id if present
        if state:
            user_id = state

        return self._complete_oauth(code, query_params.get("redirect_uri", ""), user_id)

    def _handle_oauth_callback_post(
        self,
        body: Dict[str, Any],
        user_id: str,
    ) -> HandlerResult:
        """Handle OAuth callback (POST)."""
        code = body.get("code")
        redirect_uri = body.get("redirect_uri", "http://localhost:3000/inbox/callback")
        state = body.get("state")

        if not code:
            return error_response("Missing authorization code", 400)

        # Use state as user_id if present
        if state:
            user_id = state

        return self._complete_oauth(code, redirect_uri, user_id)

    def _complete_oauth(
        self,
        code: str,
        redirect_uri: str,
        user_id: str,
    ) -> HandlerResult:
        """Complete OAuth flow and store tokens."""
        try:
            from aragora.connectors.enterprise.communication.gmail import GmailConnector

            connector = GmailConnector()

            # Run authentication
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                success = loop.run_until_complete(
                    connector.authenticate(code=code, redirect_uri=redirect_uri)
                )

                if not success:
                    return error_response("Authentication failed", 401)

                # Get user profile
                profile = loop.run_until_complete(connector.get_user_info())

            finally:
                loop.close()

            # Save state
            state = GmailUserState(
                user_id=user_id,
                email_address=profile.get("emailAddress", ""),
                access_token=connector._access_token or "",
                refresh_token=connector._refresh_token or "",
                token_expiry=connector._token_expiry,
                history_id=str(profile.get("historyId", "")),
                connected_at=datetime.now(timezone.utc),
            )
            save_user_state(state)

            return json_response({
                "success": True,
                "email_address": state.email_address,
                "user_id": user_id,
            })

        except Exception as e:
            logger.error(f"[Gmail] OAuth completion failed: {e}")
            return error_response(safe_error_message(e, "Authentication"), 500)

    def _start_sync(self, body: Dict[str, Any], user_id: str) -> HandlerResult:
        """Start email sync for user."""
        state = get_user_state(user_id)

        if not state or not state.refresh_token:
            return error_response("Not connected - please authenticate first", 401)

        full_sync = body.get("full_sync", False)
        max_messages = body.get("max_messages", 500)
        labels = body.get("labels", ["INBOX"])

        # Check if sync already running
        if user_id in _sync_jobs and _sync_jobs[user_id].get("status") == "running":
            return json_response({
                "message": "Sync already in progress",
                "status": "running",
                "progress": _sync_jobs[user_id].get("progress", 0),
            })

        # Start sync in background
        _sync_jobs[user_id] = {
            "status": "running",
            "progress": 0,
            "messages_synced": 0,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "error": None,
        }

        # Run sync in thread
        import threading
        thread = threading.Thread(
            target=self._run_sync,
            args=(user_id, state, full_sync, max_messages, labels),
            daemon=True,
        )
        thread.start()

        return json_response({
            "message": "Sync started",
            "status": "running",
            "job_id": user_id,
        })

    def _run_sync(
        self,
        user_id: str,
        state: GmailUserState,
        full_sync: bool,
        max_messages: int,
        labels: List[str],
    ) -> None:
        """Run email sync (background thread)."""
        try:
            from aragora.connectors.enterprise.communication.gmail import GmailConnector

            connector = GmailConnector(
                labels=labels,
                max_results=max_messages,
            )

            # Restore tokens
            connector._access_token = state.access_token
            connector._refresh_token = state.refresh_token
            connector._token_expiry = state.token_expiry

            # Run sync
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                result = loop.run_until_complete(
                    connector.sync(full_sync=full_sync, max_items=max_messages)
                )
            finally:
                loop.close()

            # Update user state
            state.indexed_count += result.items_synced
            state.last_sync = datetime.now(timezone.utc)
            state.history_id = result.new_cursor or state.history_id
            save_user_state(state)

            # Update job status
            _sync_jobs[user_id] = {
                "status": "completed",
                "progress": 100,
                "messages_synced": result.items_synced,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": None,
            }

            logger.info(f"[Gmail] Sync completed for {user_id}: {result.items_synced} messages")

        except Exception as e:
            logger.error(f"[Gmail] Sync failed for {user_id}: {e}")
            _sync_jobs[user_id] = {
                "status": "failed",
                "progress": 0,
                "messages_synced": 0,
                "error": str(e),
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }

    def _get_sync_status(self, user_id: str) -> HandlerResult:
        """Get sync status for user."""
        state = get_user_state(user_id)
        job = _sync_jobs.get(user_id, {})

        return json_response({
            "connected": bool(state and state.refresh_token),
            "email_address": state.email_address if state else None,
            "indexed_count": state.indexed_count if state else 0,
            "last_sync": state.last_sync.isoformat() if state and state.last_sync else None,
            "job_status": job.get("status", "idle"),
            "job_progress": job.get("progress", 0),
            "job_messages_synced": job.get("messages_synced", 0),
            "job_error": job.get("error"),
        })

    def _list_messages(
        self,
        user_id: str,
        query_params: Dict[str, Any],
    ) -> HandlerResult:
        """List indexed messages for user."""
        state = get_user_state(user_id)

        if not state or not state.refresh_token:
            return error_response("Not connected", 401)

        limit = int(query_params.get("limit", 50))
        offset = int(query_params.get("offset", 0))
        query = query_params.get("query", "")

        try:
            from aragora.connectors.enterprise.communication.gmail import GmailConnector

            connector = GmailConnector(max_results=limit)
            connector._access_token = state.access_token
            connector._refresh_token = state.refresh_token
            connector._token_expiry = state.token_expiry

            # Fetch messages
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                results = loop.run_until_complete(
                    connector.search(query=query, limit=limit)
                )
            finally:
                loop.close()

            return json_response({
                "messages": [
                    {
                        "id": r.id.replace("gmail-", ""),
                        "subject": r.title,
                        "from": r.author,
                        "snippet": r.content,
                        "date": r.metadata.get("date"),
                        "url": r.url,
                    }
                    for r in results
                ],
                "total": len(results),
                "limit": limit,
                "offset": offset,
            })

        except Exception as e:
            logger.error(f"[Gmail] List messages failed: {e}")
            return error_response(safe_error_message(e, "Failed to list messages"), 500)

    def _get_message(self, user_id: str, message_id: str) -> HandlerResult:
        """Get a single message by ID."""
        state = get_user_state(user_id)

        if not state or not state.refresh_token:
            return error_response("Not connected", 401)

        try:
            from aragora.connectors.enterprise.communication.gmail import GmailConnector

            connector = GmailConnector()
            connector._access_token = state.access_token
            connector._refresh_token = state.refresh_token
            connector._token_expiry = state.token_expiry

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                msg = loop.run_until_complete(connector.get_message(message_id))
            finally:
                loop.close()

            return json_response(msg.to_dict())

        except Exception as e:
            logger.error(f"[Gmail] Get message failed: {e}")
            return error_response(safe_error_message(e, "Failed to get message"), 500)

    def _search(self, user_id: str, body: Dict[str, Any]) -> HandlerResult:
        """Search emails."""
        state = get_user_state(user_id)

        if not state or not state.refresh_token:
            return error_response("Not connected", 401)

        query = body.get("query", "")
        limit = body.get("limit", 20)

        if not query:
            return error_response("Query is required", 400)

        try:
            from aragora.connectors.enterprise.communication.gmail import GmailConnector

            connector = GmailConnector(max_results=limit)
            connector._access_token = state.access_token
            connector._refresh_token = state.refresh_token
            connector._token_expiry = state.token_expiry

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                results = loop.run_until_complete(
                    connector.search(query=query, limit=limit)
                )
            finally:
                loop.close()

            return json_response({
                "query": query,
                "results": [
                    {
                        "id": r.id.replace("gmail-", ""),
                        "subject": r.title,
                        "from": r.author,
                        "snippet": r.content,
                        "date": r.metadata.get("date"),
                        "url": r.url,
                    }
                    for r in results
                ],
                "count": len(results),
            })

        except Exception as e:
            logger.error(f"[Gmail] Search failed: {e}")
            return error_response(safe_error_message(e, "Search"), 500)

    def _disconnect(self, user_id: str) -> HandlerResult:
        """Disconnect Gmail account."""
        deleted = delete_user_state(user_id)

        if user_id in _sync_jobs:
            del _sync_jobs[user_id]

        return json_response({
            "success": True,
            "was_connected": deleted,
        })


# Export for handler registration
__all__ = ["GmailIngestHandler"]
