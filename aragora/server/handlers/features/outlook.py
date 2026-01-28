"""
Outlook/Microsoft 365 Email Handler.

Provides REST APIs for Outlook/M365 email integration:
- OAuth2 authentication flow
- Message and folder management
- Thread/conversation retrieval
- Send and reply
- Search with OData syntax

Endpoints:
- GET  /api/v1/outlook/oauth/url          - Get OAuth authorization URL
- POST /api/v1/outlook/oauth/callback     - Handle OAuth callback
- GET  /api/v1/outlook/folders            - List mail folders
- GET  /api/v1/outlook/messages           - List messages
- GET  /api/v1/outlook/messages/{id}      - Get message details
- GET  /api/v1/outlook/conversations/{id} - Get conversation thread
- POST /api/v1/outlook/send               - Send new message
- POST /api/v1/outlook/reply              - Reply to message
- GET  /api/v1/outlook/search             - Search messages
- POST /api/v1/outlook/messages/{id}/read - Mark as read/unread
- POST /api/v1/outlook/messages/{id}/move - Move message
- DELETE /api/v1/outlook/messages/{id}    - Delete message
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    success_response,
)
from aragora.server.handlers.utils.decorators import require_permission

logger = logging.getLogger(__name__)


# =============================================================================
# In-Memory Storage (replace with database in production)
# =============================================================================

# Store connector instances per user/workspace
_outlook_connectors: Dict[str, Any] = {}  # key: f"{workspace_id}:{user_id}"
_oauth_states: Dict[str, Dict[str, Any]] = {}  # state -> {workspace_id, user_id, redirect_uri}
_storage_lock = threading.Lock()


# =============================================================================
# Handler Functions
# =============================================================================


def _get_connector_key(workspace_id: str, user_id: str) -> str:
    """Generate unique key for connector storage."""
    return f"{workspace_id}:{user_id}"


async def _get_or_create_connector(
    workspace_id: str,
    user_id: str,
    folders: Optional[List[str]] = None,
) -> Any:
    """Get existing connector or create new one."""
    key = _get_connector_key(workspace_id, user_id)

    with _storage_lock:
        if key in _outlook_connectors:
            return _outlook_connectors[key]

    # Create new connector
    try:
        from aragora.connectors.enterprise.communication.outlook import OutlookConnector

        connector = OutlookConnector(
            folders=folders,
            exclude_folders=["Junk Email", "Deleted Items"],
            max_results=100,
            user_id="me",
        )

        with _storage_lock:
            _outlook_connectors[key] = connector

        return connector

    except ImportError:
        logger.error("OutlookConnector not available")
        return None


async def handle_get_oauth_url(
    workspace_id: str,
    user_id: str,
    redirect_uri: str,
) -> Dict[str, Any]:
    """
    Get OAuth authorization URL for Outlook.

    GET /api/v1/outlook/oauth/url?redirect_uri=...
    """
    try:
        connector = await _get_or_create_connector(workspace_id, user_id)
        if not connector:
            return {"success": False, "error": "Outlook connector not available"}

        if not connector.is_configured():
            return {
                "success": False,
                "error": "Outlook not configured. Set OUTLOOK_CLIENT_ID and OUTLOOK_CLIENT_SECRET.",
            }

        # Generate state for CSRF protection
        state = str(uuid4())

        with _storage_lock:
            _oauth_states[state] = {
                "workspace_id": workspace_id,
                "user_id": user_id,
                "redirect_uri": redirect_uri,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

        url = connector.get_oauth_url(redirect_uri=redirect_uri, state=state)

        return {
            "success": True,
            "auth_url": url,
            "state": state,
        }

    except Exception as e:
        logger.exception(f"Failed to get OAuth URL: {e}")
        return {"success": False, "error": str(e)}


async def handle_oauth_callback(
    code: str,
    state: str,
    redirect_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Handle OAuth callback.

    POST /api/v1/outlook/oauth/callback
    {
        "code": "authorization_code",
        "state": "state_from_auth_url"
    }
    """
    try:
        # Validate state
        with _storage_lock:
            state_data = _oauth_states.pop(state, None)

        if not state_data:
            return {"success": False, "error": "Invalid or expired state"}

        workspace_id = state_data["workspace_id"]
        user_id = state_data["user_id"]
        stored_redirect_uri = state_data["redirect_uri"]

        # Use stored redirect_uri if not provided
        actual_redirect_uri = redirect_uri or stored_redirect_uri

        connector = await _get_or_create_connector(workspace_id, user_id)
        if not connector:
            return {"success": False, "error": "Outlook connector not available"}

        # Exchange code for tokens
        success = await connector.authenticate(
            code=code,
            redirect_uri=actual_redirect_uri,
        )

        if success:
            # Get user profile
            try:
                profile = await connector.get_user_info()
                email = profile.get("mail") or profile.get("userPrincipalName", "")
            except Exception as e:
                logger.warning(f"Could not get user profile: {e}")
                email = ""

            return {
                "success": True,
                "email": email,
                "workspace_id": workspace_id,
                "user_id": user_id,
            }
        else:
            return {"success": False, "error": "Authentication failed"}

    except Exception as e:
        logger.exception(f"OAuth callback failed: {e}")
        return {"success": False, "error": str(e)}


@require_permission("connectors:read")
async def handle_list_folders(
    workspace_id: str,
    user_id: str,
) -> Dict[str, Any]:
    """
    List mail folders.

    GET /api/v1/outlook/folders
    """
    try:
        connector = await _get_or_create_connector(workspace_id, user_id)
        if not connector:
            return {"success": False, "error": "Outlook connector not available"}

        folders = await connector.list_folders()

        return {
            "success": True,
            "folders": [
                {
                    "id": f.id,
                    "display_name": f.display_name,
                    "unread_count": f.unread_item_count,
                    "total_count": f.total_item_count,
                    "child_folder_count": f.child_folder_count,
                    "is_hidden": f.is_hidden,
                }
                for f in folders
            ],
            "total": len(folders),
        }

    except Exception as e:
        logger.exception(f"Failed to list folders: {e}")
        return {"success": False, "error": str(e)}


@require_permission("connectors:read")
async def handle_list_messages(
    workspace_id: str,
    user_id: str,
    folder_id: Optional[str] = None,
    max_results: int = 50,
    page_token: Optional[str] = None,
    filter_query: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List messages.

    GET /api/v1/outlook/messages?folder_id=...&max_results=50
    """
    try:
        connector = await _get_or_create_connector(workspace_id, user_id)
        if not connector:
            return {"success": False, "error": "Outlook connector not available"}

        message_ids, next_page = await connector.list_messages(
            folder_id=folder_id,
            max_results=max_results,
            page_token=page_token,
            query=filter_query,
        )

        # Fetch message details in parallel (limit concurrency)
        messages = []
        semaphore = asyncio.Semaphore(5)

        async def fetch_message(msg_id: str):
            async with semaphore:
                try:
                    msg = await connector.get_message(msg_id, include_body=False)
                    return (
                        msg.to_dict()
                        if hasattr(msg, "to_dict")
                        else {
                            "id": msg.id,
                            "thread_id": msg.thread_id,
                            "subject": msg.subject,
                            "from_address": msg.from_address,
                            "to_addresses": msg.to_addresses,
                            "date": msg.date.isoformat() if msg.date else None,
                            "snippet": msg.snippet,
                            "is_read": msg.is_read,
                            "is_starred": msg.is_starred,
                            "is_important": msg.is_important,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to fetch message {msg_id}: {e}")
                    return None

        results = await asyncio.gather(*[fetch_message(mid) for mid in message_ids])
        messages = [m for m in results if m is not None]

        return {
            "success": True,
            "messages": messages,
            "total": len(messages),
            "next_page_token": next_page,
        }

    except Exception as e:
        logger.exception(f"Failed to list messages: {e}")
        return {"success": False, "error": str(e)}


async def handle_get_message(
    workspace_id: str,
    user_id: str,
    message_id: str,
    include_attachments: bool = False,
) -> Dict[str, Any]:
    """
    Get message details.

    GET /api/v1/outlook/messages/{message_id}?include_attachments=true
    """
    try:
        connector = await _get_or_create_connector(workspace_id, user_id)
        if not connector:
            return {"success": False, "error": "Outlook connector not available"}

        message = await connector.get_message(message_id, include_body=True)

        result = {
            "id": message.id,
            "thread_id": message.thread_id,
            "subject": message.subject,
            "from_address": message.from_address,
            "to_addresses": message.to_addresses,
            "cc_addresses": message.cc_addresses,
            "bcc_addresses": message.bcc_addresses,
            "date": message.date.isoformat() if message.date else None,
            "body_text": message.body_text,
            "body_html": message.body_html,
            "snippet": message.snippet,
            "is_read": message.is_read,
            "is_starred": message.is_starred,
            "is_important": message.is_important,
            "labels": message.labels,
        }

        if include_attachments:
            attachments = await connector.get_message_attachments(message_id)
            result["attachments"] = [
                {
                    "id": a.id,
                    "filename": a.filename,
                    "mime_type": a.mime_type,
                    "size": a.size,
                }
                for a in attachments
            ]

        return {"success": True, "message": result}

    except Exception as e:
        logger.exception(f"Failed to get message: {e}")
        return {"success": False, "error": str(e)}


async def handle_get_conversation(
    workspace_id: str,
    user_id: str,
    conversation_id: str,
    max_messages: int = 50,
) -> Dict[str, Any]:
    """
    Get conversation thread.

    GET /api/v1/outlook/conversations/{conversation_id}
    """
    try:
        connector = await _get_or_create_connector(workspace_id, user_id)
        if not connector:
            return {"success": False, "error": "Outlook connector not available"}

        thread = await connector.get_conversation(conversation_id, max_messages)

        return {
            "success": True,
            "conversation": {
                "id": thread.id,
                "subject": thread.subject,
                "message_count": thread.message_count,
                "participants": thread.participants,
                "last_message_date": (
                    thread.last_message_date.isoformat() if thread.last_message_date else None
                ),
                "snippet": thread.snippet,
                "messages": [
                    {
                        "id": m.id,
                        "subject": m.subject,
                        "from_address": m.from_address,
                        "to_addresses": m.to_addresses,
                        "date": m.date.isoformat() if m.date else None,
                        "body_text": m.body_text,
                        "snippet": m.snippet,
                        "is_read": m.is_read,
                    }
                    for m in thread.messages
                ],
            },
        }

    except Exception as e:
        logger.exception(f"Failed to get conversation: {e}")
        return {"success": False, "error": str(e)}


@require_permission("connectors:write")
async def handle_send_message(
    workspace_id: str,
    user_id: str,
    to_addresses: List[str],
    subject: str,
    body: str,
    body_type: str = "text",
    cc_addresses: Optional[List[str]] = None,
    bcc_addresses: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Send a new email.

    POST /api/v1/outlook/send
    {
        "to": ["recipient@example.com"],
        "subject": "Subject",
        "body": "Message body",
        "body_type": "html",
        "cc": ["cc@example.com"]
    }
    """
    try:
        connector = await _get_or_create_connector(workspace_id, user_id)
        if not connector:
            return {"success": False, "error": "Outlook connector not available"}

        html_body = body if body_type == "html" else None
        plain_body = body if body_type != "html" else None

        result = await connector.send_message(
            to=to_addresses,
            subject=subject,
            body=plain_body or "",
            cc=cc_addresses,
            bcc=bcc_addresses,
            html_body=html_body,
        )

        return {
            "success": result.get("success", True),
            "message": "Email sent successfully",
        }

    except Exception as e:
        logger.exception(f"Failed to send message: {e}")
        return {"success": False, "error": str(e)}


async def handle_reply_message(
    workspace_id: str,
    user_id: str,
    message_id: str,
    body: str,
    body_type: str = "text",
    reply_all: bool = False,
    cc_addresses: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Reply to a message.

    POST /api/v1/outlook/reply
    {
        "message_id": "...",
        "body": "Reply body",
        "reply_all": false
    }
    """
    try:
        connector = await _get_or_create_connector(workspace_id, user_id)
        if not connector:
            return {"success": False, "error": "Outlook connector not available"}

        html_body = body if body_type == "html" else None

        result = await connector.reply_to_message(
            original_message_id=message_id,
            body=body,
            cc=cc_addresses,
            html_body=html_body,
            reply_all=reply_all,
        )

        return {
            "success": result.get("success", True),
            "message": "Reply sent successfully",
            "in_reply_to": message_id,
        }

    except Exception as e:
        logger.exception(f"Failed to reply to message: {e}")
        return {"success": False, "error": str(e)}


async def handle_search_messages(
    workspace_id: str,
    user_id: str,
    query: str,
    max_results: int = 25,
    folder_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search messages.

    GET /api/v1/outlook/search?q=...&max_results=25
    """
    try:
        connector = await _get_or_create_connector(workspace_id, user_id)
        if not connector:
            return {"success": False, "error": "Outlook connector not available"}

        results = await connector.search(query=query, limit=max_results)

        return {
            "success": True,
            "query": query,
            "results": [
                {
                    "id": r.source_id if hasattr(r, "source_id") else r.id,
                    "title": r.title,
                    "snippet": r.content[:200] if r.content else "",
                    "author": r.author,
                    "url": r.url,
                }
                for r in results
            ],
            "total": len(results),
        }

    except Exception as e:
        logger.exception(f"Failed to search messages: {e}")
        return {"success": False, "error": str(e)}


async def handle_mark_read(
    workspace_id: str,
    user_id: str,
    message_id: str,
    is_read: bool = True,
) -> Dict[str, Any]:
    """
    Mark message as read or unread.

    POST /api/v1/outlook/messages/{message_id}/read
    {
        "is_read": true
    }
    """
    try:
        key = _get_connector_key(workspace_id, user_id)

        with _storage_lock:
            connector = _outlook_connectors.get(key)

        if not connector:
            return {"success": False, "error": "Not authenticated with Outlook"}

        # Use the Graph API directly
        endpoint = f"/messages/{message_id}"
        await connector._api_request(
            endpoint,
            method="PATCH",
            json_data={"isRead": is_read},
        )

        return {
            "success": True,
            "message_id": message_id,
            "is_read": is_read,
        }

    except Exception as e:
        logger.exception(f"Failed to mark message: {e}")
        return {"success": False, "error": str(e)}


async def handle_move_message(
    workspace_id: str,
    user_id: str,
    message_id: str,
    destination_folder_id: str,
) -> Dict[str, Any]:
    """
    Move message to a different folder.

    POST /api/v1/outlook/messages/{message_id}/move
    {
        "destination_folder_id": "..."
    }
    """
    try:
        key = _get_connector_key(workspace_id, user_id)

        with _storage_lock:
            connector = _outlook_connectors.get(key)

        if not connector:
            return {"success": False, "error": "Not authenticated with Outlook"}

        # Use the Graph API directly
        endpoint = f"/messages/{message_id}/move"
        await connector._api_request(
            endpoint,
            method="POST",
            json_data={"destinationId": destination_folder_id},
        )

        return {
            "success": True,
            "message_id": message_id,
            "destination_folder_id": destination_folder_id,
        }

    except Exception as e:
        logger.exception(f"Failed to move message: {e}")
        return {"success": False, "error": str(e)}


@require_permission("connectors:delete")
async def handle_delete_message(
    workspace_id: str,
    user_id: str,
    message_id: str,
    permanent: bool = False,
) -> Dict[str, Any]:
    """
    Delete a message.

    DELETE /api/v1/outlook/messages/{message_id}?permanent=false
    """
    try:
        key = _get_connector_key(workspace_id, user_id)

        with _storage_lock:
            connector = _outlook_connectors.get(key)

        if not connector:
            return {"success": False, "error": "Not authenticated with Outlook"}

        if permanent:
            # Permanently delete
            endpoint = f"/messages/{message_id}"
            await connector._api_request(endpoint, method="DELETE")
        else:
            # Move to Deleted Items
            folders = await connector.list_folders()
            deleted_folder = next((f for f in folders if f.display_name == "Deleted Items"), None)
            if deleted_folder:
                endpoint = f"/messages/{message_id}/move"
                await connector._api_request(
                    endpoint,
                    method="POST",
                    json_data={"destinationId": deleted_folder.id},
                )
            else:
                # Fall back to permanent delete
                endpoint = f"/messages/{message_id}"
                await connector._api_request(endpoint, method="DELETE")

        return {
            "success": True,
            "message_id": message_id,
            "deleted": True,
            "permanent": permanent,
        }

    except Exception as e:
        logger.exception(f"Failed to delete message: {e}")
        return {"success": False, "error": str(e)}


async def handle_get_status(
    workspace_id: str,
    user_id: str,
) -> Dict[str, Any]:
    """
    Get Outlook connection status.

    GET /api/v1/outlook/status
    """
    try:
        key = _get_connector_key(workspace_id, user_id)

        with _storage_lock:
            connector = _outlook_connectors.get(key)

        if not connector:
            return {
                "success": True,
                "connected": False,
                "email": None,
            }

        # Check if we have valid tokens
        try:
            profile = await connector.get_user_info()
            email = profile.get("mail") or profile.get("userPrincipalName", "")

            return {
                "success": True,
                "connected": True,
                "email": email,
                "display_name": profile.get("displayName"),
            }
        except (ConnectionError, TimeoutError, PermissionError, ValueError):
            return {
                "success": True,
                "connected": False,
                "email": None,
                "error": "Token expired or invalid",
            }

    except Exception as e:
        logger.exception(f"Failed to get status: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# Handler Class
# =============================================================================


class OutlookHandler(BaseHandler):
    """
    HTTP handler for Outlook email endpoints.

    Integrates with the Aragora server routing system.
    """

    ROUTES = [
        "/api/v1/outlook/oauth/url",
        "/api/v1/outlook/oauth/callback",
        "/api/v1/outlook/folders",
        "/api/v1/outlook/messages",
        "/api/v1/outlook/send",
        "/api/v1/outlook/reply",
        "/api/v1/outlook/search",
        "/api/v1/outlook/status",
    ]

    ROUTE_PREFIXES = [
        "/api/v1/outlook/messages/",
        "/api/v1/outlook/conversations/",
    ]

    def __init__(self, ctx: Dict[str, Any]):
        """Initialize with server context."""
        super().__init__(ctx)  # type: ignore[arg-type]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        if path in self.ROUTES:
            return True
        for prefix in self.ROUTE_PREFIXES:
            if path.startswith(prefix):
                return True
        return False

    def handle(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route Outlook endpoint requests."""
        return None

    async def handle_get_oauth_url(self, params: Dict[str, Any]) -> HandlerResult:
        """GET /api/v1/outlook/oauth/url"""
        redirect_uri = params.get("redirect_uri")
        if not redirect_uri:
            return error_response("redirect_uri required", 400)

        result = await handle_get_oauth_url(
            workspace_id=params.get("workspace_id", "default"),
            user_id=self._get_user_id(),
            redirect_uri=redirect_uri,
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_oauth_callback(self, data: Dict[str, Any]) -> HandlerResult:
        """POST /api/v1/outlook/oauth/callback"""
        code = data.get("code")
        state = data.get("state")

        if not code or not state:
            return error_response("code and state required", 400)

        result = await handle_oauth_callback(
            code=code,
            state=state,
            redirect_uri=data.get("redirect_uri"),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_get_folders(self, params: Dict[str, Any]) -> HandlerResult:
        """GET /api/v1/outlook/folders"""
        result = await handle_list_folders(
            workspace_id=params.get("workspace_id", "default"),
            user_id=self._get_user_id(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_get_messages(self, params: Dict[str, Any]) -> HandlerResult:
        """GET /api/v1/outlook/messages"""
        result = await handle_list_messages(
            workspace_id=params.get("workspace_id", "default"),
            user_id=self._get_user_id(),
            folder_id=params.get("folder_id"),
            max_results=int(params.get("max_results", 50)),
            page_token=params.get("page_token"),
            filter_query=params.get("filter"),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_get_message(self, params: Dict[str, Any], message_id: str) -> HandlerResult:
        """GET /api/v1/outlook/messages/{message_id}"""
        result = await handle_get_message(
            workspace_id=params.get("workspace_id", "default"),
            user_id=self._get_user_id(),
            message_id=message_id,
            include_attachments=params.get("include_attachments", "false") == "true",
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 404)

    async def handle_get_conversation(
        self, params: Dict[str, Any], conversation_id: str
    ) -> HandlerResult:
        """GET /api/v1/outlook/conversations/{conversation_id}"""
        result = await handle_get_conversation(
            workspace_id=params.get("workspace_id", "default"),
            user_id=self._get_user_id(),
            conversation_id=conversation_id,
            max_messages=int(params.get("max_messages", 50)),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 404)

    async def handle_post_send(self, data: Dict[str, Any]) -> HandlerResult:
        """POST /api/v1/outlook/send"""
        to = data.get("to")
        subject = data.get("subject")
        body = data.get("body")

        if not to or not subject or not body:
            return error_response("to, subject, and body required", 400)

        result = await handle_send_message(
            workspace_id=data.get("workspace_id", "default"),
            user_id=self._get_user_id(),
            to_addresses=to if isinstance(to, list) else [to],
            subject=subject,
            body=body,
            body_type=data.get("body_type", "text"),
            cc_addresses=data.get("cc"),
            bcc_addresses=data.get("bcc"),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_reply(self, data: Dict[str, Any]) -> HandlerResult:
        """POST /api/v1/outlook/reply"""
        message_id = data.get("message_id")
        body = data.get("body")

        if not message_id or not body:
            return error_response("message_id and body required", 400)

        result = await handle_reply_message(
            workspace_id=data.get("workspace_id", "default"),
            user_id=self._get_user_id(),
            message_id=message_id,
            body=body,
            body_type=data.get("body_type", "text"),
            reply_all=data.get("reply_all", False),
            cc_addresses=data.get("cc"),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_get_search(self, params: Dict[str, Any]) -> HandlerResult:
        """GET /api/v1/outlook/search"""
        query = params.get("q")
        if not query:
            return error_response("q (query) required", 400)

        result = await handle_search_messages(
            workspace_id=params.get("workspace_id", "default"),
            user_id=self._get_user_id(),
            query=query,
            max_results=int(params.get("max_results", 25)),
            folder_id=params.get("folder_id"),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_get_status(self, params: Dict[str, Any]) -> HandlerResult:
        """GET /api/v1/outlook/status"""
        result = await handle_get_status(
            workspace_id=params.get("workspace_id", "default"),
            user_id=self._get_user_id(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_mark_read(self, data: Dict[str, Any], message_id: str) -> HandlerResult:
        """POST /api/v1/outlook/messages/{message_id}/read"""
        result = await handle_mark_read(
            workspace_id=data.get("workspace_id", "default"),
            user_id=self._get_user_id(),
            message_id=message_id,
            is_read=data.get("is_read", True),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_move(self, data: Dict[str, Any], message_id: str) -> HandlerResult:
        """POST /api/v1/outlook/messages/{message_id}/move"""
        destination = data.get("destination_folder_id")
        if not destination:
            return error_response("destination_folder_id required", 400)

        result = await handle_move_message(
            workspace_id=data.get("workspace_id", "default"),
            user_id=self._get_user_id(),
            message_id=message_id,
            destination_folder_id=destination,
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_delete_message(self, params: Dict[str, Any], message_id: str) -> HandlerResult:
        """DELETE /api/v1/outlook/messages/{message_id}"""
        result = await handle_delete_message(
            workspace_id=params.get("workspace_id", "default"),
            user_id=self._get_user_id(),
            message_id=message_id,
            permanent=params.get("permanent", "false") == "true",
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    def _get_user_id(self) -> str:
        """Get user ID from auth context."""
        auth_ctx = self.ctx.get("auth_context")
        if auth_ctx and hasattr(auth_ctx, "user_id"):
            return auth_ctx.user_id
        return "default"


__all__ = [
    "OutlookHandler",
    "handle_get_oauth_url",
    "handle_oauth_callback",
    "handle_list_folders",
    "handle_list_messages",
    "handle_get_message",
    "handle_get_conversation",
    "handle_send_message",
    "handle_reply_message",
    "handle_search_messages",
    "handle_mark_read",
    "handle_move_message",
    "handle_delete_message",
    "handle_get_status",
]
