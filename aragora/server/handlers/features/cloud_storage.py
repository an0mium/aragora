"""
Cloud Storage Handler - API endpoints for cloud storage integration.

Routes:
- GET  /api/cloud/status - Get connection status for all providers
- GET  /api/cloud/{provider}/auth/url - Get OAuth authorization URL
- POST /api/cloud/{provider}/auth/callback - Handle OAuth callback
- GET  /api/cloud/{provider}/files - List files in a folder
- GET  /api/cloud/{provider}/file/{id} - Get file metadata
- POST /api/cloud/{provider}/download - Download file content
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import handler base
try:
    from ..base import (
        BaseHandler,
        HandlerResult,
        error_response,
        json_response,
    )

    HANDLER_BASE_AVAILABLE = True
except ImportError:
    HANDLER_BASE_AVAILABLE = False
    logger.warning(
        "Handler base not available - CloudStorageHandler will have limited functionality"
    )


# Supported providers
PROVIDERS = ["google_drive", "onedrive", "dropbox", "s3"]

# In-memory token storage (use Redis/DB in production)
_tokens: Dict[str, Dict[str, str]] = {}


def get_provider_connector(provider: str):
    """Get connector instance for a provider."""
    if provider == "google_drive":
        from aragora.connectors.enterprise.documents.gdrive import GoogleDriveConnector

        return GoogleDriveConnector()
    elif provider == "onedrive":
        from aragora.connectors.enterprise.documents.onedrive import OneDriveConnector

        return OneDriveConnector()
    elif provider == "dropbox":
        from aragora.connectors.enterprise.documents.dropbox import DropboxConnector

        return DropboxConnector()
    elif provider == "s3":
        from aragora.connectors.enterprise.documents.s3 import S3Connector

        return S3Connector()
    return None


def get_provider_status(provider: str) -> Dict[str, Any]:
    """Get connection status for a provider."""
    connector = get_provider_connector(provider)
    if not connector:
        return {"connected": False, "configured": False}

    is_configured = connector.is_configured()
    has_token = provider in _tokens and "access_token" in _tokens[provider]

    return {
        "connected": is_configured and has_token,
        "configured": is_configured,
        "account_name": _tokens.get(provider, {}).get("account_name"),
    }


def get_all_provider_status() -> Dict[str, Dict[str, Any]]:
    """Get status for all providers."""
    return {p: get_provider_status(p) for p in PROVIDERS}


async def get_auth_url(provider: str, redirect_uri: str, state: str = "") -> Optional[str]:
    """Generate OAuth authorization URL for a provider."""
    connector = get_provider_connector(provider)
    if not connector:
        return None

    if hasattr(connector, "get_oauth_url"):
        return connector.get_oauth_url(redirect_uri, state)

    return None


async def handle_auth_callback(
    provider: str,
    code: str,
    redirect_uri: str,
) -> bool:
    """Handle OAuth callback and store tokens."""
    connector = get_provider_connector(provider)
    if not connector:
        return False

    try:
        success = await connector.authenticate(code=code, redirect_uri=redirect_uri)
        if success:
            # Store tokens
            _tokens[provider] = {
                "access_token": connector._access_token,
                "refresh_token": getattr(connector, "_refresh_token", None),
            }

            # Try to get account name
            if hasattr(connector, "get_user_info"):
                try:
                    user_info = await connector.get_user_info()
                    _tokens[provider]["account_name"] = (
                        user_info.get("displayName")
                        or user_info.get("name")
                        or user_info.get("email", "")
                    )
                except Exception:  # noqa: BLE001 - Optional user info enrichment
                    pass

            return True
    except Exception as e:
        logger.error(f"Auth callback failed for {provider}: {e}")

    return False


async def list_files(
    provider: str,
    path: str = "/",
    page_size: int = 100,
) -> List[Dict[str, Any]]:
    """List files in a folder."""
    connector = get_provider_connector(provider)
    if not connector:
        return []

    # Restore tokens
    if provider in _tokens:
        connector._access_token = _tokens[provider].get("access_token")
        connector._refresh_token = _tokens[provider].get("refresh_token")

    files = []
    try:
        if hasattr(connector, "list_files"):
            async for file in connector.list_files(path):
                files.append(
                    {
                        "id": file.id,
                        "name": file.name,
                        "path": getattr(file, "path", path + "/" + file.name),
                        "size": getattr(file, "size", 0),
                        "mime_type": getattr(file, "mime_type", "application/octet-stream"),
                        "modified_time": str(getattr(file, "modified_time", "")),
                        "is_folder": False,
                        "web_url": getattr(file, "web_url", getattr(file, "web_view_link", "")),
                    }
                )
                if len(files) >= page_size:
                    break

        if hasattr(connector, "list_folders"):
            async for folder in connector.list_folders(path):
                files.append(
                    {
                        "id": folder.id,
                        "name": folder.name,
                        "path": getattr(folder, "path", path + "/" + folder.name),
                        "size": 0,
                        "mime_type": "application/vnd.google-apps.folder",
                        "is_folder": True,
                    }
                )

    except Exception as e:
        logger.error(f"Failed to list files for {provider}: {e}")

    return files


async def download_file(provider: str, file_id: str) -> Optional[bytes]:
    """Download file content."""
    connector = get_provider_connector(provider)
    if not connector:
        return None

    # Restore tokens
    if provider in _tokens:
        connector._access_token = _tokens[provider].get("access_token")
        connector._refresh_token = _tokens[provider].get("refresh_token")

    try:
        if hasattr(connector, "download_file"):
            return await connector.download_file(file_id)
    except Exception as e:
        logger.error(f"Failed to download file from {provider}: {e}")

    return None


# HTTP Handler
if HANDLER_BASE_AVAILABLE:

    class CloudStorageHandler(BaseHandler):
        """HTTP handler for cloud storage operations."""

        ROUTES = [
            "/api/cloud/status",
            "/api/cloud/google_drive/auth/url",
            "/api/cloud/google_drive/auth/callback",
            "/api/cloud/google_drive/files",
            "/api/cloud/onedrive/auth/url",
            "/api/cloud/onedrive/auth/callback",
            "/api/cloud/onedrive/files",
            "/api/cloud/dropbox/auth/url",
            "/api/cloud/dropbox/auth/callback",
            "/api/cloud/dropbox/files",
        ]

        def can_handle(self, path: str, method: str = "GET") -> bool:
            """Check if this handler can process the given path."""
            return path.startswith("/api/cloud/")

        def handle(
            self,
            path: str,
            query_params: Dict[str, Any],
            handler: Any,
        ) -> Optional[HandlerResult]:
            """Route cloud storage requests."""
            if path == "/api/cloud/status":
                return json_response(get_all_provider_status())

            # Parse provider from path: /api/cloud/{provider}/...
            parts = path.split("/")
            if len(parts) < 4:
                return error_response("Invalid path", 400)

            provider = parts[3]
            if provider not in PROVIDERS:
                return error_response(f"Unknown provider: {provider}", 400)

            action = parts[4] if len(parts) > 4 else ""

            if action == "auth":
                sub_action = parts[5] if len(parts) > 5 else ""

                if sub_action == "url":
                    return self._get_auth_url(provider, query_params)

                if sub_action == "callback":
                    # Handle OAuth callback via POST
                    if handler.command != "POST":
                        return error_response("Method not allowed", 405)
                    return None  # Let handle_post handle it

            if action == "files":
                return self._list_files(provider, query_params)

            return error_response("Not found", 404)

        def handle_post(
            self,
            path: str,
            body: Dict[str, Any],
            handler: Any,
        ) -> Optional[HandlerResult]:
            """Handle POST requests."""
            parts = path.split("/")
            if len(parts) < 6:
                return error_response("Invalid path", 400)

            provider = parts[3]
            action = parts[4]
            sub_action = parts[5]

            if action == "auth" and sub_action == "callback":
                return self._handle_auth_callback(provider, body)

            if action == "download":
                return self._download_file(provider, body)

            return error_response("Not found", 404)

        def _get_auth_url(
            self,
            provider: str,
            query_params: Dict[str, Any],
        ) -> HandlerResult:
            """Generate OAuth authorization URL."""
            redirect_uri = query_params.get("redirect_uri", "http://localhost:3000/auth/callback")
            state = query_params.get("state", "")

            import asyncio

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            url = loop.run_until_complete(get_auth_url(provider, redirect_uri, state))

            if not url:
                return error_response("Provider not configured", 400)

            return json_response({"url": url})

        def _handle_auth_callback(
            self,
            provider: str,
            body: Dict[str, Any],
        ) -> HandlerResult:
            """Handle OAuth callback."""
            code = body.get("code")
            redirect_uri = body.get("redirect_uri", "http://localhost:3000/auth/callback")

            if not code:
                return error_response("Missing authorization code", 400)

            import asyncio

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            success = loop.run_until_complete(handle_auth_callback(provider, code, redirect_uri))

            if success:
                return json_response({"success": True})
            return error_response("Authentication failed", 401)

        def _list_files(
            self,
            provider: str,
            query_params: Dict[str, Any],
        ) -> HandlerResult:
            """List files in a folder."""
            path = query_params.get("path", "/")
            page_size = int(query_params.get("limit", 100))

            import asyncio

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            files = loop.run_until_complete(list_files(provider, path, page_size))

            return json_response({"files": files, "path": path})

        def _download_file(
            self,
            provider: str,
            body: Dict[str, Any],
        ) -> HandlerResult:
            """Download a file."""
            file_id = body.get("file_id")
            if not file_id:
                return error_response("Missing file_id", 400)

            import asyncio

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            content = loop.run_until_complete(download_file(provider, file_id))

            if content is None:
                return error_response("Download failed", 500)

            import base64

            return json_response(
                {
                    "content": base64.b64encode(content).decode(),
                    "size": len(content),
                }
            )
