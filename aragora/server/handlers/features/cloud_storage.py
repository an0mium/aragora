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

import hashlib
import logging
import os
import re
from typing import Any, cast

from aragora.server.validation.query_params import safe_query_int


logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependencies
try:
    from ..utils.rate_limit import rate_limit
except ImportError:
    # Fallback: no-op decorator if rate_limit not available
    def rate_limit(**kwargs):  # type: ignore[misc]
        def decorator(fn):  # type: ignore[no-untyped-def]
            return fn

        return decorator


# Try to import handler base
try:
    from ..base import (
        HandlerResult,
        error_response,
        json_response,
    handle_errors,
    )
    from ..secure import ForbiddenError, SecureHandler, UnauthorizedError

    HANDLER_BASE_AVAILABLE = True
except ImportError:
    HANDLER_BASE_AVAILABLE = False
    logger.warning(
        "Handler base not available - CloudStorageHandler will have limited functionality"
    )

# Permission constants for cloud storage operations
CLOUD_READ_PERMISSION = "cloud:read"
CLOUD_WRITE_PERMISSION = "cloud:write"

# Supported providers
PROVIDERS = ["google_drive", "onedrive", "dropbox", "s3"]

# --- Security constants ---

# Maximum download file size (100MB default, configurable via env)
MAX_DOWNLOAD_SIZE_BYTES = int(os.environ.get("ARAGORA_MAX_DOWNLOAD_SIZE", 100 * 1024 * 1024))

# Safe filename pattern - alphanumeric, hyphens, underscores, dots, spaces
SAFE_FILENAME_PATTERN = re.compile(r"^[\w\-. ]+$")

# Safe file_id pattern - provider IDs are typically alphanumeric with hyphens/underscores
SAFE_FILE_ID_PATTERN = re.compile(r"^[\w\-.:=+/]{1,512}$")

# Maximum length for OAuth authorization codes
MAX_AUTH_CODE_LENGTH = 4096

# Maximum length for redirect URIs
MAX_REDIRECT_URI_LENGTH = 2048

# Path traversal sequences that must be rejected
_PATH_TRAVERSAL_SEQUENCES = ("..", "~", "\x00")

# In-memory token storage (use Redis/DB in production)
_tokens: dict[str, dict[str, str]] = {}


def _validate_path(path: str) -> str | None:
    """Validate a file listing path to prevent path traversal.

    Returns the sanitised path, or None if the path is invalid.
    """
    if not path:
        return "/"

    # Reject null bytes and traversal sequences
    for seq in _PATH_TRAVERSAL_SEQUENCES:
        if seq in path:
            return None

    # Normalise repeated slashes
    path = re.sub(r"/+", "/", path)

    return path


def _validate_file_id(file_id: str) -> bool:
    """Validate a file ID to prevent injection attacks."""
    if not file_id or not isinstance(file_id, str):
        return False
    if len(file_id) > 512:
        return False
    return bool(SAFE_FILE_ID_PATTERN.match(file_id))


def get_provider_connector(provider: str) -> Any:
    """Get connector instance for a provider.

    Returns a connector instance for the specified cloud storage provider.
    The connectors are enterprise document connectors that provide OAuth authentication,
    file listing, and download capabilities.

    Note: The OneDrive and Dropbox connectors extend EnterpriseConnector but don't
    implement all abstract methods from BaseConnector (search/fetch). They are designed
    for sync operations rather than search. Similarly, S3Connector requires a bucket
    parameter for full functionality but can be instantiated with an empty bucket for
    configuration checking purposes.

    Returns:
        A connector instance with is_configured property, get_oauth_url method,
        authenticate method, list_files/list_folders async generators,
        download_file method, and _access_token/_refresh_token attributes.
        Returns None if provider is not supported.
    """
    if provider == "google_drive":
        from aragora.connectors.enterprise.documents.gdrive import GoogleDriveConnector

        # GoogleDriveConnector is a fully concrete implementation
        return GoogleDriveConnector()

    if provider == "onedrive":
        from aragora.connectors.enterprise.documents.onedrive import OneDriveConnector

        # OneDriveConnector doesn't implement search/fetch/name from BaseConnector,
        # but those methods aren't used in this handler. We cast to Any to allow
        # instantiation since the abstract methods aren't needed for our use case.
        return cast(Any, OneDriveConnector)()

    if provider == "dropbox":
        from aragora.connectors.enterprise.documents.dropbox import DropboxConnector

        # DropboxConnector doesn't implement search/fetch/name from BaseConnector,
        # but those methods aren't used in this handler. We cast to Any to allow
        # instantiation since the abstract methods aren't needed for our use case.
        return cast(Any, DropboxConnector)()

    if provider == "s3":
        from aragora.connectors.enterprise.documents.s3 import S3Connector

        # S3Connector requires bucket parameter - use empty string for config checking.
        # S3 doesn't support OAuth so this connector has limited use in this handler.
        return S3Connector(bucket="")

    return None


def get_provider_status(provider: str) -> dict[str, Any]:
    """Get connection status for a provider."""
    connector = get_provider_connector(provider)
    if not connector:
        return {"connected": False, "configured": False}

    # is_configured may be a property or a method depending on connector
    is_configured_attr = connector.is_configured
    is_configured: bool = (
        is_configured_attr() if callable(is_configured_attr) else bool(is_configured_attr)
    )
    has_token = provider in _tokens and "access_token" in _tokens[provider]

    return {
        "connected": is_configured and has_token,
        "configured": is_configured,
        "account_name": _tokens.get(provider, {}).get("account_name"),
    }


def get_all_provider_status() -> dict[str, dict[str, Any]]:
    """Get status for all providers."""
    return {p: get_provider_status(p) for p in PROVIDERS}


async def get_auth_url(provider: str, redirect_uri: str, state: str = "") -> str | None:
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
                except (KeyError, AttributeError, TypeError) as e:
                    # Optional user info enrichment - non-critical
                    logger.debug("Could not get user info for %s: %s", provider, e)

            return True
    except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError, KeyError) as e:
        logger.error("Auth callback failed for %s: %s", provider, e)

    return False


async def list_files(
    provider: str,
    path: str = "/",
    page_size: int = 100,
) -> list[dict[str, Any]]:
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

    except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError, AttributeError) as e:
        logger.error("Failed to list files for %s: %s", provider, e)

    return files


async def download_file(provider: str, file_id: str) -> bytes | None:
    """Download file content.

    Enforces MAX_DOWNLOAD_SIZE_BYTES to prevent memory exhaustion.
    """
    connector = get_provider_connector(provider)
    if not connector:
        return None

    # Restore tokens
    if provider in _tokens:
        connector._access_token = _tokens[provider].get("access_token")
        connector._refresh_token = _tokens[provider].get("refresh_token")

    try:
        if hasattr(connector, "download_file"):
            content = await connector.download_file(file_id)
            # Enforce download size limit
            if content is not None and len(content) > MAX_DOWNLOAD_SIZE_BYTES:
                logger.warning(
                    "Downloaded file exceeds size limit: %d > %d bytes",
                    len(content),
                    MAX_DOWNLOAD_SIZE_BYTES,
                )
                return None
            return content
    except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
        logger.error("Failed to download file from %s: %s", provider, e)

    return None


# HTTP Handler
if HANDLER_BASE_AVAILABLE:

    class CloudStorageHandler(SecureHandler):
        """HTTP handler for cloud storage operations."""

        ROUTES = [
            "/api/v1/cloud/status",
            "/api/v1/cloud/google_drive/auth/url",
            "/api/v1/cloud/google_drive/auth/callback",
            "/api/v1/cloud/google_drive/files",
            "/api/v1/cloud/onedrive/auth/url",
            "/api/v1/cloud/onedrive/auth/callback",
            "/api/v1/cloud/onedrive/files",
            "/api/v1/cloud/dropbox/auth/url",
            "/api/v1/cloud/dropbox/auth/callback",
            "/api/v1/cloud/dropbox/files",
        ]

        def can_handle(self, path: str, method: str = "GET") -> bool:
            """Check if this handler can process the given path."""
            return path.startswith("/api/v1/cloud/")

        @rate_limit(requests_per_minute=60)
        async def handle(
            self,
            path: str,
            query_params: dict[str, Any],
            handler: Any,
        ) -> HandlerResult | None:
            """Route cloud storage requests."""
            # RBAC: Require authentication and cloud:read permission
            try:
                auth_context = await self.get_auth_context(handler, require_auth=True)
                self.check_permission(auth_context, CLOUD_READ_PERMISSION)
            except UnauthorizedError:
                return error_response("Authentication required", 401)
            except ForbiddenError as e:
                logger.warning("Handler error: %s", e)
                return error_response("Permission denied", 403)

            if path == "/api/v1/cloud/status":
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
                    return await self._get_auth_url(provider, query_params)

                if sub_action == "callback":
                    # Handle OAuth callback via POST
                    if handler.command != "POST":
                        return error_response("Method not allowed", 405)
                    return None  # Let handle_post handle it

            if action == "files":
                return await self._list_files(provider, query_params)

            return error_response("Not found", 404)

        @handle_errors("cloud storage creation")
        @rate_limit(requests_per_minute=30)
        async def handle_post(
            self,
            path: str,
            query_params: dict[str, Any],
            handler: Any,
        ) -> HandlerResult | None:
            """Handle POST requests."""
            # RBAC: Require authentication and cloud:write permission
            try:
                auth_context = await self.get_auth_context(handler, require_auth=True)
                self.check_permission(auth_context, CLOUD_WRITE_PERMISSION)
            except UnauthorizedError:
                return error_response("Authentication required", 401)
            except ForbiddenError as e:
                logger.warning("Handler error: %s", e)
                return error_response("Permission denied", 403)

            body = self.read_json_body(handler)
            if body is None:
                return error_response("Invalid JSON body", 400)

            parts = path.split("/")
            if len(parts) < 6:
                return error_response("Invalid path", 400)

            provider = parts[3]
            if provider not in PROVIDERS:
                return error_response(f"Unknown provider: {provider}", 400)

            action = parts[4]
            sub_action = parts[5]

            if action == "auth" and sub_action == "callback":
                return await self._handle_auth_callback(provider, body)

            if action == "download":
                return await self._download_file(provider, body)

            return error_response("Not found", 404)

        async def _get_auth_url(
            self,
            provider: str,
            query_params: dict[str, Any],
        ) -> HandlerResult:
            """Generate OAuth authorization URL."""
            redirect_uri = query_params.get("redirect_uri", "http://localhost:3000/auth/callback")
            state = query_params.get("state", "")

            # Validate redirect_uri length
            if len(redirect_uri) > MAX_REDIRECT_URI_LENGTH:
                return error_response("redirect_uri too long", 400)

            # Validate redirect_uri scheme
            if not redirect_uri.startswith(("http://", "https://")):
                return error_response("Invalid redirect_uri scheme", 400)

            url = await get_auth_url(provider, redirect_uri, state)

            if not url:
                return error_response("Provider not configured", 400)

            return json_response({"url": url})

        async def _handle_auth_callback(
            self,
            provider: str,
            body: dict[str, Any],
        ) -> HandlerResult:
            """Handle OAuth callback."""
            code = body.get("code")
            redirect_uri = body.get("redirect_uri", "http://localhost:3000/auth/callback")

            if not code:
                return error_response("Missing authorization code", 400)

            # Validate authorization code length to prevent abuse
            if not isinstance(code, str) or len(code) > MAX_AUTH_CODE_LENGTH:
                return error_response("Invalid authorization code", 400)

            # Validate redirect_uri
            if not isinstance(redirect_uri, str) or len(redirect_uri) > MAX_REDIRECT_URI_LENGTH:
                return error_response("Invalid redirect_uri", 400)

            if not redirect_uri.startswith(("http://", "https://")):
                return error_response("Invalid redirect_uri scheme", 400)

            success = await handle_auth_callback(provider, code, redirect_uri)

            if success:
                return json_response({"success": True})
            return error_response("Authentication failed", 401)

        async def _list_files(
            self,
            provider: str,
            query_params: dict[str, Any],
        ) -> HandlerResult:
            """List files in a folder."""
            raw_path = query_params.get("path", "/")
            page_size = safe_query_int(query_params, "limit", default=100, min_val=1, max_val=1000)

            # Validate path to prevent traversal attacks
            path = _validate_path(raw_path)
            if path is None:
                return error_response("Invalid path: traversal sequences not allowed", 400)

            files = await list_files(provider, path, page_size)

            return json_response({"files": files, "path": path})

        async def _download_file(
            self,
            provider: str,
            body: dict[str, Any],
        ) -> HandlerResult:
            """Download a file with size limit enforcement and checksum."""
            import base64

            file_id = body.get("file_id")
            if not file_id:
                return error_response("Missing file_id", 400)

            # Validate file_id format to prevent injection
            if not _validate_file_id(file_id):
                return error_response("Invalid file_id format", 400)

            content = await download_file(provider, file_id)

            if content is None:
                return error_response("Download failed or file exceeds size limit", 500)

            # Compute SHA-256 checksum for integrity verification
            checksum = hashlib.sha256(content).hexdigest()

            return json_response(
                {
                    "content": base64.b64encode(content).decode(),
                    "size": len(content),
                    "checksum_sha256": checksum,
                }
            )
