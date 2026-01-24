"""
Dropbox Enterprise Connector.

Provides integration with Dropbox:
- OAuth2 authentication
- File and folder traversal
- Content download
- Cursor-based incremental sync

Requires Dropbox OAuth2 app credentials.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Set
from urllib.parse import urlencode

from aragora.connectors.enterprise.base import (
    EnterpriseConnector,
    SyncItem,
    SyncState,
)
from aragora.connectors.exceptions import (
    ConnectorAPIError,
    ConnectorAuthError,
    ConnectorNotFoundError,
    ConnectorRateLimitError,
)
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)


# Supported file extensions for indexing
SUPPORTED_EXTENSIONS: Set[str] = {
    ".txt",
    ".md",
    ".json",
    ".xml",
    ".csv",
    ".yaml",
    ".yml",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".py",
    ".js",
    ".ts",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".html",
    ".css",
    ".sql",
    ".sh",
    ".rb",
    ".php",
}


@dataclass
class DropboxFile:
    """A Dropbox file."""

    id: str
    name: str
    path_lower: str
    path_display: str
    size: int = 0
    content_hash: Optional[str] = None
    modified_time: Optional[datetime] = None
    is_downloadable: bool = True
    shared_folder_id: Optional[str] = None


@dataclass
class DropboxFolder:
    """A Dropbox folder."""

    id: str
    name: str
    path_lower: str
    path_display: str
    shared_folder_id: Optional[str] = None


class DropboxConnector(EnterpriseConnector):
    """
    Enterprise connector for Dropbox.

    Features:
    - OAuth2 authentication
    - Personal and team folders
    - File/folder traversal
    - Cursor-based incremental sync
    - Shared folder support

    Authentication:
    - OAuth2 with refresh token (recommended)
    - Access token (short-lived)

    Usage:
        connector = DropboxConnector(
            app_key="dropbox-app-key",
            app_secret="dropbox-app-secret",
        )
        await connector.authenticate(refresh_token="user-refresh-token")

        # List files in root
        async for item in connector.list_files("/"):
            print(f"{item.name}: {item.size} bytes")

        # Download file content
        content = await connector.download_file(file_path)
    """

    CONNECTOR_TYPE = "dropbox"
    DISPLAY_NAME = "Dropbox"

    # Dropbox API endpoints
    API_BASE = "https://api.dropboxapi.com/2"
    CONTENT_BASE = "https://content.dropboxapi.com/2"
    AUTH_URL = "https://www.dropbox.com/oauth2"

    def __init__(
        self,
        app_key: Optional[str] = None,
        app_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
        root_path: str = "",
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize Dropbox connector.

        Args:
            app_key: Dropbox app key
            app_secret: Dropbox app secret
            access_token: Pre-existing access token
            refresh_token: OAuth2 refresh token
            root_path: Root path to sync from (default: entire Dropbox)
            include_patterns: Glob patterns for files to include
            exclude_patterns: Glob patterns for files to exclude
        """
        super().__init__()  # type: ignore[call-arg]

        self.app_key = app_key or os.environ.get("DROPBOX_APP_KEY", "")
        self.app_secret = app_secret or os.environ.get("DROPBOX_APP_SECRET", "")
        self._access_token = access_token
        self._refresh_token = refresh_token
        self.root_path = root_path
        self.include_patterns = include_patterns or ["*"]
        self.exclude_patterns = exclude_patterns or []

        self._session: Optional[Any] = None
        self._token_expires: Optional[datetime] = None

    @property
    def source_type(self) -> SourceType:
        """Get the provenance source type."""
        return SourceType.DROPBOX  # type: ignore[attr-defined]

    @property
    def is_configured(self) -> bool:
        """Check if connector is properly configured."""
        return bool(self.app_key and self.app_secret)

    async def _get_session(self):
        """Get or create aiohttp session with timeout protection."""
        if self._session is None:
            import aiohttp

            from aragora.http_client import DEFAULT_TIMEOUT

            self._session = aiohttp.ClientSession(timeout=DEFAULT_TIMEOUT)
        return self._session

    async def close(self):
        """Close the session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def authenticate(
        self,
        code: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        refresh_token: Optional[str] = None,
    ) -> bool:
        """
        Authenticate with Dropbox.

        Args:
            code: Authorization code from OAuth flow
            redirect_uri: Redirect URI used in OAuth flow
            refresh_token: Refresh token for token renewal

        Returns:
            True if authentication successful
        """
        if refresh_token:
            self._refresh_token = refresh_token

        if self._refresh_token:
            return await self._refresh_access_token()

        if code and redirect_uri:
            return await self._exchange_code(code, redirect_uri)

        return bool(self._access_token)

    async def _exchange_code(self, code: str, redirect_uri: str) -> bool:
        """Exchange authorization code for tokens."""
        session = await self._get_session()

        url = f"{self.AUTH_URL}/token"
        data = {
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        }

        # Use HTTP Basic Auth
        import base64

        credentials = base64.b64encode(f"{self.app_key}:{self.app_secret}".encode()).decode()
        headers = {"Authorization": f"Basic {credentials}"}

        async with session.post(url, data=data, headers=headers) as resp:
            if resp.status == 200:
                result = await resp.json()
                self._access_token = result["access_token"]
                self._refresh_token = result.get("refresh_token")
                expires_in = result.get("expires_in", 14400)
                self._token_expires = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(
                    seconds=expires_in
                )
                return True
            else:
                error = await resp.text()
                logger.error(f"Token exchange failed: {error}")
                return False

    async def _refresh_access_token(self) -> bool:
        """Refresh the access token."""
        if not self._refresh_token:
            return False

        session = await self._get_session()

        url = f"{self.AUTH_URL}/token"
        data = {
            "refresh_token": self._refresh_token,
            "grant_type": "refresh_token",
        }

        import base64

        credentials = base64.b64encode(f"{self.app_key}:{self.app_secret}".encode()).decode()
        headers = {"Authorization": f"Basic {credentials}"}

        async with session.post(url, data=data, headers=headers) as resp:
            if resp.status == 200:
                result = await resp.json()
                self._access_token = result["access_token"]
                expires_in = result.get("expires_in", 14400)
                self._token_expires = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(
                    seconds=expires_in
                )
                return True
            else:
                error = await resp.text()
                logger.error(f"Token refresh failed: {error}")
                return False

    async def _ensure_valid_token(self):
        """Ensure we have a valid access token."""
        if not self._access_token:
            raise ValueError("Not authenticated")

        if (
            self._token_expires
            and datetime.now(timezone.utc).replace(tzinfo=None) >= self._token_expires
        ):
            await self._refresh_access_token()

    async def _api_request(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        is_content: bool = False,
    ) -> Dict[str, Any]:
        """Make an authenticated API request."""
        await self._ensure_valid_token()
        session = await self._get_session()

        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

        base = self.CONTENT_BASE if is_content else self.API_BASE
        url = f"{base}{endpoint}"

        import json

        async with session.post(url, headers=headers, data=json.dumps(data or {})) as resp:
            if resp.status >= 400:
                error = await resp.text()
                logger.error(f"API request failed: {resp.status} {error}")
                # Map HTTP status to appropriate exception type
                if resp.status in (401, 403):
                    raise ConnectorAuthError(
                        f"Dropbox authentication failed: {resp.status}",
                        connector_name="dropbox",
                    )
                elif resp.status == 429:
                    raise ConnectorRateLimitError(
                        "Dropbox rate limit exceeded",
                        connector_name="dropbox",
                    )
                elif resp.status == 404:
                    raise ConnectorNotFoundError(
                        "Dropbox resource not found",
                        connector_name="dropbox",
                    )
                else:
                    raise ConnectorAPIError(
                        f"Dropbox API error: {resp.status}",
                        connector_name="dropbox",
                        status_code=resp.status,
                    )
            return await resp.json()

    async def get_account_info(self) -> Dict[str, Any]:
        """Get current account info."""
        return await self._api_request("/users/get_current_account")

    async def list_files(
        self,
        folder_path: str = "",
        recursive: bool = False,
        page_size: int = 100,
    ) -> AsyncIterator[DropboxFile]:
        """
        List files in a folder.

        Args:
            folder_path: Path to the folder (empty string for root)
            recursive: Include files in subfolders
            page_size: Number of items per page

        Yields:
            DropboxFile objects
        """
        # Normalize path
        path = folder_path if folder_path else ""
        if path and not path.startswith("/"):
            path = "/" + path

        data = {
            "path": path,
            "recursive": recursive,
            "limit": page_size,
        }

        result = await self._api_request("/files/list_folder", data)

        while True:
            for entry in result.get("entries", []):
                # Skip folders for file listing
                if entry[".tag"] == "folder":
                    continue

                # Check extension filter
                name = entry.get("name", "")
                "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""

                yield DropboxFile(
                    id=entry.get("id", ""),
                    name=name,
                    path_lower=entry.get("path_lower", ""),
                    path_display=entry.get("path_display", ""),
                    size=entry.get("size", 0),
                    content_hash=entry.get("content_hash"),
                    modified_time=self._parse_datetime(entry.get("client_modified")),
                    is_downloadable=entry.get("is_downloadable", True),
                )

            # Handle pagination
            if not result.get("has_more"):
                break

            cursor = result.get("cursor")
            result = await self._api_request(
                "/files/list_folder/continue",
                {"cursor": cursor},
            )

    async def list_folders(
        self,
        parent_path: str = "",
    ) -> AsyncIterator[DropboxFolder]:
        """
        List folders in a path.

        Args:
            parent_path: Path to list folders from

        Yields:
            DropboxFolder objects
        """
        path = parent_path if parent_path else ""
        if path and not path.startswith("/"):
            path = "/" + path

        data = {"path": path, "recursive": False}

        result = await self._api_request("/files/list_folder", data)

        while True:
            for entry in result.get("entries", []):
                if entry[".tag"] != "folder":
                    continue

                yield DropboxFolder(
                    id=entry.get("id", ""),
                    name=entry.get("name", ""),
                    path_lower=entry.get("path_lower", ""),
                    path_display=entry.get("path_display", ""),
                    shared_folder_id=entry.get("shared_folder_id"),
                )

            if not result.get("has_more"):
                break

            cursor = result.get("cursor")
            result = await self._api_request(
                "/files/list_folder/continue",
                {"cursor": cursor},
            )

    async def download_file(self, file_path: str) -> bytes:
        """
        Download file content.

        Args:
            file_path: Dropbox file path

        Returns:
            File content as bytes
        """
        await self._ensure_valid_token()
        session = await self._get_session()

        import json

        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Dropbox-API-Arg": json.dumps({"path": file_path}),
        }

        url = f"{self.CONTENT_BASE}/files/download"

        async with session.post(url, headers=headers) as resp:
            if resp.status >= 400:
                error = await resp.text()
                # Map HTTP status to appropriate exception type
                if resp.status in (401, 403):
                    raise ConnectorAuthError(
                        f"Dropbox download auth failed: {resp.status}",
                        connector_name="dropbox",
                    )
                elif resp.status == 429:
                    raise ConnectorRateLimitError(
                        "Dropbox download rate limited",
                        connector_name="dropbox",
                    )
                elif resp.status == 404:
                    raise ConnectorNotFoundError(
                        f"Dropbox file not found: {file_path}",
                        connector_name="dropbox",
                        resource_id=file_path,
                    )
                else:
                    raise ConnectorAPIError(
                        f"Dropbox download failed: {resp.status} {error}",
                        connector_name="dropbox",
                        status_code=resp.status,
                    )
            return await resp.read()

    async def get_file_metadata(self, file_path: str) -> DropboxFile:
        """Get metadata for a specific file."""
        result = await self._api_request("/files/get_metadata", {"path": file_path})

        return DropboxFile(
            id=result.get("id", ""),
            name=result.get("name", ""),
            path_lower=result.get("path_lower", ""),
            path_display=result.get("path_display", ""),
            size=result.get("size", 0),
            content_hash=result.get("content_hash"),
            modified_time=self._parse_datetime(result.get("client_modified")),
            is_downloadable=result.get("is_downloadable", True),
        )

    async def search_files(
        self,
        query: str,
        max_results: int = 50,
        file_extensions: Optional[List[str]] = None,
    ) -> AsyncIterator[DropboxFile]:
        """
        Search for files.

        Args:
            query: Search query
            max_results: Maximum results to return
            file_extensions: Filter by extensions

        Yields:
            Matching DropboxFile objects
        """
        data: Dict[str, Any] = {
            "query": query,
            "options": {
                "max_results": min(max_results, 100),
                "file_status": "active",
            },
        }

        if file_extensions:
            data["options"]["file_extensions"] = file_extensions

        result = await self._api_request("/files/search_v2", data)

        for match in result.get("matches", [])[:max_results]:
            metadata = match.get("metadata", {}).get("metadata", {})
            if metadata.get(".tag") == "folder":
                continue

            yield DropboxFile(
                id=metadata.get("id", ""),
                name=metadata.get("name", ""),
                path_lower=metadata.get("path_lower", ""),
                path_display=metadata.get("path_display", ""),
                size=metadata.get("size", 0),
                content_hash=metadata.get("content_hash"),
            )

    async def sync_items(  # type: ignore[override]
        self,
        state: Optional[SyncState] = None,
    ) -> AsyncIterator[SyncItem]:
        """
        Incremental sync using cursor.

        Args:
            state: Previous sync state with cursor

        Yields:
            SyncItem for each changed file
        """
        if state and state.cursor:
            # Continue from cursor
            result = await self._api_request(
                "/files/list_folder/continue",
                {"cursor": state.cursor},
            )
        else:
            # Start fresh
            path = self.root_path if self.root_path else ""
            result = await self._api_request(
                "/files/list_folder",
                {"path": path, "recursive": True},
            )

        while True:
            for entry in result.get("entries", []):
                if entry[".tag"] == "folder":
                    continue

                name = entry.get("name", "")
                ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""

                if ext not in SUPPORTED_EXTENSIONS:
                    continue

                if entry[".tag"] == "deleted":
                    yield SyncItem(  # type: ignore[call-arg]
                        id=entry.get("path_lower", ""),
                        name=name,
                        action="delete",
                    )
                else:
                    yield SyncItem(  # type: ignore[call-arg]
                        id=entry.get("id", ""),
                        name=name,
                        action="update" if state else "create",
                        size=entry.get("size", 0),
                        modified=self._parse_datetime(entry.get("client_modified")),
                        metadata={
                            "path": entry.get("path_display"),
                            "content_hash": entry.get("content_hash"),
                        },
                    )

            if not result.get("has_more"):
                break

            cursor = result.get("cursor")
            result = await self._api_request(
                "/files/list_folder/continue",
                {"cursor": cursor},
            )

        # Yield cursor for next sync
        final_cursor = result.get("cursor")
        if final_cursor:
            yield SyncItem(  # type: ignore[call-arg]
                id="__sync_state__",
                name="",
                action="state",
                metadata={"cursor": final_cursor},
            )

    def _parse_datetime(self, dt_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO datetime string."""
        if not dt_str:
            return None
        try:
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except ValueError:
            return None

    def get_oauth_url(self, redirect_uri: str, state: str = "") -> str:
        """
        Generate OAuth authorization URL.

        Args:
            redirect_uri: Redirect URI after auth
            state: Optional state parameter

        Returns:
            Authorization URL
        """
        params = {
            "client_id": self.app_key,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "token_access_type": "offline",  # Get refresh token
        }
        if state:
            params["state"] = state

        query = urlencode(params)
        return f"{self.AUTH_URL}/authorize?{query}"
