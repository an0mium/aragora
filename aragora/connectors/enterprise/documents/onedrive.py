"""
Microsoft OneDrive Enterprise Connector.

Provides full integration with OneDrive and SharePoint Online:
- OAuth2 authentication via Microsoft Graph
- File and folder traversal
- Office document export
- Delta sync for incremental updates
- Shared folder support

Requires Azure AD OAuth2 credentials.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Set
from urllib.parse import quote

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


# Office MIME types and their export formats
OFFICE_MIMES: Dict[str, str] = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.ms-excel": "xlsx",
    "application/msword": "doc",
}

# Supported file types for indexing
SUPPORTED_EXTENSIONS: Set[str] = {
    ".txt",
    ".md",
    ".json",
    ".xml",
    ".csv",
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
}


@dataclass
class OneDriveFile:
    """A OneDrive file."""

    id: str
    name: str
    mime_type: str
    size: int = 0
    created_time: Optional[datetime] = None
    modified_time: Optional[datetime] = None
    web_url: str = ""
    parent_id: Optional[str] = None
    parent_path: str = ""
    download_url: Optional[str] = None
    shared: bool = False
    drive_id: Optional[str] = None


@dataclass
class OneDriveFolder:
    """A OneDrive folder."""

    id: str
    name: str
    parent_id: Optional[str] = None
    path: str = ""
    child_count: int = 0


class OneDriveConnector(EnterpriseConnector):
    """
    Enterprise connector for Microsoft OneDrive.

    Features:
    - OAuth2 authentication via Microsoft Graph
    - Personal OneDrive and SharePoint sites
    - Office document text extraction
    - Folder traversal with include/exclude patterns
    - Delta sync for incremental updates
    - Webhook support for real-time updates

    Authentication:
    - OAuth2 with refresh token (recommended)
    - Client credentials (for app-only access)

    Usage:
        connector = OneDriveConnector(
            client_id="azure-app-id",
            client_secret="azure-app-secret",
            tenant_id="azure-tenant-id",
        )
        await connector.authenticate(refresh_token="user-refresh-token")

        # List files in root
        async for item in connector.list_files("/"):
            print(f"{item.name}: {item.size} bytes")

        # Download file content
        content = await connector.download_file(file_id)
    """

    CONNECTOR_TYPE = "onedrive"
    DISPLAY_NAME = "Microsoft OneDrive"

    # Microsoft Graph API base
    GRAPH_BASE = "https://graph.microsoft.com/v1.0"
    AUTH_URL = "https://login.microsoftonline.com"

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        tenant_id: Optional[str] = None,
        access_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
        drive_id: Optional[str] = None,  # Specific drive to use
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize OneDrive connector.

        Args:
            client_id: Azure AD application client ID
            client_secret: Azure AD application client secret
            tenant_id: Azure AD tenant ID (or 'common' for multi-tenant)
            access_token: Pre-existing access token
            refresh_token: OAuth2 refresh token for token renewal
            drive_id: Specific drive ID to use (default: user's OneDrive)
            include_patterns: Glob patterns for files to include
            exclude_patterns: Glob patterns for files to exclude
        """
        super().__init__()

        self.client_id = client_id or os.environ.get("ONEDRIVE_CLIENT_ID", "")
        self.client_secret = client_secret or os.environ.get("ONEDRIVE_CLIENT_SECRET", "")
        self.tenant_id = tenant_id or os.environ.get("ONEDRIVE_TENANT_ID", "common")
        self._access_token = access_token
        self._refresh_token = refresh_token
        self.drive_id = drive_id
        self.include_patterns = include_patterns or ["*"]
        self.exclude_patterns = exclude_patterns or []

        self._session: Optional[Any] = None
        self._token_expires: Optional[datetime] = None

    @property
    def source_type(self) -> SourceType:
        """Get the provenance source type."""
        return SourceType.ONEDRIVE

    def is_configured(self) -> bool:
        """Check if connector is properly configured."""
        return bool(self.client_id and self.client_secret)

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
        Authenticate with Microsoft Graph.

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

        url = f"{self.AUTH_URL}/{self.tenant_id}/oauth2/v2.0/token"
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }

        async with session.post(url, data=data) as resp:
            if resp.status == 200:
                result = await resp.json()
                self._access_token = result["access_token"]
                self._refresh_token = result.get("refresh_token")
                expires_in = result.get("expires_in", 3600)
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

        url = f"{self.AUTH_URL}/{self.tenant_id}/oauth2/v2.0/token"
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self._refresh_token,
            "grant_type": "refresh_token",
        }

        async with session.post(url, data=data) as resp:
            if resp.status == 200:
                result = await resp.json()
                self._access_token = result["access_token"]
                self._refresh_token = result.get("refresh_token", self._refresh_token)
                expires_in = result.get("expires_in", 3600)
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
        method: str,
        endpoint: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make an authenticated API request."""
        await self._ensure_valid_token()
        session = await self._get_session()

        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }
        headers.update(kwargs.pop("headers", {}))

        url = f"{self.GRAPH_BASE}{endpoint}"

        async with session.request(method, url, headers=headers, **kwargs) as resp:
            if resp.status == 204:
                return {}
            if resp.status >= 400:
                error = await resp.text()
                logger.error(f"API request failed: {resp.status} {error}")
                # Map HTTP status to appropriate exception type
                if resp.status in (401, 403):
                    raise ConnectorAuthError(
                        f"OneDrive authentication failed: {resp.status}",
                        connector_name="onedrive",
                    )
                elif resp.status == 429:
                    raise ConnectorRateLimitError(
                        "OneDrive rate limit exceeded",
                        connector_name="onedrive",
                    )
                elif resp.status == 404:
                    raise ConnectorNotFoundError(
                        "OneDrive resource not found",
                        connector_name="onedrive",
                    )
                else:
                    raise ConnectorAPIError(
                        f"OneDrive API error: {resp.status}",
                        connector_name="onedrive",
                        status_code=resp.status,
                    )
            return await resp.json()

    def _get_drive_path(self) -> str:
        """Get the base path for the drive."""
        if self.drive_id:
            return f"/drives/{self.drive_id}"
        return "/me/drive"

    async def get_user_info(self) -> Dict[str, Any]:
        """Get current user info."""
        return await self._api_request("GET", "/me")

    async def list_drives(self) -> List[Dict[str, Any]]:
        """List available drives for the user."""
        result = await self._api_request("GET", "/me/drives")
        return result.get("value", [])

    async def list_files(
        self,
        folder_path: str = "/",
        page_size: int = 100,
    ) -> AsyncIterator[OneDriveFile]:
        """
        List files in a folder.

        Args:
            folder_path: Path to the folder (default: root)
            page_size: Number of items per page

        Yields:
            OneDriveFile objects
        """
        drive_path = self._get_drive_path()

        if folder_path == "/" or not folder_path:
            endpoint = f"{drive_path}/root/children"
        else:
            # URL encode the path
            encoded_path = quote(folder_path.lstrip("/"))
            endpoint = f"{drive_path}/root:/{encoded_path}:/children"

        params = {"$top": page_size}

        while endpoint:
            result = await self._api_request("GET", endpoint, params=params)

            for item in result.get("value", []):
                # Skip folders for file listing
                if "folder" in item:
                    continue

                yield OneDriveFile(
                    id=item["id"],
                    name=item["name"],
                    mime_type=item.get("file", {}).get("mimeType", "application/octet-stream"),
                    size=item.get("size", 0),
                    created_time=self._parse_datetime(item.get("createdDateTime")),
                    modified_time=self._parse_datetime(item.get("lastModifiedDateTime")),
                    web_url=item.get("webUrl", ""),
                    parent_id=item.get("parentReference", {}).get("id"),
                    parent_path=item.get("parentReference", {}).get("path", ""),
                    download_url=item.get("@microsoft.graph.downloadUrl"),
                    shared="shared" in item,
                    drive_id=item.get("parentReference", {}).get("driveId"),
                )

            # Handle pagination
            next_link = result.get("@odata.nextLink")
            if next_link:
                # Parse the next link - it's a full URL
                endpoint = next_link.replace(self.GRAPH_BASE, "")
                params = {}
            else:
                endpoint = None

    async def list_folders(
        self,
        parent_path: str = "/",
    ) -> AsyncIterator[OneDriveFolder]:
        """
        List folders in a path.

        Args:
            parent_path: Path to list folders from

        Yields:
            OneDriveFolder objects
        """
        drive_path = self._get_drive_path()

        if parent_path == "/" or not parent_path:
            endpoint = f"{drive_path}/root/children"
        else:
            encoded_path = quote(parent_path.lstrip("/"))
            endpoint = f"{drive_path}/root:/{encoded_path}:/children"

        params = {"$filter": "folder ne null"}

        while endpoint:
            result = await self._api_request("GET", endpoint, params=params)

            for item in result.get("value", []):
                if "folder" not in item:
                    continue

                yield OneDriveFolder(
                    id=item["id"],
                    name=item["name"],
                    parent_id=item.get("parentReference", {}).get("id"),
                    path=item.get("parentReference", {}).get("path", "") + "/" + item["name"],
                    child_count=item.get("folder", {}).get("childCount", 0),
                )

            next_link = result.get("@odata.nextLink")
            if next_link:
                endpoint = next_link.replace(self.GRAPH_BASE, "")
                params = {}
            else:
                endpoint = None

    async def download_file(self, file_id: str) -> bytes:
        """
        Download file content.

        Args:
            file_id: OneDrive file ID

        Returns:
            File content as bytes
        """
        await self._ensure_valid_token()
        session = await self._get_session()

        # Get download URL
        drive_path = self._get_drive_path()
        endpoint = f"{drive_path}/items/{file_id}/content"
        url = f"{self.GRAPH_BASE}{endpoint}"

        headers = {"Authorization": f"Bearer {self._access_token}"}

        async with session.get(url, headers=headers, allow_redirects=True) as resp:
            if resp.status >= 400:
                # Map HTTP status to appropriate exception type
                if resp.status in (401, 403):
                    raise ConnectorAuthError(
                        f"OneDrive download auth failed: {resp.status}",
                        connector_name="onedrive",
                    )
                elif resp.status == 429:
                    raise ConnectorRateLimitError(
                        "OneDrive download rate limited",
                        connector_name="onedrive",
                    )
                elif resp.status == 404:
                    raise ConnectorNotFoundError(
                        f"OneDrive file not found: {file_id}",
                        connector_name="onedrive",
                        resource_id=file_id,
                    )
                else:
                    raise ConnectorAPIError(
                        f"OneDrive download failed: {resp.status}",
                        connector_name="onedrive",
                        status_code=resp.status,
                    )
            return await resp.read()

    async def get_file_metadata(self, file_id: str) -> OneDriveFile:
        """Get metadata for a specific file."""
        drive_path = self._get_drive_path()
        result = await self._api_request("GET", f"{drive_path}/items/{file_id}")

        return OneDriveFile(
            id=result["id"],
            name=result["name"],
            mime_type=result.get("file", {}).get("mimeType", "application/octet-stream"),
            size=result.get("size", 0),
            created_time=self._parse_datetime(result.get("createdDateTime")),
            modified_time=self._parse_datetime(result.get("lastModifiedDateTime")),
            web_url=result.get("webUrl", ""),
            parent_id=result.get("parentReference", {}).get("id"),
            download_url=result.get("@microsoft.graph.downloadUrl"),
        )

    async def search_files(
        self,
        query: str,
        max_results: int = 50,
    ) -> AsyncIterator[OneDriveFile]:
        """
        Search for files.

        Args:
            query: Search query
            max_results: Maximum results to return

        Yields:
            Matching OneDriveFile objects
        """
        drive_path = self._get_drive_path()
        endpoint = f"{drive_path}/root/search(q='{quote(query)}')"
        params = {"$top": min(max_results, 200)}

        result = await self._api_request("GET", endpoint, params=params)

        for item in result.get("value", [])[:max_results]:
            if "folder" in item:
                continue

            yield OneDriveFile(
                id=item["id"],
                name=item["name"],
                mime_type=item.get("file", {}).get("mimeType", "application/octet-stream"),
                size=item.get("size", 0),
                web_url=item.get("webUrl", ""),
            )

    async def sync_items(
        self,
        state: Optional[SyncState] = None,
    ) -> AsyncIterator[SyncItem]:
        """
        Incremental sync using delta API.

        Args:
            state: Previous sync state

        Yields:
            SyncItem for each changed file
        """
        drive_path = self._get_drive_path()

        if state and state.cursor:
            # Use delta link from previous sync
            endpoint = state.cursor.replace(self.GRAPH_BASE, "")
        else:
            endpoint = f"{drive_path}/root/delta"

        while endpoint:
            result = await self._api_request("GET", endpoint)

            for item in result.get("value", []):
                # Skip folders
                if "folder" in item:
                    continue

                # Check if deleted
                if "deleted" in item:
                    yield SyncItem(
                        id=item["id"],
                        name=item.get("name", ""),
                        action="delete",
                    )
                    continue

                # Check extension filter
                name = item.get("name", "")
                ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
                if ext not in SUPPORTED_EXTENSIONS:
                    continue

                yield SyncItem(
                    id=item["id"],
                    name=name,
                    action="update" if state else "create",
                    size=item.get("size", 0),
                    modified=self._parse_datetime(item.get("lastModifiedDateTime")),
                    metadata={
                        "mime_type": item.get("file", {}).get("mimeType"),
                        "web_url": item.get("webUrl"),
                        "path": item.get("parentReference", {}).get("path", ""),
                    },
                )

            # Get next page or delta link
            next_link = result.get("@odata.nextLink")
            delta_link = result.get("@odata.deltaLink")

            if next_link:
                endpoint = next_link.replace(self.GRAPH_BASE, "")
            elif delta_link:
                # Store delta link for next sync
                yield SyncItem(
                    id="__sync_state__",
                    name="",
                    action="state",
                    metadata={"delta_link": delta_link},
                )
                endpoint = None
            else:
                endpoint = None

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
        scopes = "files.read files.read.all offline_access user.read"
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": scopes,
            "response_mode": "query",
        }
        if state:
            params["state"] = state

        query = "&".join(f"{k}={quote(v)}" for k, v in params.items())
        return f"{self.AUTH_URL}/{self.tenant_id}/oauth2/v2.0/authorize?{query}"
