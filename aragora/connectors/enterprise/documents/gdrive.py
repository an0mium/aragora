"""
Google Drive Enterprise Connector.

Provides full integration with Google Drive and Shared Drives:
- OAuth2 authentication flow
- File and folder traversal
- Google Docs/Sheets/Slides export to text
- Incremental sync via Changes API
- Shared Drive support

Requires Google Cloud OAuth2 credentials.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Set

from aragora.connectors.enterprise.base import (
    EnterpriseConnector,
    SyncItem,
    SyncState,
)
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)


# Google Workspace MIME types and their export formats
GOOGLE_WORKSPACE_MIMES: Dict[str, str] = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
    "application/vnd.google-apps.drawing": "image/png",
}

# Supported file types for indexing
SUPPORTED_MIMES: Set[str] = {
    "text/plain",
    "text/html",
    "text/markdown",
    "text/csv",
    "application/json",
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


@dataclass
class DriveFile:
    """A Google Drive file."""

    id: str
    name: str
    mime_type: str
    size: int = 0
    created_time: Optional[datetime] = None
    modified_time: Optional[datetime] = None
    web_view_link: str = ""
    parents: List[str] = field(default_factory=list)
    owners: List[str] = field(default_factory=list)
    shared: bool = False
    drive_id: Optional[str] = None  # For Shared Drives


@dataclass
class DriveFolder:
    """A Google Drive folder."""

    id: str
    name: str
    parent_id: Optional[str] = None
    drive_id: Optional[str] = None


class GoogleDriveConnector(EnterpriseConnector):
    """
    Enterprise connector for Google Drive.

    Features:
    - OAuth2 authentication
    - My Drive and Shared Drives support
    - Google Workspace document export
    - Folder traversal with include/exclude patterns
    - Incremental sync via Changes API
    - Webhook support for real-time updates

    Authentication:
    - OAuth2 with refresh token (recommended)
    - Service account (for domain-wide access)

    Usage:
        connector = GoogleDriveConnector(
            folder_ids=["root"],  # Optional: specific folders to sync
            include_shared_drives=True,
        )
        result = await connector.sync()
    """

    def __init__(
        self,
        folder_ids: Optional[List[str]] = None,
        include_shared_drives: bool = True,
        include_trashed: bool = False,
        export_google_docs: bool = True,
        max_file_size_mb: int = 100,
        exclude_patterns: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize Google Drive connector.

        Args:
            folder_ids: Specific folder IDs to sync (None = entire drive)
            include_shared_drives: Whether to include Shared Drives
            include_trashed: Whether to include trashed files
            export_google_docs: Export Google Docs/Sheets/Slides as text
            max_file_size_mb: Maximum file size to sync in MB
            exclude_patterns: File name patterns to exclude
        """
        super().__init__(connector_id="gdrive", **kwargs)

        self.folder_ids = folder_ids
        self.include_shared_drives = include_shared_drives
        self.include_trashed = include_trashed
        self.export_google_docs = export_google_docs
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.exclude_patterns = set(exclude_patterns or [])

        # Cache for folder paths
        self._folder_cache: Dict[str, DriveFolder] = {}
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

    @property
    def source_type(self) -> SourceType:
        return SourceType.DOCUMENT

    @property
    def name(self) -> str:
        return "Google Drive"

    async def _get_access_token(self) -> str:
        """Get valid access token, refreshing if needed."""
        now = datetime.now(timezone.utc)

        if self._access_token and self._token_expiry and now < self._token_expiry:
            return self._access_token

        # Get credentials
        client_id = await self.credentials.get_credential("GDRIVE_CLIENT_ID")
        client_secret = await self.credentials.get_credential("GDRIVE_CLIENT_SECRET")
        refresh_token = await self.credentials.get_credential("GDRIVE_REFRESH_TOKEN")

        if not all([client_id, client_secret, refresh_token]):
            raise ValueError(
                "Google Drive credentials not configured. "
                "Set GDRIVE_CLIENT_ID, GDRIVE_CLIENT_SECRET, and GDRIVE_REFRESH_TOKEN"
            )

        # Refresh access token
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            response.raise_for_status()
            data = response.json()

        self._access_token = data["access_token"]
        expires_in = data.get("expires_in", 3600)
        from datetime import timedelta
        self._token_expiry = now.replace(microsecond=0) + timedelta(seconds=expires_in - 60)

        return self._access_token

    async def _api_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make a request to Google Drive API."""
        import httpx

        token = await self._get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

        url = f"https://www.googleapis.com/drive/v3{endpoint}"

        async with httpx.AsyncClient() as client:
            response = await client.request(
                method,
                url,
                headers=headers,
                params=params,
                timeout=60,
                **kwargs,
            )
            response.raise_for_status()
            return response.json() if response.content else {}

    async def _download_file(self, file_id: str, mime_type: Optional[str] = None) -> bytes:
        """Download file content."""
        import httpx

        token = await self._get_access_token()
        headers = {"Authorization": f"Bearer {token}"}

        # For Google Workspace files, use export
        if mime_type and mime_type in GOOGLE_WORKSPACE_MIMES:
            export_mime = GOOGLE_WORKSPACE_MIMES[mime_type]
            url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export"
            params = {"mimeType": export_mime}
        else:
            url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
            params = {"alt": "media"}

        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers=headers,
                params=params,
                timeout=120,
                follow_redirects=True,
            )
            response.raise_for_status()
            return response.content

    async def _list_files(
        self,
        folder_id: Optional[str] = None,
        page_token: Optional[str] = None,
        modified_after: Optional[datetime] = None,
    ) -> tuple[List[DriveFile], Optional[str]]:
        """List files in a folder or entire drive."""
        query_parts = []

        if folder_id:
            query_parts.append(f"'{folder_id}' in parents")

        if not self.include_trashed:
            query_parts.append("trashed = false")

        if modified_after:
            modified_str = modified_after.strftime("%Y-%m-%dT%H:%M:%S")
            query_parts.append(f"modifiedTime > '{modified_str}'")

        # Filter to supported file types
        mime_queries = [f"mimeType = '{m}'" for m in SUPPORTED_MIMES]
        if self.export_google_docs:
            mime_queries.extend([f"mimeType = '{m}'" for m in GOOGLE_WORKSPACE_MIMES.keys()])
        mime_queries.append("mimeType = 'application/vnd.google-apps.folder'")

        query_parts.append(f"({' or '.join(mime_queries)})")

        query = " and ".join(query_parts) if query_parts else None

        params: Dict[str, Any] = {
            "pageSize": 100,
            "fields": "nextPageToken,files(id,name,mimeType,size,createdTime,modifiedTime,webViewLink,parents,owners,shared,driveId)",
        }

        if query:
            params["q"] = query
        if page_token:
            params["pageToken"] = page_token

        # Include shared drives if enabled
        if self.include_shared_drives:
            params["includeItemsFromAllDrives"] = True
            params["supportsAllDrives"] = True

        data = await self._api_request("/files", params=params)

        files = []
        for item in data.get("files", []):
            # Parse timestamps
            created = None
            modified = None
            if item.get("createdTime"):
                try:
                    created = datetime.fromisoformat(item["createdTime"].replace("Z", "+00:00"))
                except ValueError:
                    pass
            if item.get("modifiedTime"):
                try:
                    modified = datetime.fromisoformat(item["modifiedTime"].replace("Z", "+00:00"))
                except ValueError:
                    pass

            files.append(
                DriveFile(
                    id=item["id"],
                    name=item["name"],
                    mime_type=item.get("mimeType", "application/octet-stream"),
                    size=int(item.get("size", 0)),
                    created_time=created,
                    modified_time=modified,
                    web_view_link=item.get("webViewLink", ""),
                    parents=item.get("parents", []),
                    owners=[o.get("displayName", "") for o in item.get("owners", [])],
                    shared=item.get("shared", False),
                    drive_id=item.get("driveId"),
                )
            )

        return files, data.get("nextPageToken")

    async def _list_shared_drives(self) -> List[Dict[str, Any]]:
        """List all accessible Shared Drives."""
        if not self.include_shared_drives:
            return []

        drives = []
        page_token = None

        while True:
            params: Dict[str, Any] = {"pageSize": 100}
            if page_token:
                params["pageToken"] = page_token

            data = await self._api_request("/drives", params=params)

            drives.extend(data.get("drives", []))

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return drives

    async def _get_changes(
        self,
        start_page_token: str,
    ) -> tuple[List[DriveFile], str]:
        """Get changes since a token for incremental sync."""
        files = []
        page_token = start_page_token

        while page_token:
            params: Dict[str, Any] = {
                "pageToken": page_token,
                "pageSize": 100,
                "fields": "nextPageToken,newStartPageToken,changes(fileId,removed,file(id,name,mimeType,size,createdTime,modifiedTime,webViewLink,parents,owners,shared,driveId))",
            }

            if self.include_shared_drives:
                params["includeItemsFromAllDrives"] = True
                params["supportsAllDrives"] = True

            data = await self._api_request("/changes", params=params)

            for change in data.get("changes", []):
                if change.get("removed"):
                    continue  # Skip deleted files

                file_data = change.get("file", {})
                if not file_data:
                    continue

                # Parse timestamps
                created = None
                modified = None
                if file_data.get("createdTime"):
                    try:
                        created = datetime.fromisoformat(file_data["createdTime"].replace("Z", "+00:00"))
                    except ValueError:
                        pass
                if file_data.get("modifiedTime"):
                    try:
                        modified = datetime.fromisoformat(file_data["modifiedTime"].replace("Z", "+00:00"))
                    except ValueError:
                        pass

                files.append(
                    DriveFile(
                        id=file_data["id"],
                        name=file_data["name"],
                        mime_type=file_data.get("mimeType", "application/octet-stream"),
                        size=int(file_data.get("size", 0)),
                        created_time=created,
                        modified_time=modified,
                        web_view_link=file_data.get("webViewLink", ""),
                        parents=file_data.get("parents", []),
                        owners=[o.get("displayName", "") for o in file_data.get("owners", [])],
                        shared=file_data.get("shared", False),
                        drive_id=file_data.get("driveId"),
                    )
                )

            page_token = data.get("nextPageToken")
            if data.get("newStartPageToken"):
                return files, data["newStartPageToken"]

        return files, start_page_token

    async def _get_start_page_token(self) -> str:
        """Get the starting page token for changes tracking."""
        params: Dict[str, Any] = {}
        if self.include_shared_drives:
            params["supportsAllDrives"] = True

        data = await self._api_request("/changes/startPageToken", params=params)
        return data["startPageToken"]

    def _should_skip_file(self, file: DriveFile) -> bool:
        """Check if file should be skipped based on filters."""
        # Skip folders
        if file.mime_type == "application/vnd.google-apps.folder":
            return True

        # Skip files over size limit (Google Workspace files have size=0)
        if file.size > self.max_file_size_bytes:
            return True

        # Skip excluded patterns
        for pattern in self.exclude_patterns:
            if pattern in file.name:
                return True

        # Skip unsupported MIME types
        if file.mime_type not in SUPPORTED_MIMES and file.mime_type not in GOOGLE_WORKSPACE_MIMES:
            return True

        return False

    async def _extract_text(self, file: DriveFile) -> str:
        """Extract text content from a file."""
        try:
            content = await self._download_file(file.id, file.mime_type)

            # For Google Workspace exports, content is already text
            if file.mime_type in GOOGLE_WORKSPACE_MIMES:
                return content.decode("utf-8", errors="replace")

            # For text files
            if file.mime_type.startswith("text/"):
                return content.decode("utf-8", errors="replace")

            # For other files, try to extract text
            # In production, would use document parsers
            if file.mime_type == "application/pdf":
                # Would use PDF parser here
                return f"[PDF content from {file.name}]"

            if file.mime_type in (
                "application/msword",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ):
                # Would use DOCX parser here
                return f"[Word document content from {file.name}]"

            return content.decode("utf-8", errors="replace")[:10000]

        except Exception as e:
            logger.warning(f"[{self.name}] Failed to extract text from {file.name}: {e}")
            return ""

    async def sync_items(
        self,
        state: SyncState,
        batch_size: int = 100,
    ) -> AsyncIterator[SyncItem]:
        """
        Yield Google Drive files for syncing.

        Uses Changes API for incremental sync when cursor is available.
        """
        items_yielded = 0

        # Check for incremental sync
        if state.cursor:
            # Use Changes API for incremental sync
            logger.info(f"[{self.name}] Starting incremental sync from token {state.cursor[:20]}...")
            files, new_token = await self._get_changes(state.cursor)
            state.cursor = new_token

            for file in files:
                if self._should_skip_file(file):
                    continue

                text_content = await self._extract_text(file)
                if not text_content:
                    continue

                yield SyncItem(
                    id=f"gdrive-{file.id}",
                    content=text_content[:50000],
                    source_type="document",
                    source_id=f"gdrive/{file.id}",
                    title=file.name,
                    url=file.web_view_link,
                    author=file.owners[0] if file.owners else "",
                    created_at=file.created_time,
                    updated_at=file.modified_time,
                    domain="enterprise/gdrive",
                    confidence=0.85,
                    metadata={
                        "file_id": file.id,
                        "mime_type": file.mime_type,
                        "size": file.size,
                        "shared": file.shared,
                        "drive_id": file.drive_id,
                    },
                )

                items_yielded += 1
                if items_yielded >= batch_size:
                    await asyncio.sleep(0)

            return

        # Full sync
        logger.info(f"[{self.name}] Starting full sync...")

        # Get start token for future incremental syncs
        state.cursor = await self._get_start_page_token()

        # Determine folders to sync
        folders_to_process: List[Optional[str]] = []

        if self.folder_ids:
            folders_to_process = list(self.folder_ids)
        else:
            # Sync entire My Drive
            folders_to_process = [None]  # None means root

            # Also sync Shared Drives
            if self.include_shared_drives:
                shared_drives = await self._list_shared_drives()
                for drive in shared_drives:
                    folders_to_process.append(drive["id"])

        state.items_total = len(folders_to_process)

        for folder_id in folders_to_process:
            page_token = None

            while True:
                files, page_token = await self._list_files(
                    folder_id=folder_id,
                    page_token=page_token,
                )

                for file in files:
                    if self._should_skip_file(file):
                        continue

                    text_content = await self._extract_text(file)
                    if not text_content:
                        continue

                    yield SyncItem(
                        id=f"gdrive-{file.id}",
                        content=text_content[:50000],
                        source_type="document",
                        source_id=f"gdrive/{file.id}",
                        title=file.name,
                        url=file.web_view_link,
                        author=file.owners[0] if file.owners else "",
                        created_at=file.created_time,
                        updated_at=file.modified_time,
                        domain="enterprise/gdrive",
                        confidence=0.85,
                        metadata={
                            "file_id": file.id,
                            "mime_type": file.mime_type,
                            "size": file.size,
                            "shared": file.shared,
                            "drive_id": file.drive_id,
                            "folder_id": folder_id,
                        },
                    )

                    items_yielded += 1
                    if items_yielded >= batch_size:
                        await asyncio.sleep(0)

                if not page_token:
                    break

    async def search(
        self,
        query: str,
        limit: int = 10,
        folder_id: Optional[str] = None,
        **kwargs,
    ) -> list:
        """Search Google Drive files."""
        from aragora.connectors.base import Evidence

        # Build search query
        q_parts = [f"fullText contains '{query}'"]

        if folder_id:
            q_parts.append(f"'{folder_id}' in parents")

        if not self.include_trashed:
            q_parts.append("trashed = false")

        params: Dict[str, Any] = {
            "q": " and ".join(q_parts),
            "pageSize": limit,
            "fields": "files(id,name,mimeType,webViewLink,modifiedTime)",
        }

        if self.include_shared_drives:
            params["includeItemsFromAllDrives"] = True
            params["supportsAllDrives"] = True

        try:
            data = await self._api_request("/files", params=params)

            results = []
            for item in data.get("files", []):
                results.append(
                    Evidence(
                        id=f"gdrive-{item['id']}",
                        source_type=self.source_type,
                        source_id=item["id"],
                        content="",  # Would fetch content on demand
                        title=item.get("name", ""),
                        url=item.get("webViewLink", ""),
                        confidence=0.8,
                        metadata={
                            "mime_type": item.get("mimeType", ""),
                        },
                    )
                )

            return results

        except Exception as e:
            logger.error(f"[{self.name}] Search failed: {e}")
            return []

    async def fetch(self, evidence_id: str) -> Optional[Any]:
        """Fetch a specific Google Drive file."""
        from aragora.connectors.base import Evidence

        # Extract file ID
        if evidence_id.startswith("gdrive-"):
            file_id = evidence_id[7:]
        else:
            file_id = evidence_id

        try:
            params: Dict[str, Any] = {
                "fields": "id,name,mimeType,size,createdTime,modifiedTime,webViewLink,owners",
            }

            if self.include_shared_drives:
                params["supportsAllDrives"] = True

            data = await self._api_request(f"/files/{file_id}", params=params)

            # Extract text content
            file = DriveFile(
                id=data["id"],
                name=data["name"],
                mime_type=data.get("mimeType", ""),
                size=int(data.get("size", 0)),
                web_view_link=data.get("webViewLink", ""),
                owners=[o.get("displayName", "") for o in data.get("owners", [])],
            )

            content = await self._extract_text(file)

            return Evidence(
                id=evidence_id,
                source_type=self.source_type,
                source_id=file_id,
                content=content,
                title=data.get("name", ""),
                url=data.get("webViewLink", ""),
                author=file.owners[0] if file.owners else "",
                confidence=0.85,
                metadata={
                    "mime_type": data.get("mimeType", ""),
                    "size": int(data.get("size", 0)),
                },
            )

        except Exception as e:
            logger.error(f"[{self.name}] Fetch failed: {e}")
            return None

    async def handle_webhook(self, payload: Dict[str, Any]) -> bool:
        """Handle Google Drive push notification webhook."""
        resource_state = payload.get("resourceState", "")
        resource_id = payload.get("resourceId", "")

        logger.info(f"[{self.name}] Webhook: {resource_state} on resource {resource_id}")

        if resource_state in ["change", "add", "update"]:
            # Trigger incremental sync
            asyncio.create_task(self.sync(max_items=100))
            return True

        return False


__all__ = ["GoogleDriveConnector", "DriveFile", "DriveFolder"]
