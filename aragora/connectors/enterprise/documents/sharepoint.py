"""
SharePoint Enterprise Connector.

Provides full integration with Microsoft SharePoint Online:
- Document library crawling and indexing
- Site/subsite traversal
- List item extraction
- Incremental sync via change tokens
- Webhook support for real-time updates

Requires Microsoft Graph API credentials.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Set

from aragora.connectors.enterprise.base import (
    EnterpriseConnector,
    SyncItem,
    SyncState,
)
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)

# File types to index
INDEXABLE_EXTENSIONS = {
    ".docx",
    ".doc",
    ".pdf",
    ".txt",
    ".md",
    ".rtf",
    ".xlsx",
    ".xls",
    ".csv",
    ".pptx",
    ".ppt",
    ".html",
    ".htm",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
}

# Maximum file size to index (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024


@dataclass
class SharePointSite:
    """A SharePoint site."""

    id: str
    name: str
    display_name: str
    web_url: str
    created: Optional[datetime] = None
    last_modified: Optional[datetime] = None


@dataclass
class SharePointDrive:
    """A SharePoint document library (drive)."""

    id: str
    name: str
    drive_type: str
    web_url: str
    site_id: str


@dataclass
class SharePointItem:
    """A SharePoint file or folder."""

    id: str
    name: str
    path: str
    web_url: str
    size: int = 0
    mime_type: str = ""
    is_folder: bool = False
    created_by: str = ""
    modified_by: str = ""
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    content: str = ""
    etag: str = ""


class SharePointConnector(EnterpriseConnector):
    """
    Enterprise connector for Microsoft SharePoint Online.

    Features:
    - Document library indexing
    - Site collection traversal
    - List item extraction
    - Incremental sync via delta tokens
    - Webhook support for real-time updates
    - Automatic text extraction from Office documents

    Authentication:
    - Requires Azure AD app registration
    - Uses client credentials flow (app-only)
    - Credentials: SHAREPOINT_TENANT_ID, SHAREPOINT_CLIENT_ID, SHAREPOINT_CLIENT_SECRET

    Usage:
        connector = SharePointConnector(
            site_url="https://contoso.sharepoint.com/sites/engineering",
        )
        result = await connector.sync()
    """

    def __init__(
        self,
        site_url: str,
        include_subsites: bool = True,
        include_lists: bool = False,
        file_extensions: Optional[Set[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize SharePoint connector.

        Args:
            site_url: SharePoint site URL (e.g., https://contoso.sharepoint.com/sites/team)
            include_subsites: Whether to crawl subsites
            include_lists: Whether to crawl list items
            file_extensions: File extensions to index (default: common documents)
            exclude_paths: Path patterns to exclude (e.g., ["Archive/", "_private/"])
        """
        # Extract tenant and site from URL
        match = re.match(r"https://([^.]+)\.sharepoint\.com(/sites/([^/]+))?", site_url)
        if not match:
            raise ValueError(f"Invalid SharePoint URL: {site_url}")

        self.tenant = match.group(1)
        self.site_path = match.group(3) or "root"

        connector_id = f"sharepoint_{self.tenant}_{self.site_path}"
        super().__init__(connector_id=connector_id, **kwargs)

        self.site_url = site_url
        self.include_subsites = include_subsites
        self.include_lists = include_lists
        self.file_extensions = file_extensions or INDEXABLE_EXTENSIONS
        self.exclude_paths = exclude_paths or ["_catalogs/", "_private/", "Forms/"]

        # Cache
        self._access_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None
        self._sites_cache: Dict[str, SharePointSite] = {}

    @property
    def source_type(self) -> SourceType:
        return SourceType.DOCUMENT

    @property
    def name(self) -> str:
        return f"SharePoint ({self.tenant}/{self.site_path})"

    async def _get_access_token(self) -> str:
        """Get or refresh Microsoft Graph access token."""
        now = datetime.now(timezone.utc)

        if self._access_token and self._token_expires and now < self._token_expires:
            return self._access_token

        tenant_id = await self.credentials.get_credential("SHAREPOINT_TENANT_ID")
        client_id = await self.credentials.get_credential("SHAREPOINT_CLIENT_ID")
        client_secret = await self.credentials.get_credential("SHAREPOINT_CLIENT_SECRET")

        if not all([tenant_id, client_id, client_secret]):
            raise ValueError(
                "SharePoint credentials not configured. "
                "Set SHAREPOINT_TENANT_ID, SHAREPOINT_CLIENT_ID, SHAREPOINT_CLIENT_SECRET"
            )

        try:
            import httpx

            token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    token_url,
                    data={
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "scope": "https://graph.microsoft.com/.default",
                        "grant_type": "client_credentials",
                    },
                )
                response.raise_for_status()
                data = response.json()

                self._access_token = data["access_token"]
                expires_in = data.get("expires_in", 3600)
                # Add buffer before expiry (5 minutes)
                self._token_expires = now + timedelta(seconds=expires_in - 300)

                return self._access_token

        except Exception as e:
            raise RuntimeError(f"Failed to get SharePoint access token: {e}")

    async def _graph_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make a request to Microsoft Graph API."""
        import httpx

        token = await self._get_access_token()
        url = f"https://graph.microsoft.com/v1.0{endpoint}"

        async with httpx.AsyncClient() as client:
            response = await client.request(
                method,
                url,
                headers={"Authorization": f"Bearer {token}"},
                params=params,
                json=json_data,
                timeout=60,
            )
            response.raise_for_status()
            return response.json() if response.content else {}

    async def _get_site(self) -> SharePointSite:
        """Get the root site information."""
        if self.site_path == "root":
            endpoint = f"/sites/{self.tenant}.sharepoint.com"
        else:
            endpoint = f"/sites/{self.tenant}.sharepoint.com:/sites/{self.site_path}"

        data = await self._graph_request(endpoint)

        return SharePointSite(
            id=data["id"],
            name=data.get("name", ""),
            display_name=data.get("displayName", ""),
            web_url=data.get("webUrl", ""),
            created=(
                datetime.fromisoformat(data["createdDateTime"].replace("Z", "+00:00"))
                if data.get("createdDateTime")
                else None
            ),
            last_modified=(
                datetime.fromisoformat(data["lastModifiedDateTime"].replace("Z", "+00:00"))
                if data.get("lastModifiedDateTime")
                else None
            ),
        )

    async def _get_subsites(self, site_id: str) -> List[SharePointSite]:
        """Get subsites of a site."""
        if not self.include_subsites:
            return []

        try:
            data = await self._graph_request(f"/sites/{site_id}/sites")
            sites = []

            for item in data.get("value", []):
                sites.append(
                    SharePointSite(
                        id=item["id"],
                        name=item.get("name", ""),
                        display_name=item.get("displayName", ""),
                        web_url=item.get("webUrl", ""),
                    )
                )

            return sites

        except Exception as e:
            logger.warning(f"[{self.name}] Failed to get subsites: {e}")
            return []

    async def _get_drives(self, site_id: str) -> List[SharePointDrive]:
        """Get document libraries (drives) for a site."""
        data = await self._graph_request(f"/sites/{site_id}/drives")
        drives = []

        for item in data.get("value", []):
            drives.append(
                SharePointDrive(
                    id=item["id"],
                    name=item.get("name", ""),
                    drive_type=item.get("driveType", ""),
                    web_url=item.get("webUrl", ""),
                    site_id=site_id,
                )
            )

        return drives

    async def _get_drive_items(
        self,
        drive_id: str,
        folder_id: str = "root",
        delta_token: Optional[str] = None,
    ) -> AsyncIterator[tuple[SharePointItem, Optional[str]]]:
        """
        Get items from a drive, optionally using delta for incremental sync.

        Yields:
            Tuple of (item, new_delta_token)
        """
        if delta_token:
            endpoint = f"/drives/{drive_id}/root/delta"
            params = {"token": delta_token}
        else:
            endpoint = f"/drives/{drive_id}/items/{folder_id}/children"
            params = {"$top": "200"}

        new_delta_token = None

        while endpoint:
            data = await self._graph_request(endpoint, params=params)

            for item in data.get("value", []):
                if item.get("deleted"):
                    continue

                is_folder = "folder" in item
                name = item.get("name", "")
                path = item.get("parentReference", {}).get("path", "") + "/" + name

                # Check exclusions
                if any(excl in path for excl in self.exclude_paths):
                    continue

                # Skip non-indexable files
                if not is_folder:
                    ext = Path(name).suffix.lower()
                    if ext not in self.file_extensions:
                        continue

                    size = item.get("size", 0)
                    if size > MAX_FILE_SIZE:
                        logger.debug(f"[{self.name}] Skipping large file: {path} ({size} bytes)")
                        continue

                yield SharePointItem(
                    id=item["id"],
                    name=name,
                    path=path,
                    web_url=item.get("webUrl", ""),
                    size=item.get("size", 0),
                    mime_type=item.get("file", {}).get("mimeType", ""),
                    is_folder=is_folder,
                    created_by=item.get("createdBy", {}).get("user", {}).get("displayName", ""),
                    modified_by=item.get("lastModifiedBy", {})
                    .get("user", {})
                    .get("displayName", ""),
                    created_at=(
                        datetime.fromisoformat(item["createdDateTime"].replace("Z", "+00:00"))
                        if item.get("createdDateTime")
                        else None
                    ),
                    modified_at=(
                        datetime.fromisoformat(item["lastModifiedDateTime"].replace("Z", "+00:00"))
                        if item.get("lastModifiedDateTime")
                        else None
                    ),
                    etag=item.get("eTag", ""),
                ), new_delta_token

                # Recurse into folders
                if is_folder:
                    async for nested_item, _ in self._get_drive_items(drive_id, item["id"]):
                        yield nested_item, None

            # Handle pagination
            endpoint = data.get("@odata.nextLink", "")
            if endpoint:
                # Extract path from full URL
                endpoint = endpoint.replace("https://graph.microsoft.com/v1.0", "")
                params = {}

            # Capture delta token
            if "@odata.deltaLink" in data:
                delta_link = data["@odata.deltaLink"]
                # Extract token from delta link
                if "token=" in delta_link:
                    new_delta_token = delta_link.split("token=")[1].split("&")[0]

    async def _get_file_content(self, drive_id: str, item_id: str) -> str:
        """Get file content as text."""
        import httpx

        token = await self._get_access_token()
        url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{item_id}/content"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=120,
                    follow_redirects=True,
                )
                response.raise_for_status()

                # Try to decode as text
                try:
                    return response.text
                except Exception as e:
                    # Binary content, try base64
                    logger.debug(f"[{self.name}] Text decode failed, using base64: {e}")
                    return base64.b64encode(response.content).decode()[:1000] + "..."

        except Exception as e:
            logger.warning(f"[{self.name}] Failed to get file content: {e}")
            return ""

    async def sync_items(
        self,
        state: SyncState,
        batch_size: int = 100,
    ) -> AsyncIterator[SyncItem]:
        """
        Yield SharePoint items for syncing.

        Crawls document libraries and optionally lists from the site.
        """
        # Get root site
        site = await self._get_site()
        self._sites_cache[site.id] = site

        # Collect all sites to process
        sites_to_process = [site]
        if self.include_subsites:
            subsites = await self._get_subsites(site.id)
            sites_to_process.extend(subsites)
            for subsite in subsites:
                self._sites_cache[subsite.id] = subsite

        state.items_total = len(sites_to_process)
        items_yielded = 0

        for site in sites_to_process:
            # Get drives (document libraries)
            drives = await self._get_drives(site.id)

            for drive in drives:
                # Get delta token from cursor if available
                delta_token = None
                if state.cursor:
                    try:
                        cursor_data = json.loads(state.cursor)
                        delta_token = cursor_data.get(f"drive_{drive.id}")
                    except json.JSONDecodeError as e:
                        logger.debug(f"Invalid cursor JSON, starting fresh sync: {e}")

                async for item, new_delta in self._get_drive_items(
                    drive.id, delta_token=delta_token
                ):
                    if item.is_folder:
                        continue

                    # Get file content
                    content = await self._get_file_content(drive.id, item.id)
                    if not content:
                        continue

                    yield SyncItem(
                        id=f"sp-{item.id}",
                        content=content[:50000],  # Limit content size
                        source_type="document",
                        source_id=f"sharepoint/{self.tenant}/{site.name}/{item.path}",
                        title=item.name,
                        url=item.web_url,
                        author=item.modified_by or item.created_by,
                        created_at=item.created_at,
                        updated_at=item.modified_at,
                        domain="enterprise/sharepoint",
                        confidence=0.8,
                        metadata={
                            "site_id": site.id,
                            "site_name": site.name,
                            "drive_id": drive.id,
                            "drive_name": drive.name,
                            "path": item.path,
                            "mime_type": item.mime_type,
                            "size": item.size,
                            "etag": item.etag,
                        },
                    )

                    items_yielded += 1

                    # Update cursor with new delta token
                    if new_delta:
                        cursor_data = {}
                        if state.cursor:
                            try:
                                cursor_data = json.loads(state.cursor)
                            except json.JSONDecodeError as e:
                                logger.debug(f"Invalid cursor JSON during update: {e}")
                        cursor_data[f"drive_{drive.id}"] = new_delta
                        state.cursor = json.dumps(cursor_data)

                    if items_yielded >= batch_size:
                        # Yield control for batch processing
                        await asyncio.sleep(0)

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs,
    ) -> list:
        """Search SharePoint content via Microsoft Graph Search API."""
        from aragora.connectors.base import Evidence

        try:
            data = await self._graph_request(
                "/search/query",
                method="POST",
                json_data={
                    "requests": [
                        {
                            "entityTypes": ["driveItem"],
                            "query": {"queryString": query},
                            "from": 0,
                            "size": limit,
                        }
                    ]
                },
            )

            results = []
            for response in data.get("value", []):
                for hit in response.get("hitsContainers", [{}])[0].get("hits", []):
                    resource = hit.get("resource", {})
                    results.append(
                        Evidence(
                            id=f"sp-search-{resource.get('id', '')}",
                            source_type=self.source_type,
                            source_id=resource.get("webUrl", ""),
                            content=hit.get("summary", resource.get("name", "")),
                            title=resource.get("name", ""),
                            url=resource.get("webUrl", ""),
                            confidence=0.8,
                            metadata={
                                "rank": hit.get("rank", 0),
                                "size": resource.get("size", 0),
                            },
                        )
                    )

            return results

        except Exception as e:
            logger.error(f"[{self.name}] Search failed: {e}")
            return []

    async def fetch(self, evidence_id: str) -> Optional[Any]:
        """Fetch a specific SharePoint item."""
        from aragora.connectors.base import Evidence

        # Extract item ID from evidence_id
        if evidence_id.startswith("sp-"):
            item_id = evidence_id[3:]
        else:
            item_id = evidence_id

        try:
            # Search for the item across drives
            site = await self._get_site()
            drives = await self._get_drives(site.id)

            for drive in drives:
                try:
                    data = await self._graph_request(f"/drives/{drive.id}/items/{item_id}")
                    content = await self._get_file_content(drive.id, item_id)

                    return Evidence(
                        id=evidence_id,
                        source_type=self.source_type,
                        source_id=data.get("webUrl", ""),
                        content=content[:50000],
                        title=data.get("name", ""),
                        url=data.get("webUrl", ""),
                        author=data.get("lastModifiedBy", {})
                        .get("user", {})
                        .get("displayName", ""),
                        created_at=data.get("createdDateTime"),
                        confidence=0.8,
                    )
                except Exception as e:
                    logger.debug(f"[{self.name}] Failed to create fetch result: {e}")
                    continue

            return None

        except Exception as e:
            logger.error(f"[{self.name}] Fetch failed: {e}")
            return None

    async def handle_webhook(self, payload: Dict[str, Any]) -> bool:
        """Handle SharePoint webhook notification."""
        # SharePoint webhooks send subscription validation
        if "validationToken" in payload:
            logger.info(f"[{self.name}] Webhook validation request")
            return True

        # Process change notifications
        for notification in payload.get("value", []):
            resource = notification.get("resource", "")
            change_type = notification.get("changeType", "")

            logger.info(f"[{self.name}] Webhook: {change_type} on {resource}")

            # Trigger incremental sync
            asyncio.create_task(self.sync(max_items=50))

        return True


__all__ = ["SharePointConnector", "SharePointSite", "SharePointDrive", "SharePointItem"]
