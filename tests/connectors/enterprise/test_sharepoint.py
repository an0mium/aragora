"""
Tests for SharePoint Enterprise Connector.

Tests the Microsoft SharePoint Online integration including:
- Document library crawling and indexing
- Site/subsite traversal
- List item extraction
- Incremental sync via delta tokens
- Webhook support for real-time updates
"""

import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.enterprise.base import SyncState, SyncStatus
from aragora.connectors.enterprise.documents.sharepoint import (
    SharePointConnector,
    SharePointSite,
    SharePointDrive,
    SharePointItem,
    INDEXABLE_EXTENSIONS,
    MAX_FILE_SIZE,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_credentials():
    """Mock credential provider with Azure AD credentials."""
    from tests.connectors.enterprise.conftest import MockCredentialProvider

    return MockCredentialProvider(
        {
            "SHAREPOINT_TENANT_ID": "tenant-123-456",
            "SHAREPOINT_CLIENT_ID": "client-abc-def",
            "SHAREPOINT_CLIENT_SECRET": "secret_value_xyz",
        }
    )


@pytest.fixture
def sharepoint_connector(mock_credentials, tmp_path):
    """Create a SharePoint connector for testing."""
    return SharePointConnector(
        site_url="https://contoso.sharepoint.com/sites/engineering",
        include_subsites=True,
        include_lists=False,
        credentials=mock_credentials,
        state_dir=tmp_path / "sync_state",
    )


@pytest.fixture
def sample_site():
    """Sample SharePoint site data."""
    return {
        "id": "site-123-456",
        "name": "engineering",
        "displayName": "Engineering Team",
        "webUrl": "https://contoso.sharepoint.com/sites/engineering",
        "createdDateTime": "2024-01-01T10:00:00Z",
        "lastModifiedDateTime": "2024-01-15T14:30:00Z",
    }


@pytest.fixture
def sample_subsites():
    """Sample subsites data."""
    return [
        {
            "id": "subsite-001",
            "name": "frontend",
            "displayName": "Frontend Team",
            "webUrl": "https://contoso.sharepoint.com/sites/engineering/frontend",
        },
        {
            "id": "subsite-002",
            "name": "backend",
            "displayName": "Backend Team",
            "webUrl": "https://contoso.sharepoint.com/sites/engineering/backend",
        },
    ]


@pytest.fixture
def sample_drives():
    """Sample document libraries (drives)."""
    return [
        {
            "id": "drive-001",
            "name": "Documents",
            "driveType": "documentLibrary",
            "webUrl": "https://contoso.sharepoint.com/sites/engineering/Documents",
        },
        {
            "id": "drive-002",
            "name": "Shared Assets",
            "driveType": "documentLibrary",
            "webUrl": "https://contoso.sharepoint.com/sites/engineering/SharedAssets",
        },
    ]


@pytest.fixture
def sample_drive_items():
    """Sample drive items."""
    return [
        {
            "id": "item-001",
            "name": "Project Spec.docx",
            "webUrl": "https://contoso.sharepoint.com/sites/engineering/Documents/Project%20Spec.docx",
            "size": 50000,
            "file": {
                "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            },
            "parentReference": {"path": "/drives/drive-001/root:"},
            "createdBy": {"user": {"displayName": "Alice Smith"}},
            "lastModifiedBy": {"user": {"displayName": "Bob Jones"}},
            "createdDateTime": "2024-01-10T09:00:00Z",
            "lastModifiedDateTime": "2024-01-16T11:30:00Z",
            "eTag": '"abc123"',
        },
        {
            "id": "folder-001",
            "name": "Archive",
            "webUrl": "https://contoso.sharepoint.com/sites/engineering/Documents/Archive",
            "folder": {},
            "parentReference": {"path": "/drives/drive-001/root:"},
            "createdDateTime": "2024-01-05T08:00:00Z",
            "lastModifiedDateTime": "2024-01-05T08:00:00Z",
        },
        {
            "id": "item-002",
            "name": "data.csv",
            "webUrl": "https://contoso.sharepoint.com/sites/engineering/Documents/data.csv",
            "size": 25000,
            "file": {"mimeType": "text/csv"},
            "parentReference": {"path": "/drives/drive-001/root:"},
            "createdBy": {"user": {"displayName": "Charlie Brown"}},
            "lastModifiedBy": {"user": {"displayName": "Charlie Brown"}},
            "createdDateTime": "2024-01-12T10:00:00Z",
            "lastModifiedDateTime": "2024-01-17T09:00:00Z",
            "eTag": '"def456"',
        },
    ]


# =============================================================================
# Initialization Tests
# =============================================================================


class TestSharePointConnectorInit:
    """Test SharePointConnector initialization."""

    def test_init_with_site_url(self, mock_credentials, tmp_path):
        """Test initialization with valid site URL."""
        connector = SharePointConnector(
            site_url="https://contoso.sharepoint.com/sites/engineering",
            credentials=mock_credentials,
            state_dir=tmp_path,
        )
        assert connector.tenant == "contoso"
        assert connector.site_path == "engineering"
        assert connector.connector_id == "sharepoint_contoso_engineering"

    def test_init_with_root_url(self, mock_credentials, tmp_path):
        """Test initialization with root SharePoint URL."""
        connector = SharePointConnector(
            site_url="https://contoso.sharepoint.com",
            credentials=mock_credentials,
            state_dir=tmp_path,
        )
        assert connector.tenant == "contoso"
        assert connector.site_path == "root"

    def test_init_invalid_url(self, mock_credentials, tmp_path):
        """Test initialization fails with invalid URL."""
        with pytest.raises(ValueError, match="Invalid SharePoint URL"):
            SharePointConnector(
                site_url="https://invalid-url.com/sites/test",
                credentials=mock_credentials,
                state_dir=tmp_path,
            )

    def test_init_with_custom_options(self, mock_credentials, tmp_path):
        """Test initialization with custom options."""
        custom_extensions = {".pdf", ".docx"}
        custom_excludes = ["Archive/", "Draft/"]

        connector = SharePointConnector(
            site_url="https://contoso.sharepoint.com/sites/team",
            include_subsites=False,
            include_lists=True,
            file_extensions=custom_extensions,
            exclude_paths=custom_excludes,
            credentials=mock_credentials,
            state_dir=tmp_path,
        )

        assert connector.include_subsites is False
        assert connector.include_lists is True
        assert connector.file_extensions == custom_extensions
        assert connector.exclude_paths == custom_excludes

    def test_default_file_extensions(self, sharepoint_connector):
        """Test default file extensions include common documents."""
        extensions = sharepoint_connector.file_extensions
        assert ".docx" in extensions
        assert ".pdf" in extensions
        assert ".xlsx" in extensions
        assert ".md" in extensions

    def test_default_exclude_paths(self, sharepoint_connector):
        """Test default exclude paths."""
        excludes = sharepoint_connector.exclude_paths
        assert "_catalogs/" in excludes
        assert "_private/" in excludes
        assert "Forms/" in excludes

    def test_source_type(self, sharepoint_connector):
        """Test source type property."""
        from aragora.reasoning.provenance import SourceType

        assert sharepoint_connector.source_type == SourceType.DOCUMENT

    def test_name_property(self, sharepoint_connector):
        """Test name property."""
        assert sharepoint_connector.name == "SharePoint (contoso/engineering)"


# =============================================================================
# Access Token Tests
# =============================================================================


class TestAccessToken:
    """Test Azure AD access token management."""

    @pytest.mark.asyncio
    async def test_get_access_token_fresh(self, sharepoint_connector):
        """Test getting a fresh access token."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "new_access_token",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            token = await sharepoint_connector._get_access_token()

            assert token == "new_access_token"
            assert sharepoint_connector._access_token == "new_access_token"
            assert sharepoint_connector._token_expires is not None

    @pytest.mark.asyncio
    async def test_get_access_token_cached(self, sharepoint_connector):
        """Test using cached access token."""
        sharepoint_connector._access_token = "cached_token"
        sharepoint_connector._token_expires = datetime.now(timezone.utc) + timedelta(hours=1)

        token = await sharepoint_connector._get_access_token()

        assert token == "cached_token"

    @pytest.mark.asyncio
    async def test_get_access_token_expired(self, sharepoint_connector):
        """Test refreshing expired access token."""
        sharepoint_connector._access_token = "expired_token"
        sharepoint_connector._token_expires = datetime.now(timezone.utc) - timedelta(hours=1)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "refreshed_token",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            token = await sharepoint_connector._get_access_token()

            assert token == "refreshed_token"

    @pytest.mark.asyncio
    async def test_get_access_token_missing_credentials(self, tmp_path):
        """Test error when credentials are missing."""
        from tests.connectors.enterprise.conftest import MockCredentialProvider

        empty_credentials = MockCredentialProvider({})

        connector = SharePointConnector(
            site_url="https://contoso.sharepoint.com/sites/test",
            credentials=empty_credentials,
            state_dir=tmp_path,
        )

        with pytest.raises(ValueError, match="credentials not configured"):
            await connector._get_access_token()


# =============================================================================
# Graph API Tests
# =============================================================================


class TestGraphApiRequest:
    """Test Microsoft Graph API requests."""

    @pytest.mark.asyncio
    async def test_graph_request_success(self, sharepoint_connector):
        """Test successful Graph API request."""
        sharepoint_connector._access_token = "test_token"
        sharepoint_connector._token_expires = datetime.now(timezone.utc) + timedelta(hours=1)

        mock_response = MagicMock()
        mock_response.json.return_value = {"value": []}
        mock_response.content = b'{"value": []}'
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            result = await sharepoint_connector._graph_request("/sites")

            assert result == {"value": []}
            mock_client.request.assert_called_once()

    @pytest.mark.asyncio
    async def test_graph_request_with_params(self, sharepoint_connector):
        """Test Graph API request with parameters."""
        sharepoint_connector._access_token = "test_token"
        sharepoint_connector._token_expires = datetime.now(timezone.utc) + timedelta(hours=1)

        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.content = b"{}"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            await sharepoint_connector._graph_request(
                "/sites/site-id/drives",
                params={"$top": "100"},
            )

            call_kwargs = mock_client.request.call_args[1]
            assert call_kwargs["params"]["$top"] == "100"


# =============================================================================
# Site Retrieval Tests
# =============================================================================


class TestSiteRetrieval:
    """Test site retrieval functionality."""

    @pytest.mark.asyncio
    async def test_get_site_success(self, sharepoint_connector, sample_site):
        """Test getting root site information."""
        sharepoint_connector._access_token = "test_token"
        sharepoint_connector._token_expires = datetime.now(timezone.utc) + timedelta(hours=1)

        with patch.object(sharepoint_connector, "_graph_request", return_value=sample_site):
            site = await sharepoint_connector._get_site()

            assert site.id == "site-123-456"
            assert site.name == "engineering"
            assert site.display_name == "Engineering Team"
            assert site.created is not None
            assert site.last_modified is not None

    @pytest.mark.asyncio
    async def test_get_subsites(self, sharepoint_connector, sample_subsites):
        """Test getting subsites."""
        sharepoint_connector._access_token = "test_token"
        sharepoint_connector._token_expires = datetime.now(timezone.utc) + timedelta(hours=1)

        with patch.object(
            sharepoint_connector,
            "_graph_request",
            return_value={"value": sample_subsites},
        ):
            subsites = await sharepoint_connector._get_subsites("site-123")

            assert len(subsites) == 2
            assert subsites[0].id == "subsite-001"
            assert subsites[0].name == "frontend"
            assert subsites[1].name == "backend"

    @pytest.mark.asyncio
    async def test_get_subsites_disabled(self, mock_credentials, tmp_path):
        """Test subsites are not fetched when disabled."""
        connector = SharePointConnector(
            site_url="https://contoso.sharepoint.com/sites/test",
            include_subsites=False,
            credentials=mock_credentials,
            state_dir=tmp_path,
        )

        subsites = await connector._get_subsites("site-123")
        assert subsites == []

    @pytest.mark.asyncio
    async def test_get_subsites_error_handling(self, sharepoint_connector):
        """Test subsite retrieval error handling."""
        sharepoint_connector._access_token = "test_token"
        sharepoint_connector._token_expires = datetime.now(timezone.utc) + timedelta(hours=1)

        with patch.object(
            sharepoint_connector,
            "_graph_request",
            side_effect=Exception("API Error"),
        ):
            subsites = await sharepoint_connector._get_subsites("site-123")
            assert subsites == []


# =============================================================================
# Drive Retrieval Tests
# =============================================================================


class TestDriveRetrieval:
    """Test document library (drive) retrieval."""

    @pytest.mark.asyncio
    async def test_get_drives(self, sharepoint_connector, sample_drives):
        """Test getting document libraries."""
        sharepoint_connector._access_token = "test_token"
        sharepoint_connector._token_expires = datetime.now(timezone.utc) + timedelta(hours=1)

        with patch.object(
            sharepoint_connector,
            "_graph_request",
            return_value={"value": sample_drives},
        ):
            drives = await sharepoint_connector._get_drives("site-123")

            assert len(drives) == 2
            assert drives[0].id == "drive-001"
            assert drives[0].name == "Documents"
            assert drives[0].drive_type == "documentLibrary"
            assert drives[1].name == "Shared Assets"


# =============================================================================
# Drive Items Tests
# =============================================================================


class TestDriveItems:
    """Test drive item retrieval."""

    @pytest.mark.asyncio
    async def test_get_drive_items(self, sharepoint_connector, sample_drive_items):
        """Test getting drive items."""
        sharepoint_connector._access_token = "test_token"
        sharepoint_connector._token_expires = datetime.now(timezone.utc) + timedelta(hours=1)

        # Only include files (no folders) to avoid recursion in tests
        files_only = [item for item in sample_drive_items if "folder" not in item]

        with patch.object(
            sharepoint_connector,
            "_graph_request",
            return_value={"value": files_only},
        ):
            items = []
            async for item, delta in sharepoint_connector._get_drive_items("drive-001"):
                items.append(item)

            # Should get the file items
            assert len(items) == 2  # Project Spec.docx and data.csv

    @pytest.mark.asyncio
    async def test_get_drive_items_skips_excluded_paths(self, sharepoint_connector):
        """Test excluded paths are skipped."""
        items_with_excluded = [
            {
                "id": "item-excluded",
                "name": "catalog.xml",
                "file": {"mimeType": "text/xml"},
                "parentReference": {"path": "/drives/drive-001/root:/_catalogs"},
                "size": 1000,
            },
        ]

        with patch.object(
            sharepoint_connector,
            "_graph_request",
            return_value={"value": items_with_excluded},
        ):
            items = []
            async for item, delta in sharepoint_connector._get_drive_items("drive-001"):
                items.append(item)

            assert len(items) == 0  # Should be excluded

    @pytest.mark.asyncio
    async def test_get_drive_items_skips_non_indexable(self, sharepoint_connector):
        """Test non-indexable file types are skipped."""
        items_with_binary = [
            {
                "id": "item-binary",
                "name": "image.png",
                "file": {"mimeType": "image/png"},
                "parentReference": {"path": "/drives/drive-001/root:"},
                "size": 5000,
            },
        ]

        with patch.object(
            sharepoint_connector,
            "_graph_request",
            return_value={"value": items_with_binary},
        ):
            items = []
            async for item, delta in sharepoint_connector._get_drive_items("drive-001"):
                items.append(item)

            assert len(items) == 0  # .png not in INDEXABLE_EXTENSIONS

    @pytest.mark.asyncio
    async def test_get_drive_items_skips_large_files(self, sharepoint_connector):
        """Test large files are skipped."""
        large_file = [
            {
                "id": "item-large",
                "name": "huge.docx",
                "file": {
                    "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                },
                "parentReference": {"path": "/drives/drive-001/root:"},
                "size": MAX_FILE_SIZE + 1,
            },
        ]

        with patch.object(
            sharepoint_connector,
            "_graph_request",
            return_value={"value": large_file},
        ):
            items = []
            async for item, delta in sharepoint_connector._get_drive_items("drive-001"):
                items.append(item)

            assert len(items) == 0

    @pytest.mark.asyncio
    async def test_get_drive_items_with_delta_token(self, sharepoint_connector):
        """Test incremental sync with delta token."""
        sharepoint_connector._access_token = "test_token"
        sharepoint_connector._token_expires = datetime.now(timezone.utc) + timedelta(hours=1)

        response_with_delta = {
            "value": [
                {
                    "id": "item-new",
                    "name": "new_file.txt",
                    "file": {"mimeType": "text/plain"},
                    "parentReference": {"path": "/drives/drive-001/root:"},
                    "size": 1000,
                    "createdDateTime": "2024-01-18T10:00:00Z",
                    "lastModifiedDateTime": "2024-01-18T10:00:00Z",
                },
            ],
            "@odata.deltaLink": "https://graph.microsoft.com/v1.0/drives/drive-001/root/delta?token=new_token_xyz",
        }

        with patch.object(
            sharepoint_connector,
            "_graph_request",
            return_value=response_with_delta,
        ):
            items = []
            async for item, delta in sharepoint_connector._get_drive_items(
                "drive-001", delta_token="old_token"
            ):
                items.append(item)

            assert len(items) == 1
            assert items[0].name == "new_file.txt"


# =============================================================================
# File Content Tests
# =============================================================================


class TestFileContent:
    """Test file content retrieval."""

    @pytest.mark.asyncio
    async def test_get_file_content_text(self, sharepoint_connector):
        """Test getting text file content."""
        sharepoint_connector._access_token = "test_token"
        sharepoint_connector._token_expires = datetime.now(timezone.utc) + timedelta(hours=1)

        mock_response = MagicMock()
        mock_response.text = "File text content here"
        mock_response.content = b"File text content here"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            content = await sharepoint_connector._get_file_content("drive-001", "item-001")

            assert content == "File text content here"

    @pytest.mark.asyncio
    async def test_get_file_content_error(self, sharepoint_connector):
        """Test file content error handling."""
        sharepoint_connector._access_token = "test_token"
        sharepoint_connector._token_expires = datetime.now(timezone.utc) + timedelta(hours=1)

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Download failed")
            MockClient.return_value.__aenter__.return_value = mock_client

            content = await sharepoint_connector._get_file_content("drive-001", "item-001")

            assert content == ""


# =============================================================================
# Sync Tests
# =============================================================================


class TestSync:
    """Test sync_items functionality."""

    @pytest.mark.asyncio
    async def test_sync_items_full(self, sharepoint_connector, sample_site):
        """Test full sync."""
        state = SyncState(connector_id="sharepoint", status=SyncStatus.IDLE)

        call_count = {"children": 0}

        # Mock the API calls - return empty after first call to avoid recursion
        async def mock_graph_request(endpoint, **kwargs):
            # Match _get_site() endpoint pattern: /sites/{tenant}.sharepoint.com or /sites/{tenant}.sharepoint.com:/sites/{path}
            if (
                endpoint.startswith("/sites/")
                and "/drives" not in endpoint
                and "/sites/" not in endpoint[7:]
            ):
                return sample_site
            elif "/drives" in endpoint and "/items/" not in endpoint:
                return {
                    "value": [
                        {
                            "id": "drive-001",
                            "name": "Documents",
                            "driveType": "documentLibrary",
                            "webUrl": "",
                        }
                    ]
                }
            elif "/sites/" in endpoint and endpoint.endswith("/sites"):
                return {"value": []}  # No subsites (endpoint like /sites/{id}/sites)
            elif "/children" in endpoint:
                call_count["children"] += 1
                if call_count["children"] == 1:
                    return {
                        "value": [
                            {
                                "id": "item-001",
                                "name": "test.txt",
                                "file": {"mimeType": "text/plain"},
                                "parentReference": {"path": "/drives/drive-001/root:"},
                                "size": 500,
                                "webUrl": "https://contoso.sharepoint.com/test.txt",
                                "createdBy": {"user": {"displayName": "Alice"}},
                                "lastModifiedBy": {"user": {"displayName": "Bob"}},
                                "createdDateTime": "2024-01-15T10:00:00Z",
                                "lastModifiedDateTime": "2024-01-16T14:00:00Z",
                            },
                        ]
                    }
                return {"value": []}  # Empty for subsequent calls
            return sample_site  # Default to sample_site for unknown endpoints

        with patch.object(sharepoint_connector, "_graph_request", side_effect=mock_graph_request):
            with patch.object(
                sharepoint_connector,
                "_get_file_content",
                new_callable=AsyncMock,
                return_value="File content",
            ):
                items = []
                async for item in sharepoint_connector.sync_items(state, batch_size=10):
                    items.append(item)

        # Should yield items from drives
        assert len(items) >= 1
        if items:
            assert items[0].source_type == "document"
            assert items[0].domain == "enterprise/sharepoint"

    @pytest.mark.asyncio
    async def test_sync_items_metadata(self, sharepoint_connector, sample_site):
        """Test sync item metadata is correct."""
        state = SyncState(connector_id="sharepoint", status=SyncStatus.IDLE)

        test_item = {
            "id": "item-meta",
            "name": "metadata_test.docx",
            "file": {
                "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            },
            "parentReference": {"path": "/drives/drive-001/root:/Documents"},
            "size": 10000,
            "webUrl": "https://contoso.sharepoint.com/sites/engineering/Documents/metadata_test.docx",
            "createdBy": {"user": {"displayName": "Author Name"}},
            "lastModifiedBy": {"user": {"displayName": "Editor Name"}},
            "createdDateTime": "2024-01-10T09:00:00Z",
            "lastModifiedDateTime": "2024-01-17T15:00:00Z",
            "eTag": '"etag123"',
        }

        call_count = {"children": 0}

        async def mock_graph_request(endpoint, **kwargs):
            # Match _get_site() endpoint pattern
            if (
                endpoint.startswith("/sites/")
                and "/drives" not in endpoint
                and "/sites/" not in endpoint[7:]
            ):
                return sample_site
            elif "/drives" in endpoint and "/items/" not in endpoint:
                return {
                    "value": [
                        {
                            "id": "drive-001",
                            "name": "Docs",
                            "driveType": "documentLibrary",
                            "webUrl": "",
                        }
                    ]
                }
            elif "/sites/" in endpoint and endpoint.endswith("/sites"):
                return {"value": []}  # No subsites
            elif "/children" in endpoint:
                call_count["children"] += 1
                if call_count["children"] == 1:
                    return {"value": [test_item]}
                return {"value": []}
            return sample_site

        with patch.object(sharepoint_connector, "_graph_request", side_effect=mock_graph_request):
            with patch.object(
                sharepoint_connector,
                "_get_file_content",
                new_callable=AsyncMock,
                return_value="Content",
            ):
                items = []
                async for item in sharepoint_connector.sync_items(state, batch_size=10):
                    items.append(item)

        assert len(items) >= 1
        item = items[0]
        assert item.id == "sp-item-meta"
        assert item.title == "metadata_test.docx"
        assert "site_id" in item.metadata
        assert "drive_id" in item.metadata
        assert (
            item.metadata.get("mime_type")
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )


# =============================================================================
# Search Tests
# =============================================================================


class TestSearch:
    """Test search functionality."""

    @pytest.mark.asyncio
    async def test_search_success(self, sharepoint_connector):
        """Test successful search."""
        search_response = {
            "value": [
                {
                    "hitsContainers": [
                        {
                            "hits": [
                                {
                                    "rank": 1,
                                    "summary": "Found in quarterly report...",
                                    "resource": {
                                        "id": "item-search-001",
                                        "name": "Q4 Report.docx",
                                        "webUrl": "https://contoso.sharepoint.com/Q4Report.docx",
                                        "size": 50000,
                                    },
                                },
                            ]
                        }
                    ]
                }
            ]
        }

        with patch.object(sharepoint_connector, "_graph_request", return_value=search_response):
            results = await sharepoint_connector.search("quarterly report", limit=5)

            assert len(results) == 1
            assert results[0].title == "Q4 Report.docx"
            assert results[0].content == "Found in quarterly report..."

    @pytest.mark.asyncio
    async def test_search_error_handling(self, sharepoint_connector):
        """Test search error handling."""
        with patch.object(
            sharepoint_connector,
            "_graph_request",
            side_effect=Exception("Search API Error"),
        ):
            results = await sharepoint_connector.search("test query")
            assert results == []


# =============================================================================
# Fetch Tests
# =============================================================================


class TestFetch:
    """Test fetch functionality."""

    @pytest.mark.asyncio
    async def test_fetch_success(self, sharepoint_connector, sample_site):
        """Test successful item fetch."""
        item_data = {
            "id": "item-fetch-001",
            "name": "fetched_doc.docx",
            "webUrl": "https://contoso.sharepoint.com/fetched_doc.docx",
            "lastModifiedBy": {"user": {"displayName": "Fetcher"}},
            "createdDateTime": "2024-01-15T10:00:00Z",
        }

        drives_data = [
            {"id": "drive-001", "name": "Documents", "driveType": "documentLibrary", "webUrl": ""}
        ]

        async def mock_graph_request(endpoint, **kwargs):
            # Match _get_site() endpoint pattern
            if (
                endpoint.startswith("/sites/")
                and "/drives" not in endpoint
                and "/sites/" not in endpoint[7:]
            ):
                return sample_site
            elif "/drives" in endpoint and "/items/" not in endpoint:
                return {"value": drives_data}
            elif "/items/" in endpoint:
                return item_data
            return sample_site

        with patch.object(sharepoint_connector, "_graph_request", side_effect=mock_graph_request):
            with patch.object(
                sharepoint_connector,
                "_get_file_content",
                new_callable=AsyncMock,
                return_value="Fetched content",
            ):
                evidence = await sharepoint_connector.fetch("sp-item-fetch-001")

                assert evidence is not None
                assert evidence.id == "sp-item-fetch-001"
                assert evidence.title == "fetched_doc.docx"
                assert evidence.content == "Fetched content"

    @pytest.mark.asyncio
    async def test_fetch_with_raw_id(self, sharepoint_connector, sample_site):
        """Test fetch with raw item ID (no prefix)."""
        item_data = {
            "id": "rawitem123",
            "name": "raw.txt",
            "webUrl": "https://contoso.sharepoint.com/raw.txt",
            "lastModifiedBy": {"user": {"displayName": "User"}},
        }

        drives_data = [
            {"id": "drive-001", "name": "Documents", "driveType": "documentLibrary", "webUrl": ""}
        ]

        async def mock_graph_request(endpoint, **kwargs):
            # Match _get_site() endpoint pattern
            if (
                endpoint.startswith("/sites/")
                and "/drives" not in endpoint
                and "/sites/" not in endpoint[7:]
            ):
                return sample_site
            elif "/drives" in endpoint and "/items/" not in endpoint:
                return {"value": drives_data}
            elif "/items/" in endpoint:
                return item_data
            return sample_site

        with patch.object(sharepoint_connector, "_graph_request", side_effect=mock_graph_request):
            with patch.object(
                sharepoint_connector,
                "_get_file_content",
                new_callable=AsyncMock,
                return_value="Content",
            ):
                evidence = await sharepoint_connector.fetch("rawitem123")
                assert evidence is not None

    @pytest.mark.asyncio
    async def test_fetch_not_found(self, sharepoint_connector, sample_site):
        """Test fetch when item not found."""
        drives_data = [
            {"id": "drive-001", "name": "Documents", "driveType": "documentLibrary", "webUrl": ""}
        ]

        async def mock_graph_request(endpoint, **kwargs):
            # Match _get_site() endpoint pattern
            if (
                endpoint.startswith("/sites/")
                and "/drives" not in endpoint
                and "/sites/" not in endpoint[7:]
            ):
                return sample_site
            elif "/drives" in endpoint and "/items/" not in endpoint:
                return {"value": drives_data}
            elif "/items/" in endpoint:
                raise Exception("Item not found")
            return sample_site

        with patch.object(sharepoint_connector, "_graph_request", side_effect=mock_graph_request):
            evidence = await sharepoint_connector.fetch("nonexistent")
            assert evidence is None


# =============================================================================
# Webhook Tests
# =============================================================================


class TestWebhook:
    """Test webhook handling."""

    @pytest.mark.asyncio
    async def test_handle_webhook_validation(self, sharepoint_connector):
        """Test webhook validation request."""
        payload = {"validationToken": "token_abc_123"}

        result = await sharepoint_connector.handle_webhook(payload)
        assert result is True

    @pytest.mark.asyncio
    async def test_handle_webhook_change_notification(self, sharepoint_connector):
        """Test handling change notification."""
        payload = {
            "value": [
                {
                    "resource": "drives/drive-001/items/item-001",
                    "changeType": "updated",
                }
            ]
        }

        with patch.object(sharepoint_connector, "sync", new_callable=AsyncMock):
            result = await sharepoint_connector.handle_webhook(payload)
            assert result is True

    @pytest.mark.asyncio
    async def test_handle_webhook_multiple_notifications(self, sharepoint_connector):
        """Test handling multiple notifications."""
        payload = {
            "value": [
                {"resource": "item1", "changeType": "created"},
                {"resource": "item2", "changeType": "updated"},
                {"resource": "item3", "changeType": "deleted"},
            ]
        }

        with patch.object(sharepoint_connector, "sync", new_callable=AsyncMock):
            result = await sharepoint_connector.handle_webhook(payload)
            assert result is True


# =============================================================================
# Data Classes Tests
# =============================================================================


class TestDataClasses:
    """Test SharePoint data classes."""

    def test_sharepoint_site_creation(self):
        """Test SharePointSite dataclass."""
        site = SharePointSite(
            id="site-123",
            name="engineering",
            display_name="Engineering Team",
            web_url="https://contoso.sharepoint.com/sites/engineering",
            created=datetime(2024, 1, 1, tzinfo=timezone.utc),
            last_modified=datetime(2024, 1, 15, tzinfo=timezone.utc),
        )

        assert site.id == "site-123"
        assert site.display_name == "Engineering Team"
        assert site.created is not None

    def test_sharepoint_drive_creation(self):
        """Test SharePointDrive dataclass."""
        drive = SharePointDrive(
            id="drive-123",
            name="Documents",
            drive_type="documentLibrary",
            web_url="https://contoso.sharepoint.com/sites/eng/Documents",
            site_id="site-123",
        )

        assert drive.id == "drive-123"
        assert drive.drive_type == "documentLibrary"

    def test_sharepoint_item_creation(self):
        """Test SharePointItem dataclass."""
        item = SharePointItem(
            id="item-123",
            name="test.docx",
            path="/Documents/test.docx",
            web_url="https://contoso.sharepoint.com/Documents/test.docx",
            size=50000,
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            is_folder=False,
            created_by="Alice",
            modified_by="Bob",
        )

        assert item.id == "item-123"
        assert item.is_folder is False
        assert item.size == 50000


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Test module constants."""

    def test_indexable_extensions(self):
        """Test indexable extensions include common document types."""
        assert ".docx" in INDEXABLE_EXTENSIONS
        assert ".pdf" in INDEXABLE_EXTENSIONS
        assert ".xlsx" in INDEXABLE_EXTENSIONS
        assert ".txt" in INDEXABLE_EXTENSIONS
        assert ".md" in INDEXABLE_EXTENSIONS
        assert ".json" in INDEXABLE_EXTENSIONS

    def test_max_file_size(self):
        """Test maximum file size is reasonable."""
        assert MAX_FILE_SIZE == 10 * 1024 * 1024  # 10MB
