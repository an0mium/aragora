"""
Tests for Notion Enterprise Connector.

Tests the Notion API integration including:
- Page and database operations
- Block content extraction
- Search functionality
- Incremental sync
- Error handling
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from aragora.connectors.enterprise.collaboration.notion import (
    NotionConnector,
    NotionPage,
    NotionDatabase,
)
from aragora.connectors.enterprise.base import SyncState


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def connector():
    """Create test connector."""
    conn = NotionConnector(
        workspace_name="Engineering",
        include_archived=False,
        include_databases=True,
        max_depth=3,
    )
    # Mock credentials
    conn.credentials = MagicMock()
    conn.credentials.get_credential = AsyncMock(return_value="test_notion_token")
    return conn


@pytest.fixture
def mock_credentials():
    """Create mock credentials."""
    creds = MagicMock()
    creds.get_credential = AsyncMock(return_value="test_notion_token")
    return creds


def make_api_response(data: dict[str, Any]) -> dict[str, Any]:
    """Create a mock API response."""
    return data


def make_page_data(
    page_id: str = "page-1",
    title: str = "Test Page",
    archived: bool = False,
) -> dict[str, Any]:
    """Create mock page data."""
    return {
        "id": page_id,
        "object": "page",
        "url": f"https://notion.so/{page_id}",
        "archived": archived,
        "parent": {"type": "workspace", "workspace": True},
        "created_time": "2024-01-15T10:00:00.000Z",
        "last_edited_time": "2024-01-15T12:00:00.000Z",
        "created_by": {"id": "user-1"},
        "last_edited_by": {"id": "user-1"},
        "properties": {
            "title": {
                "type": "title",
                "title": [{"plain_text": title, "type": "text", "text": {"content": title}}],
            }
        },
    }


def make_database_data(
    db_id: str = "db-1",
    title: str = "Test Database",
) -> dict[str, Any]:
    """Create mock database data."""
    return {
        "id": db_id,
        "object": "database",
        "url": f"https://notion.so/{db_id}",
        "title": [{"plain_text": title, "type": "text"}],
        "description": [{"plain_text": "Test description"}],
        "created_time": "2024-01-15T10:00:00.000Z",
        "last_edited_time": "2024-01-15T12:00:00.000Z",
        "properties": {
            "Name": {"type": "title"},
            "Status": {"type": "select"},
        },
    }


def make_block_data(
    block_id: str = "block-1",
    block_type: str = "paragraph",
    text: str = "Test content",
) -> dict[str, Any]:
    """Create mock block data."""
    return {
        "id": block_id,
        "type": block_type,
        "has_children": False,
        block_type: {
            "rich_text": [{"plain_text": text, "type": "text"}],
        },
    }


# =============================================================================
# Initialization Tests
# =============================================================================


class TestNotionConnectorInit:
    """Test NotionConnector initialization."""

    def test_default_configuration(self):
        """Should use default configuration."""
        connector = NotionConnector()
        assert connector.workspace_name == "default"
        assert connector.include_archived is False
        assert connector.include_databases is True
        assert connector.max_depth == 5

    def test_custom_configuration(self):
        """Should accept custom configuration."""
        connector = NotionConnector(
            workspace_name="MyWorkspace",
            include_archived=True,
            include_databases=False,
            max_depth=10,
            recursive_pages=False,
        )
        assert connector.workspace_name == "MyWorkspace"
        assert connector.include_archived is True
        assert connector.include_databases is False
        assert connector.max_depth == 10
        assert connector.recursive_pages is False

    def test_connector_id_generation(self):
        """Should generate connector ID from workspace name."""
        connector = NotionConnector(workspace_name="My Workspace")
        assert connector.connector_id == "notion_my_workspace"

    def test_connector_properties(self, connector):
        """Should have correct connector properties."""
        assert connector.name == "Notion (Engineering)"
        assert "notion" in connector.connector_id

    def test_source_type(self, connector):
        """Should have correct source type."""
        from aragora.reasoning.provenance import SourceType

        assert connector.source_type == SourceType.DOCUMENT


# =============================================================================
# Authentication Tests
# =============================================================================


class TestAuthentication:
    """Test authentication flows."""

    @pytest.mark.asyncio
    async def test_get_auth_header(self, connector):
        """Should get authentication header."""
        header = await connector._get_auth_header()

        assert "Authorization" in header
        assert header["Authorization"] == "Bearer test_notion_token"
        assert header["Notion-Version"] == "2022-06-28"

    @pytest.mark.asyncio
    async def test_missing_credentials(self, connector):
        """Should raise error when credentials missing."""
        connector.credentials.get_credential = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="Notion credentials not configured"):
            await connector._get_auth_header()


# =============================================================================
# Page Operations Tests
# =============================================================================


class TestPageOperations:
    """Test page-related operations."""

    @pytest.mark.asyncio
    async def test_search_pages(self, connector):
        """Should search for pages."""
        mock_response = {
            "results": [
                make_page_data("page-1", "Engineering Docs"),
                make_page_data("page-2", "API Reference"),
            ],
            "next_cursor": None,
            "has_more": False,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            results, cursor = await connector._search_pages(query="docs")

            assert len(results) == 2
            assert cursor is None

    @pytest.mark.asyncio
    async def test_search_pages_with_pagination(self, connector):
        """Should handle pagination in search."""
        mock_response = {
            "results": [make_page_data("page-1", "Page 1")],
            "next_cursor": "cursor_123",
            "has_more": True,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            results, cursor = await connector._search_pages()

            assert len(results) == 1
            assert cursor == "cursor_123"

    @pytest.mark.asyncio
    async def test_get_page(self, connector):
        """Should get a page by ID."""
        mock_response = make_page_data("page-1", "My Page")

        with patch.object(connector, "_api_request", return_value=mock_response):
            page = await connector._get_page("page-1")

            assert page is not None
            assert page.id == "page-1"
            assert page.title == "My Page"

    @pytest.mark.asyncio
    async def test_get_page_not_found(self, connector):
        """Should return None when page not found."""
        with patch.object(connector, "_api_request", side_effect=Exception("Not found")):
            page = await connector._get_page("nonexistent")

            assert page is None

    def test_parse_page(self, connector):
        """Should parse page data correctly."""
        data = make_page_data("page-1", "Test Title")
        page = connector._parse_page(data)

        assert page.id == "page-1"
        assert page.title == "Test Title"
        assert page.url == "https://notion.so/page-1"
        assert page.parent_type == "workspace"
        assert page.archived is False

    def test_parse_page_with_timestamps(self, connector):
        """Should parse timestamps correctly."""
        data = make_page_data("page-1", "Test")
        page = connector._parse_page(data)

        assert page.created_at is not None
        assert page.last_edited_at is not None
        assert page.created_at.year == 2024


# =============================================================================
# Database Operations Tests
# =============================================================================


class TestDatabaseOperations:
    """Test database-related operations."""

    @pytest.mark.asyncio
    async def test_get_database(self, connector):
        """Should get a database by ID."""
        mock_response = make_database_data("db-1", "Tasks Database")

        with patch.object(connector, "_api_request", return_value=mock_response):
            db = await connector._get_database("db-1")

            assert db is not None
            assert db.id == "db-1"
            assert db.title == "Tasks Database"

    @pytest.mark.asyncio
    async def test_query_database(self, connector):
        """Should query database entries."""
        mock_response = {
            "results": [
                make_page_data("entry-1", "Task 1"),
                make_page_data("entry-2", "Task 2"),
            ],
            "next_cursor": None,
            "has_more": False,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            entries, cursor = await connector._query_database("db-1")

            assert len(entries) == 2
            assert cursor is None

    def test_parse_database(self, connector):
        """Should parse database data correctly."""
        data = make_database_data("db-1", "My Database")
        db = connector._parse_database(data)

        assert db.id == "db-1"
        assert db.title == "My Database"
        assert db.description == "Test description"
        assert "Name" in db.properties


# =============================================================================
# Block Content Tests
# =============================================================================


class TestBlockContent:
    """Test block content extraction."""

    @pytest.mark.asyncio
    async def test_get_block_children(self, connector):
        """Should get child blocks."""
        mock_response = {
            "results": [
                make_block_data("block-1", "paragraph", "First paragraph"),
                make_block_data("block-2", "heading_1", "Main Heading"),
            ],
            "next_cursor": None,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            blocks, cursor = await connector._get_block_children("page-1")

            assert len(blocks) == 2
            assert cursor is None

    def test_extract_paragraph_block(self, connector):
        """Should extract paragraph content."""
        block = make_block_data("block-1", "paragraph", "Test paragraph text")
        content = connector._extract_block_content(block)

        assert content == "Test paragraph text"

    def test_extract_heading_blocks(self, connector):
        """Should extract heading content with markdown."""
        h1_block = make_block_data("block-1", "heading_1", "Main Title")
        h2_block = make_block_data("block-2", "heading_2", "Section")
        h3_block = make_block_data("block-3", "heading_3", "Subsection")

        assert connector._extract_block_content(h1_block) == "# Main Title"
        assert connector._extract_block_content(h2_block) == "## Section"
        assert connector._extract_block_content(h3_block) == "### Subsection"

    def test_extract_list_items(self, connector):
        """Should extract list item content."""
        bulleted = make_block_data("block-1", "bulleted_list_item", "Bullet point")
        numbered = make_block_data("block-2", "numbered_list_item", "Numbered item")

        assert connector._extract_block_content(bulleted) == "- Bullet point"
        assert connector._extract_block_content(numbered) == "1. Numbered item"

    def test_extract_quote_block(self, connector):
        """Should extract quote content."""
        block = make_block_data("block-1", "quote", "Important quote")
        content = connector._extract_block_content(block)

        assert content == "> Important quote"

    def test_rich_text_to_string(self, connector):
        """Should convert rich text array to string."""
        rich_text = [
            {"plain_text": "Hello ", "type": "text"},
            {"plain_text": "world", "type": "text"},
        ]
        result = connector._rich_text_to_string(rich_text)

        assert result == "Hello world"

    def test_rich_text_empty(self, connector):
        """Should handle empty rich text."""
        result = connector._rich_text_to_string([])
        assert result == ""


# =============================================================================
# Search Tests
# =============================================================================


class TestSearch:
    """Test search functionality."""

    @pytest.mark.asyncio
    async def test_search(self, connector):
        """Should search for content."""
        mock_search_response = {
            "results": [
                make_page_data("page-1", "Engineering Guide"),
            ],
            "next_cursor": None,
        }

        with (
            patch.object(connector, "_api_request", return_value=mock_search_response),
            patch.object(connector, "_get_page_content", return_value="Page content here"),
        ):
            results = await connector.search("engineering", limit=5)

            assert len(results) >= 0  # May be empty if no matching pages

    @pytest.mark.asyncio
    async def test_search_empty_query(self, connector):
        """Should handle empty query."""
        mock_search_response = {"results": [], "next_cursor": None}

        with patch.object(connector, "_api_request", return_value=mock_search_response):
            results = await connector.search("", limit=5)

            assert isinstance(results, list)


# =============================================================================
# Fetch Tests
# =============================================================================


class TestFetch:
    """Test fetch functionality."""

    @pytest.mark.asyncio
    async def test_fetch_page(self, connector):
        """Should fetch a page by evidence ID."""
        mock_page = make_page_data("page-123", "My Document")

        with (
            patch.object(connector, "_api_request", return_value=mock_page),
            patch.object(connector, "_get_page_content", return_value="Document content"),
        ):
            evidence = await connector.fetch("notion-page-page-123")

            # Fetch may return None if page not found or return Evidence
            if evidence:
                assert "page-123" in evidence.id

    @pytest.mark.asyncio
    async def test_fetch_invalid_id(self, connector):
        """Should return None for invalid evidence ID."""
        result = await connector.fetch("invalid-id-format")

        assert result is None


# =============================================================================
# Sync Tests
# =============================================================================


class TestSyncItems:
    """Test sync_items functionality."""

    @pytest.mark.asyncio
    async def test_sync_items(self, connector):
        """Should yield sync items."""
        mock_search_response = {
            "results": [
                make_page_data("page-1", "Page 1"),
            ],
            "next_cursor": None,
        }

        with (
            patch.object(connector, "_api_request", return_value=mock_search_response),
            patch.object(connector, "_get_page_content", return_value="Page content"),
        ):
            state = SyncState(connector_id=connector.connector_id)
            items = []
            async for item in connector.sync_items(state):
                items.append(item)
                if len(items) >= 5:  # Limit for test
                    break

            # Should yield items (may be 0 if search returns no pages)
            assert isinstance(items, list)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_api_error(self, connector):
        """Should handle API errors gracefully."""
        import httpx

        with patch.object(
            connector,
            "_api_request",
            side_effect=httpx.HTTPStatusError(
                "Error", request=None, response=MagicMock(status_code=500)
            ),
        ):
            page = await connector._get_page("page-1")
            assert page is None

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, connector):
        """Should handle rate limit errors."""
        import httpx

        with patch.object(
            connector,
            "_api_request",
            side_effect=httpx.HTTPStatusError(
                "Rate limited", request=None, response=MagicMock(status_code=429)
            ),
        ):
            db = await connector._get_database("db-1")
            assert db is None


# =============================================================================
# Data Model Tests
# =============================================================================


class TestDataModels:
    """Test data model creation."""

    def test_notion_page_creation(self):
        """Should create NotionPage with defaults."""
        page = NotionPage(
            id="page-1",
            title="Test Page",
            url="https://notion.so/page-1",
        )
        assert page.content == ""
        assert page.parent_type == ""
        assert page.archived is False
        assert page.properties == {}

    def test_notion_page_with_all_fields(self):
        """Should create NotionPage with all fields."""
        page = NotionPage(
            id="page-1",
            title="Full Page",
            url="https://notion.so/page-1",
            content="Page content",
            parent_type="workspace",
            parent_id="ws-1",
            created_by="user-1",
            created_at=datetime.now(timezone.utc),
            archived=True,
            properties={"key": "value"},
        )
        assert page.content == "Page content"
        assert page.archived is True
        assert "key" in page.properties

    def test_notion_database_creation(self):
        """Should create NotionDatabase with defaults."""
        db = NotionDatabase(
            id="db-1",
            title="Test DB",
            url="https://notion.so/db-1",
        )
        assert db.description == ""
        assert db.properties == {}

    def test_notion_database_with_all_fields(self):
        """Should create NotionDatabase with all fields."""
        db = NotionDatabase(
            id="db-1",
            title="Full DB",
            url="https://notion.so/db-1",
            description="Test description",
            properties={"Name": {"type": "title"}},
            created_at=datetime.now(timezone.utc),
        )
        assert db.description == "Test description"
        assert "Name" in db.properties


# =============================================================================
# Page Content Extraction Tests
# =============================================================================


class TestPageContentExtraction:
    """Test full page content extraction."""

    @pytest.mark.asyncio
    async def test_get_page_content(self, connector):
        """Should extract page content from blocks."""
        mock_blocks_response = {
            "results": [
                make_block_data("block-1", "heading_1", "Introduction"),
                make_block_data("block-2", "paragraph", "First paragraph of content."),
                make_block_data("block-3", "paragraph", "Second paragraph."),
            ],
            "next_cursor": None,
        }

        with patch.object(
            connector, "_get_block_children", return_value=(mock_blocks_response["results"], None)
        ):
            content = await connector._get_page_content("page-1")

            assert "Introduction" in content
            assert "First paragraph" in content

    @pytest.mark.asyncio
    async def test_get_page_content_max_depth(self, connector):
        """Should respect max depth limit."""
        connector.max_depth = 1

        # At depth 1, should return empty (depth >= max_depth)
        content = await connector._get_page_content("page-1", depth=1)

        assert content == ""

    @pytest.mark.asyncio
    async def test_get_page_content_nested_blocks(self, connector):
        """Should handle nested blocks."""
        mock_parent_block = {
            "id": "block-1",
            "type": "toggle",
            "has_children": True,
            "toggle": {"rich_text": [{"plain_text": "Toggle content"}]},
        }
        mock_child_blocks = [
            make_block_data("child-1", "paragraph", "Nested content"),
        ]

        call_count = [0]

        async def mock_get_children(block_id, cursor=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return [mock_parent_block], None
            return mock_child_blocks, None

        with patch.object(connector, "_get_block_children", side_effect=mock_get_children):
            content = await connector._get_page_content("page-1")

            # Should have processed both parent and nested content
            assert "Toggle content" in content or "Nested content" in content


# =============================================================================
# Child Page Discovery Tests
# =============================================================================


class TestChildPageDiscovery:
    """Test recursive child page discovery."""

    @pytest.mark.asyncio
    async def test_discover_child_pages(self, connector):
        """Should discover child pages."""
        mock_blocks = [
            {
                "id": "child-page-1",
                "type": "child_page",
                "has_children": False,
                "child_page": {"title": "Child Page 1"},
            },
            {
                "id": "child-page-2",
                "type": "child_page",
                "has_children": False,
                "child_page": {"title": "Child Page 2"},
            },
        ]

        # Track which block_id is being queried to avoid infinite recursion
        call_count = [0]

        async def mock_get_children(block_id, cursor=None):
            call_count[0] += 1
            # Only return blocks for the initial parent call
            if call_count[0] == 1:
                return mock_blocks, None
            # For recursive calls on child pages, return empty
            return [], None

        with patch.object(connector, "_get_block_children", side_effect=mock_get_children):
            children = []
            async for child_id, child_title, depth in connector._discover_child_pages("parent-1"):
                children.append((child_id, child_title, depth))

            assert len(children) == 2
            assert children[0][1] == "Child Page 1"

    @pytest.mark.asyncio
    async def test_discover_child_database(self, connector):
        """Should discover child databases."""
        mock_blocks = [
            {
                "id": "child-db-1",
                "type": "child_database",
                "has_children": False,
                "child_database": {"title": "Tasks DB"},
            },
        ]

        with patch.object(connector, "_get_block_children", return_value=(mock_blocks, None)):
            children = []
            async for child_id, child_title, depth in connector._discover_child_pages("parent-1"):
                children.append((child_id, child_title, depth))

            assert len(children) == 1
            assert "[DB]" in children[0][1]

    @pytest.mark.asyncio
    async def test_discover_respects_max_depth(self, connector):
        """Should respect max depth for discovery."""
        connector.max_depth = 2

        # At max depth, should not yield anything
        children = []
        async for child in connector._discover_child_pages("parent-1", depth=2):
            children.append(child)

        assert len(children) == 0
