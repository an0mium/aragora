"""Tests for Notion connector bidirectional support."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from aragora.connectors.enterprise.collaboration.notion import (
    NotionConnector,
    NotionPage,
    NotionDatabase,
)


class TestNotionWriteOperations:
    """Tests for Notion write operations."""

    @pytest.fixture
    def connector(self):
        """Create a test connector."""
        return NotionConnector(workspace_name="test")

    @pytest.mark.asyncio
    async def test_create_page_success(self, connector):
        """Test successful page creation."""
        mock_response = {
            "id": "page-123",
            "url": "https://notion.so/page-123",
            "properties": {
                "title": {"title": [{"plain_text": "New Page"}]},
            },
            "created_time": "2024-01-01T00:00:00.000Z",
            "last_edited_time": "2024-01-01T00:00:00.000Z",
            "parent": {"page_id": "parent-123"},
        }

        with patch.object(
            connector,
            "_api_request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await connector.create_page(
                parent_id="parent-123",
                title="New Page",
                content="Hello world",
            )

            assert result is not None
            assert result.id == "page-123"
            assert result.title == "New Page"

    @pytest.mark.asyncio
    async def test_create_page_failure(self, connector):
        """Test page creation failure."""
        with patch.object(
            connector,
            "_api_request",
            new_callable=AsyncMock,
            side_effect=Exception("API error"),
        ):
            result = await connector.create_page(
                parent_id="parent-123",
                title="New Page",
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_update_page_properties(self, connector):
        """Test updating page properties."""
        mock_response = {
            "id": "page-123",
            "url": "https://notion.so/page-123",
            "properties": {
                "title": {"title": [{"plain_text": "Updated Title"}]},
            },
            "created_time": "2024-01-01T00:00:00.000Z",
            "last_edited_time": "2024-01-02T00:00:00.000Z",
            "parent": {"page_id": "parent-123"},
        }

        with patch.object(
            connector,
            "_api_request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await connector.update_page(
                page_id="page-123",
                properties={"title": {"title": [{"text": {"content": "Updated Title"}}]}},
            )

            assert result is not None
            assert result.title == "Updated Title"

    @pytest.mark.asyncio
    async def test_archive_page(self, connector):
        """Test archiving a page."""
        mock_response = {
            "id": "page-123",
            "url": "https://notion.so/page-123",
            "properties": {"title": {"title": [{"plain_text": "Page"}]}},
            "archived": True,
            "created_time": "2024-01-01T00:00:00.000Z",
            "last_edited_time": "2024-01-02T00:00:00.000Z",
            "parent": {"page_id": "parent-123"},
        }

        with patch.object(
            connector,
            "_api_request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await connector.archive_page("page-123")

            assert result is True

    @pytest.mark.asyncio
    async def test_append_content(self, connector):
        """Test appending content to a page."""
        with patch.object(
            connector,
            "_api_request",
            new_callable=AsyncMock,
            return_value={},
        ) as mock_request:
            result = await connector.append_content(
                page_id="page-123",
                content="Additional content",
            )

            assert result is True
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == "/blocks/page-123/children"
            assert call_args[1]["method"] == "PATCH"

    @pytest.mark.asyncio
    async def test_delete_block(self, connector):
        """Test deleting a block."""
        with patch.object(
            connector,
            "_api_request",
            new_callable=AsyncMock,
            return_value={},
        ) as mock_request:
            result = await connector.delete_block("block-123")

            assert result is True
            mock_request.assert_called_once()
            assert mock_request.call_args[1]["method"] == "DELETE"

    @pytest.mark.asyncio
    async def test_add_database_entry(self, connector):
        """Test adding a database entry."""
        mock_response = {
            "id": "entry-123",
            "url": "https://notion.so/entry-123",
            "properties": {
                "Name": {"title": [{"plain_text": "New Entry"}]},
                "Status": {"select": {"name": "Active"}},
            },
            "created_time": "2024-01-01T00:00:00.000Z",
            "last_edited_time": "2024-01-01T00:00:00.000Z",
            "parent": {"database_id": "db-123"},
        }

        with patch.object(
            connector,
            "_api_request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await connector.add_database_entry(
                database_id="db-123",
                properties={
                    "Name": {"title": [{"text": {"content": "New Entry"}}]},
                    "Status": {"select": {"name": "Active"}},
                },
            )

            assert result is not None
            assert result.id == "entry-123"


class TestNotionTextToBlocks:
    """Tests for text to Notion blocks conversion."""

    @pytest.fixture
    def connector(self):
        """Create a test connector."""
        return NotionConnector(workspace_name="test")

    def test_heading_conversion(self, connector):
        """Test heading text to blocks."""
        text = "# Main Heading\n\n## Sub Heading\n\n### Minor Heading"
        blocks = connector._text_to_blocks(text)

        assert len(blocks) == 3
        assert blocks[0]["type"] == "heading_1"
        assert blocks[1]["type"] == "heading_2"
        assert blocks[2]["type"] == "heading_3"

    def test_bullet_list_conversion(self, connector):
        """Test bullet list to blocks."""
        text = "- Item 1\n- Item 2\n- Item 3"
        blocks = connector._text_to_blocks(text)

        assert len(blocks) == 3
        for block in blocks:
            assert block["type"] == "bulleted_list_item"

    def test_code_block_conversion(self, connector):
        """Test code block conversion."""
        text = "```python\nprint('hello')\n```"
        blocks = connector._text_to_blocks(text)

        assert len(blocks) == 1
        assert blocks[0]["type"] == "code"
        assert blocks[0]["code"]["language"] == "python"

    def test_quote_conversion(self, connector):
        """Test quote conversion."""
        text = "> This is a quote"
        blocks = connector._text_to_blocks(text)

        assert len(blocks) == 1
        assert blocks[0]["type"] == "quote"

    def test_paragraph_conversion(self, connector):
        """Test regular paragraph conversion."""
        text = "This is a regular paragraph."
        blocks = connector._text_to_blocks(text)

        assert len(blocks) == 1
        assert blocks[0]["type"] == "paragraph"

    def test_long_paragraph_chunking(self, connector):
        """Test that long paragraphs are chunked."""
        long_text = "x" * 3000
        blocks = connector._text_to_blocks(long_text)

        # Should be split into multiple paragraph blocks
        assert len(blocks) >= 2
        for block in blocks:
            assert len(block["paragraph"]["rich_text"][0]["text"]["content"]) <= 1900


class TestNotionBuildProperty:
    """Tests for property building helper."""

    @pytest.fixture
    def connector(self):
        """Create a test connector."""
        return NotionConnector(workspace_name="test")

    def test_build_title_property(self, connector):
        """Test building title property."""
        result = connector._build_property("title", "My Title")

        assert result["title"][0]["text"]["content"] == "My Title"

    def test_build_number_property(self, connector):
        """Test building number property."""
        result = connector._build_property("number", 42)

        assert result["number"] == 42.0

    def test_build_select_property(self, connector):
        """Test building select property."""
        result = connector._build_property("select", "Option A")

        assert result["select"]["name"] == "Option A"

    def test_build_multi_select_property(self, connector):
        """Test building multi-select property."""
        result = connector._build_property("multi_select", ["A", "B", "C"])

        assert len(result["multi_select"]) == 3

    def test_build_checkbox_property(self, connector):
        """Test building checkbox property."""
        result = connector._build_property("checkbox", True)

        assert result["checkbox"] is True

    def test_build_url_property(self, connector):
        """Test building URL property."""
        result = connector._build_property("url", "https://example.com")

        assert result["url"] == "https://example.com"

    def test_build_date_property(self, connector):
        """Test building date property."""
        result = connector._build_property("date", "2024-01-01")

        assert result["date"]["start"] == "2024-01-01"
