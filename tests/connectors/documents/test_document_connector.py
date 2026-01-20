"""Tests for DocumentConnector and Evidence integration."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.connectors.documents.connector import (
    DocumentConnector,
    DocumentEvidence,
)
from aragora.connectors.documents.parser import (
    DocumentFormat,
    DocumentChunk,
    ParsedDocument,
    DocumentTable,
)


class TestDocumentConnector:
    """Tests for DocumentConnector."""

    def test_connector_init(self):
        """Test connector initialization with default options."""
        connector = DocumentConnector()

        assert connector.name == "documents"
        assert connector.parser is not None
        assert connector._parsed_docs == {}

    def test_connector_init_with_options(self):
        """Test connector initialization with custom options."""
        connector = DocumentConnector(
            max_pages=50,
            extract_tables=False,
            chunk_size=500,
            chunk_overlap=100,
        )

        assert connector.parser._max_pages == 50
        assert connector.parser._extract_tables is False
        assert connector.parser._chunk_size == 500

    @pytest.mark.asyncio
    async def test_connect_always_succeeds(self):
        """Test connect method always returns True."""
        connector = DocumentConnector()
        assert await connector.connect() is True

    @pytest.mark.asyncio
    async def test_disconnect_clears_cache(self):
        """Test disconnect clears parsed documents cache."""
        connector = DocumentConnector()
        connector._parsed_docs["test"] = MagicMock()

        await connector.disconnect()

        assert connector._parsed_docs == {}

    @pytest.mark.asyncio
    async def test_search_empty_when_no_docs(self):
        """Test search returns empty list when no documents parsed."""
        connector = DocumentConnector()

        results = await connector.search("test query", limit=5)

        assert results == []

    @pytest.mark.asyncio
    async def test_search_finds_matching_content(self):
        """Test search finds content matching query terms."""
        connector = DocumentConnector()

        # Add a mock parsed document
        doc = ParsedDocument(
            format=DocumentFormat.PDF,
            filename="test.pdf",
            content="This document discusses machine learning algorithms.",
            chunks=[
                DocumentChunk(
                    content="This document discusses machine learning algorithms.",
                    page=1,
                    section="Introduction",
                )
            ],
            tables=[],
            metadata={"source_path": "/path/to/test.pdf"},
        )
        connector._parsed_docs["test_doc"] = doc

        results = await connector.search("machine learning", limit=5)

        assert len(results) == 1
        assert results[0]["title"] == "test.pdf (Section 1)"
        assert "machine learning" in results[0]["content"].lower()
        assert results[0]["format"] == "pdf"

    @pytest.mark.asyncio
    async def test_search_finds_table_content(self):
        """Test search finds content in tables."""
        connector = DocumentConnector()

        doc = ParsedDocument(
            format=DocumentFormat.XLSX,
            filename="data.xlsx",
            content="",
            chunks=[],
            tables=[
                DocumentTable(
                    data=[["Name", "Score"], ["Alice", "95"], ["Bob", "87"]],
                    headers=["Name", "Score"],
                    page=1,
                )
            ],
            metadata={"source_path": "/path/to/data.xlsx"},
        )
        connector._parsed_docs["test_doc"] = doc

        results = await connector.search("Alice", limit=5)

        assert len(results) == 1
        assert "Alice" in results[0]["content"]
        assert results[0]["source"] == "document_table"

    @pytest.mark.asyncio
    async def test_parse_file_nonexistent(self):
        """Test parse_file returns None for nonexistent file."""
        connector = DocumentConnector()

        result = await connector.parse_file("/nonexistent/path/file.pdf")

        assert result is None

    @pytest.mark.asyncio
    async def test_parse_bytes_success(self):
        """Test parse_bytes with valid content."""
        connector = DocumentConnector()

        # Mock the parser
        mock_doc = ParsedDocument(
            format=DocumentFormat.TXT,
            filename="test.txt",
            content="Hello world",
            chunks=[DocumentChunk(content="Hello world", page=1)],
            tables=[],
            metadata={},
        )
        connector.parser.parse = MagicMock(return_value=mock_doc)

        result = await connector.parse_bytes(b"Hello world", "test.txt")

        assert result is not None
        assert result.content == "Hello world"
        assert len(connector._parsed_docs) == 1

    @pytest.mark.asyncio
    async def test_parse_from_url(self):
        """Test parse_from_url sets source URL in metadata."""
        connector = DocumentConnector()

        mock_doc = ParsedDocument(
            format=DocumentFormat.TXT,
            filename="doc.txt",
            content="Content",
            chunks=[DocumentChunk(content="Content", page=1)],
            tables=[],
            metadata={},
        )
        connector.parser.parse = MagicMock(return_value=mock_doc)

        result = await connector.parse_from_url(
            "https://example.com/doc.txt",
            b"Content",
        )

        assert result is not None
        assert result.metadata.get("source_url") == "https://example.com/doc.txt"

    def test_get_parsed_documents(self):
        """Test get_parsed_documents returns copy of cache."""
        connector = DocumentConnector()
        mock_doc = MagicMock()
        connector._parsed_docs["test"] = mock_doc

        docs = connector.get_parsed_documents()

        assert "test" in docs
        assert docs is not connector._parsed_docs  # Should be a copy

    def test_clear_documents(self):
        """Test clear_documents empties cache."""
        connector = DocumentConnector()
        connector._parsed_docs["test"] = MagicMock()

        connector.clear_documents()

        assert connector._parsed_docs == {}

    def test_remove_document_exists(self):
        """Test remove_document removes existing document."""
        connector = DocumentConnector()
        connector._parsed_docs["test"] = MagicMock()

        result = connector.remove_document("test")

        assert result is True
        assert "test" not in connector._parsed_docs

    def test_remove_document_not_exists(self):
        """Test remove_document returns False for nonexistent document."""
        connector = DocumentConnector()

        result = connector.remove_document("nonexistent")

        assert result is False

    def test_reliability_scores(self):
        """Test reliability scores for different formats."""
        connector = DocumentConnector()

        assert connector._get_reliability(DocumentFormat.PDF) == 0.85
        assert connector._get_reliability(DocumentFormat.JSON) == 0.90
        assert connector._get_reliability(DocumentFormat.HTML) == 0.65
        assert connector._get_reliability(None) == 0.60


class TestDocumentEvidence:
    """Tests for DocumentEvidence helper class."""

    def test_from_parsed_document_chunks(self):
        """Test converting document chunks to evidence snippets."""
        doc = ParsedDocument(
            format=DocumentFormat.PDF,
            filename="report.pdf",
            title="Annual Report",
            content="Full content here",
            chunks=[
                DocumentChunk(content="First section content", page=1, section="Intro"),
                DocumentChunk(content="Second section content", page=2, section="Body"),
            ],
            tables=[],
            metadata={"source_path": "/path/to/report.pdf"},
        )

        snippets = DocumentEvidence.from_parsed_document(doc)

        assert len(snippets) == 2
        assert snippets[0]["source"] == "document"
        assert snippets[0]["title"] == "Annual Report"
        assert snippets[0]["reliability_score"] == 0.85  # PDF score
        assert snippets[0]["metadata"]["filename"] == "report.pdf"

    def test_from_parsed_document_tables(self):
        """Test converting document tables to evidence snippets."""
        doc = ParsedDocument(
            format=DocumentFormat.XLSX,
            filename="data.xlsx",
            title="Sales Data",
            content="",
            chunks=[],
            tables=[
                DocumentTable(
                    data=[["Q1", "100"], ["Q2", "150"]],
                    headers=["Quarter", "Sales"],
                    page=1,
                    caption="Quarterly sales",
                )
            ],
            metadata={"source_url": "https://example.com/data.xlsx"},
        )

        snippets = DocumentEvidence.from_parsed_document(doc)

        assert len(snippets) == 1
        assert snippets[0]["source"] == "document_table"
        assert snippets[0]["reliability_score"] == 0.95  # XLSX + 0.05 for table
        assert "Quarter" in snippets[0]["content"]

    def test_from_parsed_document_truncates_long_content(self):
        """Test that long content is truncated."""
        long_content = "x" * 2000
        doc = ParsedDocument(
            format=DocumentFormat.TXT,
            filename="long.txt",
            content=long_content,
            chunks=[DocumentChunk(content=long_content, page=1)],
            tables=[],
            metadata={},
        )

        snippets = DocumentEvidence.from_parsed_document(doc, max_snippet_length=100)

        assert len(snippets[0]["content"]) <= 103  # 100 + "..."

    def test_format_table_text(self):
        """Test table formatting."""
        data = [["A", "B"], ["1", "2"], ["3", "4"]]
        headers = ["Col1", "Col2"]

        text = DocumentEvidence._format_table_text(data, headers)

        assert "Col1" in text
        assert "Col2" in text
        assert "A | B" in text

    def test_format_table_text_limits_rows(self):
        """Test table formatting limits rows."""
        data = [[str(i), str(i+1)] for i in range(20)]

        text = DocumentEvidence._format_table_text(data)

        assert "5 more rows" in text


class TestEvidenceCollectorIntegration:
    """Integration tests for EvidenceCollector with documents."""

    @pytest.mark.asyncio
    async def test_extract_document_paths(self):
        """Test extraction of document paths from task description."""
        from aragora.evidence.collector import EvidenceCollector

        collector = EvidenceCollector(
            connectors={},
            allowed_domains=set(),
        )

        task = 'Analyze the report at /path/to/report.pdf and compare with "./docs/data.xlsx"'
        paths = collector._extract_document_paths(task)

        assert "/path/to/report.pdf" in paths
        assert "./docs/data.xlsx" in paths

    def test_is_document_url(self):
        """Test document URL detection."""
        from aragora.evidence.collector import EvidenceCollector

        collector = EvidenceCollector(
            connectors={},
            allowed_domains=set(),
        )

        assert collector._is_document_url("https://example.com/report.pdf") is True
        assert collector._is_document_url("https://example.com/data.xlsx") is True
        assert collector._is_document_url("https://example.com/page.html") is True
        assert collector._is_document_url("https://example.com/api/data") is False
        assert collector._is_document_url("https://example.com/image.png") is False

    @pytest.mark.asyncio
    async def test_parse_document_file_integration(self):
        """Test parsing document file through EvidenceCollector."""
        from aragora.evidence.collector import EvidenceCollector

        collector = EvidenceCollector(
            connectors={},
            allowed_domains=set(),
        )

        # Should return None for nonexistent file
        result = await collector.parse_document_file("/nonexistent/file.pdf")
        assert result is None

    @pytest.mark.asyncio
    async def test_parse_document_bytes_integration(self):
        """Test parsing document bytes through EvidenceCollector."""
        from aragora.evidence.collector import EvidenceCollector

        collector = EvidenceCollector(
            connectors={},
            allowed_domains=set(),
        )

        # Parse simple text content
        with patch.object(
            collector,
            "_get_document_connector",
            return_value=MagicMock(
                parse_bytes=AsyncMock(
                    return_value=ParsedDocument(
                        format=DocumentFormat.TXT,
                        filename="test.txt",
                        content="Hello",
                        chunks=[DocumentChunk(content="Hello", page=1)],
                        tables=[],
                        metadata={},
                    )
                )
            ),
        ):
            result = await collector.parse_document_bytes(b"Hello", "test.txt")

            assert result is not None
            assert len(result) == 1
            assert result[0].snippet == "Hello"
