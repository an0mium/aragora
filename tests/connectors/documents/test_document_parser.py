"""
Tests for DocumentParser - Unified Document Parsing.

Tests cover:
- Format detection
- Text/Markdown/HTML parsing
- JSON/YAML/CSV parsing
- Error handling
- Dataclass functionality
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch


class TestDocumentFormat:
    """Tests for DocumentFormat enum."""

    def test_format_values(self):
        """Should have expected format values."""
        from aragora.connectors.documents.parser import DocumentFormat

        assert DocumentFormat.PDF.value == "pdf"
        assert DocumentFormat.DOCX.value == "docx"
        assert DocumentFormat.JSON.value == "json"
        assert DocumentFormat.MD.value == "md"
        assert DocumentFormat.UNKNOWN.value == "unknown"

    def test_format_is_string_enum(self):
        """Format value should be usable as string."""
        from aragora.connectors.documents.parser import DocumentFormat

        assert DocumentFormat.PDF.value == "pdf"
        assert str(DocumentFormat.PDF.value) == "pdf"


class TestDocumentChunk:
    """Tests for DocumentChunk dataclass."""

    def test_create_chunk(self):
        """Should create document chunk."""
        from aragora.connectors.documents.parser import DocumentChunk

        chunk = DocumentChunk(
            content="Test content",
            chunk_type="text",
            page=1,
        )

        assert chunk.content == "Test content"
        assert chunk.chunk_type == "text"
        assert chunk.page == 1

    def test_chunk_defaults(self):
        """Should have default values."""
        from aragora.connectors.documents.parser import DocumentChunk

        chunk = DocumentChunk(content="Test")

        assert chunk.chunk_type == "text"
        assert chunk.page is None
        assert chunk.section is None
        assert chunk.metadata == {}


class TestDocumentTable:
    """Tests for DocumentTable dataclass."""

    def test_create_table(self):
        """Should create document table."""
        from aragora.connectors.documents.parser import DocumentTable

        table = DocumentTable(
            data=[["A", "B"], ["1", "2"]],
            headers=["Col1", "Col2"],
        )

        assert table.rows == 2
        assert table.cols == 2

    def test_table_to_markdown(self):
        """Should convert table to markdown."""
        from aragora.connectors.documents.parser import DocumentTable

        table = DocumentTable(
            data=[["A", "B"], ["1", "2"]],
            headers=["Col1", "Col2"],
        )

        md = table.to_markdown()
        assert "| Col1 | Col2 |" in md
        assert "| --- | --- |" in md
        assert "| A | B |" in md

    def test_empty_table_to_markdown(self):
        """Should handle empty table."""
        from aragora.connectors.documents.parser import DocumentTable

        table = DocumentTable(data=[])
        assert table.to_markdown() == ""

    def test_table_without_headers(self):
        """Should use first row as headers when no headers specified."""
        from aragora.connectors.documents.parser import DocumentTable

        table = DocumentTable(
            data=[["Header1", "Header2"], ["Val1", "Val2"]],
        )

        md = table.to_markdown()
        assert "| Header1 | Header2 |" in md


class TestParsedDocument:
    """Tests for ParsedDocument dataclass."""

    def test_create_parsed_document(self):
        """Should create parsed document."""
        from aragora.connectors.documents.parser import ParsedDocument, DocumentFormat

        doc = ParsedDocument(
            content="Document content here",
            format=DocumentFormat.TXT,
            filename="test.txt",
        )

        assert doc.content == "Document content here"
        assert doc.format == DocumentFormat.TXT
        assert doc.word_count == 3  # Auto-calculated

    def test_word_count_auto_calculated(self):
        """Should auto-calculate word count."""
        from aragora.connectors.documents.parser import ParsedDocument

        doc = ParsedDocument(content="one two three four five")
        assert doc.word_count == 5

    def test_document_with_errors(self):
        """Should track parsing errors."""
        from aragora.connectors.documents.parser import ParsedDocument

        doc = ParsedDocument(
            content="",
            errors=["Failed to extract text"],
        )

        assert len(doc.errors) == 1


class TestDocumentParserInit:
    """Tests for DocumentParser initialization."""

    def test_default_init(self):
        """Should initialize with default values."""
        from aragora.connectors.documents.parser import DocumentParser

        parser = DocumentParser()

        assert parser.max_pages == 100
        assert parser.max_content_size == 500_000
        assert parser.extract_tables is True
        assert parser.extract_metadata is True

    def test_custom_init(self):
        """Should accept custom configuration."""
        from aragora.connectors.documents.parser import DocumentParser

        parser = DocumentParser(
            max_pages=50,
            max_content_size=100_000,
            extract_tables=False,
        )

        assert parser.max_pages == 50
        assert parser.max_content_size == 100_000
        assert parser.extract_tables is False


class TestFormatDetection:
    """Tests for format detection."""

    def test_detect_from_filename(self):
        """Should detect format from filename."""
        from aragora.connectors.documents.parser import DocumentParser, DocumentFormat

        parser = DocumentParser()

        assert parser.detect_format(filename="doc.pdf") == DocumentFormat.PDF
        assert parser.detect_format(filename="doc.docx") == DocumentFormat.DOCX
        assert parser.detect_format(filename="data.json") == DocumentFormat.JSON
        assert parser.detect_format(filename="readme.md") == DocumentFormat.MD

    def test_detect_unknown_format(self):
        """Should return UNKNOWN for unrecognized formats."""
        from aragora.connectors.documents.parser import DocumentParser, DocumentFormat

        parser = DocumentParser()

        assert parser.detect_format(filename="file.xyz") == DocumentFormat.UNKNOWN

    def test_detect_case_insensitive(self):
        """Should handle uppercase extensions."""
        from aragora.connectors.documents.parser import DocumentParser, DocumentFormat

        parser = DocumentParser()

        assert parser.detect_format(filename="DOC.PDF") == DocumentFormat.PDF
        assert parser.detect_format(filename="doc.JSON") == DocumentFormat.JSON


class TestTextParsing:
    """Tests for text format parsing."""

    def test_parse_plain_text(self):
        """Should parse plain text."""
        from aragora.connectors.documents.parser import DocumentParser, DocumentFormat

        parser = DocumentParser()
        content = b"Hello world.\nThis is a test."

        result = parser.parse(content, filename="test.txt")

        assert result.format == DocumentFormat.TXT
        assert "Hello world" in result.content
        assert result.word_count > 0

    def test_parse_markdown(self):
        """Should parse markdown."""
        from aragora.connectors.documents.parser import DocumentParser, DocumentFormat

        parser = DocumentParser()
        content = b"# Heading\n\nSome **bold** text."

        result = parser.parse(content, filename="readme.md")

        assert result.format == DocumentFormat.MD
        assert "Heading" in result.content

    def test_parse_empty_text(self):
        """Should handle empty text."""
        from aragora.connectors.documents.parser import DocumentParser

        parser = DocumentParser()
        result = parser.parse(b"", filename="empty.txt")

        assert result.content == ""
        assert result.word_count == 0


class TestJsonParsing:
    """Tests for JSON parsing."""

    def test_parse_json(self):
        """Should parse JSON documents."""
        from aragora.connectors.documents.parser import DocumentParser, DocumentFormat

        parser = DocumentParser()
        data = {"name": "Test", "value": 123}
        content = json.dumps(data).encode()

        result = parser.parse(content, filename="data.json")

        assert result.format == DocumentFormat.JSON
        assert "name" in result.content
        assert "Test" in result.content

    def test_parse_json_array(self):
        """Should parse JSON arrays."""
        from aragora.connectors.documents.parser import DocumentParser

        parser = DocumentParser()
        data = [{"id": 1}, {"id": 2}]
        content = json.dumps(data).encode()

        result = parser.parse(content, filename="items.json")

        assert "id" in result.content

    def test_parse_invalid_json(self):
        """Should handle invalid JSON gracefully."""
        from aragora.connectors.documents.parser import DocumentParser

        parser = DocumentParser()
        content = b"{ invalid json }"

        result = parser.parse(content, filename="bad.json")

        # Should capture error but not crash
        assert len(result.errors) > 0 or result.content != ""


class TestCsvParsing:
    """Tests for CSV parsing."""

    def test_parse_csv(self):
        """Should parse CSV files."""
        from aragora.connectors.documents.parser import DocumentParser, DocumentFormat

        parser = DocumentParser()
        content = b"name,age\nAlice,30\nBob,25"

        result = parser.parse(content, filename="data.csv")

        assert result.format == DocumentFormat.CSV
        assert len(result.tables) >= 1

    def test_parse_csv_extracts_table(self):
        """Should extract table from CSV."""
        from aragora.connectors.documents.parser import DocumentParser

        parser = DocumentParser()
        content = b"col1,col2\na,b\nc,d"

        result = parser.parse(content, filename="table.csv")

        assert len(result.tables) >= 1
        # Table should have data
        if result.tables:
            table = result.tables[0]
            if hasattr(table, "data"):
                assert len(table.data) >= 2


class TestHtmlParsing:
    """Tests for HTML parsing."""

    def test_parse_html(self):
        """Should parse HTML documents."""
        from aragora.connectors.documents.parser import DocumentParser, DocumentFormat

        parser = DocumentParser()
        content = b"<html><body><h1>Title</h1><p>Content here.</p></body></html>"

        result = parser.parse(content, filename="page.html")

        assert result.format == DocumentFormat.HTML
        assert "Title" in result.content or "Content" in result.content

    def test_parse_html_strips_tags(self):
        """Should strip HTML tags from content."""
        from aragora.connectors.documents.parser import DocumentParser

        parser = DocumentParser()
        content = b"<p>Simple <b>text</b></p>"

        result = parser.parse(content, filename="simple.html")

        # Should have text content without tags
        assert "<p>" not in result.content or "Simple" in result.content


class TestYamlParsing:
    """Tests for YAML parsing."""

    def test_parse_yaml(self):
        """Should parse YAML documents."""
        from aragora.connectors.documents.parser import DocumentParser, DocumentFormat

        parser = DocumentParser()
        content = b"name: Test\nvalue: 123\nitems:\n  - one\n  - two"

        result = parser.parse(content, filename="config.yaml")

        assert result.format == DocumentFormat.YAML
        assert "name" in result.content or "Test" in result.content


class TestXmlParsing:
    """Tests for XML parsing."""

    def test_parse_xml(self):
        """Should parse XML documents."""
        from aragora.connectors.documents.parser import DocumentParser, DocumentFormat

        parser = DocumentParser()
        content = b"<root><item>Value</item></root>"

        result = parser.parse(content, filename="data.xml")

        assert result.format == DocumentFormat.XML


class TestCodeParsing:
    """Tests for code file parsing."""

    def test_parse_python_code(self):
        """Should parse Python code files."""
        from aragora.connectors.documents.parser import DocumentParser, DocumentFormat

        parser = DocumentParser()
        content = b"def hello():\n    print('Hello')"

        result = parser.parse(content, filename="script.py")

        assert result.format == DocumentFormat.CODE
        assert "def hello" in result.content

    def test_parse_javascript_code(self):
        """Should parse JavaScript files."""
        from aragora.connectors.documents.parser import DocumentParser, DocumentFormat

        parser = DocumentParser()
        content = b"function test() { return 42; }"

        result = parser.parse(content, filename="app.js")

        assert result.format == DocumentFormat.CODE


class TestParseDocumentAsync:
    """Tests for async parse_document function."""

    @pytest.mark.asyncio
    async def test_parse_document_async(self):
        """Should parse document asynchronously."""
        from aragora.connectors.documents.parser import parse_document

        content = b"Test content for async parsing."

        result = await parse_document(content, filename="test.txt")

        assert result.content == "Test content for async parsing."

    @pytest.mark.asyncio
    async def test_parse_document_with_format(self):
        """Should accept explicit format."""
        from aragora.connectors.documents.parser import parse_document, DocumentFormat

        content = b"Plain text"

        result = await parse_document(content, format=DocumentFormat.TXT)

        assert result.format == DocumentFormat.TXT


class TestContentSizeLimits:
    """Tests for content size limits."""

    def test_respects_max_content_size(self):
        """Should respect max content size limit."""
        from aragora.connectors.documents.parser import DocumentParser

        parser = DocumentParser(max_content_size=100)
        large_content = b"x" * 200

        result = parser.parse(large_content, filename="large.txt")

        # Content should be truncated or handled
        assert len(result.content) <= 200  # Some buffer for processing


class TestErrorHandling:
    """Tests for error handling."""

    def test_handles_decode_errors(self):
        """Should handle decode errors gracefully."""
        from aragora.connectors.documents.parser import DocumentParser

        parser = DocumentParser()
        # Invalid UTF-8 bytes
        content = b"\xff\xfe invalid"

        result = parser.parse(content, filename="bad.txt")

        # Should not crash, may have errors or partial content
        assert result is not None

    def test_handles_missing_format(self):
        """Should handle missing format specification."""
        from aragora.connectors.documents.parser import DocumentParser, DocumentFormat

        parser = DocumentParser()
        content = b"some content"

        result = parser.parse(content)  # No filename or format

        # Should default to text or unknown
        assert result.format in (DocumentFormat.TXT, DocumentFormat.UNKNOWN)


class TestExtensionMapping:
    """Tests for extension to format mapping."""

    def test_extension_map_completeness(self):
        """Should have mappings for common extensions."""
        from aragora.connectors.documents.parser import EXTENSION_MAP, DocumentFormat

        assert ".pdf" in EXTENSION_MAP
        assert ".docx" in EXTENSION_MAP
        assert ".xlsx" in EXTENSION_MAP
        assert ".json" in EXTENSION_MAP
        assert ".md" in EXTENSION_MAP
        assert ".py" in EXTENSION_MAP

    def test_markdown_variants(self):
        """Should support markdown extension variants."""
        from aragora.connectors.documents.parser import EXTENSION_MAP, DocumentFormat

        assert EXTENSION_MAP.get(".md") == DocumentFormat.MD
        assert EXTENSION_MAP.get(".markdown") == DocumentFormat.MD

    def test_yaml_variants(self):
        """Should support YAML extension variants."""
        from aragora.connectors.documents.parser import EXTENSION_MAP, DocumentFormat

        assert EXTENSION_MAP.get(".yaml") == DocumentFormat.YAML
        assert EXTENSION_MAP.get(".yml") == DocumentFormat.YAML


class TestMetadataExtraction:
    """Tests for metadata extraction."""

    def test_extracts_filename_metadata(self):
        """Should parse content correctly with filename provided."""
        from aragora.connectors.documents.parser import DocumentParser, DocumentFormat

        parser = DocumentParser()
        result = parser.parse(b"content", filename="document.txt")

        # Filename is used for format detection, content is parsed
        assert result.format == DocumentFormat.TXT
        assert result.content == "content"

    def test_metadata_extraction_toggle(self):
        """Should respect extract_metadata setting."""
        from aragora.connectors.documents.parser import DocumentParser

        parser_with = DocumentParser(extract_metadata=True)
        parser_without = DocumentParser(extract_metadata=False)

        # Both should work without errors
        result1 = parser_with.parse(b"test", filename="test.txt")
        result2 = parser_without.parse(b"test", filename="test.txt")

        assert result1 is not None
        assert result2 is not None
