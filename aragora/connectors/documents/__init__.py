"""
Document Connectors and Parsers.

Provides unified document ingestion for omnivorous capabilities:
- PDF parsing with table extraction
- Word document (.docx) parsing
- Excel spreadsheet (.xlsx) parsing
- PowerPoint presentation (.pptx) parsing
- Structured data (JSON, YAML, XML, CSV)
- Plain text and Markdown

Usage:
    from aragora.connectors.documents import parse_document, DocumentParser

    # Parse a document
    result = await parse_document(content, filename="report.pdf")
    print(result.content)
    print(result.tables)

    # Use parser with custom options
    parser = DocumentParser(max_pages=50, extract_tables=True)
    result = parser.parse(content, format=DocumentFormat.PDF)
"""

from aragora.connectors.documents.parser import (
    DocumentParser,
    DocumentFormat,
    DocumentChunk,
    ParsedDocument,
    parse_document,
    EXTENSION_MAP,
)

__all__ = [
    "DocumentParser",
    "DocumentFormat",
    "DocumentChunk",
    "ParsedDocument",
    "parse_document",
    "EXTENSION_MAP",
]
