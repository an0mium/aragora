"""
IBM Docling adapter for advanced document layout analysis.

Docling excels at extracting complex structures from documents:
- Tables with merged cells and spanning
- Document layout and reading order
- Figure captions and references
- Mathematical equations

Best used in combination with UnstructuredParser for comprehensive extraction.

Requirements:
    pip install docling

Usage:
    from aragora.documents.ingestion.docling_adapter import (
        DoclingParser,
        parse_with_docling,
    )

    parser = DoclingParser()
    result = parser.parse(content, filename="report.pdf")

    # Get extracted tables
    tables = result.tables
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Check for docling library
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logger.info("docling not available - install with: pip install docling")


@dataclass
class ExtractedTable:
    """A table extracted from a document."""

    id: str = ""
    page_number: int = 0
    rows: list[list[str]] = field(default_factory=list)
    headers: list[str] = field(default_factory=list)
    caption: str = ""
    bounding_box: Optional[dict[str, float]] = None

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def col_count(self) -> int:
        return len(self.headers) if self.headers else (len(self.rows[0]) if self.rows else 0)

    def to_markdown(self) -> str:
        """Convert table to Markdown format."""
        if not self.rows and not self.headers:
            return ""

        lines = []

        # Headers
        if self.headers:
            lines.append("| " + " | ".join(self.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(self.headers)) + " |")
        elif self.rows:
            # Use first row as header
            lines.append("| " + " | ".join(self.rows[0]) + " |")
            lines.append("| " + " | ".join(["---"] * len(self.rows[0])) + " |")
            self.rows = self.rows[1:]

        # Data rows
        for row in self.rows:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def to_csv(self) -> str:
        """Convert table to CSV format."""
        import csv
        import io as io_module

        output = io_module.StringIO()
        writer = csv.writer(output)

        if self.headers:
            writer.writerow(self.headers)
        for row in self.rows:
            writer.writerow(row)

        return output.getvalue()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "page_number": self.page_number,
            "rows": self.rows,
            "headers": self.headers,
            "caption": self.caption,
            "bounding_box": self.bounding_box,
            "row_count": self.row_count,
            "col_count": self.col_count,
        }


@dataclass
class ExtractedFigure:
    """A figure/image extracted from a document."""

    id: str = ""
    page_number: int = 0
    caption: str = ""
    alt_text: str = ""
    bounding_box: Optional[dict[str, float]] = None
    image_data: Optional[bytes] = None  # Raw image bytes if extracted

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "page_number": self.page_number,
            "caption": self.caption,
            "alt_text": self.alt_text,
            "bounding_box": self.bounding_box,
            "has_image_data": self.image_data is not None,
        }


@dataclass
class DoclingResult:
    """Result of Docling document analysis."""

    text: str
    tables: list[ExtractedTable]
    figures: list[ExtractedFigure]
    page_count: int
    headings: list[str]
    metadata: dict[str, Any]
    parser_name: str = "docling"
    parse_duration_ms: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "tables": [t.to_dict() for t in self.tables],
            "figures": [f.to_dict() for f in self.figures],
            "page_count": self.page_count,
            "headings": self.headings,
            "metadata": self.metadata,
            "parser_name": self.parser_name,
            "parse_duration_ms": self.parse_duration_ms,
            "error": self.error,
        }


class DoclingParser:
    """
    Document parser using IBM Docling.

    Specialized for extracting structured content like tables,
    figures, and complex layouts from PDF documents.
    """

    SUPPORTED_FORMATS = {".pdf"}  # Docling primarily supports PDF

    def __init__(
        self,
        extract_tables: bool = True,
        extract_figures: bool = True,
        extract_images: bool = False,  # Image extraction can be expensive
        ocr_enabled: bool = True,
    ):
        """
        Initialize Docling parser.

        Args:
            extract_tables: Extract table structures
            extract_figures: Extract figure captions
            extract_images: Extract actual image data (can be memory-intensive)
            ocr_enabled: Enable OCR for scanned documents
        """
        self.extract_tables = extract_tables
        self.extract_figures = extract_figures
        self.extract_images = extract_images
        self.ocr_enabled = ocr_enabled

        self._converter: Optional[Any] = None

    def _get_converter(self):
        """Get or create the document converter."""
        if not DOCLING_AVAILABLE:
            raise RuntimeError("Docling is not installed. Install with: pip install docling")

        if self._converter is None:
            # Configure pipeline options
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = self.ocr_enabled
            pipeline_options.do_table_structure = self.extract_tables

            self._converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF],
                pdf_pipeline_options=pipeline_options,
            )

        return self._converter

    def parse(
        self,
        content: bytes,
        filename: str,
    ) -> DoclingResult:
        """
        Parse a document using Docling.

        Args:
            content: Raw file content
            filename: Original filename

        Returns:
            DoclingResult with extracted content
        """
        start_time = time.monotonic()

        ext = Path(filename).suffix.lower()
        if ext not in self.SUPPORTED_FORMATS:
            return DoclingResult(
                text="",
                tables=[],
                figures=[],
                page_count=0,
                headings=[],
                metadata={},
                error=f"Unsupported format: {ext}. Docling supports: {self.SUPPORTED_FORMATS}",
            )

        if not DOCLING_AVAILABLE:
            return DoclingResult(
                text="",
                tables=[],
                figures=[],
                page_count=0,
                headings=[],
                metadata={},
                error="Docling is not installed",
            )

        try:
            result = self._parse_with_docling(content, filename)
            result.parse_duration_ms = int((time.monotonic() - start_time) * 1000)
            return result
        except Exception as e:
            logger.error(f"Docling parsing failed: {e}")
            return DoclingResult(
                text="",
                tables=[],
                figures=[],
                page_count=0,
                headings=[],
                metadata={},
                parse_duration_ms=int((time.monotonic() - start_time) * 1000),
                error=str(e),
            )

    def _parse_with_docling(self, content: bytes, filename: str) -> DoclingResult:
        """Internal parsing using Docling."""
        import tempfile

        # Write content to temp file (Docling requires file path)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            converter = self._get_converter()

            # Convert document
            result = converter.convert(temp_path)
            doc = result.document

            # Extract text
            text = doc.export_to_markdown()

            # Extract tables
            tables = []
            if self.extract_tables:
                for i, table in enumerate(doc.tables):
                    extracted = ExtractedTable(
                        id=f"table_{i}",
                        page_number=getattr(table, "page_no", 0),
                        rows=[],
                        headers=[],
                        caption=getattr(table, "caption", ""),
                    )

                    # Extract table data
                    if hasattr(table, "data"):
                        data = table.data
                        if hasattr(data, "headers"):
                            extracted.headers = [str(h) for h in data.headers]
                        if hasattr(data, "rows"):
                            extracted.rows = [[str(cell) for cell in row] for row in data.rows]

                    tables.append(extracted)

            # Extract figures
            figures: list[ExtractedFigure] = []
            if self.extract_figures:
                for i, figure in enumerate(doc.pictures):
                    fig_extracted = ExtractedFigure(
                        id=f"figure_{i}",
                        page_number=getattr(figure, "page_no", 0),
                        caption=getattr(figure, "caption", ""),
                    )

                    # Extract image data if requested
                    if self.extract_images and hasattr(figure, "image"):
                        fig_extracted.image_data = figure.image

                    figures.append(fig_extracted)

            # Extract headings
            headings = []
            for item in doc.document_index:
                if hasattr(item, "text"):
                    headings.append(item.text)

            # Get metadata
            metadata = {}
            if hasattr(doc, "metadata"):
                for key in ["title", "author", "subject", "keywords", "creator"]:
                    if hasattr(doc.metadata, key):
                        value = getattr(doc.metadata, key)
                        if value:
                            metadata[key] = value

            return DoclingResult(
                text=text,
                tables=tables,
                figures=figures,
                page_count=doc.page_count if hasattr(doc, "page_count") else 1,
                headings=headings,
                metadata=metadata,
            )

        finally:
            temp_path.unlink(missing_ok=True)

    def extract_tables_only(
        self,
        content: bytes,
        filename: str,
    ) -> list[ExtractedTable]:
        """
        Extract only tables from a document.

        Convenience method for table-focused extraction.

        Args:
            content: Raw file content
            filename: Original filename

        Returns:
            List of extracted tables
        """
        result = self.parse(content, filename)
        return result.tables

    def is_supported(self, filename: str) -> bool:
        """Check if file format is supported."""
        ext = Path(filename).suffix.lower()
        return ext in self.SUPPORTED_FORMATS


def parse_with_docling(
    content: bytes,
    filename: str,
    extract_tables: bool = True,
    extract_figures: bool = True,
) -> DoclingResult:
    """
    Convenience function to parse a document with Docling.

    Args:
        content: Raw file content
        filename: Original filename
        extract_tables: Extract table structures
        extract_figures: Extract figure captions

    Returns:
        DoclingResult with extracted content
    """
    parser = DoclingParser(
        extract_tables=extract_tables,
        extract_figures=extract_figures,
    )
    return parser.parse(content, filename)


def get_tables_from_pdf(content: bytes, filename: str = "document.pdf") -> list[ExtractedTable]:
    """
    Extract tables from a PDF document.

    Convenience function for table extraction.

    Args:
        content: PDF file content
        filename: Optional filename

    Returns:
        List of extracted tables
    """
    parser = DoclingParser(
        extract_tables=True,
        extract_figures=False,
        extract_images=False,
    )
    return parser.extract_tables_only(content, filename)


__all__ = [
    "DoclingParser",
    "DoclingResult",
    "ExtractedTable",
    "ExtractedFigure",
    "parse_with_docling",
    "get_tables_from_pdf",
    "DOCLING_AVAILABLE",
]
