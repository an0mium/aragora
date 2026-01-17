"""
Document ingestion: batch upload, parsing, and streaming.

Integrates with:
- Unstructured.io for 25+ format support
- IBM Docling for advanced table/layout extraction
"""

from aragora.documents.ingestion.unstructured_adapter import (
    UnstructuredParser,
    ParseResult,
    ParsedElement,
    parse_document,
    get_supported_formats,
    UNSTRUCTURED_AVAILABLE,
)
from aragora.documents.ingestion.docling_adapter import (
    DoclingParser,
    DoclingResult,
    ExtractedTable,
    ExtractedFigure,
    parse_with_docling,
    get_tables_from_pdf,
    DOCLING_AVAILABLE,
)
from aragora.documents.ingestion.batch_processor import (
    BatchProcessor,
    DocumentJob,
    JobStatus,
    JobPriority,
    BatchResult,
    get_batch_processor,
)

__all__ = [
    # Unstructured
    "UnstructuredParser",
    "ParseResult",
    "ParsedElement",
    "parse_document",
    "get_supported_formats",
    "UNSTRUCTURED_AVAILABLE",
    # Docling
    "DoclingParser",
    "DoclingResult",
    "ExtractedTable",
    "ExtractedFigure",
    "parse_with_docling",
    "get_tables_from_pdf",
    "DOCLING_AVAILABLE",
    # Batch processor
    "BatchProcessor",
    "DocumentJob",
    "JobStatus",
    "JobPriority",
    "BatchResult",
    "get_batch_processor",
]
