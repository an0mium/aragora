"""Deprecated import location for the document parsing/storage surface.

The document parsing surface moved down to :mod:`aragora.documents.parsing`
during the P4a layering work so that lower-layer modules (e.g.
``aragora.core.decision_router``) can reach it without importing
``aragora.server``. Importing from ``aragora.server.documents`` still works but
is deprecated; import from ``aragora.documents.parsing`` instead.
"""

from __future__ import annotations

import warnings

from aragora.documents.parsing import (
    DOCX_AVAILABLE,
    PDF_AVAILABLE,
    SUPPORTED_EXTENSIONS,
    VALID_DOC_ID_PATTERN,
    DocumentStore,
    ParsedDocument,
    _safe_path,
    _validate_doc_id,
    generate_doc_id,
    get_supported_formats,
    parse_docx,
    parse_document,
    parse_pdf,
    parse_text,
)

warnings.warn(
    "aragora.server.documents is deprecated; import from aragora.documents.parsing instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ParsedDocument",
    "DocumentStore",
    "parse_document",
    "parse_text",
    "parse_pdf",
    "parse_docx",
    "generate_doc_id",
    "get_supported_formats",
    "SUPPORTED_EXTENSIONS",
    "VALID_DOC_ID_PATTERN",
    "PDF_AVAILABLE",
    "DOCX_AVAILABLE",
    "_safe_path",
    "_validate_doc_id",
]
