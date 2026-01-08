"""
Document management endpoint handlers.

Endpoints:
- GET /api/documents - List all uploaded documents
- GET /api/documents/formats - Get supported file formats
- GET /api/documents/{doc_id} - Get a document by ID
"""

from typing import Optional
from .base import (
    BaseHandler, HandlerResult, json_response, error_response,
)


class DocumentHandler(BaseHandler):
    """Handler for document-related endpoints."""

    ROUTES = [
        "/api/documents",
        "/api/documents/formats",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in self.ROUTES:
            return True
        # Handle /api/documents/{doc_id} pattern
        if path.startswith("/api/documents/") and path.count("/") == 3:
            return True
        return False

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route document requests to appropriate methods."""
        if path == "/api/documents":
            return self._list_documents()

        if path == "/api/documents/formats":
            return self._get_supported_formats()

        if path.startswith("/api/documents/"):
            # Extract doc_id from /api/documents/{doc_id}
            doc_id, err = self.extract_path_param(path, 2, "document_id")
            if err:
                return err
            return self._get_document(doc_id)

        return None

    def get_document_store(self):
        """Get document store instance."""
        return self.ctx.get("document_store")

    def _list_documents(self) -> HandlerResult:
        """List all uploaded documents."""
        store = self.get_document_store()
        if not store:
            return json_response({
                "documents": [],
                "count": 0,
                "error": "Document storage not configured"
            })

        try:
            docs = store.list_all()
            return json_response({
                "documents": docs,
                "count": len(docs)
            })
        except Exception as e:
            return error_response(f"Failed to list documents: {e}", 500)

    def _get_supported_formats(self) -> HandlerResult:
        """Get list of supported document formats."""
        try:
            from aragora.server.documents import get_supported_formats
            formats = get_supported_formats()
            return json_response(formats)
        except ImportError:
            return json_response({
                "extensions": [".txt", ".md", ".pdf"],
                "note": "Document parsing module not fully loaded"
            })

    def _get_document(self, doc_id: str) -> HandlerResult:
        """Get a document by ID."""
        store = self.get_document_store()
        if not store:
            return error_response("Document storage not configured", 500)

        try:
            doc = store.get(doc_id)
            if doc:
                return json_response(doc.to_dict())
            return error_response(f"Document not found: {doc_id}", 404)
        except Exception as e:
            return error_response(f"Failed to get document: {e}", 500)
