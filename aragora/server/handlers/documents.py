"""
Document management endpoint handlers.

Endpoints:
- GET /api/documents - List all uploaded documents
- GET /api/documents/formats - Get supported file formats
- GET /api/documents/{doc_id} - Get a document by ID
- POST /api/documents/upload - Upload a document
"""

import logging
import os
import threading
import time
from typing import Optional

from .base import (
    BaseHandler, HandlerResult, json_response, error_response, handle_errors,
)

logger = logging.getLogger(__name__)

# DoS protection
MAX_MULTIPART_PARTS = 10


class DocumentHandler(BaseHandler):
    """Handler for document-related endpoints."""

    ROUTES = [
        "/api/documents",
        "/api/documents/formats",
        "/api/documents/upload",
    ]

    # Upload rate limiting (IP-based)
    _upload_counts: dict[str, list] = {}
    _upload_counts_lock = threading.Lock()
    MAX_UPLOADS_PER_MINUTE = 5
    MAX_UPLOADS_PER_HOUR = 30

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in self.ROUTES:
            return True
        # Handle /api/documents/{doc_id} pattern
        if path.startswith("/api/documents/") and path.count("/") == 3:
            return True
        return False

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route GET document requests to appropriate methods."""
        if path == "/api/documents":
            return self._list_documents()

        if path == "/api/documents/formats":
            return self._get_supported_formats()

        if path.startswith("/api/documents/") and not path.endswith("/upload"):
            # Extract doc_id from /api/documents/{doc_id}
            doc_id, err = self.extract_path_param(path, 2, "document_id")
            if err:
                return err
            return self._get_document(doc_id)

        return None

    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route POST document requests to appropriate methods."""
        if path == "/api/documents/upload":
            return self._upload_document(handler)
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

    def _check_upload_rate_limit(self, handler) -> Optional[HandlerResult]:
        """Check IP-based upload rate limit.

        Returns error response if rate limited, None if allowed.
        """
        # Get client IP
        client_ip = self._get_client_ip(handler)

        now = time.time()
        one_minute_ago = now - 60
        one_hour_ago = now - 3600

        with DocumentHandler._upload_counts_lock:
            if client_ip not in DocumentHandler._upload_counts:
                DocumentHandler._upload_counts[client_ip] = []

            # Clean up old entries
            DocumentHandler._upload_counts[client_ip] = [
                ts for ts in DocumentHandler._upload_counts[client_ip]
                if ts > one_hour_ago
            ]

            timestamps = DocumentHandler._upload_counts[client_ip]

            # Check per-minute limit
            recent_minute = sum(1 for ts in timestamps if ts > one_minute_ago)
            if recent_minute >= DocumentHandler.MAX_UPLOADS_PER_MINUTE:
                return error_response(
                    f"Upload rate limit exceeded. Max {DocumentHandler.MAX_UPLOADS_PER_MINUTE} per minute.",
                    429
                )

            # Check per-hour limit
            if len(timestamps) >= DocumentHandler.MAX_UPLOADS_PER_HOUR:
                return error_response(
                    f"Upload rate limit exceeded. Max {DocumentHandler.MAX_UPLOADS_PER_HOUR} per hour.",
                    429
                )

            # Record this upload
            DocumentHandler._upload_counts[client_ip].append(now)

        return None

    def _get_client_ip(self, handler) -> str:
        """Get client IP address, respecting trusted proxy headers."""
        remote_ip = handler.client_address[0] if hasattr(handler, 'client_address') else 'unknown'
        # For simplicity, just return remote IP (full proxy handling is in unified_server)
        return remote_ip

    @handle_errors("document upload")
    def _upload_document(self, handler) -> HandlerResult:
        """Handle document upload. Rate limited by IP.

        Accepts multipart/form-data or raw file upload with X-Filename header.
        """
        # Check rate limit
        rate_limit_error = self._check_upload_rate_limit(handler)
        if rate_limit_error:
            return rate_limit_error

        store = self.get_document_store()
        if not store:
            return error_response("Document storage not configured", 500)

        # Get content length
        try:
            content_length = int(handler.headers.get('Content-Length', '0'))
        except ValueError:
            return error_response("Invalid Content-Length header", 400)

        if content_length == 0:
            return error_response("No content provided", 400)

        # Check max size (10MB)
        max_size = 10 * 1024 * 1024
        if content_length > max_size:
            return error_response("File too large. Max size: 10MB", 413)

        content_type = handler.headers.get('Content-Type', '')

        # Parse file from request
        file_content, filename = self._parse_upload(handler, content_type, content_length)

        if not file_content or not filename:
            return error_response("No file found in upload", 400)

        # Import document parsing
        try:
            from aragora.server.documents import parse_document, SUPPORTED_EXTENSIONS
        except ImportError:
            return error_response("Document parsing module not available", 500)

        # Validate file extension
        ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
        if ext not in SUPPORTED_EXTENSIONS:
            return json_response({
                "error": f"Unsupported file type: {ext}",
                "supported": list(SUPPORTED_EXTENSIONS)
            }, status=400)

        # Parse and store document
        from aragora.server.error_utils import safe_error_message
        try:
            doc = parse_document(file_content, filename)
            doc_id = store.add(doc)

            return json_response({
                "success": True,
                "document": {
                    "id": doc_id,
                    "filename": doc.filename,
                    "word_count": doc.word_count,
                    "page_count": doc.page_count,
                    "preview": doc.preview,
                }
            })
        except ImportError as e:
            return error_response(safe_error_message(e, "document_import"), 400)
        except Exception as e:
            return error_response(safe_error_message(e, "document_parsing"), 500)

    def _parse_upload(self, handler, content_type: str, content_length: int) -> tuple[Optional[bytes], Optional[str]]:
        """Parse file content and filename from upload request.

        Returns (file_content, filename) or (None, None) on failure.
        """
        if 'multipart/form-data' in content_type:
            return self._parse_multipart(handler, content_type, content_length)
        else:
            return self._parse_raw_upload(handler, content_length)

    def _parse_multipart(self, handler, content_type: str, content_length: int) -> tuple[Optional[bytes], Optional[str]]:
        """Parse multipart form data upload."""
        # Parse boundary
        boundary = None
        for part in content_type.split(';'):
            if 'boundary=' in part:
                parts = part.split('=', 1)
                if len(parts) == 2 and parts[1].strip():
                    boundary = parts[1].strip()
                break

        if not boundary:
            return None, None

        body = handler.rfile.read(content_length)
        boundary_bytes = f'--{boundary}'.encode()
        parts = body.split(boundary_bytes)

        # DoS protection
        if len(parts) > MAX_MULTIPART_PARTS:
            return None, None

        for part in parts:
            if b'Content-Disposition' not in part:
                continue

            try:
                header_end = part.index(b'\r\n\r\n')
                headers_raw = part[:header_end].decode('utf-8', errors='ignore')
                file_data = part[header_end + 4:]

                # Remove trailing boundary markers
                if file_data.endswith(b'--\r\n'):
                    file_data = file_data[:-4]
                elif file_data.endswith(b'\r\n'):
                    file_data = file_data[:-2]

                # Extract and sanitize filename
                if 'filename="' in headers_raw:
                    start = headers_raw.index('filename="') + 10
                    end = headers_raw.index('"', start)
                    raw_filename = headers_raw[start:end]
                    filename = os.path.basename(raw_filename)

                    # Reject suspicious patterns
                    if not filename or '\x00' in filename or '..' in filename:
                        continue
                    if filename.strip('.').strip() == '':
                        continue

                    return file_data, filename
            except (ValueError, IndexError):
                continue

        return None, None

    def _parse_raw_upload(self, handler, content_length: int) -> tuple[Optional[bytes], Optional[str]]:
        """Parse raw file upload with X-Filename header."""
        raw_filename = handler.headers.get('X-Filename', 'document.txt')
        filename = os.path.basename(raw_filename)

        # Reject suspicious patterns
        if not filename or '\x00' in filename or '..' in filename:
            return None, None

        file_content = handler.rfile.read(content_length)
        return file_content, filename
