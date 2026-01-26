"""
Document management endpoint handlers.

Endpoints:
- GET /api/documents - List all uploaded documents
- GET /api/documents/formats - Get supported file formats
- GET /api/documents/{doc_id} - Get a document by ID
- POST /api/documents/upload - Upload a document
- DELETE /api/documents/{doc_id} - Delete a document by ID
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from aragora.rbac.decorators import require_permission

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    require_user_auth,
    safe_error_message,
)

logger = logging.getLogger(__name__)

# Knowledge processing enabled by default (can be disabled via env var)
KNOWLEDGE_PROCESSING_DEFAULT = (
    os.environ.get("ARAGORA_KNOWLEDGE_AUTO_PROCESS", "true").lower() == "true"
)

# DoS protection
MAX_MULTIPART_PARTS = 10
MAX_FILENAME_LENGTH = 255
MIN_FILE_SIZE = 1  # Minimum 1 byte


class UploadErrorCode(Enum):
    """Specific error codes for upload failures."""

    RATE_LIMITED = "rate_limited"
    FILE_TOO_LARGE = "file_too_large"
    FILE_TOO_SMALL = "file_too_small"
    INVALID_CONTENT_LENGTH = "invalid_content_length"
    NO_CONTENT = "no_content"
    UNSUPPORTED_FORMAT = "unsupported_format"
    INVALID_FILENAME = "invalid_filename"
    FILENAME_TOO_LONG = "filename_too_long"
    CORRUPTED_UPLOAD = "corrupted_upload"
    MULTIPART_PARSE_ERROR = "multipart_parse_error"
    MISSING_BOUNDARY = "missing_boundary"
    STORAGE_NOT_CONFIGURED = "storage_not_configured"
    PARSING_FAILED = "parsing_failed"
    STORAGE_FAILED = "storage_failed"


@dataclass
class UploadError:
    """Structured upload error response."""

    code: UploadErrorCode
    message: str
    details: Optional[dict] = None

    def to_response(self, status: int = 400) -> HandlerResult:
        """Convert to error response."""
        payload: dict[str, Any] = {
            "error": self.message,
            "error_code": self.code.value,
        }
        if self.details:
            payload["details"] = self.details
        return json_response(payload, status=status)


class DocumentHandler(BaseHandler):
    """Handler for document-related endpoints."""

    ROUTES = [
        "/api/v1/documents",
        "/api/v1/documents/formats",
        "/api/v1/documents/upload",
    ]

    # Upload rate limiting (IP-based with LRU eviction)
    _upload_counts: OrderedDict[str, list] = OrderedDict()
    _upload_counts_lock = threading.Lock()
    MAX_UPLOADS_PER_MINUTE = 5
    MAX_UPLOADS_PER_HOUR = 30
    MAX_TRACKED_IPS = 10000  # Prevent unbounded memory growth

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in self.ROUTES:
            return True
        # Handle /api/documents/{doc_id} pattern
        if path.startswith("/api/v1/documents/") and path.count("/") == 4:
            return True
        return False

    @require_permission("documents:read")
    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route GET document requests to appropriate methods."""
        if path == "/api/v1/documents":
            return self._list_documents()

        if path == "/api/v1/documents/formats":
            return self._get_supported_formats()

        if path.startswith("/api/v1/documents/") and not path.endswith("/upload"):
            # Extract doc_id from /api/documents/{doc_id}
            doc_id, err = self.extract_path_param(path, 3, "document_id")
            if err:
                return err
            return self._get_document(doc_id)

        return None

    @require_permission("documents:create")
    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route POST document requests to appropriate methods."""
        if path == "/api/v1/documents/upload":
            # Extract knowledge processing options from query params
            process_knowledge = query_params.get("process_knowledge", [None])[0]
            if process_knowledge is None:
                process_knowledge = KNOWLEDGE_PROCESSING_DEFAULT
            else:
                process_knowledge = process_knowledge.lower() == "true"

            workspace_id = query_params.get("workspace_id", ["default"])[0]

            return self._upload_document(
                handler,
                process_knowledge=process_knowledge,
                workspace_id=workspace_id,
            )
        return None

    @require_permission("documents:delete")
    def handle_delete(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Route DELETE document requests to appropriate methods."""
        if path.startswith("/api/v1/documents/") and not path.endswith("/upload"):
            # Extract doc_id from /api/documents/{doc_id}
            doc_id, err = self.extract_path_param(path, 3, "document_id")
            if err:
                return err
            return self._delete_document(doc_id)
        return None

    def _delete_document(self, doc_id: str) -> HandlerResult:
        """Delete a document by ID."""
        store = self.get_document_store()
        if not store:
            return error_response("Document storage not configured", 500)

        try:
            # Check if document exists
            doc = store.get(doc_id)
            if not doc:
                return error_response(f"Document not found: {doc_id}", 404)

            # Delete the document
            success = store.delete(doc_id)
            if success:
                logger.info(f"Document deleted: {doc_id}")
                return json_response(
                    {"success": True, "message": f"Document {doc_id} deleted successfully"}
                )
            else:
                return error_response(f"Failed to delete document: {doc_id}", 500)
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return error_response(safe_error_message(e, "delete document"), 500)

    def get_document_store(self):
        """Get document store instance."""
        return self.ctx.get("document_store")

    def _list_documents(self) -> HandlerResult:
        """List all uploaded documents."""
        store = self.get_document_store()
        if not store:
            return json_response(
                {"documents": [], "count": 0, "error": "Document storage not configured"}
            )

        try:
            docs = store.list_all()
            return json_response({"documents": docs, "count": len(docs)})
        except Exception as e:
            return error_response(safe_error_message(e, "list documents"), 500)

    def _get_supported_formats(self) -> HandlerResult:
        """Get list of supported document formats."""
        try:
            from aragora.server.documents import get_supported_formats

            formats = get_supported_formats()
            return json_response(formats)
        except ImportError:
            return json_response(
                {
                    "extensions": [".txt", ".md", ".pdf"],
                    "note": "Document parsing module not fully loaded",
                }
            )

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
            return error_response(safe_error_message(e, "get document"), 500)

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
            else:
                # Move to end for LRU tracking
                DocumentHandler._upload_counts.move_to_end(client_ip)

            # Clean up old entries for this IP
            DocumentHandler._upload_counts[client_ip] = [
                ts for ts in DocumentHandler._upload_counts[client_ip] if ts > one_hour_ago
            ]

            # Remove empty entries to prevent memory leak
            if not DocumentHandler._upload_counts[client_ip]:
                del DocumentHandler._upload_counts[client_ip]
                # Re-add for the new upload
                DocumentHandler._upload_counts[client_ip] = []

            timestamps = DocumentHandler._upload_counts[client_ip]

            # Check per-minute limit
            recent_minute = sum(1 for ts in timestamps if ts > one_minute_ago)
            if recent_minute >= DocumentHandler.MAX_UPLOADS_PER_MINUTE:
                return error_response(
                    f"Upload rate limit exceeded. Max {DocumentHandler.MAX_UPLOADS_PER_MINUTE} per minute.",
                    429,
                )

            # Check per-hour limit
            if len(timestamps) >= DocumentHandler.MAX_UPLOADS_PER_HOUR:
                return error_response(
                    f"Upload rate limit exceeded. Max {DocumentHandler.MAX_UPLOADS_PER_HOUR} per hour.",
                    429,
                )

            # Enforce max tracked IPs (LRU eviction)
            while len(DocumentHandler._upload_counts) > DocumentHandler.MAX_TRACKED_IPS:
                # Remove oldest (first) entry
                DocumentHandler._upload_counts.popitem(last=False)

            # Record this upload
            DocumentHandler._upload_counts[client_ip].append(now)

        return None

    def _get_client_ip(self, handler) -> str:
        """Get client IP address, respecting trusted proxy headers."""
        remote_ip = handler.client_address[0] if hasattr(handler, "client_address") else "unknown"
        # For simplicity, just return remote IP (full proxy handling is in unified_server)
        return remote_ip

    @require_user_auth
    @handle_errors("document upload")
    def _upload_document(
        self,
        handler,
        user=None,
        process_knowledge: bool = True,
        workspace_id: str = "default",
    ) -> HandlerResult:
        """Handle document upload. Rate limited by IP.

        Accepts multipart/form-data or raw file upload with X-Filename header.
        Returns structured error codes for client handling.

        Args:
            handler: HTTP request handler
            user: Authenticated user (from decorator)
            process_knowledge: Whether to process through knowledge pipeline
            workspace_id: Workspace ID for knowledge processing
        """
        # Check rate limit
        rate_limit_error = self._check_upload_rate_limit(handler)
        if rate_limit_error:
            return rate_limit_error

        store = self.get_document_store()
        if not store:
            return UploadError(
                UploadErrorCode.STORAGE_NOT_CONFIGURED, "Document storage not configured"
            ).to_response(500)

        # Get and validate content length
        try:
            content_length = int(handler.headers.get("Content-Length", "0"))
        except ValueError:
            return UploadError(
                UploadErrorCode.INVALID_CONTENT_LENGTH,
                "Invalid Content-Length header",
                {"header_value": handler.headers.get("Content-Length", "")},
            ).to_response(400)

        if content_length == 0:
            return UploadError(
                UploadErrorCode.NO_CONTENT,
                "No content provided. Include file data in request body.",
            ).to_response(400)

        if content_length < MIN_FILE_SIZE:
            return UploadError(
                UploadErrorCode.FILE_TOO_SMALL,
                f"File too small. Minimum size: {MIN_FILE_SIZE} bytes",
                {"received_bytes": content_length},
            ).to_response(400)

        # Check max size (10MB)
        max_size = 10 * 1024 * 1024
        if content_length > max_size:
            return UploadError(
                UploadErrorCode.FILE_TOO_LARGE,
                "File too large. Maximum size: 10MB",
                {"received_bytes": content_length, "max_bytes": max_size},
            ).to_response(413)

        content_type = handler.headers.get("Content-Type", "")

        # Parse file from request
        file_content, filename, parse_error = self._parse_upload_with_error(
            handler, content_type, content_length
        )

        if parse_error:
            return parse_error.to_response(400)

        if not file_content or not filename:
            return UploadError(
                UploadErrorCode.CORRUPTED_UPLOAD,
                "Could not extract file from upload. Ensure file is included in request.",
            ).to_response(400)

        # Validate filename length
        if len(filename) > MAX_FILENAME_LENGTH:
            return UploadError(
                UploadErrorCode.FILENAME_TOO_LONG,
                f"Filename too long. Maximum: {MAX_FILENAME_LENGTH} characters",
                {"filename_length": len(filename), "max_length": MAX_FILENAME_LENGTH},
            ).to_response(400)

        # Verify actual content matches declared length (detect truncation)
        if len(file_content) != content_length:
            return UploadError(
                UploadErrorCode.CORRUPTED_UPLOAD,
                "Upload appears truncated or corrupted",
                {"expected_bytes": content_length, "received_bytes": len(file_content)},
            ).to_response(400)

        # Import document parsing
        try:
            from aragora.server.documents import SUPPORTED_EXTENSIONS, parse_document
        except ImportError:
            return UploadError(
                UploadErrorCode.PARSING_FAILED, "Document parsing module not available"
            ).to_response(500)

        # Validate file extension
        ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
        if ext not in SUPPORTED_EXTENSIONS:
            return UploadError(
                UploadErrorCode.UNSUPPORTED_FORMAT,
                f"Unsupported file type: {ext}",
                {"extension": ext, "supported": list(SUPPORTED_EXTENSIONS)},
            ).to_response(400)

        # Parse and store document
        from aragora.server.errors import safe_error_message

        try:
            doc = parse_document(file_content, filename)
            doc_id = store.add(doc)

            logger.info(f"Document uploaded: {filename} ({len(file_content)} bytes) -> {doc_id}")

            # Build response
            response_data: dict[str, Any] = {
                "success": True,
                "document": {
                    "id": doc_id,
                    "filename": doc.filename,
                    "word_count": doc.word_count,
                    "page_count": doc.page_count,
                    "preview": doc.preview,
                },
            }

            # Process through knowledge pipeline if enabled
            if process_knowledge:
                try:
                    from aragora.knowledge.integration import process_uploaded_document

                    knowledge_result = process_uploaded_document(
                        content=file_content,
                        filename=filename,
                        workspace_id=workspace_id,
                        document_id=doc_id,
                        async_processing=True,  # Queue for background processing
                    )
                    response_data.update(knowledge_result)
                    logger.info(
                        f"Knowledge processing queued for {doc_id}: "
                        f"{knowledge_result.get('knowledge_processing', {}).get('job_id', 'N/A')}"
                    )
                except ImportError:
                    logger.warning("Knowledge pipeline not available, skipping")
                except Exception as ke:
                    logger.warning(f"Knowledge processing failed, document still uploaded: {ke}")
                    response_data["knowledge_processing"] = {
                        "status": "failed",
                        "error": str(ke)[:200],
                    }

            return json_response(response_data)
        except ImportError as e:
            logger.error(f"Document import error: {e}")
            return UploadError(
                UploadErrorCode.PARSING_FAILED,
                safe_error_message(e, "document_import"),
                {"error_type": "ImportError"},
            ).to_response(400)
        except ValueError as e:
            # Common for malformed PDFs, etc
            logger.warning(f"Document parse error for {filename}: {e}")
            return UploadError(
                UploadErrorCode.PARSING_FAILED,
                f"Could not parse document: {safe_error_message(e, 'document_parsing')}",
                {"filename": filename, "error_type": "ValueError"},
            ).to_response(400)
        except Exception as e:
            logger.error(f"Document storage error: {e}")
            return UploadError(
                UploadErrorCode.STORAGE_FAILED,
                safe_error_message(e, "document_storage"),
                {"error_type": type(e).__name__},
            ).to_response(500)

    def _parse_upload_with_error(
        self, handler, content_type: str, content_length: int
    ) -> tuple[Optional[bytes], Optional[str], Optional[UploadError]]:
        """Parse file content and filename from upload request with detailed errors.

        Returns (file_content, filename, error) tuple.
        On success: (content, filename, None)
        On failure: (None, None, UploadError)
        """
        if "multipart/form-data" in content_type:
            return self._parse_multipart_with_error(handler, content_type, content_length)
        else:
            return self._parse_raw_upload_with_error(handler, content_length)

    def _parse_upload(
        self, handler, content_type: str, content_length: int
    ) -> tuple[Optional[bytes], Optional[str]]:
        """Parse file content and filename from upload request.

        Returns (file_content, filename) or (None, None) on failure.
        Legacy method - use _parse_upload_with_error for better error handling.
        """
        if "multipart/form-data" in content_type:
            content, filename, _ = self._parse_multipart_with_error(
                handler, content_type, content_length
            )
            return content, filename
        else:
            content, filename, _ = self._parse_raw_upload_with_error(handler, content_length)
            return content, filename

    def _parse_multipart_with_error(
        self, handler, content_type: str, content_length: int
    ) -> tuple[Optional[bytes], Optional[str], Optional[UploadError]]:
        """Parse multipart form data upload with detailed errors."""
        # Parse boundary
        boundary = None
        for part in content_type.split(";"):
            if "boundary=" in part:
                parts = part.split("=", 1)
                if len(parts) == 2 and parts[1].strip():
                    boundary = parts[1].strip().strip('"')
                break

        if not boundary:
            return (
                None,
                None,
                UploadError(
                    UploadErrorCode.MISSING_BOUNDARY,
                    "Missing boundary in multipart/form-data Content-Type header",
                    {"content_type": content_type},
                ),
            )

        try:
            body: bytes = handler.rfile.read(content_length)
        except Exception as e:
            return (
                None,
                None,
                UploadError(
                    UploadErrorCode.CORRUPTED_UPLOAD, f"Failed to read upload body: {str(e)[:100]}"
                ),
            )

        boundary_bytes = f"--{boundary}".encode()
        body_parts: list[bytes] = body.split(boundary_bytes)

        # DoS protection
        if len(body_parts) > MAX_MULTIPART_PARTS:
            return (
                None,
                None,
                UploadError(
                    UploadErrorCode.MULTIPART_PARSE_ERROR,
                    f"Too many parts in multipart upload. Maximum: {MAX_MULTIPART_PARTS}",
                    {"part_count": len(body_parts), "max_parts": MAX_MULTIPART_PARTS},
                ),
            )

        for part in body_parts:
            if b"Content-Disposition" not in part:
                continue

            try:
                header_end = part.index(b"\r\n\r\n")
                headers_raw = part[:header_end].decode("utf-8", errors="ignore")
                file_data = part[header_end + 4 :]

                # Remove trailing boundary markers
                if file_data.endswith(b"--\r\n"):
                    file_data = file_data[:-4]
                elif file_data.endswith(b"\r\n"):
                    file_data = file_data[:-2]

                # Extract and sanitize filename
                if 'filename="' in headers_raw:
                    start = headers_raw.index('filename="') + 10
                    end = headers_raw.index('"', start)
                    raw_filename = headers_raw[start:end]
                    filename = os.path.basename(raw_filename)

                    # Reject suspicious patterns with specific errors
                    if not filename:
                        return (
                            None,
                            None,
                            UploadError(
                                UploadErrorCode.INVALID_FILENAME, "Empty filename in upload"
                            ),
                        )
                    if "\x00" in filename:
                        return (
                            None,
                            None,
                            UploadError(
                                UploadErrorCode.INVALID_FILENAME,
                                "Filename contains null bytes (potential attack)",
                            ),
                        )
                    if ".." in filename:
                        return (
                            None,
                            None,
                            UploadError(
                                UploadErrorCode.INVALID_FILENAME,
                                "Filename contains path traversal sequence (..)",
                            ),
                        )
                    if filename.strip(".").strip() == "":
                        return (
                            None,
                            None,
                            UploadError(
                                UploadErrorCode.INVALID_FILENAME,
                                "Filename cannot be only dots or whitespace",
                            ),
                        )

                    return file_data, filename, None
            except (ValueError, IndexError) as e:
                logger.debug(f"Multipart part parse error: {e}")
                continue

        return (
            None,
            None,
            UploadError(
                UploadErrorCode.MULTIPART_PARSE_ERROR,
                "No valid file found in multipart upload. Ensure field name is 'file' and filename is provided.",
            ),
        )

    def _parse_multipart(
        self, handler, content_type: str, content_length: int
    ) -> tuple[Optional[bytes], Optional[str]]:
        """Parse multipart form data upload (legacy)."""
        content, filename, _ = self._parse_multipart_with_error(
            handler, content_type, content_length
        )
        return content, filename

    def _parse_raw_upload_with_error(
        self, handler, content_length: int
    ) -> tuple[Optional[bytes], Optional[str], Optional[UploadError]]:
        """Parse raw file upload with X-Filename header (with detailed errors)."""
        raw_filename = handler.headers.get("X-Filename", "document.txt")
        filename = os.path.basename(raw_filename)

        # Reject suspicious patterns with specific errors
        if not filename:
            return (
                None,
                None,
                UploadError(
                    UploadErrorCode.INVALID_FILENAME,
                    "Empty filename. Provide X-Filename header with valid filename.",
                ),
            )
        if "\x00" in filename:
            return (
                None,
                None,
                UploadError(
                    UploadErrorCode.INVALID_FILENAME,
                    "Filename contains null bytes (potential attack)",
                ),
            )
        if ".." in filename:
            return (
                None,
                None,
                UploadError(
                    UploadErrorCode.INVALID_FILENAME,
                    "Filename contains path traversal sequence (..)",
                ),
            )

        try:
            file_content = handler.rfile.read(content_length)
        except Exception as e:
            return (
                None,
                None,
                UploadError(
                    UploadErrorCode.CORRUPTED_UPLOAD, f"Failed to read upload body: {str(e)[:100]}"
                ),
            )

        return file_content, filename, None

    def _parse_raw_upload(
        self, handler, content_length: int
    ) -> tuple[Optional[bytes], Optional[str]]:
        """Parse raw file upload with X-Filename header (legacy)."""
        content, filename, _ = self._parse_raw_upload_with_error(handler, content_length)
        return content, filename
