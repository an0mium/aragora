"""
Cloud Storage Handler for Aragora.

Provides REST API endpoints for cloud storage operations:
- Upload files to cloud storage (S3, GCS, Azure Blob)
- Download files from cloud storage
- List files with filtering and pagination
- Delete files
- Get file metadata
- Generate presigned URLs
- Manage storage quotas

Endpoints:
    GET  /api/v2/storage/files                    - List files with filtering
    POST /api/v2/storage/files                    - Upload a file
    GET  /api/v2/storage/files/:file_id           - Get file metadata
    GET  /api/v2/storage/files/:file_id/download  - Download file
    DELETE /api/v2/storage/files/:file_id         - Delete a file
    POST /api/v2/storage/files/:file_id/presign   - Generate presigned URL
    GET  /api/v2/storage/quota                    - Get storage quota usage
    GET  /api/v2/storage/buckets                  - List available buckets
    POST /api/v2/storage/buckets                  - Create a bucket
    DELETE /api/v2/storage/buckets/:bucket_id     - Delete a bucket

Stability: STABLE
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from aragora.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    get_circuit_breaker,
)
from aragora.server.errors import safe_error_message
from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)
from aragora.server.handlers.utils.decorators import require_permission
from aragora.server.handlers.utils.rate_limit import rate_limit
from aragora.server.validation.query_params import safe_query_int

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Configuration
# =============================================================================

# Maximum file size (100MB default, configurable via env)
MAX_FILE_SIZE_BYTES = int(os.environ.get("ARAGORA_MAX_UPLOAD_SIZE", 100 * 1024 * 1024))

# Allowed file extensions (configurable)
ALLOWED_EXTENSIONS = frozenset(
    os.environ.get(
        "ARAGORA_ALLOWED_EXTENSIONS",
        ".txt,.pdf,.doc,.docx,.xls,.xlsx,.csv,.json,.xml,.png,.jpg,.jpeg,.gif,.mp3,.mp4,.wav,.zip,.tar,.gz",
    ).split(",")
)

# Circuit breaker configuration for cloud storage
CLOUD_STORAGE_CB_NAME = "cloud_storage"
CLOUD_STORAGE_CB_FAILURE_THRESHOLD = 5
CLOUD_STORAGE_CB_COOLDOWN_SECONDS = 30

# Safe filename pattern
SAFE_FILENAME_PATTERN = re.compile(r"^[\w\-. ]+$")


class StorageProvider(str, Enum):
    """Supported cloud storage providers."""

    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"
    LOCAL = "local"


class FileStatus(str, Enum):
    """File status values."""

    PENDING = "pending"
    UPLOADING = "uploading"
    AVAILABLE = "available"
    DELETED = "deleted"
    FAILED = "failed"


@dataclass
class FileMetadata:
    """Metadata for a stored file."""

    id: str
    filename: str
    original_filename: str
    content_type: str
    size_bytes: int
    checksum: str
    bucket: str
    path: str
    status: FileStatus
    created_at: datetime
    updated_at: datetime
    owner_id: str
    tenant_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    expires_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "filename": self.filename,
            "original_filename": self.original_filename,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "bucket": self.bucket,
            "path": self.path,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "owner_id": self.owner_id,
            "tenant_id": self.tenant_id,
            "metadata": self.metadata,
            "tags": self.tags,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass
class BucketInfo:
    """Information about a storage bucket."""

    id: str
    name: str
    provider: StorageProvider
    region: str
    created_at: datetime
    owner_id: str
    tenant_id: Optional[str] = None
    is_public: bool = False
    versioning_enabled: bool = False
    lifecycle_rules: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider.value,
            "region": self.region,
            "created_at": self.created_at.isoformat(),
            "owner_id": self.owner_id,
            "tenant_id": self.tenant_id,
            "is_public": self.is_public,
            "versioning_enabled": self.versioning_enabled,
            "lifecycle_rules": self.lifecycle_rules,
        }


@dataclass
class StorageQuota:
    """Storage quota information."""

    total_bytes: int
    used_bytes: int
    file_count: int
    max_file_size_bytes: int
    tenant_id: Optional[str] = None

    @property
    def available_bytes(self) -> int:
        """Calculate available storage."""
        return max(0, self.total_bytes - self.used_bytes)

    @property
    def usage_percent(self) -> float:
        """Calculate usage percentage."""
        if self.total_bytes == 0:
            return 0.0
        return (self.used_bytes / self.total_bytes) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_bytes": self.total_bytes,
            "used_bytes": self.used_bytes,
            "available_bytes": self.available_bytes,
            "file_count": self.file_count,
            "max_file_size_bytes": self.max_file_size_bytes,
            "usage_percent": round(self.usage_percent, 2),
            "tenant_id": self.tenant_id,
        }


@runtime_checkable
class CloudStorageBackend(Protocol):
    """Protocol for cloud storage backend implementations."""

    async def upload_file(
        self,
        bucket: str,
        path: str,
        data: bytes,
        content_type: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Upload a file and return the storage URL."""
        ...

    async def download_file(self, bucket: str, path: str) -> bytes:
        """Download a file and return its contents."""
        ...

    async def delete_file(self, bucket: str, path: str) -> bool:
        """Delete a file and return success status."""
        ...

    async def file_exists(self, bucket: str, path: str) -> bool:
        """Check if a file exists."""
        ...

    async def get_presigned_url(
        self,
        bucket: str,
        path: str,
        expires_in_seconds: int = 3600,
        method: str = "GET",
    ) -> str:
        """Generate a presigned URL for the file."""
        ...

    async def list_files(
        self,
        bucket: str,
        prefix: str | None = None,
        limit: int = 100,
        continuation_token: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """List files in a bucket with optional prefix filter."""
        ...


class LocalStorageBackend:
    """Local filesystem storage backend for development/testing."""

    def __init__(self, base_path: str | Path | None = None):
        """Initialize local storage backend."""
        self.base_path = Path(
            base_path or os.environ.get("ARAGORA_STORAGE_PATH", "/tmp/aragora_storage")
        )
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def upload_file(
        self,
        bucket: str,
        path: str,
        data: bytes,
        content_type: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Upload a file to local storage."""
        bucket_path = self.base_path / bucket
        bucket_path.mkdir(parents=True, exist_ok=True)
        file_path = bucket_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(data)
        return f"file://{file_path}"

    async def download_file(self, bucket: str, path: str) -> bytes:
        """Download a file from local storage."""
        file_path = self.base_path / bucket / path
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {bucket}/{path}")
        return file_path.read_bytes()

    async def delete_file(self, bucket: str, path: str) -> bool:
        """Delete a file from local storage."""
        file_path = self.base_path / bucket / path
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    async def file_exists(self, bucket: str, path: str) -> bool:
        """Check if a file exists in local storage."""
        file_path = self.base_path / bucket / path
        return file_path.exists()

    async def get_presigned_url(
        self,
        bucket: str,
        path: str,
        expires_in_seconds: int = 3600,
        method: str = "GET",
    ) -> str:
        """Generate a presigned URL (returns local file path for local storage)."""
        file_path = self.base_path / bucket / path
        return f"file://{file_path}"

    async def list_files(
        self,
        bucket: str,
        prefix: str | None = None,
        limit: int = 100,
        continuation_token: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """List files in local storage."""
        bucket_path = self.base_path / bucket
        if not bucket_path.exists():
            return [], None

        files = []
        for file_path in bucket_path.rglob("*"):
            if file_path.is_file():
                rel_path = str(file_path.relative_to(bucket_path))
                if prefix and not rel_path.startswith(prefix):
                    continue
                stat = file_path.stat()
                files.append(
                    {
                        "path": rel_path,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(
                            stat.st_mtime, tz=timezone.utc
                        ).isoformat(),
                    }
                )
                if len(files) >= limit:
                    break

        return files, None


class CloudStorageHandler(BaseHandler):
    """
    HTTP handler for cloud storage operations.

    Provides REST API access to cloud storage with circuit breaker
    protection and rate limiting.

    Stability: STABLE
    """

    ROUTES = [
        "/api/v2/storage/files",
        "/api/v2/storage/files/*",
        "/api/v2/storage/quota",
        "/api/v2/storage/buckets",
        "/api/v2/storage/buckets/*",
    ]

    def __init__(self, server_context: dict[str, Any]):
        """Initialize with server context."""
        super().__init__(server_context)
        self._backend: CloudStorageBackend | None = None
        self._files: dict[str, FileMetadata] = {}
        self._buckets: dict[str, BucketInfo] = {}
        self._circuit_breaker: CircuitBreaker | None = None

    def _get_backend(self) -> CloudStorageBackend:
        """Get or create storage backend (lazy initialization)."""
        if self._backend is None:
            # Check for configured backend in context
            backend = self.ctx.get("cloud_storage_backend")
            if backend is not None:
                self._backend = backend
            else:
                # Default to local storage for development
                self._backend = LocalStorageBackend()
        return self._backend

    def _get_circuit_breaker(self) -> CircuitBreaker:
        """Get or create circuit breaker for cloud storage operations."""
        if self._circuit_breaker is None:
            self._circuit_breaker = get_circuit_breaker(
                CLOUD_STORAGE_CB_NAME,
                failure_threshold=CLOUD_STORAGE_CB_FAILURE_THRESHOLD,
                cooldown_seconds=CLOUD_STORAGE_CB_COOLDOWN_SECONDS,
            )
        return self._circuit_breaker

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the request."""
        if path.startswith("/api/v2/storage/"):
            return method in ("GET", "POST", "DELETE")
        return False

    def _validate_filename(self, filename: str) -> tuple[bool, str]:
        """Validate a filename for safety."""
        if not filename:
            return False, "Filename is required"
        if len(filename) > 255:
            return False, "Filename too long (max 255 characters)"
        if not SAFE_FILENAME_PATTERN.match(filename):
            return False, "Filename contains invalid characters"
        ext = Path(filename).suffix.lower()
        if ext and ext not in ALLOWED_EXTENSIONS:
            return False, f"File extension not allowed: {ext}"
        return True, ""

    def _generate_file_id(self) -> str:
        """Generate a unique file ID."""
        return f"file_{uuid.uuid4().hex[:16]}"

    def _generate_bucket_id(self) -> str:
        """Generate a unique bucket ID."""
        return f"bucket_{uuid.uuid4().hex[:12]}"

    def _compute_checksum(self, data: bytes) -> str:
        """Compute SHA-256 checksum of file data."""
        return f"sha256:{hashlib.sha256(data).hexdigest()}"

    @rate_limit(requests_per_minute=60)
    async def handle(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult | None:
        """Route GET requests to appropriate handler method."""
        method: str = getattr(handler, "command", "GET") if handler else "GET"
        if method != "GET":
            return None

        query_params = query_params or {}

        try:
            # Check circuit breaker
            cb = self._get_circuit_breaker()
            if not cb.can_proceed():
                logger.warning("Cloud storage circuit breaker is open")
                return error_response(
                    "Cloud storage service temporarily unavailable",
                    503,
                )

            # Quota endpoint
            if path == "/api/v2/storage/quota":
                return await self._get_quota(handler)

            # List buckets
            if path == "/api/v2/storage/buckets":
                return await self._list_buckets(query_params, handler)

            # List files
            if path == "/api/v2/storage/files":
                return await self._list_files(query_params, handler)

            # File-specific routes
            if path.startswith("/api/v2/storage/files/"):
                parts = path.split("/")
                # Path: /api/v2/storage/files/:file_id -> ["", "api", "v2", "storage", "files", file_id]
                if len(parts) < 6:
                    return error_response("Invalid file path", 400)

                file_id = parts[5]

                # Download endpoint
                if len(parts) > 6 and parts[6] == "download":
                    return await self._download_file(file_id, handler)

                # Get file metadata
                return await self._get_file(file_id, handler)

            # Bucket-specific routes
            if path.startswith("/api/v2/storage/buckets/"):
                parts = path.split("/")
                # Path: /api/v2/storage/buckets/:bucket_id -> ["", "api", "v2", "storage", "buckets", bucket_id]
                if len(parts) < 6:
                    return error_response("Invalid bucket path", 400)

                bucket_id = parts[5]
                return await self._get_bucket(bucket_id, handler)

            return None

        except CircuitOpenError:
            logger.warning("Cloud storage circuit breaker tripped")
            return error_response(
                "Cloud storage service temporarily unavailable",
                503,
            )
        except Exception as e:
            logger.exception(f"Error handling cloud storage GET request: {e}")
            cb = self._get_circuit_breaker()
            cb.record_failure()
            return error_response(safe_error_message(e, "cloud storage upload"), 500)

    @rate_limit(requests_per_minute=30)
    @require_permission("debates:write")
    async def handle_post(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult | None:
        """Route POST requests to appropriate handler method."""
        body = self.read_json_body(handler) or {}
        query_params = query_params or {}

        try:
            # Check circuit breaker
            cb = self._get_circuit_breaker()
            if not cb.can_proceed():
                logger.warning("Cloud storage circuit breaker is open")
                return error_response(
                    "Cloud storage service temporarily unavailable",
                    503,
                )

            # Upload file
            if path == "/api/v2/storage/files":
                return await self._upload_file(body, handler)

            # Create bucket
            if path == "/api/v2/storage/buckets":
                return await self._create_bucket(body, handler)

            # File-specific POST routes
            if path.startswith("/api/v2/storage/files/"):
                parts = path.split("/")
                # Path: /api/v2/storage/files/:file_id/presign -> ["", "api", "v2", "storage", "files", file_id, "presign"]
                if len(parts) < 6:
                    return error_response("Invalid file path", 400)

                file_id = parts[5]

                # Presigned URL endpoint
                if len(parts) > 6 and parts[6] == "presign":
                    return await self._generate_presigned_url(file_id, body, handler)

            return None

        except CircuitOpenError:
            logger.warning("Cloud storage circuit breaker tripped")
            return error_response(
                "Cloud storage service temporarily unavailable",
                503,
            )
        except Exception as e:
            logger.exception(f"Error handling cloud storage POST request: {e}")
            cb = self._get_circuit_breaker()
            cb.record_failure()
            return error_response(safe_error_message(e, "cloud storage list"), 500)

    @rate_limit(requests_per_minute=20)
    @require_permission("debates:delete")
    async def handle_delete(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult | None:
        """Route DELETE requests to appropriate handler method."""
        try:
            # Check circuit breaker
            cb = self._get_circuit_breaker()
            if not cb.can_proceed():
                logger.warning("Cloud storage circuit breaker is open")
                return error_response(
                    "Cloud storage service temporarily unavailable",
                    503,
                )

            # Delete file
            if path.startswith("/api/v2/storage/files/"):
                parts = path.split("/")
                # Path: /api/v2/storage/files/:file_id -> ["", "api", "v2", "storage", "files", file_id]
                if len(parts) < 6:
                    return error_response("Invalid file path", 400)
                file_id = parts[5]
                return await self._delete_file(file_id, handler)

            # Delete bucket
            if path.startswith("/api/v2/storage/buckets/"):
                parts = path.split("/")
                # Path: /api/v2/storage/buckets/:bucket_id -> ["", "api", "v2", "storage", "buckets", bucket_id]
                if len(parts) < 6:
                    return error_response("Invalid bucket path", 400)
                bucket_id = parts[5]
                return await self._delete_bucket(bucket_id, handler)

            return None

        except CircuitOpenError:
            logger.warning("Cloud storage circuit breaker tripped")
            return error_response(
                "Cloud storage service temporarily unavailable",
                503,
            )
        except Exception as e:
            logger.exception(f"Error handling cloud storage DELETE request: {e}")
            cb = self._get_circuit_breaker()
            cb.record_failure()
            return error_response(safe_error_message(e, "cloud storage delete"), 500)

    @require_permission("storage:read")
    async def _list_files(
        self,
        query_params: dict[str, str],
        handler: Any,
    ) -> HandlerResult:
        """List files with filtering and pagination."""
        bucket = query_params.get("bucket", "default")
        prefix = query_params.get("prefix")
        limit = safe_query_int(query_params, "limit", default=20, min_val=1, max_val=100)
        offset = safe_query_int(query_params, "offset", default=0, min_val=0, max_val=10000)

        # Filter files by bucket and prefix
        files = [
            f
            for f in self._files.values()
            if f.bucket == bucket and f.status == FileStatus.AVAILABLE
        ]

        if prefix:
            files = [f for f in files if f.path.startswith(prefix)]

        # Sort by created_at descending
        files.sort(key=lambda f: f.created_at, reverse=True)

        total = len(files)
        files = files[offset : offset + limit]

        # Record success
        cb = self._get_circuit_breaker()
        cb.record_success()

        return json_response(
            {
                "files": [f.to_dict() for f in files],
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": total,
                    "has_more": offset + len(files) < total,
                },
            }
        )

    @require_permission("storage:read")
    async def _get_file(self, file_id: str, handler: Any) -> HandlerResult:
        """Get file metadata by ID."""
        file_meta = self._files.get(file_id)
        if not file_meta:
            return error_response("File not found", 404)

        cb = self._get_circuit_breaker()
        cb.record_success()

        return json_response(file_meta.to_dict())

    @require_permission("storage:read")
    async def _download_file(self, file_id: str, handler: Any) -> HandlerResult:
        """Download a file."""
        file_meta = self._files.get(file_id)
        if not file_meta:
            return error_response("File not found", 404)

        if file_meta.status != FileStatus.AVAILABLE:
            return error_response(f"File not available (status: {file_meta.status.value})", 400)

        backend = self._get_backend()
        try:
            data = await backend.download_file(file_meta.bucket, file_meta.path)

            cb = self._get_circuit_breaker()
            cb.record_success()

            # Return file data with proper content type
            return HandlerResult(
                status_code=200,
                content_type=file_meta.content_type,
                body=data,
                headers={
                    "Content-Disposition": f'attachment; filename="{file_meta.original_filename}"',
                    "Content-Length": str(len(data)),
                },
            )
        except FileNotFoundError:
            return error_response("File not found in storage", 404)
        except Exception as e:
            cb = self._get_circuit_breaker()
            cb.record_failure()
            logger.error(f"Failed to download file {file_id}: {e}")
            return error_response("Failed to download file", 500)

    @require_permission("storage:write")
    async def _upload_file(self, body: dict[str, Any], handler: Any) -> HandlerResult:
        """Upload a new file."""
        filename = body.get("filename", "")
        content_b64 = body.get("content")  # Base64 encoded content
        bucket = body.get("bucket", "default")
        tags = body.get("tags", [])
        metadata = body.get("metadata", {})

        # Validate filename
        valid, err = self._validate_filename(filename)
        if not valid:
            return error_response(err, 400)

        # Decode content
        if not content_b64:
            return error_response("File content is required", 400)

        import base64

        try:
            data = base64.b64decode(content_b64)
        except Exception:
            return error_response("Invalid base64 content", 400)

        # Check file size
        if len(data) > MAX_FILE_SIZE_BYTES:
            return error_response(
                f"File too large (max {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB)",
                400,
            )

        # Generate file metadata
        file_id = self._generate_file_id()
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        checksum = self._compute_checksum(data)
        now = datetime.now(timezone.utc)
        path = f"{now.strftime('%Y/%m/%d')}/{file_id}/{filename}"

        # Get owner from auth context
        user = self.get_current_user(handler)
        owner_id = user.user_id if user else "anonymous"

        # Upload to backend
        backend = self._get_backend()
        try:
            await backend.upload_file(
                bucket=bucket,
                path=path,
                data=data,
                content_type=content_type,
                metadata={"owner": owner_id, "checksum": checksum},
            )
        except Exception as e:
            cb = self._get_circuit_breaker()
            cb.record_failure()
            logger.error(f"Failed to upload file: {e}")
            return error_response("Failed to upload file to storage", 500)

        # Create metadata record
        file_meta = FileMetadata(
            id=file_id,
            filename=filename,
            original_filename=filename,
            content_type=content_type,
            size_bytes=len(data),
            checksum=checksum,
            bucket=bucket,
            path=path,
            status=FileStatus.AVAILABLE,
            created_at=now,
            updated_at=now,
            owner_id=owner_id,
            metadata=metadata,
            tags=tags,
        )
        self._files[file_id] = file_meta

        cb = self._get_circuit_breaker()
        cb.record_success()

        return json_response(
            {
                "file": file_meta.to_dict(),
                "message": f"File uploaded: {file_id}",
            },
            status=201,
        )

    @require_permission("storage:delete")
    async def _delete_file(self, file_id: str, handler: Any) -> HandlerResult:
        """Delete a file."""
        file_meta = self._files.get(file_id)
        if not file_meta:
            return error_response("File not found", 404)

        backend = self._get_backend()
        try:
            await backend.delete_file(file_meta.bucket, file_meta.path)
        except Exception as e:
            cb = self._get_circuit_breaker()
            cb.record_failure()
            logger.error(f"Failed to delete file {file_id} from storage: {e}")
            # Continue to mark as deleted even if backend delete fails

        # Mark as deleted
        file_meta.status = FileStatus.DELETED
        file_meta.updated_at = datetime.now(timezone.utc)

        cb = self._get_circuit_breaker()
        cb.record_success()

        return json_response(
            {
                "deleted": True,
                "file_id": file_id,
                "message": f"File {file_id} deleted",
            }
        )

    @require_permission("storage:write")
    async def _generate_presigned_url(
        self,
        file_id: str,
        body: dict[str, Any],
        handler: Any,
    ) -> HandlerResult:
        """Generate a presigned URL for file access."""
        file_meta = self._files.get(file_id)
        if not file_meta:
            return error_response("File not found", 404)

        expires_in = body.get("expires_in_seconds", 3600)
        method = body.get("method", "GET").upper()

        if expires_in < 60 or expires_in > 86400:
            return error_response("expires_in_seconds must be between 60 and 86400", 400)

        if method not in ("GET", "PUT"):
            return error_response("method must be GET or PUT", 400)

        backend = self._get_backend()
        try:
            url = await backend.get_presigned_url(
                bucket=file_meta.bucket,
                path=file_meta.path,
                expires_in_seconds=expires_in,
                method=method,
            )
        except Exception as e:
            cb = self._get_circuit_breaker()
            cb.record_failure()
            logger.error(f"Failed to generate presigned URL: {e}")
            return error_response("Failed to generate presigned URL", 500)

        cb = self._get_circuit_breaker()
        cb.record_success()

        return json_response(
            {
                "file_id": file_id,
                "url": url,
                "expires_in_seconds": expires_in,
                "method": method,
            }
        )

    @require_permission("storage:read")
    async def _get_quota(self, handler: Any) -> HandlerResult:
        """Get storage quota usage."""
        # Calculate current usage
        used_bytes = sum(
            f.size_bytes for f in self._files.values() if f.status == FileStatus.AVAILABLE
        )
        file_count = sum(1 for f in self._files.values() if f.status == FileStatus.AVAILABLE)

        # Default quota (1GB, configurable)
        total_bytes = int(os.environ.get("ARAGORA_STORAGE_QUOTA_BYTES", 1024 * 1024 * 1024))

        quota = StorageQuota(
            total_bytes=total_bytes,
            used_bytes=used_bytes,
            file_count=file_count,
            max_file_size_bytes=MAX_FILE_SIZE_BYTES,
        )

        cb = self._get_circuit_breaker()
        cb.record_success()

        return json_response(
            {
                "quota": quota.to_dict(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    @require_permission("storage:read")
    async def _list_buckets(
        self,
        query_params: dict[str, str],
        handler: Any,
    ) -> HandlerResult:
        """List available buckets."""
        buckets = list(self._buckets.values())

        # Add default bucket if not exists
        if not buckets:
            default_bucket = BucketInfo(
                id="default",
                name="default",
                provider=StorageProvider.LOCAL,
                region="local",
                created_at=datetime.now(timezone.utc),
                owner_id="system",
            )
            buckets = [default_bucket]

        cb = self._get_circuit_breaker()
        cb.record_success()

        return json_response(
            {
                "buckets": [b.to_dict() for b in buckets],
                "count": len(buckets),
            }
        )

    @require_permission("storage:read")
    async def _get_bucket(self, bucket_id: str, handler: Any) -> HandlerResult:
        """Get bucket information by ID."""
        bucket = self._buckets.get(bucket_id)
        if not bucket:
            # Check for default bucket
            if bucket_id == "default":
                bucket = BucketInfo(
                    id="default",
                    name="default",
                    provider=StorageProvider.LOCAL,
                    region="local",
                    created_at=datetime.now(timezone.utc),
                    owner_id="system",
                )
            else:
                return error_response("Bucket not found", 404)

        cb = self._get_circuit_breaker()
        cb.record_success()

        return json_response(bucket.to_dict())

    @require_permission("storage:admin")
    async def _create_bucket(self, body: dict[str, Any], handler: Any) -> HandlerResult:
        """Create a new bucket."""
        name = body.get("name", "").strip()
        if not name:
            return error_response("Bucket name is required", 400)

        if not re.match(r"^[a-z0-9][a-z0-9\-]{1,61}[a-z0-9]$", name):
            return error_response(
                "Bucket name must be 3-63 characters, lowercase letters, numbers, and hyphens",
                400,
            )

        # Check if bucket already exists
        if any(b.name == name for b in self._buckets.values()):
            return error_response(f"Bucket '{name}' already exists", 409)

        bucket_id = self._generate_bucket_id()
        user = self.get_current_user(handler)
        owner_id = user.user_id if user else "anonymous"

        bucket = BucketInfo(
            id=bucket_id,
            name=name,
            provider=StorageProvider(body.get("provider", "local")),
            region=body.get("region", "local"),
            created_at=datetime.now(timezone.utc),
            owner_id=owner_id,
            is_public=body.get("is_public", False),
            versioning_enabled=body.get("versioning_enabled", False),
        )
        self._buckets[bucket_id] = bucket

        cb = self._get_circuit_breaker()
        cb.record_success()

        return json_response(
            {
                "bucket": bucket.to_dict(),
                "message": f"Bucket created: {name}",
            },
            status=201,
        )

    @require_permission("storage:admin")
    async def _delete_bucket(self, bucket_id: str, handler: Any) -> HandlerResult:
        """Delete a bucket."""
        if bucket_id == "default":
            return error_response("Cannot delete default bucket", 400)

        bucket = self._buckets.get(bucket_id)
        if not bucket:
            return error_response("Bucket not found", 404)

        # Check if bucket has files
        files_in_bucket = [
            f
            for f in self._files.values()
            if f.bucket == bucket.name and f.status == FileStatus.AVAILABLE
        ]
        if files_in_bucket:
            return error_response(
                f"Bucket contains {len(files_in_bucket)} files. Delete files first.",
                400,
            )

        del self._buckets[bucket_id]

        cb = self._get_circuit_breaker()
        cb.record_success()

        return json_response(
            {
                "deleted": True,
                "bucket_id": bucket_id,
                "message": f"Bucket {bucket_id} deleted",
            }
        )


# Handler factory function for registration
def create_cloud_storage_handler(server_context: dict[str, Any]) -> CloudStorageHandler:
    """Factory function for handler registration."""
    return CloudStorageHandler(server_context)
