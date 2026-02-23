"""Tests for cloud storage handler (aragora/server/handlers/cloud_storage.py).

Covers all routes and behavior of the CloudStorageHandler class:
- can_handle() routing
- GET    /api/v2/storage/files              - List files with filtering
- POST   /api/v2/storage/files              - Upload a file
- GET    /api/v2/storage/files/:id          - Get file metadata
- GET    /api/v2/storage/files/:id/download - Download file
- DELETE /api/v2/storage/files/:id          - Delete a file
- POST   /api/v2/storage/files/:id/presign  - Generate presigned URL
- GET    /api/v2/storage/quota              - Get storage quota usage
- GET    /api/v2/storage/buckets            - List available buckets
- POST   /api/v2/storage/buckets            - Create a bucket
- DELETE /api/v2/storage/buckets/:id        - Delete a bucket
- Circuit breaker integration
- Validation and error paths
- Dataclass serialization
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.cloud_storage import (
    ALLOWED_EXTENSIONS,
    CLOUD_STORAGE_CB_NAME,
    MAX_FILE_SIZE_BYTES,
    SAFE_FILENAME_PATTERN,
    BucketInfo,
    CloudStorageHandler,
    FileMetadata,
    FileStatus,
    LocalStorageBackend,
    StorageProvider,
    StorageQuota,
    create_cloud_storage_handler,
)
from aragora.server.handlers.base import HandlerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: HandlerResult) -> dict:
    """Extract the JSON body from a HandlerResult."""
    if isinstance(result, HandlerResult):
        if isinstance(result.body, bytes):
            return json.loads(result.body.decode("utf-8"))
        return result.body
    if isinstance(result, dict):
        return result.get("body", result)
    return {}


def _status(result: HandlerResult) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, HandlerResult):
        return result.status_code
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return 200


class MockHTTPHandler:
    """Mock HTTP handler for testing (simulates BaseHTTPRequestHandler)."""

    def __init__(self, body: dict[str, Any] | None = None):
        self.rfile = MagicMock()
        self.command = "GET"
        self._body = body
        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers = {"Content-Length": str(len(body_bytes))}
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {"Content-Length": "2"}


def _make_handler(
    body: dict[str, Any] | None = None, method: str = "GET"
) -> MockHTTPHandler:
    """Create a MockHTTPHandler with optional body and method."""
    h = MockHTTPHandler(body=body)
    h.command = method
    return h


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a CloudStorageHandler with mock backend."""
    h = CloudStorageHandler(server_context={})
    # Pre-configure a mock backend to avoid filesystem access
    h._backend = AsyncMock()
    h._backend.upload_file = AsyncMock(return_value="file://test/path")
    h._backend.download_file = AsyncMock(return_value=b"file content")
    h._backend.delete_file = AsyncMock(return_value=True)
    h._backend.file_exists = AsyncMock(return_value=True)
    h._backend.get_presigned_url = AsyncMock(
        return_value="https://presigned.example.com/file"
    )
    h._backend.list_files = AsyncMock(return_value=([], None))
    return h


@pytest.fixture(autouse=True)
def _reset_circuit_breaker():
    """Reset circuit breakers between tests."""
    from aragora.resilience import reset_all_circuit_breakers

    reset_all_circuit_breakers()
    yield
    reset_all_circuit_breakers()


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    """Reset rate limiters between tests."""
    from aragora.server.handlers.utils.rate_limit import clear_all_limiters

    clear_all_limiters()
    yield
    clear_all_limiters()


def _sample_b64_content(data: bytes = b"Hello, World!") -> str:
    """Return base64-encoded content for upload tests."""
    return base64.b64encode(data).decode()


def _add_sample_file(handler, file_id="file_abc123", bucket="default", status=FileStatus.AVAILABLE):
    """Add a sample file to handler's internal file store."""
    now = datetime.now(timezone.utc)
    fm = FileMetadata(
        id=file_id,
        filename="test.txt",
        original_filename="test.txt",
        content_type="text/plain",
        size_bytes=100,
        checksum="sha256:abc123",
        bucket=bucket,
        path=f"2026/01/01/{file_id}/test.txt",
        status=status,
        created_at=now,
        updated_at=now,
        owner_id="test-user-001",
        metadata={"key": "value"},
        tags=["test"],
    )
    handler._files[file_id] = fm
    return fm


def _add_sample_bucket(handler, bucket_id="bucket_test123", name="test-bucket"):
    """Add a sample bucket to handler's internal bucket store."""
    now = datetime.now(timezone.utc)
    bi = BucketInfo(
        id=bucket_id,
        name=name,
        provider=StorageProvider.LOCAL,
        region="local",
        created_at=now,
        owner_id="test-user-001",
    )
    handler._buckets[bucket_id] = bi
    return bi


# ============================================================================
# can_handle routing
# ============================================================================


class TestCanHandle:
    """Verify that can_handle correctly accepts or rejects paths."""

    def test_files_path_get(self, handler):
        assert handler.can_handle("/api/v2/storage/files", "GET")

    def test_files_path_post(self, handler):
        assert handler.can_handle("/api/v2/storage/files", "POST")

    def test_files_path_delete(self, handler):
        assert handler.can_handle("/api/v2/storage/files/file_123", "DELETE")

    def test_quota_path(self, handler):
        assert handler.can_handle("/api/v2/storage/quota", "GET")

    def test_buckets_path(self, handler):
        assert handler.can_handle("/api/v2/storage/buckets", "GET")

    def test_buckets_post(self, handler):
        assert handler.can_handle("/api/v2/storage/buckets", "POST")

    def test_bucket_delete(self, handler):
        assert handler.can_handle("/api/v2/storage/buckets/bucket_123", "DELETE")

    def test_file_download_path(self, handler):
        assert handler.can_handle("/api/v2/storage/files/file_123/download", "GET")

    def test_file_presign_path(self, handler):
        assert handler.can_handle("/api/v2/storage/files/file_123/presign", "POST")

    def test_rejects_v1_path(self, handler):
        assert not handler.can_handle("/api/v1/storage/files", "GET")

    def test_rejects_different_prefix(self, handler):
        assert not handler.can_handle("/api/v2/other/files", "GET")

    def test_rejects_unsupported_method(self, handler):
        assert not handler.can_handle("/api/v2/storage/files", "PUT")

    def test_rejects_patch_method(self, handler):
        assert not handler.can_handle("/api/v2/storage/files", "PATCH")

    def test_rejects_empty_path(self, handler):
        assert not handler.can_handle("", "GET")

    def test_rejects_root_path(self, handler):
        assert not handler.can_handle("/", "GET")

    def test_rejects_partial_prefix(self, handler):
        assert not handler.can_handle("/api/v2/storage", "GET")


# ============================================================================
# Initialization
# ============================================================================


class TestHandlerInit:
    """Test handler initialization."""

    def test_init_with_empty_context(self):
        h = CloudStorageHandler({})
        assert h.ctx == {}

    def test_init_with_server_context(self):
        ctx = {"some_key": "value"}
        h = CloudStorageHandler(server_context=ctx)
        assert h.ctx == ctx

    def test_routes_defined(self, handler):
        assert len(handler.ROUTES) > 0
        assert "/api/v2/storage/files" in handler.ROUTES

    def test_initial_state_empty(self):
        h = CloudStorageHandler({})
        assert h._files == {}
        assert h._buckets == {}
        assert h._circuit_breaker is None

    def test_factory_function(self):
        h = create_cloud_storage_handler({})
        assert isinstance(h, CloudStorageHandler)

    def test_backend_defaults_to_none(self):
        h = CloudStorageHandler({})
        assert h._backend is None

    def test_get_backend_creates_local_default(self):
        h = CloudStorageHandler({})
        backend = h._get_backend()
        assert isinstance(backend, LocalStorageBackend)

    def test_get_backend_from_context(self):
        mock_backend = AsyncMock()
        h = CloudStorageHandler({"cloud_storage_backend": mock_backend})
        backend = h._get_backend()
        assert backend is mock_backend

    def test_get_backend_caches(self):
        h = CloudStorageHandler({})
        b1 = h._get_backend()
        b2 = h._get_backend()
        assert b1 is b2


# ============================================================================
# Filename Validation
# ============================================================================


class TestFilenameValidation:
    """Test _validate_filename method."""

    def test_valid_filename(self, handler):
        valid, err = handler._validate_filename("document.pdf")
        assert valid is True
        assert err == ""

    def test_valid_filename_with_spaces(self, handler):
        valid, err = handler._validate_filename("my document.pdf")
        assert valid is True

    def test_valid_filename_with_dashes(self, handler):
        valid, err = handler._validate_filename("my-file_name.txt")
        assert valid is True

    def test_empty_filename(self, handler):
        valid, err = handler._validate_filename("")
        assert valid is False
        assert "required" in err.lower()

    def test_too_long_filename(self, handler):
        valid, err = handler._validate_filename("a" * 256 + ".txt")
        assert valid is False
        assert "too long" in err.lower()

    def test_max_length_filename(self, handler):
        # 255 chars total
        name = "a" * 251 + ".txt"
        valid, err = handler._validate_filename(name)
        assert valid is True

    def test_invalid_characters(self, handler):
        valid, err = handler._validate_filename("file<script>.txt")
        assert valid is False
        assert "invalid characters" in err.lower()

    def test_disallowed_extension(self, handler):
        valid, err = handler._validate_filename("script.exe")
        assert valid is False
        assert "not allowed" in err.lower()

    def test_allowed_extensions(self, handler):
        for ext in [".txt", ".pdf", ".json", ".csv", ".png"]:
            valid, err = handler._validate_filename(f"file{ext}")
            assert valid is True, f"Extension {ext} should be allowed"

    def test_no_extension(self, handler):
        valid, err = handler._validate_filename("README")
        assert valid is True


# ============================================================================
# Utility Methods
# ============================================================================


class TestUtilityMethods:
    """Test utility methods on the handler."""

    def test_generate_file_id(self, handler):
        fid = handler._generate_file_id()
        assert fid.startswith("file_")
        assert len(fid) > 5

    def test_file_ids_are_unique(self, handler):
        ids = {handler._generate_file_id() for _ in range(100)}
        assert len(ids) == 100

    def test_generate_bucket_id(self, handler):
        bid = handler._generate_bucket_id()
        assert bid.startswith("bucket_")

    def test_bucket_ids_are_unique(self, handler):
        ids = {handler._generate_bucket_id() for _ in range(100)}
        assert len(ids) == 100

    def test_compute_checksum(self, handler):
        cs = handler._compute_checksum(b"hello")
        assert cs.startswith("sha256:")
        assert len(cs) > 10

    def test_checksum_deterministic(self, handler):
        data = b"test data"
        assert handler._compute_checksum(data) == handler._compute_checksum(data)

    def test_checksum_differs(self, handler):
        assert handler._compute_checksum(b"a") != handler._compute_checksum(b"b")


# ============================================================================
# GET /api/v2/storage/files - List Files
# ============================================================================


class TestListFiles:
    """Test listing files with filtering and pagination."""

    @pytest.mark.asyncio
    async def test_list_empty(self, handler):
        h = _make_handler()
        result = await handler.handle("/api/v2/storage/files", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert body["files"] == []
        assert body["pagination"]["total"] == 0

    @pytest.mark.asyncio
    async def test_list_with_files(self, handler):
        _add_sample_file(handler, "file_1")
        _add_sample_file(handler, "file_2")
        h = _make_handler()
        result = await handler.handle("/api/v2/storage/files", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert len(body["files"]) == 2
        assert body["pagination"]["total"] == 2

    @pytest.mark.asyncio
    async def test_list_filters_by_bucket(self, handler):
        _add_sample_file(handler, "file_1", bucket="default")
        _add_sample_file(handler, "file_2", bucket="other")
        h = _make_handler()
        result = await handler.handle("/api/v2/storage/files", {"bucket": "default"}, h)
        body = _body(result)
        assert len(body["files"]) == 1

    @pytest.mark.asyncio
    async def test_list_filters_by_prefix(self, handler):
        fm1 = _add_sample_file(handler, "file_1")
        fm1.path = "images/photo.jpg"
        fm2 = _add_sample_file(handler, "file_2")
        fm2.path = "docs/readme.txt"
        h = _make_handler()
        result = await handler.handle(
            "/api/v2/storage/files", {"prefix": "images/"}, h
        )
        body = _body(result)
        assert len(body["files"]) == 1

    @pytest.mark.asyncio
    async def test_list_excludes_deleted(self, handler):
        _add_sample_file(handler, "file_1", status=FileStatus.AVAILABLE)
        _add_sample_file(handler, "file_2", status=FileStatus.DELETED)
        h = _make_handler()
        result = await handler.handle("/api/v2/storage/files", {}, h)
        body = _body(result)
        assert len(body["files"]) == 1

    @pytest.mark.asyncio
    async def test_list_pagination_limit(self, handler):
        for i in range(5):
            _add_sample_file(handler, f"file_{i}")
        h = _make_handler()
        result = await handler.handle(
            "/api/v2/storage/files", {"limit": "2"}, h
        )
        body = _body(result)
        assert len(body["files"]) == 2
        assert body["pagination"]["total"] == 5
        assert body["pagination"]["has_more"] is True

    @pytest.mark.asyncio
    async def test_list_pagination_offset(self, handler):
        for i in range(5):
            _add_sample_file(handler, f"file_{i}")
        h = _make_handler()
        result = await handler.handle(
            "/api/v2/storage/files", {"offset": "3", "limit": "10"}, h
        )
        body = _body(result)
        assert len(body["files"]) == 2

    @pytest.mark.asyncio
    async def test_list_sorted_by_created_at_desc(self, handler):
        fm1 = _add_sample_file(handler, "file_old")
        fm1.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        fm2 = _add_sample_file(handler, "file_new")
        fm2.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        h = _make_handler()
        result = await handler.handle("/api/v2/storage/files", {}, h)
        body = _body(result)
        assert body["files"][0]["id"] == "file_new"
        assert body["files"][1]["id"] == "file_old"

    @pytest.mark.asyncio
    async def test_list_non_get_returns_none(self, handler):
        h = _make_handler(method="POST")
        result = await handler.handle("/api/v2/storage/files", {}, h)
        assert result is None


# ============================================================================
# GET /api/v2/storage/files/:file_id - Get File
# ============================================================================


class TestGetFile:
    """Test getting file metadata by ID."""

    @pytest.mark.asyncio
    async def test_get_existing_file(self, handler):
        _add_sample_file(handler, "file_abc")
        h = _make_handler()
        result = await handler.handle("/api/v2/storage/files/file_abc", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == "file_abc"
        assert body["filename"] == "test.txt"

    @pytest.mark.asyncio
    async def test_get_nonexistent_file(self, handler):
        h = _make_handler()
        result = await handler.handle("/api/v2/storage/files/nonexistent", {}, h)
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_get_file_includes_metadata(self, handler):
        fm = _add_sample_file(handler, "file_meta")
        fm.metadata = {"custom": "data"}
        fm.tags = ["tag1", "tag2"]
        h = _make_handler()
        result = await handler.handle("/api/v2/storage/files/file_meta", {}, h)
        body = _body(result)
        assert body["metadata"] == {"custom": "data"}
        assert body["tags"] == ["tag1", "tag2"]


# ============================================================================
# GET /api/v2/storage/files/:file_id/download - Download File
# ============================================================================


class TestDownloadFile:
    """Test downloading files."""

    @pytest.mark.asyncio
    async def test_download_success(self, handler):
        _add_sample_file(handler, "file_dl")
        h = _make_handler()
        result = await handler.handle(
            "/api/v2/storage/files/file_dl/download", {}, h
        )
        assert _status(result) == 200
        assert result.content_type == "text/plain"
        assert result.body == b"file content"
        assert "Content-Disposition" in result.headers

    @pytest.mark.asyncio
    async def test_download_nonexistent(self, handler):
        h = _make_handler()
        result = await handler.handle(
            "/api/v2/storage/files/nonexistent/download", {}, h
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_download_unavailable_status(self, handler):
        _add_sample_file(handler, "file_pending", status=FileStatus.PENDING)
        h = _make_handler()
        result = await handler.handle(
            "/api/v2/storage/files/file_pending/download", {}, h
        )
        assert _status(result) == 400
        assert "not available" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_download_deleted_file(self, handler):
        _add_sample_file(handler, "file_del", status=FileStatus.DELETED)
        h = _make_handler()
        result = await handler.handle(
            "/api/v2/storage/files/file_del/download", {}, h
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_download_backend_not_found(self, handler):
        _add_sample_file(handler, "file_miss")
        handler._backend.download_file.side_effect = FileNotFoundError("gone")
        h = _make_handler()
        result = await handler.handle(
            "/api/v2/storage/files/file_miss/download", {}, h
        )
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_download_backend_error(self, handler):
        _add_sample_file(handler, "file_err")
        handler._backend.download_file.side_effect = OSError("disk failure")
        h = _make_handler()
        result = await handler.handle(
            "/api/v2/storage/files/file_err/download", {}, h
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_download_content_length_header(self, handler):
        _add_sample_file(handler, "file_cl")
        handler._backend.download_file.return_value = b"12345"
        h = _make_handler()
        result = await handler.handle(
            "/api/v2/storage/files/file_cl/download", {}, h
        )
        assert result.headers["Content-Length"] == "5"


# ============================================================================
# POST /api/v2/storage/files - Upload File
# ============================================================================


class TestUploadFile:
    """Test file upload."""

    @pytest.mark.asyncio
    async def test_upload_success(self, handler):
        body = {
            "filename": "test.txt",
            "content": _sample_b64_content(),
            "bucket": "default",
        }
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post("/api/v2/storage/files", {}, h)
        assert _status(result) == 201
        data = _body(result)
        assert "file" in data
        assert data["file"]["filename"] == "test.txt"
        assert data["file"]["status"] == "available"
        assert "message" in data

    @pytest.mark.asyncio
    async def test_upload_adds_to_file_store(self, handler):
        body = {
            "filename": "report.pdf",
            "content": _sample_b64_content(),
        }
        h = _make_handler(body=body, method="POST")
        await handler.handle_post("/api/v2/storage/files", {}, h)
        assert len(handler._files) == 1

    @pytest.mark.asyncio
    async def test_upload_missing_filename(self, handler):
        body = {"content": _sample_b64_content()}
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post("/api/v2/storage/files", {}, h)
        assert _status(result) == 400
        assert "required" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_upload_empty_filename(self, handler):
        body = {"filename": "", "content": _sample_b64_content()}
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post("/api/v2/storage/files", {}, h)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_upload_missing_content(self, handler):
        body = {"filename": "test.txt"}
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post("/api/v2/storage/files", {}, h)
        assert _status(result) == 400
        assert "content" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_upload_invalid_base64(self, handler):
        body = {"filename": "test.txt", "content": "not-valid-base64!!!"}
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post("/api/v2/storage/files", {}, h)
        assert _status(result) == 400
        assert "base64" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_upload_file_too_large(self, handler):
        # Create data larger than MAX_FILE_SIZE_BYTES
        with patch(
            "aragora.server.handlers.cloud_storage.MAX_FILE_SIZE_BYTES", 10
        ):
            body = {
                "filename": "big.txt",
                "content": _sample_b64_content(b"x" * 100),
            }
            h = _make_handler(body=body, method="POST")
            result = await handler.handle_post("/api/v2/storage/files", {}, h)
        assert _status(result) == 400
        assert "too large" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_upload_invalid_filename_chars(self, handler):
        body = {
            "filename": "file<script>.txt",
            "content": _sample_b64_content(),
        }
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post("/api/v2/storage/files", {}, h)
        assert _status(result) == 400
        assert "invalid characters" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_upload_disallowed_extension(self, handler):
        body = {
            "filename": "virus.exe",
            "content": _sample_b64_content(),
        }
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post("/api/v2/storage/files", {}, h)
        assert _status(result) == 400
        assert "not allowed" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_upload_with_tags_and_metadata(self, handler):
        body = {
            "filename": "tagged.json",
            "content": _sample_b64_content(b'{"key": "val"}'),
            "tags": ["important", "project-x"],
            "metadata": {"department": "engineering"},
        }
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post("/api/v2/storage/files", {}, h)
        assert _status(result) == 201
        data = _body(result)
        assert data["file"]["tags"] == ["important", "project-x"]
        assert data["file"]["metadata"] == {"department": "engineering"}

    @pytest.mark.asyncio
    async def test_upload_custom_bucket(self, handler):
        body = {
            "filename": "test.csv",
            "content": _sample_b64_content(),
            "bucket": "my-bucket",
        }
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post("/api/v2/storage/files", {}, h)
        assert _status(result) == 201
        data = _body(result)
        assert data["file"]["bucket"] == "my-bucket"

    @pytest.mark.asyncio
    async def test_upload_backend_error(self, handler):
        handler._backend.upload_file.side_effect = OSError("upload failed")
        body = {
            "filename": "test.txt",
            "content": _sample_b64_content(),
        }
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post("/api/v2/storage/files", {}, h)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_upload_computes_checksum(self, handler):
        content = b"unique content for checksum"
        body = {
            "filename": "check.txt",
            "content": _sample_b64_content(content),
        }
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post("/api/v2/storage/files", {}, h)
        assert _status(result) == 201
        data = _body(result)
        assert data["file"]["checksum"].startswith("sha256:")

    @pytest.mark.asyncio
    async def test_upload_guesses_content_type(self, handler):
        body = {
            "filename": "image.png",
            "content": _sample_b64_content(b"\x89PNG"),
        }
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post("/api/v2/storage/files", {}, h)
        assert _status(result) == 201
        data = _body(result)
        assert "png" in data["file"]["content_type"].lower()


# ============================================================================
# DELETE /api/v2/storage/files/:file_id - Delete File
# ============================================================================


class TestDeleteFile:
    """Test file deletion."""

    @pytest.mark.asyncio
    async def test_delete_success(self, handler):
        _add_sample_file(handler, "file_del")
        h = _make_handler(method="DELETE")
        result = await handler.handle_delete(
            "/api/v2/storage/files/file_del", {}, h
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["deleted"] is True
        assert body["file_id"] == "file_del"

    @pytest.mark.asyncio
    async def test_delete_marks_as_deleted(self, handler):
        _add_sample_file(handler, "file_del")
        h = _make_handler(method="DELETE")
        await handler.handle_delete(
            "/api/v2/storage/files/file_del", {}, h
        )
        assert handler._files["file_del"].status == FileStatus.DELETED

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, handler):
        h = _make_handler(method="DELETE")
        result = await handler.handle_delete(
            "/api/v2/storage/files/nonexistent", {}, h
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_delete_backend_error_still_marks_deleted(self, handler):
        _add_sample_file(handler, "file_berr")
        handler._backend.delete_file.side_effect = OSError("backend down")
        h = _make_handler(method="DELETE")
        result = await handler.handle_delete(
            "/api/v2/storage/files/file_berr", {}, h
        )
        assert _status(result) == 200
        assert handler._files["file_berr"].status == FileStatus.DELETED

    @pytest.mark.asyncio
    async def test_delete_invalid_path(self, handler):
        h = _make_handler(method="DELETE")
        result = await handler.handle_delete(
            "/api/v2/storage/files", {}, h
        )
        # Path doesn't match /api/v2/storage/files/* (no file_id), returns None
        assert result is None


# ============================================================================
# POST /api/v2/storage/files/:file_id/presign - Generate Presigned URL
# ============================================================================


class TestPresignedUrl:
    """Test presigned URL generation."""

    @pytest.mark.asyncio
    async def test_presign_success(self, handler):
        _add_sample_file(handler, "file_ps")
        body = {"expires_in_seconds": 600, "method": "GET"}
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post(
            "/api/v2/storage/files/file_ps/presign", {}, h
        )
        assert _status(result) == 200
        data = _body(result)
        assert data["file_id"] == "file_ps"
        assert "url" in data
        assert data["expires_in_seconds"] == 600
        assert data["method"] == "GET"

    @pytest.mark.asyncio
    async def test_presign_default_values(self, handler):
        _add_sample_file(handler, "file_ps2")
        h = _make_handler(body={}, method="POST")
        result = await handler.handle_post(
            "/api/v2/storage/files/file_ps2/presign", {}, h
        )
        assert _status(result) == 200
        data = _body(result)
        assert data["expires_in_seconds"] == 3600
        assert data["method"] == "GET"

    @pytest.mark.asyncio
    async def test_presign_put_method(self, handler):
        _add_sample_file(handler, "file_ps3")
        body = {"method": "PUT"}
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post(
            "/api/v2/storage/files/file_ps3/presign", {}, h
        )
        assert _status(result) == 200
        data = _body(result)
        assert data["method"] == "PUT"

    @pytest.mark.asyncio
    async def test_presign_nonexistent_file(self, handler):
        h = _make_handler(body={}, method="POST")
        result = await handler.handle_post(
            "/api/v2/storage/files/nonexistent/presign", {}, h
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_presign_expires_too_short(self, handler):
        _add_sample_file(handler, "file_short")
        body = {"expires_in_seconds": 30}
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post(
            "/api/v2/storage/files/file_short/presign", {}, h
        )
        assert _status(result) == 400
        assert "between" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_presign_expires_too_long(self, handler):
        _add_sample_file(handler, "file_long")
        body = {"expires_in_seconds": 100000}
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post(
            "/api/v2/storage/files/file_long/presign", {}, h
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_presign_invalid_method(self, handler):
        _add_sample_file(handler, "file_badm")
        body = {"method": "DELETE"}
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post(
            "/api/v2/storage/files/file_badm/presign", {}, h
        )
        assert _status(result) == 400
        assert "must be GET or PUT" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_presign_backend_error(self, handler):
        _add_sample_file(handler, "file_pserr")
        handler._backend.get_presigned_url.side_effect = OSError("fail")
        h = _make_handler(body={}, method="POST")
        result = await handler.handle_post(
            "/api/v2/storage/files/file_pserr/presign", {}, h
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_presign_exact_boundary_60(self, handler):
        _add_sample_file(handler, "file_b60")
        body = {"expires_in_seconds": 60}
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post(
            "/api/v2/storage/files/file_b60/presign", {}, h
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_presign_exact_boundary_86400(self, handler):
        _add_sample_file(handler, "file_b86400")
        body = {"expires_in_seconds": 86400}
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post(
            "/api/v2/storage/files/file_b86400/presign", {}, h
        )
        assert _status(result) == 200


# ============================================================================
# GET /api/v2/storage/quota - Storage Quota
# ============================================================================


class TestStorageQuota:
    """Test storage quota endpoint."""

    @pytest.mark.asyncio
    async def test_quota_empty(self, handler):
        h = _make_handler()
        result = await handler.handle("/api/v2/storage/quota", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert "quota" in body
        q = body["quota"]
        assert q["used_bytes"] == 0
        assert q["file_count"] == 0
        assert q["usage_percent"] == 0.0

    @pytest.mark.asyncio
    async def test_quota_with_files(self, handler):
        fm1 = _add_sample_file(handler, "file_q1")
        fm1.size_bytes = 500
        fm2 = _add_sample_file(handler, "file_q2")
        fm2.size_bytes = 300
        h = _make_handler()
        result = await handler.handle("/api/v2/storage/quota", {}, h)
        body = _body(result)
        q = body["quota"]
        assert q["used_bytes"] == 800
        assert q["file_count"] == 2

    @pytest.mark.asyncio
    async def test_quota_excludes_deleted(self, handler):
        fm = _add_sample_file(handler, "file_qdel", status=FileStatus.DELETED)
        fm.size_bytes = 1000
        h = _make_handler()
        result = await handler.handle("/api/v2/storage/quota", {}, h)
        body = _body(result)
        assert body["quota"]["used_bytes"] == 0
        assert body["quota"]["file_count"] == 0

    @pytest.mark.asyncio
    async def test_quota_has_generated_at(self, handler):
        h = _make_handler()
        result = await handler.handle("/api/v2/storage/quota", {}, h)
        body = _body(result)
        assert "generated_at" in body


# ============================================================================
# GET /api/v2/storage/buckets - List Buckets
# ============================================================================


class TestListBuckets:
    """Test listing buckets."""

    @pytest.mark.asyncio
    async def test_list_empty_returns_default(self, handler):
        h = _make_handler()
        result = await handler.handle("/api/v2/storage/buckets", {}, h)
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 1
        assert body["buckets"][0]["name"] == "default"

    @pytest.mark.asyncio
    async def test_list_with_custom_buckets(self, handler):
        _add_sample_bucket(handler, "b1", "my-bucket-1")
        _add_sample_bucket(handler, "b2", "my-bucket-2")
        h = _make_handler()
        result = await handler.handle("/api/v2/storage/buckets", {}, h)
        body = _body(result)
        assert body["count"] == 2


# ============================================================================
# GET /api/v2/storage/buckets/:bucket_id - Get Bucket
# ============================================================================


class TestGetBucket:
    """Test getting individual bucket info."""

    @pytest.mark.asyncio
    async def test_get_default_bucket(self, handler):
        h = _make_handler()
        result = await handler.handle(
            "/api/v2/storage/buckets/default", {}, h
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["name"] == "default"
        assert body["provider"] == "local"

    @pytest.mark.asyncio
    async def test_get_custom_bucket(self, handler):
        _add_sample_bucket(handler, "b123", "my-bucket")
        h = _make_handler()
        result = await handler.handle(
            "/api/v2/storage/buckets/b123", {}, h
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["name"] == "my-bucket"

    @pytest.mark.asyncio
    async def test_get_nonexistent_bucket(self, handler):
        h = _make_handler()
        result = await handler.handle(
            "/api/v2/storage/buckets/nonexistent", {}, h
        )
        assert _status(result) == 404


# ============================================================================
# POST /api/v2/storage/buckets - Create Bucket
# ============================================================================


class TestCreateBucket:
    """Test bucket creation."""

    @pytest.mark.asyncio
    async def test_create_success(self, handler):
        body = {"name": "my-new-bucket"}
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post("/api/v2/storage/buckets", {}, h)
        assert _status(result) == 201
        data = _body(result)
        assert data["bucket"]["name"] == "my-new-bucket"
        assert "message" in data

    @pytest.mark.asyncio
    async def test_create_adds_to_store(self, handler):
        body = {"name": "store-bucket"}
        h = _make_handler(body=body, method="POST")
        await handler.handle_post("/api/v2/storage/buckets", {}, h)
        assert len(handler._buckets) == 1

    @pytest.mark.asyncio
    async def test_create_with_provider(self, handler):
        body = {"name": "s3-bucket", "provider": "s3", "region": "us-east-1"}
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post("/api/v2/storage/buckets", {}, h)
        assert _status(result) == 201
        data = _body(result)
        assert data["bucket"]["provider"] == "s3"
        assert data["bucket"]["region"] == "us-east-1"

    @pytest.mark.asyncio
    async def test_create_with_options(self, handler):
        body = {
            "name": "public-bucket",
            "is_public": True,
            "versioning_enabled": True,
        }
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post("/api/v2/storage/buckets", {}, h)
        assert _status(result) == 201
        data = _body(result)
        assert data["bucket"]["is_public"] is True
        assert data["bucket"]["versioning_enabled"] is True

    @pytest.mark.asyncio
    async def test_create_missing_name(self, handler):
        body = {}
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post("/api/v2/storage/buckets", {}, h)
        assert _status(result) == 400
        assert "required" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_create_empty_name(self, handler):
        body = {"name": ""}
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post("/api/v2/storage/buckets", {}, h)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_whitespace_name(self, handler):
        body = {"name": "   "}
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post("/api/v2/storage/buckets", {}, h)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_invalid_name_uppercase(self, handler):
        body = {"name": "MyBucket"}
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post("/api/v2/storage/buckets", {}, h)
        assert _status(result) == 400
        assert "lowercase" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_create_invalid_name_too_short(self, handler):
        body = {"name": "ab"}
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post("/api/v2/storage/buckets", {}, h)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_invalid_name_special_chars(self, handler):
        body = {"name": "bucket_name!"}
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post("/api/v2/storage/buckets", {}, h)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_duplicate_name(self, handler):
        _add_sample_bucket(handler, "b1", "existing-bucket")
        body = {"name": "existing-bucket"}
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post("/api/v2/storage/buckets", {}, h)
        assert _status(result) == 409
        assert "already exists" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_create_valid_bucket_names(self, handler):
        valid_names = ["my-bucket-123", "data-store", "abc", "a1b"]
        for name in valid_names:
            h2 = CloudStorageHandler(server_context={})
            h2._backend = AsyncMock()
            body = {"name": name}
            hh = _make_handler(body=body, method="POST")
            result = await h2.handle_post("/api/v2/storage/buckets", {}, hh)
            assert _status(result) == 201, f"Name '{name}' should be valid"


# ============================================================================
# DELETE /api/v2/storage/buckets/:bucket_id - Delete Bucket
# ============================================================================


class TestDeleteBucket:
    """Test bucket deletion."""

    @pytest.mark.asyncio
    async def test_delete_success(self, handler):
        _add_sample_bucket(handler, "b_del", "deletable")
        h = _make_handler(method="DELETE")
        result = await handler.handle_delete(
            "/api/v2/storage/buckets/b_del", {}, h
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["deleted"] is True
        assert "b_del" not in handler._buckets

    @pytest.mark.asyncio
    async def test_delete_default_bucket_forbidden(self, handler):
        h = _make_handler(method="DELETE")
        result = await handler.handle_delete(
            "/api/v2/storage/buckets/default", {}, h
        )
        assert _status(result) == 400
        assert "default" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_bucket(self, handler):
        h = _make_handler(method="DELETE")
        result = await handler.handle_delete(
            "/api/v2/storage/buckets/nonexistent", {}, h
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_delete_bucket_with_files(self, handler):
        bi = _add_sample_bucket(handler, "b_full", "full-bucket")
        # Add a file in this bucket
        fm = _add_sample_file(handler, "file_in_bucket", bucket="full-bucket")
        h = _make_handler(method="DELETE")
        result = await handler.handle_delete(
            "/api/v2/storage/buckets/b_full", {}, h
        )
        assert _status(result) == 400
        assert "files" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_delete_bucket_with_only_deleted_files(self, handler):
        _add_sample_bucket(handler, "b_delf", "del-bucket")
        _add_sample_file(handler, "file_df", bucket="del-bucket", status=FileStatus.DELETED)
        h = _make_handler(method="DELETE")
        result = await handler.handle_delete(
            "/api/v2/storage/buckets/b_delf", {}, h
        )
        # Deleted files don't count
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_delete_bucket_invalid_path(self, handler):
        h = _make_handler(method="DELETE")
        result = await handler.handle_delete(
            "/api/v2/storage/buckets", {}, h
        )
        assert result is None


# ============================================================================
# Circuit Breaker Integration
# ============================================================================


class TestCircuitBreakerIntegration:
    """Test circuit breaker behavior in handlers."""

    @pytest.mark.asyncio
    async def test_handle_circuit_open(self, handler):
        cb = handler._get_circuit_breaker()
        # Force circuit open
        for _ in range(cb.failure_threshold + 1):
            cb.record_failure()
        h = _make_handler()
        result = await handler.handle("/api/v2/storage/files", {}, h)
        assert _status(result) == 503
        assert "temporarily unavailable" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_handle_post_circuit_open(self, handler):
        cb = handler._get_circuit_breaker()
        for _ in range(cb.failure_threshold + 1):
            cb.record_failure()
        body = {"filename": "test.txt", "content": _sample_b64_content()}
        h = _make_handler(body=body, method="POST")
        result = await handler.handle_post("/api/v2/storage/files", {}, h)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_handle_delete_circuit_open(self, handler):
        _add_sample_file(handler, "file_cb")
        cb = handler._get_circuit_breaker()
        for _ in range(cb.failure_threshold + 1):
            cb.record_failure()
        h = _make_handler(method="DELETE")
        result = await handler.handle_delete(
            "/api/v2/storage/files/file_cb", {}, h
        )
        assert _status(result) == 503

    def test_circuit_breaker_caches(self, handler):
        cb1 = handler._get_circuit_breaker()
        cb2 = handler._get_circuit_breaker()
        assert cb1 is cb2


# ============================================================================
# Error Handling in handle() method
# ============================================================================


class TestHandleErrorPaths:
    """Test error paths in the main handle dispatch."""

    @pytest.mark.asyncio
    async def test_handle_non_get_returns_none(self, handler):
        h = _make_handler(method="POST")
        result = await handler.handle("/api/v2/storage/files", {}, h)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_none_query_params(self, handler):
        h = _make_handler()
        result = await handler.handle("/api/v2/storage/files", None, h)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_handle_unknown_path(self, handler):
        h = _make_handler()
        result = await handler.handle("/api/v2/storage/unknown", {}, h)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_file_path_too_short(self, handler):
        h = _make_handler()
        # Path with less than 6 parts after split
        result = await handler.handle("/api/v2/storage/files/", {}, h)
        # Parts: ["", "api", "v2", "storage", "files", ""] -- 6 parts, so file_id=""
        # This should get file with id "" which won't exist
        if result is not None:
            assert _status(result) in (400, 404)

    @pytest.mark.asyncio
    async def test_handle_bucket_path_too_short(self, handler):
        h = _make_handler()
        # Should not crash on bucket path
        result = await handler.handle("/api/v2/storage/buckets/", {}, h)
        if result is not None:
            assert _status(result) in (400, 404)

    @pytest.mark.asyncio
    async def test_handle_post_unknown_path_returns_none(self, handler):
        h = _make_handler(body={}, method="POST")
        result = await handler.handle_post("/api/v2/storage/unknown", {}, h)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_delete_non_matching(self, handler):
        h = _make_handler(method="DELETE")
        result = await handler.handle_delete("/api/v2/storage/unknown", {}, h)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_post_file_path_no_presign(self, handler):
        """POST to /api/v2/storage/files/:id without /presign should return None."""
        _add_sample_file(handler, "file_nop")
        h = _make_handler(body={}, method="POST")
        result = await handler.handle_post(
            "/api/v2/storage/files/file_nop", {}, h
        )
        assert result is None


# ============================================================================
# Dataclass Serialization
# ============================================================================


class TestDataclasses:
    """Test dataclass serialization methods."""

    def test_file_metadata_to_dict(self):
        now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        fm = FileMetadata(
            id="file_test",
            filename="test.txt",
            original_filename="test.txt",
            content_type="text/plain",
            size_bytes=1024,
            checksum="sha256:abc",
            bucket="default",
            path="2026/01/15/file_test/test.txt",
            status=FileStatus.AVAILABLE,
            created_at=now,
            updated_at=now,
            owner_id="user-1",
            tenant_id="tenant-1",
            metadata={"key": "val"},
            tags=["tag1"],
            expires_at=now,
        )
        d = fm.to_dict()
        assert d["id"] == "file_test"
        assert d["status"] == "available"
        assert d["created_at"] == now.isoformat()
        assert d["tenant_id"] == "tenant-1"
        assert d["metadata"] == {"key": "val"}
        assert d["tags"] == ["tag1"]
        assert d["expires_at"] == now.isoformat()

    def test_file_metadata_no_tenant_no_expires(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        fm = FileMetadata(
            id="f1", filename="a.txt", original_filename="a.txt",
            content_type="text/plain", size_bytes=0, checksum="sha256:x",
            bucket="b", path="p", status=FileStatus.PENDING,
            created_at=now, updated_at=now, owner_id="u",
        )
        d = fm.to_dict()
        assert d["tenant_id"] is None
        assert d["expires_at"] is None
        assert d["status"] == "pending"

    def test_bucket_info_to_dict(self):
        now = datetime(2026, 2, 1, tzinfo=timezone.utc)
        bi = BucketInfo(
            id="b1",
            name="my-bucket",
            provider=StorageProvider.S3,
            region="us-west-2",
            created_at=now,
            owner_id="user-1",
            is_public=True,
            versioning_enabled=True,
            lifecycle_rules=[{"days": 30, "action": "delete"}],
        )
        d = bi.to_dict()
        assert d["id"] == "b1"
        assert d["name"] == "my-bucket"
        assert d["provider"] == "s3"
        assert d["region"] == "us-west-2"
        assert d["is_public"] is True
        assert d["versioning_enabled"] is True
        assert len(d["lifecycle_rules"]) == 1

    def test_storage_quota_to_dict(self):
        sq = StorageQuota(
            total_bytes=1024 * 1024 * 1024,
            used_bytes=512 * 1024 * 1024,
            file_count=42,
            max_file_size_bytes=100 * 1024 * 1024,
            tenant_id="t1",
        )
        d = sq.to_dict()
        assert d["total_bytes"] == 1024 * 1024 * 1024
        assert d["used_bytes"] == 512 * 1024 * 1024
        assert d["available_bytes"] == 512 * 1024 * 1024
        assert d["file_count"] == 42
        assert d["usage_percent"] == 50.0
        assert d["tenant_id"] == "t1"

    def test_storage_quota_zero_total(self):
        sq = StorageQuota(
            total_bytes=0, used_bytes=0, file_count=0,
            max_file_size_bytes=100,
        )
        assert sq.usage_percent == 0.0
        assert sq.available_bytes == 0

    def test_storage_quota_full(self):
        sq = StorageQuota(
            total_bytes=100, used_bytes=100, file_count=5,
            max_file_size_bytes=50,
        )
        assert sq.usage_percent == 100.0
        assert sq.available_bytes == 0

    def test_storage_quota_over_limit(self):
        sq = StorageQuota(
            total_bytes=100, used_bytes=150, file_count=10,
            max_file_size_bytes=50,
        )
        assert sq.available_bytes == 0  # max(0, negative)


# ============================================================================
# Enums
# ============================================================================


class TestEnums:
    """Test enum values."""

    def test_storage_providers(self):
        assert StorageProvider.S3.value == "s3"
        assert StorageProvider.GCS.value == "gcs"
        assert StorageProvider.AZURE.value == "azure"
        assert StorageProvider.LOCAL.value == "local"

    def test_file_statuses(self):
        assert FileStatus.PENDING.value == "pending"
        assert FileStatus.UPLOADING.value == "uploading"
        assert FileStatus.AVAILABLE.value == "available"
        assert FileStatus.DELETED.value == "deleted"
        assert FileStatus.FAILED.value == "failed"


# ============================================================================
# Constants
# ============================================================================


class TestConstants:
    """Test module-level constants."""

    def test_max_file_size_positive(self):
        assert MAX_FILE_SIZE_BYTES > 0

    def test_allowed_extensions_is_frozenset(self):
        assert isinstance(ALLOWED_EXTENSIONS, frozenset)

    def test_common_extensions_present(self):
        for ext in [".txt", ".pdf", ".json", ".csv", ".png"]:
            assert ext in ALLOWED_EXTENSIONS

    def test_safe_filename_pattern(self):
        assert SAFE_FILENAME_PATTERN.match("hello.txt")
        assert SAFE_FILENAME_PATTERN.match("my file.pdf")
        assert not SAFE_FILENAME_PATTERN.match("file<bad>.txt")
        assert not SAFE_FILENAME_PATTERN.match("file/path.txt")


# ============================================================================
# LocalStorageBackend
# ============================================================================


class TestLocalStorageBackend:
    """Test the local filesystem storage backend."""

    @pytest.mark.asyncio
    async def test_upload_and_download(self, tmp_path):
        backend = LocalStorageBackend(base_path=tmp_path)
        bucket = "test-bucket"
        path = "files/test.txt"
        data = b"hello world"

        await backend.upload_file(bucket, path, data, "text/plain")
        result = await backend.download_file(bucket, path)
        assert result == data

    @pytest.mark.asyncio
    async def test_file_exists(self, tmp_path):
        backend = LocalStorageBackend(base_path=tmp_path)
        assert not await backend.file_exists("bucket", "missing.txt")

        await backend.upload_file("bucket", "exists.txt", b"data", "text/plain")
        assert await backend.file_exists("bucket", "exists.txt")

    @pytest.mark.asyncio
    async def test_delete_file(self, tmp_path):
        backend = LocalStorageBackend(base_path=tmp_path)
        await backend.upload_file("bucket", "del.txt", b"data", "text/plain")
        assert await backend.delete_file("bucket", "del.txt") is True
        assert await backend.file_exists("bucket", "del.txt") is False

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, tmp_path):
        backend = LocalStorageBackend(base_path=tmp_path)
        assert await backend.delete_file("bucket", "nope.txt") is False

    @pytest.mark.asyncio
    async def test_download_nonexistent(self, tmp_path):
        backend = LocalStorageBackend(base_path=tmp_path)
        with pytest.raises(FileNotFoundError):
            await backend.download_file("bucket", "nope.txt")

    @pytest.mark.asyncio
    async def test_presigned_url(self, tmp_path):
        backend = LocalStorageBackend(base_path=tmp_path)
        await backend.upload_file("bucket", "f.txt", b"d", "text/plain")
        url = await backend.get_presigned_url("bucket", "f.txt")
        assert url.startswith("file://")

    @pytest.mark.asyncio
    async def test_list_files_empty_bucket(self, tmp_path):
        backend = LocalStorageBackend(base_path=tmp_path)
        files, token = await backend.list_files("empty-bucket")
        assert files == []
        assert token is None

    @pytest.mark.asyncio
    async def test_list_files_with_content(self, tmp_path):
        backend = LocalStorageBackend(base_path=tmp_path)
        await backend.upload_file("bucket", "a.txt", b"aaa", "text/plain")
        await backend.upload_file("bucket", "b.txt", b"bbb", "text/plain")
        files, token = await backend.list_files("bucket")
        assert len(files) == 2
        paths = [f["path"] for f in files]
        assert "a.txt" in paths
        assert "b.txt" in paths

    @pytest.mark.asyncio
    async def test_list_files_with_prefix(self, tmp_path):
        backend = LocalStorageBackend(base_path=tmp_path)
        await backend.upload_file("bucket", "images/a.png", b"d", "image/png")
        await backend.upload_file("bucket", "docs/b.txt", b"d", "text/plain")
        files, _ = await backend.list_files("bucket", prefix="images/")
        assert len(files) == 1

    @pytest.mark.asyncio
    async def test_list_files_with_limit(self, tmp_path):
        backend = LocalStorageBackend(base_path=tmp_path)
        for i in range(5):
            await backend.upload_file("bucket", f"f{i}.txt", b"d", "text/plain")
        files, _ = await backend.list_files("bucket", limit=3)
        assert len(files) == 3

    @pytest.mark.asyncio
    async def test_path_traversal_protection(self, tmp_path):
        backend = LocalStorageBackend(base_path=tmp_path)
        with pytest.raises(ValueError, match="traversal"):
            backend._safe_path("bucket", "../../etc/passwd")

    @pytest.mark.asyncio
    async def test_upload_returns_file_url(self, tmp_path):
        backend = LocalStorageBackend(base_path=tmp_path)
        url = await backend.upload_file("b", "f.txt", b"data", "text/plain")
        assert url.startswith("file://")
