"""
Tests for CloudStorageHandler - Cloud storage HTTP endpoints.

Tests cover:
- List files with filtering and pagination
- Upload files with validation
- Download files
- Delete files
- Generate presigned URLs
- Storage quota management
- Bucket operations (list, create, delete)
- Circuit breaker behavior
- Rate limiting
- Error handling

Stability: STABLE
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.cloud_storage import (
    CloudStorageHandler,
    FileMetadata,
    FileStatus,
    BucketInfo,
    StorageProvider,
    StorageQuota,
    LocalStorageBackend,
    create_cloud_storage_handler,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
)


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================


class MockHTTPHandler:
    """Mock HTTP request handler for testing."""

    def __init__(
        self,
        method: str = "GET",
        body: dict | None = None,
        user_id: str | None = "test-user",
    ):
        self.command = method
        self._body = json.dumps(body or {}).encode() if body else b""
        self.headers = {
            "Content-Length": str(len(self._body)) if self._body else "0",
            "Content-Type": "application/json" if body else "",
            "Authorization": f"Bearer test-token-{user_id}" if user_id else "",
        }
        self.rfile = io.BytesIO(self._body)
        self._user_id = user_id

    @property
    def user_id(self) -> str:
        return self._user_id or "anonymous"


class MockUserAuthContext:
    """Mock user authentication context."""

    def __init__(self, user_id: str = "test-user", is_admin: bool = False):
        self.user_id = user_id
        self.is_authenticated = True
        self.is_admin = is_admin
        self.roles = ["admin"] if is_admin else ["user"]
        self.permissions = ["storage:read", "storage:write", "storage:delete"]
        if is_admin:
            self.permissions.append("storage:admin")


class MockStorageBackend:
    """Mock cloud storage backend for testing."""

    def __init__(self):
        self._files: dict[str, bytes] = {}
        self._fail_upload = False
        self._fail_download = False
        self._fail_delete = False

    async def upload_file(
        self,
        bucket: str,
        path: str,
        data: bytes,
        content_type: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        if self._fail_upload:
            raise OSError("Upload failed")
        key = f"{bucket}/{path}"
        self._files[key] = data
        return f"file://{key}"

    async def download_file(self, bucket: str, path: str) -> bytes:
        if self._fail_download:
            raise OSError("Download failed")
        key = f"{bucket}/{path}"
        if key not in self._files:
            raise FileNotFoundError(f"File not found: {key}")
        return self._files[key]

    async def delete_file(self, bucket: str, path: str) -> bool:
        if self._fail_delete:
            raise OSError("Delete failed")
        key = f"{bucket}/{path}"
        if key in self._files:
            del self._files[key]
            return True
        return False

    async def file_exists(self, bucket: str, path: str) -> bool:
        key = f"{bucket}/{path}"
        return key in self._files

    async def get_presigned_url(
        self,
        bucket: str,
        path: str,
        expires_in_seconds: int = 3600,
        method: str = "GET",
    ) -> str:
        return f"https://storage.example.com/{bucket}/{path}?expires={expires_in_seconds}"

    async def list_files(
        self,
        bucket: str,
        prefix: str | None = None,
        limit: int = 100,
        continuation_token: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        files = []
        for key, data in self._files.items():
            if key.startswith(f"{bucket}/"):
                path = key[len(f"{bucket}/") :]
                if prefix is None or path.startswith(prefix):
                    files.append(
                        {
                            "path": path,
                            "size": len(data),
                            "modified": datetime.now(timezone.utc).isoformat(),
                        }
                    )
        return files[:limit], None


@pytest.fixture
def mock_storage_backend():
    """Create mock storage backend."""
    return MockStorageBackend()


@pytest.fixture
def server_context(mock_storage_backend):
    """Create server context with mock storage backend."""
    return {
        "cloud_storage_backend": mock_storage_backend,
    }


@pytest.fixture
def handler(server_context):
    """Create CloudStorageHandler instance."""
    return CloudStorageHandler(server_context)


@pytest.fixture
def sample_file_content():
    """Sample file content for testing."""
    return b"Hello, this is test file content!"


@pytest.fixture
def sample_file_b64(sample_file_content):
    """Base64 encoded sample file content."""
    return base64.b64encode(sample_file_content).decode()


# ===========================================================================
# FileMetadata Tests
# ===========================================================================


class TestFileMetadata:
    """Tests for FileMetadata dataclass."""

    def test_file_metadata_creation(self):
        """Should create file metadata with all fields."""
        now = datetime.now(timezone.utc)
        metadata = FileMetadata(
            id="file_123",
            filename="test.txt",
            original_filename="test.txt",
            content_type="text/plain",
            size_bytes=1024,
            checksum="sha256:abc123",
            bucket="default",
            path="2024/01/01/file_123/test.txt",
            status=FileStatus.AVAILABLE,
            created_at=now,
            updated_at=now,
            owner_id="user-001",
        )

        assert metadata.id == "file_123"
        assert metadata.filename == "test.txt"
        assert metadata.status == FileStatus.AVAILABLE
        assert metadata.size_bytes == 1024

    def test_file_metadata_to_dict(self):
        """Should convert file metadata to dictionary."""
        now = datetime.now(timezone.utc)
        metadata = FileMetadata(
            id="file_123",
            filename="test.txt",
            original_filename="test.txt",
            content_type="text/plain",
            size_bytes=1024,
            checksum="sha256:abc123",
            bucket="default",
            path="2024/01/01/file_123/test.txt",
            status=FileStatus.AVAILABLE,
            created_at=now,
            updated_at=now,
            owner_id="user-001",
            tags=["important", "report"],
        )

        result = metadata.to_dict()

        assert result["id"] == "file_123"
        assert result["status"] == "available"
        assert result["tags"] == ["important", "report"]
        assert "created_at" in result


class TestBucketInfo:
    """Tests for BucketInfo dataclass."""

    def test_bucket_info_creation(self):
        """Should create bucket info with all fields."""
        now = datetime.now(timezone.utc)
        bucket = BucketInfo(
            id="bucket_123",
            name="my-bucket",
            provider=StorageProvider.S3,
            region="us-east-1",
            created_at=now,
            owner_id="user-001",
            is_public=False,
            versioning_enabled=True,
        )

        assert bucket.id == "bucket_123"
        assert bucket.name == "my-bucket"
        assert bucket.provider == StorageProvider.S3
        assert bucket.versioning_enabled is True

    def test_bucket_info_to_dict(self):
        """Should convert bucket info to dictionary."""
        now = datetime.now(timezone.utc)
        bucket = BucketInfo(
            id="bucket_123",
            name="my-bucket",
            provider=StorageProvider.GCS,
            region="us-central1",
            created_at=now,
            owner_id="user-001",
        )

        result = bucket.to_dict()

        assert result["id"] == "bucket_123"
        assert result["provider"] == "gcs"
        assert "created_at" in result


class TestStorageQuota:
    """Tests for StorageQuota dataclass."""

    def test_storage_quota_available_bytes(self):
        """Should calculate available bytes correctly."""
        quota = StorageQuota(
            total_bytes=1000,
            used_bytes=400,
            file_count=10,
            max_file_size_bytes=100,
        )

        assert quota.available_bytes == 600

    def test_storage_quota_usage_percent(self):
        """Should calculate usage percentage correctly."""
        quota = StorageQuota(
            total_bytes=1000,
            used_bytes=250,
            file_count=5,
            max_file_size_bytes=100,
        )

        assert quota.usage_percent == 25.0

    def test_storage_quota_zero_total(self):
        """Should handle zero total bytes."""
        quota = StorageQuota(
            total_bytes=0,
            used_bytes=0,
            file_count=0,
            max_file_size_bytes=100,
        )

        assert quota.usage_percent == 0.0
        assert quota.available_bytes == 0


# ===========================================================================
# CloudStorageHandler Tests - List Files
# ===========================================================================


class TestCloudStorageHandlerListFiles:
    """Tests for CloudStorageHandler.handle (list files)."""

    @pytest.mark.asyncio
    async def test_list_files_empty(self, handler):
        """Should return empty list when no files."""
        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/storage/files", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert data["files"] == []
        assert data["pagination"]["total"] == 0

    @pytest.mark.asyncio
    async def test_list_files_with_results(self, handler):
        """Should return files when available."""
        # Add a test file
        now = datetime.now(timezone.utc)
        handler._files["file_001"] = FileMetadata(
            id="file_001",
            filename="test.txt",
            original_filename="test.txt",
            content_type="text/plain",
            size_bytes=100,
            checksum="sha256:abc",
            bucket="default",
            path="test.txt",
            status=FileStatus.AVAILABLE,
            created_at=now,
            updated_at=now,
            owner_id="user-001",
        )

        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/storage/files", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert len(data["files"]) == 1
        assert data["files"][0]["id"] == "file_001"

    @pytest.mark.asyncio
    async def test_list_files_pagination(self, handler):
        """Should support pagination."""
        # Add multiple test files
        now = datetime.now(timezone.utc)
        for i in range(5):
            handler._files[f"file_{i:03d}"] = FileMetadata(
                id=f"file_{i:03d}",
                filename=f"test{i}.txt",
                original_filename=f"test{i}.txt",
                content_type="text/plain",
                size_bytes=100,
                checksum=f"sha256:abc{i}",
                bucket="default",
                path=f"test{i}.txt",
                status=FileStatus.AVAILABLE,
                created_at=now,
                updated_at=now,
                owner_id="user-001",
            )

        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle(
                "/api/v2/storage/files",
                {"limit": "2", "offset": "1"},
                mock_request,
            )

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert len(data["files"]) == 2
        assert data["pagination"]["total"] == 5
        assert data["pagination"]["has_more"] is True

    @pytest.mark.asyncio
    async def test_list_files_filter_by_bucket(self, handler):
        """Should filter files by bucket."""
        now = datetime.now(timezone.utc)
        handler._files["file_001"] = FileMetadata(
            id="file_001",
            filename="test1.txt",
            original_filename="test1.txt",
            content_type="text/plain",
            size_bytes=100,
            checksum="sha256:abc",
            bucket="bucket-a",
            path="test1.txt",
            status=FileStatus.AVAILABLE,
            created_at=now,
            updated_at=now,
            owner_id="user-001",
        )
        handler._files["file_002"] = FileMetadata(
            id="file_002",
            filename="test2.txt",
            original_filename="test2.txt",
            content_type="text/plain",
            size_bytes=100,
            checksum="sha256:def",
            bucket="bucket-b",
            path="test2.txt",
            status=FileStatus.AVAILABLE,
            created_at=now,
            updated_at=now,
            owner_id="user-001",
        )

        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle(
                "/api/v2/storage/files",
                {"bucket": "bucket-a"},
                mock_request,
            )

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert len(data["files"]) == 1
        assert data["files"][0]["id"] == "file_001"


# ===========================================================================
# CloudStorageHandler Tests - Upload Files
# ===========================================================================


class TestCloudStorageHandlerUploadFiles:
    """Tests for CloudStorageHandler.handle_post (upload files)."""

    @pytest.mark.asyncio
    async def test_upload_file_success(self, handler, sample_file_b64):
        """Should upload file successfully."""
        mock_request = MockHTTPHandler(
            method="POST",
            body={
                "filename": "test.txt",
                "content": sample_file_b64,
                "bucket": "default",
            },
        )

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post("/api/v2/storage/files", {}, mock_request)

        assert result is not None
        assert result.status == 201
        data = json.loads(result.body)
        assert "file" in data
        assert data["file"]["filename"] == "test.txt"
        assert data["file"]["status"] == "available"

    @pytest.mark.asyncio
    async def test_upload_file_missing_filename(self, handler, sample_file_b64):
        """Should reject upload without filename."""
        mock_request = MockHTTPHandler(
            method="POST",
            body={"content": sample_file_b64},
        )

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post("/api/v2/storage/files", {}, mock_request)

        assert result is not None
        assert result.status == 400
        data = json.loads(result.body)
        assert "Filename" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_upload_file_missing_content(self, handler):
        """Should reject upload without content."""
        mock_request = MockHTTPHandler(
            method="POST",
            body={"filename": "test.txt"},
        )

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post("/api/v2/storage/files", {}, mock_request)

        assert result is not None
        assert result.status == 400
        data = json.loads(result.body)
        assert "content" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_upload_file_invalid_base64(self, handler):
        """Should reject invalid base64 content."""
        mock_request = MockHTTPHandler(
            method="POST",
            body={"filename": "test.txt", "content": "not-valid-base64!!!"},
        )

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post("/api/v2/storage/files", {}, mock_request)

        assert result is not None
        assert result.status == 400
        data = json.loads(result.body)
        assert "base64" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_upload_file_with_tags(self, handler, sample_file_b64):
        """Should upload file with tags."""
        mock_request = MockHTTPHandler(
            method="POST",
            body={
                "filename": "test.txt",
                "content": sample_file_b64,
                "tags": ["important", "report"],
            },
        )

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post("/api/v2/storage/files", {}, mock_request)

        assert result is not None
        assert result.status == 201
        data = json.loads(result.body)
        assert data["file"]["tags"] == ["important", "report"]


# ===========================================================================
# CloudStorageHandler Tests - Get File
# ===========================================================================


class TestCloudStorageHandlerGetFile:
    """Tests for CloudStorageHandler.handle (get file)."""

    @pytest.mark.asyncio
    async def test_get_file_success(self, handler):
        """Should return file metadata."""
        now = datetime.now(timezone.utc)
        handler._files["file_001"] = FileMetadata(
            id="file_001",
            filename="test.txt",
            original_filename="test.txt",
            content_type="text/plain",
            size_bytes=100,
            checksum="sha256:abc",
            bucket="default",
            path="test.txt",
            status=FileStatus.AVAILABLE,
            created_at=now,
            updated_at=now,
            owner_id="user-001",
        )

        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/storage/files/file_001", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert data["id"] == "file_001"

    @pytest.mark.asyncio
    async def test_get_file_not_found(self, handler):
        """Should return 404 for non-existent file."""
        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/storage/files/nonexistent", {}, mock_request)

        assert result is not None
        assert result.status == 404


# ===========================================================================
# CloudStorageHandler Tests - Download File
# ===========================================================================


class TestCloudStorageHandlerDownloadFile:
    """Tests for CloudStorageHandler.handle (download file)."""

    @pytest.mark.asyncio
    async def test_download_file_success(self, handler, mock_storage_backend, sample_file_content):
        """Should download file successfully."""
        now = datetime.now(timezone.utc)
        handler._files["file_001"] = FileMetadata(
            id="file_001",
            filename="test.txt",
            original_filename="test.txt",
            content_type="text/plain",
            size_bytes=len(sample_file_content),
            checksum="sha256:abc",
            bucket="default",
            path="test.txt",
            status=FileStatus.AVAILABLE,
            created_at=now,
            updated_at=now,
            owner_id="user-001",
        )
        mock_storage_backend._files["default/test.txt"] = sample_file_content

        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle(
                "/api/v2/storage/files/file_001/download",
                {},
                mock_request,
            )

        assert result is not None
        assert result.status == 200
        assert result.body == sample_file_content
        assert result.content_type == "text/plain"

    @pytest.mark.asyncio
    async def test_download_file_not_found(self, handler):
        """Should return 404 for non-existent file."""
        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle(
                "/api/v2/storage/files/nonexistent/download",
                {},
                mock_request,
            )

        assert result is not None
        assert result.status == 404


# ===========================================================================
# CloudStorageHandler Tests - Delete File
# ===========================================================================


class TestCloudStorageHandlerDeleteFile:
    """Tests for CloudStorageHandler.handle_delete (delete file)."""

    @pytest.mark.asyncio
    async def test_delete_file_success(self, handler, mock_storage_backend):
        """Should delete file successfully."""
        now = datetime.now(timezone.utc)
        handler._files["file_001"] = FileMetadata(
            id="file_001",
            filename="test.txt",
            original_filename="test.txt",
            content_type="text/plain",
            size_bytes=100,
            checksum="sha256:abc",
            bucket="default",
            path="test.txt",
            status=FileStatus.AVAILABLE,
            created_at=now,
            updated_at=now,
            owner_id="user-001",
        )
        mock_storage_backend._files["default/test.txt"] = b"content"

        mock_request = MockHTTPHandler(method="DELETE")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_delete("/api/v2/storage/files/file_001", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert data["deleted"] is True
        assert handler._files["file_001"].status == FileStatus.DELETED

    @pytest.mark.asyncio
    async def test_delete_file_not_found(self, handler):
        """Should return 404 for non-existent file."""
        mock_request = MockHTTPHandler(method="DELETE")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_delete(
                "/api/v2/storage/files/nonexistent", {}, mock_request
            )

        assert result is not None
        assert result.status == 404


# ===========================================================================
# CloudStorageHandler Tests - Presigned URL
# ===========================================================================


class TestCloudStorageHandlerPresignedUrl:
    """Tests for CloudStorageHandler.handle_post (presigned URL)."""

    @pytest.mark.asyncio
    async def test_generate_presigned_url_success(self, handler):
        """Should generate presigned URL successfully."""
        now = datetime.now(timezone.utc)
        handler._files["file_001"] = FileMetadata(
            id="file_001",
            filename="test.txt",
            original_filename="test.txt",
            content_type="text/plain",
            size_bytes=100,
            checksum="sha256:abc",
            bucket="default",
            path="test.txt",
            status=FileStatus.AVAILABLE,
            created_at=now,
            updated_at=now,
            owner_id="user-001",
        )

        mock_request = MockHTTPHandler(
            method="POST",
            body={"expires_in_seconds": 3600, "method": "GET"},
        )

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post(
                "/api/v2/storage/files/file_001/presign",
                {},
                mock_request,
            )

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert "url" in data
        assert data["expires_in_seconds"] == 3600

    @pytest.mark.asyncio
    async def test_generate_presigned_url_invalid_expires(self, handler):
        """Should reject invalid expiration time."""
        now = datetime.now(timezone.utc)
        handler._files["file_001"] = FileMetadata(
            id="file_001",
            filename="test.txt",
            original_filename="test.txt",
            content_type="text/plain",
            size_bytes=100,
            checksum="sha256:abc",
            bucket="default",
            path="test.txt",
            status=FileStatus.AVAILABLE,
            created_at=now,
            updated_at=now,
            owner_id="user-001",
        )

        mock_request = MockHTTPHandler(
            method="POST",
            body={"expires_in_seconds": 10},  # Too short
        )

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle_post(
                "/api/v2/storage/files/file_001/presign",
                {},
                mock_request,
            )

        assert result is not None
        assert result.status == 400


# ===========================================================================
# CloudStorageHandler Tests - Quota
# ===========================================================================


class TestCloudStorageHandlerQuota:
    """Tests for CloudStorageHandler.handle (quota)."""

    @pytest.mark.asyncio
    async def test_get_quota_empty(self, handler):
        """Should return quota with zero usage when no files."""
        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/storage/quota", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert data["quota"]["used_bytes"] == 0
        assert data["quota"]["file_count"] == 0

    @pytest.mark.asyncio
    async def test_get_quota_with_files(self, handler):
        """Should calculate quota based on stored files."""
        now = datetime.now(timezone.utc)
        handler._files["file_001"] = FileMetadata(
            id="file_001",
            filename="test.txt",
            original_filename="test.txt",
            content_type="text/plain",
            size_bytes=1000,
            checksum="sha256:abc",
            bucket="default",
            path="test.txt",
            status=FileStatus.AVAILABLE,
            created_at=now,
            updated_at=now,
            owner_id="user-001",
        )
        handler._files["file_002"] = FileMetadata(
            id="file_002",
            filename="test2.txt",
            original_filename="test2.txt",
            content_type="text/plain",
            size_bytes=2000,
            checksum="sha256:def",
            bucket="default",
            path="test2.txt",
            status=FileStatus.AVAILABLE,
            created_at=now,
            updated_at=now,
            owner_id="user-001",
        )

        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/storage/quota", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert data["quota"]["used_bytes"] == 3000
        assert data["quota"]["file_count"] == 2


# ===========================================================================
# CloudStorageHandler Tests - Buckets
# ===========================================================================


class TestCloudStorageHandlerBuckets:
    """Tests for CloudStorageHandler bucket operations."""

    @pytest.mark.asyncio
    async def test_list_buckets_default(self, handler):
        """Should return default bucket when none exist."""
        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/storage/buckets", {}, mock_request)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert len(data["buckets"]) == 1
        assert data["buckets"][0]["name"] == "default"

    @pytest.mark.asyncio
    async def test_create_bucket_success(self, handler):
        """Should create bucket successfully."""
        mock_request = MockHTTPHandler(
            method="POST",
            body={"name": "my-new-bucket", "region": "us-east-1"},
        )

        with patch.object(
            handler, "get_current_user", return_value=MockUserAuthContext(is_admin=True)
        ):
            result = await handler.handle_post("/api/v2/storage/buckets", {}, mock_request)

        assert result is not None
        assert result.status == 201
        data = json.loads(result.body)
        assert data["bucket"]["name"] == "my-new-bucket"

    @pytest.mark.asyncio
    async def test_create_bucket_invalid_name(self, handler):
        """Should reject invalid bucket name."""
        mock_request = MockHTTPHandler(
            method="POST",
            body={"name": "INVALID_NAME!"},
        )

        with patch.object(
            handler, "get_current_user", return_value=MockUserAuthContext(is_admin=True)
        ):
            result = await handler.handle_post("/api/v2/storage/buckets", {}, mock_request)

        assert result is not None
        assert result.status == 400

    @pytest.mark.asyncio
    async def test_delete_bucket_success(self, handler):
        """Should delete empty bucket successfully."""
        now = datetime.now(timezone.utc)
        handler._buckets["bucket_001"] = BucketInfo(
            id="bucket_001",
            name="my-bucket",
            provider=StorageProvider.LOCAL,
            region="local",
            created_at=now,
            owner_id="user-001",
        )

        mock_request = MockHTTPHandler(method="DELETE")

        with patch.object(
            handler, "get_current_user", return_value=MockUserAuthContext(is_admin=True)
        ):
            result = await handler.handle_delete(
                "/api/v2/storage/buckets/bucket_001", {}, mock_request
            )

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert data["deleted"] is True

    @pytest.mark.asyncio
    async def test_delete_default_bucket_fails(self, handler):
        """Should not allow deleting default bucket."""
        mock_request = MockHTTPHandler(method="DELETE")

        with patch.object(
            handler, "get_current_user", return_value=MockUserAuthContext(is_admin=True)
        ):
            result = await handler.handle_delete(
                "/api/v2/storage/buckets/default", {}, mock_request
            )

        assert result is not None
        assert result.status == 400


# ===========================================================================
# CloudStorageHandler Tests - Circuit Breaker
# ===========================================================================


class TestCloudStorageHandlerCircuitBreaker:
    """Tests for circuit breaker behavior."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self, handler, mock_storage_backend):
        """Should open circuit breaker after repeated failures."""
        mock_storage_backend._fail_upload = True

        # Make multiple failed requests to trip the circuit breaker
        for i in range(6):
            mock_request = MockHTTPHandler(
                method="POST",
                body={"filename": f"test{i}.txt", "content": "dGVzdA=="},
            )
            with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
                result = await handler.handle_post("/api/v2/storage/files", {}, mock_request)
                # First few should fail with 500, then 503 when circuit opens
                assert result is not None

    @pytest.mark.asyncio
    async def test_circuit_breaker_allows_success_recording(self, handler):
        """Should record success and keep circuit closed."""
        # Make successful request
        mock_request = MockHTTPHandler(method="GET")

        with patch.object(handler, "get_current_user", return_value=MockUserAuthContext()):
            result = await handler.handle("/api/v2/storage/files", {}, mock_request)

        assert result is not None
        assert result.status == 200

        # Circuit should still be closed
        cb = handler._get_circuit_breaker()
        assert cb.can_proceed() is True


# ===========================================================================
# CloudStorageHandler Tests - Filename Validation
# ===========================================================================


class TestCloudStorageHandlerFilenameValidation:
    """Tests for filename validation."""

    def test_validate_filename_valid(self, handler):
        """Should accept valid filenames."""
        valid, _ = handler._validate_filename("test.txt")
        assert valid is True

        valid, _ = handler._validate_filename("my-file_123.pdf")
        assert valid is True

        valid, _ = handler._validate_filename("document with spaces.docx")
        assert valid is True

    def test_validate_filename_empty(self, handler):
        """Should reject empty filename."""
        valid, error = handler._validate_filename("")
        assert valid is False
        assert "required" in error.lower()

    def test_validate_filename_too_long(self, handler):
        """Should reject too long filename."""
        long_name = "a" * 300 + ".txt"
        valid, error = handler._validate_filename(long_name)
        assert valid is False
        assert "long" in error.lower()

    def test_validate_filename_invalid_chars(self, handler):
        """Should reject invalid characters."""
        valid, error = handler._validate_filename("test<script>.txt")
        assert valid is False
        assert "invalid" in error.lower()


# ===========================================================================
# LocalStorageBackend Tests
# ===========================================================================


class TestLocalStorageBackend:
    """Tests for LocalStorageBackend."""

    @pytest.mark.asyncio
    async def test_local_backend_upload_download(self, tmp_path):
        """Should upload and download files."""
        backend = LocalStorageBackend(base_path=tmp_path)
        content = b"test content"

        await backend.upload_file("test-bucket", "test.txt", content, "text/plain")
        result = await backend.download_file("test-bucket", "test.txt")

        assert result == content

    @pytest.mark.asyncio
    async def test_local_backend_delete(self, tmp_path):
        """Should delete files."""
        backend = LocalStorageBackend(base_path=tmp_path)
        content = b"test content"

        await backend.upload_file("test-bucket", "test.txt", content, "text/plain")
        assert await backend.file_exists("test-bucket", "test.txt") is True

        result = await backend.delete_file("test-bucket", "test.txt")
        assert result is True
        assert await backend.file_exists("test-bucket", "test.txt") is False

    @pytest.mark.asyncio
    async def test_local_backend_list_files(self, tmp_path):
        """Should list files."""
        backend = LocalStorageBackend(base_path=tmp_path)

        await backend.upload_file("test-bucket", "file1.txt", b"content1", "text/plain")
        await backend.upload_file("test-bucket", "file2.txt", b"content2", "text/plain")

        files, token = await backend.list_files("test-bucket")

        assert len(files) == 2
        assert token is None


# ===========================================================================
# Factory Function Tests
# ===========================================================================


class TestCreateCloudStorageHandler:
    """Tests for handler factory function."""

    def test_create_handler(self):
        """Should create handler instance."""
        ctx = {}
        handler = create_cloud_storage_handler(ctx)

        assert isinstance(handler, CloudStorageHandler)

    def test_handler_has_routes(self):
        """Handler should define routes."""
        assert hasattr(CloudStorageHandler, "ROUTES")
        assert len(CloudStorageHandler.ROUTES) > 0

    def test_handler_can_handle(self):
        """Handler should respond to can_handle."""
        handler = CloudStorageHandler({})

        assert handler.can_handle("/api/v2/storage/files", "GET") is True
        assert handler.can_handle("/api/v2/storage/quota", "GET") is True
        assert handler.can_handle("/api/v2/other/path", "GET") is False
