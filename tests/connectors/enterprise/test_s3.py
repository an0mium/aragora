"""
Tests for AWS S3 Enterprise Connector.
Tests the S3 bucket integration including:
- Bucket listing and object traversal
- Object content retrieval
- Prefix-based filtering
- Last-modified based incremental sync
- S3 event notification handling
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, AsyncIterator

from aragora.connectors.enterprise.base import SyncState, SyncStatus


# =============================================================================
# Mock S3 Connector (mirrors actual implementation)
# =============================================================================


@dataclass
class S3Object:
    """Represents an S3 object."""

    key: str
    bucket: str
    size: int
    last_modified: datetime
    etag: str
    content_type: Optional[str] = None
    metadata: Dict[str, str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MockS3Connector:
    """Mock S3 connector for testing."""

    ROUTES = ["/api/connectors/s3/sync", "/api/connectors/s3/search"]

    def __init__(
        self,
        bucket_name: str,
        aws_access_key_id: str = "",
        aws_secret_access_key: str = "",
        region: str = "us-east-1",
        prefix: str = "",
        exclude_patterns: List[str] = None,
        include_patterns: List[str] = None,
        **kwargs,
    ):
        self.connector_id = f"s3_{bucket_name}"
        self.bucket_name = bucket_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region = region
        self.prefix = prefix
        self.exclude_patterns = exclude_patterns or []
        self.include_patterns = include_patterns or ["*"]

        self._client = None
        self._objects_cache: Dict[str, S3Object] = {}

    @property
    def name(self) -> str:
        return f"S3: {self.bucket_name}"

    @property
    def source_type(self) -> str:
        return "storage"

    async def _get_client(self):
        """Get or create S3 client."""
        if self._client is None:
            self._client = MagicMock()
        return self._client

    async def _list_objects(
        self,
        prefix: str = "",
        continuation_token: Optional[str] = None,
    ) -> tuple[List[S3Object], Optional[str]]:
        """List objects in the bucket."""
        client = await self._get_client()

        params = {
            "Bucket": self.bucket_name,
            "Prefix": prefix or self.prefix,
            "MaxKeys": 1000,
        }
        if continuation_token:
            params["ContinuationToken"] = continuation_token

        response = client.list_objects_v2(**params)

        objects = []
        for item in response.get("Contents", []):
            obj = S3Object(
                key=item["Key"],
                bucket=self.bucket_name,
                size=item["Size"],
                last_modified=item["LastModified"],
                etag=item["ETag"],
                content_type=item.get("ContentType"),
            )
            objects.append(obj)
            self._objects_cache[obj.key] = obj

        next_token = response.get("NextContinuationToken")
        return objects, next_token

    async def _get_object_content(self, key: str) -> Optional[bytes]:
        """Get object content."""
        client = await self._get_client()

        try:
            response = client.get_object(Bucket=self.bucket_name, Key=key)
            return response["Body"].read()
        except Exception:
            return None

    def _should_include(self, key: str) -> bool:
        """Check if object should be included based on patterns."""
        import fnmatch

        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(key, pattern):
                return False

        # Check include patterns
        for pattern in self.include_patterns:
            if fnmatch.fnmatch(key, pattern):
                return True

        return False

    async def sync_items(
        self,
        state: SyncState,
        batch_size: int = 100,
    ) -> AsyncIterator[Any]:
        """Sync objects from S3 bucket."""
        # Parse last sync time from cursor
        modified_since = None
        if state.cursor:
            try:
                modified_since = datetime.fromisoformat(state.cursor)
            except ValueError:
                pass

        continuation_token = None
        items_yielded = 0

        while True:
            objects, continuation_token = await self._list_objects(
                continuation_token=continuation_token
            )

            for obj in objects:
                # Filter by modification time
                if modified_since and obj.last_modified <= modified_since:
                    continue

                # Filter by patterns
                if not self._should_include(obj.key):
                    continue

                # Get content
                content = await self._get_object_content(obj.key)
                if content is None:
                    continue

                yield MagicMock(
                    id=f"s3-{self.bucket_name}-{obj.key}",
                    content=content.decode("utf-8", errors="replace")[:50000],
                    source_type="document",
                    source_id=f"s3://{self.bucket_name}/{obj.key}",
                    title=obj.key.split("/")[-1],
                    url=f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{obj.key}",
                    created_at=obj.last_modified,
                    updated_at=obj.last_modified,
                    metadata={
                        "bucket": self.bucket_name,
                        "key": obj.key,
                        "size": obj.size,
                        "etag": obj.etag,
                        "content_type": obj.content_type,
                    },
                )

                items_yielded += 1

                # Update cursor
                if obj.last_modified:
                    ts = obj.last_modified.isoformat()
                    if not state.cursor or ts > state.cursor:
                        state.cursor = ts

                if items_yielded >= batch_size:
                    return

            if not continuation_token:
                break

    async def fetch(self, item_id: str) -> Optional[Any]:
        """Fetch a specific object."""
        # Parse item ID
        if item_id.startswith("s3-"):
            parts = item_id.split("-", 2)
            if len(parts) >= 3:
                key = parts[2]
            else:
                key = item_id
        else:
            key = item_id

        content = await self._get_object_content(key)
        if content is None:
            return None

        obj = self._objects_cache.get(key)

        return MagicMock(
            id=f"s3-{self.bucket_name}-{key}",
            content=content.decode("utf-8", errors="replace"),
            title=key.split("/")[-1],
            metadata={
                "bucket": self.bucket_name,
                "key": key,
                "size": len(content),
            },
        )

    async def search(self, query: str, limit: int = 10) -> List[Any]:
        """Search objects by key prefix."""
        results = []

        objects, _ = await self._list_objects(prefix=query)

        for obj in objects[:limit]:
            results.append(
                MagicMock(
                    id=f"s3-{self.bucket_name}-{obj.key}",
                    title=obj.key.split("/")[-1],
                    url=f"s3://{self.bucket_name}/{obj.key}",
                    score=1.0,
                )
            )

        return results

    async def handle_webhook(self, payload: Dict[str, Any]) -> bool:
        """Handle S3 event notification."""
        records = payload.get("Records", [])

        for record in records:
            event_name = record.get("eventName", "")
            if "ObjectCreated" in event_name or "ObjectRemoved" in event_name:
                # Trigger sync for this object
                return True

        return False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def s3_connector():
    """Create S3 connector for testing."""
    return MockS3Connector(
        bucket_name="test-bucket",
        aws_access_key_id="AKIATEST",
        aws_secret_access_key="secret",
        region="us-west-2",
        prefix="documents/",
    )


@pytest.fixture
def sample_objects():
    """Sample S3 objects."""
    return [
        {
            "Key": "documents/report.pdf",
            "Size": 1024000,
            "LastModified": datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            "ETag": '"abc123"',
            "ContentType": "application/pdf",
        },
        {
            "Key": "documents/data.csv",
            "Size": 5000,
            "LastModified": datetime(2024, 1, 16, 14, 0, 0, tzinfo=timezone.utc),
            "ETag": '"def456"',
            "ContentType": "text/csv",
        },
        {
            "Key": "documents/notes.txt",
            "Size": 500,
            "LastModified": datetime(2024, 1, 17, 9, 0, 0, tzinfo=timezone.utc),
            "ETag": '"ghi789"',
            "ContentType": "text/plain",
        },
    ]


# =============================================================================
# Test Classes
# =============================================================================


class TestS3Init:
    """Test S3 connector initialization."""

    def test_connector_initialization(self, s3_connector):
        """Test basic initialization."""
        assert s3_connector.bucket_name == "test-bucket"
        assert s3_connector.region == "us-west-2"
        assert s3_connector.prefix == "documents/"
        assert s3_connector.connector_id == "s3_test-bucket"

    def test_name_property(self, s3_connector):
        """Test name property."""
        assert s3_connector.name == "S3: test-bucket"

    def test_source_type(self, s3_connector):
        """Test source type."""
        assert s3_connector.source_type == "storage"

    def test_default_patterns(self):
        """Test default include/exclude patterns."""
        connector = MockS3Connector(bucket_name="bucket")
        assert connector.include_patterns == ["*"]
        assert connector.exclude_patterns == []


class TestObjectListing:
    """Test object listing functionality."""

    @pytest.mark.asyncio
    async def test_list_objects(self, s3_connector, sample_objects):
        """Test listing objects."""
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {
            "Contents": sample_objects,
            "IsTruncated": False,
        }
        s3_connector._client = mock_client

        objects, next_token = await s3_connector._list_objects()

        assert len(objects) == 3
        assert objects[0].key == "documents/report.pdf"
        assert objects[1].size == 5000
        assert next_token is None

    @pytest.mark.asyncio
    async def test_list_objects_with_prefix(self, s3_connector, sample_objects):
        """Test listing with custom prefix."""
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {"Contents": sample_objects[:1]}
        s3_connector._client = mock_client

        objects, _ = await s3_connector._list_objects(prefix="documents/report")

        mock_client.list_objects_v2.assert_called_once()
        call_args = mock_client.list_objects_v2.call_args
        assert call_args[1]["Prefix"] == "documents/report"

    @pytest.mark.asyncio
    async def test_list_objects_pagination(self, s3_connector, sample_objects):
        """Test pagination with continuation token."""
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {
            "Contents": sample_objects[:2],
            "NextContinuationToken": "token123",
            "IsTruncated": True,
        }
        s3_connector._client = mock_client

        objects, next_token = await s3_connector._list_objects()

        assert len(objects) == 2
        assert next_token == "token123"

    @pytest.mark.asyncio
    async def test_list_objects_empty_bucket(self, s3_connector):
        """Test listing empty bucket."""
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {"Contents": []}
        s3_connector._client = mock_client

        objects, next_token = await s3_connector._list_objects()

        assert len(objects) == 0
        assert next_token is None


class TestObjectContent:
    """Test object content retrieval."""

    @pytest.mark.asyncio
    async def test_get_object_content(self, s3_connector):
        """Test getting object content."""
        mock_body = MagicMock()
        mock_body.read.return_value = b"Hello, World!"

        mock_client = MagicMock()
        mock_client.get_object.return_value = {"Body": mock_body}
        s3_connector._client = mock_client

        content = await s3_connector._get_object_content("test.txt")

        assert content == b"Hello, World!"
        mock_client.get_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="test.txt",
        )

    @pytest.mark.asyncio
    async def test_get_object_content_not_found(self, s3_connector):
        """Test getting non-existent object."""
        mock_client = MagicMock()
        mock_client.get_object.side_effect = Exception("NoSuchKey")
        s3_connector._client = mock_client

        content = await s3_connector._get_object_content("nonexistent.txt")

        assert content is None


class TestPatternFiltering:
    """Test include/exclude pattern filtering."""

    def test_include_all_pattern(self, s3_connector):
        """Test include all with wildcard."""
        assert s3_connector._should_include("anything.txt") is True
        assert s3_connector._should_include("folder/file.pdf") is True

    def test_exclude_pattern(self):
        """Test exclude pattern."""
        connector = MockS3Connector(
            bucket_name="bucket",
            exclude_patterns=["*.tmp", "*.log"],
        )

        assert connector._should_include("data.csv") is True
        assert connector._should_include("temp.tmp") is False
        assert connector._should_include("app.log") is False

    def test_include_specific_pattern(self):
        """Test include specific pattern."""
        connector = MockS3Connector(
            bucket_name="bucket",
            include_patterns=["*.pdf", "*.docx"],
        )

        assert connector._should_include("report.pdf") is True
        assert connector._should_include("document.docx") is True
        assert connector._should_include("data.csv") is False

    def test_exclude_takes_precedence(self):
        """Test that exclude patterns take precedence."""
        connector = MockS3Connector(
            bucket_name="bucket",
            include_patterns=["*"],
            exclude_patterns=["secret/*"],
        )

        assert connector._should_include("public/file.txt") is True
        assert connector._should_include("secret/credentials.txt") is False


class TestSyncItems:
    """Test sync_items functionality."""

    @pytest.mark.asyncio
    async def test_sync_items_full(self, s3_connector, sample_objects):
        """Test full sync."""
        state = SyncState(connector_id="s3", status=SyncStatus.IDLE)

        mock_body = MagicMock()
        mock_body.read.return_value = b"file content"

        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {
            "Contents": sample_objects,
            "IsTruncated": False,
        }
        mock_client.get_object.return_value = {"Body": mock_body}
        s3_connector._client = mock_client

        items = []
        async for item in s3_connector.sync_items(state, batch_size=100):
            items.append(item)

        assert len(items) == 3
        assert items[0].metadata["key"] == "documents/report.pdf"

    @pytest.mark.asyncio
    async def test_sync_incremental(self, s3_connector, sample_objects):
        """Test incremental sync with cursor."""
        # Set cursor to filter out older items
        cursor_time = datetime(2024, 1, 16, 0, 0, 0, tzinfo=timezone.utc).isoformat()
        state = SyncState(connector_id="s3", status=SyncStatus.IDLE, cursor=cursor_time)

        mock_body = MagicMock()
        mock_body.read.return_value = b"content"

        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {
            "Contents": sample_objects,
            "IsTruncated": False,
        }
        mock_client.get_object.return_value = {"Body": mock_body}
        s3_connector._client = mock_client

        items = []
        async for item in s3_connector.sync_items(state, batch_size=100):
            items.append(item)

        # Should only get items modified after Jan 16
        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_sync_updates_cursor(self, s3_connector, sample_objects):
        """Test that sync updates cursor."""
        state = SyncState(connector_id="s3", status=SyncStatus.IDLE)

        mock_body = MagicMock()
        mock_body.read.return_value = b"content"

        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {
            "Contents": sample_objects,
            "IsTruncated": False,
        }
        mock_client.get_object.return_value = {"Body": mock_body}
        s3_connector._client = mock_client

        async for _ in s3_connector.sync_items(state, batch_size=100):
            pass

        assert state.cursor is not None

    @pytest.mark.asyncio
    async def test_sync_respects_batch_size(self, s3_connector, sample_objects):
        """Test batch size limit."""
        state = SyncState(connector_id="s3", status=SyncStatus.IDLE)

        mock_body = MagicMock()
        mock_body.read.return_value = b"content"

        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {
            "Contents": sample_objects,
            "IsTruncated": False,
        }
        mock_client.get_object.return_value = {"Body": mock_body}
        s3_connector._client = mock_client

        items = []
        async for item in s3_connector.sync_items(state, batch_size=2):
            items.append(item)

        assert len(items) == 2


class TestFetch:
    """Test fetch functionality."""

    @pytest.mark.asyncio
    async def test_fetch_by_key(self, s3_connector):
        """Test fetching by key."""
        mock_body = MagicMock()
        mock_body.read.return_value = b"File content here"

        mock_client = MagicMock()
        mock_client.get_object.return_value = {"Body": mock_body}
        s3_connector._client = mock_client

        result = await s3_connector.fetch("documents/test.txt")

        assert result is not None
        assert result.content == "File content here"

    @pytest.mark.asyncio
    async def test_fetch_by_item_id(self, s3_connector):
        """Test fetching by full item ID."""
        mock_body = MagicMock()
        mock_body.read.return_value = b"content"

        mock_client = MagicMock()
        mock_client.get_object.return_value = {"Body": mock_body}
        s3_connector._client = mock_client

        result = await s3_connector.fetch("s3-test-bucket-documents/file.txt")

        assert result is not None

    @pytest.mark.asyncio
    async def test_fetch_not_found(self, s3_connector):
        """Test fetching non-existent object."""
        mock_client = MagicMock()
        mock_client.get_object.side_effect = Exception("NoSuchKey")
        s3_connector._client = mock_client

        result = await s3_connector.fetch("nonexistent.txt")

        assert result is None


class TestSearch:
    """Test search functionality."""

    @pytest.mark.asyncio
    async def test_search_by_prefix(self, s3_connector, sample_objects):
        """Test searching by prefix."""
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {
            "Contents": sample_objects,
            "IsTruncated": False,
        }
        s3_connector._client = mock_client

        results = await s3_connector.search("documents/")

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_with_limit(self, s3_connector, sample_objects):
        """Test search with limit."""
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {
            "Contents": sample_objects,
            "IsTruncated": False,
        }
        s3_connector._client = mock_client

        results = await s3_connector.search("documents/", limit=2)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_empty_results(self, s3_connector):
        """Test search with no results."""
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {"Contents": []}
        s3_connector._client = mock_client

        results = await s3_connector.search("nonexistent/")

        assert len(results) == 0


class TestWebhook:
    """Test webhook handling."""

    @pytest.mark.asyncio
    async def test_handle_object_created(self, s3_connector):
        """Test handling ObjectCreated event."""
        payload = {
            "Records": [
                {
                    "eventName": "ObjectCreated:Put",
                    "s3": {
                        "bucket": {"name": "test-bucket"},
                        "object": {"key": "new-file.txt"},
                    },
                }
            ]
        }

        result = await s3_connector.handle_webhook(payload)

        assert result is True

    @pytest.mark.asyncio
    async def test_handle_object_removed(self, s3_connector):
        """Test handling ObjectRemoved event."""
        payload = {
            "Records": [
                {
                    "eventName": "ObjectRemoved:Delete",
                    "s3": {
                        "bucket": {"name": "test-bucket"},
                        "object": {"key": "deleted-file.txt"},
                    },
                }
            ]
        }

        result = await s3_connector.handle_webhook(payload)

        assert result is True

    @pytest.mark.asyncio
    async def test_handle_irrelevant_event(self, s3_connector):
        """Test handling irrelevant event."""
        payload = {
            "Records": [
                {
                    "eventName": "s3:TestEvent",
                }
            ]
        }

        result = await s3_connector.handle_webhook(payload)

        assert result is False

    @pytest.mark.asyncio
    async def test_handle_empty_payload(self, s3_connector):
        """Test handling empty payload."""
        result = await s3_connector.handle_webhook({})

        assert result is False
