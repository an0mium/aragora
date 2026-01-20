"""
Pytest fixtures for enterprise connector tests.

Provides mock credentials, API responses, and connector instances
for testing without external dependencies.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.enterprise.base import (
    SyncItem,
    SyncState,
    SyncStatus,
    SyncResult,
)


# =============================================================================
# Mock Credentials
# =============================================================================


class MockCredentialProvider:
    """Mock credential provider for testing."""

    def __init__(self, credentials: Optional[Dict[str, str]] = None):
        self._credentials = credentials or {}

    async def get_credential(self, key: str) -> Optional[str]:
        """Return mock credential."""
        return self._credentials.get(key)

    def set_credential(self, key: str, value: str) -> None:
        """Set mock credential for testing."""
        self._credentials[key] = value


@pytest.fixture
def mock_credentials() -> MockCredentialProvider:
    """Create mock credential provider."""
    return MockCredentialProvider(
        {
            "MONGO_USER": "test_user",
            "MONGO_PASSWORD": "test_password",
            "SLACK_BOT_TOKEN": "xoxb-test-token-12345",
            "NOTION_API_KEY": "secret_test_notion_key",
            "CONFLUENCE_API_TOKEN": "test_confluence_token",
        }
    )


# =============================================================================
# Sync State Fixtures
# =============================================================================


@pytest.fixture
def fresh_sync_state() -> SyncState:
    """Create a fresh sync state for testing."""
    return SyncState(
        connector_id="test_connector",
        tenant_id="test_tenant",
        status=SyncStatus.IDLE,
    )


@pytest.fixture
def in_progress_sync_state() -> SyncState:
    """Create an in-progress sync state."""
    return SyncState(
        connector_id="test_connector",
        tenant_id="test_tenant",
        cursor="cursor_abc123",
        last_sync_at=datetime.now(timezone.utc),
        items_synced=50,
        items_total=100,
        status=SyncStatus.RUNNING,
        started_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def completed_sync_state() -> SyncState:
    """Create a completed sync state."""
    return SyncState(
        connector_id="test_connector",
        tenant_id="test_tenant",
        cursor="cursor_final",
        last_sync_at=datetime.now(timezone.utc),
        last_item_id="item_100",
        last_item_timestamp=datetime.now(timezone.utc),
        items_synced=100,
        items_total=100,
        status=SyncStatus.COMPLETED,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )


# =============================================================================
# MongoDB Fixtures
# =============================================================================


@pytest.fixture
def mock_mongo_documents() -> List[Dict[str, Any]]:
    """Sample MongoDB documents for testing."""
    return [
        {
            "_id": "doc1",
            "title": "First Document",
            "content": "This is the first document content.",
            "updated_at": datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
            "tags": ["test", "sample"],
        },
        {
            "_id": "doc2",
            "title": "Second Document",
            "content": "This is the second document content.",
            "updated_at": datetime(2024, 1, 16, 14, 45, tzinfo=timezone.utc),
            "tags": ["test"],
        },
        {
            "_id": "doc3",
            "title": "Third Document",
            "content": "Nested content with data.",
            "updated_at": datetime(2024, 1, 17, 9, 0, tzinfo=timezone.utc),
            "metadata": {"author": "tester", "version": 1},
        },
    ]


@pytest.fixture
def mock_mongo_client(mock_mongo_documents):
    """Create a mock MongoDB client."""
    mock_collection = MagicMock()
    mock_collection.find = MagicMock(return_value=AsyncIteratorMock(mock_mongo_documents))
    mock_collection.count_documents = AsyncMock(return_value=len(mock_mongo_documents))

    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_collection)
    mock_db.list_collection_names = AsyncMock(return_value=["test_collection", "users"])

    mock_client = MagicMock()
    mock_client.__getitem__ = MagicMock(return_value=mock_db)

    return mock_client


# =============================================================================
# Slack Fixtures
# =============================================================================


@pytest.fixture
def mock_slack_channels() -> List[Dict[str, Any]]:
    """Sample Slack channels for testing."""
    return [
        {
            "id": "C001",
            "name": "general",
            "is_private": False,
            "is_archived": False,
            "topic": {"value": "Company-wide announcements"},
            "purpose": {"value": "General discussion"},
            "num_members": 50,
            "created": 1609459200,
        },
        {
            "id": "C002",
            "name": "engineering",
            "is_private": False,
            "is_archived": False,
            "topic": {"value": "Engineering team"},
            "purpose": {"value": "Engineering discussions"},
            "num_members": 20,
            "created": 1609459200,
        },
    ]


@pytest.fixture
def mock_slack_messages() -> List[Dict[str, Any]]:
    """Sample Slack messages for testing."""
    return [
        {
            "ts": "1704067200.000001",
            "text": "Hello everyone!",
            "user": "U001",
            "type": "message",
        },
        {
            "ts": "1704067300.000002",
            "text": "Welcome to the channel",
            "user": "U002",
            "type": "message",
            "thread_ts": "1704067200.000001",
            "reply_count": 2,
        },
        {
            "ts": "1704067400.000003",
            "text": "Check out this file",
            "user": "U001",
            "type": "message",
            "files": [{"id": "F001", "name": "document.pdf"}],
        },
    ]


@pytest.fixture
def mock_slack_users() -> List[Dict[str, Any]]:
    """Sample Slack users for testing."""
    return [
        {
            "id": "U001",
            "name": "alice",
            "real_name": "Alice Smith",
            "profile": {"display_name": "alice.smith", "email": "alice@example.com"},
            "is_bot": False,
        },
        {
            "id": "U002",
            "name": "bob",
            "real_name": "Bob Jones",
            "profile": {"display_name": "bob.jones", "email": "bob@example.com"},
            "is_bot": False,
        },
    ]


@pytest.fixture
def mock_slack_client(mock_slack_channels, mock_slack_messages, mock_slack_users):
    """Create a mock Slack WebClient."""
    mock_client = MagicMock()

    # Mock conversations_list
    mock_client.conversations_list = AsyncMock(
        return_value={"channels": mock_slack_channels, "response_metadata": {"next_cursor": ""}}
    )

    # Mock conversations_history
    mock_client.conversations_history = AsyncMock(
        return_value={"messages": mock_slack_messages, "has_more": False}
    )

    # Mock users_list
    mock_client.users_list = AsyncMock(
        return_value={"members": mock_slack_users, "response_metadata": {"next_cursor": ""}}
    )

    # Mock conversations_replies
    mock_client.conversations_replies = AsyncMock(
        return_value={"messages": mock_slack_messages[:2]}
    )

    return mock_client


# =============================================================================
# Notion Fixtures
# =============================================================================


@pytest.fixture
def mock_notion_pages() -> List[Dict[str, Any]]:
    """Sample Notion pages for testing."""
    return [
        {
            "id": "page-001",
            "object": "page",
            "created_time": "2024-01-15T10:00:00.000Z",
            "last_edited_time": "2024-01-16T14:30:00.000Z",
            "properties": {
                "title": {"title": [{"plain_text": "Project Overview"}]},
                "Status": {"select": {"name": "In Progress"}},
            },
            "url": "https://notion.so/page-001",
        },
        {
            "id": "page-002",
            "object": "page",
            "created_time": "2024-01-14T09:00:00.000Z",
            "last_edited_time": "2024-01-17T11:00:00.000Z",
            "properties": {
                "title": {"title": [{"plain_text": "Technical Specs"}]},
                "Status": {"select": {"name": "Completed"}},
            },
            "url": "https://notion.so/page-002",
        },
    ]


@pytest.fixture
def mock_notion_blocks() -> List[Dict[str, Any]]:
    """Sample Notion blocks for testing."""
    return [
        {
            "id": "block-001",
            "type": "paragraph",
            "paragraph": {"rich_text": [{"plain_text": "This is a paragraph."}]},
        },
        {
            "id": "block-002",
            "type": "heading_1",
            "heading_1": {"rich_text": [{"plain_text": "Section Title"}]},
        },
        {
            "id": "block-003",
            "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": [{"plain_text": "First item"}]},
        },
    ]


# =============================================================================
# Confluence Fixtures
# =============================================================================


@pytest.fixture
def mock_confluence_pages() -> List[Dict[str, Any]]:
    """Sample Confluence pages for testing."""
    return [
        {
            "id": "123456",
            "type": "page",
            "title": "Engineering Guidelines",
            "space": {"key": "ENG", "name": "Engineering"},
            "body": {"storage": {"value": "<p>Guidelines content here.</p>"}},
            "version": {"number": 5},
            "_links": {"webui": "/display/ENG/Engineering+Guidelines"},
            "history": {"lastUpdated": {"when": "2024-01-16T10:00:00.000Z"}},
        },
        {
            "id": "789012",
            "type": "page",
            "title": "API Documentation",
            "space": {"key": "DEV", "name": "Development"},
            "body": {"storage": {"value": "<h1>API Docs</h1><p>Documentation.</p>"}},
            "version": {"number": 12},
            "_links": {"webui": "/display/DEV/API+Documentation"},
            "history": {"lastUpdated": {"when": "2024-01-17T15:30:00.000Z"}},
        },
    ]


# =============================================================================
# Async Iterator Helper
# =============================================================================


class AsyncIteratorMock:
    """Mock async iterator for database cursors."""

    def __init__(self, items: List[Any]):
        self._items = items
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item

    def sort(self, *args, **kwargs):
        """Support sort chaining."""
        return self

    def limit(self, *args, **kwargs):
        """Support limit chaining."""
        return self

    def skip(self, *args, **kwargs):
        """Support skip chaining."""
        return self


# =============================================================================
# Generic Sync Item Fixtures
# =============================================================================


@pytest.fixture
def sample_sync_items() -> List[SyncItem]:
    """Create sample sync items for testing."""
    return [
        SyncItem(
            id="item-001",
            content="First item content for testing.",
            source_type="database",
            source_id="mongodb://localhost/test/items/001",
            title="First Item",
            url="mongodb://localhost/test?id=001",
            updated_at=datetime.now(timezone.utc),
            domain="operational/data",
            confidence=0.9,
            metadata={"collection": "items", "type": "document"},
        ),
        SyncItem(
            id="item-002",
            content="Second item content with more details.",
            source_type="collaboration",
            source_id="slack://workspace/channel/message",
            title="Slack Message",
            url="https://slack.com/archives/C001/p1234567890",
            updated_at=datetime.now(timezone.utc),
            domain="operational/communication",
            confidence=0.85,
            metadata={"channel": "general", "user": "alice"},
        ),
    ]
