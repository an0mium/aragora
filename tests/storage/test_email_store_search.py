"""Tests for EmailStore FTS5 search functionality."""

import pytest
import tempfile
from pathlib import Path

from aragora.storage.email_store import EmailStore, reset_email_store


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield str(Path(tmpdir) / "test_email_store.db")


@pytest.fixture
def store(temp_db_path):
    """Create a fresh EmailStore for testing."""
    reset_email_store()
    store = EmailStore(temp_db_path)
    yield store
    reset_email_store()


@pytest.fixture
def store_with_messages(store):
    """Create store with sample messages."""
    # Create a shared inbox first
    store.create_shared_inbox(
        inbox_id="inbox_123",
        workspace_id="ws_456",
        name="Support Inbox",
    )

    # Add sample messages
    messages = [
        {
            "id": "msg_1",
            "subject": "Help with login issue",
            "from_address": "john@example.com",
            "snippet": "I am having trouble logging into my account. Password reset is not working.",
            "status": "open",
        },
        {
            "id": "msg_2",
            "subject": "Billing question about invoice",
            "from_address": "jane@company.org",
            "snippet": "I received an invoice but the amount seems incorrect. Can you check?",
            "status": "assigned",
            "assigned_to": "agent_1",
        },
        {
            "id": "msg_3",
            "subject": "Feature request for dashboard",
            "from_address": "bob@startup.io",
            "snippet": "Would love to see a new chart feature on the dashboard for metrics.",
            "status": "open",
        },
        {
            "id": "msg_4",
            "subject": "Account password reset request",
            "from_address": "alice@domain.net",
            "snippet": "Please help me reset my account password. I forgot it.",
            "status": "resolved",
        },
        {
            "id": "msg_5",
            "subject": "Integration API documentation",
            "from_address": "dev@techcorp.com",
            "snippet": "Where can I find the API documentation for integrations?",
            "status": "open",
        },
    ]

    for msg in messages:
        store.save_message(
            message_id=msg["id"],
            inbox_id="inbox_123",
            workspace_id="ws_456",
            subject=msg["subject"],
            from_address=msg["from_address"],
            snippet=msg["snippet"],
            status=msg.get("status", "open"),
            assigned_to=msg.get("assigned_to"),
        )

    return store


class TestFTSInitialization:
    """Tests for FTS5 table initialization."""

    def test_fts_lazy_init(self, store):
        """Test that FTS is initialized lazily on first search."""
        # Check table doesn't exist initially (it's created lazily)
        check = store.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='shared_inbox_messages_fts'"
        )
        assert check is None

        # Trigger initialization
        store._ensure_fts_table()

        # Now table should exist
        check = store.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='shared_inbox_messages_fts'"
        )
        assert check is not None

    def test_fts_idempotent_init(self, store):
        """Test that multiple init calls are safe."""
        store._ensure_fts_table()
        store._ensure_fts_table()  # Should not error

        # Table should still exist
        check = store.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='shared_inbox_messages_fts'"
        )
        assert check is not None


class TestSearchMessages:
    """Tests for full-text search functionality."""

    def test_search_by_subject(self, store_with_messages):
        """Test searching messages by subject."""
        results = store_with_messages.search_messages(
            inbox_id="inbox_123",
            query="login",
        )

        assert len(results) == 1
        assert results[0]["id"] == "msg_1"

    def test_search_by_snippet(self, store_with_messages):
        """Test searching messages by snippet content."""
        results = store_with_messages.search_messages(
            inbox_id="inbox_123",
            query="password",
        )

        # Should find msg_1 (password reset not working) and msg_4 (password reset)
        assert len(results) == 2
        message_ids = [r["id"] for r in results]
        assert "msg_1" in message_ids
        assert "msg_4" in message_ids

    def test_search_by_sender(self, store_with_messages):
        """Test searching messages by sender email domain."""
        # Search for company to find jane@company.org
        results = store_with_messages.search_messages(
            inbox_id="inbox_123",
            query="company",
        )

        assert len(results) == 1
        assert results[0]["id"] == "msg_2"

    def test_search_with_status_filter(self, store_with_messages):
        """Test search with status filter."""
        # Search for "password" but only open messages
        results = store_with_messages.search_messages(
            inbox_id="inbox_123",
            query="password",
            status="open",
        )

        assert len(results) == 1
        assert results[0]["id"] == "msg_1"
        assert results[0]["status"] == "open"

    def test_search_with_assignee_filter(self, store_with_messages):
        """Test search with assignee filter."""
        results = store_with_messages.search_messages(
            inbox_id="inbox_123",
            query="invoice",
            assigned_to="agent_1",
        )

        assert len(results) == 1
        assert results[0]["assigned_to"] == "agent_1"

    def test_search_no_results(self, store_with_messages):
        """Test search with no matching results."""
        results = store_with_messages.search_messages(
            inbox_id="inbox_123",
            query="nonexistent term xyz",
        )

        assert len(results) == 0

    def test_search_empty_query(self, store_with_messages):
        """Test search with empty query returns all messages."""
        results = store_with_messages.search_messages(
            inbox_id="inbox_123",
            query="",
        )

        # Should fall back to list_inbox_messages
        assert len(results) == 5

    def test_search_with_pagination(self, store_with_messages):
        """Test search with limit and offset."""
        # First add more messages to test pagination
        for i in range(5):
            store_with_messages.save_message(
                message_id=f"msg_page_{i}",
                inbox_id="inbox_123",
                workspace_id="ws_456",
                subject=f"Support request number {i}",
                from_address=f"user{i}@test.com",
                snippet=f"This is support request {i} needing help",
                status="open",
            )

        # Re-init FTS to pick up new messages
        store_with_messages._ensure_fts_table()
        for i in range(5):
            store_with_messages.index_message_for_search(f"msg_page_{i}")

        # Search for common term "request"
        results_page1 = store_with_messages.search_messages(
            inbox_id="inbox_123",
            query="request",
            limit=2,
            offset=0,
        )

        results_page2 = store_with_messages.search_messages(
            inbox_id="inbox_123",
            query="request",
            limit=2,
            offset=2,
        )

        assert len(results_page1) == 2
        assert len(results_page2) >= 1

    def test_search_wrong_inbox(self, store_with_messages):
        """Test search returns nothing for wrong inbox."""
        results = store_with_messages.search_messages(
            inbox_id="wrong_inbox",
            query="password",
        )

        assert len(results) == 0


class TestSearchWithSnippets:
    """Tests for search with highlighted snippets."""

    def test_search_with_snippets(self, store_with_messages):
        """Test search returns highlighted snippets."""
        results = store_with_messages.search_messages_with_snippets(
            inbox_id="inbox_123",
            query="password",
            limit=10,
        )

        assert len(results) >= 1
        assert "snippet_highlight" in results[0]
        assert "message_id" in results[0]
        assert "rank" in results[0]

    def test_snippets_empty_query(self, store_with_messages):
        """Test snippets returns empty for empty query."""
        results = store_with_messages.search_messages_with_snippets(
            inbox_id="inbox_123",
            query="",
        )

        assert len(results) == 0


class TestSearchQueryEscaping:
    """Tests for FTS query escaping."""

    def test_escape_special_chars(self, store):
        """Test that special FTS characters are escaped and terms are quoted."""
        # These are dangerous FTS5 characters
        dangerous_query = "test*query^with:special(chars)"

        escaped = store._escape_fts_query(dangerous_query)

        # Should be quoted terms without dangerous chars
        assert "*" not in escaped
        assert "^" not in escaped
        assert ":" not in escaped
        assert "(" not in escaped
        assert ")" not in escaped
        # Should have quoted terms
        assert '"' in escaped

    def test_multi_word_becomes_quoted_terms(self, store):
        """Test that multi-word queries become quoted terms."""
        query = "hello world test"
        escaped = store._escape_fts_query(query)

        # Each word should be quoted
        assert '"hello"' in escaped
        assert '"world"' in escaped
        assert '"test"' in escaped


class TestIndexMessage:
    """Tests for message indexing."""

    def test_index_new_message(self, store):
        """Test indexing a message for search."""
        store.create_shared_inbox("inbox_1", "ws_1", "Test")
        store.save_message(
            message_id="msg_new",
            inbox_id="inbox_1",
            workspace_id="ws_1",
            subject="New unique subject",
            snippet="This is a unique message for testing",
        )

        # Index it
        store.index_message_for_search("msg_new")

        # Search for it
        results = store.search_messages("inbox_1", "unique")
        assert len(results) == 1
        assert results[0]["id"] == "msg_new"

    def test_reindex_message(self, store):
        """Test re-indexing updates the search index."""
        store.create_shared_inbox("inbox_1", "ws_1", "Test")
        store.save_message(
            message_id="msg_update",
            inbox_id="inbox_1",
            workspace_id="ws_1",
            subject="Original subject",
            snippet="Original content",
        )
        store.index_message_for_search("msg_update")

        # Update the message
        store.save_message(
            message_id="msg_update",
            inbox_id="inbox_1",
            workspace_id="ws_1",
            subject="Updated subject",
            snippet="Updated content with new keywords",
        )
        store.index_message_for_search("msg_update")

        # Search for new content
        results = store.search_messages("inbox_1", "Updated")
        assert len(results) == 1

        # Old content should not be found
        results_old = store.search_messages("inbox_1", "Original")
        assert len(results_old) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
