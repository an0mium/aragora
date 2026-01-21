"""Tests for ApprovalRequestStore backends."""

import asyncio
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from aragora.storage.approval_request_store import (
    ApprovalRequestItem,
    InMemoryApprovalRequestStore,
    SQLiteApprovalRequestStore,
    get_approval_request_store,
    reset_approval_request_store,
)


@pytest.fixture
def sample_request_data():
    """Sample approval request data for testing."""
    return {
        "request_id": "approval-test-001",
        "workflow_id": "workflow-123",
        "step_id": "step-1",
        "title": "Review deployment plan",
        "status": "pending",
        "description": "Please review the deployment plan for production",
        "request_data": {"deployment_version": "1.0.0", "target_env": "prod"},
        "requester_id": "user-123",
        "workspace_id": "ws-456",
        "priority": 2,
        "tags": ["deployment", "prod"],
    }


@pytest.fixture
def sample_request_data_2():
    """Second sample for listing tests."""
    return {
        "request_id": "approval-test-002",
        "workflow_id": "workflow-456",
        "step_id": "step-2",
        "title": "Approve security change",
        "status": "pending",
        "description": "Security configuration update",
        "request_data": {"change_type": "security"},
        "requester_id": "user-789",
        "workspace_id": "ws-456",
        "priority": 1,
        "tags": ["security"],
    }


class TestApprovalRequestItem:
    """Tests for ApprovalRequestItem dataclass."""

    def test_default_timestamps(self):
        """Test that timestamps are set by default."""
        item = ApprovalRequestItem(
            request_id="test-1",
            workflow_id="wf-1",
            step_id="step-1",
            title="Test request",
        )
        assert item.created_at != ""
        assert item.updated_at != ""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        item = ApprovalRequestItem(
            request_id="test-1",
            workflow_id="wf-1",
            step_id="step-1",
            title="Test request",
            status="approved",
            request_data={"key": "value"},
        )
        d = item.to_dict()
        assert d["request_id"] == "test-1"
        assert d["workflow_id"] == "wf-1"
        assert d["status"] == "approved"
        assert d["request_data"] == {"key": "value"}

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "request_id": "test-1",
            "workflow_id": "wf-1",
            "step_id": "step-1",
            "title": "Test request",
            "status": "rejected",
            "response_data": {"reason": "Denied"},
        }
        item = ApprovalRequestItem.from_dict(data)
        assert item.request_id == "test-1"
        assert item.status == "rejected"
        assert item.response_data == {"reason": "Denied"}

    def test_json_roundtrip(self):
        """Test JSON serialization roundtrip."""
        item = ApprovalRequestItem(
            request_id="test-1",
            workflow_id="wf-1",
            step_id="step-1",
            title="Test request",
            request_data={"nested": {"data": 123}},
        )
        json_str = item.to_json()
        restored = ApprovalRequestItem.from_json(json_str)
        assert restored.request_id == item.request_id
        assert restored.request_data == item.request_data


class TestInMemoryApprovalRequestStore:
    """Tests for InMemoryApprovalRequestStore."""

    @pytest.fixture
    def store(self):
        """Create a fresh in-memory store."""
        return InMemoryApprovalRequestStore()

    @pytest.mark.asyncio
    async def test_save_and_get(self, store, sample_request_data):
        """Test saving and retrieving a request."""
        await store.save(sample_request_data)
        result = await store.get(sample_request_data["request_id"])
        assert result is not None
        assert result["request_id"] == sample_request_data["request_id"]
        assert result["title"] == sample_request_data["title"]

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        """Test getting a non-existent request."""
        result = await store.get("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_requires_request_id(self, store):
        """Test that save requires request_id."""
        with pytest.raises(ValueError, match="request_id is required"):
            await store.save({"workflow_id": "test"})

    @pytest.mark.asyncio
    async def test_delete(self, store, sample_request_data):
        """Test deleting a request."""
        await store.save(sample_request_data)
        deleted = await store.delete(sample_request_data["request_id"])
        assert deleted is True
        result = await store.get(sample_request_data["request_id"])
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        """Test deleting a non-existent request."""
        deleted = await store.delete("nonexistent-id")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_list_all(self, store, sample_request_data, sample_request_data_2):
        """Test listing all requests."""
        await store.save(sample_request_data)
        await store.save(sample_request_data_2)
        all_requests = await store.list_all()
        assert len(all_requests) == 2

    @pytest.mark.asyncio
    async def test_list_by_status(self, store, sample_request_data, sample_request_data_2):
        """Test listing requests by status."""
        await store.save(sample_request_data)  # pending
        sample_request_data_2["status"] = "approved"
        await store.save(sample_request_data_2)
        pending = await store.list_by_status("pending")
        assert len(pending) == 1
        assert pending[0]["request_id"] == sample_request_data["request_id"]

    @pytest.mark.asyncio
    async def test_list_by_workflow(self, store, sample_request_data, sample_request_data_2):
        """Test listing requests by workflow."""
        await store.save(sample_request_data)
        await store.save(sample_request_data_2)
        workflow_requests = await store.list_by_workflow("workflow-123")
        assert len(workflow_requests) == 1
        assert workflow_requests[0]["request_id"] == sample_request_data["request_id"]

    @pytest.mark.asyncio
    async def test_list_pending(self, store, sample_request_data, sample_request_data_2):
        """Test listing pending requests."""
        await store.save(sample_request_data)
        sample_request_data_2["status"] = "approved"
        await store.save(sample_request_data_2)
        pending = await store.list_pending()
        assert len(pending) == 1
        assert pending[0]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_list_expired(self, store, sample_request_data):
        """Test listing expired requests."""
        # Set expires_at in the past
        past_time = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        sample_request_data["expires_at"] = past_time
        await store.save(sample_request_data)

        expired = await store.list_expired()
        assert len(expired) == 1
        assert expired[0]["request_id"] == sample_request_data["request_id"]

    @pytest.mark.asyncio
    async def test_list_expired_excludes_responded(self, store, sample_request_data):
        """Test that expired list excludes already-responded requests."""
        past_time = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        sample_request_data["expires_at"] = past_time
        sample_request_data["status"] = "approved"  # Already responded
        await store.save(sample_request_data)

        expired = await store.list_expired()
        assert len(expired) == 0

    @pytest.mark.asyncio
    async def test_respond(self, store, sample_request_data):
        """Test responding to a request."""
        await store.save(sample_request_data)
        response_data = {"comment": "Looks good"}
        responded = await store.respond(
            sample_request_data["request_id"],
            "approved",
            "reviewer-123",
            response_data=response_data,
        )
        assert responded is True
        result = await store.get(sample_request_data["request_id"])
        assert result["status"] == "approved"
        assert result["responder_id"] == "reviewer-123"
        assert result["responded_at"] is not None
        assert result["response_data"] == response_data

    @pytest.mark.asyncio
    async def test_respond_reject(self, store, sample_request_data):
        """Test rejecting a request."""
        await store.save(sample_request_data)
        responded = await store.respond(
            sample_request_data["request_id"],
            "rejected",
            "reviewer-456",
            response_data={"reason": "Not ready for production"},
        )
        assert responded is True
        result = await store.get(sample_request_data["request_id"])
        assert result["status"] == "rejected"

    @pytest.mark.asyncio
    async def test_respond_nonexistent(self, store):
        """Test responding to non-existent request."""
        responded = await store.respond("nonexistent-id", "approved", "user-1")
        assert responded is False


class TestSQLiteApprovalRequestStore:
    """Tests for SQLiteApprovalRequestStore."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a SQLite store with temp database."""
        db_path = tmp_path / "test_approval_requests.db"
        return SQLiteApprovalRequestStore(db_path=db_path)

    @pytest.mark.asyncio
    async def test_save_and_get(self, store, sample_request_data):
        """Test saving and retrieving a request."""
        await store.save(sample_request_data)
        result = await store.get(sample_request_data["request_id"])
        assert result is not None
        assert result["request_id"] == sample_request_data["request_id"]

    @pytest.mark.asyncio
    async def test_persistence(self, tmp_path, sample_request_data):
        """Test that data persists across store instances."""
        db_path = tmp_path / "persistence_test.db"

        # Save with first instance
        store1 = SQLiteApprovalRequestStore(db_path=db_path)
        await store1.save(sample_request_data)

        # Retrieve with second instance
        store2 = SQLiteApprovalRequestStore(db_path=db_path)
        result = await store2.get(sample_request_data["request_id"])
        assert result is not None
        assert result["request_id"] == sample_request_data["request_id"]

    @pytest.mark.asyncio
    async def test_list_pending_ordered_by_priority(
        self, store, sample_request_data, sample_request_data_2
    ):
        """Test that list_pending returns results ordered by priority."""
        await store.save(sample_request_data)  # priority 2
        await store.save(sample_request_data_2)  # priority 1
        pending = await store.list_pending()
        assert len(pending) == 2
        # Priority 1 (higher priority) should be first
        assert pending[0]["priority"] == 1

    @pytest.mark.asyncio
    async def test_respond_full_cycle(self, store, sample_request_data):
        """Test full respond cycle with SQLite persistence."""
        await store.save(sample_request_data)

        # Respond
        await store.respond(
            sample_request_data["request_id"],
            "approved",
            "reviewer-999",
            response_data={"approval_note": "Ship it!"},
        )

        # Verify
        result = await store.get(sample_request_data["request_id"])
        assert result["status"] == "approved"
        assert result["responder_id"] == "reviewer-999"
        assert result["response_data"]["approval_note"] == "Ship it!"


class TestGlobalStoreAccessor:
    """Tests for global store accessor functions."""

    def setup_method(self):
        """Reset store before each test."""
        reset_approval_request_store()

    def teardown_method(self):
        """Reset store after each test."""
        reset_approval_request_store()

    def test_get_default_store(self, monkeypatch, tmp_path):
        """Test getting default store uses SQLite."""
        monkeypatch.setenv("ARAGORA_DATA_DIR", str(tmp_path))
        store = get_approval_request_store()
        assert isinstance(store, SQLiteApprovalRequestStore)

    def test_get_memory_store(self, monkeypatch):
        """Test getting memory store via env var."""
        monkeypatch.setenv("ARAGORA_APPROVAL_STORE_BACKEND", "memory")
        store = get_approval_request_store()
        assert isinstance(store, InMemoryApprovalRequestStore)

    def test_singleton_behavior(self, monkeypatch):
        """Test that get_approval_request_store returns singleton."""
        monkeypatch.setenv("ARAGORA_APPROVAL_STORE_BACKEND", "memory")
        store1 = get_approval_request_store()
        store2 = get_approval_request_store()
        assert store1 is store2
