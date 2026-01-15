"""
Tests for batch debate operations handler mixin.

Tests:
- BatchOperationsMixin class structure
- Error handling for invalid input
- Module exports
"""

import pytest
from unittest.mock import MagicMock, patch

from aragora.server.handlers.debates.batch import BatchOperationsMixin


class TestBatchOperationsMixinStructure:
    """Tests for BatchOperationsMixin class structure."""

    def test_mixin_has_submit_batch(self):
        """Mixin should have _submit_batch method."""
        assert hasattr(BatchOperationsMixin, "_submit_batch")

    def test_mixin_has_get_batch_status(self):
        """Mixin should have _get_batch_status method."""
        assert hasattr(BatchOperationsMixin, "_get_batch_status")

    def test_mixin_has_list_batches(self):
        """Mixin should have _list_batches method."""
        assert hasattr(BatchOperationsMixin, "_list_batches")

    def test_mixin_has_get_queue_status(self):
        """Mixin should have _get_queue_status method."""
        assert hasattr(BatchOperationsMixin, "_get_queue_status")

    def test_mixin_has_create_debate_executor(self):
        """Mixin should have _create_debate_executor method."""
        assert hasattr(BatchOperationsMixin, "_create_debate_executor")


class TestModuleExports:
    """Tests for module exports."""

    def test_exports_batch_operations_mixin(self):
        """Should export BatchOperationsMixin."""
        from aragora.server.handlers.debates import batch
        assert "BatchOperationsMixin" in batch.__all__


class TestBatchStatusValidation:
    """Tests for batch status endpoint validation."""

    def test_get_batch_status_validates_id(self):
        """_get_batch_status should validate batch ID."""
        # Create a mock class that uses the mixin
        class MockHandler(BatchOperationsMixin):
            pass

        handler = MockHandler()

        # Test with invalid characters
        result = handler._get_batch_status("../../../etc/passwd")
        # Should return 400 for invalid ID
        assert result.status_code == 400

    def test_get_batch_status_sanitizes_input(self):
        """_get_batch_status should sanitize input."""
        class MockHandler(BatchOperationsMixin):
            pass

        handler = MockHandler()

        # Test with SQL injection attempt
        result = handler._get_batch_status("'; DROP TABLE batches;--")
        assert result.status_code == 400


class TestListBatchesValidation:
    """Tests for list batches endpoint."""

    def test_list_batches_handles_no_queue(self):
        """_list_batches should handle no queue gracefully."""
        class MockHandler(BatchOperationsMixin):
            pass

        handler = MockHandler()

        # Mock get_debate_queue_sync at the import location inside the method
        with patch('aragora.server.debate_queue.get_debate_queue_sync', return_value=None):
            result = handler._list_batches(limit=10)
            import json
            body = json.loads(result.body.decode('utf-8'))
            assert body["batches"] == []
            assert body["count"] == 0

    def test_list_batches_validates_status_filter(self):
        """_list_batches should validate status filter."""
        class MockHandler(BatchOperationsMixin):
            pass

        handler = MockHandler()

        # Mock get_debate_queue_sync
        with patch('aragora.server.debate_queue.get_debate_queue_sync') as mock:
            mock.return_value = MagicMock()
            # Invalid status should return error
            result = handler._list_batches(limit=10, status_filter="invalid_status")
            assert result.status_code == 400


class TestQueueStatus:
    """Tests for queue status endpoint."""

    def test_get_queue_status_no_queue(self):
        """_get_queue_status should handle no queue."""
        class MockHandler(BatchOperationsMixin):
            pass

        handler = MockHandler()

        with patch('aragora.server.debate_queue.get_debate_queue_sync', return_value=None):
            result = handler._get_queue_status()
            import json
            body = json.loads(result.body.decode('utf-8'))
            assert body["active"] is False
            assert "Queue not initialized" in body["message"]

    def test_get_queue_status_with_queue(self):
        """_get_queue_status should return queue stats."""
        class MockHandler(BatchOperationsMixin):
            pass

        handler = MockHandler()

        # Create mock queue
        mock_queue = MagicMock()
        mock_queue.max_concurrent = 5
        mock_queue._active_count = 2
        mock_queue.list_batches.return_value = [
            {"status": "pending"},
            {"status": "processing"},
            {"status": "completed"},
        ]

        with patch('aragora.server.debate_queue.get_debate_queue_sync', return_value=mock_queue):
            result = handler._get_queue_status()
            import json
            body = json.loads(result.body.decode('utf-8'))
            assert body["active"] is True
            assert body["max_concurrent"] == 5
            assert body["active_count"] == 2
            assert body["total_batches"] == 3


class TestCreateDebateExecutor:
    """Tests for _create_debate_executor method."""

    def test_returns_callable(self):
        """_create_debate_executor should return a callable."""
        class MockHandler(BatchOperationsMixin):
            pass

        handler = MockHandler()
        executor = handler._create_debate_executor()
        assert callable(executor)
