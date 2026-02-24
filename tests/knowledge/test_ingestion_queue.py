"""Tests for Knowledge Mound ingestion dead letter queue (Sprint 16A)."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.knowledge.mound.ingestion_queue import IngestionDeadLetterQueue


@pytest.fixture
def dlq(tmp_path):
    """Create a DLQ with a temporary database."""
    db_path = str(tmp_path / "test_dlq.db")
    return IngestionDeadLetterQueue(db_path=db_path)


class TestIngestionDeadLetterQueue:
    """Tests for IngestionDeadLetterQueue."""

    def test_enqueue_stores_failed_item(self, dlq):
        """Enqueue stores a failed ingestion item."""
        dlq.enqueue("debate-1", {"task": "test"}, "connection refused")

        items = dlq.list_failed()
        assert len(items) == 1
        assert items[0]["debate_id"] == "debate-1"
        assert "connection refused" in items[0]["error"]

    def test_list_failed_returns_recent_items(self, dlq):
        """list_failed returns items in reverse chronological order."""
        dlq.enqueue("debate-1", {"task": "first"}, "error 1")
        dlq.enqueue("debate-2", {"task": "second"}, "error 2")

        items = dlq.list_failed(limit=10)
        assert len(items) == 2
        # Most recent first
        assert items[0]["debate_id"] == "debate-2"
        assert items[1]["debate_id"] == "debate-1"

    def test_list_failed_respects_limit(self, dlq):
        """list_failed returns at most `limit` items."""
        for i in range(5):
            dlq.enqueue(f"debate-{i}", {"n": i}, f"error {i}")

        items = dlq.list_failed(limit=3)
        assert len(items) == 3

    def test_process_queue_retries_and_removes_successes(self, dlq):
        """process_queue retries items and removes successful ones."""
        dlq.enqueue("debate-1", {"task": "ok"}, "temp error")
        dlq.enqueue("debate-2", {"task": "also ok"}, "temp error")

        mock_fn = MagicMock()
        successes = dlq.process_queue(ingest_fn=mock_fn)

        assert successes == 2
        assert mock_fn.call_count == 2
        assert dlq.list_failed() == []

    def test_process_queue_keeps_failures(self, dlq):
        """process_queue keeps items that still fail on retry."""
        dlq.enqueue("debate-1", {"task": "broken"}, "permanent error")

        mock_fn = MagicMock(side_effect=RuntimeError("still broken"))
        successes = dlq.process_queue(ingest_fn=mock_fn)

        assert successes == 0
        items = dlq.list_failed()
        assert len(items) == 1
        assert items[0]["retry_count"] == 1

    def test_process_queue_noop_without_fn(self, dlq):
        """process_queue returns 0 when no ingest_fn provided."""
        dlq.enqueue("debate-1", {"task": "test"}, "error")
        assert dlq.process_queue() == 0

    def test_clear_removes_all_items(self, dlq):
        """clear removes all items from the queue."""
        for i in range(3):
            dlq.enqueue(f"debate-{i}", {}, f"error {i}")

        removed = dlq.clear()
        assert removed == 3
        assert dlq.list_failed() == []


class TestOrchestratorRetry:
    """Test that orchestrator_runner retries ingestion with backoff."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_third_attempt(self):
        """Ingestion succeeds after retrying on transient errors."""
        from aragora.debate.orchestrator_runner import handle_debate_completion

        arena = MagicMock()
        arena._trackers = MagicMock()
        arena.extensions = MagicMock()
        arena.agents = []
        arena._budget_coordinator = MagicMock()
        arena._queue_for_supabase_sync = MagicMock()

        call_count = 0

        async def _ingest_with_failures(result):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return True

        arena._ingest_debate_outcome = _ingest_with_failures

        state = MagicMock()
        state.debate_id = "test-retry"
        state.debate_status = "completed"
        state.gupp_bead_id = None
        state.gupp_hook_entries = []

        ctx = MagicMock()
        ctx.result = MagicMock()
        ctx.result.to_dict.return_value = {"task": "test"}
        ctx.debate_id = "test-retry"
        state.ctx = ctx

        # The KM ingestion runs via asyncio.create_task (background).
        # We need the background coroutine to actually execute, so we
        # capture the coro and run it after handle_debate_completion returns.
        captured_coros = []
        _orig_create_task = asyncio.create_task

        def _capture_create_task(coro, **kwargs):
            captured_coros.append(coro)
            # Return a mock task that supports add_done_callback
            mock_task = MagicMock()
            mock_task.cancelled.return_value = False
            mock_task.exception.return_value = None
            return mock_task

        with patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("asyncio.create_task", side_effect=_capture_create_task):
            await handle_debate_completion(arena, state)

        # Now run the captured background coroutine(s)
        for coro in captured_coros:
            await coro

        assert call_count == 3  # Succeeded on 3rd try
