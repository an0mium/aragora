"""Tests for workflow outcomes → supermemory integration (B3).

Verifies that:
1. WORKFLOW_COMPLETE events store success outcomes in supermemory
2. WORKFLOW_FAILED events store failure outcomes with error details
3. Cross-workflow learning metadata is properly tagged
4. Handlers are graceful when supermemory unavailable
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from aragora.events.types import StreamEvent, StreamEventType


def _make_event(event_type: str, data: dict) -> StreamEvent:
    """Create a StreamEvent with the given data."""
    return StreamEvent(
        type=StreamEventType(event_type),
        data=data,
    )


def _get_handler():
    """Get the workflow outcome handler from BasicHandlersMixin."""
    from aragora.events.cross_subscribers.handlers.basic import BasicHandlersMixin

    mixin = BasicHandlersMixin.__new__(BasicHandlersMixin)
    return mixin._handle_workflow_outcome_to_supermemory


class TestWorkflowOutcomeToSupermemory:
    """Test _handle_workflow_outcome_to_supermemory handler."""

    def test_skips_empty_workflow_id(self):
        handler = _get_handler()
        event = _make_event(
            "workflow_complete",
            {"workflow_id": "", "success": True},
        )
        # Should not raise
        handler(event)

    def test_stores_successful_outcome(self):
        handler = _get_handler()
        mock_sm = MagicMock()

        with patch(
            "aragora.memory.supermemory.get_supermemory",
            return_value=mock_sm,
            create=True,
        ):
            event = _make_event(
                "workflow_complete",
                {
                    "workflow_id": "wf-123",
                    "definition_id": "vendor_review",
                    "success": True,
                    "duration_ms": 5000,
                    "steps_executed": 4,
                },
            )
            handler(event)
            mock_sm.store.assert_called_once()
            call_kwargs = mock_sm.store.call_args
            assert "completed successfully" in call_kwargs.kwargs.get(
                "content", call_kwargs[1].get("content", "")
            ) or "completed successfully" in str(call_kwargs)

    def test_stores_failed_outcome_with_error(self):
        handler = _get_handler()
        mock_sm = MagicMock()

        with patch(
            "aragora.memory.supermemory.get_supermemory",
            return_value=mock_sm,
            create=True,
        ):
            event = _make_event(
                "workflow_failed",
                {
                    "workflow_id": "wf-456",
                    "definition_id": "cost_audit",
                    "success": False,
                    "duration_ms": 1500,
                    "steps_executed": 2,
                    "error": "API timeout in step 3",
                },
            )
            handler(event)
            mock_sm.store.assert_called_once()
            call_args = mock_sm.store.call_args
            # Verify error is included in content
            content = call_args.kwargs.get("content", "") or call_args[1].get("content", "")
            assert "failed" in content
            assert "API timeout" in content

    def test_metadata_includes_workflow_details(self):
        handler = _get_handler()
        mock_sm = MagicMock()

        with patch(
            "aragora.memory.supermemory.get_supermemory",
            return_value=mock_sm,
            create=True,
        ):
            event = _make_event(
                "workflow_complete",
                {
                    "workflow_id": "wf-789",
                    "definition_id": "security_scan",
                    "success": True,
                    "duration_ms": 3000,
                    "steps_executed": 5,
                },
            )
            handler(event)
            call_args = mock_sm.store.call_args
            metadata = call_args.kwargs.get("metadata", {}) or call_args[1].get("metadata", {})
            assert metadata["workflow_id"] == "wf-789"
            assert metadata["definition_id"] == "security_scan"
            assert metadata["success"] is True
            assert metadata["steps_executed"] == 5

    def test_tags_include_workflow_outcome(self):
        handler = _get_handler()
        mock_sm = MagicMock()

        with patch(
            "aragora.memory.supermemory.get_supermemory",
            return_value=mock_sm,
            create=True,
        ):
            event = _make_event(
                "workflow_complete",
                {
                    "workflow_id": "wf-abc",
                    "definition_id": "budget_review",
                    "success": True,
                },
            )
            handler(event)
            call_args = mock_sm.store.call_args
            tags = call_args.kwargs.get("tags", []) or call_args[1].get("tags", [])
            assert "workflow_outcome" in tags
            assert "workflow:budget_review" in tags

    def test_source_tagged_with_workflow_id(self):
        handler = _get_handler()
        mock_sm = MagicMock()

        with patch(
            "aragora.memory.supermemory.get_supermemory",
            return_value=mock_sm,
            create=True,
        ):
            event = _make_event(
                "workflow_complete",
                {
                    "workflow_id": "wf-def",
                    "definition_id": "test",
                    "success": True,
                },
            )
            handler(event)
            call_args = mock_sm.store.call_args
            source = call_args.kwargs.get("source", "") or call_args[1].get("source", "")
            assert source == "workflow:wf-def"

    def test_graceful_when_supermemory_unavailable(self):
        handler = _get_handler()
        with patch(
            "aragora.memory.supermemory.get_supermemory",
            return_value=None,
            create=True,
        ):
            event = _make_event(
                "workflow_complete",
                {"workflow_id": "wf-ghi", "success": True},
            )
            # Should not raise
            handler(event)

    def test_graceful_on_import_error(self):
        handler = _get_handler()
        with patch.dict("sys.modules", {"aragora.memory.supermemory": None}):
            event = _make_event(
                "workflow_complete",
                {"workflow_id": "wf-jkl", "success": True},
            )
            handler(event)

    def test_graceful_on_store_exception(self):
        handler = _get_handler()
        mock_sm = MagicMock()
        mock_sm.store.side_effect = RuntimeError("test error")

        with patch(
            "aragora.memory.supermemory.get_supermemory",
            return_value=mock_sm,
            create=True,
        ):
            event = _make_event(
                "workflow_complete",
                {"workflow_id": "wf-mno", "success": True},
            )
            # Should handle exception gracefully
            handler(event)


class TestCrossSubscriberRegistration:
    """Test that workflow handlers are registered in CrossSubscriberManager."""

    def test_workflow_complete_handler_registered(self):
        from aragora.events.cross_subscribers.manager import CrossSubscriberManager

        manager = CrossSubscriberManager()
        subs = manager._subscribers.get(StreamEventType.WORKFLOW_COMPLETE, [])
        names = [name for name, _ in subs]
        assert "workflow_complete_to_supermemory" in names

    def test_workflow_failed_handler_registered(self):
        from aragora.events.cross_subscribers.manager import CrossSubscriberManager

        manager = CrossSubscriberManager()
        subs = manager._subscribers.get(StreamEventType.WORKFLOW_FAILED, [])
        names = [name for name, _ in subs]
        assert "workflow_failed_to_supermemory" in names
