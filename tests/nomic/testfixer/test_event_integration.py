"""
Tests for TestFixer → Nomic Loop event integration.

Verifies:
1. Orchestrator emits events at key points
2. TestFixer event type registration and naming conventions

``TestFixerHandlersMixin`` (formerly ``aragora.events.subscribers.testfixer_handlers``)
was a dead duplicate of the live cross-subscriber path and was deleted; its handler
tests went with it (P4a Batch E2c).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch
import pytest

from aragora.events.types import StreamEvent, StreamEventType


class TestOrchestratorEventEmission:
    """Tests for event emission from TestFixerOrchestrator."""

    @pytest.fixture
    def mock_event_emitter(self):
        """Create a mock event emitter that collects events."""
        events = []

        async def collect_event(event: StreamEvent):
            events.append(event)

        collect_event.events = events
        return collect_event

    @pytest.fixture
    def orchestrator_with_emitter(self, mock_event_emitter, tmp_path):
        """Create orchestrator with event emitter."""
        from aragora.nomic.testfixer.orchestrator import (
            TestFixerOrchestrator,
            FixLoopConfig,
        )

        return TestFixerOrchestrator(
            repo_path=tmp_path,
            test_command="pytest tests/ -q",
            config=FixLoopConfig(max_iterations=2),
            event_emitter=mock_event_emitter,
        )

    @pytest.mark.asyncio
    async def test_emit_event_with_async_emitter(
        self, orchestrator_with_emitter, mock_event_emitter
    ):
        """Test that _emit_event works with async emitter."""
        await orchestrator_with_emitter._emit_event(
            StreamEventType.TESTFIXER_FAILURE_DETECTED,
            {"test_name": "test_example", "error_type": "AssertionError"},
        )

        assert len(mock_event_emitter.events) == 1
        event = mock_event_emitter.events[0]
        assert event.type == StreamEventType.TESTFIXER_FAILURE_DETECTED
        assert event.data["test_name"] == "test_example"
        assert event.data["run_id"] == orchestrator_with_emitter.run_id

    @pytest.mark.asyncio
    async def test_emit_event_with_sync_emitter(self, tmp_path):
        """Test that _emit_event works with sync emitter."""
        from aragora.nomic.testfixer.orchestrator import (
            TestFixerOrchestrator,
            FixLoopConfig,
        )

        events = []

        def sync_emitter(event: StreamEvent):
            events.append(event)

        orchestrator = TestFixerOrchestrator(
            repo_path=tmp_path,
            test_command="pytest",
            config=FixLoopConfig(max_iterations=1),
            event_emitter=sync_emitter,
        )

        await orchestrator._emit_event(
            StreamEventType.TESTFIXER_LOOP_COMPLETE,
            {"status": "success"},
        )

        assert len(events) == 1
        assert events[0].type == StreamEventType.TESTFIXER_LOOP_COMPLETE

    @pytest.mark.asyncio
    async def test_emit_event_no_emitter(self, tmp_path):
        """Test that _emit_event handles no emitter gracefully."""
        from aragora.nomic.testfixer.orchestrator import (
            TestFixerOrchestrator,
            FixLoopConfig,
        )

        orchestrator = TestFixerOrchestrator(
            repo_path=tmp_path,
            test_command="pytest",
            config=FixLoopConfig(max_iterations=1),
            # No event_emitter
        )

        # Should not raise
        await orchestrator._emit_event(
            StreamEventType.TESTFIXER_FAILURE_DETECTED,
            {"test_name": "test_example"},
        )

    @pytest.mark.asyncio
    async def test_emit_event_handles_emitter_error(self, tmp_path):
        """Test that _emit_event handles emitter errors gracefully."""
        from aragora.nomic.testfixer.orchestrator import (
            TestFixerOrchestrator,
            FixLoopConfig,
        )

        def failing_emitter(event: StreamEvent):
            raise RuntimeError("Emitter failed")

        orchestrator = TestFixerOrchestrator(
            repo_path=tmp_path,
            test_command="pytest",
            config=FixLoopConfig(max_iterations=1),
            event_emitter=failing_emitter,
        )

        # Should not raise, just log
        await orchestrator._emit_event(
            StreamEventType.TESTFIXER_FAILURE_DETECTED,
            {"test_name": "test_example"},
        )


class TestEventTypes:
    """Tests for TestFixer event types."""

    def test_all_testfixer_event_types_exist(self):
        """Test that all expected testfixer event types are defined."""
        expected_types = [
            "TESTFIXER_FAILURE_DETECTED",
            "TESTFIXER_ANALYSIS_COMPLETE",
            "TESTFIXER_FIX_PROPOSED",
            "TESTFIXER_FIX_APPLIED",
            "TESTFIXER_FIX_REVERTED",
            "TESTFIXER_ITERATION_COMPLETE",
            "TESTFIXER_LOOP_COMPLETE",
            "TESTFIXER_PATTERN_LEARNED",
        ]

        for type_name in expected_types:
            assert hasattr(StreamEventType, type_name), f"Missing event type: {type_name}"

    def test_event_type_values_are_snake_case(self):
        """Test that event type values follow naming convention."""
        testfixer_types = [t for t in StreamEventType if t.name.startswith("TESTFIXER_")]

        for event_type in testfixer_types:
            assert event_type.value.startswith("testfixer_")
            assert event_type.value == event_type.value.lower()


class TestEndToEndFlow:
    """Tests for end-to-end event flow."""

    @pytest.mark.asyncio
    async def test_meta_planner_context_with_failures(self):
        """Test that PlanningContext accepts test failures."""
        from aragora.nomic.meta_planner import PlanningContext

        context = PlanningContext(
            test_failures=[
                "tests/test_auth.py::test_login (AssertionError): Expected success",
                "tests/test_api.py::test_get_user (TypeError): None is not callable",
            ],
        )

        assert len(context.test_failures) == 2
        assert "test_login" in context.test_failures[0]
        assert "test_get_user" in context.test_failures[1]
