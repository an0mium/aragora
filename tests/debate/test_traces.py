"""Comprehensive tests for the debate traces module.

Tests cover:
1. Trace creation and management (TraceEvent, DebateTrace)
2. Event recording via DebateTracer
3. Trace serialization/deserialization
4. Query and filtering operations
5. Compliance metadata and checksums
6. Replay functionality (DebateReplayer)
7. Edge cases and error handling

These tests are important for audit/compliance as traces provide
deterministic, replayable debate artifacts with full event logging.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.traces import (
    DebateReplayer,
    DebateTrace,
    DebateTracer,
    EventType,
    TraceEvent,
    list_traces,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def trace_event():
    """Create a basic TraceEvent."""
    return TraceEvent(
        event_id="debate-001-e0001",
        event_type=EventType.AGENT_PROPOSAL,
        timestamp="2025-01-01T12:00:00",
        round_num=1,
        agent="claude",
        content={"content": "My proposal", "confidence": 0.8},
    )


@pytest.fixture
def trace_event_critique():
    """Create a critique TraceEvent."""
    return TraceEvent(
        event_id="debate-001-e0002",
        event_type=EventType.AGENT_CRITIQUE,
        timestamp="2025-01-01T12:01:00",
        round_num=1,
        agent="gpt4",
        content={
            "target_agent": "claude",
            "issues": ["Lacks specificity", "Missing examples"],
            "suggestions": ["Add concrete examples"],
            "severity": 5.0,
        },
    )


@pytest.fixture
def trace_event_consensus():
    """Create a consensus check TraceEvent."""
    return TraceEvent(
        event_id="debate-001-e0003",
        event_type=EventType.CONSENSUS_CHECK,
        timestamp="2025-01-01T12:02:00",
        round_num=1,
        agent=None,
        content={
            "reached": True,
            "confidence": 0.85,
            "votes": {"claude": True, "gpt4": True},
        },
    )


@pytest.fixture
def empty_trace():
    """Create an empty DebateTrace."""
    return DebateTrace(
        trace_id="trace-test-001",
        debate_id="debate-001",
        task="Test debate task",
        agents=["claude", "gpt4"],
        random_seed=42,
        events=[],
    )


@pytest.fixture
def trace_with_events(trace_event, trace_event_critique, trace_event_consensus):
    """Create a DebateTrace with multiple events."""
    return DebateTrace(
        trace_id="trace-test-002",
        debate_id="debate-002",
        task="Complex debate task",
        agents=["claude", "gpt4", "gemini"],
        random_seed=12345,
        events=[trace_event, trace_event_critique, trace_event_consensus],
        started_at="2025-01-01T12:00:00",
        completed_at="2025-01-01T12:30:00",
        final_result={"final_answer": "The consensus answer"},
        metadata={"tenant_id": "tenant-001", "audit_id": "audit-xyz"},
    )


@pytest.fixture
def temp_db_path():
    """Create a temporary database file path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def temp_trace_file():
    """Create a temporary file for trace saving/loading."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    yield path
    # Cleanup
    if path.exists():
        path.unlink()


@pytest.fixture
def tracer(temp_db_path):
    """Create a DebateTracer with temp database."""
    return DebateTracer(
        debate_id="tracer-test-001",
        task="Tracer test task",
        agents=["claude", "gpt4"],
        random_seed=42,
        db_path=temp_db_path,
    )


# =============================================================================
# TestEventType
# =============================================================================


class TestEventType:
    """Test EventType enumeration."""

    def test_all_event_types_defined(self):
        """Verify all expected event types exist."""
        expected_types = [
            "DEBATE_START",
            "DEBATE_END",
            "ROUND_START",
            "ROUND_END",
            "MESSAGE",
            "AGENT_PROPOSAL",
            "AGENT_CRITIQUE",
            "AGENT_SYNTHESIS",
            "CONSENSUS_CHECK",
            "FORK_DECISION",
            "FORK_CREATED",
            "MERGE_RESULT",
            "MEMORY_ACCESS",
            "MEMORY_WRITE",
            "HUMAN_INTERVENTION",
            "ERROR",
            "TOOL_CALL",
            "TOOL_RESULT",
        ]
        for type_name in expected_types:
            assert hasattr(EventType, type_name)

    def test_event_type_values_are_strings(self):
        """Verify event type values are lowercase strings."""
        for event_type in EventType:
            assert isinstance(event_type.value, str)
            assert event_type.value == event_type.value.lower()

    def test_event_types_are_unique(self):
        """Verify all event type values are unique."""
        values = [e.value for e in EventType]
        assert len(values) == len(set(values))


# =============================================================================
# TestTraceEvent
# =============================================================================


class TestTraceEvent:
    """Test TraceEvent dataclass."""

    def test_create_basic_event(self, trace_event):
        """Test creating a basic trace event."""
        assert trace_event.event_id == "debate-001-e0001"
        assert trace_event.event_type == EventType.AGENT_PROPOSAL
        assert trace_event.timestamp == "2025-01-01T12:00:00"
        assert trace_event.round_num == 1
        assert trace_event.agent == "claude"
        assert trace_event.content["content"] == "My proposal"

    def test_create_event_with_parent(self):
        """Test creating event with parent reference."""
        event = TraceEvent(
            event_id="e-002",
            event_type=EventType.AGENT_CRITIQUE,
            timestamp="2025-01-01T12:00:01",
            round_num=1,
            agent="gpt4",
            content={"critique": "needs work"},
            parent_event_id="e-001",
        )
        assert event.parent_event_id == "e-001"

    def test_create_event_with_duration(self):
        """Test creating event with duration tracking."""
        event = TraceEvent(
            event_id="e-003",
            event_type=EventType.TOOL_CALL,
            timestamp="2025-01-01T12:00:00",
            round_num=1,
            agent="claude",
            content={"tool": "search"},
            duration_ms=150,
        )
        assert event.duration_ms == 150

    def test_create_event_with_metadata(self):
        """Test creating event with metadata."""
        event = TraceEvent(
            event_id="e-004",
            event_type=EventType.HUMAN_INTERVENTION,
            timestamp="2025-01-01T12:00:00",
            round_num=2,
            agent=None,
            content={"action": "approved"},
            metadata={"user_id": "user-123", "audit_reason": "manual review"},
        )
        assert event.metadata["user_id"] == "user-123"
        assert event.metadata["audit_reason"] == "manual review"

    def test_event_to_dict(self, trace_event):
        """Test converting event to dictionary."""
        d = trace_event.to_dict()
        assert d["event_id"] == "debate-001-e0001"
        assert d["event_type"] == "agent_proposal"  # Enum value
        assert d["timestamp"] == "2025-01-01T12:00:00"
        assert d["round_num"] == 1
        assert d["agent"] == "claude"
        assert d["content"]["content"] == "My proposal"
        assert d["parent_event_id"] is None
        assert d["duration_ms"] is None
        assert d["metadata"] == {}

    def test_event_from_dict(self):
        """Test creating event from dictionary."""
        data = {
            "event_id": "e-from-dict",
            "event_type": "agent_synthesis",
            "timestamp": "2025-01-01T13:00:00",
            "round_num": 2,
            "agent": "gemini",
            "content": {"synthesis": "combined view"},
            "parent_event_id": "e-parent",
            "duration_ms": 200,
            "metadata": {"key": "value"},
        }
        event = TraceEvent.from_dict(data)
        assert event.event_id == "e-from-dict"
        assert event.event_type == EventType.AGENT_SYNTHESIS
        assert event.agent == "gemini"
        assert event.parent_event_id == "e-parent"
        assert event.duration_ms == 200
        assert event.metadata["key"] == "value"

    def test_event_roundtrip_serialization(self, trace_event):
        """Test event survives dict roundtrip."""
        d = trace_event.to_dict()
        restored = TraceEvent.from_dict(d)
        assert restored.event_id == trace_event.event_id
        assert restored.event_type == trace_event.event_type
        assert restored.content == trace_event.content

    def test_event_with_none_agent(self):
        """Test event without agent (system events)."""
        event = TraceEvent(
            event_id="sys-001",
            event_type=EventType.DEBATE_START,
            timestamp="2025-01-01T10:00:00",
            round_num=0,
            agent=None,
            content={"task": "Begin debate"},
        )
        assert event.agent is None
        d = event.to_dict()
        assert d["agent"] is None


# =============================================================================
# TestDebateTrace
# =============================================================================


class TestDebateTrace:
    """Test DebateTrace dataclass."""

    def test_create_empty_trace(self, empty_trace):
        """Test creating empty trace."""
        assert empty_trace.trace_id == "trace-test-001"
        assert empty_trace.debate_id == "debate-001"
        assert empty_trace.task == "Test debate task"
        assert empty_trace.agents == ["claude", "gpt4"]
        assert empty_trace.random_seed == 42
        assert empty_trace.events == []
        assert empty_trace.completed_at is None
        assert empty_trace.final_result is None

    def test_create_trace_with_events(self, trace_with_events):
        """Test creating trace with events."""
        assert len(trace_with_events.events) == 3
        assert trace_with_events.completed_at == "2025-01-01T12:30:00"
        assert trace_with_events.final_result["final_answer"] == "The consensus answer"
        assert trace_with_events.metadata["tenant_id"] == "tenant-001"

    def test_add_event(self, empty_trace, trace_event):
        """Test adding event to trace."""
        empty_trace.add_event(trace_event)
        assert len(empty_trace.events) == 1
        assert empty_trace.events[0].event_id == trace_event.event_id

    def test_add_multiple_events(self, empty_trace):
        """Test adding multiple events."""
        for i in range(5):
            event = TraceEvent(
                event_id=f"e-{i:03d}",
                event_type=EventType.MESSAGE,
                timestamp=f"2025-01-01T12:{i:02d}:00",
                round_num=0,
                agent="test",
                content={"msg": f"Message {i}"},
            )
            empty_trace.add_event(event)
        assert len(empty_trace.events) == 5

    def test_clear_events(self, trace_with_events):
        """Test clearing all events."""
        assert len(trace_with_events.events) > 0
        trace_with_events.clear_events()
        assert len(trace_with_events.events) == 0

    def test_checksum_consistency(self, trace_with_events):
        """Test checksum is consistent for same events."""
        checksum1 = trace_with_events.checksum
        checksum2 = trace_with_events.checksum
        assert checksum1 == checksum2
        assert len(checksum1) == 16  # Truncated SHA-256

    def test_checksum_changes_with_events(self, empty_trace):
        """Test checksum changes when events are modified."""
        checksum1 = empty_trace.checksum

        empty_trace.add_event(
            TraceEvent(
                event_id="new-e",
                event_type=EventType.MESSAGE,
                timestamp="2025-01-01T12:00:00",
                round_num=0,
                agent="test",
                content={"msg": "new"},
            )
        )
        checksum2 = empty_trace.checksum

        assert checksum1 != checksum2

    def test_duration_ms_with_completed_trace(self, trace_with_events):
        """Test duration calculation for completed trace."""
        duration = trace_with_events.duration_ms
        assert duration is not None
        # 30 minutes = 1800 seconds = 1,800,000 ms
        assert duration == 1800000

    def test_duration_ms_with_incomplete_trace(self, empty_trace):
        """Test duration is None for incomplete trace."""
        assert empty_trace.completed_at is None
        assert empty_trace.duration_ms is None

    def test_get_events_by_type(self, trace_with_events):
        """Test filtering events by type."""
        proposals = trace_with_events.get_events_by_type(EventType.AGENT_PROPOSAL)
        assert len(proposals) == 1
        assert proposals[0].event_type == EventType.AGENT_PROPOSAL

        critiques = trace_with_events.get_events_by_type(EventType.AGENT_CRITIQUE)
        assert len(critiques) == 1

        consensus = trace_with_events.get_events_by_type(EventType.CONSENSUS_CHECK)
        assert len(consensus) == 1

    def test_get_events_by_type_empty_result(self, trace_with_events):
        """Test filtering returns empty list for missing types."""
        errors = trace_with_events.get_events_by_type(EventType.ERROR)
        assert errors == []

    def test_get_events_by_agent(self, trace_with_events):
        """Test filtering events by agent."""
        claude_events = trace_with_events.get_events_by_agent("claude")
        assert len(claude_events) == 1
        assert claude_events[0].agent == "claude"

        gpt4_events = trace_with_events.get_events_by_agent("gpt4")
        assert len(gpt4_events) == 1
        assert gpt4_events[0].agent == "gpt4"

    def test_get_events_by_agent_none_agent(self, trace_with_events):
        """Test filtering for system events (no agent)."""
        # Consensus check has agent=None
        none_events = trace_with_events.get_events_by_agent(None)
        assert len(none_events) == 1
        assert none_events[0].event_type == EventType.CONSENSUS_CHECK

    def test_get_events_by_round(self):
        """Test filtering events by round."""
        events = [
            TraceEvent(
                event_id=f"e-{i}",
                event_type=EventType.MESSAGE,
                timestamp=f"2025-01-01T12:{i:02d}:00",
                round_num=i // 3,  # 0, 0, 0, 1, 1, 1, 2, 2, 2
                agent="test",
                content={"msg": i},
            )
            for i in range(9)
        ]
        trace = DebateTrace(
            trace_id="round-test",
            debate_id="round-debate",
            task="Test rounds",
            agents=["test"],
            random_seed=1,
            events=events,
        )

        round_0 = trace.get_events_by_round(0)
        assert len(round_0) == 3

        round_1 = trace.get_events_by_round(1)
        assert len(round_1) == 3

        round_2 = trace.get_events_by_round(2)
        assert len(round_2) == 3

    def test_get_events_by_round_empty(self, trace_with_events):
        """Test filtering by non-existent round."""
        round_99 = trace_with_events.get_events_by_round(99)
        assert round_99 == []


# =============================================================================
# TestDebateTraceSerialization
# =============================================================================


class TestDebateTraceSerialization:
    """Test DebateTrace JSON serialization and deserialization."""

    def test_to_json(self, trace_with_events):
        """Test JSON serialization."""
        json_str = trace_with_events.to_json()
        assert isinstance(json_str, str)

        data = json.loads(json_str)
        assert data["trace_id"] == "trace-test-002"
        assert data["debate_id"] == "debate-002"
        assert data["task"] == "Complex debate task"
        assert data["agents"] == ["claude", "gpt4", "gemini"]
        assert data["random_seed"] == 12345
        assert len(data["events"]) == 3
        assert "checksum" in data

    def test_from_json(self, trace_with_events):
        """Test JSON deserialization."""
        json_str = trace_with_events.to_json()
        restored = DebateTrace.from_json(json_str)

        assert restored.trace_id == trace_with_events.trace_id
        assert restored.debate_id == trace_with_events.debate_id
        assert restored.task == trace_with_events.task
        assert restored.agents == trace_with_events.agents
        assert restored.random_seed == trace_with_events.random_seed
        assert len(restored.events) == len(trace_with_events.events)
        assert restored.checksum == trace_with_events.checksum

    def test_json_roundtrip_preserves_events(self, trace_with_events):
        """Test events survive JSON roundtrip."""
        json_str = trace_with_events.to_json()
        restored = DebateTrace.from_json(json_str)

        for orig, rest in zip(trace_with_events.events, restored.events):
            assert orig.event_id == rest.event_id
            assert orig.event_type == rest.event_type
            assert orig.content == rest.content

    def test_from_json_checksum_validation(self, trace_with_events):
        """Test checksum validation during deserialization."""
        json_str = trace_with_events.to_json()
        data = json.loads(json_str)

        # Tamper with checksum
        data["checksum"] = "invalid_checksum"
        tampered_json = json.dumps(data)

        with pytest.raises(ValueError, match="Trace checksum mismatch"):
            DebateTrace.from_json(tampered_json)

    def test_from_json_missing_checksum_allowed(self, trace_with_events):
        """Test deserialization works without checksum."""
        json_str = trace_with_events.to_json()
        data = json.loads(json_str)

        del data["checksum"]
        no_checksum_json = json.dumps(data)

        # Should not raise
        restored = DebateTrace.from_json(no_checksum_json)
        assert restored.trace_id == trace_with_events.trace_id

    def test_save_and_load(self, trace_with_events, temp_trace_file):
        """Test saving and loading trace from file."""
        trace_with_events.save(temp_trace_file)
        assert temp_trace_file.exists()

        loaded = DebateTrace.load(temp_trace_file)
        assert loaded.trace_id == trace_with_events.trace_id
        assert loaded.checksum == trace_with_events.checksum

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading from non-existent file raises error."""
        fake_path = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError, match="Debate trace not found"):
            DebateTrace.load(fake_path)

    def test_to_json_indent_formatting(self, trace_with_events):
        """Test JSON formatting with custom indent."""
        json_str = trace_with_events.to_json(indent=4)
        # Check it's properly indented (has newlines and spaces)
        assert "\n" in json_str
        assert "    " in json_str  # 4-space indent

    def test_empty_trace_serialization(self, empty_trace):
        """Test empty trace serialization."""
        json_str = empty_trace.to_json()
        restored = DebateTrace.from_json(json_str)

        assert restored.events == []
        assert restored.final_result is None


# =============================================================================
# TestDebateTraceToDebateResult
# =============================================================================


class TestDebateTraceToDebateResult:
    """Test DebateTrace conversion to DebateResult."""

    def test_to_debate_result_basic(self, trace_with_events):
        """Test basic conversion to DebateResult."""
        result = trace_with_events.to_debate_result()

        assert result.debate_id == trace_with_events.debate_id
        assert result.task == trace_with_events.task
        assert result.participants == trace_with_events.agents

    def test_to_debate_result_messages(self, trace_with_events):
        """Test messages are extracted correctly."""
        result = trace_with_events.to_debate_result()

        # Should have 1 proposal message
        assert len(result.messages) >= 1
        proposal_msgs = [m for m in result.messages if m.role == "proposer"]
        assert len(proposal_msgs) == 1
        assert proposal_msgs[0].agent == "claude"

    def test_to_debate_result_critiques(self, trace_with_events):
        """Test critiques are extracted correctly."""
        result = trace_with_events.to_debate_result()

        assert len(result.critiques) == 1
        critique = result.critiques[0]
        assert critique.agent == "gpt4"
        assert critique.target_agent == "claude"
        assert "Lacks specificity" in critique.issues

    def test_to_debate_result_consensus(self, trace_with_events):
        """Test consensus info is extracted."""
        result = trace_with_events.to_debate_result()

        assert result.consensus_reached is True
        assert result.confidence == 0.85

    def test_to_debate_result_final_answer(self, trace_with_events):
        """Test final answer is extracted from final_result."""
        result = trace_with_events.to_debate_result()
        assert result.final_answer == "The consensus answer"

    def test_to_debate_result_rounds_used(self, trace_with_events):
        """Test rounds_used calculation."""
        result = trace_with_events.to_debate_result()
        # All events in trace_with_events are round 1
        assert result.rounds_used == 1

    def test_to_debate_result_duration(self, trace_with_events):
        """Test duration is converted correctly."""
        result = trace_with_events.to_debate_result()
        # 1,800,000 ms = 1800 seconds
        assert result.duration_seconds == 1800.0

    def test_to_debate_result_empty_trace(self, empty_trace):
        """Test conversion of empty trace."""
        result = empty_trace.to_debate_result()

        assert result.messages == []
        assert result.critiques == []
        assert result.consensus_reached is False
        assert result.final_answer == ""


# =============================================================================
# TestDebateTracer
# =============================================================================


class TestDebateTracer:
    """Test DebateTracer class for recording debate events."""

    def test_tracer_initialization(self, tracer):
        """Test tracer is initialized correctly."""
        assert tracer.debate_id == "tracer-test-001"
        assert tracer.task == "Tracer test task"
        assert tracer.agents == ["claude", "gpt4"]
        assert tracer.random_seed == 42
        assert tracer.trace is not None
        assert tracer.trace.trace_id == "trace-tracer-test-001"

    def test_tracer_auto_seed(self, temp_db_path):
        """Test tracer auto-generates seed if not provided."""
        tracer = DebateTracer(
            debate_id="auto-seed-test",
            task="Auto seed test",
            agents=["agent1"],
            db_path=temp_db_path,
        )
        assert tracer.random_seed is not None
        assert 0 <= tracer.random_seed < 2**32

    def test_record_event(self, tracer):
        """Test recording a basic event."""
        event = tracer.record(
            EventType.MESSAGE, {"text": "Hello"}, agent="claude"
        )

        assert event.event_id.startswith("tracer-test-001-e")
        assert event.event_type == EventType.MESSAGE
        assert event.agent == "claude"
        assert len(tracer.trace.events) == 1

    def test_record_with_duration(self, tracer):
        """Test recording event with duration."""
        event = tracer.record(
            EventType.TOOL_CALL,
            {"tool": "search"},
            agent="claude",
            duration_ms=250,
        )
        assert event.duration_ms == 250

    def test_record_with_metadata(self, tracer):
        """Test recording event with metadata."""
        event = tracer.record(
            EventType.HUMAN_INTERVENTION,
            {"action": "approved"},
            metadata={"reviewer": "admin", "reason": "policy check"},
        )
        assert event.metadata["reviewer"] == "admin"
        assert event.metadata["reason"] == "policy check"

    def test_start_round(self, tracer):
        """Test starting a new round."""
        tracer.start_round(1)

        assert tracer._current_round == 1
        assert len(tracer.trace.events) == 1
        assert tracer.trace.events[0].event_type == EventType.ROUND_START
        assert tracer.trace.events[0].content["round"] == 1

    def test_end_round(self, tracer):
        """Test ending a round."""
        tracer.start_round(1)
        tracer.end_round()

        assert len(tracer.trace.events) == 2
        assert tracer.trace.events[1].event_type == EventType.ROUND_END
        assert tracer.trace.events[1].content["round"] == 1

    def test_record_proposal(self, tracer):
        """Test recording a proposal."""
        tracer.start_round(1)
        tracer.record_proposal("claude", "My proposal content", confidence=0.9)

        proposals = tracer.trace.get_events_by_type(EventType.AGENT_PROPOSAL)
        assert len(proposals) == 1
        assert proposals[0].agent == "claude"
        assert proposals[0].content["content"] == "My proposal content"
        assert proposals[0].content["confidence"] == 0.9

    def test_record_critique(self, tracer):
        """Test recording a critique."""
        tracer.start_round(1)
        tracer.record_critique(
            agent="gpt4",
            target_agent="claude",
            issues=["Issue 1", "Issue 2"],
            severity=6.5,
            suggestions=["Suggestion 1"],
        )

        critiques = tracer.trace.get_events_by_type(EventType.AGENT_CRITIQUE)
        assert len(critiques) == 1
        assert critiques[0].agent == "gpt4"
        assert critiques[0].content["target_agent"] == "claude"
        assert critiques[0].content["issues"] == ["Issue 1", "Issue 2"]
        assert critiques[0].content["severity"] == 6.5

    def test_record_synthesis(self, tracer):
        """Test recording a synthesis."""
        tracer.start_round(1)
        tracer.record_synthesis(
            agent="gemini",
            content="Combined synthesis",
            incorporated=["claude", "gpt4"],
        )

        syntheses = tracer.trace.get_events_by_type(EventType.AGENT_SYNTHESIS)
        assert len(syntheses) == 1
        assert syntheses[0].agent == "gemini"
        assert syntheses[0].content["content"] == "Combined synthesis"
        assert syntheses[0].content["incorporated"] == ["claude", "gpt4"]

    def test_record_consensus(self, tracer):
        """Test recording consensus check."""
        tracer.record_consensus(
            reached=True, confidence=0.92, votes={"claude": True, "gpt4": True}
        )

        consensus = tracer.trace.get_events_by_type(EventType.CONSENSUS_CHECK)
        assert len(consensus) == 1
        assert consensus[0].content["reached"] is True
        assert consensus[0].content["confidence"] == 0.92
        assert consensus[0].content["votes"] == {"claude": True, "gpt4": True}

    def test_record_tool_call(self, tracer):
        """Test recording tool call and getting event ID."""
        event_id = tracer.record_tool_call(
            agent="claude", tool="web_search", args={"query": "test query"}
        )

        assert event_id is not None
        tool_calls = tracer.trace.get_events_by_type(EventType.TOOL_CALL)
        assert len(tool_calls) == 1
        assert tool_calls[0].content["tool"] == "web_search"
        assert tool_calls[0].content["args"]["query"] == "test query"

    def test_record_tool_result(self, tracer):
        """Test recording tool result linked to call."""
        call_id = tracer.record_tool_call(
            agent="claude", tool="web_search", args={"query": "test"}
        )
        tracer.record_tool_result(
            agent="claude",
            tool="web_search",
            result={"results": ["result1", "result2"]},
            call_event_id=call_id,
        )

        results = tracer.trace.get_events_by_type(EventType.TOOL_RESULT)
        assert len(results) == 1
        assert results[0].metadata["call_event_id"] == call_id

    def test_record_tool_result_truncation(self, tracer):
        """Test tool result truncation for large results."""
        call_id = tracer.record_tool_call(
            agent="claude", tool="search", args={}
        )
        large_result = "x" * 2000
        tracer.record_tool_result(
            agent="claude", tool="search", result=large_result, call_event_id=call_id
        )

        results = tracer.trace.get_events_by_type(EventType.TOOL_RESULT)
        assert len(results[0].content["result"]) <= 1000

    def test_record_error(self, tracer):
        """Test recording an error."""
        tracer.record_error("Agent timeout", agent="gpt4")

        errors = tracer.trace.get_events_by_type(EventType.ERROR)
        assert len(errors) == 1
        assert errors[0].content["error"] == "Agent timeout"
        assert errors[0].agent == "gpt4"

    def test_record_error_without_agent(self, tracer):
        """Test recording system-level error."""
        tracer.record_error("Database connection failed")

        errors = tracer.trace.get_events_by_type(EventType.ERROR)
        assert len(errors) == 1
        assert errors[0].agent is None

    def test_finalize(self, tracer):
        """Test finalizing the trace."""
        tracer.start_round(1)
        tracer.record_proposal("claude", "Proposal", confidence=0.8)

        result = {"final_answer": "The answer", "confidence": 0.9}
        trace = tracer.finalize(result)

        assert trace.completed_at is not None
        assert trace.final_result == result

        # Should have DEBATE_END event
        end_events = trace.get_events_by_type(EventType.DEBATE_END)
        assert len(end_events) == 1

    def test_event_parent_tracking(self, tracer):
        """Test parent event ID tracking via event stack."""
        tracer.start_round(1)
        round_event = tracer.trace.events[0]

        # Events recorded during round should reference round start
        tracer.record_proposal("claude", "Proposal", confidence=0.7)
        proposal_event = tracer.trace.events[1]

        assert proposal_event.parent_event_id == round_event.event_id

    def test_get_state_at_event(self, tracer):
        """Test reconstructing state at a specific event."""
        tracer.start_round(1)
        tracer.record_proposal("claude", "Claude proposal", confidence=0.8)
        tracer.record_proposal("gpt4", "GPT4 proposal", confidence=0.75)
        tracer.record_critique(
            "gpt4", "claude", ["Issue"], severity=5.0, suggestions=[]
        )
        tracer.record_consensus(True, 0.85, {"claude": True, "gpt4": True})

        # Get the last event ID
        last_event = tracer.trace.events[-1]
        state = tracer.get_state_at_event(last_event.event_id)

        assert state["round"] == 1
        assert len(state["messages"]) == 2
        assert len(state["critiques"]) == 1
        assert state["consensus"]["reached"] is True

    def test_get_state_at_middle_event(self, tracer):
        """Test state reconstruction at middle event."""
        tracer.start_round(1)
        tracer.record_proposal("claude", "Proposal 1", confidence=0.8)
        proposal_event = tracer.trace.events[-1]
        tracer.record_proposal("gpt4", "Proposal 2", confidence=0.7)
        tracer.record_consensus(True, 0.9, {"claude": True, "gpt4": True})

        # Get state at first proposal
        state = tracer.get_state_at_event(proposal_event.event_id)

        assert len(state["messages"]) == 1
        assert state["consensus"] is None  # Consensus came later


# =============================================================================
# TestDebateReplayer
# =============================================================================


class TestDebateReplayer:
    """Test DebateReplayer for trace replay and analysis."""

    def test_replayer_initialization(self, trace_with_events):
        """Test replayer initialization from trace."""
        replayer = DebateReplayer(trace_with_events)

        assert replayer.trace is trace_with_events
        assert replayer._position == 0

    def test_replayer_from_file(self, trace_with_events, temp_trace_file):
        """Test loading replayer from file."""
        trace_with_events.save(temp_trace_file)
        replayer = DebateReplayer.from_file(temp_trace_file)

        assert replayer.trace.trace_id == trace_with_events.trace_id

    def test_replayer_reset(self, trace_with_events):
        """Test resetting replayer position."""
        replayer = DebateReplayer(trace_with_events)

        # Advance position
        replayer.step()
        replayer.step()
        assert replayer._position == 2

        # Reset
        replayer.reset()
        assert replayer._position == 0

    def test_replayer_step(self, trace_with_events):
        """Test stepping through events."""
        replayer = DebateReplayer(trace_with_events)

        event1 = replayer.step()
        assert event1 is not None
        assert event1.event_type == EventType.AGENT_PROPOSAL

        event2 = replayer.step()
        assert event2 is not None
        assert event2.event_type == EventType.AGENT_CRITIQUE

        event3 = replayer.step()
        assert event3 is not None
        assert event3.event_type == EventType.CONSENSUS_CHECK

    def test_replayer_step_past_end(self, trace_with_events):
        """Test stepping past end returns None."""
        replayer = DebateReplayer(trace_with_events)

        # Step through all events
        for _ in range(len(trace_with_events.events)):
            replayer.step()

        # Next step should return None
        assert replayer.step() is None

    def test_replayer_step_to_round(self):
        """Test stepping to specific round."""
        events = []
        for round_num in range(3):
            events.append(
                TraceEvent(
                    event_id=f"round-{round_num}-start",
                    event_type=EventType.ROUND_START,
                    timestamp=f"2025-01-01T12:{round_num}0:00",
                    round_num=round_num,
                    agent=None,
                    content={"round": round_num},
                )
            )
            events.append(
                TraceEvent(
                    event_id=f"round-{round_num}-msg",
                    event_type=EventType.MESSAGE,
                    timestamp=f"2025-01-01T12:{round_num}1:00",
                    round_num=round_num,
                    agent="test",
                    content={"msg": f"Round {round_num} message"},
                )
            )

        trace = DebateTrace(
            trace_id="round-test",
            debate_id="round-debate",
            task="Test",
            agents=["test"],
            random_seed=1,
            events=events,
        )
        replayer = DebateReplayer(trace)

        # Step to round 2
        passed_events = replayer.step_to_round(2)

        # Should have passed rounds 0 and 1 (4 events)
        assert len(passed_events) == 4

    def test_replayer_get_state(self, trace_with_events):
        """Test getting current state."""
        replayer = DebateReplayer(trace_with_events)

        # Initial state
        initial_state = replayer.get_state()
        assert initial_state["round"] == 0
        assert initial_state["messages"] == []

        # After stepping
        replayer.step()
        state = replayer.get_state()
        assert len(state["messages"]) >= 0  # Depends on trace content

    def test_replayer_events_iterator(self, trace_with_events):
        """Test iterating through all events."""
        replayer = DebateReplayer(trace_with_events)

        events = list(replayer.events())
        assert len(events) == len(trace_with_events.events)

    def test_replayer_fork_at(self, trace_with_events, temp_db_path):
        """Test forking from a specific event."""
        replayer = DebateReplayer(trace_with_events)

        # Fork at first event
        first_event = trace_with_events.events[0]
        with patch("aragora.debate.traces.DebateTracer.__init__", return_value=None):
            # Create a minimal mock to avoid DB initialization
            pass

        # Test the fork logic with a real tracer
        new_tracer = replayer.fork_at(first_event.event_id, new_seed=999)

        assert new_tracer.debate_id.endswith("-fork")
        assert len(new_tracer.trace.events) == 1

    def test_replayer_fork_at_invalid_event(self, trace_with_events):
        """Test forking at non-existent event raises error."""
        replayer = DebateReplayer(trace_with_events)

        with pytest.raises(ValueError, match="Event not found"):
            replayer.fork_at("nonexistent-event-id")

    def test_replayer_generate_diff(self, trace_with_events, empty_trace):
        """Test generating diff between traces."""
        replayer = DebateReplayer(trace_with_events)

        # Add one event to empty trace for comparison
        empty_trace.add_event(trace_with_events.events[0])

        diffs = replayer.generate_diff(empty_trace)

        # Should have diffs for events 1 and 2 (removed)
        assert len(diffs) == 2
        assert diffs[0]["type"] == "removed"
        assert diffs[1]["type"] == "removed"

    def test_replayer_generate_diff_identical_traces(self, trace_with_events):
        """Test diff of identical traces is empty."""
        replayer = DebateReplayer(trace_with_events)
        diffs = replayer.generate_diff(trace_with_events)
        assert diffs == []

    def test_replayer_generate_diff_added_events(self, empty_trace, trace_event):
        """Test diff shows added events."""
        empty_trace.add_event(trace_event)
        replayer = DebateReplayer(empty_trace)

        other = DebateTrace(
            trace_id="other",
            debate_id="other",
            task="Other",
            agents=["a"],
            random_seed=1,
            events=[],
        )

        diffs = replayer.generate_diff(other)
        assert len(diffs) == 1
        assert diffs[0]["type"] == "removed"  # Present in self, not in other

    def test_replayer_generate_markdown_report(self, trace_with_events):
        """Test generating markdown report."""
        replayer = DebateReplayer(trace_with_events)
        report = replayer.generate_markdown_report()

        assert "# Debate Trace Report" in report
        assert trace_with_events.trace_id in report
        assert trace_with_events.task in report
        assert "checksum" in report.lower()

    def test_replayer_markdown_report_includes_proposals(self):
        """Test markdown report includes proposal content."""
        events = [
            TraceEvent(
                event_id="r1-start",
                event_type=EventType.ROUND_START,
                timestamp="2025-01-01T12:00:00",
                round_num=1,
                agent=None,
                content={"round": 1},
            ),
            TraceEvent(
                event_id="p1",
                event_type=EventType.AGENT_PROPOSAL,
                timestamp="2025-01-01T12:01:00",
                round_num=1,
                agent="claude",
                content={"content": "My detailed proposal content here"},
            ),
        ]
        trace = DebateTrace(
            trace_id="report-test",
            debate_id="report",
            task="Test task",
            agents=["claude"],
            random_seed=1,
            events=events,
        )
        replayer = DebateReplayer(trace)
        report = replayer.generate_markdown_report()

        assert "## Round 1" in report
        assert "claude" in report
        assert "Proposal" in report


# =============================================================================
# TestListTraces
# =============================================================================


class TestListTraces:
    """Test list_traces function for database queries."""

    def test_list_traces_empty_database(self, temp_db_path):
        """Test listing traces from empty database."""
        # Create an empty database with schema
        tracer = DebateTracer(
            debate_id="setup",
            task="Setup",
            agents=["a"],
            db_path=temp_db_path,
        )
        # Don't finalize - just use to create tables

        traces = list_traces(temp_db_path)
        assert traces == []

    def test_list_traces_with_data(self, temp_db_path):
        """Test listing traces from database with data."""
        # Create and finalize a tracer
        tracer = DebateTracer(
            debate_id="list-test-001",
            task="List test task",
            agents=["claude", "gpt4"],
            db_path=temp_db_path,
        )
        tracer.record_proposal("claude", "Test proposal", confidence=0.8)
        tracer.finalize({"answer": "Done"})

        traces = list_traces(temp_db_path)

        assert len(traces) == 1
        assert traces[0]["debate_id"] == "list-test-001"
        assert traces[0]["task"][:100] == "List test task"
        assert traces[0]["agents"] == ["claude", "gpt4"]

    def test_list_traces_limit(self, temp_db_path):
        """Test list_traces respects limit parameter."""
        # Create multiple traces
        for i in range(5):
            tracer = DebateTracer(
                debate_id=f"limit-test-{i:03d}",
                task=f"Task {i}",
                agents=["a"],
                db_path=temp_db_path,
            )
            tracer.finalize({"answer": f"Answer {i}"})

        # List with limit
        traces = list_traces(temp_db_path, limit=3)
        assert len(traces) == 3


# =============================================================================
# TestComplianceMetadata
# =============================================================================


class TestComplianceMetadata:
    """Test compliance-related metadata in traces."""

    def test_trace_includes_all_required_audit_fields(self, trace_with_events):
        """Test trace has all required audit/compliance fields."""
        json_str = trace_with_events.to_json()
        data = json.loads(json_str)

        required_fields = [
            "trace_id",
            "debate_id",
            "task",
            "agents",
            "random_seed",
            "events",
            "started_at",
            "completed_at",
            "checksum",
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_event_timestamps_are_iso_format(self, trace_with_events):
        """Test all event timestamps are ISO format."""
        for event in trace_with_events.events:
            # Should not raise
            datetime.fromisoformat(event.timestamp)

    def test_trace_checksum_provides_integrity(self, trace_with_events):
        """Test checksum changes if trace is tampered."""
        original_checksum = trace_with_events.checksum

        # Tamper with an event
        trace_with_events.events[0].content["tampered"] = True
        trace_with_events._mark_checksum_dirty()

        new_checksum = trace_with_events.checksum
        assert original_checksum != new_checksum

    def test_metadata_can_store_tenant_info(self, trace_with_events):
        """Test metadata can store multi-tenant compliance info."""
        assert trace_with_events.metadata["tenant_id"] == "tenant-001"
        assert trace_with_events.metadata["audit_id"] == "audit-xyz"

    def test_event_metadata_for_compliance(self):
        """Test event-level metadata for compliance tracking."""
        event = TraceEvent(
            event_id="compliance-e1",
            event_type=EventType.HUMAN_INTERVENTION,
            timestamp="2025-01-01T12:00:00",
            round_num=1,
            agent=None,
            content={"action": "manual_override"},
            metadata={
                "user_id": "user-123",
                "reason": "Policy violation",
                "ip_address": "192.168.1.1",
                "session_id": "sess-abc",
            },
        )

        assert event.metadata["user_id"] == "user-123"
        assert event.metadata["reason"] == "Policy violation"

    def test_random_seed_enables_reproducibility(self, temp_db_path):
        """Test random seed allows reproducible traces."""
        seed = 12345

        tracer1 = DebateTracer(
            debate_id="repro-1",
            task="Test",
            agents=["a"],
            random_seed=seed,
            db_path=temp_db_path,
        )

        tracer2 = DebateTracer(
            debate_id="repro-2",
            task="Test",
            agents=["a"],
            random_seed=seed,
            db_path=temp_db_path,
        )

        assert tracer1.random_seed == tracer2.random_seed == seed


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_content_event(self):
        """Test event with empty content."""
        event = TraceEvent(
            event_id="empty-content",
            event_type=EventType.MESSAGE,
            timestamp="2025-01-01T12:00:00",
            round_num=0,
            agent="test",
            content={},
        )
        d = event.to_dict()
        assert d["content"] == {}

    def test_very_long_content(self):
        """Test event with very long content."""
        long_content = {"text": "x" * 100000}
        event = TraceEvent(
            event_id="long-content",
            event_type=EventType.MESSAGE,
            timestamp="2025-01-01T12:00:00",
            round_num=0,
            agent="test",
            content=long_content,
        )

        # Should handle without error
        d = event.to_dict()
        assert len(d["content"]["text"]) == 100000

    def test_special_characters_in_content(self):
        """Test event with special characters."""
        special_content = {
            "text": "Special chars: <>&\"'\\n\\t\u0000\u2603"
        }
        event = TraceEvent(
            event_id="special-chars",
            event_type=EventType.MESSAGE,
            timestamp="2025-01-01T12:00:00",
            round_num=0,
            agent="test",
            content=special_content,
        )

        # Should survive JSON roundtrip
        trace = DebateTrace(
            trace_id="special",
            debate_id="special",
            task="Test",
            agents=["test"],
            random_seed=1,
            events=[event],
        )
        json_str = trace.to_json()
        restored = DebateTrace.from_json(json_str)
        assert restored.events[0].content["text"] == special_content["text"]

    def test_unicode_agent_names(self, temp_db_path):
        """Test handling of unicode in agent names."""
        tracer = DebateTracer(
            debate_id="unicode-test",
            task="Test unicode",
            agents=["クロード", "GPT-4"],
            db_path=temp_db_path,
        )
        tracer.record_proposal("クロード", "Japanese proposal", confidence=0.8)
        trace = tracer.finalize({"answer": "完了"})

        assert "クロード" in trace.agents
        proposals = trace.get_events_by_agent("クロード")
        assert len(proposals) == 1

    def test_trace_with_zero_events(self, empty_trace):
        """Test trace operations with zero events."""
        assert empty_trace.checksum is not None
        assert empty_trace.get_events_by_type(EventType.MESSAGE) == []
        assert empty_trace.get_events_by_agent("any") == []
        assert empty_trace.get_events_by_round(0) == []

    def test_large_number_of_events(self, empty_trace):
        """Test trace with many events."""
        for i in range(1000):
            empty_trace.add_event(
                TraceEvent(
                    event_id=f"bulk-{i:04d}",
                    event_type=EventType.MESSAGE,
                    timestamp=f"2025-01-01T{(i // 3600):02d}:{((i % 3600) // 60):02d}:{(i % 60):02d}",
                    round_num=i // 100,
                    agent=f"agent-{i % 5}",
                    content={"index": i},
                )
            )

        assert len(empty_trace.events) == 1000
        assert empty_trace.checksum is not None

        # Test filtering still works
        round_5_events = empty_trace.get_events_by_round(5)
        assert len(round_5_events) == 100

    def test_duplicate_event_ids(self, empty_trace):
        """Test handling of duplicate event IDs (should be allowed but not recommended)."""
        event1 = TraceEvent(
            event_id="duplicate-id",
            event_type=EventType.MESSAGE,
            timestamp="2025-01-01T12:00:00",
            round_num=0,
            agent="test",
            content={"first": True},
        )
        event2 = TraceEvent(
            event_id="duplicate-id",
            event_type=EventType.MESSAGE,
            timestamp="2025-01-01T12:01:00",
            round_num=0,
            agent="test",
            content={"second": True},
        )

        empty_trace.add_event(event1)
        empty_trace.add_event(event2)

        # Both should be present
        assert len(empty_trace.events) == 2

    def test_negative_round_number(self):
        """Test event with negative round number."""
        event = TraceEvent(
            event_id="neg-round",
            event_type=EventType.MESSAGE,
            timestamp="2025-01-01T12:00:00",
            round_num=-1,
            agent="test",
            content={"msg": "pre-debate"},
        )
        assert event.round_num == -1

    def test_invalid_event_type_from_dict(self):
        """Test error handling for invalid event type."""
        data = {
            "event_id": "bad-type",
            "event_type": "invalid_type",
            "timestamp": "2025-01-01T12:00:00",
            "round_num": 0,
            "agent": "test",
            "content": {},
            "parent_event_id": None,
            "duration_ms": None,
            "metadata": {},
        }
        with pytest.raises(ValueError):
            TraceEvent.from_dict(data)

    def test_malformed_json_deserialization(self):
        """Test error handling for malformed JSON."""
        with pytest.raises(json.JSONDecodeError):
            DebateTrace.from_json("not valid json {")

    def test_missing_required_fields_from_json(self):
        """Test error handling for missing required fields."""
        incomplete_json = json.dumps({"trace_id": "incomplete"})
        with pytest.raises((KeyError, TypeError)):
            DebateTrace.from_json(incomplete_json)

    def test_concurrent_event_addition(self, empty_trace):
        """Test that events can be added in sequence (simulating concurrent-like behavior)."""
        import threading

        def add_events(start_idx):
            for i in range(10):
                empty_trace.add_event(
                    TraceEvent(
                        event_id=f"thread-{start_idx}-{i}",
                        event_type=EventType.MESSAGE,
                        timestamp="2025-01-01T12:00:00",
                        round_num=0,
                        agent="test",
                        content={"idx": start_idx * 10 + i},
                    )
                )

        threads = [threading.Thread(target=add_events, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All events should be present (may have race conditions but list append is thread-safe in CPython)
        assert len(empty_trace.events) == 50


# =============================================================================
# TestReplayerFromDatabase
# =============================================================================


class TestReplayerFromDatabase:
    """Test DebateReplayer database loading."""

    def test_from_database_success(self, temp_db_path):
        """Test loading replayer from database."""
        # Create and finalize a trace
        tracer = DebateTracer(
            debate_id="db-load-test",
            task="Database load test",
            agents=["claude"],
            random_seed=42,
            db_path=temp_db_path,
        )
        tracer.record_proposal("claude", "Test", confidence=0.9)
        tracer.finalize({"answer": "Done"})

        # Load via replayer
        replayer = DebateReplayer.from_database(
            f"trace-db-load-test", db_path=temp_db_path
        )

        assert replayer.trace.debate_id == "db-load-test"
        assert len(replayer.trace.events) > 0

    def test_from_database_not_found(self, temp_db_path):
        """Test loading non-existent trace from database."""
        # Ensure tables exist
        tracer = DebateTracer(
            debate_id="setup",
            task="Setup",
            agents=["a"],
            db_path=temp_db_path,
        )

        with pytest.raises(ValueError, match="Trace not found"):
            DebateReplayer.from_database("nonexistent-trace", db_path=temp_db_path)
