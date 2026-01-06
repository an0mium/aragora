"""Tests for the Debate Tracing and Replay module."""
import pytest
import tempfile
import sqlite3
import random
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from aragora.debate.traces import (
    EventType,
    TraceEvent,
    DebateTrace,
    DebateTracer,
    DebateReplayer,
    list_traces,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create temporary SQLite database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    # Cleanup
    if db_path.exists():
        os.unlink(db_path)


@pytest.fixture
def temp_file():
    """Create temporary file for trace saving."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        file_path = Path(f.name)
    yield file_path
    # Cleanup
    if file_path.exists():
        os.unlink(file_path)


@pytest.fixture
def trace_event():
    """Create a basic TraceEvent."""
    return TraceEvent(
        event_id="debate-001-e0001",
        event_type=EventType.AGENT_PROPOSAL,
        timestamp="2025-01-01T12:00:00",
        round_num=1,
        agent="claude",
        content={"text": "My proposal", "confidence": 0.8},
    )


@pytest.fixture
def trace_event_with_metadata():
    """Create a TraceEvent with all fields populated."""
    return TraceEvent(
        event_id="debate-001-e0002",
        event_type=EventType.TOOL_RESULT,
        timestamp="2025-01-01T12:01:00",
        round_num=1,
        agent="gpt4",
        content={"tool": "search", "result": "found data"},
        parent_event_id="debate-001-e0001",
        duration_ms=1500,
        metadata={"call_event_id": "debate-001-e0001"},
    )


@pytest.fixture
def debate_trace(trace_event):
    """Create a basic DebateTrace."""
    return DebateTrace(
        trace_id="trace-debate-001",
        debate_id="debate-001",
        task="Design a system architecture",
        agents=["claude", "gpt4"],
        random_seed=42,
        events=[trace_event],
        started_at="2025-01-01T12:00:00",
    )


@pytest.fixture
def completed_debate_trace(trace_event):
    """Create a completed DebateTrace."""
    return DebateTrace(
        trace_id="trace-debate-001",
        debate_id="debate-001",
        task="Design a system",
        agents=["claude", "gpt4"],
        random_seed=42,
        events=[trace_event],
        started_at="2025-01-01T12:00:00",
        completed_at="2025-01-01T12:05:00",
        final_result={"conclusion": "Use microservices"},
    )


@pytest.fixture
def tracer(temp_db):
    """Create a DebateTracer with temp database."""
    return DebateTracer(
        debate_id="test-debate",
        task="Test task",
        agents=["agent1", "agent2"],
        db_path=str(temp_db),
        random_seed=42,
    )


@pytest.fixture
def tracer_with_events(temp_db):
    """Create a tracer with pre-recorded events."""
    tracer = DebateTracer(
        debate_id="test-debate",
        task="Test task",
        agents=["agent1", "agent2"],
        db_path=str(temp_db),
        random_seed=42,
    )
    tracer.start_round(1)
    tracer.record_proposal("agent1", "First proposal", 0.8)
    tracer.record_proposal("agent2", "Second proposal", 0.7)
    tracer.record_critique("agent1", "agent2", ["Issue 1"], 0.5, ["Fix it"])
    tracer.record_consensus(False, 0.6, {"agent1": True, "agent2": False})
    tracer.end_round()
    return tracer


@pytest.fixture
def replayer(debate_trace):
    """Create a DebateReplayer."""
    return DebateReplayer(debate_trace)


# =============================================================================
# TestEventType
# =============================================================================


class TestEventType:
    """Tests for EventType enum."""

    def test_all_values_defined(self):
        """Should define all expected values."""
        assert len(EventType) == 18

    def test_debate_start_value(self):
        """Should have DEBATE_START value."""
        assert EventType.DEBATE_START.value == "debate_start"

    def test_debate_end_value(self):
        """Should have DEBATE_END value."""
        assert EventType.DEBATE_END.value == "debate_end"

    def test_round_start_value(self):
        """Should have ROUND_START value."""
        assert EventType.ROUND_START.value == "round_start"

    def test_round_end_value(self):
        """Should have ROUND_END value."""
        assert EventType.ROUND_END.value == "round_end"

    def test_agent_proposal_value(self):
        """Should have AGENT_PROPOSAL value."""
        assert EventType.AGENT_PROPOSAL.value == "agent_proposal"

    def test_agent_critique_value(self):
        """Should have AGENT_CRITIQUE value."""
        assert EventType.AGENT_CRITIQUE.value == "agent_critique"

    def test_consensus_check_value(self):
        """Should have CONSENSUS_CHECK value."""
        assert EventType.CONSENSUS_CHECK.value == "consensus_check"

    def test_tool_call_value(self):
        """Should have TOOL_CALL value."""
        assert EventType.TOOL_CALL.value == "tool_call"

    def test_error_value(self):
        """Should have ERROR value."""
        assert EventType.ERROR.value == "error"


# =============================================================================
# TestTraceEvent
# =============================================================================


class TestTraceEvent:
    """Tests for TraceEvent dataclass."""

    def test_creation_with_required_fields(self):
        """Should create event with required fields."""
        event = TraceEvent(
            event_id="e-001",
            event_type=EventType.MESSAGE,
            timestamp="2025-01-01T12:00:00",
            round_num=1,
            agent="claude",
            content={"text": "Hello"},
        )
        assert event.event_id == "e-001"
        assert event.event_type == EventType.MESSAGE
        assert event.agent == "claude"

    def test_default_values(self, trace_event):
        """Should have correct default values."""
        assert trace_event.parent_event_id is None
        assert trace_event.duration_ms is None
        assert trace_event.metadata == {}

    def test_to_dict_all_fields(self, trace_event_with_metadata):
        """Should serialize all fields to dict."""
        data = trace_event_with_metadata.to_dict()
        assert data["event_id"] == "debate-001-e0002"
        assert data["event_type"] == "tool_result"
        assert data["timestamp"] == "2025-01-01T12:01:00"
        assert data["round_num"] == 1
        assert data["agent"] == "gpt4"
        assert data["parent_event_id"] == "debate-001-e0001"
        assert data["duration_ms"] == 1500
        assert data["metadata"]["call_event_id"] == "debate-001-e0001"

    def test_to_dict_enum_conversion(self, trace_event):
        """Should convert enum to string value."""
        data = trace_event.to_dict()
        assert data["event_type"] == "agent_proposal"
        assert isinstance(data["event_type"], str)

    def test_from_dict_basic(self):
        """Should deserialize from dict."""
        data = {
            "event_id": "e-003",
            "event_type": "agent_synthesis",
            "timestamp": "2025-01-01T12:00:00",
            "round_num": 2,
            "agent": "gpt4",
            "content": {"text": "Synthesis"},
            "parent_event_id": None,
            "duration_ms": None,
            "metadata": {},
        }
        event = TraceEvent.from_dict(data)
        assert event.event_id == "e-003"
        assert event.event_type == EventType.AGENT_SYNTHESIS
        assert event.agent == "gpt4"

    def test_from_dict_enum_reconstruction(self):
        """Should reconstruct enum from string value."""
        data = {
            "event_id": "e-004",
            "event_type": "consensus_check",
            "timestamp": "2025-01-01T12:00:00",
            "round_num": 1,
            "agent": None,
            "content": {},
            "parent_event_id": None,
            "duration_ms": None,
            "metadata": {},
        }
        event = TraceEvent.from_dict(data)
        assert event.event_type == EventType.CONSENSUS_CHECK
        assert isinstance(event.event_type, EventType)

    def test_serialization_round_trip(self, trace_event_with_metadata):
        """Should preserve all data through round-trip."""
        data = trace_event_with_metadata.to_dict()
        restored = TraceEvent.from_dict(data)
        assert restored.event_id == trace_event_with_metadata.event_id
        assert restored.event_type == trace_event_with_metadata.event_type
        assert restored.parent_event_id == trace_event_with_metadata.parent_event_id
        assert restored.metadata == trace_event_with_metadata.metadata


# =============================================================================
# TestDebateTrace
# =============================================================================


class TestDebateTrace:
    """Tests for DebateTrace dataclass."""

    def test_creation_with_required_fields(self):
        """Should create trace with required fields."""
        trace = DebateTrace(
            trace_id="t-001",
            debate_id="d-001",
            task="Test task",
            agents=["a1", "a2"],
            random_seed=42,
        )
        assert trace.trace_id == "t-001"
        assert trace.debate_id == "d-001"
        assert trace.random_seed == 42

    def test_default_values(self, debate_trace):
        """Should have correct default values."""
        assert debate_trace.events is not None
        assert debate_trace.completed_at is None
        assert debate_trace.final_result is None
        assert debate_trace.metadata == {}

    def test_checksum_deterministic(self, debate_trace):
        """Should generate same checksum for same events."""
        checksum1 = debate_trace.checksum
        checksum2 = debate_trace.checksum
        assert checksum1 == checksum2
        assert len(checksum1) == 16

    def test_checksum_changes_with_events(self, debate_trace, trace_event_with_metadata):
        """Should generate different checksum when events change."""
        checksum1 = debate_trace.checksum
        debate_trace.events.append(trace_event_with_metadata)
        checksum2 = debate_trace.checksum
        assert checksum1 != checksum2

    def test_duration_ms_with_timestamps(self, completed_debate_trace):
        """Should calculate duration when completed."""
        duration = completed_debate_trace.duration_ms
        assert duration is not None
        assert duration == 5 * 60 * 1000  # 5 minutes in ms

    def test_duration_ms_without_completed(self, debate_trace):
        """Should return None when not completed."""
        assert debate_trace.duration_ms is None

    def test_get_events_by_type(self, debate_trace, trace_event_with_metadata):
        """Should filter events by type."""
        debate_trace.events.append(trace_event_with_metadata)
        proposals = debate_trace.get_events_by_type(EventType.AGENT_PROPOSAL)
        assert len(proposals) == 1
        assert proposals[0].event_type == EventType.AGENT_PROPOSAL

    def test_get_events_by_agent(self, debate_trace, trace_event_with_metadata):
        """Should filter events by agent."""
        debate_trace.events.append(trace_event_with_metadata)
        claude_events = debate_trace.get_events_by_agent("claude")
        gpt_events = debate_trace.get_events_by_agent("gpt4")
        assert len(claude_events) == 1
        assert len(gpt_events) == 1

    def test_get_events_by_round(self, debate_trace):
        """Should filter events by round."""
        round1_events = debate_trace.get_events_by_round(1)
        round2_events = debate_trace.get_events_by_round(2)
        assert len(round1_events) == 1
        assert len(round2_events) == 0

    def test_get_events_empty_result(self, debate_trace):
        """Should return empty list for no matches."""
        errors = debate_trace.get_events_by_type(EventType.ERROR)
        assert errors == []


# =============================================================================
# TestDebateTraceSerialization
# =============================================================================


class TestDebateTraceSerialization:
    """Tests for DebateTrace serialization."""

    def test_to_json_structure(self, debate_trace):
        """Should serialize to valid JSON."""
        json_str = debate_trace.to_json()
        data = json.loads(json_str)
        assert "trace_id" in data
        assert "debate_id" in data
        assert "events" in data
        assert isinstance(data["events"], list)

    def test_to_json_includes_checksum(self, debate_trace):
        """Should include checksum in JSON."""
        json_str = debate_trace.to_json()
        data = json.loads(json_str)
        assert "checksum" in data
        assert data["checksum"] == debate_trace.checksum

    def test_from_json_basic(self, debate_trace):
        """Should deserialize from JSON."""
        json_str = debate_trace.to_json()
        restored = DebateTrace.from_json(json_str)
        assert restored.trace_id == debate_trace.trace_id
        assert restored.debate_id == debate_trace.debate_id
        assert len(restored.events) == len(debate_trace.events)

    def test_from_json_checksum_validation(self, debate_trace):
        """Should validate checksum on load."""
        json_str = debate_trace.to_json()
        # Should not raise
        restored = DebateTrace.from_json(json_str)
        assert restored.checksum == debate_trace.checksum

    def test_from_json_checksum_mismatch_raises(self, debate_trace):
        """Should raise ValueError on checksum mismatch."""
        json_str = debate_trace.to_json()
        data = json.loads(json_str)
        data["checksum"] = "invalid_checksum"
        with pytest.raises(ValueError, match="checksum mismatch"):
            DebateTrace.from_json(json.dumps(data))

    def test_save_creates_file(self, debate_trace, temp_file):
        """Should save trace to file."""
        debate_trace.save(temp_file)
        assert temp_file.exists()
        content = temp_file.read_text()
        assert debate_trace.trace_id in content

    def test_load_reads_file(self, debate_trace, temp_file):
        """Should load trace from file."""
        debate_trace.save(temp_file)
        loaded = DebateTrace.load(temp_file)
        assert loaded.trace_id == debate_trace.trace_id

    def test_save_load_round_trip(self, completed_debate_trace, temp_file):
        """Should preserve all data through save/load."""
        completed_debate_trace.save(temp_file)
        loaded = DebateTrace.load(temp_file)
        assert loaded.trace_id == completed_debate_trace.trace_id
        assert loaded.completed_at == completed_debate_trace.completed_at
        assert loaded.final_result == completed_debate_trace.final_result
        assert loaded.checksum == completed_debate_trace.checksum


# =============================================================================
# TestDebateTracerInit
# =============================================================================


class TestDebateTracerInit:
    """Tests for DebateTracer initialization."""

    def test_initialization_with_required_fields(self, temp_db):
        """Should initialize with required fields."""
        tracer = DebateTracer(
            debate_id="d-001",
            task="Test",
            agents=["a1"],
            db_path=str(temp_db),
        )
        assert tracer.debate_id == "d-001"
        assert tracer.task == "Test"
        assert tracer.agents == ["a1"]

    def test_sets_random_seed(self, temp_db):
        """Should set random seed for determinism."""
        tracer = DebateTracer(
            debate_id="d-001",
            task="Test",
            agents=["a1"],
            db_path=str(temp_db),
            random_seed=12345,
        )
        assert tracer.random_seed == 12345

    def test_creates_trace_object(self, tracer):
        """Should create internal trace object."""
        assert tracer.trace is not None
        assert isinstance(tracer.trace, DebateTrace)
        assert tracer.trace.debate_id == "test-debate"

    def test_initializes_event_counter(self, tracer):
        """Should initialize event counter to zero."""
        assert tracer._event_counter == 0

    def test_initializes_event_stack(self, tracer):
        """Should initialize empty event stack."""
        assert tracer._event_stack == []

    def test_db_path_creates_database(self, temp_db):
        """Should create database file."""
        DebateTracer(
            debate_id="d-001",
            task="Test",
            agents=["a1"],
            db_path=str(temp_db),
        )
        assert temp_db.exists()


# =============================================================================
# TestInitDb
# =============================================================================


class TestInitDb:
    """Tests for DebateTracer._init_db method."""

    def test_creates_traces_table(self, temp_db):
        """Should create traces table."""
        DebateTracer("d-001", "Test", ["a1"], db_path=str(temp_db))
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='traces'"
            )
            assert cursor.fetchone() is not None

    def test_creates_trace_events_table(self, temp_db):
        """Should create trace_events table."""
        DebateTracer("d-001", "Test", ["a1"], db_path=str(temp_db))
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='trace_events'"
            )
            assert cursor.fetchone() is not None

    def test_creates_index(self, temp_db):
        """Should create index on trace_id."""
        DebateTracer("d-001", "Test", ["a1"], db_path=str(temp_db))
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_trace_events_trace'"
            )
            assert cursor.fetchone() is not None

    def test_idempotent_on_reinit(self, temp_db):
        """Should be idempotent on re-initialization."""
        DebateTracer("d-001", "Test", ["a1"], db_path=str(temp_db))
        # Should not raise
        DebateTracer("d-002", "Test2", ["a2"], db_path=str(temp_db))


# =============================================================================
# TestRecord
# =============================================================================


class TestRecord:
    """Tests for DebateTracer.record method."""

    def test_generates_sequential_event_id(self, tracer):
        """Should generate sequential event IDs."""
        e1 = tracer.record(EventType.MESSAGE, {"text": "First"})
        e2 = tracer.record(EventType.MESSAGE, {"text": "Second"})
        assert e1.event_id == "test-debate-e0001"
        assert e2.event_id == "test-debate-e0002"

    def test_records_timestamp(self, tracer):
        """Should record timestamp on event."""
        event = tracer.record(EventType.MESSAGE, {"text": "Test"})
        assert event.timestamp is not None
        # Should be valid ISO format
        datetime.fromisoformat(event.timestamp)

    def test_sets_parent_from_stack(self, tracer):
        """Should set parent from event stack."""
        tracer.start_round(1)
        event = tracer.record(EventType.MESSAGE, {"text": "Test"})
        assert event.parent_event_id is not None
        assert event.parent_event_id == tracer._event_stack[-1]

    def test_no_parent_when_stack_empty(self, tracer):
        """Should have no parent when stack is empty."""
        event = tracer.record(EventType.MESSAGE, {"text": "Test"})
        assert event.parent_event_id is None

    def test_appends_to_trace_events(self, tracer):
        """Should append event to trace."""
        initial_count = len(tracer.trace.events)
        tracer.record(EventType.MESSAGE, {"text": "Test"})
        assert len(tracer.trace.events) == initial_count + 1

    def test_returns_created_event(self, tracer):
        """Should return the created event."""
        event = tracer.record(EventType.MESSAGE, {"text": "Test"}, agent="claude")
        assert isinstance(event, TraceEvent)
        assert event.agent == "claude"
        assert event.content == {"text": "Test"}

    def test_increments_counter(self, tracer):
        """Should increment event counter."""
        assert tracer._event_counter == 0
        tracer.record(EventType.MESSAGE, {"text": "Test"})
        assert tracer._event_counter == 1


# =============================================================================
# TestRoundManagement
# =============================================================================


class TestRoundManagement:
    """Tests for DebateTracer round management."""

    def test_start_round_records_event(self, tracer):
        """Should record ROUND_START event."""
        tracer.start_round(1)
        events = tracer.trace.get_events_by_type(EventType.ROUND_START)
        assert len(events) == 1
        assert events[0].content["round"] == 1

    def test_start_round_pushes_to_stack(self, tracer):
        """Should push event ID to stack."""
        assert len(tracer._event_stack) == 0
        tracer.start_round(1)
        assert len(tracer._event_stack) == 1

    def test_start_round_updates_current_round(self, tracer):
        """Should update current round number."""
        assert tracer._current_round == 0
        tracer.start_round(3)
        assert tracer._current_round == 3

    def test_end_round_records_event(self, tracer):
        """Should record ROUND_END event."""
        tracer.start_round(1)
        tracer.end_round()
        events = tracer.trace.get_events_by_type(EventType.ROUND_END)
        assert len(events) == 1

    def test_end_round_pops_from_stack(self, tracer):
        """Should pop from event stack."""
        tracer.start_round(1)
        assert len(tracer._event_stack) == 1
        tracer.end_round()
        assert len(tracer._event_stack) == 0

    def test_nested_rounds_parent_tracking(self, tracer):
        """Should track parents through nested rounds."""
        tracer.start_round(1)
        round1_id = tracer._event_stack[-1]
        proposal = tracer.record(EventType.AGENT_PROPOSAL, {"content": "Test"}, agent="a1")
        assert proposal.parent_event_id == round1_id


# =============================================================================
# TestSpecializedRecordMethods
# =============================================================================


class TestSpecializedRecordMethods:
    """Tests for specialized record methods."""

    def test_record_proposal(self, tracer):
        """Should record proposal event."""
        tracer.record_proposal("claude", "My proposal", 0.8)
        events = tracer.trace.get_events_by_type(EventType.AGENT_PROPOSAL)
        assert len(events) == 1
        assert events[0].agent == "claude"
        assert events[0].content["content"] == "My proposal"

    def test_record_proposal_with_confidence(self, tracer):
        """Should include confidence in proposal."""
        tracer.record_proposal("claude", "Proposal", 0.95)
        event = tracer.trace.events[-1]
        assert event.content["confidence"] == 0.95

    def test_record_critique(self, tracer):
        """Should record critique event."""
        tracer.record_critique("gpt4", "claude", ["Issue 1", "Issue 2"], 0.7, ["Fix A"])
        events = tracer.trace.get_events_by_type(EventType.AGENT_CRITIQUE)
        assert len(events) == 1
        assert events[0].content["target_agent"] == "claude"
        assert len(events[0].content["issues"]) == 2

    def test_record_critique_with_severity(self, tracer):
        """Should include severity in critique."""
        tracer.record_critique("gpt4", "claude", ["Bug"], 0.9, ["Fix it"])
        event = tracer.trace.events[-1]
        assert event.content["severity"] == 0.9

    def test_record_synthesis(self, tracer):
        """Should record synthesis event."""
        tracer.record_synthesis("claude", "Combined proposal", ["suggestion1"])
        events = tracer.trace.get_events_by_type(EventType.AGENT_SYNTHESIS)
        assert len(events) == 1
        assert events[0].content["incorporated"] == ["suggestion1"]

    def test_record_consensus(self, tracer):
        """Should record consensus event."""
        tracer.record_consensus(True, 0.85, {"claude": True, "gpt4": True})
        events = tracer.trace.get_events_by_type(EventType.CONSENSUS_CHECK)
        assert len(events) == 1
        assert events[0].content["reached"] is True
        assert events[0].content["confidence"] == 0.85

    def test_record_tool_call(self, tracer):
        """Should record tool call and return event ID."""
        event_id = tracer.record_tool_call("claude", "search", {"query": "test"})
        assert event_id is not None
        events = tracer.trace.get_events_by_type(EventType.TOOL_CALL)
        assert len(events) == 1

    def test_record_tool_result(self, tracer):
        """Should record tool result with reference."""
        call_id = tracer.record_tool_call("claude", "search", {})
        tracer.record_tool_result("claude", "search", "Found results", call_id)
        events = tracer.trace.get_events_by_type(EventType.TOOL_RESULT)
        assert len(events) == 1
        assert events[0].metadata["call_event_id"] == call_id

    def test_record_tool_result_truncation(self, tracer):
        """Should truncate long tool results."""
        long_result = "A" * 2000
        call_id = tracer.record_tool_call("claude", "search", {})
        tracer.record_tool_result("claude", "search", long_result, call_id)
        event = tracer.trace.events[-1]
        assert len(event.content["result"]) == 1000

    def test_record_error(self, tracer):
        """Should record error event."""
        tracer.record_error("Something went wrong", agent="claude")
        events = tracer.trace.get_events_by_type(EventType.ERROR)
        assert len(events) == 1
        assert events[0].content["error"] == "Something went wrong"


# =============================================================================
# TestFinalize
# =============================================================================


class TestFinalize:
    """Tests for DebateTracer.finalize method."""

    def test_sets_completed_at(self, tracer):
        """Should set completed_at timestamp."""
        assert tracer.trace.completed_at is None
        tracer.finalize({"result": "Done"})
        assert tracer.trace.completed_at is not None

    def test_sets_final_result(self, tracer):
        """Should set final result."""
        result = {"conclusion": "Use microservices"}
        tracer.finalize(result)
        assert tracer.trace.final_result == result

    def test_records_debate_end_event(self, tracer):
        """Should record DEBATE_END event."""
        tracer.finalize({"result": "Done"})
        events = tracer.trace.get_events_by_type(EventType.DEBATE_END)
        assert len(events) == 1

    def test_calls_save_trace(self, tracer, temp_db):
        """Should save trace to database."""
        tracer.finalize({"result": "Done"})
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM traces")
            count = cursor.fetchone()[0]
            assert count == 1

    def test_returns_completed_trace(self, tracer):
        """Should return the completed trace."""
        trace = tracer.finalize({"result": "Done"})
        assert isinstance(trace, DebateTrace)
        assert trace.completed_at is not None


# =============================================================================
# TestSaveTrace
# =============================================================================


class TestSaveTrace:
    """Tests for DebateTracer._save_trace method."""

    def test_inserts_into_traces_table(self, tracer, temp_db):
        """Should insert into traces table."""
        tracer.record(EventType.MESSAGE, {"text": "Test"})
        tracer._save_trace()
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT trace_id FROM traces")
            row = cursor.fetchone()
            assert row[0] == tracer.trace.trace_id

    def test_inserts_into_trace_events_table(self, tracer, temp_db):
        """Should insert events into trace_events table."""
        tracer.record(EventType.MESSAGE, {"text": "Test"})
        tracer._save_trace()
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trace_events")
            count = cursor.fetchone()[0]
            assert count == 1

    def test_upsert_replaces_existing(self, tracer, temp_db):
        """Should replace existing trace on upsert."""
        tracer.record(EventType.MESSAGE, {"text": "First"})
        tracer._save_trace()
        tracer.record(EventType.MESSAGE, {"text": "Second"})
        tracer._save_trace()
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM traces")
            count = cursor.fetchone()[0]
            assert count == 1

    def test_stores_checksum(self, tracer, temp_db):
        """Should store checksum in database."""
        tracer.record(EventType.MESSAGE, {"text": "Test"})
        tracer._save_trace()
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT checksum FROM traces")
            checksum = cursor.fetchone()[0]
            assert checksum == tracer.trace.checksum

    def test_stores_full_trace_json(self, tracer, temp_db):
        """Should store full trace JSON."""
        tracer.record(EventType.MESSAGE, {"text": "Test"})
        tracer._save_trace()
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT trace_json FROM traces")
            trace_json = cursor.fetchone()[0]
            assert tracer.trace.trace_id in trace_json


# =============================================================================
# TestGetStateAtEvent
# =============================================================================


class TestGetStateAtEvent:
    """Tests for DebateTracer.get_state_at_event method."""

    def test_returns_state_dict(self, tracer_with_events):
        """Should return state dictionary."""
        event_id = tracer_with_events.trace.events[-1].event_id
        state = tracer_with_events.get_state_at_event(event_id)
        assert isinstance(state, dict)
        assert "round" in state
        assert "messages" in state

    def test_tracks_current_round(self, tracer_with_events):
        """Should track current round number."""
        event_id = tracer_with_events.trace.events[-1].event_id
        state = tracer_with_events.get_state_at_event(event_id)
        assert state["round"] == 1

    def test_collects_messages(self, tracer_with_events):
        """Should collect messages up to event."""
        event_id = tracer_with_events.trace.events[-1].event_id
        state = tracer_with_events.get_state_at_event(event_id)
        assert len(state["messages"]) == 2

    def test_collects_critiques(self, tracer_with_events):
        """Should collect critiques up to event."""
        event_id = tracer_with_events.trace.events[-1].event_id
        state = tracer_with_events.get_state_at_event(event_id)
        assert len(state["critiques"]) == 1

    def test_tracks_agents_acted(self, tracer_with_events):
        """Should track which agents have acted."""
        event_id = tracer_with_events.trace.events[-1].event_id
        state = tracer_with_events.get_state_at_event(event_id)
        assert "agent1" in state["agents_acted"]
        assert "agent2" in state["agents_acted"]

    def test_tracks_consensus(self, tracer_with_events):
        """Should track consensus state."""
        event_id = tracer_with_events.trace.events[-1].event_id
        state = tracer_with_events.get_state_at_event(event_id)
        assert state["consensus"] is not None
        assert state["consensus"]["reached"] is False

    def test_stops_at_target_event(self, tracer_with_events):
        """Should stop at target event."""
        # Get second event (first proposal)
        events = tracer_with_events.trace.get_events_by_type(EventType.AGENT_PROPOSAL)
        first_proposal_id = events[0].event_id
        state = tracer_with_events.get_state_at_event(first_proposal_id)
        assert len(state["messages"]) == 1

    def test_handles_event_not_found(self, tracer_with_events):
        """Should return full state if event not found."""
        state = tracer_with_events.get_state_at_event("nonexistent")
        # Should process all events
        assert len(state["messages"]) == 2


# =============================================================================
# TestDebateReplayerInit
# =============================================================================


class TestDebateReplayerInit:
    """Tests for DebateReplayer initialization."""

    def test_initialization_with_trace(self, debate_trace):
        """Should initialize with trace."""
        replayer = DebateReplayer(debate_trace)
        assert replayer.trace == debate_trace

    def test_sets_position_to_zero(self, debate_trace):
        """Should set position to zero."""
        replayer = DebateReplayer(debate_trace)
        assert replayer._position == 0

    def test_restores_random_seed(self, debate_trace):
        """Should restore random seed for determinism."""
        replayer = DebateReplayer(debate_trace)
        # Random should be seeded with trace's seed
        value1 = random.random()
        replayer.reset()
        value2 = random.random()
        assert value1 == value2

    def test_from_file_loads_trace(self, debate_trace, temp_file):
        """Should load replayer from file."""
        debate_trace.save(temp_file)
        replayer = DebateReplayer.from_file(temp_file)
        assert replayer.trace.trace_id == debate_trace.trace_id

    def test_from_database_loads_trace(self, tracer_with_events, temp_db):
        """Should load replayer from database."""
        tracer_with_events.finalize({"result": "Done"})
        replayer = DebateReplayer.from_database(
            tracer_with_events.trace.trace_id,
            db_path=str(temp_db),
        )
        assert replayer.trace.trace_id == tracer_with_events.trace.trace_id

    def test_from_database_not_found_raises(self, temp_db):
        """Should raise ValueError when trace not found."""
        # Create empty db
        DebateTracer("d-001", "Test", ["a1"], db_path=str(temp_db))
        with pytest.raises(ValueError, match="Trace not found"):
            DebateReplayer.from_database("nonexistent", db_path=str(temp_db))


# =============================================================================
# TestStep
# =============================================================================


class TestStep:
    """Tests for DebateReplayer.step method."""

    def test_returns_first_event(self, replayer):
        """Should return first event on first step."""
        event = replayer.step()
        assert event is not None
        assert event == replayer.trace.events[0]

    def test_increments_position(self, replayer):
        """Should increment position after step."""
        assert replayer._position == 0
        replayer.step()
        assert replayer._position == 1

    def test_returns_sequential_events(self, tracer_with_events):
        """Should return events in sequence."""
        replayer = DebateReplayer(tracer_with_events.trace)
        events = []
        while True:
            event = replayer.step()
            if event is None:
                break
            events.append(event)
        assert len(events) == len(tracer_with_events.trace.events)

    def test_returns_none_at_end(self, replayer):
        """Should return None at end of events."""
        # Step through all events
        while replayer.step() is not None:
            pass
        assert replayer.step() is None

    def test_reset_restores_position(self, replayer):
        """Should reset position to zero."""
        replayer.step()
        replayer.step()
        replayer.reset()
        assert replayer._position == 0

    def test_reset_reseeds_random(self, debate_trace):
        """Should reseed random on reset."""
        replayer = DebateReplayer(debate_trace)
        random.random()  # Consume some random
        replayer.reset()
        value1 = random.random()
        replayer.reset()
        value2 = random.random()
        assert value1 == value2


# =============================================================================
# TestStepToRound
# =============================================================================


class TestStepToRound:
    """Tests for DebateReplayer.step_to_round method."""

    def test_finds_target_round(self, tracer_with_events):
        """Should find target round."""
        replayer = DebateReplayer(tracer_with_events.trace)
        replayer.step_to_round(1)
        # Position should be at ROUND_START for round 1
        current = replayer.trace.events[replayer._position]
        assert current.event_type == EventType.ROUND_START

    def test_returns_skipped_events(self, tracer_with_events):
        """Should return skipped events."""
        # Add second round
        tracer_with_events.start_round(2)
        tracer_with_events.record_proposal("agent1", "Round 2", 0.8)
        tracer_with_events.end_round()

        replayer = DebateReplayer(tracer_with_events.trace)
        skipped = replayer.step_to_round(2)
        # Should have skipped all round 1 events
        assert len(skipped) > 0

    def test_stops_at_round_start(self, tracer_with_events):
        """Should stop at ROUND_START event."""
        replayer = DebateReplayer(tracer_with_events.trace)
        replayer.step_to_round(1)
        event = replayer.trace.events[replayer._position]
        assert event.event_type == EventType.ROUND_START
        assert event.content["round"] == 1

    def test_handles_round_not_found(self, tracer_with_events):
        """Should handle round not found."""
        replayer = DebateReplayer(tracer_with_events.trace)
        skipped = replayer.step_to_round(99)
        # Should step through all events
        assert replayer._position == len(replayer.trace.events)


# =============================================================================
# TestGetState
# =============================================================================


class TestGetState:
    """Tests for DebateReplayer.get_state method."""

    def test_returns_current_state(self, tracer_with_events):
        """Should return current state."""
        replayer = DebateReplayer(tracer_with_events.trace)
        replayer.step()  # ROUND_START
        replayer.step()  # First proposal
        state = replayer.get_state()
        assert "round" in state
        assert "messages" in state

    def test_state_updates_with_step(self, tracer_with_events):
        """Should update state as we step."""
        replayer = DebateReplayer(tracer_with_events.trace)
        replayer.step()  # ROUND_START
        state1 = replayer.get_state()
        replayer.step()  # First proposal
        state2 = replayer.get_state()
        assert len(state2["messages"]) > len(state1["messages"])

    def test_state_after_reset(self, tracer_with_events):
        """Should return initial state after reset."""
        replayer = DebateReplayer(tracer_with_events.trace)
        replayer.step()
        replayer.step()
        replayer.reset()
        state = replayer.get_state()
        assert state["round"] == 0
        assert state["messages"] == []


# =============================================================================
# TestForkAt
# =============================================================================


class TestForkAt:
    """Tests for DebateReplayer.fork_at method."""

    def test_creates_new_tracer(self, tracer_with_events):
        """Should create new DebateTracer."""
        replayer = DebateReplayer(tracer_with_events.trace)
        event_id = tracer_with_events.trace.events[2].event_id
        forked = replayer.fork_at(event_id)
        assert isinstance(forked, DebateTracer)

    def test_copies_events_up_to_fork(self, tracer_with_events):
        """Should copy events up to fork point."""
        replayer = DebateReplayer(tracer_with_events.trace)
        event_id = tracer_with_events.trace.events[2].event_id
        forked = replayer.fork_at(event_id)
        assert len(forked.trace.events) == 3  # Events 0, 1, 2

    def test_sets_new_debate_id(self, tracer_with_events):
        """Should set new debate ID with -fork suffix."""
        replayer = DebateReplayer(tracer_with_events.trace)
        event_id = tracer_with_events.trace.events[1].event_id
        forked = replayer.fork_at(event_id)
        assert "-fork" in forked.debate_id

    def test_resets_event_counter(self, tracer_with_events):
        """Should reset event counter for continuation."""
        replayer = DebateReplayer(tracer_with_events.trace)
        event_id = tracer_with_events.trace.events[2].event_id
        forked = replayer.fork_at(event_id)
        assert forked._event_counter == 3

    def test_restores_current_round(self, tracer_with_events):
        """Should restore current round from events."""
        replayer = DebateReplayer(tracer_with_events.trace)
        event_id = tracer_with_events.trace.events[2].event_id
        forked = replayer.fork_at(event_id)
        assert forked._current_round == 1

    def test_uses_new_seed_if_provided(self, tracer_with_events):
        """Should use new random seed if provided."""
        replayer = DebateReplayer(tracer_with_events.trace)
        event_id = tracer_with_events.trace.events[1].event_id
        forked = replayer.fork_at(event_id, new_seed=99999)
        assert forked.random_seed == 99999

    def test_raises_on_event_not_found(self, tracer_with_events):
        """Should raise ValueError for unknown event."""
        replayer = DebateReplayer(tracer_with_events.trace)
        with pytest.raises(ValueError, match="Event not found"):
            replayer.fork_at("nonexistent-event")


# =============================================================================
# TestGenerateDiff
# =============================================================================


class TestGenerateDiff:
    """Tests for DebateReplayer.generate_diff method."""

    def test_identical_traces_no_diff(self, debate_trace):
        """Should return empty diff for identical traces."""
        replayer = DebateReplayer(debate_trace)
        diffs = replayer.generate_diff(debate_trace)
        assert diffs == []

    def test_detects_added_events(self, debate_trace, trace_event_with_metadata):
        """Should detect added events."""
        replayer = DebateReplayer(debate_trace)
        other = DebateTrace(
            trace_id="t-002",
            debate_id="d-002",
            task="Task",
            agents=["a1"],
            random_seed=42,
            events=debate_trace.events + [trace_event_with_metadata],
            started_at=debate_trace.started_at,
        )
        diffs = replayer.generate_diff(other)
        assert any(d["type"] == "added" for d in diffs)

    def test_detects_removed_events(self, debate_trace, trace_event_with_metadata):
        """Should detect removed events."""
        other = DebateTrace(
            trace_id="t-002",
            debate_id="d-002",
            task="Task",
            agents=["a1"],
            random_seed=42,
            events=[],
            started_at=debate_trace.started_at,
        )
        replayer = DebateReplayer(debate_trace)
        diffs = replayer.generate_diff(other)
        assert any(d["type"] == "removed" for d in diffs)

    def test_detects_changed_events(self, debate_trace):
        """Should detect changed events."""
        # Create trace with same event but different content
        changed_event = TraceEvent(
            event_id=debate_trace.events[0].event_id,
            event_type=debate_trace.events[0].event_type,
            timestamp=debate_trace.events[0].timestamp,
            round_num=debate_trace.events[0].round_num,
            agent=debate_trace.events[0].agent,
            content={"different": "content"},
        )
        other = DebateTrace(
            trace_id="t-002",
            debate_id="d-002",
            task="Task",
            agents=["a1"],
            random_seed=42,
            events=[changed_event],
            started_at=debate_trace.started_at,
        )
        replayer = DebateReplayer(debate_trace)
        diffs = replayer.generate_diff(other)
        assert any(d["type"] == "changed" for d in diffs)

    def test_handles_empty_traces(self):
        """Should handle empty traces."""
        trace1 = DebateTrace(
            trace_id="t-001",
            debate_id="d-001",
            task="Task",
            agents=["a1"],
            random_seed=42,
            events=[],
            started_at="2025-01-01T12:00:00",
        )
        trace2 = DebateTrace(
            trace_id="t-002",
            debate_id="d-002",
            task="Task",
            agents=["a1"],
            random_seed=42,
            events=[],
            started_at="2025-01-01T12:00:00",
        )
        replayer = DebateReplayer(trace1)
        diffs = replayer.generate_diff(trace2)
        assert diffs == []


# =============================================================================
# TestGenerateMarkdownReport
# =============================================================================


class TestGenerateMarkdownReport:
    """Tests for DebateReplayer.generate_markdown_report method."""

    def test_generates_markdown_string(self, tracer_with_events):
        """Should generate markdown string."""
        replayer = DebateReplayer(tracer_with_events.trace)
        report = replayer.generate_markdown_report()
        assert isinstance(report, str)
        assert len(report) > 0

    def test_includes_debate_header(self, tracer_with_events):
        """Should include debate header."""
        replayer = DebateReplayer(tracer_with_events.trace)
        report = replayer.generate_markdown_report()
        assert "# Debate Trace Report" in report
        assert tracer_with_events.trace.trace_id in report

    def test_includes_round_sections(self, tracer_with_events):
        """Should include round sections."""
        replayer = DebateReplayer(tracer_with_events.trace)
        report = replayer.generate_markdown_report()
        assert "## Round 1" in report

    def test_includes_proposals(self, tracer_with_events):
        """Should include proposals."""
        replayer = DebateReplayer(tracer_with_events.trace)
        report = replayer.generate_markdown_report()
        assert "(Proposal)" in report

    def test_includes_critiques(self, tracer_with_events):
        """Should include critiques."""
        replayer = DebateReplayer(tracer_with_events.trace)
        report = replayer.generate_markdown_report()
        assert "(Critique)" in report
        assert "Severity" in report

    def test_includes_consensus(self, tracer_with_events):
        """Should include consensus checks."""
        replayer = DebateReplayer(tracer_with_events.trace)
        report = replayer.generate_markdown_report()
        assert "Consensus Check" in report
        assert "Reached" in report


# =============================================================================
# TestListTraces
# =============================================================================


class TestListTraces:
    """Tests for list_traces function."""

    def test_returns_empty_for_no_traces(self, temp_db):
        """Should return empty list when no traces."""
        # Initialize db
        DebateTracer("d-001", "Test", ["a1"], db_path=str(temp_db))
        # Don't finalize, so no traces saved
        traces = list_traces(db_path=str(temp_db))
        assert traces == []

    def test_returns_trace_metadata(self, tracer_with_events, temp_db):
        """Should return trace metadata."""
        tracer_with_events.finalize({"result": "Done"})
        traces = list_traces(db_path=str(temp_db))
        assert len(traces) == 1
        assert "trace_id" in traces[0]
        assert "debate_id" in traces[0]
        assert "checksum" in traces[0]

    def test_respects_limit(self, temp_db):
        """Should respect limit parameter."""
        for i in range(5):
            tracer = DebateTracer(f"d-{i}", "Test", ["a1"], db_path=str(temp_db))
            tracer.finalize({"result": f"Result {i}"})

        traces = list_traces(db_path=str(temp_db), limit=3)
        assert len(traces) == 3

    def test_orders_by_started_at_desc(self, temp_db):
        """Should order by started_at descending."""
        for i in range(3):
            tracer = DebateTracer(f"d-{i}", "Test", ["a1"], db_path=str(temp_db))
            tracer.finalize({"result": f"Result {i}"})

        traces = list_traces(db_path=str(temp_db))
        # Most recent first
        for i in range(len(traces) - 1):
            assert traces[i]["started_at"] >= traces[i + 1]["started_at"]


# =============================================================================
# TestDeterministicReplay
# =============================================================================


class TestDeterministicReplay:
    """Tests for deterministic replay with seeded RNG."""

    def test_same_seed_same_sequence(self, temp_db):
        """Should produce same random sequence with same seed."""
        tracer1 = DebateTracer("d-1", "Test", ["a1"], db_path=str(temp_db), random_seed=42)
        seq1 = [random.random() for _ in range(5)]

        tracer2 = DebateTracer("d-2", "Test", ["a1"], db_path=str(temp_db), random_seed=42)
        seq2 = [random.random() for _ in range(5)]

        assert seq1 == seq2

    def test_different_seed_different_sequence(self, temp_db):
        """Should produce different random sequence with different seed."""
        tracer1 = DebateTracer("d-1", "Test", ["a1"], db_path=str(temp_db), random_seed=42)
        seq1 = [random.random() for _ in range(5)]

        tracer2 = DebateTracer("d-2", "Test", ["a1"], db_path=str(temp_db), random_seed=99)
        seq2 = [random.random() for _ in range(5)]

        assert seq1 != seq2

    def test_replay_restores_seed(self, tracer_with_events):
        """Should restore random seed on replay."""
        # Record random values during initial tracing
        initial_seed = tracer_with_events.random_seed

        # Create replayer - should restore seed
        replayer = DebateReplayer(tracer_with_events.trace)
        value1 = random.random()

        # Reset and get same value
        replayer.reset()
        value2 = random.random()

        assert value1 == value2
