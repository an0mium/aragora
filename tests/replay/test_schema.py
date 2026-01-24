"""
Tests for the replay schema module.

Tests ReplayEvent and ReplayMeta dataclasses.
"""

import json
import time

import pytest

from aragora.replay.schema import ReplayEvent, ReplayMeta, SCHEMA_VERSION


# =============================================================================
# ReplayEvent Tests
# =============================================================================


class TestReplayEvent:
    """Tests for ReplayEvent dataclass."""

    def test_create_event(self):
        """Test creating a replay event with all fields."""
        event = ReplayEvent(
            event_id="evt-123",
            timestamp=1704067200.0,
            offset_ms=5000,
            event_type="turn",
            source="claude",
            content="My proposal is...",
            metadata={"round": 1},
        )

        assert event.event_id == "evt-123"
        assert event.timestamp == 1704067200.0
        assert event.offset_ms == 5000
        assert event.event_type == "turn"
        assert event.source == "claude"
        assert event.content == "My proposal is..."
        assert event.metadata == {"round": 1}

    def test_default_metadata(self):
        """Test that metadata defaults to empty dict."""
        event = ReplayEvent(
            event_id="evt-456",
            timestamp=time.time(),
            offset_ms=0,
            event_type="system",
            source="system",
            content="Debate started",
        )

        assert event.metadata == {}

    def test_to_jsonl(self):
        """Test JSONL serialization."""
        event = ReplayEvent(
            event_id="evt-789",
            timestamp=1704067200.0,
            offset_ms=1000,
            event_type="vote",
            source="gpt",
            content="approve",
            metadata={"reasoning": "Good proposal"},
        )

        jsonl = event.to_jsonl()
        parsed = json.loads(jsonl)

        assert parsed["event_id"] == "evt-789"
        assert parsed["timestamp"] == 1704067200.0
        assert parsed["offset_ms"] == 1000
        assert parsed["event_type"] == "vote"
        assert parsed["source"] == "gpt"
        assert parsed["content"] == "approve"
        assert parsed["metadata"]["reasoning"] == "Good proposal"

    def test_from_jsonl(self):
        """Test JSONL deserialization."""
        jsonl = '{"event_id": "evt-abc", "timestamp": 1704067200.0, "offset_ms": 2000, "event_type": "turn", "source": "claude", "content": "Hello", "metadata": {}}'

        event = ReplayEvent.from_jsonl(jsonl)

        assert event.event_id == "evt-abc"
        assert event.timestamp == 1704067200.0
        assert event.offset_ms == 2000
        assert event.event_type == "turn"
        assert event.source == "claude"
        assert event.content == "Hello"

    def test_roundtrip_serialization(self):
        """Test that to_jsonl/from_jsonl roundtrip preserves data."""
        original = ReplayEvent(
            event_id="evt-roundtrip",
            timestamp=1704067200.0,
            offset_ms=3000,
            event_type="audience_input",
            source="user-123",
            content="What about edge cases?",
            metadata={"loop_id": "loop-1"},
        )

        jsonl = original.to_jsonl()
        restored = ReplayEvent.from_jsonl(jsonl)

        assert restored.event_id == original.event_id
        assert restored.timestamp == original.timestamp
        assert restored.offset_ms == original.offset_ms
        assert restored.event_type == original.event_type
        assert restored.source == original.source
        assert restored.content == original.content
        assert restored.metadata == original.metadata

    def test_from_jsonl_invalid_json(self):
        """Test that invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            ReplayEvent.from_jsonl("not valid json")

    def test_unicode_content(self):
        """Test handling of unicode content."""
        event = ReplayEvent(
            event_id="evt-unicode",
            timestamp=time.time(),
            offset_ms=0,
            event_type="turn",
            source="agent",
            content="Unicode test: \u4e2d\u6587 \U0001f680 \u00e9",
        )

        jsonl = event.to_jsonl()
        restored = ReplayEvent.from_jsonl(jsonl)

        assert restored.content == event.content


# =============================================================================
# ReplayMeta Tests
# =============================================================================


class TestReplayMeta:
    """Tests for ReplayMeta dataclass."""

    def test_create_meta(self):
        """Test creating replay metadata with all fields."""
        meta = ReplayMeta(
            debate_id="debate-123",
            topic="AI Safety",
            proposal="Implement alignment checks",
            agents=[{"id": "claude", "name": "Claude"}, {"id": "gpt", "name": "GPT"}],
            started_at="2024-01-01T00:00:00Z",
            ended_at="2024-01-01T00:10:00Z",
            duration_ms=600000,
            status="completed",
            final_verdict="approve",
            vote_tally={"approve": 2, "reject": 0},
            event_count=50,
            tags=["ai", "safety"],
        )

        assert meta.debate_id == "debate-123"
        assert meta.schema_version == SCHEMA_VERSION
        assert len(meta.agents) == 2
        assert meta.status == "completed"
        assert meta.final_verdict == "approve"
        assert meta.event_count == 50

    def test_default_values(self):
        """Test default values for optional fields."""
        meta = ReplayMeta()

        assert meta.schema_version == SCHEMA_VERSION
        assert meta.debate_id == ""
        assert meta.topic == ""
        assert meta.proposal == ""
        assert meta.agents == []
        assert meta.started_at == ""
        assert meta.ended_at is None
        assert meta.duration_ms is None
        assert meta.status == "in_progress"
        assert meta.final_verdict is None
        assert meta.vote_tally == {}
        assert meta.event_count == 0
        assert meta.tags == []

    def test_to_json(self):
        """Test JSON serialization."""
        meta = ReplayMeta(
            debate_id="debate-456",
            topic="API Design",
            proposal="Use REST",
            agents=[{"id": "claude", "name": "Claude"}],
            started_at="2024-01-01T00:00:00Z",
            status="in_progress",
        )

        json_str = meta.to_json()
        parsed = json.loads(json_str)

        assert parsed["debate_id"] == "debate-456"
        assert parsed["topic"] == "API Design"
        assert parsed["schema_version"] == SCHEMA_VERSION
        assert len(parsed["agents"]) == 1

    def test_from_json(self):
        """Test JSON deserialization."""
        json_str = """{
            "schema_version": "1.0",
            "debate_id": "debate-789",
            "topic": "Testing",
            "proposal": "Write more tests",
            "agents": [{"id": "agent1", "name": "Agent 1"}],
            "started_at": "2024-01-01T00:00:00Z",
            "ended_at": null,
            "duration_ms": null,
            "status": "in_progress",
            "final_verdict": null,
            "vote_tally": {},
            "event_count": 10,
            "tags": ["test"]
        }"""

        meta = ReplayMeta.from_json(json_str)

        assert meta.debate_id == "debate-789"
        assert meta.topic == "Testing"
        assert meta.event_count == 10
        assert meta.tags == ["test"]

    def test_roundtrip_serialization(self):
        """Test that to_json/from_json roundtrip preserves data."""
        original = ReplayMeta(
            debate_id="debate-roundtrip",
            topic="Roundtrip Test",
            proposal="Test serialization",
            agents=[{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
            started_at="2024-01-01T00:00:00Z",
            ended_at="2024-01-01T00:05:00Z",
            duration_ms=300000,
            status="completed",
            final_verdict="approve",
            vote_tally={"approve": 1, "reject": 1},
            event_count=25,
            tags=["test", "roundtrip"],
        )

        json_str = original.to_json()
        restored = ReplayMeta.from_json(json_str)

        assert restored.debate_id == original.debate_id
        assert restored.topic == original.topic
        assert restored.agents == original.agents
        assert restored.status == original.status
        assert restored.vote_tally == original.vote_tally
        assert restored.tags == original.tags

    def test_from_json_invalid_json(self):
        """Test that invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            ReplayMeta.from_json("{not valid json}")

    def test_schema_version_constant(self):
        """Test that schema version is correct."""
        assert SCHEMA_VERSION == "1.0"
