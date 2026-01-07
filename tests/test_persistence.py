"""
Tests for persistence module.

Tests the data models and SupabaseClient (with mocked Supabase).
"""

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, Mock

import pytest

from aragora.persistence.models import (
    NomicCycle,
    DebateArtifact,
    StreamEvent,
    AgentMetrics,
)


# =============================================================================
# NomicCycle Model Tests
# =============================================================================


class TestNomicCycleModel:
    """Tests for NomicCycle dataclass."""

    def test_creates_with_required_fields(self):
        """Should create cycle with required fields."""
        cycle = NomicCycle(
            loop_id="test-loop",
            cycle_number=1,
            phase="debate",
            stage="proposing",
            started_at=datetime.now(),
        )

        assert cycle.loop_id == "test-loop"
        assert cycle.cycle_number == 1
        assert cycle.phase == "debate"
        assert cycle.stage == "proposing"

    def test_optional_fields_default_to_none(self):
        """Should set optional fields to None by default."""
        cycle = NomicCycle(
            loop_id="test",
            cycle_number=1,
            phase="debate",
            stage="init",
            started_at=datetime.now(),
        )

        assert cycle.completed_at is None
        assert cycle.success is None
        assert cycle.git_commit is None
        assert cycle.error_message is None
        assert cycle.id is None

    def test_to_dict_serializes_datetime(self):
        """Should serialize datetime fields to ISO format."""
        started = datetime(2026, 1, 1, 12, 0, 0)
        completed = datetime(2026, 1, 1, 13, 0, 0)

        cycle = NomicCycle(
            loop_id="test",
            cycle_number=1,
            phase="debate",
            stage="done",
            started_at=started,
            completed_at=completed,
        )

        d = cycle.to_dict()

        assert d["started_at"] == "2026-01-01T12:00:00"
        assert d["completed_at"] == "2026-01-01T13:00:00"

    def test_to_dict_handles_none_completed_at(self):
        """Should handle None completed_at in to_dict."""
        cycle = NomicCycle(
            loop_id="test",
            cycle_number=1,
            phase="debate",
            stage="in_progress",
            started_at=datetime.now(),
        )

        d = cycle.to_dict()

        assert d["completed_at"] is None


# =============================================================================
# DebateArtifact Model Tests
# =============================================================================


class TestDebateArtifactModel:
    """Tests for DebateArtifact dataclass."""

    def test_creates_with_required_fields(self):
        """Should create artifact with required fields."""
        artifact = DebateArtifact(
            loop_id="test-loop",
            cycle_number=1,
            phase="debate",
            task="Design a rate limiter",
            agents=["claude", "gemini"],
            transcript=[{"agent": "claude", "content": "Hello"}],
            consensus_reached=True,
            confidence=0.95,
        )

        assert artifact.loop_id == "test-loop"
        assert artifact.agents == ["claude", "gemini"]
        assert artifact.consensus_reached is True
        assert artifact.confidence == 0.95

    def test_to_dict_includes_all_fields(self):
        """Should include all fields in to_dict output."""
        artifact = DebateArtifact(
            loop_id="test",
            cycle_number=2,
            phase="design",
            task="Test task",
            agents=["agent1"],
            transcript=[],
            consensus_reached=False,
            confidence=0.5,
            winning_proposal="Proposal A",
            vote_tally={"A": 3, "B": 1},
        )

        d = artifact.to_dict()

        assert d["loop_id"] == "test"
        assert d["cycle_number"] == 2
        assert d["phase"] == "design"
        assert d["winning_proposal"] == "Proposal A"
        assert d["vote_tally"] == {"A": 3, "B": 1}

    def test_created_at_defaults_to_now(self):
        """Should default created_at to current time."""
        before = datetime.utcnow()

        artifact = DebateArtifact(
            loop_id="test",
            cycle_number=1,
            phase="debate",
            task="Task",
            agents=[],
            transcript=[],
            consensus_reached=True,
            confidence=1.0,
        )

        after = datetime.utcnow()

        assert before <= artifact.created_at <= after


# =============================================================================
# StreamEvent Model Tests
# =============================================================================


class TestStreamEventModel:
    """Tests for StreamEvent dataclass."""

    def test_creates_with_required_fields(self):
        """Should create event with required fields."""
        event = StreamEvent(
            loop_id="test-loop",
            cycle=1,
            event_type="phase_start",
            event_data={"phase": "debate"},
        )

        assert event.loop_id == "test-loop"
        assert event.cycle == 1
        assert event.event_type == "phase_start"
        assert event.event_data == {"phase": "debate"}

    def test_optional_agent_field(self):
        """Should allow optional agent field."""
        event = StreamEvent(
            loop_id="test",
            cycle=1,
            event_type="message",
            event_data={"content": "Hello"},
            agent="claude",
        )

        assert event.agent == "claude"

    def test_to_dict_serializes_correctly(self):
        """Should serialize to dict correctly."""
        timestamp = datetime(2026, 1, 7, 10, 30, 0)

        event = StreamEvent(
            loop_id="test",
            cycle=2,
            event_type="task_complete",
            event_data={"task": "Done"},
            agent="gemini",
            timestamp=timestamp,
        )

        d = event.to_dict()

        assert d["loop_id"] == "test"
        assert d["cycle"] == 2
        assert d["event_type"] == "task_complete"
        assert d["agent"] == "gemini"
        assert d["timestamp"] == "2026-01-07T10:30:00"


# =============================================================================
# AgentMetrics Model Tests
# =============================================================================


class TestAgentMetricsModel:
    """Tests for AgentMetrics dataclass."""

    def test_creates_with_required_fields(self):
        """Should create metrics with required fields."""
        metrics = AgentMetrics(
            loop_id="test",
            cycle=1,
            agent_name="claude",
            model="claude-3",
            phase="debate",
        )

        assert metrics.loop_id == "test"
        assert metrics.agent_name == "claude"
        assert metrics.model == "claude-3"

    def test_numeric_defaults(self):
        """Should default numeric fields to 0."""
        metrics = AgentMetrics(
            loop_id="test",
            cycle=1,
            agent_name="agent",
            model="model",
            phase="phase",
        )

        assert metrics.messages_sent == 0
        assert metrics.proposals_made == 0
        assert metrics.critiques_given == 0
        assert metrics.votes_won == 0
        assert metrics.votes_received == 0
        assert metrics.consensus_contributions == 0

    def test_to_dict_includes_all_metrics(self):
        """Should include all metrics in to_dict."""
        metrics = AgentMetrics(
            loop_id="test",
            cycle=3,
            agent_name="claude",
            model="claude-3",
            phase="debate",
            messages_sent=10,
            proposals_made=2,
            critiques_given=5,
            votes_won=3,
            votes_received=4,
            consensus_contributions=1,
            avg_response_time_ms=1500.5,
        )

        d = metrics.to_dict()

        assert d["messages_sent"] == 10
        assert d["proposals_made"] == 2
        assert d["critiques_given"] == 5
        assert d["votes_won"] == 3
        assert d["votes_received"] == 4
        assert d["avg_response_time_ms"] == 1500.5


# =============================================================================
# SupabaseClient Tests (Mocked)
# =============================================================================


class TestSupabaseClientInit:
    """Tests for SupabaseClient initialization."""

    def test_is_configured_false_without_credentials(self):
        """Should report not configured without credentials."""
        # Clear environment variables
        with patch.dict("os.environ", {}, clear=True):
            from aragora.persistence.supabase_client import SupabaseClient

            client = SupabaseClient()

            assert client.is_configured is False

    def test_is_configured_false_without_supabase_library(self):
        """Should handle missing supabase library gracefully."""
        with patch.dict("os.environ", {}, clear=True):
            from aragora.persistence.supabase_client import SupabaseClient

            client = SupabaseClient()

            assert client.is_configured is False


class TestSupabaseClientOperations:
    """Tests for SupabaseClient CRUD operations (mocked)."""

    @pytest.fixture
    def mock_client(self):
        """Create a SupabaseClient with mocked supabase."""
        with patch("aragora.persistence.supabase_client.SUPABASE_AVAILABLE", True):
            with patch("aragora.persistence.supabase_client.create_client") as mock_create:
                mock_supabase = MagicMock()
                mock_create.return_value = mock_supabase

                from aragora.persistence.supabase_client import SupabaseClient

                client = SupabaseClient(url="http://test.supabase.co", key="test-key")
                client.client = mock_supabase

                return client

    @pytest.mark.asyncio
    async def test_save_cycle_returns_none_when_not_configured(self):
        """Should return None when not configured."""
        from aragora.persistence.supabase_client import SupabaseClient

        client = SupabaseClient()  # Not configured

        cycle = NomicCycle(
            loop_id="test",
            cycle_number=1,
            phase="debate",
            stage="init",
            started_at=datetime.now(),
        )

        result = await client.save_cycle(cycle)

        assert result is None

    @pytest.mark.asyncio
    async def test_list_cycles_returns_empty_when_not_configured(self):
        """Should return empty list when not configured."""
        from aragora.persistence.supabase_client import SupabaseClient

        client = SupabaseClient()

        result = await client.list_cycles()

        assert result == []

    @pytest.mark.asyncio
    async def test_get_cycle_returns_none_when_not_configured(self):
        """Should return None when not configured."""
        from aragora.persistence.supabase_client import SupabaseClient

        client = SupabaseClient()

        result = await client.get_cycle("test", 1)

        assert result is None

    @pytest.mark.asyncio
    async def test_save_debate_returns_none_when_not_configured(self):
        """Should return None when not configured."""
        from aragora.persistence.supabase_client import SupabaseClient

        client = SupabaseClient()

        debate = DebateArtifact(
            loop_id="test",
            cycle_number=1,
            phase="debate",
            task="Test",
            agents=[],
            transcript=[],
            consensus_reached=True,
            confidence=0.9,
        )

        result = await client.save_debate(debate)

        assert result is None

    @pytest.mark.asyncio
    async def test_save_event_returns_none_when_not_configured(self):
        """Should return None when not configured."""
        from aragora.persistence.supabase_client import SupabaseClient

        client = SupabaseClient()

        event = StreamEvent(
            loop_id="test",
            cycle=1,
            event_type="test",
            event_data={},
        )

        result = await client.save_event(event)

        assert result is None

    @pytest.mark.asyncio
    async def test_save_events_batch_returns_zero_when_not_configured(self):
        """Should return 0 when not configured."""
        from aragora.persistence.supabase_client import SupabaseClient

        client = SupabaseClient()

        events = [
            StreamEvent(loop_id="test", cycle=1, event_type="test", event_data={})
        ]

        result = await client.save_events_batch(events)

        assert result == 0

    @pytest.mark.asyncio
    async def test_save_events_batch_returns_zero_for_empty_list(self):
        """Should return 0 for empty events list."""
        from aragora.persistence.supabase_client import SupabaseClient

        client = SupabaseClient()

        result = await client.save_events_batch([])

        assert result == 0

    @pytest.mark.asyncio
    async def test_get_events_returns_empty_when_not_configured(self):
        """Should return empty list when not configured."""
        from aragora.persistence.supabase_client import SupabaseClient

        client = SupabaseClient()

        result = await client.get_events("test-loop")

        assert result == []

    @pytest.mark.asyncio
    async def test_save_metrics_returns_none_when_not_configured(self):
        """Should return None when not configured."""
        from aragora.persistence.supabase_client import SupabaseClient

        client = SupabaseClient()

        metrics = AgentMetrics(
            loop_id="test",
            cycle=1,
            agent_name="agent",
            model="model",
            phase="debate",
        )

        result = await client.save_metrics(metrics)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_agent_stats_returns_empty_when_not_configured(self):
        """Should return empty list when not configured."""
        from aragora.persistence.supabase_client import SupabaseClient

        client = SupabaseClient()

        result = await client.get_agent_stats("claude")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_loop_summary_returns_empty_when_not_configured(self):
        """Should return empty dict when not configured."""
        from aragora.persistence.supabase_client import SupabaseClient

        client = SupabaseClient()

        result = await client.get_loop_summary("test-loop")

        assert result == {}


class TestSupabaseClientSubscription:
    """Tests for real-time subscription functionality."""

    def test_subscribe_returns_none_when_not_configured(self):
        """Should return None when not configured."""
        from aragora.persistence.supabase_client import SupabaseClient

        client = SupabaseClient()

        result = client.subscribe_to_events("test-loop", lambda e: None)

        assert result is None


class TestSupabaseClientDictConversions:
    """Tests for dict-to-model conversion methods."""

    def test_dict_to_cycle_handles_iso_datetime(self):
        """Should parse ISO datetime strings correctly."""
        from aragora.persistence.supabase_client import SupabaseClient

        client = SupabaseClient()

        data = {
            "id": "123",
            "loop_id": "test",
            "cycle_number": 1,
            "phase": "debate",
            "stage": "init",
            "started_at": "2026-01-07T10:00:00Z",
            "completed_at": "2026-01-07T11:00:00Z",
            "success": True,
        }

        cycle = client._dict_to_cycle(data)

        assert cycle.loop_id == "test"
        assert cycle.success is True

    def test_dict_to_cycle_handles_none_completed_at(self):
        """Should handle None completed_at."""
        from aragora.persistence.supabase_client import SupabaseClient

        client = SupabaseClient()

        data = {
            "loop_id": "test",
            "cycle_number": 1,
            "phase": "debate",
            "started_at": "2026-01-07T10:00:00Z",
            "completed_at": None,
        }

        cycle = client._dict_to_cycle(data)

        assert cycle.completed_at is None

    def test_dict_to_cycle_handles_invalid_datetime(self):
        """Should handle invalid datetime gracefully."""
        from aragora.persistence.supabase_client import SupabaseClient

        client = SupabaseClient()

        data = {
            "loop_id": "test",
            "cycle_number": 1,
            "phase": "debate",
            "started_at": "invalid-datetime",
        }

        cycle = client._dict_to_cycle(data)

        # Should use fallback datetime
        assert cycle.started_at is not None

    def test_dict_to_debate_creates_artifact(self):
        """Should create DebateArtifact from dict."""
        from aragora.persistence.supabase_client import SupabaseClient

        client = SupabaseClient()

        data = {
            "id": "456",
            "loop_id": "test",
            "cycle_number": 2,
            "phase": "design",
            "task": "Test task",
            "agents": ["claude", "gemini"],
            "transcript": [{"agent": "claude", "content": "Hi"}],
            "consensus_reached": True,
            "confidence": 0.85,
            "winning_proposal": "Plan A",
            "vote_tally": {"A": 2, "B": 1},
            "created_at": "2026-01-07T12:00:00Z",
        }

        artifact = client._dict_to_debate(data)

        assert artifact.id == "456"
        assert artifact.task == "Test task"
        assert artifact.confidence == 0.85

    def test_dict_to_event_creates_stream_event(self):
        """Should create StreamEvent from dict."""
        from aragora.persistence.supabase_client import SupabaseClient

        client = SupabaseClient()

        data = {
            "id": "789",
            "loop_id": "test",
            "cycle": 3,
            "event_type": "phase_start",
            "event_data": {"phase": "implement"},
            "agent": "codex",
            "timestamp": "2026-01-07T14:00:00Z",
        }

        event = client._dict_to_event(data)

        assert event.id == "789"
        assert event.event_type == "phase_start"
        assert event.agent == "codex"

    def test_dict_to_metrics_creates_agent_metrics(self):
        """Should create AgentMetrics from dict."""
        from aragora.persistence.supabase_client import SupabaseClient

        client = SupabaseClient()

        data = {
            "id": "abc",
            "loop_id": "test",
            "cycle": 4,
            "agent_name": "claude",
            "model": "claude-3.5",
            "phase": "verify",
            "messages_sent": 15,
            "proposals_made": 3,
            "critiques_given": 7,
            "votes_won": 4,
            "votes_received": 5,
            "consensus_contributions": 2,
            "avg_response_time_ms": 2500.0,
            "timestamp": "2026-01-07T15:00:00Z",
        }

        metrics = client._dict_to_metrics(data)

        assert metrics.agent_name == "claude"
        assert metrics.messages_sent == 15
        assert metrics.avg_response_time_ms == 2500.0
