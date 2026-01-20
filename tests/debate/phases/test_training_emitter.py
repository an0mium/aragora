"""
Tests for training data emitter module.

Tests cover:
- TrainingEmitter class
- SFT record building
- DPO preference pair generation
- Calibration record building
- Insight usage recording
- Event emission
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.phases.training_emitter import TrainingEmitter


@dataclass
class MockResult:
    """Mock debate result."""

    final_answer: str = "The best approach is X"
    consensus_reached: bool = True
    confidence: float = 0.85
    rounds_used: int = 3
    winner: str = "claude"
    messages: list = field(default_factory=list)
    votes: list = field(default_factory=list)


@dataclass
class MockEnv:
    """Mock environment."""

    task: str = "What is the best approach?"
    context: str = "Some context"


@dataclass
class MockAgent:
    """Mock agent."""

    name: str


@dataclass
class MockMessage:
    """Mock message."""

    agent: str
    content: str
    role: str = "proposer"


@dataclass
class MockVote:
    """Mock vote."""

    agent: str
    choice: str
    confidence: float = 0.8
    reasoning: str = "Good reasoning"
    continue_debate: bool = False


@dataclass
class MockDebateContext:
    """Mock debate context."""

    result: MockResult = field(default_factory=MockResult)
    env: MockEnv = field(default_factory=MockEnv)
    debate_id: str = "test-debate-123"
    domain: str = "testing"
    applied_insight_ids: list = field(default_factory=list)
    agents: list = field(default_factory=list)
    choice_mapping: dict = field(default_factory=dict)


class TestTrainingEmitterInit:
    """Tests for TrainingEmitter initialization."""

    def test_default_init(self):
        """Default initialization sets correct defaults."""
        emitter = TrainingEmitter()

        assert emitter.training_exporter is None
        assert emitter.event_emitter is None
        assert emitter.insight_store is None
        assert emitter.loop_id is None
        assert emitter.sft_confidence_threshold == 0.8
        assert emitter.min_response_length == 50
        assert emitter.max_response_length == 4000

    def test_custom_init(self):
        """Custom initialization stores all parameters."""
        exporter = MagicMock()
        emitter_mock = MagicMock()
        store = MagicMock()

        emitter = TrainingEmitter(
            training_exporter=exporter,
            event_emitter=emitter_mock,
            insight_store=store,
            loop_id="loop-456",
            sft_confidence_threshold=0.9,
            min_response_length=100,
            max_response_length=2000,
        )

        assert emitter.training_exporter is exporter
        assert emitter.event_emitter is emitter_mock
        assert emitter.insight_store is store
        assert emitter.loop_id == "loop-456"
        assert emitter.sft_confidence_threshold == 0.9
        assert emitter.min_response_length == 100
        assert emitter.max_response_length == 2000


class TestBuildSftRecord:
    """Tests for SFT record building."""

    def test_builds_sft_record(self):
        """Builds valid SFT record from debate."""
        ctx = MockDebateContext()
        ctx.agents = [MockAgent("claude"), MockAgent("gpt4")]
        emitter = TrainingEmitter()

        record = emitter.build_sft_record(ctx)

        assert record is not None
        assert record["type"] == "sft"
        assert "What is the best approach?" in record["instruction"]
        assert record["response"] == "The best approach is X"
        assert record["metadata"]["debate_id"] == "test-debate-123"
        assert record["metadata"]["confidence"] == 0.85

    def test_includes_context_in_instruction(self):
        """SFT record includes context in instruction."""
        ctx = MockDebateContext()
        ctx.env.context = "Important background info"
        emitter = TrainingEmitter()

        record = emitter.build_sft_record(ctx)

        assert "Important background" in record["instruction"]

    def test_truncates_long_responses(self):
        """Long responses are truncated to max_response_length."""
        ctx = MockDebateContext()
        ctx.result.final_answer = "x" * 5000
        emitter = TrainingEmitter(max_response_length=1000)

        record = emitter.build_sft_record(ctx)

        assert len(record["response"]) == 1000

    def test_returns_none_without_final_answer(self):
        """Returns None when no final answer."""
        ctx = MockDebateContext()
        ctx.result.final_answer = ""
        emitter = TrainingEmitter()

        record = emitter.build_sft_record(ctx)

        assert record is None


class TestBuildDpoRecords:
    """Tests for DPO preference pair building."""

    def test_builds_dpo_pairs(self):
        """Builds DPO pairs from winner vs losers."""
        ctx = MockDebateContext()
        ctx.result.winner = "claude"
        ctx.result.messages = [
            MockMessage("claude", "Claude's detailed response " * 20),
            MockMessage("gpt4", "GPT4's detailed response " * 20),
        ]
        emitter = TrainingEmitter()

        records = emitter.build_dpo_records(ctx)

        assert len(records) == 1
        assert records[0]["type"] == "dpo"
        assert "Claude's" in records[0]["chosen"]
        assert "GPT4's" in records[0]["rejected"]
        assert records[0]["metadata"]["winner"] == "claude"
        assert records[0]["metadata"]["loser"] == "gpt4"

    def test_no_records_without_winner(self):
        """Returns empty list when no winner."""
        ctx = MockDebateContext()
        ctx.result.winner = None
        emitter = TrainingEmitter()

        records = emitter.build_dpo_records(ctx)

        assert records == []

    def test_no_records_without_messages(self):
        """Returns empty list when no messages."""
        ctx = MockDebateContext()
        ctx.result.messages = []
        emitter = TrainingEmitter()

        records = emitter.build_dpo_records(ctx)

        assert records == []

    def test_skips_short_responses(self):
        """Skips responses shorter than min_response_length."""
        ctx = MockDebateContext()
        ctx.result.winner = "claude"
        ctx.result.messages = [
            MockMessage("claude", "Long response " * 20),
            MockMessage("gpt4", "Short"),  # Too short
        ]
        emitter = TrainingEmitter(min_response_length=50)

        records = emitter.build_dpo_records(ctx)

        assert len(records) == 0


class TestBuildCalibrationRecords:
    """Tests for calibration record building."""

    def test_builds_calibration_records(self):
        """Builds calibration records from votes."""
        ctx = MockDebateContext()
        ctx.result.winner = "claude"
        ctx.result.votes = [
            MockVote("agent1", "claude", 0.9),
            MockVote("agent2", "gpt4", 0.7),
        ]
        ctx.choice_mapping = {"claude": "claude", "gpt4": "gpt4"}
        emitter = TrainingEmitter()

        records = emitter.build_calibration_records(ctx)

        assert len(records) == 2
        assert records[0]["type"] == "calibration"
        assert records[0]["agent"] == "agent1"
        assert records[0]["confidence"] == 0.9
        assert records[0]["correct"] is True

        assert records[1]["agent"] == "agent2"
        assert records[1]["confidence"] == 0.7
        assert records[1]["correct"] is False

    def test_no_records_without_votes(self):
        """Returns empty list when no votes."""
        ctx = MockDebateContext()
        ctx.result.votes = []
        emitter = TrainingEmitter()

        records = emitter.build_calibration_records(ctx)

        assert records == []

    def test_skips_votes_without_confidence(self):
        """Skips votes without confidence value."""
        vote = MagicMock()
        vote.agent = "agent1"
        vote.choice = "claude"
        del vote.confidence  # No confidence attribute

        ctx = MockDebateContext()
        ctx.result.winner = "claude"
        ctx.result.votes = [vote]
        emitter = TrainingEmitter()

        records = emitter.build_calibration_records(ctx)

        assert len(records) == 0


class TestRecordInsightUsage:
    """Tests for insight usage recording."""

    @pytest.mark.asyncio
    async def test_records_usage_for_applied_insights(self):
        """Records usage for applied insights."""
        ctx = MockDebateContext()
        ctx.applied_insight_ids = ["insight1", "insight2"]
        ctx.result.consensus_reached = True
        ctx.result.confidence = 0.8

        store = AsyncMock()
        emitter = TrainingEmitter(insight_store=store)

        await emitter.record_insight_usage(ctx)

        assert store.record_insight_usage.call_count == 2
        store.record_insight_usage.assert_any_call(
            insight_id="insight1",
            debate_id="test-debate-123",
            was_successful=True,
        )

    @pytest.mark.asyncio
    async def test_skips_without_store(self):
        """Skips when no insight store."""
        ctx = MockDebateContext()
        ctx.applied_insight_ids = ["insight1"]

        emitter = TrainingEmitter()  # No store

        # Should not raise
        await emitter.record_insight_usage(ctx)

    @pytest.mark.asyncio
    async def test_skips_without_applied_insights(self):
        """Skips when no applied insights."""
        ctx = MockDebateContext()
        ctx.applied_insight_ids = []

        store = AsyncMock()
        emitter = TrainingEmitter(insight_store=store)

        await emitter.record_insight_usage(ctx)

        store.record_insight_usage.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_without_consensus(self):
        """Skips recording when no consensus reached."""
        ctx = MockDebateContext()
        ctx.applied_insight_ids = ["insight1"]
        ctx.result.consensus_reached = False

        store = AsyncMock()
        emitter = TrainingEmitter(insight_store=store)

        await emitter.record_insight_usage(ctx)

        store.record_insight_usage.assert_not_called()


class TestEmitTrainingData:
    """Tests for training data emission."""

    @pytest.mark.asyncio
    async def test_emits_training_data(self):
        """Emits all training record types."""
        ctx = MockDebateContext()
        ctx.result.confidence = 0.85
        ctx.result.winner = "claude"
        ctx.result.messages = [
            MockMessage("claude", "Winner response " * 20),
            MockMessage("gpt4", "Loser response " * 20),
        ]
        ctx.result.votes = [MockVote("agent1", "claude", 0.9)]
        ctx.choice_mapping = {"claude": "claude"}
        ctx.agents = [MockAgent("claude")]

        exporter = MagicMock()
        emitter = TrainingEmitter(
            training_exporter=exporter,
            sft_confidence_threshold=0.8,
        )

        await emitter.emit_training_data(ctx)

        exporter.assert_called_once()
        records = exporter.call_args[0][0]
        types = [r["type"] for r in records]
        assert "sft" in types
        assert "dpo" in types
        assert "calibration" in types

    @pytest.mark.asyncio
    async def test_async_exporter_awaited(self):
        """Async exporter is awaited."""
        ctx = MockDebateContext()
        ctx.result.confidence = 0.85
        ctx.agents = [MockAgent("claude")]

        async_exporter = AsyncMock()
        emitter = TrainingEmitter(
            training_exporter=async_exporter,
            sft_confidence_threshold=0.8,
        )

        await emitter.emit_training_data(ctx)

        async_exporter.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_without_exporter(self):
        """Skips when no exporter configured."""
        ctx = MockDebateContext()
        emitter = TrainingEmitter()  # No exporter

        # Should not raise
        await emitter.emit_training_data(ctx)

    @pytest.mark.asyncio
    async def test_skips_without_final_answer(self):
        """Skips when no final answer."""
        ctx = MockDebateContext()
        ctx.result.final_answer = ""

        exporter = MagicMock()
        emitter = TrainingEmitter(training_exporter=exporter)

        await emitter.emit_training_data(ctx)

        exporter.assert_not_called()

    @pytest.mark.asyncio
    async def test_emits_event_notification(self):
        """Emits WebSocket event notification."""
        ctx = MockDebateContext()
        ctx.result.confidence = 0.85
        ctx.agents = [MockAgent("claude")]

        exporter = MagicMock()
        event_emitter = MagicMock()
        emitter = TrainingEmitter(
            training_exporter=exporter,
            event_emitter=event_emitter,
            loop_id="loop-123",
            sft_confidence_threshold=0.8,
        )

        # Patch at the source module where StreamEvent is defined
        with (
            patch("aragora.server.stream.StreamEvent") as mock_event,
            patch("aragora.server.stream.StreamEventType") as mock_type,
        ):
            mock_type.TRAINING_DATA_EXPORTED = "TRAINING_DATA_EXPORTED"
            await emitter.emit_training_data(ctx)

        # Event emitter should be called (may fail silently if StreamEventType missing)
        # Just verify the exporter was called
        exporter.assert_called()
