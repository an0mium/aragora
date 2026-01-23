"""Tests for Arena↔Control Plane Bridge."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.control_plane.arena_bridge import (
    AgentMetrics,
    ArenaControlPlaneBridge,
    ArenaEventAdapter,
    get_arena_bridge,
    init_arena_bridge,
    set_arena_bridge,
)
from aragora.control_plane.deliberation import (
    DeliberationSLA,
    DeliberationTask,
    SLAComplianceLevel,
)
from aragora.control_plane.deliberation_events import DeliberationEventType


class TestDeliberationEventType:
    """Tests for DeliberationEventType enum."""

    def test_lifecycle_events(self):
        """Test lifecycle event values."""
        assert DeliberationEventType.DELIBERATION_STARTED.value == "deliberation.started"
        assert DeliberationEventType.DELIBERATION_COMPLETED.value == "deliberation.completed"
        assert DeliberationEventType.DELIBERATION_FAILED.value == "deliberation.failed"

    def test_round_events(self):
        """Test round event values."""
        assert DeliberationEventType.ROUND_START.value == "deliberation.round_start"
        assert DeliberationEventType.ROUND_END.value == "deliberation.round_end"

    def test_sla_events(self):
        """Test SLA event values."""
        assert DeliberationEventType.SLA_WARNING.value == "deliberation.sla_warning"
        assert DeliberationEventType.SLA_CRITICAL.value == "deliberation.sla_critical"
        assert DeliberationEventType.SLA_VIOLATED.value == "deliberation.sla_violated"


class TestAgentMetrics:
    """Tests for AgentMetrics dataclass."""

    def test_default_values(self):
        """Test default values."""
        metrics = AgentMetrics(agent_id="test-agent")
        assert metrics.response_count == 0
        assert metrics.vote_count == 0
        assert metrics.critique_count == 0
        assert metrics.total_confidence == 0.0
        assert metrics.position_history == []
        assert metrics.contributed_to_final is False

    def test_accumulation(self):
        """Test metric accumulation."""
        metrics = AgentMetrics(agent_id="test-agent")
        metrics.response_count += 1
        metrics.vote_count += 2
        metrics.total_confidence += 0.8

        assert metrics.response_count == 1
        assert metrics.vote_count == 2
        assert metrics.total_confidence == 0.8


class TestArenaEventAdapter:
    """Tests for ArenaEventAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create test adapter."""
        return ArenaEventAdapter(
            task_id="test-task-123",
            stream_server=None,
            shared_state=None,
        )

    def test_init(self, adapter):
        """Test adapter initialization."""
        assert adapter.task_id == "test-task-123"
        assert adapter._current_round == 0
        assert adapter._total_rounds == 0
        assert adapter._agent_metrics == {}

    def test_ensure_agent_metrics(self, adapter):
        """Test agent metrics creation."""
        metrics = adapter._ensure_agent_metrics("claude")
        assert metrics.agent_id == "claude"

        # Same agent returns same metrics
        metrics2 = adapter._ensure_agent_metrics("claude")
        assert metrics is metrics2

    @pytest.mark.asyncio
    async def test_on_debate_start(self, adapter):
        """Test debate start event."""
        await adapter.on_debate_start(
            task="Test question",
            agents=["claude", "gpt-4"],
            rounds=3,
        )

        assert adapter._total_rounds == 3
        assert "claude" in adapter._agent_metrics
        assert "gpt-4" in adapter._agent_metrics

    @pytest.mark.asyncio
    async def test_on_round_start(self, adapter):
        """Test round start event."""
        await adapter.on_round_start(round_num=2, total_rounds=5)

        assert adapter._current_round == 2
        assert adapter._total_rounds == 5

    @pytest.mark.asyncio
    async def test_on_agent_message(self, adapter):
        """Test agent message tracking."""
        await adapter.on_agent_message(
            agent="claude",
            content="Test response content",
            role="proposer",
            round_num=1,
        )

        metrics = adapter.get_agent_metrics()["claude"]
        assert metrics.response_count == 1
        assert len(metrics.position_history) == 1

    @pytest.mark.asyncio
    async def test_on_vote(self, adapter):
        """Test vote tracking."""
        await adapter.on_vote(
            agent="claude",
            choice="option_a",
            confidence=0.85,
            reasoning="Good reasoning",
        )

        metrics = adapter.get_agent_metrics()["claude"]
        assert metrics.vote_count == 1
        assert metrics.total_confidence == 0.85

    @pytest.mark.asyncio
    async def test_on_critique(self, adapter):
        """Test critique tracking."""
        await adapter.on_critique(
            critic="gpt-4",
            target="claude",
            issues=["Issue 1", "Issue 2"],
            severity=7.5,
        )

        metrics = adapter.get_agent_metrics()["gpt-4"]
        assert metrics.critique_count == 1

    def test_summarize_votes(self, adapter):
        """Test vote summarization."""
        votes = {
            "claude": "option_a",
            "gpt-4": "option_a",
            "gemini": "option_b",
        }

        distribution = adapter._summarize_votes(votes)
        assert distribution["option_a"] == 2
        assert distribution["option_b"] == 1


class TestArenaControlPlaneBridge:
    """Tests for ArenaControlPlaneBridge."""

    @pytest.fixture
    def bridge(self):
        """Create test bridge."""
        return ArenaControlPlaneBridge(
            stream_server=None,
            shared_state=None,
            elo_callback=None,
        )

    def test_init(self, bridge):
        """Test bridge initialization."""
        assert bridge.stream_server is None
        assert bridge.shared_state is None
        assert bridge.elo_callback is None

    def test_create_event_hooks(self, bridge):
        """Test event hooks creation."""
        adapter = ArenaEventAdapter(task_id="test-123")
        task = DeliberationTask(
            question="Test question?",
            sla=DeliberationSLA(timeout_seconds=60),
        )

        hooks = bridge._create_event_hooks(adapter, task)

        assert "on_debate_start" in hooks
        assert "on_round_start" in hooks
        assert "on_round_end" in hooks
        assert "on_agent_message" in hooks
        assert "on_proposal" in hooks
        assert "on_critique" in hooks
        assert "on_vote" in hooks
        assert "on_consensus" in hooks
        assert "on_agent_error" in hooks

    def test_extract_agent_performance_empty(self, bridge):
        """Test performance extraction with no data."""
        result = MagicMock()
        result.winner = None
        result.final_answer = ""

        performances = bridge._extract_agent_performance(result, {})
        assert performances == {}

    def test_extract_agent_performance(self, bridge):
        """Test performance extraction with data."""
        result = MagicMock()
        result.winner = "claude"
        result.final_answer = "The answer is X"

        adapter_metrics = {
            "claude": AgentMetrics(
                agent_id="claude",
                response_count=3,
                vote_count=2,
                total_confidence=1.6,
                position_history=["pos1", "pos1"],
            ),
            "gpt-4": AgentMetrics(
                agent_id="gpt-4",
                response_count=2,
                vote_count=2,
                total_confidence=1.4,
                position_history=["pos2", "pos3"],
            ),
        }

        performances = bridge._extract_agent_performance(result, adapter_metrics)

        assert "claude" in performances
        assert "gpt-4" in performances
        assert performances["claude"].agent_id == "claude"
        assert performances["claude"].response_count == 3
        assert performances["claude"].average_confidence == 0.8
        assert performances["claude"].contributed_to_consensus is True
        assert performances["gpt-4"].position_changed is True


class TestBridgeSingleton:
    """Tests for bridge singleton functions."""

    def test_get_set_bridge(self):
        """Test get/set bridge functions."""
        # Clear any existing bridge
        set_arena_bridge(None)
        assert get_arena_bridge() is None

        # Set a bridge
        bridge = ArenaControlPlaneBridge()
        set_arena_bridge(bridge)
        assert get_arena_bridge() is bridge

        # Clean up
        set_arena_bridge(None)

    def test_init_bridge(self):
        """Test init_arena_bridge function."""
        # Clear any existing bridge
        set_arena_bridge(None)

        bridge = init_arena_bridge(
            stream_server=None,
            shared_state=None,
        )

        assert bridge is not None
        assert get_arena_bridge() is bridge

        # Clean up
        set_arena_bridge(None)


class TestSLAMonitoring:
    """Tests for SLA monitoring functionality."""

    @pytest.fixture
    def bridge(self):
        """Create test bridge."""
        return ArenaControlPlaneBridge()

    @pytest.mark.asyncio
    async def test_sla_warning_triggered(self, bridge):
        """Test that SLA warning is triggered at threshold."""
        sla_events = []

        async def capture_sla(event_type, data):
            sla_events.append((event_type, data))

        adapter = ArenaEventAdapter(
            task_id="test-sla",
            sla_callback=lambda et, d: sla_events.append((et, d)),
        )

        # Simulate time passage past warning threshold
        task = DeliberationTask(
            question="Test?",
            sla=DeliberationSLA(
                timeout_seconds=10.0,
                warning_threshold=0.5,  # Warning at 50%
            ),
        )
        task.metrics.started_at = time.time() - 6  # 6 seconds elapsed (60%)

        # Check SLA compliance
        compliance = task.sla.get_compliance_level(6.0)
        assert compliance == SLAComplianceLevel.WARNING

    @pytest.mark.asyncio
    async def test_sla_critical_triggered(self):
        """Test that SLA critical is triggered at threshold."""
        task = DeliberationTask(
            question="Test?",
            sla=DeliberationSLA(
                timeout_seconds=10.0,
                critical_threshold=0.9,  # Critical at 90%
            ),
        )

        # 9.5 seconds elapsed (95%)
        compliance = task.sla.get_compliance_level(9.5)
        assert compliance == SLAComplianceLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_sla_violated(self):
        """Test that SLA violation is detected."""
        task = DeliberationTask(
            question="Test?",
            sla=DeliberationSLA(timeout_seconds=10.0),
        )

        # 11 seconds elapsed (110%)
        compliance = task.sla.get_compliance_level(11.0)
        assert compliance == SLAComplianceLevel.VIOLATED
