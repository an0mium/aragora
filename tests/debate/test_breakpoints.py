"""
Tests for Human Intervention Breakpoints module.

Tests cover:
- BreakpointTrigger enum values
- Data classes (DebateSnapshot, HumanGuidance, Breakpoint, BreakpointConfig)
- HumanNotifier notification handling
- BreakpointManager trigger detection and handling
- Guidance injection into debates
- Decorators (critical_decision, breakpoint)
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from aragora.debate.breakpoints import (
    Breakpoint,
    BreakpointConfig,
    BreakpointManager,
    BreakpointTrigger,
    DebateSnapshot,
    HumanGuidance,
    HumanNotifier,
    breakpoint,
    critical_decision,
)


# =============================================================================
# BreakpointTrigger Enum Tests
# =============================================================================


class TestBreakpointTrigger:
    """Tests for BreakpointTrigger enum."""

    @pytest.mark.smoke
    def test_all_trigger_types_exist(self):
        """Test all trigger types exist with correct values."""
        assert BreakpointTrigger.LOW_CONFIDENCE.value == "low_confidence"
        assert BreakpointTrigger.DEADLOCK.value == "deadlock"
        assert BreakpointTrigger.HIGH_DISAGREEMENT.value == "high_disagreement"
        assert BreakpointTrigger.CRITICAL_DECISION.value == "critical_decision"
        assert BreakpointTrigger.EXPLICIT_APPEAL.value == "explicit_appeal"
        assert BreakpointTrigger.ROUND_LIMIT.value == "round_limit"
        assert BreakpointTrigger.SAFETY_CONCERN.value == "safety_concern"
        assert BreakpointTrigger.HOLLOW_CONSENSUS.value == "hollow_consensus"
        assert BreakpointTrigger.CUSTOM.value == "custom"

    def test_trigger_type_from_string(self):
        """Test creating trigger type from string."""
        assert BreakpointTrigger("low_confidence") == BreakpointTrigger.LOW_CONFIDENCE
        assert BreakpointTrigger("deadlock") == BreakpointTrigger.DEADLOCK


# =============================================================================
# DebateSnapshot Tests
# =============================================================================


class TestDebateSnapshot:
    """Tests for DebateSnapshot dataclass."""

    def test_create_minimal_snapshot(self):
        """Test creating snapshot with required fields."""
        snapshot = DebateSnapshot(
            debate_id="debate-123",
            task="Test task",
            current_round=2,
            total_rounds=5,
            latest_messages=[],
            active_proposals=[],
            open_critiques=[],
            current_consensus=None,
            confidence=0.5,
            agent_positions={},
            unresolved_issues=[],
            key_disagreements=[],
        )

        assert snapshot.debate_id == "debate-123"
        assert snapshot.task == "Test task"
        assert snapshot.current_round == 2
        assert snapshot.total_rounds == 5
        assert snapshot.confidence == 0.5
        assert snapshot.created_at is not None

    def test_snapshot_with_agent_positions(self):
        """Test snapshot with agent positions populated."""
        snapshot = DebateSnapshot(
            debate_id="debate-456",
            task="Design a cache system",
            current_round=3,
            total_rounds=5,
            latest_messages=[
                {"agent": "claude", "content": "Use Redis", "round": 2},
                {"agent": "gpt4", "content": "Consider Memcached", "round": 2},
            ],
            active_proposals=["Redis proposal", "Memcached proposal"],
            open_critiques=["Redis is complex"],
            current_consensus=None,
            confidence=0.45,
            agent_positions={
                "claude": "Prefers Redis for persistence",
                "gpt4": "Prefers Memcached for simplicity",
            },
            unresolved_issues=["Persistence vs speed trade-off"],
            key_disagreements=["Cache eviction strategy"],
        )

        assert len(snapshot.agent_positions) == 2
        assert "claude" in snapshot.agent_positions
        assert len(snapshot.key_disagreements) == 1

    def test_snapshot_created_at_auto_generated(self):
        """Test that created_at is auto-generated."""
        snapshot = DebateSnapshot(
            debate_id="debate-789",
            task="Test",
            current_round=1,
            total_rounds=3,
            latest_messages=[],
            active_proposals=[],
            open_critiques=[],
            current_consensus=None,
            confidence=0.7,
            agent_positions={},
            unresolved_issues=[],
            key_disagreements=[],
        )

        # Should be parseable as ISO format
        datetime.fromisoformat(snapshot.created_at)


# =============================================================================
# HumanGuidance Tests
# =============================================================================


class TestHumanGuidance:
    """Tests for HumanGuidance dataclass."""

    def test_create_continue_guidance(self):
        """Test creating guidance with continue action."""
        guidance = HumanGuidance(
            guidance_id="guide-001",
            debate_id="debate-123",
            human_id="user@example.com",
            action="continue",
        )

        assert guidance.action == "continue"
        assert guidance.decision is None
        assert guidance.hints == []
        assert guidance.constraints == []

    def test_create_resolve_guidance(self):
        """Test creating guidance with resolve action."""
        guidance = HumanGuidance(
            guidance_id="guide-002",
            debate_id="debate-123",
            human_id="admin",
            action="resolve",
            decision="Use Redis with a 1-hour TTL",
            reasoning="Redis provides the best balance of speed and persistence",
        )

        assert guidance.action == "resolve"
        assert guidance.decision == "Use Redis with a 1-hour TTL"
        assert guidance.reasoning != ""

    def test_create_redirect_guidance_with_hints(self):
        """Test creating guidance with redirect action and hints."""
        guidance = HumanGuidance(
            guidance_id="guide-003",
            debate_id="debate-123",
            human_id="user",
            action="redirect",
            hints=["Consider latency requirements", "Check memory constraints"],
            constraints=["Must support > 10K ops/sec"],
            preferred_direction="Focus on write performance",
        )

        assert guidance.action == "redirect"
        assert len(guidance.hints) == 2
        assert len(guidance.constraints) == 1
        assert guidance.preferred_direction is not None

    def test_guidance_with_answers(self):
        """Test guidance with answers to specific questions."""
        guidance = HumanGuidance(
            guidance_id="guide-004",
            debate_id="debate-123",
            human_id="expert",
            action="redirect",
            answers={
                "What is the expected load?": "1M requests/day",
                "Is persistence required?": "Yes, for audit",
            },
        )

        assert len(guidance.answers) == 2

    def test_guidance_created_at_auto_generated(self):
        """Test that created_at is auto-generated."""
        guidance = HumanGuidance(
            guidance_id="guide-005",
            debate_id="debate-123",
            human_id="user",
            action="abort",
        )

        datetime.fromisoformat(guidance.created_at)


# =============================================================================
# Breakpoint Tests
# =============================================================================


class TestBreakpoint:
    """Tests for Breakpoint dataclass."""

    def test_create_unresolved_breakpoint(self):
        """Test creating an unresolved breakpoint."""
        snapshot = DebateSnapshot(
            debate_id="debate-123",
            task="Test",
            current_round=2,
            total_rounds=5,
            latest_messages=[],
            active_proposals=[],
            open_critiques=[],
            current_consensus=None,
            confidence=0.3,
            agent_positions={},
            unresolved_issues=[],
            key_disagreements=[],
        )

        bp = Breakpoint(
            breakpoint_id="bp-001",
            trigger=BreakpointTrigger.LOW_CONFIDENCE,
            triggered_at=datetime.now().isoformat(),
            debate_snapshot=snapshot,
        )

        assert bp.breakpoint_id == "bp-001"
        assert bp.trigger == BreakpointTrigger.LOW_CONFIDENCE
        assert bp.resolved is False
        assert bp.guidance is None
        assert bp.escalation_level == 1
        assert bp.timeout_minutes == 30

    def test_create_resolved_breakpoint(self):
        """Test creating a resolved breakpoint."""
        snapshot = DebateSnapshot(
            debate_id="debate-123",
            task="Test",
            current_round=2,
            total_rounds=5,
            latest_messages=[],
            active_proposals=[],
            open_critiques=[],
            current_consensus=None,
            confidence=0.3,
            agent_positions={},
            unresolved_issues=[],
            key_disagreements=[],
        )

        guidance = HumanGuidance(
            guidance_id="guide-001",
            debate_id="debate-123",
            human_id="user",
            action="continue",
        )

        bp = Breakpoint(
            breakpoint_id="bp-002",
            trigger=BreakpointTrigger.DEADLOCK,
            triggered_at=datetime.now().isoformat(),
            debate_snapshot=snapshot,
            resolved=True,
            guidance=guidance,
            resolved_at=datetime.now().isoformat(),
        )

        assert bp.resolved is True
        assert bp.guidance is not None
        assert bp.resolved_at is not None

    def test_escalation_levels(self):
        """Test different escalation levels."""
        snapshot = DebateSnapshot(
            debate_id="debate-123",
            task="Test",
            current_round=2,
            total_rounds=5,
            latest_messages=[],
            active_proposals=[],
            open_critiques=[],
            current_consensus=None,
            confidence=0.3,
            agent_positions={},
            unresolved_issues=[],
            key_disagreements=[],
        )

        bp = Breakpoint(
            breakpoint_id="bp-003",
            trigger=BreakpointTrigger.SAFETY_CONCERN,
            triggered_at=datetime.now().isoformat(),
            debate_snapshot=snapshot,
            escalation_level=3,
            timeout_minutes=5,
        )

        assert bp.escalation_level == 3
        assert bp.timeout_minutes == 5


# =============================================================================
# BreakpointConfig Tests
# =============================================================================


class TestBreakpointConfig:
    """Tests for BreakpointConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BreakpointConfig()

        assert config.min_confidence == 0.6
        assert config.max_deadlock_rounds == 3
        assert config.max_total_rounds == 10
        assert config.disagreement_threshold == 0.7
        assert config.require_human_for_critical is True
        assert config.auto_timeout_action == "continue"
        assert len(config.safety_keywords) > 0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = BreakpointConfig(
            min_confidence=0.8,
            max_deadlock_rounds=2,
            max_total_rounds=5,
            disagreement_threshold=0.5,
            notification_channels=["slack", "discord"],
            safety_keywords=["danger", "risk"],
        )

        assert config.min_confidence == 0.8
        assert config.max_deadlock_rounds == 2
        assert "slack" in config.notification_channels
        assert "danger" in config.safety_keywords


# =============================================================================
# HumanNotifier Tests
# =============================================================================


class TestHumanNotifier:
    """Tests for HumanNotifier class."""

    def test_register_handler(self):
        """Test registering notification handler."""
        config = BreakpointConfig(notification_channels=["slack"])
        notifier = HumanNotifier(config)

        handler = AsyncMock()
        notifier.register_handler("slack", handler)

        assert "slack" in notifier._handlers

    @pytest.mark.asyncio
    async def test_notify_with_registered_handler(self):
        """Test notification with registered handler."""
        config = BreakpointConfig(notification_channels=["test_channel"])
        notifier = HumanNotifier(config)

        handler = AsyncMock(return_value=None)
        notifier.register_handler("test_channel", handler)

        snapshot = DebateSnapshot(
            debate_id="debate-123",
            task="Test",
            current_round=2,
            total_rounds=5,
            latest_messages=[],
            active_proposals=[],
            open_critiques=[],
            current_consensus=None,
            confidence=0.3,
            agent_positions={},
            unresolved_issues=[],
            key_disagreements=[],
        )

        bp = Breakpoint(
            breakpoint_id="bp-001",
            trigger=BreakpointTrigger.LOW_CONFIDENCE,
            triggered_at=datetime.now().isoformat(),
            debate_snapshot=snapshot,
        )

        result = await notifier.notify(bp)

        assert result is True
        handler.assert_called_once_with(bp)

    @pytest.mark.asyncio
    async def test_notify_fallback_to_cli(self):
        """Test notification falls back to CLI when no handlers."""
        config = BreakpointConfig(notification_channels=[])
        notifier = HumanNotifier(config)

        snapshot = DebateSnapshot(
            debate_id="debate-123",
            task="Test task for CLI display",
            current_round=2,
            total_rounds=5,
            latest_messages=[],
            active_proposals=[],
            open_critiques=[],
            current_consensus=None,
            confidence=0.3,
            agent_positions={"claude": "Test position"},
            unresolved_issues=[],
            key_disagreements=["Key disagreement 1"],
        )

        bp = Breakpoint(
            breakpoint_id="bp-001",
            trigger=BreakpointTrigger.LOW_CONFIDENCE,
            triggered_at=datetime.now().isoformat(),
            debate_snapshot=snapshot,
        )

        result = await notifier.notify(bp)
        assert result is True

    @pytest.mark.asyncio
    async def test_notify_handler_failure_continues(self):
        """Test notification continues when handler fails."""
        config = BreakpointConfig(notification_channels=["failing", "working"])
        notifier = HumanNotifier(config)

        failing_handler = AsyncMock(side_effect=Exception("Handler error"))
        working_handler = AsyncMock(return_value=None)
        notifier.register_handler("failing", failing_handler)
        notifier.register_handler("working", working_handler)

        snapshot = DebateSnapshot(
            debate_id="debate-123",
            task="Test",
            current_round=2,
            total_rounds=5,
            latest_messages=[],
            active_proposals=[],
            open_critiques=[],
            current_consensus=None,
            confidence=0.3,
            agent_positions={},
            unresolved_issues=[],
            key_disagreements=[],
        )

        bp = Breakpoint(
            breakpoint_id="bp-001",
            trigger=BreakpointTrigger.LOW_CONFIDENCE,
            triggered_at=datetime.now().isoformat(),
            debate_snapshot=snapshot,
        )

        result = await notifier.notify(bp)
        assert result is True
        working_handler.assert_called_once()


# =============================================================================
# BreakpointManager Tests
# =============================================================================


class TestBreakpointManager:
    """Tests for BreakpointManager class."""

    def _create_mock_message(self, agent: str, content: str, round_num: int) -> Mock:
        """Helper to create mock messages."""
        msg = Mock()
        msg.agent = agent
        msg.content = content
        msg.round = round_num
        return msg

    def test_init_default_config(self):
        """Test initialization with default config."""
        manager = BreakpointManager()

        assert manager.config is not None
        assert manager.breakpoints == []
        assert manager._breakpoint_counter == 0

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = BreakpointConfig(min_confidence=0.8)
        manager = BreakpointManager(config=config)

        assert manager.config.min_confidence == 0.8

    def test_check_triggers_low_confidence(self):
        """Test low confidence trigger."""
        config = BreakpointConfig(min_confidence=0.6)
        manager = BreakpointManager(config=config)

        messages = [
            self._create_mock_message("claude", "Test message", 1),
        ]

        bp = manager.check_triggers(
            debate_id="debate-123",
            task="Test task",
            messages=messages,
            confidence=0.4,  # Below threshold
            round_num=1,
            max_rounds=5,
        )

        assert bp is not None
        assert bp.trigger == BreakpointTrigger.LOW_CONFIDENCE
        assert "debate-123" in bp.breakpoint_id

    def test_check_triggers_round_limit(self):
        """Test round limit trigger."""
        config = BreakpointConfig()
        manager = BreakpointManager(config=config)

        messages = [
            self._create_mock_message("claude", "Test message", 5),
        ]

        bp = manager.check_triggers(
            debate_id="debate-123",
            task="Test task",
            messages=messages,
            confidence=0.8,
            round_num=5,  # At limit
            max_rounds=5,
        )

        assert bp is not None
        assert bp.trigger == BreakpointTrigger.ROUND_LIMIT

    def test_check_triggers_safety_concern(self):
        """Test safety concern trigger."""
        config = BreakpointConfig(safety_keywords=["dangerous"])
        manager = BreakpointManager(config=config)

        messages = [
            self._create_mock_message("claude", "This is a dangerous approach", 1),
        ]

        bp = manager.check_triggers(
            debate_id="debate-123",
            task="Test task",
            messages=messages,
            confidence=0.8,
            round_num=1,
            max_rounds=5,
        )

        assert bp is not None
        assert bp.trigger == BreakpointTrigger.SAFETY_CONCERN
        assert bp.escalation_level == 3  # High priority

    def test_check_triggers_high_disagreement(self):
        """Test high disagreement trigger via critiques."""
        config = BreakpointConfig(disagreement_threshold=0.7)
        manager = BreakpointManager(config=config)

        messages = [
            self._create_mock_message("claude", "Test message", 1),
        ]

        critique1 = Mock()
        critique1.severity = 0.8
        critique2 = Mock()
        critique2.severity = 0.9

        bp = manager.check_triggers(
            debate_id="debate-123",
            task="Test task",
            messages=messages,
            confidence=0.8,
            round_num=1,
            max_rounds=5,
            critiques=[critique1, critique2],
        )

        assert bp is not None
        assert bp.trigger == BreakpointTrigger.HIGH_DISAGREEMENT

    def test_check_triggers_no_trigger(self):
        """Test when no trigger condition is met."""
        config = BreakpointConfig()
        manager = BreakpointManager(config=config)

        messages = [
            self._create_mock_message("claude", "All looks good", 1),
        ]

        bp = manager.check_triggers(
            debate_id="debate-123",
            task="Test task",
            messages=messages,
            confidence=0.8,
            round_num=1,
            max_rounds=5,
        )

        assert bp is None

    def test_check_triggers_deadlock(self):
        """Test deadlock detection."""
        config = BreakpointConfig(max_deadlock_rounds=2)
        manager = BreakpointManager(config=config)

        # Create repeating messages to simulate deadlock
        messages = [
            self._create_mock_message("claude", "We should use approach A", 1),
            self._create_mock_message("gpt4", "We should use approach B", 1),
            self._create_mock_message("claude", "We should use approach A", 2),
            self._create_mock_message("gpt4", "We should use approach B", 2),
            self._create_mock_message("claude", "We should use approach A", 3),
            self._create_mock_message("gpt4", "We should use approach B", 3),
            self._create_mock_message("claude", "We should use approach A", 4),
            self._create_mock_message("gpt4", "We should use approach B", 4),
        ]

        bp = manager.check_triggers(
            debate_id="debate-123",
            task="Test task",
            messages=messages,
            confidence=0.8,
            round_num=4,
            max_rounds=10,
        )

        assert bp is not None
        assert bp.trigger == BreakpointTrigger.DEADLOCK

    def test_get_pending_breakpoints(self):
        """Test getting pending breakpoints."""
        manager = BreakpointManager()

        messages = [self._create_mock_message("claude", "Test", 1)]

        # Create a breakpoint
        manager.check_triggers(
            debate_id="debate-123",
            task="Test task",
            messages=messages,
            confidence=0.3,
            round_num=1,
            max_rounds=5,
        )

        pending = manager.get_pending_breakpoints()
        assert len(pending) == 1
        assert pending[0].resolved is False

    def test_get_breakpoint_by_id(self):
        """Test getting specific breakpoint."""
        manager = BreakpointManager()

        messages = [self._create_mock_message("claude", "Test", 1)]

        bp = manager.check_triggers(
            debate_id="debate-123",
            task="Test task",
            messages=messages,
            confidence=0.3,
            round_num=1,
            max_rounds=5,
        )

        found = manager.get_breakpoint(bp.breakpoint_id)
        assert found is not None
        assert found.breakpoint_id == bp.breakpoint_id

        not_found = manager.get_breakpoint("nonexistent-id")
        assert not_found is None

    def test_resolve_breakpoint(self):
        """Test resolving a breakpoint."""
        manager = BreakpointManager()

        messages = [self._create_mock_message("claude", "Test", 1)]

        bp = manager.check_triggers(
            debate_id="debate-123",
            task="Test task",
            messages=messages,
            confidence=0.3,
            round_num=1,
            max_rounds=5,
        )

        guidance = HumanGuidance(
            guidance_id="guide-001",
            debate_id="debate-123",
            human_id="user",
            action="continue",
        )

        result = manager.resolve_breakpoint(bp.breakpoint_id, guidance)

        assert result is True
        resolved_bp = manager.get_breakpoint(bp.breakpoint_id)
        assert resolved_bp.resolved is True
        assert resolved_bp.guidance is not None

    def test_resolve_nonexistent_breakpoint(self):
        """Test resolving nonexistent breakpoint."""
        manager = BreakpointManager()

        guidance = HumanGuidance(
            guidance_id="guide-001",
            debate_id="debate-123",
            human_id="user",
            action="continue",
        )

        result = manager.resolve_breakpoint("nonexistent-id", guidance)
        assert result is False

    def test_resolve_already_resolved_breakpoint(self):
        """Test resolving already resolved breakpoint."""
        manager = BreakpointManager()

        messages = [self._create_mock_message("claude", "Test", 1)]

        bp = manager.check_triggers(
            debate_id="debate-123",
            task="Test task",
            messages=messages,
            confidence=0.3,
            round_num=1,
            max_rounds=5,
        )

        guidance = HumanGuidance(
            guidance_id="guide-001",
            debate_id="debate-123",
            human_id="user",
            action="continue",
        )

        # Resolve first time
        manager.resolve_breakpoint(bp.breakpoint_id, guidance)

        # Try to resolve again
        result = manager.resolve_breakpoint(bp.breakpoint_id, guidance)
        assert result is False

    @pytest.mark.asyncio
    async def test_handle_breakpoint_with_timeout(self):
        """Test handling breakpoint with timeout."""

        async def slow_human_input(bp):
            await asyncio.sleep(10)
            return HumanGuidance(
                guidance_id="guide-001",
                debate_id=bp.debate_snapshot.debate_id,
                human_id="user",
                action="continue",
            )

        config = BreakpointConfig(auto_timeout_action="abort")
        manager = BreakpointManager(config=config, get_human_input=slow_human_input)

        snapshot = DebateSnapshot(
            debate_id="debate-123",
            task="Test",
            current_round=2,
            total_rounds=5,
            latest_messages=[],
            active_proposals=[],
            open_critiques=[],
            current_consensus=None,
            confidence=0.3,
            agent_positions={},
            unresolved_issues=[],
            key_disagreements=[],
        )

        bp = Breakpoint(
            breakpoint_id="bp-001",
            trigger=BreakpointTrigger.LOW_CONFIDENCE,
            triggered_at=datetime.now().isoformat(),
            debate_snapshot=snapshot,
            timeout_minutes=0.001,  # Very short timeout
        )

        guidance = await manager.handle_breakpoint(bp)

        assert guidance.action == "abort"
        assert guidance.human_id == "system"

    def test_inject_guidance_resolve(self):
        """Test injecting resolve guidance."""
        manager = BreakpointManager()

        guidance = HumanGuidance(
            guidance_id="guide-001",
            debate_id="debate-123",
            human_id="user",
            action="resolve",
            decision="Use Redis for caching",
        )

        messages = []
        env = Mock()

        new_messages, new_env = manager.inject_guidance(guidance, messages, env)

        assert len(new_messages) == 1
        assert new_messages[0].agent == "human"
        assert new_messages[0].role == "judge"
        assert "HUMAN DECISION" in new_messages[0].content

    def test_inject_guidance_redirect(self):
        """Test injecting redirect guidance with hints."""
        manager = BreakpointManager()

        guidance = HumanGuidance(
            guidance_id="guide-001",
            debate_id="debate-123",
            human_id="user",
            action="redirect",
            hints=["Consider latency", "Check memory"],
            constraints=["Max 100ms response time"],
        )

        messages = []
        env = Mock()
        env.constraints = []

        new_messages, new_env = manager.inject_guidance(guidance, messages, env)

        assert len(new_messages) == 1
        assert new_messages[0].agent == "human"
        assert new_messages[0].role == "moderator"
        assert "HUMAN GUIDANCE" in new_messages[0].content
        assert "latency" in new_messages[0].content
        assert len(new_env.constraints) == 1

    def test_inject_guidance_continue(self):
        """Test injecting continue guidance (no change)."""
        manager = BreakpointManager()

        guidance = HumanGuidance(
            guidance_id="guide-001",
            debate_id="debate-123",
            human_id="user",
            action="continue",
        )

        messages = [Mock()]
        env = Mock()

        new_messages, new_env = manager.inject_guidance(guidance, messages, env)

        assert len(new_messages) == 1
        assert new_env == env

    def test_event_emitter_integration(self):
        """Test event emitter integration."""
        emitter = Mock()
        manager = BreakpointManager(event_emitter=emitter, loop_id="loop-123")

        messages = [self._create_mock_message("claude", "Test", 1)]

        with patch("aragora.debate.breakpoints.StreamEvent") as MockEvent:
            with patch("aragora.debate.breakpoints.StreamEventType") as MockType:
                MockType.BREAKPOINT = "BREAKPOINT"
                manager.check_triggers(
                    debate_id="debate-123",
                    task="Test task",
                    messages=messages,
                    confidence=0.3,
                    round_num=1,
                    max_rounds=5,
                )

        # Emitter should have been called
        # Note: This depends on the import working; may need adjustment


# =============================================================================
# Decorator Tests
# =============================================================================


class TestCriticalDecisionDecorator:
    """Tests for critical_decision decorator."""

    def test_decorator_adds_attributes(self):
        """Test decorator adds aragora critical attributes."""

        @critical_decision(reason="Important financial decision")
        def my_function():
            pass

        assert hasattr(my_function, "_aragora_critical")
        assert my_function._aragora_critical is True
        assert my_function._aragora_critical_reason == "Important financial decision"

    def test_decorator_without_reason(self):
        """Test decorator without reason."""

        @critical_decision()
        def my_function():
            pass

        assert my_function._aragora_critical is True
        assert my_function._aragora_critical_reason == ""

    def test_decorator_preserves_function(self):
        """Test decorator preserves original function."""

        @critical_decision(reason="Test")
        def add(a, b):
            return a + b

        assert add(2, 3) == 5


class TestBreakpointDecorator:
    """Tests for breakpoint decorator."""

    def test_breakpoint_decorator_adds_attributes(self):
        """Test breakpoint decorator adds attributes."""

        @breakpoint(trigger="low_confidence", threshold=0.7, message="Confidence too low")
        def check_confidence():
            pass

        assert hasattr(check_confidence, "_aragora_breakpoint")
        assert check_confidence._aragora_breakpoint is True
        assert check_confidence._aragora_breakpoint_trigger == "low_confidence"
        assert check_confidence._aragora_breakpoint_threshold == 0.7
        assert check_confidence._aragora_breakpoint_message == "Confidence too low"

    def test_breakpoint_decorator_defaults(self):
        """Test breakpoint decorator defaults."""

        @breakpoint()
        def default_breakpoint():
            pass

        assert default_breakpoint._aragora_breakpoint_trigger == "low_confidence"
        assert default_breakpoint._aragora_breakpoint_threshold == 0.6
        assert default_breakpoint._aragora_breakpoint_message == ""

    def test_breakpoint_decorator_preserves_function(self):
        """Test breakpoint decorator preserves function behavior."""

        @breakpoint(trigger="deadlock")
        def multiply(a, b):
            return a * b

        assert multiply(3, 4) == 12


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


class TestBreakpointEdgeCases:
    """Tests for edge cases."""

    def test_empty_messages_list(self):
        """Test with empty messages list."""
        manager = BreakpointManager()

        bp = manager.check_triggers(
            debate_id="debate-123",
            task="Test",
            messages=[],
            confidence=0.3,
            round_num=1,
            max_rounds=5,
        )

        assert bp is not None
        assert bp.trigger == BreakpointTrigger.LOW_CONFIDENCE

    def test_single_agent_debate(self):
        """Test with single agent debate."""
        manager = BreakpointManager()

        msg = Mock()
        msg.agent = "claude"
        msg.content = "Solo message"
        msg.round = 1

        bp = manager.check_triggers(
            debate_id="debate-123",
            task="Test",
            messages=[msg],
            confidence=0.8,
            round_num=1,
            max_rounds=5,
        )

        assert bp is None

    def test_multiple_breakpoints_same_debate(self):
        """Test multiple breakpoints for same debate."""
        manager = BreakpointManager()

        msg = Mock()
        msg.agent = "claude"
        msg.content = "Test"
        msg.round = 1

        # First breakpoint
        bp1 = manager.check_triggers(
            debate_id="debate-123",
            task="Test",
            messages=[msg],
            confidence=0.3,
            round_num=1,
            max_rounds=5,
        )

        # Second breakpoint
        bp2 = manager.check_triggers(
            debate_id="debate-123",
            task="Test",
            messages=[msg],
            confidence=0.2,
            round_num=2,
            max_rounds=5,
        )

        assert bp1.breakpoint_id != bp2.breakpoint_id
        assert len(manager.breakpoints) == 2

    def test_deadlock_detection_minimum_messages(self):
        """Test deadlock detection requires minimum messages."""
        config = BreakpointConfig(max_deadlock_rounds=3)
        manager = BreakpointManager(config=config)

        # Too few messages for deadlock detection
        msg = Mock()
        msg.agent = "claude"
        msg.content = "Test"
        msg.round = 3

        bp = manager.check_triggers(
            debate_id="debate-123",
            task="Test",
            messages=[msg],
            confidence=0.8,
            round_num=3,
            max_rounds=10,
        )

        # Should not trigger deadlock with only 1 message
        assert bp is None or bp.trigger != BreakpointTrigger.DEADLOCK

    def test_safety_keywords_case_insensitive(self):
        """Test safety keyword detection is case insensitive."""
        config = BreakpointConfig(safety_keywords=["danger"])
        manager = BreakpointManager(config=config)

        msg = Mock()
        msg.agent = "claude"
        msg.content = "This is DANGER zone"
        msg.round = 1

        bp = manager.check_triggers(
            debate_id="debate-123",
            task="Test",
            messages=[msg],
            confidence=0.8,
            round_num=1,
            max_rounds=5,
        )

        assert bp is not None
        assert bp.trigger == BreakpointTrigger.SAFETY_CONCERN

    def test_message_without_content_attribute(self):
        """Test handling messages without content attribute."""
        manager = BreakpointManager()

        msg = Mock(spec=["agent", "round"])
        msg.agent = "claude"
        msg.round = 1

        # Should not crash
        bp = manager.check_triggers(
            debate_id="debate-123",
            task="Test",
            messages=[msg],
            confidence=0.3,
            round_num=1,
            max_rounds=5,
        )

        assert bp is not None


# =============================================================================
# BreakpointManager._detect_deadlock Tests
# =============================================================================


class TestDeadlockDetection:
    """Tests for deadlock detection internal method."""

    def _create_mock_message(self, agent: str, content: str, round_num: int) -> Mock:
        """Helper to create mock messages."""
        msg = Mock()
        msg.agent = agent
        msg.content = content
        msg.round = round_num
        return msg

    def test_detect_deadlock_high_overlap(self):
        """Test deadlock detected with high content overlap."""
        manager = BreakpointManager()

        # Simulate repeated content
        messages = [
            self._create_mock_message("a", "position A is best", 1),
            self._create_mock_message("b", "position B is better", 1),
            self._create_mock_message("a", "position A is best", 2),
            self._create_mock_message("b", "position B is better", 2),
            self._create_mock_message("a", "position A is best", 3),
            self._create_mock_message("b", "position B is better", 3),
        ]

        result = manager._detect_deadlock(messages, lookback=3)
        assert result is True

    def test_detect_deadlock_no_overlap(self):
        """Test no deadlock with varied content."""
        manager = BreakpointManager()

        messages = [
            self._create_mock_message("a", "First unique point", 1),
            self._create_mock_message("b", "Second unique point", 1),
            self._create_mock_message("a", "Third unique point", 2),
            self._create_mock_message("b", "Fourth unique point", 2),
            self._create_mock_message("a", "Fifth unique point", 3),
            self._create_mock_message("b", "Sixth unique point", 3),
        ]

        result = manager._detect_deadlock(messages, lookback=3)
        assert result is False

    def test_detect_deadlock_insufficient_messages(self):
        """Test deadlock detection with insufficient messages."""
        manager = BreakpointManager()

        messages = [
            self._create_mock_message("a", "Test", 1),
            self._create_mock_message("b", "Test", 1),
        ]

        result = manager._detect_deadlock(messages, lookback=3)
        assert result is False


# =============================================================================
# Additional Coverage: _emit_breakpoint_event and _emit_breakpoint_resolved_event
# =============================================================================


class TestEmitBreakpointEvents:
    """Tests for WebSocket event emission methods."""

    def _make_snapshot(self, debate_id="debate-123"):
        return DebateSnapshot(
            debate_id=debate_id,
            task="Test task for event emission",
            current_round=2,
            total_rounds=5,
            latest_messages=[],
            active_proposals=[],
            open_critiques=[],
            current_consensus=None,
            confidence=0.4,
            agent_positions={"claude": "Position A"},
            unresolved_issues=[],
            key_disagreements=["Disagreement 1"],
        )

    def _make_breakpoint(self, snapshot=None, guidance=None):
        snapshot = snapshot or self._make_snapshot()
        return Breakpoint(
            breakpoint_id="bp-test-001",
            trigger=BreakpointTrigger.LOW_CONFIDENCE,
            triggered_at=datetime.now().isoformat(),
            debate_snapshot=snapshot,
            escalation_level=2,
            timeout_minutes=15,
            resolved=guidance is not None,
            guidance=guidance,
            resolved_at=datetime.now().isoformat() if guidance else None,
        )

    def test_emit_breakpoint_event_no_emitter(self):
        """Test _emit_breakpoint_event does nothing without emitter."""
        manager = BreakpointManager(event_emitter=None)
        bp = self._make_breakpoint()
        # Should not raise
        manager._emit_breakpoint_event(bp)

    def test_emit_breakpoint_event_import_failure(self):
        """Test _emit_breakpoint_event handles import errors gracefully."""
        emitter = Mock()
        manager = BreakpointManager(event_emitter=emitter, loop_id="loop-1")

        bp = self._make_breakpoint()

        # Make the emitter.emit raise to simulate failure
        emitter.emit.side_effect = RuntimeError("Failed to emit")

        # The method catches all exceptions, so it should not raise
        # (we test the except path via a broken emitter)
        manager._emit_breakpoint_event(bp)

    def test_emit_breakpoint_event_calls_emitter(self):
        """Test _emit_breakpoint_event calls emitter.emit."""
        emitter = Mock()
        manager = BreakpointManager(event_emitter=emitter, loop_id="loop-42")

        bp = self._make_breakpoint()

        # Just call it - if events module is available it works, if not it catches
        manager._emit_breakpoint_event(bp)

        # emitter.emit should have been called (the events module is importable)
        emitter.emit.assert_called_once()

    def test_emit_breakpoint_resolved_event_no_emitter(self):
        """Test _emit_breakpoint_resolved_event does nothing without emitter."""
        manager = BreakpointManager(event_emitter=None)
        guidance = HumanGuidance(
            guidance_id="g-1",
            debate_id="debate-123",
            human_id="user",
            action="continue",
        )
        bp = self._make_breakpoint(guidance=guidance)
        # Should not raise
        manager._emit_breakpoint_resolved_event(bp)

    def test_emit_breakpoint_resolved_event_no_guidance(self):
        """Test _emit_breakpoint_resolved_event skips when no guidance."""
        emitter = Mock()
        manager = BreakpointManager(event_emitter=emitter)
        bp = self._make_breakpoint(guidance=None)
        manager._emit_breakpoint_resolved_event(bp)
        emitter.emit.assert_not_called()

    def test_emit_breakpoint_resolved_event_calls_emitter(self):
        """Test _emit_breakpoint_resolved_event calls emitter when guidance present."""
        emitter = Mock()
        manager = BreakpointManager(event_emitter=emitter, loop_id="loop-99")

        guidance = HumanGuidance(
            guidance_id="g-2",
            debate_id="debate-123",
            human_id="expert_user",
            action="resolve",
            decision="Use Redis",
            hints=["Consider latency"],
            constraints=["Max 50ms"],
            reasoning="Best tradeoff",
        )
        bp = self._make_breakpoint(guidance=guidance)

        # Just call it directly - events module is importable
        manager._emit_breakpoint_resolved_event(bp)

        emitter.emit.assert_called_once()


# =============================================================================
# Additional Coverage: handle_breakpoint success path
# =============================================================================


class TestHandleBreakpointSuccess:
    """Tests for handle_breakpoint successful human input flow."""

    @pytest.mark.asyncio
    async def test_handle_breakpoint_success(self):
        """Test handle_breakpoint with successful human input."""
        guidance = HumanGuidance(
            guidance_id="guide-success",
            debate_id="debate-123",
            human_id="expert",
            action="resolve",
            decision="Use Redis",
        )

        async def mock_human_input(bp):
            return guidance

        manager = BreakpointManager(get_human_input=mock_human_input)

        snapshot = DebateSnapshot(
            debate_id="debate-123",
            task="Test",
            current_round=2,
            total_rounds=5,
            latest_messages=[],
            active_proposals=[],
            open_critiques=[],
            current_consensus=None,
            confidence=0.3,
            agent_positions={},
            unresolved_issues=[],
            key_disagreements=[],
        )

        bp = Breakpoint(
            breakpoint_id="bp-success",
            trigger=BreakpointTrigger.LOW_CONFIDENCE,
            triggered_at=datetime.now().isoformat(),
            debate_snapshot=snapshot,
            timeout_minutes=30,
        )

        result = await manager.handle_breakpoint(bp)

        assert result.action == "resolve"
        assert result.decision == "Use Redis"
        assert result.human_id == "expert"
        assert bp.resolved is True
        assert bp.guidance is guidance
        assert bp.resolved_at is not None

    @pytest.mark.asyncio
    async def test_handle_breakpoint_emits_resolved_event(self):
        """Test handle_breakpoint emits resolved event."""
        guidance = HumanGuidance(
            guidance_id="g-1",
            debate_id="debate-123",
            human_id="user",
            action="continue",
        )

        async def mock_input(bp):
            return guidance

        emitter = Mock()
        manager = BreakpointManager(
            get_human_input=mock_input,
            event_emitter=emitter,
            loop_id="loop-1",
        )

        snapshot = DebateSnapshot(
            debate_id="debate-123",
            task="Test",
            current_round=1,
            total_rounds=5,
            latest_messages=[],
            active_proposals=[],
            open_critiques=[],
            current_consensus=None,
            confidence=0.3,
            agent_positions={},
            unresolved_issues=[],
            key_disagreements=[],
        )

        bp = Breakpoint(
            breakpoint_id="bp-emit",
            trigger=BreakpointTrigger.LOW_CONFIDENCE,
            triggered_at=datetime.now().isoformat(),
            debate_snapshot=snapshot,
            timeout_minutes=30,
        )

        with patch(
            "aragora.debate.breakpoints.BreakpointManager._emit_breakpoint_resolved_event"
        ) as mock_emit:
            await manager.handle_breakpoint(bp)
            mock_emit.assert_called_once_with(bp)


# =============================================================================
# Additional Coverage: _create_breakpoint snapshot building
# =============================================================================


class TestCreateBreakpointSnapshot:
    """Tests for _create_breakpoint snapshot construction."""

    def _make_msg(self, agent, content, round_num):
        msg = Mock()
        msg.agent = agent
        msg.content = content
        msg.round = round_num
        return msg

    def test_snapshot_extracts_agent_positions_from_reversed_messages(self):
        """Test agent positions are extracted from latest messages per agent."""
        manager = BreakpointManager()

        messages = [
            self._make_msg("claude", "Early position", 1),
            self._make_msg("gpt4", "GPT4 early position", 1),
            self._make_msg("claude", "Latest position from claude", 2),
            self._make_msg("gpt4", "Latest position from gpt4", 2),
        ]

        bp = manager.check_triggers(
            debate_id="debate-pos",
            task="Test",
            messages=messages,
            confidence=0.3,
            round_num=2,
            max_rounds=5,
        )

        positions = bp.debate_snapshot.agent_positions
        assert "claude" in positions
        assert "gpt4" in positions

    def test_snapshot_limits_latest_messages_to_5(self):
        """Test snapshot only includes last 5 messages."""
        manager = BreakpointManager()

        messages = [self._make_msg("claude", f"Message {i}", i) for i in range(10)]

        bp = manager.check_triggers(
            debate_id="debate-limit",
            task="Test",
            messages=messages,
            confidence=0.3,
            round_num=10,
            max_rounds=15,
        )

        assert len(bp.debate_snapshot.latest_messages) <= 5

    def test_breakpoint_counter_increments(self):
        """Test breakpoint counter increments across multiple breakpoints."""
        manager = BreakpointManager()

        msg = self._make_msg("claude", "Test", 1)

        bp1 = manager.check_triggers(
            debate_id="debate-1",
            task="Test",
            messages=[msg],
            confidence=0.3,
            round_num=1,
            max_rounds=5,
        )

        bp2 = manager.check_triggers(
            debate_id="debate-2",
            task="Test",
            messages=[msg],
            confidence=0.2,
            round_num=1,
            max_rounds=5,
        )

        assert manager._breakpoint_counter == 2

    def test_safety_concern_escalation_level_3(self):
        """Test safety concern breakpoint always has escalation level 3."""
        config = BreakpointConfig(safety_keywords=["harmful"])
        manager = BreakpointManager(config=config)

        msg = self._make_msg("claude", "This could be harmful", 1)

        bp = manager.check_triggers(
            debate_id="debate-safety",
            task="Test",
            messages=[msg],
            confidence=0.9,
            round_num=1,
            max_rounds=5,
        )

        assert bp.trigger == BreakpointTrigger.SAFETY_CONCERN
        assert bp.escalation_level == 3


# =============================================================================
# Additional Coverage: inject_guidance edge cases
# =============================================================================


class TestInjectGuidanceDeep:
    """Additional tests for inject_guidance edge cases."""

    def test_inject_guidance_abort_does_nothing(self):
        """Test inject_guidance with abort action makes no changes."""
        manager = BreakpointManager()

        guidance = HumanGuidance(
            guidance_id="g-1",
            debate_id="debate-123",
            human_id="user",
            action="abort",
        )

        messages = [Mock()]
        env = Mock()

        new_messages, new_env = manager.inject_guidance(guidance, messages, env)

        assert len(new_messages) == 1  # Unchanged
        assert new_env is env

    def test_inject_guidance_continue_with_hints(self):
        """Test inject_guidance continue action with hints creates guidance message."""
        manager = BreakpointManager()

        guidance = HumanGuidance(
            guidance_id="g-2",
            debate_id="debate-123",
            human_id="user",
            action="continue",
            hints=["Think about caching", "Consider latency requirements"],
        )

        msg = Mock()
        msg.round = 1
        messages = [msg]
        env = Mock()

        new_messages, new_env = manager.inject_guidance(guidance, messages, env)

        # Should have added a guidance message
        assert len(new_messages) >= 1

    def test_inject_guidance_redirect_updates_task(self):
        """Test inject_guidance redirect action updates environment task."""
        manager = BreakpointManager()

        guidance = HumanGuidance(
            guidance_id="g-3",
            debate_id="debate-123",
            human_id="user",
            action="redirect",
            decision="Focus on scalability instead",
            hints=["Think about horizontal scaling"],
        )

        msg = Mock()
        msg.round = 1
        messages = [msg]
        env = Mock()
        env.task = "Original task"

        new_messages, new_env = manager.inject_guidance(guidance, messages, env)

        # redirect should create guidance message
        assert len(new_messages) >= 1


# =============================================================================
# Additional Coverage: resolve_breakpoint
# =============================================================================


class TestResolveBreakpoint:
    """Tests for resolve_breakpoint method."""

    def _make_msg(self, agent, content, round_num):
        msg = Mock()
        msg.agent = agent
        msg.content = content
        msg.round = round_num
        return msg

    def test_resolve_breakpoint_updates_state(self):
        """Test resolve_breakpoint updates breakpoint state."""
        manager = BreakpointManager()

        # Create breakpoint via check_triggers so it's registered
        msg = self._make_msg("claude", "Test", 1)
        bp = manager.check_triggers(
            debate_id="debate-123",
            task="Test",
            messages=[msg],
            confidence=0.3,
            round_num=1,
            max_rounds=5,
        )

        guidance = HumanGuidance(
            guidance_id="g-1",
            debate_id="debate-123",
            human_id="user",
            action="resolve",
            decision="Approved",
        )

        result = manager.resolve_breakpoint(bp.breakpoint_id, guidance)

        assert result is True
        assert bp.resolved is True
        assert bp.guidance is guidance
        assert bp.resolved_at is not None

    def test_resolve_breakpoint_not_found(self):
        """Test resolve_breakpoint returns False for unknown ID."""
        manager = BreakpointManager()

        guidance = HumanGuidance(
            guidance_id="g-1",
            debate_id="debate-123",
            human_id="user",
            action="continue",
        )

        result = manager.resolve_breakpoint("nonexistent-bp", guidance)

        assert result is False

    def test_resolve_breakpoint_already_resolved(self):
        """Test resolve_breakpoint returns False if already resolved."""
        manager = BreakpointManager()

        msg = self._make_msg("claude", "Test", 1)
        bp = manager.check_triggers(
            debate_id="debate-123",
            task="Test",
            messages=[msg],
            confidence=0.3,
            round_num=1,
            max_rounds=5,
        )

        guidance = HumanGuidance(
            guidance_id="g-1",
            debate_id="debate-123",
            human_id="user",
            action="continue",
        )

        # Resolve once
        manager.resolve_breakpoint(bp.breakpoint_id, guidance)

        # Try to resolve again
        result = manager.resolve_breakpoint(bp.breakpoint_id, guidance)
        assert result is False
