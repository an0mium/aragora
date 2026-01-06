"""Tests for debate/breakpoints.py - Human intervention breakpoints."""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from aragora.debate.breakpoints import (
    BreakpointTrigger,
    DebateSnapshot,
    HumanGuidance,
    Breakpoint,
    BreakpointConfig,
    HumanNotifier,
    BreakpointManager,
    critical_decision,
    breakpoint,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_message():
    """Factory for creating mock message objects."""
    def _create(agent="claude", content="Test message", round_num=1):
        msg = MagicMock()
        msg.agent = agent
        msg.content = content
        msg.round = round_num
        return msg
    return _create


@pytest.fixture
def default_config():
    """Create default BreakpointConfig."""
    return BreakpointConfig()


@pytest.fixture
def custom_config():
    """Create custom BreakpointConfig for testing."""
    return BreakpointConfig(
        min_confidence=0.8,
        max_deadlock_rounds=2,
        max_total_rounds=5,
        disagreement_threshold=0.5,
        safety_keywords=["danger", "harm"],
    )


@pytest.fixture
def breakpoint_manager():
    """Create BreakpointManager with default config."""
    return BreakpointManager()


@pytest.fixture
def sample_snapshot():
    """Create a sample DebateSnapshot."""
    return DebateSnapshot(
        debate_id="test-debate-123",
        task="Test task description",
        current_round=2,
        total_rounds=5,
        latest_messages=[{"agent": "claude", "content": "Test", "round": 1}],
        active_proposals=["Proposal A"],
        open_critiques=["Critique 1"],
        current_consensus=None,
        confidence=0.7,
        agent_positions={"claude": "Position A", "gemini": "Position B"},
        unresolved_issues=["Issue 1"],
        key_disagreements=["Disagreement 1"],
    )


@pytest.fixture
def sample_breakpoint(sample_snapshot):
    """Create a sample Breakpoint."""
    return Breakpoint(
        breakpoint_id="bp-test-1",
        trigger=BreakpointTrigger.LOW_CONFIDENCE,
        triggered_at=datetime.now().isoformat(),
        debate_snapshot=sample_snapshot,
    )


@pytest.fixture
def sample_guidance():
    """Create sample HumanGuidance."""
    return HumanGuidance(
        guidance_id="guid-1",
        debate_id="test-debate-123",
        human_id="test-user",
        action="continue",
    )


# =============================================================================
# BreakpointTrigger Enum Tests
# =============================================================================

class TestBreakpointTriggerEnum:
    """Tests for BreakpointTrigger enum."""

    def test_all_trigger_values_exist(self):
        """All 8 trigger types should exist."""
        expected_triggers = [
            "LOW_CONFIDENCE",
            "DEADLOCK",
            "HIGH_DISAGREEMENT",
            "CRITICAL_DECISION",
            "EXPLICIT_APPEAL",
            "ROUND_LIMIT",
            "SAFETY_CONCERN",
            "CUSTOM",
        ]

        for trigger_name in expected_triggers:
            assert hasattr(BreakpointTrigger, trigger_name)

    def test_enum_values_are_strings(self):
        """All enum values should be strings."""
        for trigger in BreakpointTrigger:
            assert isinstance(trigger.value, str)


# =============================================================================
# DebateSnapshot Dataclass Tests
# =============================================================================

class TestDebateSnapshot:
    """Tests for DebateSnapshot dataclass."""

    def test_all_fields_initialized(self, sample_snapshot):
        """All 14 fields should be initialized correctly."""
        assert sample_snapshot.debate_id == "test-debate-123"
        assert sample_snapshot.task == "Test task description"
        assert sample_snapshot.current_round == 2
        assert sample_snapshot.total_rounds == 5
        assert len(sample_snapshot.latest_messages) == 1
        assert sample_snapshot.active_proposals == ["Proposal A"]
        assert sample_snapshot.open_critiques == ["Critique 1"]
        assert sample_snapshot.current_consensus is None
        assert sample_snapshot.confidence == 0.7
        assert "claude" in sample_snapshot.agent_positions
        assert sample_snapshot.unresolved_issues == ["Issue 1"]
        assert sample_snapshot.key_disagreements == ["Disagreement 1"]

    def test_created_at_auto_populated(self):
        """created_at should be auto-populated with current timestamp."""
        snapshot = DebateSnapshot(
            debate_id="test",
            task="task",
            current_round=1,
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

        # Should be a valid ISO timestamp
        assert snapshot.created_at is not None
        # Should be parseable
        datetime.fromisoformat(snapshot.created_at)

    def test_empty_lists_handled(self):
        """Empty lists should be accepted for all list fields."""
        snapshot = DebateSnapshot(
            debate_id="test",
            task="task",
            current_round=1,
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

        assert snapshot.latest_messages == []
        assert snapshot.active_proposals == []
        assert snapshot.open_critiques == []

    def test_agent_positions_dict(self):
        """agent_positions should store dict correctly."""
        positions = {"claude": "Position A", "gemini": "Position B", "gpt": "Position C"}
        snapshot = DebateSnapshot(
            debate_id="test",
            task="task",
            current_round=1,
            total_rounds=5,
            latest_messages=[],
            active_proposals=[],
            open_critiques=[],
            current_consensus=None,
            confidence=0.5,
            agent_positions=positions,
            unresolved_issues=[],
            key_disagreements=[],
        )

        assert len(snapshot.agent_positions) == 3
        assert snapshot.agent_positions["claude"] == "Position A"


# =============================================================================
# HumanGuidance Dataclass Tests
# =============================================================================

class TestHumanGuidance:
    """Tests for HumanGuidance dataclass."""

    def test_all_fields_initialized(self, sample_guidance):
        """All 9 fields should be initialized correctly."""
        assert sample_guidance.guidance_id == "guid-1"
        assert sample_guidance.debate_id == "test-debate-123"
        assert sample_guidance.human_id == "test-user"
        assert sample_guidance.action == "continue"

    def test_action_accepts_valid_values(self):
        """action should accept continue/resolve/redirect/abort."""
        for action in ["continue", "resolve", "redirect", "abort"]:
            guidance = HumanGuidance(
                guidance_id="g1",
                debate_id="d1",
                human_id="h1",
                action=action,
            )
            assert guidance.action == action

    def test_optional_fields_default(self):
        """Optional fields should default to None/empty."""
        guidance = HumanGuidance(
            guidance_id="g1",
            debate_id="d1",
            human_id="h1",
            action="continue",
        )

        assert guidance.decision is None
        assert guidance.hints == []
        assert guidance.constraints == []
        assert guidance.preferred_direction is None
        assert guidance.reasoning == ""

    def test_answers_dict_preserved(self):
        """answers dict should be preserved."""
        answers = {"q1": "answer1", "q2": "answer2"}
        guidance = HumanGuidance(
            guidance_id="g1",
            debate_id="d1",
            human_id="h1",
            action="resolve",
            answers=answers,
        )

        assert guidance.answers == answers
        assert guidance.answers["q1"] == "answer1"


# =============================================================================
# Breakpoint Dataclass Tests
# =============================================================================

class TestBreakpointDataclass:
    """Tests for Breakpoint dataclass."""

    def test_all_fields_initialized(self, sample_breakpoint):
        """All 6 fields should be initialized correctly."""
        assert sample_breakpoint.breakpoint_id == "bp-test-1"
        assert sample_breakpoint.trigger == BreakpointTrigger.LOW_CONFIDENCE
        assert sample_breakpoint.triggered_at is not None
        assert sample_breakpoint.debate_snapshot is not None

    def test_resolved_defaults_false(self, sample_snapshot):
        """resolved should default to False."""
        bp = Breakpoint(
            breakpoint_id="bp-1",
            trigger=BreakpointTrigger.DEADLOCK,
            triggered_at=datetime.now().isoformat(),
            debate_snapshot=sample_snapshot,
        )

        assert bp.resolved is False
        assert bp.guidance is None
        assert bp.resolved_at is None

    def test_escalation_level_defaults(self, sample_snapshot):
        """escalation_level should default to 1."""
        bp = Breakpoint(
            breakpoint_id="bp-1",
            trigger=BreakpointTrigger.DEADLOCK,
            triggered_at=datetime.now().isoformat(),
            debate_snapshot=sample_snapshot,
        )

        assert bp.escalation_level == 1

    def test_timeout_minutes_defaults(self, sample_snapshot):
        """timeout_minutes should default to 30."""
        bp = Breakpoint(
            breakpoint_id="bp-1",
            trigger=BreakpointTrigger.DEADLOCK,
            triggered_at=datetime.now().isoformat(),
            debate_snapshot=sample_snapshot,
        )

        assert bp.timeout_minutes == 30


# =============================================================================
# BreakpointConfig Tests
# =============================================================================

class TestBreakpointConfig:
    """Tests for BreakpointConfig dataclass."""

    def test_default_thresholds(self, default_config):
        """Default thresholds should be correct."""
        assert default_config.min_confidence == 0.6
        assert default_config.max_deadlock_rounds == 3
        assert default_config.max_total_rounds == 10
        assert default_config.disagreement_threshold == 0.7

    def test_default_safety_keywords(self, default_config):
        """All 5 default safety keywords should be present."""
        expected = ["dangerous", "harmful", "illegal", "unethical", "unsafe"]
        for keyword in expected:
            assert keyword in default_config.safety_keywords

    def test_notification_channels_defaults_empty(self, default_config):
        """notification_channels should default to empty list."""
        assert default_config.notification_channels == []

    def test_auto_timeout_action_default(self, default_config):
        """auto_timeout_action should default to 'continue'."""
        assert default_config.auto_timeout_action == "continue"


# =============================================================================
# HumanNotifier Tests
# =============================================================================

class TestHumanNotifier:
    """Tests for HumanNotifier class."""

    def test_register_handler_stores(self, default_config):
        """register_handler should store handler."""
        notifier = HumanNotifier(default_config)
        handler = AsyncMock()

        notifier.register_handler("slack", handler)

        assert "slack" in notifier._handlers
        assert notifier._handlers["slack"] is handler

    @pytest.mark.asyncio
    async def test_notify_calls_registered_handlers(self, sample_breakpoint):
        """notify should call registered handlers."""
        config = BreakpointConfig(notification_channels=["slack", "discord"])
        notifier = HumanNotifier(config)

        slack_handler = AsyncMock()
        discord_handler = AsyncMock()
        notifier.register_handler("slack", slack_handler)
        notifier.register_handler("discord", discord_handler)

        result = await notifier.notify(sample_breakpoint)

        assert result is True
        slack_handler.assert_called_once_with(sample_breakpoint)
        discord_handler.assert_called_once_with(sample_breakpoint)

    @pytest.mark.asyncio
    async def test_notify_falls_back_to_cli(self, sample_breakpoint):
        """notify should fallback to CLI when no handlers."""
        config = BreakpointConfig(notification_channels=[])
        notifier = HumanNotifier(config)

        with patch.object(notifier, "_cli_notify") as mock_cli:
            result = await notifier.notify(sample_breakpoint)

            assert result is True
            mock_cli.assert_called_once_with(sample_breakpoint)

    @pytest.mark.asyncio
    async def test_handler_exception_doesnt_crash(self, sample_breakpoint):
        """Handler exception shouldn't crash notification."""
        config = BreakpointConfig(notification_channels=["slack", "discord"])
        notifier = HumanNotifier(config)

        failing_handler = AsyncMock(side_effect=Exception("Handler error"))
        success_handler = AsyncMock()
        notifier.register_handler("slack", failing_handler)
        notifier.register_handler("discord", success_handler)

        result = await notifier.notify(sample_breakpoint)

        assert result is True
        success_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_success_status(self, sample_breakpoint):
        """notify should return True on success."""
        config = BreakpointConfig(notification_channels=["test"])
        notifier = HumanNotifier(config)
        notifier.register_handler("test", AsyncMock())

        result = await notifier.notify(sample_breakpoint)

        assert result is True


# =============================================================================
# BreakpointManager Initialization Tests
# =============================================================================

class TestBreakpointManagerInit:
    """Tests for BreakpointManager initialization."""

    def test_default_config_created(self):
        """Default config should be created when None."""
        manager = BreakpointManager()

        assert manager.config is not None
        assert manager.config.min_confidence == 0.6  # Default value

    def test_custom_config_used(self, custom_config):
        """Custom config should be used when provided."""
        manager = BreakpointManager(config=custom_config)

        assert manager.config.min_confidence == 0.8
        assert manager.config.max_deadlock_rounds == 2

    def test_notifier_initialized(self):
        """Notifier should be initialized."""
        manager = BreakpointManager()

        assert manager.notifier is not None
        assert isinstance(manager.notifier, HumanNotifier)


# =============================================================================
# check_triggers Method Tests
# =============================================================================

class TestCheckTriggers:
    """Tests for BreakpointManager.check_triggers method."""

    def test_low_confidence_triggers(self, breakpoint_manager, mock_message):
        """Should trigger when confidence below threshold."""
        messages = [mock_message()]

        result = breakpoint_manager.check_triggers(
            debate_id="test",
            task="Test task",
            messages=messages,
            confidence=0.3,  # Below default 0.6
            round_num=1,
            max_rounds=10,
        )

        assert result is not None
        assert result.trigger == BreakpointTrigger.LOW_CONFIDENCE

    def test_low_confidence_no_trigger_when_above(self, breakpoint_manager, mock_message):
        """Should not trigger when confidence at or above threshold."""
        messages = [mock_message()]

        result = breakpoint_manager.check_triggers(
            debate_id="test",
            task="Test task",
            messages=messages,
            confidence=0.7,  # Above default 0.6
            round_num=1,
            max_rounds=10,
        )

        assert result is None

    def test_deadlock_triggers(self, mock_message):
        """Should trigger after max_deadlock_rounds with repeated content."""
        config = BreakpointConfig(max_deadlock_rounds=2)
        manager = BreakpointManager(config=config)

        # Create messages with same content (simulating deadlock)
        messages = [
            mock_message(content="Same content here", round_num=1),
            mock_message(content="Same content here", round_num=2),
            mock_message(content="Same content here", round_num=3),
            mock_message(content="Same content here", round_num=4),
        ]

        result = manager.check_triggers(
            debate_id="test",
            task="Test task",
            messages=messages,
            confidence=0.8,  # Above threshold
            round_num=3,
            max_rounds=10,
        )

        assert result is not None
        assert result.trigger == BreakpointTrigger.DEADLOCK

    def test_round_limit_triggers(self, breakpoint_manager, mock_message):
        """Should trigger when round >= max_total_rounds."""
        messages = [mock_message()]

        result = breakpoint_manager.check_triggers(
            debate_id="test",
            task="Test task",
            messages=messages,
            confidence=0.8,
            round_num=10,  # At limit
            max_rounds=10,
        )

        assert result is not None
        assert result.trigger == BreakpointTrigger.ROUND_LIMIT

    def test_safety_concern_triggers(self, breakpoint_manager, mock_message):
        """Should trigger on safety keywords."""
        messages = [mock_message(content="This could be dangerous to implement")]

        result = breakpoint_manager.check_triggers(
            debate_id="test",
            task="Test task",
            messages=messages,
            confidence=0.9,
            round_num=1,
            max_rounds=10,
        )

        assert result is not None
        assert result.trigger == BreakpointTrigger.SAFETY_CONCERN
        assert result.escalation_level == 3  # High priority

    def test_safety_concern_case_insensitive(self, breakpoint_manager, mock_message):
        """Safety keyword detection should be case-insensitive."""
        messages = [mock_message(content="This is HARMFUL behavior")]

        result = breakpoint_manager.check_triggers(
            debate_id="test",
            task="Test task",
            messages=messages,
            confidence=0.9,
            round_num=1,
            max_rounds=10,
        )

        assert result is not None
        assert result.trigger == BreakpointTrigger.SAFETY_CONCERN

    def test_high_disagreement_triggers(self, breakpoint_manager, mock_message):
        """Should trigger when critique severity exceeds threshold."""
        messages = [mock_message()]

        # Create mock critiques with high severity
        critique1 = MagicMock()
        critique1.severity = 0.9
        critique2 = MagicMock()
        critique2.severity = 0.8
        critiques = [critique1, critique2]

        result = breakpoint_manager.check_triggers(
            debate_id="test",
            task="Test task",
            messages=messages,
            confidence=0.8,
            round_num=1,
            max_rounds=10,
            critiques=critiques,
        )

        assert result is not None
        assert result.trigger == BreakpointTrigger.HIGH_DISAGREEMENT

    def test_no_trigger_all_conditions_satisfied(self, breakpoint_manager, mock_message):
        """Should return None when all conditions satisfied."""
        messages = [mock_message(content="Normal content here")]

        result = breakpoint_manager.check_triggers(
            debate_id="test",
            task="Test task",
            messages=messages,
            confidence=0.8,
            round_num=1,
            max_rounds=10,
        )

        assert result is None

    def test_empty_messages_handled(self, breakpoint_manager):
        """Should handle empty messages gracefully."""
        result = breakpoint_manager.check_triggers(
            debate_id="test",
            task="Test task",
            messages=[],
            confidence=0.8,
            round_num=1,
            max_rounds=10,
        )

        assert result is None

    def test_first_matching_trigger_returned(self, mock_message):
        """Should return first matching trigger (LOW_CONFIDENCE before others)."""
        config = BreakpointConfig(min_confidence=0.9)
        manager = BreakpointManager(config=config)

        # Message that would trigger safety concern
        messages = [mock_message(content="This is dangerous")]

        # But low confidence should trigger first
        result = manager.check_triggers(
            debate_id="test",
            task="Test task",
            messages=messages,
            confidence=0.5,  # Below 0.9 threshold
            round_num=1,
            max_rounds=10,
        )

        assert result is not None
        assert result.trigger == BreakpointTrigger.LOW_CONFIDENCE


# =============================================================================
# _detect_deadlock Method Tests
# =============================================================================

class TestDetectDeadlock:
    """Tests for BreakpointManager._detect_deadlock method."""

    def test_returns_false_insufficient_messages(self, breakpoint_manager, mock_message):
        """Should return False with < lookback * 2 messages."""
        messages = [mock_message(), mock_message()]

        result = breakpoint_manager._detect_deadlock(messages, lookback=3)

        assert result is False

    def test_detects_high_content_overlap(self, breakpoint_manager, mock_message):
        """Should detect deadlock when content overlap > 50%."""
        # Create messages with same content in both periods
        messages = [
            mock_message(content="Repeated content A", round_num=1),
            mock_message(content="Repeated content B", round_num=2),
            mock_message(content="Repeated content A", round_num=3),
            mock_message(content="Repeated content B", round_num=4),
        ]

        result = breakpoint_manager._detect_deadlock(messages, lookback=2)

        assert result is True

    def test_no_deadlock_varied_content(self, breakpoint_manager, mock_message):
        """Should not detect deadlock with varied content."""
        messages = [
            mock_message(content="Unique content 1", round_num=1),
            mock_message(content="Unique content 2", round_num=2),
            mock_message(content="Unique content 3", round_num=3),
            mock_message(content="Unique content 4", round_num=4),
        ]

        result = breakpoint_manager._detect_deadlock(messages, lookback=2)

        assert result is False

    def test_case_insensitive_comparison(self, breakpoint_manager, mock_message):
        """Deadlock detection should be case-insensitive."""
        messages = [
            mock_message(content="SAME CONTENT", round_num=1),
            mock_message(content="same content", round_num=2),
            mock_message(content="Same Content", round_num=3),
            mock_message(content="SAME CONTENT", round_num=4),
        ]

        result = breakpoint_manager._detect_deadlock(messages, lookback=2)

        assert result is True

    def test_handles_empty_content(self, breakpoint_manager, mock_message):
        """Should handle messages with empty content."""
        messages = [
            mock_message(content="", round_num=1),
            mock_message(content="", round_num=2),
            mock_message(content="", round_num=3),
            mock_message(content="", round_num=4),
        ]

        # Should not raise, but may detect as deadlock (empty = same)
        result = breakpoint_manager._detect_deadlock(messages, lookback=2)
        assert isinstance(result, bool)


# =============================================================================
# _create_breakpoint Method Tests
# =============================================================================

class TestCreateBreakpoint:
    """Tests for BreakpointManager._create_breakpoint method."""

    def test_generates_unique_id(self, breakpoint_manager, mock_message):
        """Should generate unique breakpoint_id."""
        messages = [mock_message()]

        bp1 = breakpoint_manager._create_breakpoint(
            BreakpointTrigger.LOW_CONFIDENCE,
            "debate-1", "task", messages, 0.5, 1, 10,
        )
        bp2 = breakpoint_manager._create_breakpoint(
            BreakpointTrigger.DEADLOCK,
            "debate-1", "task", messages, 0.5, 2, 10,
        )

        assert bp1.breakpoint_id != bp2.breakpoint_id
        assert "debate-1" in bp1.breakpoint_id

    def test_extracts_latest_5_messages(self, breakpoint_manager, mock_message):
        """Should extract last 5 messages for snapshot."""
        messages = [mock_message(content=f"Message {i}", round_num=i) for i in range(10)]

        bp = breakpoint_manager._create_breakpoint(
            BreakpointTrigger.LOW_CONFIDENCE,
            "debate-1", "task", messages, 0.5, 1, 10,
        )

        assert len(bp.debate_snapshot.latest_messages) == 5

    def test_tracks_agent_positions(self, breakpoint_manager, mock_message):
        """Should track agent positions correctly."""
        messages = [
            mock_message(agent="claude", content="Claude position"),
            mock_message(agent="gemini", content="Gemini position"),
            mock_message(agent="claude", content="Claude updated"),
        ]

        bp = breakpoint_manager._create_breakpoint(
            BreakpointTrigger.LOW_CONFIDENCE,
            "debate-1", "task", messages, 0.5, 1, 10,
        )

        # Should have both agents, with most recent positions
        assert "claude" in bp.debate_snapshot.agent_positions
        assert "gemini" in bp.debate_snapshot.agent_positions

    def test_increments_counter(self, breakpoint_manager, mock_message):
        """Should increment breakpoint counter."""
        messages = [mock_message()]

        initial_count = breakpoint_manager._breakpoint_counter

        breakpoint_manager._create_breakpoint(
            BreakpointTrigger.LOW_CONFIDENCE,
            "debate-1", "task", messages, 0.5, 1, 10,
        )

        assert breakpoint_manager._breakpoint_counter == initial_count + 1


# =============================================================================
# inject_guidance Method Tests
# =============================================================================

class TestInjectGuidance:
    """Tests for BreakpointManager.inject_guidance method."""

    def test_resolve_adds_human_decision(self, breakpoint_manager, mock_message):
        """'resolve' action should add human decision message."""
        messages = [mock_message()]
        environment = MagicMock()

        guidance = HumanGuidance(
            guidance_id="g1",
            debate_id="d1",
            human_id="h1",
            action="resolve",
            decision="The correct answer is X",
        )

        new_messages, _ = breakpoint_manager.inject_guidance(guidance, messages, environment)

        assert len(new_messages) == 2
        assert new_messages[-1].agent == "human"
        assert new_messages[-1].role == "judge"
        assert "[HUMAN DECISION]" in new_messages[-1].content
        assert "The correct answer is X" in new_messages[-1].content

    def test_redirect_adds_hints(self, breakpoint_manager, mock_message):
        """'redirect' action should add hints as moderator message."""
        messages = [mock_message()]
        environment = MagicMock()

        guidance = HumanGuidance(
            guidance_id="g1",
            debate_id="d1",
            human_id="h1",
            action="redirect",
            hints=["Consider option A", "Look at the data"],
        )

        new_messages, _ = breakpoint_manager.inject_guidance(guidance, messages, environment)

        assert len(new_messages) == 2
        assert new_messages[-1].agent == "human"
        assert new_messages[-1].role == "moderator"
        assert "[HUMAN GUIDANCE]" in new_messages[-1].content
        assert "Consider option A" in new_messages[-1].content

    def test_continue_no_modification(self, breakpoint_manager, mock_message):
        """'continue' action should not modify messages."""
        messages = [mock_message()]
        environment = MagicMock()

        guidance = HumanGuidance(
            guidance_id="g1",
            debate_id="d1",
            human_id="h1",
            action="continue",
        )

        new_messages, new_env = breakpoint_manager.inject_guidance(guidance, messages, environment)

        assert len(new_messages) == 1

    def test_abort_no_modification(self, breakpoint_manager, mock_message):
        """'abort' action should not modify messages."""
        messages = [mock_message()]
        environment = MagicMock()

        guidance = HumanGuidance(
            guidance_id="g1",
            debate_id="d1",
            human_id="h1",
            action="abort",
        )

        new_messages, _ = breakpoint_manager.inject_guidance(guidance, messages, environment)

        assert len(new_messages) == 1

    def test_redirect_applies_constraints(self, breakpoint_manager, mock_message):
        """'redirect' should apply constraints to environment."""
        messages = [mock_message()]
        environment = MagicMock()
        environment.constraints = []

        guidance = HumanGuidance(
            guidance_id="g1",
            debate_id="d1",
            human_id="h1",
            action="redirect",
            hints=["Consider this"],
            constraints=["Must be safe", "Must be fast"],
        )

        _, new_env = breakpoint_manager.inject_guidance(guidance, messages, environment)

        assert "Must be safe" in new_env.constraints
        assert "Must be fast" in new_env.constraints


# =============================================================================
# handle_breakpoint Method Tests
# =============================================================================

class TestHandleBreakpoint:
    """Tests for BreakpointManager.handle_breakpoint method."""

    @pytest.mark.asyncio
    async def test_notifies_human(self, sample_breakpoint):
        """Should notify human when handling breakpoint."""
        mock_input = AsyncMock(return_value=HumanGuidance(
            guidance_id="g1",
            debate_id="d1",
            human_id="h1",
            action="continue",
        ))
        manager = BreakpointManager(get_human_input=mock_input)

        with patch.object(manager.notifier, "notify", new_callable=AsyncMock) as mock_notify:
            await manager.handle_breakpoint(sample_breakpoint)

            mock_notify.assert_called_once_with(sample_breakpoint)

    @pytest.mark.asyncio
    async def test_marks_resolved(self, sample_breakpoint):
        """Should mark breakpoint as resolved."""
        mock_input = AsyncMock(return_value=HumanGuidance(
            guidance_id="g1",
            debate_id="d1",
            human_id="h1",
            action="continue",
        ))
        manager = BreakpointManager(get_human_input=mock_input)

        with patch.object(manager.notifier, "notify", new_callable=AsyncMock):
            await manager.handle_breakpoint(sample_breakpoint)

        assert sample_breakpoint.resolved is True
        assert sample_breakpoint.resolved_at is not None
        assert sample_breakpoint.guidance is not None

    @pytest.mark.asyncio
    async def test_timeout_uses_default_action(self, sample_breakpoint):
        """Should use default action on timeout."""
        async def slow_input(bp):
            await asyncio.sleep(100)  # Will be interrupted
            return HumanGuidance(
                guidance_id="g1",
                debate_id="d1",
                human_id="h1",
                action="resolve",
            )

        sample_breakpoint.timeout_minutes = 0.001  # Very short timeout

        manager = BreakpointManager(get_human_input=slow_input)

        with patch.object(manager.notifier, "notify", new_callable=AsyncMock):
            guidance = await manager.handle_breakpoint(sample_breakpoint)

        assert guidance.action == "continue"  # Default
        assert "system" in guidance.human_id
        assert "timeout" in guidance.reasoning.lower()


# =============================================================================
# Decorator Tests
# =============================================================================

class TestDecorators:
    """Tests for breakpoint decorators."""

    def test_critical_decision_sets_attributes(self):
        """@critical_decision should set function attributes."""
        @critical_decision(reason="Important choice")
        def my_func():
            pass

        assert hasattr(my_func, "_aragora_critical")
        assert my_func._aragora_critical is True
        assert my_func._aragora_critical_reason == "Important choice"

    def test_breakpoint_decorator_sets_attributes(self):
        """@breakpoint decorator should set attributes."""
        @breakpoint(trigger="confidence < 0.5", threshold=0.5, message="Low confidence")
        def my_func():
            pass

        assert hasattr(my_func, "_aragora_breakpoint")
        assert my_func._aragora_breakpoint is True
        assert my_func._aragora_breakpoint_trigger == "confidence < 0.5"
        assert my_func._aragora_breakpoint_threshold == 0.5
        assert my_func._aragora_breakpoint_message == "Low confidence"


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_messages_without_content_attribute(self, breakpoint_manager):
        """Should handle messages without content attribute."""
        msg = MagicMock(spec=["agent", "round"])
        msg.agent = "test"
        msg.round = 1
        # No content attribute

        # Should not raise
        result = breakpoint_manager.check_triggers(
            debate_id="test",
            task="Test",
            messages=[msg],
            confidence=0.8,
            round_num=1,
            max_rounds=10,
        )

        assert result is None

    def test_very_long_content_truncated(self, breakpoint_manager, mock_message):
        """Should truncate very long content in snapshots."""
        long_content = "x" * 1000
        messages = [mock_message(content=long_content)]

        bp = breakpoint_manager._create_breakpoint(
            BreakpointTrigger.LOW_CONFIDENCE,
            "debate-1", "task", messages, 0.5, 1, 10,
        )

        # Content should be truncated to 200 chars in latest_messages
        for msg_dict in bp.debate_snapshot.latest_messages:
            assert len(msg_dict["content"]) <= 200

    def test_empty_critiques_list(self, breakpoint_manager, mock_message):
        """Should handle empty critiques list."""
        messages = [mock_message()]

        result = breakpoint_manager.check_triggers(
            debate_id="test",
            task="Test",
            messages=messages,
            confidence=0.8,
            round_num=1,
            max_rounds=10,
            critiques=[],  # Empty list
        )

        # Should not crash, no HIGH_DISAGREEMENT trigger
        assert result is None

    def test_confidence_exactly_at_threshold(self, breakpoint_manager, mock_message):
        """Should not trigger when confidence exactly at threshold."""
        messages = [mock_message()]

        result = breakpoint_manager.check_triggers(
            debate_id="test",
            task="Test",
            messages=messages,
            confidence=0.6,  # Exactly at default threshold
            round_num=1,
            max_rounds=10,
        )

        # 0.6 is not < 0.6, so no trigger
        assert result is None

    def test_multiple_safety_keywords(self, breakpoint_manager, mock_message):
        """Should trigger on any safety keyword."""
        messages = [
            mock_message(content="This is illegal and unethical behavior"),
        ]

        result = breakpoint_manager.check_triggers(
            debate_id="test",
            task="Test",
            messages=messages,
            confidence=0.9,
            round_num=1,
            max_rounds=10,
        )

        assert result is not None
        assert result.trigger == BreakpointTrigger.SAFETY_CONCERN
