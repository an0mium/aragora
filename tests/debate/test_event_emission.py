"""Tests for EventEmitter class in aragora.debate.event_emission."""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock, MagicMock, call


from aragora.debate.event_emission import EventEmitter


# === Test Fixtures ===


@pytest.fixture
def mock_event_bus() -> Mock:
    """Create a mock EventBus with emit_sync method."""
    bus = Mock()
    bus.emit_sync = Mock()
    return bus


@pytest.fixture
def mock_event_bridge() -> Mock:
    """Create a mock EventEmitterBridge with notify and emit_moment methods."""
    bridge = Mock()
    bridge.notify = Mock()
    bridge.emit_moment = Mock()
    return bridge


@pytest.fixture
def mock_hooks() -> dict:
    """Create mock hooks dictionary with on_agent_preview callback."""
    return {"on_agent_preview": Mock()}


@pytest.fixture
def mock_persona_manager() -> Mock:
    """Create a mock PersonaManager with get_persona method."""
    manager = Mock()
    persona = Mock()
    persona.brief_description = "A thoughtful analyst"
    manager.get_persona = Mock(return_value=persona)
    return manager


@pytest.fixture
def mock_agent() -> Mock:
    """Create a mock Agent with name attribute."""
    agent = Mock()
    agent.name = "claude"
    return agent


@pytest.fixture
def mock_moment() -> Mock:
    """Create a mock moment with to_dict method."""
    moment = Mock()
    moment.to_dict = Mock(
        return_value={"content": "test moment", "timestamp": "2024-01-01T00:00:00Z"}
    )
    moment.moment_type = "insight"
    moment.agent_name = "claude"
    return moment


@pytest.fixture
def full_emitter(
    mock_event_bus: Mock,
    mock_event_bridge: Mock,
    mock_hooks: dict,
    mock_persona_manager: Mock,
) -> EventEmitter:
    """Create an EventEmitter with all dependencies."""
    return EventEmitter(
        event_bus=mock_event_bus,
        event_bridge=mock_event_bridge,
        hooks=mock_hooks,
        persona_manager=mock_persona_manager,
    )


# === Initialization Tests ===


class TestEventEmitterInit:
    """Test EventEmitter initialization with various dependency combinations."""

    def test_init_with_all_dependencies(
        self,
        mock_event_bus: Mock,
        mock_event_bridge: Mock,
        mock_hooks: dict,
        mock_persona_manager: Mock,
    ) -> None:
        """EventEmitter can be initialized with all dependencies."""
        emitter = EventEmitter(
            event_bus=mock_event_bus,
            event_bridge=mock_event_bridge,
            hooks=mock_hooks,
            persona_manager=mock_persona_manager,
        )
        assert emitter.event_bus is mock_event_bus
        assert emitter.event_bridge is mock_event_bridge
        assert emitter.hooks is mock_hooks
        assert emitter.persona_manager is mock_persona_manager
        assert emitter._current_debate_id == ""

    def test_init_with_no_dependencies(self) -> None:
        """EventEmitter can be initialized with no dependencies."""
        emitter = EventEmitter()
        assert emitter.event_bus is None
        assert emitter.event_bridge is None
        assert emitter.hooks == {}
        assert emitter.persona_manager is None
        assert emitter._current_debate_id == ""

    def test_init_with_only_event_bus(self, mock_event_bus: Mock) -> None:
        """EventEmitter can be initialized with only event_bus."""
        emitter = EventEmitter(event_bus=mock_event_bus)
        assert emitter.event_bus is mock_event_bus
        assert emitter.event_bridge is None
        assert emitter.hooks == {}

    def test_init_with_only_event_bridge(self, mock_event_bridge: Mock) -> None:
        """EventEmitter can be initialized with only event_bridge."""
        emitter = EventEmitter(event_bridge=mock_event_bridge)
        assert emitter.event_bus is None
        assert emitter.event_bridge is mock_event_bridge
        assert emitter.hooks == {}

    def test_init_with_empty_hooks(self) -> None:
        """EventEmitter with None hooks defaults to empty dict."""
        emitter = EventEmitter(hooks=None)
        assert emitter.hooks == {}

    def test_init_with_custom_hooks(self) -> None:
        """EventEmitter preserves custom hooks dict."""
        custom_hooks = {"on_test": Mock(), "on_other": Mock()}
        emitter = EventEmitter(hooks=custom_hooks)
        assert emitter.hooks is custom_hooks


class TestSetDebateId:
    """Test set_debate_id method."""

    def test_set_debate_id(self, full_emitter: EventEmitter) -> None:
        """set_debate_id updates the current debate ID."""
        full_emitter.set_debate_id("debate-123")
        assert full_emitter._current_debate_id == "debate-123"

    def test_set_debate_id_multiple_times(self, full_emitter: EventEmitter) -> None:
        """set_debate_id can be called multiple times."""
        full_emitter.set_debate_id("debate-1")
        assert full_emitter._current_debate_id == "debate-1"
        full_emitter.set_debate_id("debate-2")
        assert full_emitter._current_debate_id == "debate-2"


# === notify_spectator Tests ===


class TestNotifySpectator:
    """Test notify_spectator method."""

    def test_notify_spectator_uses_event_bus(
        self,
        mock_event_bus: Mock,
        mock_event_bridge: Mock,
    ) -> None:
        """notify_spectator prefers EventBus over event_bridge."""
        emitter = EventEmitter(event_bus=mock_event_bus, event_bridge=mock_event_bridge)
        emitter.set_debate_id("debate-123")

        emitter.notify_spectator("test_event", data="test_data", count=5)

        mock_event_bus.emit_sync.assert_called_once_with(
            "test_event",
            debate_id="debate-123",
            data="test_data",
            count=5,
        )
        mock_event_bridge.notify.assert_not_called()

    def test_notify_spectator_fallback_to_event_bridge(
        self,
        mock_event_bridge: Mock,
    ) -> None:
        """notify_spectator falls back to event_bridge when no event_bus."""
        emitter = EventEmitter(event_bridge=mock_event_bridge)

        emitter.notify_spectator("test_event", data="test_data")

        mock_event_bridge.notify.assert_called_once_with("test_event", data="test_data")

    def test_notify_spectator_uses_provided_debate_id(
        self,
        mock_event_bus: Mock,
    ) -> None:
        """notify_spectator uses explicit debate_id over current one."""
        emitter = EventEmitter(event_bus=mock_event_bus)
        emitter.set_debate_id("current-debate")

        emitter.notify_spectator("test_event", debate_id="explicit-debate")

        mock_event_bus.emit_sync.assert_called_once_with(
            "test_event",
            debate_id="explicit-debate",
        )

    def test_notify_spectator_no_emitters(self) -> None:
        """notify_spectator does nothing when no emitters configured."""
        emitter = EventEmitter()
        # Should not raise
        emitter.notify_spectator("test_event", data="test")


# === emit_moment Tests ===


class TestEmitMoment:
    """Test emit_moment method."""

    def test_emit_moment_with_event_bus(
        self,
        mock_event_bus: Mock,
        mock_moment: Mock,
    ) -> None:
        """emit_moment uses EventBus when available."""
        emitter = EventEmitter(event_bus=mock_event_bus)
        emitter.set_debate_id("debate-123")

        emitter.emit_moment(mock_moment)

        mock_event_bus.emit_sync.assert_called_once()
        call_args = mock_event_bus.emit_sync.call_args
        assert call_args.kwargs["event_type"] == "moment"
        assert call_args.kwargs["debate_id"] == "debate-123"
        assert call_args.kwargs["moment_type"] == "insight"
        assert call_args.kwargs["agent"] == "claude"
        assert call_args.kwargs["content"] == "test moment"

    def test_emit_moment_fallback_to_event_bridge(
        self,
        mock_event_bridge: Mock,
        mock_moment: Mock,
    ) -> None:
        """emit_moment falls back to event_bridge when no event_bus."""
        emitter = EventEmitter(event_bridge=mock_event_bridge)

        emitter.emit_moment(mock_moment)

        mock_event_bridge.emit_moment.assert_called_once_with(mock_moment)

    def test_emit_moment_no_to_dict(
        self,
        mock_event_bus: Mock,
    ) -> None:
        """emit_moment handles moments without to_dict method."""
        moment = Mock(spec=[])  # No to_dict
        emitter = EventEmitter(event_bus=mock_event_bus)

        emitter.emit_moment(moment)

        # Should not emit when moment lacks to_dict
        mock_event_bus.emit_sync.assert_not_called()

    def test_emit_moment_no_emitters(self, mock_moment: Mock) -> None:
        """emit_moment does nothing when no emitters configured."""
        emitter = EventEmitter()
        # Should not raise
        emitter.emit_moment(mock_moment)

    def test_emit_moment_missing_attributes(
        self,
        mock_event_bus: Mock,
    ) -> None:
        """emit_moment handles moments with missing attributes gracefully."""
        moment = Mock()
        moment.to_dict = Mock(return_value={"data": "test"})
        # Simulate missing attributes
        del moment.moment_type
        del moment.agent_name

        emitter = EventEmitter(event_bus=mock_event_bus)
        emitter.set_debate_id("debate-123")

        emitter.emit_moment(moment)

        call_args = mock_event_bus.emit_sync.call_args
        assert call_args.kwargs["moment_type"] == "unknown"
        assert call_args.kwargs["agent"] is None


# === broadcast_health_event Tests ===


class TestBroadcastHealthEvent:
    """Test broadcast_health_event method."""

    def test_broadcast_health_event_with_event_bus(
        self,
        mock_event_bus: Mock,
    ) -> None:
        """broadcast_health_event uses EventBus when available."""
        emitter = EventEmitter(event_bus=mock_event_bus)
        emitter.set_debate_id("debate-123")

        event = {"data": {"status": "healthy", "latency_ms": 50}}
        emitter.broadcast_health_event(event)

        mock_event_bus.emit_sync.assert_called_once_with(
            event_type="health_event",
            debate_id="debate-123",
            status="healthy",
            latency_ms=50,
        )

    def test_broadcast_health_event_fallback_to_event_bridge(
        self,
        mock_event_bridge: Mock,
    ) -> None:
        """broadcast_health_event falls back to event_bridge."""
        emitter = EventEmitter(event_bridge=mock_event_bridge)

        event = {"data": {"status": "degraded"}}
        emitter.broadcast_health_event(event)

        mock_event_bridge.notify.assert_called_once_with(
            event_type="health_event",
            status="degraded",
        )

    def test_broadcast_health_event_filters_event_type_and_debate_id(
        self,
        mock_event_bus: Mock,
    ) -> None:
        """broadcast_health_event filters out event_type and debate_id from data."""
        emitter = EventEmitter(event_bus=mock_event_bus)
        emitter.set_debate_id("debate-123")

        event = {
            "data": {
                "event_type": "should_be_filtered",
                "debate_id": "should_be_filtered",
                "status": "healthy",
            }
        }
        emitter.broadcast_health_event(event)

        call_args = mock_event_bus.emit_sync.call_args
        assert "event_type" in call_args.kwargs  # The passed event_type, not filtered one
        assert call_args.kwargs["event_type"] == "health_event"
        assert call_args.kwargs["status"] == "healthy"

    def test_broadcast_health_event_handles_direct_data(
        self,
        mock_event_bus: Mock,
    ) -> None:
        """broadcast_health_event handles events without nested data key."""
        emitter = EventEmitter(event_bus=mock_event_bus)
        emitter.set_debate_id("debate-123")

        event = {"status": "healthy", "cpu_percent": 45.2}
        emitter.broadcast_health_event(event)

        call_args = mock_event_bus.emit_sync.call_args
        assert call_args.kwargs["status"] == "healthy"
        assert call_args.kwargs["cpu_percent"] == 45.2

    def test_broadcast_health_event_handles_errors_gracefully(self) -> None:
        """broadcast_health_event catches and logs errors."""
        mock_bus = Mock()
        mock_bus.emit_sync = Mock(side_effect=RuntimeError("Test error"))
        emitter = EventEmitter(event_bus=mock_bus)

        # Should not raise
        emitter.broadcast_health_event({"data": {"status": "test"}})

    def test_broadcast_health_event_handles_invalid_event(
        self,
        mock_event_bus: Mock,
    ) -> None:
        """broadcast_health_event handles non-dict data gracefully."""
        emitter = EventEmitter(event_bus=mock_event_bus)

        # Non-dict data value
        event = {"data": "not a dict"}
        emitter.broadcast_health_event(event)

        # Should emit with empty data (filtered non-dict)
        mock_event_bus.emit_sync.assert_called_once_with(
            event_type="health_event",
            debate_id="",
        )


# === Agent Preview Tests ===


class TestShouldEmitPreview:
    """Test should_emit_preview method."""

    def test_should_emit_preview_true(self, mock_hooks: dict) -> None:
        """should_emit_preview returns True when hook is registered."""
        emitter = EventEmitter(hooks=mock_hooks)
        assert emitter.should_emit_preview() is True

    def test_should_emit_preview_false(self) -> None:
        """should_emit_preview returns False when hook is not registered."""
        emitter = EventEmitter(hooks={})
        assert emitter.should_emit_preview() is False

    def test_should_emit_preview_other_hooks(self) -> None:
        """should_emit_preview returns False with different hooks."""
        emitter = EventEmitter(hooks={"on_other": Mock()})
        assert emitter.should_emit_preview() is False


class TestGetAgentRole:
    """Test get_agent_role method."""

    def test_get_agent_role_from_assignments(
        self,
        full_emitter: EventEmitter,
        mock_agent: Mock,
    ) -> None:
        """get_agent_role retrieves role from role_assignments."""
        role_assignments = {"claude": {"role": "critic", "stance": "negative"}}
        assert full_emitter.get_agent_role(mock_agent, role_assignments) == "critic"

    def test_get_agent_role_default(
        self,
        full_emitter: EventEmitter,
        mock_agent: Mock,
    ) -> None:
        """get_agent_role defaults to 'proposer'."""
        role_assignments = {}
        assert full_emitter.get_agent_role(mock_agent, role_assignments) == "proposer"

    def test_get_agent_role_missing_role_key(
        self,
        full_emitter: EventEmitter,
        mock_agent: Mock,
    ) -> None:
        """get_agent_role defaults to 'proposer' when role key is missing."""
        role_assignments = {"claude": {"stance": "negative"}}
        assert full_emitter.get_agent_role(mock_agent, role_assignments) == "proposer"


class TestGetAgentStance:
    """Test get_agent_stance method."""

    def test_get_agent_stance_from_assignments(
        self,
        full_emitter: EventEmitter,
        mock_agent: Mock,
    ) -> None:
        """get_agent_stance retrieves stance from role_assignments."""
        role_assignments = {"claude": {"role": "critic", "stance": "negative"}}
        assert full_emitter.get_agent_stance(mock_agent, role_assignments) == "negative"

    def test_get_agent_stance_default(
        self,
        full_emitter: EventEmitter,
        mock_agent: Mock,
    ) -> None:
        """get_agent_stance defaults to 'neutral'."""
        role_assignments = {}
        assert full_emitter.get_agent_stance(mock_agent, role_assignments) == "neutral"

    def test_get_agent_stance_missing_stance_key(
        self,
        full_emitter: EventEmitter,
        mock_agent: Mock,
    ) -> None:
        """get_agent_stance defaults to 'neutral' when stance key is missing."""
        role_assignments = {"claude": {"role": "critic"}}
        assert full_emitter.get_agent_stance(mock_agent, role_assignments) == "neutral"


class TestGetAgentDescription:
    """Test get_agent_description method."""

    def test_get_agent_description_with_persona(
        self,
        full_emitter: EventEmitter,
        mock_agent: Mock,
    ) -> None:
        """get_agent_description returns persona's brief_description."""
        description = full_emitter.get_agent_description(mock_agent)
        assert description == "A thoughtful analyst"

    def test_get_agent_description_no_persona_manager(
        self,
        mock_agent: Mock,
    ) -> None:
        """get_agent_description returns empty string without persona_manager."""
        emitter = EventEmitter()
        assert emitter.get_agent_description(mock_agent) == ""

    def test_get_agent_description_persona_none(
        self,
        mock_agent: Mock,
    ) -> None:
        """get_agent_description handles None persona gracefully."""
        manager = Mock()
        manager.get_persona = Mock(return_value=None)
        emitter = EventEmitter(persona_manager=manager)

        # getattr(None, "brief_description", "") returns ""
        assert emitter.get_agent_description(mock_agent) == ""


class TestBuildAgentPreview:
    """Test build_agent_preview method."""

    def test_build_agent_preview_full(
        self,
        full_emitter: EventEmitter,
        mock_agent: Mock,
    ) -> None:
        """build_agent_preview creates complete preview dict."""
        role_assignments = {"claude": {"role": "judge", "stance": "affirmative"}}
        preview = full_emitter.build_agent_preview(mock_agent, role_assignments)

        assert preview == {
            "name": "claude",
            "role": "judge",
            "stance": "affirmative",
            "description": "A thoughtful analyst",
            "strengths": [],
        }

    def test_build_agent_preview_defaults(
        self,
        mock_agent: Mock,
    ) -> None:
        """build_agent_preview uses defaults when data is missing."""
        emitter = EventEmitter()
        preview = emitter.build_agent_preview(mock_agent, {})

        assert preview == {
            "name": "claude",
            "role": "proposer",
            "stance": "neutral",
            "description": "",
            "strengths": [],
        }


class TestEmitAgentPreview:
    """Test emit_agent_preview method."""

    def test_emit_agent_preview_calls_hook(
        self,
        full_emitter: EventEmitter,
        mock_hooks: dict,
        mock_agent: Mock,
    ) -> None:
        """emit_agent_preview calls on_agent_preview hook with previews."""
        agents = [mock_agent]
        role_assignments = {"claude": {"role": "proposer", "stance": "neutral"}}

        full_emitter.emit_agent_preview(agents, role_assignments)

        mock_hooks["on_agent_preview"].assert_called_once()
        call_args = mock_hooks["on_agent_preview"].call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0]["name"] == "claude"

    def test_emit_agent_preview_multiple_agents(
        self,
        full_emitter: EventEmitter,
        mock_hooks: dict,
    ) -> None:
        """emit_agent_preview handles multiple agents."""
        agent1 = Mock()
        agent1.name = "claude"
        agent2 = Mock()
        agent2.name = "gpt4"

        role_assignments = {
            "claude": {"role": "proposer", "stance": "affirmative"},
            "gpt4": {"role": "critic", "stance": "negative"},
        }

        full_emitter.emit_agent_preview([agent1, agent2], role_assignments)

        call_args = mock_hooks["on_agent_preview"].call_args[0][0]
        assert len(call_args) == 2
        names = [p["name"] for p in call_args]
        assert "claude" in names
        assert "gpt4" in names

    def test_emit_agent_preview_no_hook(self, mock_agent: Mock) -> None:
        """emit_agent_preview does nothing when hook is not registered."""
        emitter = EventEmitter(hooks={})
        # Should not raise
        emitter.emit_agent_preview([mock_agent], {})

    def test_emit_agent_preview_handles_hook_error(
        self,
        mock_agent: Mock,
    ) -> None:
        """emit_agent_preview catches errors from hook callback."""
        failing_hook = Mock(side_effect=Exception("Hook failed"))
        emitter = EventEmitter(hooks={"on_agent_preview": failing_hook})

        # Should not raise
        emitter.emit_agent_preview([mock_agent], {})


# === Feature Event Emission Tests ===


class TestEmitCalibrationUpdate:
    """Test emit_calibration_update method."""

    def test_emit_calibration_update(self, mock_event_bus: Mock) -> None:
        """emit_calibration_update emits with correct parameters."""
        emitter = EventEmitter(event_bus=mock_event_bus)
        emitter.set_debate_id("debate-123")

        emitter.emit_calibration_update(
            agent_name="claude",
            brier_score=0.15,
            prediction_count=100,
            accuracy=0.85,
        )

        mock_event_bus.emit_sync.assert_called_once_with(
            "calibration_update",
            debate_id="debate-123",
            agent="claude",
            brier_score=0.15,
            prediction_count=100,
            accuracy=0.85,
        )


class TestEmitEvidenceFound:
    """Test emit_evidence_found method."""

    def test_emit_evidence_found(self, mock_event_bus: Mock) -> None:
        """emit_evidence_found emits with correct parameters."""
        emitter = EventEmitter(event_bus=mock_event_bus)
        emitter.set_debate_id("debate-123")

        emitter.emit_evidence_found(
            claim="The sky is blue",
            evidence_type="citation",
            source="Science Journal",
            confidence=0.9,
            excerpt="Light scattering causes blue color",
        )

        mock_event_bus.emit_sync.assert_called_once_with(
            "evidence_found",
            debate_id="debate-123",
            claim="The sky is blue",
            evidence_type="citation",
            source="Science Journal",
            confidence=0.9,
            excerpt="Light scattering causes blue color",
        )

    def test_emit_evidence_found_truncates_excerpt(self, mock_event_bus: Mock) -> None:
        """emit_evidence_found truncates long excerpts to 500 chars."""
        emitter = EventEmitter(event_bus=mock_event_bus)

        long_excerpt = "x" * 600
        emitter.emit_evidence_found(
            claim="Test",
            evidence_type="quote",
            source="Book",
            confidence=0.8,
            excerpt=long_excerpt,
        )

        call_args = mock_event_bus.emit_sync.call_args
        assert len(call_args.kwargs["excerpt"]) == 500

    def test_emit_evidence_found_empty_excerpt(self, mock_event_bus: Mock) -> None:
        """emit_evidence_found handles empty excerpt."""
        emitter = EventEmitter(event_bus=mock_event_bus)

        emitter.emit_evidence_found(
            claim="Test",
            evidence_type="fact",
            source="Source",
            confidence=0.7,
            excerpt="",
        )

        call_args = mock_event_bus.emit_sync.call_args
        assert call_args.kwargs["excerpt"] == ""


class TestEmitTraitEmerged:
    """Test emit_trait_emerged method."""

    def test_emit_trait_emerged(self, mock_event_bus: Mock) -> None:
        """emit_trait_emerged emits with correct parameters."""
        emitter = EventEmitter(event_bus=mock_event_bus)
        emitter.set_debate_id("debate-123")

        emitter.emit_trait_emerged(
            agent_name="claude",
            trait_name="analytical",
            trait_description="Shows deep analytical thinking",
            emergence_round=3,
        )

        mock_event_bus.emit_sync.assert_called_once_with(
            "trait_emerged",
            debate_id="debate-123",
            agent="claude",
            trait_name="analytical",
            trait_description="Shows deep analytical thinking",
            emergence_round=3,
        )


class TestEmitRiskWarning:
    """Test emit_risk_warning method."""

    def test_emit_risk_warning(self, mock_event_bus: Mock) -> None:
        """emit_risk_warning emits with correct parameters."""
        emitter = EventEmitter(event_bus=mock_event_bus)
        emitter.set_debate_id("debate-123")

        emitter.emit_risk_warning(
            risk_type="factual",
            severity="high",
            description="Unverified claim detected",
            affected_claims=["Claim A", "Claim B"],
        )

        mock_event_bus.emit_sync.assert_called_once_with(
            "risk_warning",
            debate_id="debate-123",
            risk_type="factual",
            severity="high",
            description="Unverified claim detected",
            affected_claims=["Claim A", "Claim B"],
        )

    def test_emit_risk_warning_no_affected_claims(self, mock_event_bus: Mock) -> None:
        """emit_risk_warning defaults affected_claims to empty list."""
        emitter = EventEmitter(event_bus=mock_event_bus)

        emitter.emit_risk_warning(
            risk_type="logical",
            severity="medium",
            description="Circular reasoning",
        )

        call_args = mock_event_bus.emit_sync.call_args
        assert call_args.kwargs["affected_claims"] == []


class TestEmitRhetoricalObservation:
    """Test emit_rhetorical_observation method."""

    def test_emit_rhetorical_observation(self, mock_event_bus: Mock) -> None:
        """emit_rhetorical_observation emits with correct parameters."""
        emitter = EventEmitter(event_bus=mock_event_bus)
        emitter.set_debate_id("debate-123")

        emitter.emit_rhetorical_observation(
            agent_name="claude",
            technique="appeal_to_authority",
            description="Used expert credentials",
            quote="As Dr. Smith says...",
            effectiveness=0.7,
        )

        mock_event_bus.emit_sync.assert_called_once_with(
            "rhetorical_observation",
            debate_id="debate-123",
            agent="claude",
            technique="appeal_to_authority",
            description="Used expert credentials",
            quote="As Dr. Smith says...",
            effectiveness=0.7,
        )

    def test_emit_rhetorical_observation_truncates_quote(
        self,
        mock_event_bus: Mock,
    ) -> None:
        """emit_rhetorical_observation truncates long quotes to 200 chars."""
        emitter = EventEmitter(event_bus=mock_event_bus)

        long_quote = "y" * 300
        emitter.emit_rhetorical_observation(
            agent_name="claude",
            technique="repetition",
            description="Repeated key point",
            quote=long_quote,
            effectiveness=0.5,
        )

        call_args = mock_event_bus.emit_sync.call_args
        assert len(call_args.kwargs["quote"]) == 200

    def test_emit_rhetorical_observation_defaults(self, mock_event_bus: Mock) -> None:
        """emit_rhetorical_observation uses default values."""
        emitter = EventEmitter(event_bus=mock_event_bus)

        emitter.emit_rhetorical_observation(
            agent_name="claude",
            technique="analogy",
            description="Used comparison",
        )

        call_args = mock_event_bus.emit_sync.call_args
        assert call_args.kwargs["quote"] == ""
        assert call_args.kwargs["effectiveness"] == 0.5


class TestEmitHollowConsensus:
    """Test emit_hollow_consensus method."""

    def test_emit_hollow_consensus(self, mock_event_bus: Mock) -> None:
        """emit_hollow_consensus emits with correct parameters."""
        emitter = EventEmitter(event_bus=mock_event_bus)
        emitter.set_debate_id("debate-123")

        emitter.emit_hollow_consensus(
            confidence=0.85,
            indicators=["Groupthink detected", "Lack of dissent"],
            recommendation="Request external review",
        )

        mock_event_bus.emit_sync.assert_called_once_with(
            "hollow_consensus",
            debate_id="debate-123",
            confidence=0.85,
            indicators=["Groupthink detected", "Lack of dissent"],
            recommendation="Request external review",
        )


class TestEmitTricksterIntervention:
    """Test emit_trickster_intervention method."""

    def test_emit_trickster_intervention(self, mock_event_bus: Mock) -> None:
        """emit_trickster_intervention emits with correct parameters."""
        emitter = EventEmitter(event_bus=mock_event_bus)
        emitter.set_debate_id("debate-123")

        emitter.emit_trickster_intervention(
            intervention_type="devils_advocate",
            challenge="What if the opposite were true?",
            target_claim="The proposal is optimal",
        )

        mock_event_bus.emit_sync.assert_called_once_with(
            "trickster_intervention",
            debate_id="debate-123",
            intervention_type="devils_advocate",
            challenge="What if the opposite were true?",
            target_claim="The proposal is optimal",
        )

    def test_emit_trickster_intervention_defaults(self, mock_event_bus: Mock) -> None:
        """emit_trickster_intervention uses default target_claim."""
        emitter = EventEmitter(event_bus=mock_event_bus)

        emitter.emit_trickster_intervention(
            intervention_type="contrarian",
            challenge="Consider alternative approaches",
        )

        call_args = mock_event_bus.emit_sync.call_args
        assert call_args.kwargs["target_claim"] == ""


class TestEmitAgentMessage:
    """Test emit_agent_message method."""

    def test_emit_agent_message(self, mock_event_bus: Mock) -> None:
        """emit_agent_message emits with correct parameters."""
        emitter = EventEmitter(event_bus=mock_event_bus)
        emitter.set_debate_id("debate-123")

        emitter.emit_agent_message(
            agent_name="claude",
            content="I propose we use a distributed cache...",
            role="proposer",
            round_num=2,
            enable_tts=True,
        )

        mock_event_bus.emit_sync.assert_called_once_with(
            "agent_message",
            debate_id="debate-123",
            agent="claude",
            content="I propose we use a distributed cache...",
            role="proposer",
            round_num=2,
            enable_tts=True,
        )

    def test_emit_agent_message_defaults(self, mock_event_bus: Mock) -> None:
        """emit_agent_message uses default values."""
        emitter = EventEmitter(event_bus=mock_event_bus)

        emitter.emit_agent_message(
            agent_name="gpt4",
            content="Critique message",
        )

        call_args = mock_event_bus.emit_sync.call_args
        assert call_args.kwargs["role"] == "proposer"
        assert call_args.kwargs["round_num"] == 0
        assert call_args.kwargs["enable_tts"] is True


class TestEmitPhaseStart:
    """Test emit_phase_start method."""

    def test_emit_phase_start(self, mock_event_bus: Mock) -> None:
        """emit_phase_start emits with correct parameters."""
        emitter = EventEmitter(event_bus=mock_event_bus)
        emitter.set_debate_id("debate-123")

        emitter.emit_phase_start(phase_name="proposal", round_number=1)

        mock_event_bus.emit_sync.assert_called_once_with(
            "phase_start",
            debate_id="debate-123",
            phase="proposal",
            round=1,
        )


class TestEmitPhaseEnd:
    """Test emit_phase_end method."""

    def test_emit_phase_end(self, mock_event_bus: Mock) -> None:
        """emit_phase_end emits with correct parameters."""
        emitter = EventEmitter(event_bus=mock_event_bus)
        emitter.set_debate_id("debate-123")

        emitter.emit_phase_end(
            phase_name="critique",
            round_number=2,
            duration_ms=1500,
            success=True,
        )

        mock_event_bus.emit_sync.assert_called_once_with(
            "phase_end",
            debate_id="debate-123",
            phase="critique",
            round=2,
            duration_ms=1500,
            success=True,
        )

    def test_emit_phase_end_failure(self, mock_event_bus: Mock) -> None:
        """emit_phase_end handles failure case."""
        emitter = EventEmitter(event_bus=mock_event_bus)

        emitter.emit_phase_end(
            phase_name="synthesis",
            round_number=3,
            duration_ms=500,
            success=False,
        )

        call_args = mock_event_bus.emit_sync.call_args
        assert call_args.kwargs["success"] is False


class TestEmitRoundStart:
    """Test emit_round_start method."""

    def test_emit_round_start(self, mock_event_bus: Mock) -> None:
        """emit_round_start emits with correct parameters."""
        emitter = EventEmitter(event_bus=mock_event_bus)
        emitter.set_debate_id("debate-123")

        emitter.emit_round_start(round_number=2, total_rounds=5)

        mock_event_bus.emit_sync.assert_called_once_with(
            "round_start",
            debate_id="debate-123",
            round=2,
            total_rounds=5,
        )


class TestEmitRoundEnd:
    """Test emit_round_end method."""

    def test_emit_round_end(self, mock_event_bus: Mock) -> None:
        """emit_round_end emits with correct parameters."""
        emitter = EventEmitter(event_bus=mock_event_bus)
        emitter.set_debate_id("debate-123")

        emitter.emit_round_end(
            round_number=3,
            total_rounds=5,
            duration_ms=30000,
        )

        mock_event_bus.emit_sync.assert_called_once_with(
            "round_end",
            debate_id="debate-123",
            round=3,
            total_rounds=5,
            duration_ms=30000,
        )


# === Edge Cases and Error Handling Tests ===


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_emit_events_with_none_dependencies(self) -> None:
        """All emit methods handle None dependencies gracefully."""
        emitter = EventEmitter()

        # None of these should raise
        emitter.notify_spectator("test")
        emitter.emit_moment(Mock(to_dict=Mock(return_value={})))
        emitter.broadcast_health_event({"status": "ok"})
        emitter.emit_agent_preview([], {})
        emitter.emit_calibration_update("agent", 0.1, 10, 0.9)
        emitter.emit_evidence_found("claim", "type", "src", 0.5)
        emitter.emit_trait_emerged("agent", "trait", "desc", 1)
        emitter.emit_risk_warning("type", "low", "desc")
        emitter.emit_rhetorical_observation("agent", "tech", "desc")
        emitter.emit_hollow_consensus(0.5, [], "rec")
        emitter.emit_trickster_intervention("type", "challenge")
        emitter.emit_agent_message("agent", "content")
        emitter.emit_phase_start("phase", 1)
        emitter.emit_phase_end("phase", 1, 100)
        emitter.emit_round_start(1, 3)
        emitter.emit_round_end(1, 3, 1000)

    def test_event_bus_takes_precedence_over_bridge(
        self,
        mock_event_bus: Mock,
        mock_event_bridge: Mock,
    ) -> None:
        """EventBus always takes precedence over event_bridge."""
        emitter = EventEmitter(
            event_bus=mock_event_bus,
            event_bridge=mock_event_bridge,
        )

        emitter.notify_spectator("test_event")

        mock_event_bus.emit_sync.assert_called_once()
        mock_event_bridge.notify.assert_not_called()

    def test_special_characters_in_event_data(self, mock_event_bus: Mock) -> None:
        """Events handle special characters in data."""
        emitter = EventEmitter(event_bus=mock_event_bus)

        emitter.emit_evidence_found(
            claim="Test with 'quotes' and \"double quotes\"",
            evidence_type="quote",
            source="Source with\nnewlines\tand tabs",
            confidence=0.9,
            excerpt="Unicode: \u2603 \u2764 \ud83d\ude00",
        )

        # Should not raise
        mock_event_bus.emit_sync.assert_called_once()

    def test_empty_agent_list_for_preview(
        self,
        full_emitter: EventEmitter,
        mock_hooks: dict,
    ) -> None:
        """emit_agent_preview handles empty agent list."""
        full_emitter.emit_agent_preview([], {})

        mock_hooks["on_agent_preview"].assert_called_once_with([])

    def test_large_number_of_agents(
        self,
        full_emitter: EventEmitter,
        mock_hooks: dict,
    ) -> None:
        """emit_agent_preview handles large number of agents."""
        agents = []
        for i in range(100):
            agent = Mock()
            agent.name = f"agent_{i}"
            agents.append(agent)

        full_emitter.emit_agent_preview(agents, {})

        call_args = mock_hooks["on_agent_preview"].call_args[0][0]
        assert len(call_args) == 100
