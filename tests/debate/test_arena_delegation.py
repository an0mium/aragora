"""
Tests for ArenaDelegation.

Tests the delegation of Arena methods to component subsystems:
- Context delegation (continuum memory)
- Checkpoint operations (outcome storage, evidence)
- Audience management (user events)
- Agent pool operations (calibration, critic selection)
- Role management (assignments, stances)
- Spectator notifications
- Grounded operations (positions, verdicts)
- Knowledge operations (context, outcome ingestion)
- Prompt building (persona, flip context)
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.arena_delegation import ArenaDelegation


# ============================================================================
# Mock Classes
# ============================================================================


class MockCheckpointOperations:
    """Mock CheckpointOperations for testing."""

    def __init__(self):
        self.stored_outcomes = []
        self.stored_evidence = []
        self.memory_outcomes_updated = []

    def store_debate_outcome(
        self, result: Any, task: str, belief_cruxes: list[str] | None = None
    ) -> None:
        self.stored_outcomes.append(
            {
                "result": result,
                "task": task,
                "belief_cruxes": belief_cruxes,
            }
        )

    def store_evidence(self, evidence_snippets: list, task: str) -> None:
        self.stored_evidence.append(
            {
                "snippets": evidence_snippets,
                "task": task,
            }
        )

    def update_memory_outcomes(self, result: Any) -> None:
        self.memory_outcomes_updated.append(result)


class MockContextDelegator:
    """Mock ContextDelegator for testing."""

    def __init__(self, context: str = "Test continuum context"):
        self._context = context

    def get_continuum_context(self) -> str:
        return self._context


class MockAudienceManager:
    """Mock AudienceManager for testing."""

    def __init__(self):
        self.events_handled = []
        self.drain_called = False

    def handle_event(self, event: Any) -> None:
        self.events_handled.append(event)

    def drain_events(self) -> None:
        self.drain_called = True


class MockAgentPool:
    """Mock AgentPool for testing."""

    def __init__(self):
        self.calibration_weights = {"agent_a": 1.3, "agent_b": 0.8}
        self.composite_scores = {
            ("agent_a", "general"): 1500.0,
            ("agent_b", "math"): 1200.0,
        }

    def _get_calibration_weight(self, agent_name: str) -> float:
        return self.calibration_weights.get(agent_name, 1.0)

    def _compute_composite_score(self, agent_name: str, domain: str = "general") -> float:
        return self.composite_scores.get((agent_name, domain), 1000.0)

    def select_critics(self, proposer: Any, candidates: list[Any] | None = None) -> list[Any]:
        # Return candidates without proposer
        if candidates is None:
            return []
        return [c for c in candidates if c != proposer][:2]


class MockRolesManager:
    """Mock RolesManager for testing."""

    def __init__(self):
        self.roles_assigned = False
        self.agreement_applied = False
        self.stances_assigned = []
        self.role_assignments_logged = []
        self.role_assignments_updated = []

    def assign_roles(self) -> None:
        self.roles_assigned = True

    def apply_agreement_intensity(self) -> None:
        self.agreement_applied = True

    def assign_stances(self, round_num: int = 0) -> None:
        self.stances_assigned.append(round_num)

    def get_stance_guidance(self, agent: Any) -> str:
        return f"Stance guidance for {getattr(agent, 'name', str(agent))}"

    def get_agreement_intensity_guidance(self) -> str:
        return "High agreement intensity expected"

    def get_role_context(self, agent: Any) -> str:
        return f"Role context for {getattr(agent, 'name', str(agent))}"

    def format_role_assignments_for_log(self) -> str:
        return "Role A -> agent_a\nRole B -> agent_b"

    def log_role_assignments(self, round_num: int) -> None:
        self.role_assignments_logged.append(round_num)

    def update_role_assignments(self, round_num: int) -> None:
        self.role_assignments_updated.append(round_num)


class MockSpectatorStream:
    """Mock SpectatorStream for testing."""

    def __init__(self, should_fail: bool = False):
        self.events_emitted = []
        self._should_fail = should_fail

    def emit(self, event_type: str, **kwargs: Any) -> None:
        if self._should_fail:
            raise RuntimeError("Spectator error")
        self.events_emitted.append({"type": event_type, **kwargs})


class MockGroundedOperations:
    """Mock GroundedOperations for testing."""

    def __init__(self):
        self.positions_recorded = []
        self.verdicts_created = []

    def record_position(
        self, agent_name: str, content: str, debate_id: str, round_num: int
    ) -> None:
        self.positions_recorded.append(
            {
                "agent_name": agent_name,
                "content": content,
                "debate_id": debate_id,
                "round_num": round_num,
            }
        )

    def create_grounded_verdict(self, result: Any) -> dict:
        self.verdicts_created.append(result)
        return {"verdict": "consensus", "confidence": 0.9}


class MockKnowledgeMoundOperations:
    """Mock KnowledgeMoundOperations for testing."""

    def __init__(self, context: str = "Relevant knowledge context"):
        self._context = context
        self.outcomes_ingested = []

    async def fetch_knowledge_context(
        self, task: str, limit: int = 10, auth_context: Any = None
    ) -> str:
        return self._context

    async def ingest_debate_outcome(self, result: Any) -> None:
        self.outcomes_ingested.append(result)


class MockPromptBuilder:
    """Mock PromptBuilder for testing."""

    def get_persona_context(self, agent: Any) -> str:
        return f"Persona: {getattr(agent, 'name', str(agent))}"

    def get_flip_context(self, agent: Any) -> str:
        return f"Flip context for {getattr(agent, 'name', str(agent))}"


class MockAgent:
    """Mock Agent for testing."""

    def __init__(self, name: str):
        self.name = name


class MockDebateResult:
    """Mock DebateResult for testing."""

    def __init__(self, consensus: str = "Test consensus"):
        self.final_answer = consensus
        self.messages = []
        self.votes = {}


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def checkpoint_ops():
    """Create mock checkpoint operations."""
    return MockCheckpointOperations()


@pytest.fixture
def context_delegator():
    """Create mock context delegator."""
    return MockContextDelegator()


@pytest.fixture
def audience_manager():
    """Create mock audience manager."""
    return MockAudienceManager()


@pytest.fixture
def agent_pool():
    """Create mock agent pool."""
    return MockAgentPool()


@pytest.fixture
def roles_manager():
    """Create mock roles manager."""
    return MockRolesManager()


@pytest.fixture
def delegation(checkpoint_ops, context_delegator, audience_manager, agent_pool, roles_manager):
    """Create ArenaDelegation with all required dependencies."""
    return ArenaDelegation(
        checkpoint_ops=checkpoint_ops,
        context_delegator=context_delegator,
        audience_manager=audience_manager,
        agent_pool=agent_pool,
        roles_manager=roles_manager,
    )


# ============================================================================
# Initialization Tests
# ============================================================================


class TestArenaDelegationInit:
    """Tests for ArenaDelegation initialization."""

    def test_init_with_required_dependencies(
        self, checkpoint_ops, context_delegator, audience_manager, agent_pool, roles_manager
    ):
        """Test initialization with required dependencies."""
        delegation = ArenaDelegation(
            checkpoint_ops=checkpoint_ops,
            context_delegator=context_delegator,
            audience_manager=audience_manager,
            agent_pool=agent_pool,
            roles_manager=roles_manager,
        )

        assert delegation._checkpoint_ops == checkpoint_ops
        assert delegation._context_delegator == context_delegator
        assert delegation._audience_manager == audience_manager
        assert delegation._agent_pool == agent_pool
        assert delegation._roles_manager == roles_manager
        assert delegation._spectator is None
        assert delegation._grounded_ops is None
        assert delegation._knowledge_ops is None
        assert delegation._prompt_builder is None

    def test_init_with_all_optional_dependencies(
        self, checkpoint_ops, context_delegator, audience_manager, agent_pool, roles_manager
    ):
        """Test initialization with all optional dependencies."""
        spectator = MockSpectatorStream()
        grounded_ops = MockGroundedOperations()
        knowledge_ops = MockKnowledgeMoundOperations()
        prompt_builder = MockPromptBuilder()

        delegation = ArenaDelegation(
            checkpoint_ops=checkpoint_ops,
            context_delegator=context_delegator,
            audience_manager=audience_manager,
            agent_pool=agent_pool,
            roles_manager=roles_manager,
            spectator=spectator,
            grounded_ops=grounded_ops,
            knowledge_ops=knowledge_ops,
            prompt_builder=prompt_builder,
        )

        assert delegation._spectator == spectator
        assert delegation._grounded_ops == grounded_ops
        assert delegation._knowledge_ops == knowledge_ops
        assert delegation._prompt_builder == prompt_builder


# ============================================================================
# Context Delegation Tests
# ============================================================================


class TestContextDelegation:
    """Tests for context delegation methods."""

    def test_get_continuum_context(self, delegation, context_delegator):
        """Test continuum context retrieval."""
        context = delegation.get_continuum_context()

        assert context == "Test continuum context"

    def test_get_continuum_context_with_different_content(
        self, checkpoint_ops, audience_manager, agent_pool, roles_manager
    ):
        """Test continuum context with custom content."""
        custom_context = "Custom memory context with historical data"
        context_delegator = MockContextDelegator(context=custom_context)

        delegation = ArenaDelegation(
            checkpoint_ops=checkpoint_ops,
            context_delegator=context_delegator,
            audience_manager=audience_manager,
            agent_pool=agent_pool,
            roles_manager=roles_manager,
        )

        context = delegation.get_continuum_context()
        assert context == custom_context


# ============================================================================
# Checkpoint Operations Tests
# ============================================================================


class TestCheckpointOperations:
    """Tests for checkpoint operation methods."""

    def test_store_debate_outcome(self, delegation, checkpoint_ops):
        """Test storing debate outcome."""
        result = MockDebateResult()
        task = "Test debate task"

        delegation.store_debate_outcome(result, task)

        assert len(checkpoint_ops.stored_outcomes) == 1
        stored = checkpoint_ops.stored_outcomes[0]
        assert stored["result"] == result
        assert stored["task"] == task
        assert stored["belief_cruxes"] is None

    def test_store_debate_outcome_with_belief_cruxes(self, delegation, checkpoint_ops):
        """Test storing debate outcome with belief cruxes."""
        result = MockDebateResult()
        task = "Test task"
        cruxes = ["crux1", "crux2", "crux3"]

        delegation.store_debate_outcome(result, task, belief_cruxes=cruxes)

        stored = checkpoint_ops.stored_outcomes[0]
        assert stored["belief_cruxes"] == ["crux1", "crux2", "crux3"]

    def test_store_debate_outcome_truncates_cruxes(self, delegation, checkpoint_ops):
        """Test that belief cruxes are truncated to 10."""
        result = MockDebateResult()
        task = "Test task"
        cruxes = [f"crux{i}" for i in range(20)]  # 20 cruxes

        delegation.store_debate_outcome(result, task, belief_cruxes=cruxes)

        stored = checkpoint_ops.stored_outcomes[0]
        assert len(stored["belief_cruxes"]) == 10

    def test_store_debate_outcome_converts_cruxes_to_strings(self, delegation, checkpoint_ops):
        """Test that belief cruxes are converted to strings."""
        result = MockDebateResult()
        task = "Test task"
        # Mix of types
        cruxes = [123, {"key": "value"}, ["list"], "string"]

        delegation.store_debate_outcome(result, task, belief_cruxes=cruxes)

        stored = checkpoint_ops.stored_outcomes[0]
        assert all(isinstance(c, str) for c in stored["belief_cruxes"])

    def test_store_evidence(self, delegation, checkpoint_ops):
        """Test storing evidence."""
        evidence = [{"text": "Evidence 1"}, {"text": "Evidence 2"}]
        task = "Test task"

        delegation.store_evidence(evidence, task)

        assert len(checkpoint_ops.stored_evidence) == 1
        stored = checkpoint_ops.stored_evidence[0]
        assert stored["snippets"] == evidence
        assert stored["task"] == task

    def test_update_memory_outcomes(self, delegation, checkpoint_ops):
        """Test updating memory outcomes."""
        result = MockDebateResult()

        delegation.update_memory_outcomes(result)

        assert len(checkpoint_ops.memory_outcomes_updated) == 1
        assert checkpoint_ops.memory_outcomes_updated[0] == result


# ============================================================================
# Audience Management Tests
# ============================================================================


class TestAudienceManagement:
    """Tests for audience management methods."""

    def test_handle_user_event(self, delegation, audience_manager):
        """Test handling user event."""
        event = MagicMock(type="USER_VOTE", data={"agent": "agent_a"})

        delegation.handle_user_event(event)

        assert len(audience_manager.events_handled) == 1
        assert audience_manager.events_handled[0] == event

    def test_drain_user_events(self, delegation, audience_manager):
        """Test draining user events."""
        delegation.drain_user_events()

        assert audience_manager.drain_called is True


# ============================================================================
# Agent Pool Operations Tests
# ============================================================================


class TestAgentPoolOperations:
    """Tests for agent pool operation methods."""

    def test_get_calibration_weight(self, delegation):
        """Test getting calibration weight."""
        weight = delegation.get_calibration_weight("agent_a")
        assert weight == 1.3

        weight = delegation.get_calibration_weight("unknown")
        assert weight == 1.0

    def test_compute_composite_judge_score(self, delegation):
        """Test computing composite judge score."""
        score = delegation.compute_composite_judge_score("agent_a", "general")
        assert score == 1500.0

        score = delegation.compute_composite_judge_score("agent_b", "math")
        assert score == 1200.0

        score = delegation.compute_composite_judge_score("unknown", "general")
        assert score == 1000.0

    def test_select_critics_for_proposal(self, delegation):
        """Test selecting critics for proposal."""
        agents = [
            MockAgent("agent_a"),
            MockAgent("agent_b"),
            MockAgent("agent_c"),
        ]

        critics = delegation.select_critics_for_proposal("agent_a", agents)

        # Should not include proposer
        critic_names = [c.name for c in critics]
        assert "agent_a" not in critic_names

    def test_select_critics_for_proposal_proposer_not_in_list(self, delegation):
        """Test selecting critics when proposer not in candidate list."""
        agents = [
            MockAgent("agent_b"),
            MockAgent("agent_c"),
        ]

        critics = delegation.select_critics_for_proposal("agent_a", agents)

        # Falls back to first agent as proposer
        assert len(critics) <= 2

    def test_select_critics_for_proposal_empty_list(self, delegation):
        """Test selecting critics with empty list."""
        critics = delegation.select_critics_for_proposal("agent_a", [])

        assert critics == []


# ============================================================================
# Role Management Tests
# ============================================================================


class TestRoleManagement:
    """Tests for role management methods."""

    def test_assign_roles(self, delegation, roles_manager):
        """Test assigning roles."""
        delegation.assign_roles()

        assert roles_manager.roles_assigned is True

    def test_apply_agreement_intensity(self, delegation, roles_manager):
        """Test applying agreement intensity."""
        delegation.apply_agreement_intensity()

        assert roles_manager.agreement_applied is True

    def test_assign_stances(self, delegation, roles_manager):
        """Test assigning stances."""
        delegation.assign_stances(round_num=2)

        assert 2 in roles_manager.stances_assigned

    def test_assign_stances_default_round(self, delegation, roles_manager):
        """Test assigning stances with default round."""
        delegation.assign_stances()

        assert 0 in roles_manager.stances_assigned

    def test_get_stance_guidance(self, delegation):
        """Test getting stance guidance."""
        agent = MockAgent("test_agent")

        guidance = delegation.get_stance_guidance(agent)

        assert "test_agent" in guidance

    def test_get_agreement_intensity_guidance(self, delegation):
        """Test getting agreement intensity guidance."""
        guidance = delegation.get_agreement_intensity_guidance()

        assert "High agreement intensity" in guidance

    def test_get_role_context(self, delegation):
        """Test getting role context."""
        agent = MockAgent("test_agent")

        context = delegation.get_role_context(agent)

        assert "test_agent" in context

    def test_format_role_assignments_for_log(self, delegation):
        """Test formatting role assignments for log."""
        formatted = delegation.format_role_assignments_for_log()

        assert "Role A" in formatted
        assert "agent_a" in formatted

    def test_log_role_assignments(self, delegation, roles_manager):
        """Test logging role assignments."""
        delegation.log_role_assignments(round_num=3)

        assert 3 in roles_manager.role_assignments_logged

    def test_update_role_assignments(self, delegation, roles_manager):
        """Test updating role assignments."""
        delegation.update_role_assignments(round_num=4)

        assert 4 in roles_manager.role_assignments_updated


# ============================================================================
# Spectator Notification Tests
# ============================================================================


class TestSpectatorNotifications:
    """Tests for spectator notification methods."""

    def test_notify_spectator_success(
        self, checkpoint_ops, context_delegator, audience_manager, agent_pool, roles_manager
    ):
        """Test successful spectator notification."""
        spectator = MockSpectatorStream()
        delegation = ArenaDelegation(
            checkpoint_ops=checkpoint_ops,
            context_delegator=context_delegator,
            audience_manager=audience_manager,
            agent_pool=agent_pool,
            roles_manager=roles_manager,
            spectator=spectator,
        )

        delegation.notify_spectator("debate_start", task="Test task", round=1)

        assert len(spectator.events_emitted) == 1
        event = spectator.events_emitted[0]
        assert event["type"] == "debate_start"
        assert event["task"] == "Test task"
        assert event["round"] == 1

    def test_notify_spectator_no_spectator(self, delegation):
        """Test notification when no spectator configured."""
        # Should not raise
        delegation.notify_spectator("debate_start", task="Test task")

    def test_notify_spectator_handles_error(
        self, checkpoint_ops, context_delegator, audience_manager, agent_pool, roles_manager
    ):
        """Test spectator notification handles errors gracefully."""
        spectator = MockSpectatorStream(should_fail=True)
        delegation = ArenaDelegation(
            checkpoint_ops=checkpoint_ops,
            context_delegator=context_delegator,
            audience_manager=audience_manager,
            agent_pool=agent_pool,
            roles_manager=roles_manager,
            spectator=spectator,
        )

        # Should not raise, just log debug
        delegation.notify_spectator("debate_start")


# ============================================================================
# Grounded Operations Tests
# ============================================================================


class TestGroundedOperations:
    """Tests for grounded operation methods."""

    def test_record_grounded_position(
        self, checkpoint_ops, context_delegator, audience_manager, agent_pool, roles_manager
    ):
        """Test recording grounded position."""
        grounded_ops = MockGroundedOperations()
        delegation = ArenaDelegation(
            checkpoint_ops=checkpoint_ops,
            context_delegator=context_delegator,
            audience_manager=audience_manager,
            agent_pool=agent_pool,
            roles_manager=roles_manager,
            grounded_ops=grounded_ops,
        )

        delegation.record_grounded_position(
            agent_name="agent_a",
            position_text="I believe X because Y",
            round_num=2,
            debate_id="debate-123",
        )

        assert len(grounded_ops.positions_recorded) == 1
        position = grounded_ops.positions_recorded[0]
        assert position["agent_name"] == "agent_a"
        assert position["content"] == "I believe X because Y"
        assert position["round_num"] == 2
        assert position["debate_id"] == "debate-123"

    def test_record_grounded_position_with_evidence_ids(
        self, checkpoint_ops, context_delegator, audience_manager, agent_pool, roles_manager
    ):
        """Test recording grounded position with evidence IDs (for API compat)."""
        grounded_ops = MockGroundedOperations()
        delegation = ArenaDelegation(
            checkpoint_ops=checkpoint_ops,
            context_delegator=context_delegator,
            audience_manager=audience_manager,
            agent_pool=agent_pool,
            roles_manager=roles_manager,
            grounded_ops=grounded_ops,
        )

        # evidence_ids accepted but not used
        delegation.record_grounded_position(
            agent_name="agent_a",
            position_text="Position text",
            round_num=1,
            evidence_ids=["ev1", "ev2"],
        )

        assert len(grounded_ops.positions_recorded) == 1

    def test_record_grounded_position_no_grounded_ops(self, delegation):
        """Test recording position when no grounded ops configured."""
        # Should not raise
        delegation.record_grounded_position(
            agent_name="agent_a",
            position_text="Position",
            round_num=1,
        )

    def test_create_grounded_verdict(
        self, checkpoint_ops, context_delegator, audience_manager, agent_pool, roles_manager
    ):
        """Test creating grounded verdict."""
        grounded_ops = MockGroundedOperations()
        delegation = ArenaDelegation(
            checkpoint_ops=checkpoint_ops,
            context_delegator=context_delegator,
            audience_manager=audience_manager,
            agent_pool=agent_pool,
            roles_manager=roles_manager,
            grounded_ops=grounded_ops,
        )

        result = MockDebateResult()
        verdict = delegation.create_grounded_verdict(result)

        assert verdict is not None
        assert verdict["verdict"] == "consensus"
        assert verdict["confidence"] == 0.9
        assert result in grounded_ops.verdicts_created

    def test_create_grounded_verdict_no_grounded_ops(self, delegation):
        """Test creating verdict when no grounded ops configured."""
        result = MockDebateResult()
        verdict = delegation.create_grounded_verdict(result)

        assert verdict is None


# ============================================================================
# Knowledge Operations Tests
# ============================================================================


class TestKnowledgeOperations:
    """Tests for knowledge operation methods."""

    @pytest.mark.asyncio
    async def test_fetch_knowledge_context(
        self, checkpoint_ops, context_delegator, audience_manager, agent_pool, roles_manager
    ):
        """Test fetching knowledge context."""
        knowledge_ops = MockKnowledgeMoundOperations()
        delegation = ArenaDelegation(
            checkpoint_ops=checkpoint_ops,
            context_delegator=context_delegator,
            audience_manager=audience_manager,
            agent_pool=agent_pool,
            roles_manager=roles_manager,
            knowledge_ops=knowledge_ops,
        )

        context = await delegation.fetch_knowledge_context("Test task", limit=5)

        assert context == "Relevant knowledge context"

    @pytest.mark.asyncio
    async def test_fetch_knowledge_context_no_knowledge_ops(self, delegation):
        """Test fetching context when no knowledge ops configured."""
        context = await delegation.fetch_knowledge_context("Test task")

        assert context is None

    @pytest.mark.asyncio
    async def test_ingest_debate_outcome(
        self, checkpoint_ops, context_delegator, audience_manager, agent_pool, roles_manager
    ):
        """Test ingesting debate outcome into knowledge."""
        knowledge_ops = MockKnowledgeMoundOperations()
        delegation = ArenaDelegation(
            checkpoint_ops=checkpoint_ops,
            context_delegator=context_delegator,
            audience_manager=audience_manager,
            agent_pool=agent_pool,
            roles_manager=roles_manager,
            knowledge_ops=knowledge_ops,
        )

        result = MockDebateResult()
        await delegation.ingest_debate_outcome(result)

        assert result in knowledge_ops.outcomes_ingested

    @pytest.mark.asyncio
    async def test_ingest_debate_outcome_no_knowledge_ops(self, delegation):
        """Test ingesting outcome when no knowledge ops configured."""
        result = MockDebateResult()
        # Should not raise
        await delegation.ingest_debate_outcome(result)


# ============================================================================
# Prompt Building Tests
# ============================================================================


class TestPromptBuilding:
    """Tests for prompt building methods."""

    def test_get_persona_context(
        self, checkpoint_ops, context_delegator, audience_manager, agent_pool, roles_manager
    ):
        """Test getting persona context."""
        prompt_builder = MockPromptBuilder()
        delegation = ArenaDelegation(
            checkpoint_ops=checkpoint_ops,
            context_delegator=context_delegator,
            audience_manager=audience_manager,
            agent_pool=agent_pool,
            roles_manager=roles_manager,
            prompt_builder=prompt_builder,
        )

        agent = MockAgent("expert_agent")
        context = delegation.get_persona_context(agent)

        assert "expert_agent" in context

    def test_get_persona_context_no_prompt_builder(self, delegation):
        """Test getting persona context when no prompt builder configured."""
        agent = MockAgent("test_agent")
        context = delegation.get_persona_context(agent)

        assert context == ""

    def test_get_flip_context(
        self, checkpoint_ops, context_delegator, audience_manager, agent_pool, roles_manager
    ):
        """Test getting flip context."""
        prompt_builder = MockPromptBuilder()
        delegation = ArenaDelegation(
            checkpoint_ops=checkpoint_ops,
            context_delegator=context_delegator,
            audience_manager=audience_manager,
            agent_pool=agent_pool,
            roles_manager=roles_manager,
            prompt_builder=prompt_builder,
        )

        agent = MockAgent("flip_agent")
        context = delegation.get_flip_context(agent)

        assert "flip_agent" in context

    def test_get_flip_context_no_prompt_builder(self, delegation):
        """Test getting flip context when no prompt builder configured."""
        agent = MockAgent("test_agent")
        context = delegation.get_flip_context(agent)

        assert context == ""


# ============================================================================
# Integration Tests
# ============================================================================


class TestArenaDelegationIntegration:
    """Integration tests for ArenaDelegation."""

    def test_full_debate_workflow(
        self, checkpoint_ops, context_delegator, audience_manager, agent_pool, roles_manager
    ):
        """Test full debate workflow using delegation."""
        spectator = MockSpectatorStream()
        grounded_ops = MockGroundedOperations()
        prompt_builder = MockPromptBuilder()

        delegation = ArenaDelegation(
            checkpoint_ops=checkpoint_ops,
            context_delegator=context_delegator,
            audience_manager=audience_manager,
            agent_pool=agent_pool,
            roles_manager=roles_manager,
            spectator=spectator,
            grounded_ops=grounded_ops,
            prompt_builder=prompt_builder,
        )

        # Setup phase
        delegation.assign_roles()
        delegation.assign_stances(round_num=0)
        assert roles_manager.roles_assigned is True

        # Get context for debate
        context = delegation.get_continuum_context()
        assert context is not None

        # Simulate debate round
        agent = MockAgent("debater_1")
        persona = delegation.get_persona_context(agent)
        assert "debater_1" in persona

        stance = delegation.get_stance_guidance(agent)
        assert "debater_1" in stance

        # Record position
        delegation.record_grounded_position(
            agent_name="debater_1",
            position_text="My argument is...",
            round_num=1,
            debate_id="test-debate",
        )
        assert len(grounded_ops.positions_recorded) == 1

        # Handle audience participation
        event = MagicMock(type="USER_VOTE", data={"agent": "debater_1"})
        delegation.handle_user_event(event)
        delegation.drain_user_events()
        assert audience_manager.drain_called is True

        # Notify spectators
        delegation.notify_spectator("round_end", round=1)
        assert len(spectator.events_emitted) == 1

        # Store outcome
        result = MockDebateResult(consensus="Final consensus")
        delegation.store_debate_outcome(result, task="Test debate")
        assert len(checkpoint_ops.stored_outcomes) == 1

        # Create verdict
        verdict = delegation.create_grounded_verdict(result)
        assert verdict is not None

    @pytest.mark.asyncio
    async def test_knowledge_workflow(
        self, checkpoint_ops, context_delegator, audience_manager, agent_pool, roles_manager
    ):
        """Test knowledge operations workflow."""
        knowledge_ops = MockKnowledgeMoundOperations(context="Domain-specific knowledge")

        delegation = ArenaDelegation(
            checkpoint_ops=checkpoint_ops,
            context_delegator=context_delegator,
            audience_manager=audience_manager,
            agent_pool=agent_pool,
            roles_manager=roles_manager,
            knowledge_ops=knowledge_ops,
        )

        # Fetch context before debate
        context = await delegation.fetch_knowledge_context("What is X?", limit=20)
        assert context == "Domain-specific knowledge"

        # Ingest outcome after debate
        result = MockDebateResult(consensus="X is Y")
        await delegation.ingest_debate_outcome(result)
        assert result in knowledge_ops.outcomes_ingested
