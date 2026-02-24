"""
Comprehensive tests for the Aragora debate orchestration engine (Arena).

Covers:
- Arena initialization with various configs
- Debate flow (propose, critique, revise, vote)
- Consensus detection and proof generation
- Timeout handling
- Agent team selection
- Error handling (agent failures, API errors)
- Factory methods (from_config, from_configs, create)
- Async context manager lifecycle
- Protocol configuration
- Partial consensus tracking

All external dependencies (API calls, agents, etc.) are mocked.
No API keys or network access required.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from aragora.core_types import (
    Agent,
    Critique,
    DebateResult,
    Environment,
    Message,
    Vote,
)
from aragora.debate.arena_config import ArenaConfig, DebateConfig, AgentConfig
from aragora.debate.consensus import (
    Claim,
    ConsensusBuilder,
    ConsensusProof,
    ConsensusVote,
    DissentRecord,
    Evidence,
    PartialConsensus,
    PartialConsensusItem,
    UnresolvedTension,
    VoteType,
    build_partial_consensus,
)
from aragora.debate.protocol import (
    DebateProtocol,
    RoundPhase,
    user_vote_multiplier,
)


# =============================================================================
# Helpers
# =============================================================================


class MockAgent(Agent):
    """Concrete mock agent for testing."""

    def __init__(self, name: str = "mock-agent", model: str = "mock-model"):
        super().__init__(name=name, model=model, role="proposer")

    async def generate(self, prompt: str, context=None) -> str:
        return f"Response from {self.name}"

    async def critique(self, proposal, task, context=None, target_agent=None) -> Critique:
        return Critique(
            agent=self.name,
            target_agent=target_agent or "unknown",
            target_content=proposal,
            issues=["Minor issue"],
            suggestions=["Consider improving"],
            severity=3.0,
            reasoning="Standard critique",
        )


def _make_env(task: str = "Design a rate limiter") -> Environment:
    return Environment(task=task)


def _make_agents(count: int = 3) -> list[MockAgent]:
    names = ["claude", "gpt4", "gemini", "grok", "mistral"]
    return [MockAgent(name=names[i], model=f"model-{i}") for i in range(count)]


def _make_protocol(**overrides) -> DebateProtocol:
    defaults = {
        "rounds": 2,
        "consensus": "majority",
        "timeout_seconds": 0,
        "use_structured_phases": False,
        "convergence_detection": False,
        "early_stopping": False,
        "enable_trickster": False,
        "enable_rhetorical_observer": False,
        "enable_calibration": False,
        "enable_evolution": False,
        "enable_research": False,
        "role_rotation": False,
        "role_matching": False,
        "enable_breakpoints": False,
        "enable_evidence_weighting": False,
        "verify_claims_during_consensus": False,
        "enable_molecule_tracking": False,
        "enable_agent_channels": False,
    }
    defaults.update(overrides)
    return DebateProtocol(**defaults)


# =============================================================================
# Arena Initialization Tests
# =============================================================================


class TestArenaInitialization:
    """Tests for Arena.__init__ with various configurations."""

    def test_basic_init(self):
        """Arena initializes with minimal arguments."""
        from aragora.debate.orchestrator import Arena

        env = _make_env()
        agents = _make_agents(2)
        protocol = _make_protocol()

        arena = Arena(environment=env, agents=agents, protocol=protocol)

        assert arena.env is env
        assert len(arena.agents) >= 2
        assert arena.protocol is not None

    def test_init_with_no_agents_raises(self):
        """Arena raises ValueError when no agents provided."""
        from aragora.debate.orchestrator import Arena

        env = _make_env()
        protocol = _make_protocol()

        with pytest.raises(ValueError, match="agents"):
            Arena(environment=env, agents=[], protocol=protocol)

    def test_init_with_protocol_defaults(self):
        """Arena uses default protocol when none specified."""
        from aragora.debate.orchestrator import Arena

        env = _make_env()
        agents = _make_agents(2)

        arena = Arena(environment=env, agents=agents)

        assert arena.protocol is not None
        assert arena.protocol.rounds > 0

    def test_init_with_debate_config(self):
        """Arena accepts DebateConfig for protocol settings."""
        from aragora.debate.orchestrator import Arena

        env = _make_env()
        agents = _make_agents(2)
        debate_config = DebateConfig(rounds=5, consensus_threshold=0.8)
        protocol = _make_protocol()

        arena = Arena(
            environment=env,
            agents=agents,
            protocol=protocol,
            debate_config=debate_config,
        )

        assert arena.env.task == "Design a rate limiter"

    def test_init_with_agent_config(self):
        """Arena accepts AgentConfig for agent management."""
        from aragora.debate.orchestrator import Arena

        env = _make_env()
        agents = _make_agents(2)
        agent_config = AgentConfig(agent_weights={"claude": 1.2, "gpt4": 1.0})
        protocol = _make_protocol()

        arena = Arena(
            environment=env,
            agents=agents,
            protocol=protocol,
            agent_config=agent_config,
        )

        assert arena.agent_weights == {"claude": 1.2, "gpt4": 1.0}

    def test_init_with_loop_id(self):
        """Arena stores loop_id for multi-debate sessions."""
        from aragora.debate.orchestrator import Arena

        env = _make_env()
        agents = _make_agents(2)
        protocol = _make_protocol()

        arena = Arena(
            environment=env,
            agents=agents,
            protocol=protocol,
            loop_id="test-loop-123",
        )

        assert arena.loop_id == "test-loop-123"

    def test_init_with_org_and_user(self):
        """Arena stores org_id and user_id for billing/audit."""
        from aragora.debate.orchestrator import Arena

        env = _make_env()
        agents = _make_agents(2)
        protocol = _make_protocol()

        arena = Arena(
            environment=env,
            agents=agents,
            protocol=protocol,
            org_id="org-test",
            user_id="user-test",
        )

        assert arena.org_id == "org-test"
        assert arena.user_id == "user-test"

    def test_init_with_circuit_breaker(self):
        """Arena accepts a CircuitBreaker for fault tolerance."""
        from aragora.debate.orchestrator import Arena
        from aragora.resilience import CircuitBreaker

        env = _make_env()
        agents = _make_agents(2)
        protocol = _make_protocol()
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=60.0)

        arena = Arena(
            environment=env,
            agents=agents,
            protocol=protocol,
            circuit_breaker=cb,
        )

        assert arena.circuit_breaker is cb


# =============================================================================
# Arena Factory Method Tests
# =============================================================================


class TestArenaFactoryMethods:
    """Tests for Arena.from_config, from_configs, and create."""

    def test_from_config(self):
        """Arena.from_config creates arena from ArenaConfig."""
        from aragora.debate.orchestrator import Arena

        env = _make_env()
        agents = _make_agents(2)
        protocol = _make_protocol()
        config = ArenaConfig(loop_id="from-config-test")

        arena = Arena.from_config(env, agents, protocol, config)

        assert isinstance(arena, Arena)
        assert arena.loop_id == "from-config-test"

    def test_from_configs(self):
        """Arena.from_configs creates arena from grouped config objects."""
        from aragora.debate.orchestrator import Arena

        env = _make_env()
        agents = _make_agents(2)
        protocol = _make_protocol()

        arena = Arena.from_configs(
            env,
            agents,
            protocol,
            debate_config=DebateConfig(rounds=3),
            agent_config=AgentConfig(use_airlock=False),
        )

        assert isinstance(arena, Arena)

    def test_create(self):
        """Arena.create produces a working Arena."""
        from aragora.debate.orchestrator import Arena

        env = _make_env()
        agents = _make_agents(2)
        protocol = _make_protocol()

        arena = Arena.create(env, agents, protocol)

        assert isinstance(arena, Arena)


# =============================================================================
# Arena Lifecycle (async context manager) Tests
# =============================================================================


class TestArenaLifecycle:
    """Tests for Arena async context manager and cleanup."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Arena works as async context manager."""
        from aragora.debate.orchestrator import Arena

        env = _make_env()
        agents = _make_agents(2)
        protocol = _make_protocol()

        async with Arena(environment=env, agents=agents, protocol=protocol) as arena:
            assert arena is not None
            assert arena.env.task == "Design a rate limiter"

    @pytest.mark.asyncio
    async def test_cleanup_called_on_exit(self):
        """Arena._cleanup is called on context manager exit."""
        from aragora.debate.orchestrator import Arena

        env = _make_env()
        agents = _make_agents(2)
        protocol = _make_protocol()

        arena = Arena(environment=env, agents=agents, protocol=protocol)
        # Patch _cleanup to verify it gets called
        arena._cleanup = AsyncMock()

        async with arena:
            pass

        arena._cleanup.assert_awaited_once()


# =============================================================================
# Debate Execution Tests
# =============================================================================


class TestDebateExecution:
    """Tests for Arena.run() debate execution flow."""

    @pytest.mark.asyncio
    async def test_run_returns_debate_result(self):
        """Arena.run() returns a DebateResult."""
        from aragora.debate.orchestrator import Arena

        env = _make_env()
        agents = _make_agents(2)
        protocol = _make_protocol()

        arena = Arena(environment=env, agents=agents, protocol=protocol)

        # Mock the internal run to avoid real execution
        mock_result = DebateResult(
            task=env.task,
            messages=[],
            critiques=[],
            votes=[],
            dissenting_views=[],
            rounds_used=2,
            consensus_reached=True,
            confidence=0.85,
            final_answer="Use token bucket algorithm",
        )
        arena._run_inner = AsyncMock(return_value=mock_result)

        result = await arena.run()

        assert isinstance(result, DebateResult)
        assert result.task == "Design a rate limiter"
        assert result.consensus_reached is True
        assert result.confidence == 0.85

    @pytest.mark.asyncio
    async def test_run_with_timeout(self):
        """Arena.run() respects timeout_seconds and returns partial result."""
        from aragora.debate.orchestrator import Arena

        env = _make_env()
        agents = _make_agents(2)
        protocol = _make_protocol(timeout_seconds=1)

        arena = Arena(environment=env, agents=agents, protocol=protocol)

        # Mock _run_inner to sleep longer than timeout
        async def slow_run(correlation_id=""):
            await asyncio.sleep(10)
            return DebateResult(task=env.task)

        arena._run_inner = slow_run

        result = await arena.run()

        # Should get a partial result due to timeout
        assert isinstance(result, DebateResult)
        assert result.task == env.task

    @pytest.mark.asyncio
    async def test_run_with_correlation_id(self):
        """Arena.run() passes correlation_id to _run_inner."""
        from aragora.debate.orchestrator import Arena

        env = _make_env()
        agents = _make_agents(2)
        protocol = _make_protocol()

        arena = Arena(environment=env, agents=agents, protocol=protocol)
        mock_result = DebateResult(task=env.task)
        arena._run_inner = AsyncMock(return_value=mock_result)

        await arena.run(correlation_id="test-corr-123")

        arena._run_inner.assert_awaited_once_with(correlation_id="test-corr-123")

    @pytest.mark.asyncio
    async def test_run_no_timeout_when_zero(self):
        """Arena.run() skips asyncio.wait_for when timeout_seconds is 0."""
        from aragora.debate.orchestrator import Arena

        env = _make_env()
        agents = _make_agents(2)
        protocol = _make_protocol(timeout_seconds=0)

        arena = Arena(environment=env, agents=agents, protocol=protocol)
        mock_result = DebateResult(
            task=env.task,
            rounds_used=2,
            consensus_reached=True,
        )
        arena._run_inner = AsyncMock(return_value=mock_result)

        result = await arena.run()

        assert result.consensus_reached is True


# =============================================================================
# Protocol Configuration Tests
# =============================================================================


class TestProtocolConfiguration:
    """Tests for DebateProtocol settings and their impact."""

    def test_default_protocol(self):
        """Default DebateProtocol has reasonable defaults."""
        protocol = DebateProtocol()

        assert protocol.rounds > 0
        assert protocol.consensus in (
            "majority", "unanimous", "judge", "none",
            "weighted", "supermajority", "any", "byzantine",
        )
        assert 0.0 <= protocol.consensus_threshold <= 1.0
        assert protocol.timeout_seconds > 0

    def test_quick_protocol(self):
        """Quick protocol for fast debates."""
        protocol = DebateProtocol(
            rounds=3,
            consensus="majority",
            use_structured_phases=False,
            timeout_seconds=300,
        )

        assert protocol.rounds == 3
        assert protocol.consensus == "majority"

    def test_high_assurance_protocol(self):
        """High-assurance protocol with strict consensus."""
        protocol = DebateProtocol(
            consensus="supermajority",
            consensus_threshold=0.8,
            enable_trickster=True,
        )

        assert protocol.consensus == "supermajority"
        assert protocol.consensus_threshold == 0.8
        assert protocol.enable_trickster is True

    def test_structured_phases(self):
        """Structured phases provide cognitive mode per round."""
        protocol = DebateProtocol(use_structured_phases=True)

        phase0 = protocol.get_round_phase(0)
        assert phase0 is not None
        assert phase0.name == "Context Gathering"
        assert phase0.cognitive_mode == "Researcher"

        phase1 = protocol.get_round_phase(1)
        assert phase1 is not None
        assert phase1.cognitive_mode == "Analyst"

    def test_round_phase_out_of_range(self):
        """Out-of-range round number returns None."""
        protocol = DebateProtocol(use_structured_phases=True)

        assert protocol.get_round_phase(999) is None

    def test_round_phase_disabled(self):
        """Disabled structured phases returns None."""
        protocol = DebateProtocol(use_structured_phases=False)

        assert protocol.get_round_phase(0) is None

    def test_with_gold_path(self):
        """Protocol.with_gold_path creates plan-enabled protocol."""
        protocol = DebateProtocol.with_gold_path(
            min_confidence=0.8,
            approval_mode="risk_based",
        )

        assert protocol.auto_create_plan is True
        assert protocol.plan_min_confidence == 0.8
        assert protocol.plan_approval_mode == "risk_based"

    def test_with_full_flywheel(self):
        """Protocol.with_full_flywheel enables all feedback loops."""
        protocol = DebateProtocol.with_full_flywheel()

        assert protocol.enable_adaptive_consensus is True
        assert protocol.enable_synthesis is True
        assert protocol.enable_knowledge_injection is True
        assert protocol.enable_trickster is True
        assert protocol.auto_create_plan is True

    def test_user_vote_multiplier_neutral(self):
        """Neutral intensity returns multiplier of 1.0."""
        protocol = DebateProtocol()

        result = user_vote_multiplier(protocol.user_vote_intensity_neutral, protocol)

        assert result == 1.0

    def test_user_vote_multiplier_low(self):
        """Low intensity returns multiplier below 1.0."""
        protocol = DebateProtocol()

        result = user_vote_multiplier(1, protocol)

        assert result == protocol.user_vote_intensity_min_multiplier

    def test_user_vote_multiplier_high(self):
        """High intensity returns multiplier above 1.0."""
        protocol = DebateProtocol()

        result = user_vote_multiplier(10, protocol)

        assert result == protocol.user_vote_intensity_max_multiplier


# =============================================================================
# Consensus Detection Tests
# =============================================================================


class TestConsensusDetection:
    """Tests for ConsensusBuilder and ConsensusProof."""

    def test_consensus_builder_basic(self):
        """ConsensusBuilder builds a proof with claims and votes."""
        builder = ConsensusBuilder(debate_id="debate-1", task="Test task")

        claim = builder.add_claim("The sky is blue", author="claude", confidence=0.9)
        builder.add_evidence(
            claim.claim_id, source="claude", content="Visual observation",
            supports=True, strength=0.8,
        )
        builder.record_vote(
            agent="claude", vote=VoteType.AGREE,
            confidence=0.9, reasoning="I agree",
        )
        builder.record_vote(
            agent="gpt4", vote=VoteType.AGREE,
            confidence=0.85, reasoning="Confirmed",
        )

        proof = builder.build(
            final_claim="The sky is blue",
            confidence=0.9,
            consensus_reached=True,
            reasoning_summary="All agents agree",
            rounds=3,
        )

        assert isinstance(proof, ConsensusProof)
        assert proof.consensus_reached is True
        assert proof.confidence == 0.9
        assert len(proof.supporting_agents) == 2
        assert len(proof.dissenting_agents) == 0
        assert proof.agreement_ratio == 1.0
        assert proof.has_strong_consensus is True

    def test_consensus_with_dissent(self):
        """ConsensusBuilder captures dissenting views."""
        builder = ConsensusBuilder(debate_id="debate-2", task="Controversial topic")

        claim = builder.add_claim("Approach A is best", author="claude", confidence=0.7)
        builder.record_vote("claude", VoteType.AGREE, 0.8, "Strong supporter")
        builder.record_vote("gpt4", VoteType.DISAGREE, 0.6, "I disagree")
        builder.record_dissent(
            agent="gpt4",
            claim_id=claim.claim_id,
            reasons=["Evidence is weak", "Better alternatives exist"],
            severity=0.7,
        )

        proof = builder.build(
            final_claim="Approach A is best",
            confidence=0.6,
            consensus_reached=False,
            reasoning_summary="Divided opinions",
        )

        assert proof.consensus_reached is False
        assert len(proof.dissenting_agents) == 1
        assert "gpt4" in proof.dissenting_agents
        assert len(proof.dissents) == 1
        assert proof.has_strong_consensus is False
        assert proof.agreement_ratio == 0.5

    def test_consensus_proof_checksum(self):
        """ConsensusProof generates stable checksum."""
        builder = ConsensusBuilder(debate_id="debate-3", task="Test")
        builder.add_claim("Claim X", author="agent1", confidence=0.5)
        builder.record_vote("agent1", VoteType.AGREE, 0.8, "Yes")

        proof = builder.build("Claim X", 0.8, True, "Summary")

        checksum1 = proof.checksum
        checksum2 = proof.checksum  # cached

        assert checksum1 == checksum2
        assert len(checksum1) == 16  # SHA-256 truncated to 16 chars

    def test_consensus_proof_serialization(self):
        """ConsensusProof serializes to dict and JSON."""
        builder = ConsensusBuilder(debate_id="debate-4", task="Serialize test")
        builder.record_vote("agent1", VoteType.AGREE, 0.9, "Agreed")

        proof = builder.build("Final claim", 0.9, True, "All agreed")

        d = proof.to_dict()
        assert d["consensus_reached"] is True
        assert d["final_claim"] == "Final claim"
        assert "checksum" in d

        json_str = proof.to_json()
        assert '"consensus_reached": true' in json_str

    def test_consensus_proof_markdown(self):
        """ConsensusProof generates readable Markdown."""
        builder = ConsensusBuilder(debate_id="debate-5", task="Markdown test")
        builder.record_vote("claude", VoteType.AGREE, 0.9, "Supporting reason")

        proof = builder.build("The answer is 42", 0.95, True, "Universal agreement")

        md = proof.to_markdown()

        assert "# Consensus Proof" in md
        assert "The answer is 42" in md
        assert "Reached" in md

    def test_consensus_blind_spots(self):
        """ConsensusProof identifies blind spots from dissent."""
        builder = ConsensusBuilder(debate_id="debate-6", task="Blind spot test")
        claim = builder.add_claim("Main claim", author="claude")
        builder.record_vote("claude", VoteType.AGREE, 0.8, "Agree")
        builder.record_vote("gpt4", VoteType.DISAGREE, 0.3, "Disagree")
        builder.record_dissent(
            "gpt4", claim.claim_id,
            reasons=["Overlooked edge case"],
            alternative="Consider fallback strategy",
            severity=0.8,
        )
        builder.record_tension(
            description="Speed vs safety tradeoff",
            agents=["claude", "gpt4"],
            options=["Fast approach", "Safe approach"],
            impact="Affects reliability",
        )

        proof = builder.build("Main claim", 0.6, False, "Disagreement")

        blind_spots = proof.get_blind_spots()
        assert len(blind_spots) >= 2  # dissent + tension

    def test_consensus_risk_correlation(self):
        """ConsensusProof groups risks by agreement level."""
        builder = ConsensusBuilder(debate_id="debate-7", task="Risk test")

        # Unanimous claim (only supporting evidence)
        claim1 = builder.add_claim("Safe choice", author="claude")
        builder.add_evidence(claim1.claim_id, "claude", "Strong evidence", supports=True, strength=0.9)

        # Contested claim (mixed evidence)
        claim2 = builder.add_claim("Risky choice", author="gpt4")
        builder.add_evidence(claim2.claim_id, "claude", "Concern", supports=False, strength=0.7)
        builder.add_evidence(claim2.claim_id, "gpt4", "Defense", supports=True, strength=0.5)

        proof = builder.build("Safe choice", 0.7, True, "Mostly agreed")

        correlation = proof.get_risk_correlation()
        assert "unanimous" in correlation
        assert "contested" in correlation

    def test_claim_net_evidence_strength(self):
        """Claim.net_evidence_strength calculates correctly."""
        claim = Claim(
            claim_id="c1", statement="Test", author="agent",
            confidence=0.5,
        )
        claim.supporting_evidence.append(
            Evidence("e1", "agent", "Support", "argument", True, 0.8)
        )
        claim.refuting_evidence.append(
            Evidence("e2", "critic", "Refute", "argument", False, 0.3)
        )

        # (0.8 - 0.3) / (0.8 + 0.3) = 0.5 / 1.1 ~ 0.4545
        strength = claim.net_evidence_strength
        assert 0.4 < strength < 0.5

    def test_claim_no_evidence_strength(self):
        """Claim with no evidence returns 0.0 strength."""
        claim = Claim(claim_id="c1", statement="Test", author="agent", confidence=0.5)

        assert claim.net_evidence_strength == 0.0

    def test_unresolved_tension(self):
        """ConsensusBuilder records unresolved tensions."""
        builder = ConsensusBuilder(debate_id="debate-8", task="Tension test")

        tension = builder.record_tension(
            description="Performance vs cost",
            agents=["claude", "gpt4"],
            options=["Optimize speed", "Minimize cost"],
            impact="Budget allocation",
            followup="Conduct A/B test",
        )

        assert tension.description == "Performance vs cost"
        assert tension.suggested_followup == "Conduct A/B test"

        proof = builder.build("Compromise", 0.5, False, "Unresolved")
        summary = proof.get_tension_summary()
        assert "Performance vs cost" in summary

    def test_conditional_vote(self):
        """ConsensusBuilder handles conditional votes."""
        builder = ConsensusBuilder(debate_id="debate-9", task="Conditional test")

        builder.record_vote(
            "claude", VoteType.CONDITIONAL, 0.7, "Agree if conditions met",
            conditions=["Must have monitoring", "Must have rollback"],
        )

        proof = builder.build("Plan A", 0.7, True, "Conditional agreement")

        # Conditional votes count as supporting
        assert "claude" in proof.supporting_agents


# =============================================================================
# Partial Consensus Tests
# =============================================================================


class TestPartialConsensus:
    """Tests for PartialConsensus tracking."""

    def test_partial_consensus_basic(self):
        """PartialConsensus tracks agreed and disagreed items."""
        partial = PartialConsensus(debate_id="debate-pc1")

        partial.add_item(PartialConsensusItem(
            item_id="item-1", topic="Architecture",
            statement="Use microservices",
            confidence=0.9, agreed=True,
            supporting_agents=["claude", "gpt4"],
        ))
        partial.add_item(PartialConsensusItem(
            item_id="item-2", topic="Database",
            statement="Use PostgreSQL",
            confidence=0.4, agreed=False,
            supporting_agents=["claude"],
            dissenting_agents=["gpt4"],
        ))

        assert len(partial.agreed_items) == 1
        assert len(partial.disagreed_items) == 1
        assert partial.consensus_ratio == 0.5
        assert 0.6 < partial.avg_confidence < 0.7  # (0.9 + 0.4) / 2

    def test_partial_consensus_actionable_items(self):
        """PartialConsensus filters actionable agreed items."""
        partial = PartialConsensus(debate_id="test")
        partial.add_item(PartialConsensusItem(
            item_id="a1", topic="Deploy", statement="Deploy to AWS",
            confidence=0.8, agreed=True, actionable=True,
        ))
        partial.add_item(PartialConsensusItem(
            item_id="a2", topic="Risk", statement="Theoretical risk",
            confidence=0.6, agreed=True, actionable=False,
        ))

        assert len(partial.actionable_items) == 1
        assert partial.actionable_items[0].item_id == "a1"

    def test_partial_consensus_summary(self):
        """PartialConsensus generates readable summary."""
        partial = PartialConsensus(debate_id="summary-test")
        partial.add_item(PartialConsensusItem(
            item_id="s1", topic="T1", statement="S1",
            confidence=0.8, agreed=True, actionable=True,
        ))

        summary = partial.summary()
        assert "Partial Consensus" in summary
        assert "1/1" in summary

    def test_partial_consensus_empty(self):
        """Empty PartialConsensus reports 0 ratio."""
        partial = PartialConsensus(debate_id="empty")

        assert partial.consensus_ratio == 0.0
        assert partial.avg_confidence == 0.0
        assert "No sub-questions" in partial.summary()

    def test_partial_consensus_serialization(self):
        """PartialConsensus serializes to dict."""
        partial = PartialConsensus(
            debate_id="ser-test",
            overall_consensus=True,
            overall_confidence=0.85,
        )
        partial.add_item(PartialConsensusItem(
            item_id="i1", topic="T", statement="S",
            confidence=0.9, agreed=True,
        ))

        d = partial.to_dict()
        assert d["debate_id"] == "ser-test"
        assert d["overall_consensus"] is True
        assert d["agreed_count"] == 1
        assert len(d["items"]) == 1

    def test_partial_consensus_item_agreement_ratio(self):
        """PartialConsensusItem calculates agreement ratio."""
        item = PartialConsensusItem(
            item_id="r1", topic="T", statement="S",
            confidence=0.7, agreed=True,
            supporting_agents=["claude", "gpt4", "gemini"],
            dissenting_agents=["grok"],
        )

        assert item.agreement_ratio == 0.75  # 3/4

    def test_build_partial_consensus_from_result(self):
        """build_partial_consensus extracts items from DebateResult."""
        result = DebateResult(
            task="Test task",
            final_answer="1. Implement caching to improve performance\n2. Add monitoring for reliability",
            confidence=0.8,
            consensus_reached=False,
            participants=["claude", "gpt4"],
            critiques=[],
            dissenting_views=["Alternative approach preferred by minority"],
        )

        partial = build_partial_consensus(result)

        assert isinstance(partial, PartialConsensus)
        assert partial.overall_consensus is False
        # Should have items from final_answer bullet points + dissenting views
        assert len(partial.items) >= 1


# =============================================================================
# Agent Team Selection Tests
# =============================================================================


class TestAgentTeamSelection:
    """Tests for agent team selection and weights."""

    def test_require_agents_returns_agents(self):
        """_require_agents returns the agent list."""
        from aragora.debate.orchestrator import Arena

        env = _make_env()
        agents = _make_agents(3)
        protocol = _make_protocol()

        arena = Arena(environment=env, agents=agents, protocol=protocol)

        result = arena._require_agents()
        assert len(result) >= 3

    def test_agent_weights_stored(self):
        """Agent weights are stored on the arena."""
        from aragora.debate.orchestrator import Arena

        env = _make_env()
        agents = _make_agents(2)
        protocol = _make_protocol()
        weights = {"claude": 1.5, "gpt4": 0.8}

        arena = Arena(
            environment=env, agents=agents, protocol=protocol,
            agent_weights=weights,
        )

        assert arena.agent_weights == weights


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in debate execution."""

    def test_empty_task_raises(self):
        """Environment with empty task raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            Environment(task="")

    def test_null_byte_task_raises(self):
        """Environment with null byte raises ValueError."""
        with pytest.raises(ValueError, match="null"):
            Environment(task="test\x00task")

    def test_very_long_task_raises(self):
        """Environment with extremely long task raises ValueError."""
        with pytest.raises(ValueError, match="maximum length"):
            Environment(task="x" * 200_000)

    @pytest.mark.asyncio
    async def test_timeout_returns_partial_result(self):
        """Timeout during debate returns partial DebateResult."""
        from aragora.debate.orchestrator import Arena

        env = _make_env()
        agents = _make_agents(2)
        protocol = _make_protocol(timeout_seconds=1)

        arena = Arena(environment=env, agents=agents, protocol=protocol)

        # Simulate slow execution
        async def slow_run(correlation_id=""):
            await asyncio.sleep(10)
            return DebateResult(task=env.task)

        arena._run_inner = slow_run

        result = await arena.run()

        assert isinstance(result, DebateResult)
        assert result.task == env.task

    @pytest.mark.asyncio
    async def test_agent_generate_failure_handled(self):
        """MockAgent can be configured to fail."""

        class FailingAgent(MockAgent):
            async def generate(self, prompt, context=None):
                raise RuntimeError("API connection failed")

        agent = FailingAgent(name="failing-agent")

        with pytest.raises(RuntimeError, match="API connection failed"):
            await agent.generate("test prompt")

    @pytest.mark.asyncio
    async def test_agent_critique_failure_handled(self):
        """Agent critique failures are catchable."""

        class FailingCritic(MockAgent):
            async def critique(self, proposal, task, context=None, target_agent=None):
                raise TimeoutError("Critique timed out")

        agent = FailingCritic(name="failing-critic")

        with pytest.raises(TimeoutError, match="timed out"):
            await agent.critique("proposal", "task")


# =============================================================================
# Core Types Tests
# =============================================================================


class TestCoreTypes:
    """Tests for core debate types: Message, Critique, Vote, DebateResult."""

    def test_message_creation(self):
        """Message stores role, agent, content, and round."""
        msg = Message(role="proposer", agent="claude", content="My proposal", round=1)

        assert msg.role == "proposer"
        assert msg.agent == "claude"
        assert msg.content == "My proposal"
        assert msg.round == 1

    def test_message_str(self):
        """Message has readable string representation."""
        msg = Message(role="critic", agent="gpt4", content="I think there is an issue with approach")

        s = str(msg)
        assert "critic" in s
        assert "gpt4" in s

    def test_critique_creation(self):
        """Critique captures structured feedback."""
        critique = Critique(
            agent="gpt4",
            target_agent="claude",
            target_content="Rate limiter proposal",
            issues=["Missing retry logic", "No monitoring"],
            suggestions=["Add exponential backoff", "Use Prometheus"],
            severity=5.0,
            reasoning="Production readiness concerns",
        )

        assert critique.agent == "gpt4"
        assert critique.target == "claude"  # backward compat property
        assert critique.severity == 5.0
        assert len(critique.issues) == 2

    def test_critique_to_prompt(self):
        """Critique formats as prompt text."""
        critique = Critique(
            agent="gpt4", target_agent="claude",
            target_content="Proposal",
            issues=["Issue 1"], suggestions=["Fix 1"],
            severity=4.0, reasoning="Because",
        )

        prompt = critique.to_prompt()
        assert "gpt4" in prompt
        assert "Issue 1" in prompt
        assert "severity: 4.0" in prompt

    def test_vote_creation(self):
        """Vote stores agent choice and confidence."""
        vote = Vote(
            agent="gemini",
            choice="claude",
            reasoning="Best proposal",
            confidence=0.9,
            continue_debate=False,
        )

        assert vote.agent == "gemini"
        assert vote.choice == "claude"
        assert vote.confidence == 0.9
        assert vote.continue_debate is False

    def test_debate_result_defaults(self):
        """DebateResult has sensible defaults."""
        result = DebateResult(task="Test")

        assert result.task == "Test"
        assert result.confidence == 0.0
        assert result.consensus_reached is False
        assert result.status == "completed"
        assert result.messages == []
        assert result.debate_id != ""  # auto-generated

    def test_debate_result_consensus_status(self):
        """DebateResult auto-sets status based on consensus."""
        result = DebateResult(task="Test", consensus_reached=True)
        assert result.status == "consensus_reached"

        result2 = DebateResult(task="Test2", consensus_reached=False)
        assert result2.status == "completed"

    def test_debate_result_round_sync(self):
        """DebateResult syncs rounds_used and rounds_completed."""
        result = DebateResult(task="Test", rounds_completed=5)
        assert result.rounds_used == 5

        result2 = DebateResult(task="Test", rounds_used=3)
        assert result2.rounds_completed == 3

    def test_debate_result_history_alias(self):
        """DebateResult.history is alias for messages."""
        msgs = [Message(role="proposer", agent="claude", content="Test")]
        result = DebateResult(task="Test", messages=msgs)

        assert result.history is result.messages
        assert len(result.history) == 1


# =============================================================================
# ArenaConfig Tests
# =============================================================================


class TestArenaConfig:
    """Tests for ArenaConfig construction and builder."""

    def test_basic_config(self):
        """ArenaConfig with defaults."""
        config = ArenaConfig()

        assert config.loop_id == ""
        assert config.strict_loop_scoping is False
        assert config.org_id == ""

    def test_config_with_kwargs(self):
        """ArenaConfig accepts keyword arguments."""
        config = ArenaConfig(
            loop_id="test-loop",
            org_id="test-org",
            use_airlock=True,
        )

        assert config.loop_id == "test-loop"
        assert config.org_id == "test-org"
        assert config.use_airlock is True

    def test_config_builder(self):
        """ArenaConfig.builder() creates config fluently."""
        config = (
            ArenaConfig.builder()
            .with_identity(loop_id="builder-loop")
            .with_core(use_airlock=True)
            .build()
        )

        assert config.loop_id == "builder-loop"
        assert config.use_airlock is True

    def test_config_equality(self):
        """ArenaConfig supports equality comparison."""
        config1 = ArenaConfig(loop_id="test")
        config2 = ArenaConfig(loop_id="test")

        assert config1 == config2

    def test_config_repr(self):
        """ArenaConfig has string representation."""
        config = ArenaConfig(loop_id="repr-test")

        r = repr(config)
        assert "ArenaConfig" in r
        assert "repr-test" in r

    def test_config_unknown_kwarg_raises(self):
        """ArenaConfig raises TypeError for unknown kwargs."""
        with pytest.raises(TypeError, match="unknown"):
            ArenaConfig(nonexistent_field=True)

    def test_config_to_arena_kwargs(self):
        """ArenaConfig.to_arena_kwargs returns dict for Arena init."""
        config = ArenaConfig(
            loop_id="kwargs-test",
            org_id="org-1",
            use_airlock=True,
        )

        kwargs = config.to_arena_kwargs()

        assert kwargs["loop_id"] == "kwargs-test"
        assert kwargs["org_id"] == "org-1"
        assert kwargs["use_airlock"] is True


# =============================================================================
# MockAgent Tests (verify test helper works)
# =============================================================================


class TestMockAgent:
    """Tests for our MockAgent helper to ensure test infrastructure works."""

    @pytest.mark.asyncio
    async def test_mock_agent_generate(self):
        """MockAgent returns a predictable response."""
        agent = MockAgent(name="test-agent")

        response = await agent.generate("What is 2+2?")

        assert "test-agent" in response

    @pytest.mark.asyncio
    async def test_mock_agent_critique(self):
        """MockAgent returns a valid Critique."""
        agent = MockAgent(name="critic-1")

        critique = await agent.critique(
            "My proposal", "Design something",
            target_agent="proposer-1",
        )

        assert isinstance(critique, Critique)
        assert critique.agent == "critic-1"
        assert critique.target_agent == "proposer-1"
        assert critique.severity == 3.0

    @pytest.mark.asyncio
    async def test_mock_agent_inherits_from_agent(self):
        """MockAgent is a proper Agent subclass."""
        agent = MockAgent()

        assert isinstance(agent, Agent)
        assert agent.role == "proposer"
        assert agent.stance == "neutral"


# =============================================================================
# VoteType Enum Tests
# =============================================================================


class TestVoteType:
    """Tests for VoteType enum values."""

    def test_vote_types_exist(self):
        """All expected vote types are defined."""
        assert VoteType.AGREE.value == "agree"
        assert VoteType.DISAGREE.value == "disagree"
        assert VoteType.ABSTAIN.value == "abstain"
        assert VoteType.CONDITIONAL.value == "conditional"


# =============================================================================
# Environment Validation Tests
# =============================================================================


class TestEnvironment:
    """Tests for Environment dataclass validation."""

    def test_valid_environment(self):
        """Valid Environment is created successfully."""
        env = Environment(task="Design a system", context="For web scale")

        assert env.task == "Design a system"
        assert env.context == "For web scale"
        assert env.max_rounds == 3

    def test_environment_with_roles(self):
        """Environment accepts custom roles."""
        env = Environment(
            task="Review code",
            roles=["proposer", "critic", "analyst"],
        )

        assert "analyst" in env.roles

    def test_environment_with_documents(self):
        """Environment stores document IDs for evidence grounding."""
        env = Environment(
            task="Analyze contract",
            documents=["doc-123", "doc-456"],
        )

        assert len(env.documents) == 2

    def test_whitespace_only_task_raises(self):
        """Environment rejects whitespace-only task."""
        with pytest.raises(ValueError, match="empty"):
            Environment(task="   ")
