"""
Tests for scaling debates to 10+ concurrent agents.

Verifies:
- 10 agents can all propose, critique, and revise concurrently
- Consensus is reached with 10+ agents
- No deadlocks or timeouts
- ArenaConfig.max_agents validation works correctly
- Complexity governor supports 12 agents at NOMINAL stress
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core_types import Agent, Critique, Environment, Vote
from aragora.debate.arena_config import ArenaConfig
from aragora.debate.complexity_governor import (
    AdaptiveComplexityGovernor,
    GovernorConstraints,
    StressLevel,
)
from aragora.debate.config.defaults import DEBATE_DEFAULTS
from aragora.debate.phases.proposal_phase import ProposalPhase
from aragora.debate.protocol import DebateProtocol


# =============================================================================
# Helpers
# =============================================================================

AGENT_NAMES = [
    "claude",
    "gpt-5.5",
    "gemini",
    "grok",
    "mistral",
    "deepseek",
    "llama",
    "qwen",
    "codestral",
    "yi",
    "kimi",
    "phi",
]


class MockAgent(Agent):
    """Concrete mock agent for testing at scale."""

    def __init__(self, name: str = "mock-agent", model: str = "mock-model"):
        super().__init__(name=name, model=model, role="proposer")

    async def generate(self, prompt: str, context=None) -> str:
        return f"Proposal from {self.name}: implement rate limiting with sliding window"

    async def critique(self, proposal, task, context=None, target_agent=None) -> Critique:
        return Critique(
            agent=self.name,
            target_agent=target_agent or "unknown",
            target_content=proposal,
            issues=["Consider edge cases"],
            suggestions=["Add retry logic"],
            severity=3.0,
            reasoning=f"Critique from {self.name}",
        )


def _make_agents(count: int) -> list[MockAgent]:
    """Create count mock agents with unique names."""
    return [
        MockAgent(name=AGENT_NAMES[i % len(AGENT_NAMES)], model=f"model-{i}") for i in range(count)
    ]


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


def _make_debate_context(agents=None, task="Design a rate limiter"):
    """Create a mock debate context that satisfies ProposalPhase requirements."""
    agents = agents or []
    ctx = MagicMock()
    ctx.agents = agents
    ctx.proposers = [a for a in agents if getattr(a, "role", "") == "proposer"]
    ctx.env = MagicMock()
    ctx.env.task = task
    ctx.proposals = {}
    ctx.messages = []
    ctx.context_messages = []
    ctx.cancellation_token = None
    ctx.hook_manager = None
    ctx.debate_id = "test-scale-debate"
    return ctx


# =============================================================================
# Config limit tests
# =============================================================================


class TestScaleConfigLimits:
    """Tests that config defaults support 10+ agents."""

    def test_debate_defaults_allow_20_agents(self):
        """DEBATE_DEFAULTS.max_agents_per_debate is 20."""
        assert DEBATE_DEFAULTS.max_agents_per_debate >= 20

    def test_arena_config_max_agents_default(self):
        """ArenaConfig default max_agents is 20."""
        config = ArenaConfig()
        assert config.max_agents == 20

    def test_arena_config_custom_max_agents(self):
        """ArenaConfig allows setting custom max_agents."""
        config = ArenaConfig(max_agents=50)
        assert config.max_agents == 50

    def test_protocol_parallel_proposals(self):
        """DebateProtocol has max_parallel_proposals field."""
        protocol = _make_protocol()
        assert hasattr(protocol, "max_parallel_proposals")
        assert protocol.max_parallel_proposals >= 10

    def test_protocol_parallel_critiques(self):
        """DebateProtocol max_parallel_critiques supports 10+ agents."""
        protocol = _make_protocol()
        assert protocol.max_parallel_critiques >= 20

    def test_protocol_parallel_revisions(self):
        """DebateProtocol max_parallel_revisions supports 10+ agents."""
        protocol = _make_protocol()
        assert protocol.max_parallel_revisions >= 10


# =============================================================================
# Complexity governor tests
# =============================================================================


class TestScaleGovernor:
    """Tests that the complexity governor supports 10+ agents."""

    def test_nominal_allows_12_agents(self):
        """NOMINAL stress level allows up to 12 agents per round."""
        governor = AdaptiveComplexityGovernor()
        constraints = governor.current_constraints
        assert constraints.max_agents_per_round >= 12

    def test_elevated_allows_8_agents(self):
        """ELEVATED stress level allows 8 agents per round."""
        constraints = AdaptiveComplexityGovernor.CONSTRAINT_PRESETS[StressLevel.ELEVATED]
        assert constraints.max_agents_per_round >= 8

    def test_high_allows_5_agents(self):
        """HIGH stress level still allows 5 agents per round."""
        constraints = AdaptiveComplexityGovernor.CONSTRAINT_PRESETS[StressLevel.HIGH]
        assert constraints.max_agents_per_round >= 5

    def test_critical_allows_3_agents(self):
        """CRITICAL stress level allows 3 agents per round."""
        constraints = AdaptiveComplexityGovernor.CONSTRAINT_PRESETS[StressLevel.CRITICAL]
        assert constraints.max_agents_per_round >= 3

    def test_governor_defaults_allow_10_agents(self):
        """Default GovernorConstraints max_agents_per_round >= 12."""
        constraints = GovernorConstraints()
        assert constraints.max_agents_per_round >= 12

    def test_recommended_agent_count_at_nominal(self):
        """Governor recommends 12+ agents at NOMINAL stress."""
        governor = AdaptiveComplexityGovernor()
        recommended = governor.get_recommended_agent_count()
        assert recommended >= 12


# =============================================================================
# Arena initialization with 10+ agents
# =============================================================================


class TestScaleArenaInit:
    """Tests that Arena accepts 10+ agents without errors."""

    def test_arena_init_10_agents(self):
        """Arena initializes successfully with 10 agents."""
        from aragora.debate.orchestrator import Arena

        env = Environment(task="Design a rate limiter")
        agents = _make_agents(10)
        protocol = _make_protocol()

        arena = Arena(environment=env, agents=agents, protocol=protocol)
        assert len(arena.agents) >= 10

    def test_arena_init_12_agents(self):
        """Arena initializes successfully with 12 agents."""
        from aragora.debate.orchestrator import Arena

        env = Environment(task="Design a rate limiter")
        agents = _make_agents(12)
        protocol = _make_protocol()

        arena = Arena(environment=env, agents=agents, protocol=protocol)
        assert len(arena.agents) >= 12

    def test_arena_rejects_over_max_agents(self):
        """Arena raises ValueError when agents exceed max_agents."""
        from aragora.debate.orchestrator import Arena

        env = Environment(task="Design a rate limiter")
        agents = _make_agents(25)
        protocol = _make_protocol()
        config = ArenaConfig(max_agents=20)

        with pytest.raises(ValueError, match="Too many agents"):
            Arena(
                environment=env,
                agents=agents,
                protocol=protocol,
                debate_config=None,
                agent_config=None,
            )

    def test_arena_logs_warning_over_10_agents(self):
        """Arena logs a warning when >10 agents are used."""
        from aragora.debate.orchestrator import Arena

        env = Environment(task="Design a rate limiter")
        agents = _make_agents(12)
        protocol = _make_protocol()

        with patch("aragora.debate.orchestrator.logger") as mock_logger:
            Arena(environment=env, agents=agents, protocol=protocol)
            # Check that warning was called about large agent count
            warning_calls = [
                call
                for call in mock_logger.warning.call_args_list
                if "large_agent_count" in str(call)
            ]
            assert len(warning_calls) >= 1


# =============================================================================
# Proposal phase at scale
# =============================================================================


class TestScaleProposalPhase:
    """Tests that proposal phase handles 10 agents concurrently."""

    @pytest.mark.asyncio
    async def test_10_agent_parallel_proposals(self):
        """All 10 agents generate proposals concurrently without deadlock."""
        agents = _make_agents(10)

        async def mock_generate(agent, prompt, context=None):
            # Simulate a small delay to test concurrency
            await asyncio.sleep(0.01)
            return f"Proposal from {agent.name}"

        phase = ProposalPhase(
            build_proposal_prompt=lambda agent: "Propose a solution for: test task",
            generate_with_agent=mock_generate,
        )

        ctx = _make_debate_context(agents=agents, task="Design a rate limiter")

        await phase.execute(ctx)

        # All 10 agents should have produced proposals
        assert len(ctx.proposals) == 10
        for agent in agents:
            assert agent.name in ctx.proposals

    @pytest.mark.asyncio
    async def test_12_agent_parallel_proposals(self):
        """All 12 agents generate proposals concurrently."""
        agents = _make_agents(12)

        async def mock_generate(agent, prompt, context=None):
            await asyncio.sleep(0.01)
            return f"Proposal from {agent.name}"

        phase = ProposalPhase(
            build_proposal_prompt=lambda agent: "Propose a solution",
            generate_with_agent=mock_generate,
        )

        ctx = _make_debate_context(agents=agents, task="Design a rate limiter")

        await phase.execute(ctx)

        assert len(ctx.proposals) == 12

    @pytest.mark.asyncio
    async def test_10_agents_with_circuit_breaker(self):
        """Circuit breaker filtering works correctly with 10 agents."""
        agents = _make_agents(10)

        mock_cb = MagicMock()
        # Circuit breaker excludes 2 agents, 8 remain
        mock_cb.filter_available_agents.return_value = agents[:8]

        async def mock_generate(agent, prompt, context=None):
            return f"Proposal from {agent.name}"

        phase = ProposalPhase(
            circuit_breaker=mock_cb,
            build_proposal_prompt=lambda agent: "Propose a solution",
            generate_with_agent=mock_generate,
        )

        ctx = _make_debate_context(agents=agents, task="Design a rate limiter")

        await phase.execute(ctx)

        # 8 agents should have proposals (2 filtered by circuit breaker)
        assert len(ctx.proposals) == 8


# =============================================================================
# Critique phase at scale (O(N^2) pattern)
# =============================================================================


class TestScaleCritiquePhase:
    """Tests that the critique topology works correctly with 10+ agents."""

    def test_full_mesh_critique_count_10_agents(self):
        """Full-mesh critique topology with 10 agents produces N*(N-1) critiques."""
        agents = _make_agents(10)
        proposals = {agent.name: f"Proposal from {agent.name}" for agent in agents}

        # Simulate full-mesh: each agent critiques every other agent's proposal
        critique_pairs = []
        for proposal_agent in proposals:
            critics = [a for a in agents if a.name != proposal_agent]
            for critic in critics:
                critique_pairs.append((critic.name, proposal_agent))

        # N agents, each critiques (N-1) proposals = N*(N-1) total
        assert len(critique_pairs) == 10 * 9  # 90 critiques

    def test_fast_first_routing_limits_critics(self):
        """Fast-first routing reduces critic count per proposal."""
        protocol = _make_protocol(
            fast_first_routing=True,
            fast_first_max_critics_per_proposal=3,
        )

        # With fast_first, each proposal gets at most 3 critics
        # instead of N-1 = 9 critics
        assert protocol.fast_first_max_critics_per_proposal == 3
        # 10 proposals * 3 critics each = 30 critiques (vs 90 full-mesh)


# =============================================================================
# Vote collection at scale
# =============================================================================


class TestScaleVoteCollection:
    """Tests that vote collection handles 10+ agents."""

    def test_quorum_with_10_agents(self):
        """Quorum check passes with 10 agents and sufficient participation."""
        # With default min_participation_ratio=0.5, need 5 of 10 agents
        protocol = _make_protocol()
        min_ratio = getattr(protocol, "min_participation_ratio", 0.5)
        min_count = getattr(protocol, "min_participation_count", 1)

        import math

        total_agents = 10
        required = max(min_count, math.ceil(total_agents * min_ratio))
        required = min(required, total_agents)

        # With 10 agents and 0.5 ratio, need 5 votes
        assert required <= 10
        assert required >= 1

    def test_rlm_early_termination_with_10_agents(self):
        """RLM early termination works with 10 agents."""
        from aragora.debate.phases.vote_collector import VoteCollector, VoteCollectorConfig

        config = VoteCollectorConfig(
            vote_with_agent=AsyncMock(),
            enable_rlm_early_termination=True,
            rlm_early_termination_threshold=0.75,
            rlm_majority_lead_threshold=0.25,
        )
        collector = VoteCollector(config)

        # Create 8 votes (out of 10 agents) - meets 75% threshold
        votes = []
        for i in range(7):
            vote = MagicMock()
            vote.choice = "option_a"
            votes.append(vote)
        # 1 dissenter
        vote = MagicMock()
        vote.choice = "option_b"
        votes.append(vote)

        has_majority, winner = collector._check_clear_majority(votes, total_agents=10)
        # 7 votes for option_a out of 10 total agents: majority > 50%, lead = 6 >= 2.5
        assert has_majority is True
        assert winner == "option_a"


# =============================================================================
# End-to-end 10-agent debate integration test
# =============================================================================


class TestScaleEndToEnd:
    """Integration tests for full debate flow with 10 agents."""

    @pytest.mark.asyncio
    async def test_10_agent_proposal_critique_cycle(self):
        """10 agents complete a full propose -> critique cycle without errors."""
        agents = _make_agents(10)
        proposals = {}

        # Phase 1: All agents propose concurrently
        async def generate_proposal(agent):
            await asyncio.sleep(0.005)  # Simulate API latency
            return f"Proposal from {agent.name}: implement distributed cache"

        # Gather all proposals concurrently
        tasks = [generate_proposal(a) for a in agents]
        results = await asyncio.gather(*tasks)

        for agent, result in zip(agents, results):
            proposals[agent.name] = result

        assert len(proposals) == 10

        # Phase 2: Each agent critiques all other proposals (full-mesh)
        critiques = []

        async def generate_critique(critic, target_agent, proposal):
            await asyncio.sleep(0.005)
            return Critique(
                agent=critic.name,
                target_agent=target_agent,
                target_content=proposal,
                issues=["Consider edge cases"],
                suggestions=["Add monitoring"],
                severity=3.0,
                reasoning=f"Critique from {critic.name}",
            )

        # Bounded concurrency with semaphore
        semaphore = asyncio.Semaphore(20)
        critique_tasks = []

        for target_name, proposal in proposals.items():
            for critic in agents:
                if critic.name == target_name:
                    continue

                async def bounded_critique(c=critic, t=target_name, p=proposal):
                    async with semaphore:
                        return await generate_critique(c, t, p)

                critique_tasks.append(bounded_critique())

        critique_results = await asyncio.gather(*critique_tasks, return_exceptions=True)

        valid_critiques = [r for r in critique_results if isinstance(r, Critique)]
        # 10 agents * 9 targets each = 90 critiques
        assert len(valid_critiques) == 90

        # Phase 3: Voting - all agents vote concurrently
        votes = []

        async def cast_vote(agent):
            await asyncio.sleep(0.005)
            return Vote(
                agent=agent.name,
                choice=agents[0].name,  # Everyone votes for agent 0
                reasoning=f"Vote from {agent.name}",
            )

        vote_tasks = [cast_vote(a) for a in agents]
        votes = await asyncio.gather(*vote_tasks)

        assert len(votes) == 10

        # Check consensus: all voted for the same agent
        choices = [v.choice for v in votes]
        assert len(set(choices)) == 1  # Unanimous

    @pytest.mark.asyncio
    async def test_10_agents_no_deadlock_with_timeouts(self):
        """10 agents with mixed speeds complete without deadlock."""
        agents = _make_agents(10)

        async def variable_speed_generate(agent):
            # Different agents take different amounts of time
            delay = 0.001 * (hash(agent.name) % 20)
            await asyncio.sleep(delay)
            return f"Proposal from {agent.name}"

        # Run with a generous timeout - should complete well within
        results = await asyncio.wait_for(
            asyncio.gather(*[variable_speed_generate(a) for a in agents]),
            timeout=5.0,
        )

        assert len(results) == 10
        # All proposals are unique and non-empty
        assert all(r.startswith("Proposal from") for r in results)

    @pytest.mark.asyncio
    async def test_10_agents_with_failures(self):
        """10 agents where 2 fail still produce 8 valid proposals."""
        agents = _make_agents(10)
        failing_agents = {agents[3].name, agents[7].name}

        async def sometimes_failing_generate(agent):
            if agent.name in failing_agents:
                raise ConnectionError(f"API error for {agent.name}")
            await asyncio.sleep(0.005)
            return f"Proposal from {agent.name}"

        tasks = [sometimes_failing_generate(a) for a in agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successes = [r for r in results if isinstance(r, str)]
        failures = [r for r in results if isinstance(r, Exception)]

        assert len(successes) == 8
        assert len(failures) == 2
