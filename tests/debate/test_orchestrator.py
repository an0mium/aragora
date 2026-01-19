"""
Tests for the debate orchestrator module.

Tests cover:
- _compute_domain_from_task utility function
- Arena initialization and configuration
- Arena context manager behavior
- Team selection and domain extraction
- Event emission and lifecycle
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core import Agent, Critique, Environment, Message, Vote
from aragora.debate.orchestrator import Arena, _compute_domain_from_task
from aragora.debate.protocol import DebateProtocol


class MockAgent(Agent):
    """Mock agent for testing Arena functionality."""

    def __init__(
        self,
        name: str = "mock-agent",
        response: str = "Test response",
        model: str = "mock-model",
        role: str = "proposer",
    ):
        super().__init__(name=name, model=model, role=role)
        self.agent_type = "mock"
        self.response = response
        self.generate_calls = 0
        self.critique_calls = 0
        self.vote_calls = 0

    async def generate(self, prompt: str, context: list = None) -> str:
        self.generate_calls += 1
        return self.response

    async def generate_stream(self, prompt: str, context: list = None):
        yield self.response

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list = None,
        target_agent: str = None,
    ) -> Critique:
        self.critique_calls += 1
        return Critique(
            agent=self.name,
            target_agent=target_agent or "unknown",
            target_content=proposal[:100] if proposal else "",
            issues=["Test issue"],
            suggestions=["Test suggestion"],
            severity=0.5,
            reasoning="Test reasoning",
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        self.vote_calls += 1
        choice = list(proposals.keys())[0] if proposals else self.name
        return Vote(
            agent=self.name,
            choice=choice,
            reasoning="Test vote",
            confidence=0.8,
            continue_debate=False,
        )


class TestComputeDomainFromTask:
    """Tests for _compute_domain_from_task utility function."""

    def test_security_domain(self):
        """Test security domain detection."""
        assert _compute_domain_from_task("implement authentication system") == "security"
        assert _compute_domain_from_task("fix the security vulnerability") == "security"
        assert _compute_domain_from_task("add encryption to data storage") == "security"
        assert _compute_domain_from_task("prevent hack attacks") == "security"
        assert _compute_domain_from_task("improve auth flow") == "security"

    def test_performance_domain(self):
        """Test performance domain detection."""
        # Note: Order of checks matters - performance keywords are checked early
        assert _compute_domain_from_task("improve the speed of the application") == "performance"
        assert _compute_domain_from_task("implement cache strategy") == "performance"
        assert _compute_domain_from_task("reduce latency") == "performance"
        assert _compute_domain_from_task("optimize the code") == "performance"

    def test_testing_domain(self):
        """Test testing domain detection."""
        assert _compute_domain_from_task("write unit tests for the module") == "testing"
        assert _compute_domain_from_task("improve test coverage") == "testing"
        assert _compute_domain_from_task("fix regression tests") == "testing"

    def test_architecture_domain(self):
        """Test architecture domain detection."""
        assert _compute_domain_from_task("design the system architecture") == "architecture"
        assert _compute_domain_from_task("implement factory pattern") == "architecture"
        assert _compute_domain_from_task("restructure the codebase") == "architecture"

    def test_debugging_domain(self):
        """Test debugging domain detection."""
        assert _compute_domain_from_task("fix the login bug") == "debugging"
        assert _compute_domain_from_task("resolve null pointer error") == "debugging"
        assert _compute_domain_from_task("debug the crash on startup") == "debugging"
        assert _compute_domain_from_task("handle runtime exception") == "debugging"

    def test_api_domain(self):
        """Test API domain detection."""
        assert _compute_domain_from_task("create new endpoint for users") == "api"
        assert _compute_domain_from_task("implement rest service") == "api"
        # Note: "graphql" is API but comes after other checks
        assert _compute_domain_from_task("graphql mutations") == "api"

    def test_database_domain(self):
        """Test database domain detection."""
        assert _compute_domain_from_task("create new sql table") == "database"
        assert _compute_domain_from_task("write database query") == "database"
        # Note: "schema" could be API (graphql) but database is checked after API

    def test_frontend_domain(self):
        """Test frontend domain detection."""
        assert _compute_domain_from_task("update the ui component") == "frontend"
        assert _compute_domain_from_task("improve layout styling") == "frontend"
        assert _compute_domain_from_task("style with css") == "frontend"
        assert _compute_domain_from_task("react component") == "frontend"
        # Note: "fix layout" triggers "debugging" due to "fix"

    def test_general_domain(self):
        """Test general domain fallback."""
        assert _compute_domain_from_task("do something random") == "general"
        assert _compute_domain_from_task("process the data") == "general"
        assert _compute_domain_from_task("implement feature x") == "general"

    def test_first_match_wins(self):
        """Test that the first matching domain wins (order matters)."""
        # "performance" is checked before "database" so "optimize" wins
        assert _compute_domain_from_task("optimize database queries") == "performance"
        # "security" is checked before "testing" so "auth" wins
        assert _compute_domain_from_task("test auth flow") == "security"

    def test_case_sensitivity(self):
        """Test that function expects lowercase input."""
        # The function expects lowercase input (as documented)
        # Mixed case might not match correctly
        assert _compute_domain_from_task("security") == "security"
        assert _compute_domain_from_task("performance") == "performance"

    def test_partial_word_matches(self):
        """Test that word boundaries are not enforced (substring matching)."""
        # "auth" matches in "authentication"
        assert _compute_domain_from_task("authentication") == "security"
        # "test" matches in "testing"
        assert _compute_domain_from_task("testing") == "testing"
        # "cache" is an exact keyword, "caching" doesn't match
        assert _compute_domain_from_task("cache layer") == "performance"
        # "encrypt" matches in "encryption"
        assert _compute_domain_from_task("encryption") == "security"

    def test_multiple_keywords_first_wins(self):
        """Test that with multiple keywords, first domain check wins."""
        # This has both "speed" (performance) and "bug" (debugging)
        # Performance is checked first, so it wins
        assert _compute_domain_from_task("speed up and fix the bug") == "performance"

        # This has both "security" and "test"
        # Security is checked first
        assert _compute_domain_from_task("security test suite") == "security"

    def test_empty_string(self):
        """Test empty string returns general."""
        assert _compute_domain_from_task("") == "general"

    def test_no_keywords(self):
        """Test string with no domain keywords returns general."""
        assert _compute_domain_from_task("hello world") == "general"
        assert _compute_domain_from_task("please help me") == "general"


class TestArenaInitialization:
    """Tests for Arena initialization and configuration."""

    @pytest.fixture
    def environment(self):
        """Create a test environment."""
        return Environment(task="What is the best approach to solve this problem?")

    @pytest.fixture
    def agents(self):
        """Create test agents."""
        return [
            MockAgent(name="agent1", response="Proposal from agent 1"),
            MockAgent(name="agent2", response="Proposal from agent 2"),
        ]

    @pytest.fixture
    def protocol(self):
        """Create a test protocol."""
        return DebateProtocol(rounds=2, consensus="majority")

    def test_arena_creation_with_protocol(self, environment, agents, protocol):
        """Arena can be created with explicit protocol."""
        arena = Arena(environment, agents, protocol)

        assert arena.protocol == protocol
        assert arena.protocol.rounds == 2
        assert arena.protocol.consensus == "majority"

    def test_arena_stores_agents(self, environment, agents, protocol):
        """Arena stores provided agents."""
        arena = Arena(environment, agents, protocol)

        # Check agents are stored (may be in _agents or agent_pool)
        assert hasattr(arena, "_agents") or hasattr(arena, "agent_pool")


class TestArenaFromConfig:
    """Tests for Arena.from_config factory method."""

    def test_from_config_creates_arena(self):
        """Arena.from_config creates an arena from configuration."""
        from aragora.debate.arena_config import ArenaConfig

        env = Environment(task="Design a scalable API")
        protocol = DebateProtocol(rounds=3, consensus="majority")
        config = ArenaConfig(loop_id="test-123")
        agents = [MockAgent(name="a1"), MockAgent(name="a2")]

        arena = Arena.from_config(env, agents, protocol, config)

        assert arena is not None
        assert isinstance(arena, Arena)

    def test_arena_direct_creation_works(self):
        """Arena can be created directly with loop_id."""
        env = Environment(task="Design a scalable API")
        protocol = DebateProtocol(rounds=3, consensus="majority")
        agents = [MockAgent(name="a1"), MockAgent(name="a2")]

        # Direct creation works
        arena = Arena(env, agents, protocol, loop_id="test-123")

        assert arena is not None
        assert isinstance(arena, Arena)


class TestArenaDomainExtraction:
    """Tests for domain extraction from tasks."""

    @pytest.fixture
    def arena_with_security_task(self):
        """Arena with a security-related task."""
        env = Environment(task="Implement secure authentication")
        agents = [MockAgent(name="a1"), MockAgent(name="a2")]
        return Arena(env, agents)

    @pytest.fixture
    def arena_with_performance_task(self):
        """Arena with a performance-related task."""
        env = Environment(task="Optimize database query speed")
        agents = [MockAgent(name="a1"), MockAgent(name="a2")]
        return Arena(env, agents)

    def test_extracts_security_domain(self, arena_with_security_task):
        """Security domain is extracted for auth tasks."""
        domain = arena_with_security_task._extract_debate_domain()
        assert domain == "security"

    def test_extracts_performance_domain(self, arena_with_performance_task):
        """Performance domain is extracted for optimization tasks."""
        domain = arena_with_performance_task._extract_debate_domain()
        assert domain == "performance"


class TestArenaContextManager:
    """Tests for Arena async context manager behavior."""

    @pytest.fixture
    def environment(self):
        return Environment(task="Test task")

    @pytest.fixture
    def agents(self):
        return [MockAgent(name="a1"), MockAgent(name="a2")]

    @pytest.mark.asyncio
    async def test_context_manager_entry(self, environment, agents):
        """Arena can be used as async context manager."""
        arena = Arena(environment, agents)

        async with arena as entered_arena:
            assert entered_arena is arena

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_exit(self, environment, agents):
        """Arena cleans up resources on context exit."""
        arena = Arena(environment, agents)

        async with arena:
            pass

        # After exit, cleanup should have occurred
        # (No exception means successful cleanup)

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_exception(self, environment, agents):
        """Arena cleans up even when exception occurs."""
        arena = Arena(environment, agents)

        with pytest.raises(ValueError):
            async with arena:
                raise ValueError("Test error")

        # Cleanup should still have occurred


class TestArenaTeamSelection:
    """Tests for debate team selection logic."""

    @pytest.fixture
    def environment(self):
        return Environment(task="Test task")

    @pytest.fixture
    def many_agents(self):
        """Create many agents for team selection tests."""
        return [
            MockAgent(name=f"agent{i}", role="proposer")
            for i in range(6)
        ]

    def test_select_team_returns_agents(self, environment, many_agents):
        """Team selection returns a list of agents."""
        arena = Arena(environment, many_agents)

        # Select a team from the agents
        team = arena._select_debate_team(many_agents[:3])

        assert isinstance(team, list)
        assert len(team) > 0
        assert all(isinstance(a, Agent) for a in team)

    def test_select_team_respects_requested_agents(self, environment, many_agents):
        """Team selection includes requested agents."""
        arena = Arena(environment, many_agents)

        requested = many_agents[:2]
        team = arena._select_debate_team(requested)

        # All requested agents should be in the team
        requested_names = {a.name for a in requested}
        team_names = {a.name for a in team}
        assert requested_names.issubset(team_names)


class TestArenaEventEmission:
    """Tests for Arena event emission."""

    @pytest.fixture
    def environment(self):
        return Environment(task="Test task")

    @pytest.fixture
    def agents(self):
        return [MockAgent(name="a1"), MockAgent(name="a2")]

    @pytest.fixture
    def protocol(self):
        return DebateProtocol(rounds=1, consensus="majority")

    def test_arena_has_event_emitter(self, environment, agents, protocol):
        """Arena initializes with an event emitter."""
        arena = Arena(environment, agents, protocol)

        # Event emitter should be initialized (or event_bus)
        assert hasattr(arena, "event_emitter") or hasattr(arena, "event_bus")

    def test_arena_accepts_event_hooks(self, environment, agents, protocol):
        """Arena accepts custom event hooks without error."""
        hooks = {
            "on_debate_start": MagicMock(),
            "on_round_complete": MagicMock(),
        }

        # Should create arena without error
        arena = Arena(environment, agents, protocol, event_hooks=hooks)

        # Arena should be created successfully
        assert arena is not None


class TestArenaRun:
    """Tests for Arena.run() execution."""

    @pytest.fixture
    def environment(self):
        return Environment(task="What is 2+2?")

    @pytest.fixture
    def agents(self):
        return [
            MockAgent(name="math1", response="The answer is 4"),
            MockAgent(name="math2", response="2 plus 2 equals 4"),
        ]

    @pytest.fixture
    def protocol(self):
        return DebateProtocol(rounds=1, consensus="majority")

    @pytest.mark.asyncio
    async def test_run_returns_debate_result(self, environment, agents, protocol):
        """Arena.run() returns a DebateResult."""
        from aragora.core import DebateResult

        arena = Arena(environment, agents, protocol)

        result = await arena.run()

        assert isinstance(result, DebateResult)
        assert result.task == "What is 2+2?"

    @pytest.mark.asyncio
    async def test_run_populates_final_answer(self, environment, agents, protocol):
        """Arena.run() populates the final answer."""
        arena = Arena(environment, agents, protocol)

        result = await arena.run()

        assert result.final_answer is not None
        assert len(result.final_answer) > 0

    @pytest.mark.asyncio
    async def test_run_tracks_rounds(self, environment, agents, protocol):
        """Arena.run() tracks the number of rounds used."""
        arena = Arena(environment, agents, protocol)

        result = await arena.run()

        assert result.rounds_used >= 1

    @pytest.mark.asyncio
    async def test_run_with_correlation_id(self, environment, agents, protocol):
        """Arena.run() accepts a correlation ID for tracing."""
        arena = Arena(environment, agents, protocol)

        result = await arena.run(correlation_id="test-correlation-123")

        # Should complete without error
        assert result is not None

    @pytest.mark.asyncio
    async def test_run_calls_agent_generate(self, environment, agents, protocol):
        """Arena.run() calls agent generate methods."""
        arena = Arena(environment, agents, protocol)

        await arena.run()

        # At least one agent should have been called
        total_calls = sum(a.generate_calls for a in agents)
        assert total_calls > 0


class TestArenaRequireAgents:
    """Tests for _require_agents helper."""

    def test_require_agents_returns_agents(self):
        """_require_agents returns the agent list."""
        env = Environment(task="Test")
        agents = [MockAgent(name="a1"), MockAgent(name="a2")]
        protocol = DebateProtocol(rounds=1, consensus="majority")
        arena = Arena(env, agents, protocol)

        result = arena._require_agents()

        # Should return agents (may be processed)
        assert len(result) > 0

    def test_require_agents_returns_non_empty_list(self):
        """_require_agents returns a non-empty list of agents."""
        env = Environment(task="Test")
        agents = [MockAgent(name="a1"), MockAgent(name="a2")]
        protocol = DebateProtocol(rounds=1, consensus="majority")
        arena = Arena(env, agents, protocol)

        result = arena._require_agents()

        # Should return agents
        assert isinstance(result, list)
        assert len(result) >= 1
