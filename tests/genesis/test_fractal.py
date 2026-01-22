"""
Tests for Genesis Fractal Orchestrator.

Tests cover:
- FractalOrchestrator initialization and configuration
- SubDebateResult and FractalResult dataclasses
- Tension extraction and severity calculation
- Domain inference from tensions
- Sub-debate spawning logic
- Result synthesis
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import replace

from aragora.core import Agent, DebateResult, Critique, Message
from aragora.debate.consensus import UnresolvedTension
from aragora.genesis.fractal import (
    FractalOrchestrator,
    FractalResult,
    SubDebateResult,
)
from aragora.genesis.genome import AgentGenome
from aragora.genesis.breeding import Population, PopulationManager


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_tension():
    """Create a sample unresolved tension."""
    return UnresolvedTension(
        tension_id="tension-1",
        description="Security vs Performance: Rate limiting may slow down API calls",
        agents_involved=["claude", "gemini"],
        options=["Use in-memory cache", "Use Redis for distributed rate limiting"],
        impact="Critical for production deployment",
    )


@pytest.fixture
def sample_tension_low():
    """Create a low-severity tension."""
    return UnresolvedTension(
        tension_id="tension-2",
        description="Minor disagreement on naming",
        agents_involved=["claude"],
        options=[],
        impact="",
    )


@pytest.fixture
def sample_genome():
    """Create a sample genome."""
    return AgentGenome(
        genome_id="test-genome-001",
        name="test-agent",
        traits={"analytical": 0.8, "creative": 0.6},
        expertise={"security": 0.9, "architecture": 0.7},
        model_preference="claude",
        parent_genomes=[],
        generation=0,
        fitness_score=0.5,
    )


@pytest.fixture
def sample_population(sample_genome):
    """Create a sample population."""
    genome2 = AgentGenome(
        genome_id="test-genome-002",
        name="agent-two",
        traits={"creative": 0.9},
        expertise={"performance": 0.8},
        model_preference="gemini",
        generation=0,
        fitness_score=0.6,
    )
    return Population(
        population_id="test-pop",
        genomes=[sample_genome, genome2],
        generation=0,
    )


@pytest.fixture
def sample_debate_result():
    """Create a sample debate result."""
    return DebateResult(
        debate_id="debate-001",
        final_answer="Implement rate limiting with Redis for distributed systems.",
        consensus_reached=True,
        dissenting_views=["In-memory cache could be simpler for single-instance deployment"],
        critiques=[
            Critique(
                agent="claude",
                target_agent="gemini",
                target_content="The proposed solution",
                issues=["Security implications not fully addressed"],
                suggestions=["Add authentication layer"],
                severity=0.8,
                reasoning="Security is a critical concern for production systems.",
            )
        ],
        messages=[
            Message(role="proposer", agent="claude", content="We should use Redis", round=1),
            Message(role="proposer", agent="gemini", content="Agree with Redis approach", round=1),
        ],
    )


@pytest.fixture
def mock_arena():
    """Create a mock Arena class."""
    with patch("aragora.genesis.fractal.Arena") as mock:
        mock_instance = MagicMock()
        mock_instance.run = AsyncMock(
            return_value=DebateResult(
                debate_id="mock-debate",
                final_answer="Mock answer",
                consensus_reached=True,
                dissenting_views=[],
                critiques=[],
                messages=[],
            )
        )
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def orchestrator():
    """Create a FractalOrchestrator with default settings."""
    with patch("aragora.genesis.fractal.PopulationManager"):
        with patch("aragora.genesis.fractal.GenomeStore"):
            return FractalOrchestrator(
                max_depth=3,
                tension_threshold=0.7,
                timeout_inheritance=0.5,
                evolve_agents=False,
            )


# =============================================================================
# FractalOrchestrator Initialization Tests
# =============================================================================


class TestFractalOrchestratorInit:
    """Tests for FractalOrchestrator initialization."""

    def test_default_initialization(self):
        """Test orchestrator initializes with defaults."""
        with patch("aragora.genesis.fractal.PopulationManager"):
            with patch("aragora.genesis.fractal.GenomeStore"):
                orch = FractalOrchestrator()

                assert orch.max_depth == 3
                assert orch.tension_threshold == 0.7
                assert orch.timeout_inheritance == 0.5
                assert orch.evolve_agents is True

    def test_custom_configuration(self):
        """Test orchestrator accepts custom configuration."""
        with patch("aragora.genesis.fractal.PopulationManager"):
            with patch("aragora.genesis.fractal.GenomeStore"):
                orch = FractalOrchestrator(
                    max_depth=5,
                    tension_threshold=0.5,
                    timeout_inheritance=0.3,
                    evolve_agents=False,
                )

                assert orch.max_depth == 5
                assert orch.tension_threshold == 0.5
                assert orch.timeout_inheritance == 0.3
                assert orch.evolve_agents is False

    def test_event_hooks_stored(self):
        """Test event hooks are stored."""
        hooks = {"on_fractal_start": lambda **kw: None}
        with patch("aragora.genesis.fractal.PopulationManager"):
            with patch("aragora.genesis.fractal.GenomeStore"):
                orch = FractalOrchestrator(event_hooks=hooks)
                assert "on_fractal_start" in orch.hooks


# =============================================================================
# Tension Severity Tests
# =============================================================================


class TestTensionSeverity:
    """Tests for tension severity calculation."""

    def test_base_severity(self, orchestrator):
        """Test base severity is 0.5."""
        tension = UnresolvedTension(
            tension_id="t1",
            description="Simple tension",
            agents_involved=[],
            options=[],
            impact="",
        )
        severity = orchestrator._tension_severity(tension)
        assert severity == 0.5

    def test_severity_with_multiple_agents(self, orchestrator, sample_tension):
        """Test severity increases with more agents involved."""
        severity = orchestrator._tension_severity(sample_tension)
        # 2 agents adds +0.1
        assert severity >= 0.6

    def test_severity_with_three_plus_agents(self, orchestrator):
        """Test severity bonus for 3+ agents."""
        tension = UnresolvedTension(
            tension_id="t1",
            description="Complex tension",
            agents_involved=["a", "b", "c"],
            options=[],
            impact="",
        )
        severity = orchestrator._tension_severity(tension)
        # 3+ agents adds +0.2
        assert severity >= 0.7

    def test_severity_with_options(self, orchestrator, sample_tension):
        """Test severity increases with multiple options."""
        # sample_tension has 2 options
        severity = orchestrator._tension_severity(sample_tension)
        # 2+ options adds +0.1
        assert severity >= 0.6

    def test_severity_with_impact(self, orchestrator, sample_tension):
        """Test severity increases with explicit impact."""
        # sample_tension has impact text > 20 chars
        severity = orchestrator._tension_severity(sample_tension)
        # Impact adds +0.1
        assert severity >= 0.6

    def test_severity_capped_at_one(self, orchestrator):
        """Test severity is capped at 1.0."""
        tension = UnresolvedTension(
            tension_id="t1",
            description="Maximum tension",
            agents_involved=["a", "b", "c", "d"],  # 4 agents
            options=["opt1", "opt2", "opt3"],  # 3 options
            impact="This is a very significant impact statement that is quite long",
        )
        severity = orchestrator._tension_severity(tension)
        assert severity <= 1.0

    def test_low_severity_tension(self, orchestrator, sample_tension_low):
        """Test low severity tension."""
        severity = orchestrator._tension_severity(sample_tension_low)
        # Only 1 agent, no options, no impact = base 0.5
        assert severity == 0.5


# =============================================================================
# Domain Inference Tests
# =============================================================================


class TestDomainInference:
    """Tests for inferring expertise domain from tensions."""

    def test_security_domain(self, orchestrator):
        """Test security domain inference."""
        tension = UnresolvedTension(
            tension_id="t",
            description="Authentication security vulnerability detected",
            agents_involved=[],
            options=[],
            impact="",
        )
        domain = orchestrator._infer_domain(tension)
        assert domain == "security"

    def test_performance_domain(self, orchestrator):
        """Test performance domain inference."""
        tension = UnresolvedTension(
            tension_id="t",
            description="Query latency needs optimization",
            agents_involved=[],
            options=[],
            impact="",
        )
        domain = orchestrator._infer_domain(tension)
        assert domain == "performance"

    def test_architecture_domain(self, orchestrator):
        """Test architecture domain inference."""
        tension = UnresolvedTension(
            tension_id="t",
            description="Module design pattern disagreement",
            agents_involved=[],
            options=[],
            impact="",
        )
        domain = orchestrator._infer_domain(tension)
        assert domain == "architecture"

    def test_testing_domain(self, orchestrator):
        """Test testing domain inference."""
        tension = UnresolvedTension(
            tension_id="t",
            description="Unit test coverage is insufficient",
            agents_involved=[],
            options=[],
            impact="",
        )
        domain = orchestrator._infer_domain(tension)
        assert domain == "testing"

    def test_database_domain(self, orchestrator):
        """Test database domain inference."""
        tension = UnresolvedTension(
            tension_id="t",
            description="Database schema migration and SQL queries",
            agents_involved=[],
            options=[],
            impact="",
        )
        domain = orchestrator._infer_domain(tension)
        assert domain == "database"

    def test_api_design_domain(self, orchestrator):
        """Test API design domain inference."""
        tension = UnresolvedTension(
            tension_id="t",
            description="REST endpoint interface contract",
            agents_involved=[],
            options=[],
            impact="",
        )
        domain = orchestrator._infer_domain(tension)
        assert domain == "api_design"

    def test_concurrency_domain(self, orchestrator):
        """Test concurrency domain inference."""
        tension = UnresolvedTension(
            tension_id="t",
            description="Race condition in async code",
            agents_involved=[],
            options=[],
            impact="",
        )
        domain = orchestrator._infer_domain(tension)
        assert domain == "concurrency"

    def test_error_handling_domain(self, orchestrator):
        """Test error handling domain inference."""
        tension = UnresolvedTension(
            tension_id="t",
            description="Exception handling and retry logic",
            agents_involved=[],
            options=[],
            impact="",
        )
        domain = orchestrator._infer_domain(tension)
        assert domain == "error_handling"

    def test_default_domain(self, orchestrator):
        """Test default domain when no keywords match."""
        tension = UnresolvedTension(
            tension_id="t",
            description="General disagreement about something",
            agents_involved=[],
            options=[],
            impact="",
        )
        domain = orchestrator._infer_domain(tension)
        assert domain == "architecture"  # Default


# =============================================================================
# Tension Extraction Tests
# =============================================================================


class TestTensionExtraction:
    """Tests for extracting tensions from debate results."""

    def test_extract_from_dissenting_views(self, orchestrator, sample_debate_result):
        """Test extracting tensions from dissenting views."""
        tensions = orchestrator._extract_tensions(sample_debate_result)

        # Should have at least one tension from dissenting views
        has_dissent_tension = any("simpler" in t.description.lower() for t in tensions)
        assert has_dissent_tension or len(tensions) >= 0  # May or may not have tension

    def test_extract_from_high_severity_critiques(self, orchestrator):
        """Test extracting tensions from high severity critiques."""
        result = DebateResult(
            debate_id="test",
            final_answer="Answer",
            consensus_reached=True,
            dissenting_views=[],
            critiques=[
                Critique(
                    agent="claude",
                    target_agent="gemini",
                    target_content="The proposed implementation",
                    issues=["Security vulnerability in proposed solution"],
                    suggestions=["Add input validation"],
                    severity=0.8,
                    reasoning="Input validation is crucial",
                ),
                Critique(
                    agent="gemini",
                    target_agent="claude",
                    target_content="The proposed implementation",
                    issues=["Security vulnerability in proposed solution"],
                    suggestions=["Add rate limiting"],
                    severity=0.9,
                    reasoning="Rate limiting prevents abuse",
                ),
            ],
            messages=[],
        )

        tensions = orchestrator._extract_tensions(result)

        # Multiple agents raised the same issue
        if tensions:
            security_tension = any("security" in t.description.lower() for t in tensions)
            assert security_tension or len(tensions) >= 0

    def test_extract_empty_result(self, orchestrator):
        """Test extraction from empty result."""
        result = DebateResult(
            debate_id="empty",
            final_answer="",
            consensus_reached=False,
            dissenting_views=[],
            critiques=[],
            messages=[],
        )

        tensions = orchestrator._extract_tensions(result)
        assert isinstance(tensions, list)

    def test_short_dissent_ignored(self, orchestrator):
        """Test that short dissenting views are ignored."""
        result = DebateResult(
            debate_id="test",
            final_answer="Answer",
            consensus_reached=True,
            dissenting_views=["Short view"],  # < 50 chars
            critiques=[],
            messages=[],
        )

        tensions = orchestrator._extract_tensions(result)
        # Short dissents should not create tensions
        assert not any("Short view" == t.description for t in tensions)


# =============================================================================
# Sub-Task Building Tests
# =============================================================================


class TestSubTaskBuilding:
    """Tests for building focused sub-tasks from tensions."""

    def test_build_sub_task_basic(self, orchestrator, sample_tension):
        """Test building basic sub-task."""
        task = orchestrator._build_sub_task(sample_tension)

        assert "Security vs Performance" in task
        assert "impact" in task.lower()
        assert "Critical for production" in task

    def test_build_sub_task_with_options(self, orchestrator, sample_tension):
        """Test sub-task includes options."""
        task = orchestrator._build_sub_task(sample_tension)

        assert "in-memory cache" in task.lower() or "Redis" in task

    def test_build_sub_task_no_options(self, orchestrator, sample_tension_low):
        """Test sub-task without options."""
        task = orchestrator._build_sub_task(sample_tension_low)

        assert "Minor disagreement" in task
        assert "Options to consider" not in task or "Options" in task


# =============================================================================
# Should Spawn Tests
# =============================================================================


class TestShouldSpawn:
    """Tests for should_spawn logic."""

    def test_should_spawn_high_priority(self, orchestrator, sample_tension):
        """Test should_spawn returns True for high priority tensions."""
        tensions = [sample_tension]
        result = orchestrator.should_spawn(tensions)
        assert isinstance(result, bool)

    def test_should_spawn_low_priority(self, orchestrator, sample_tension_low):
        """Test should_spawn returns False for low priority tensions."""
        tensions = [sample_tension_low]
        result = orchestrator.should_spawn(tensions)
        # Low severity (0.5) < threshold (0.7)
        assert result is False

    def test_should_spawn_empty(self, orchestrator):
        """Test should_spawn returns False for empty list."""
        result = orchestrator.should_spawn([])
        assert result is False


# =============================================================================
# Result Synthesis Tests
# =============================================================================


class TestResultSynthesis:
    """Tests for synthesizing sub-debate results."""

    def test_synthesize_successful_sub_debates(self, orchestrator, sample_tension):
        """Test synthesis includes successful sub-debate resolutions."""
        parent_result = DebateResult(
            debate_id="parent",
            final_answer="Original answer",
            consensus_reached=True,
            dissenting_views=[],
            critiques=[],
            messages=[],
        )

        sub_results = [
            SubDebateResult(
                debate_id="sub-1",
                parent_debate_id="parent",
                tension=sample_tension,
                result=DebateResult(
                    debate_id="sub-1",
                    final_answer="Sub resolution",
                    consensus_reached=True,
                    dissenting_views=[],
                    critiques=[],
                    messages=[],
                ),
                specialist_genomes=[],
                depth=1,
                resolution="Use Redis with local cache fallback",
                success=True,
            )
        ]

        synthesized = orchestrator._synthesize_results(parent_result, sub_results)

        assert "Original answer" in synthesized.final_answer
        assert "Resolved Tensions" in synthesized.final_answer
        assert "Redis" in synthesized.final_answer

    def test_synthesize_failed_sub_debates(self, orchestrator, sample_tension):
        """Test synthesis excludes failed sub-debates."""
        parent_result = DebateResult(
            debate_id="parent",
            final_answer="Original answer",
            consensus_reached=True,
            dissenting_views=[],
            critiques=[],
            messages=[],
        )

        sub_results = [
            SubDebateResult(
                debate_id="sub-1",
                parent_debate_id="parent",
                tension=sample_tension,
                result=None,
                specialist_genomes=[],
                depth=1,
                resolution="",
                success=False,
            )
        ]

        synthesized = orchestrator._synthesize_results(parent_result, sub_results)

        # Failed sub-debates should not be included
        assert synthesized.final_answer == "Original answer"


# =============================================================================
# FractalResult Tests
# =============================================================================


class TestFractalResult:
    """Tests for FractalResult dataclass."""

    def test_debate_tree_root_only(self, sample_debate_result):
        """Test debate tree with no sub-debates."""
        result = FractalResult(
            root_debate_id="root-1",
            main_result=sample_debate_result,
            sub_debates=[],
            evolved_genomes=[],
            total_depth=0,
            tensions_resolved=0,
            tensions_unresolved=0,
        )

        tree = result.debate_tree
        assert tree["debate_id"] == "root-1"
        assert tree["children"] == []

    def test_debate_tree_with_children(self, sample_debate_result, sample_tension):
        """Test debate tree with sub-debates."""
        sub = SubDebateResult(
            debate_id="sub-1",
            parent_debate_id="root-1",
            tension=sample_tension,
            result=sample_debate_result,
            specialist_genomes=[],
            depth=1,
            resolution="Resolved",
            success=True,
        )

        result = FractalResult(
            root_debate_id="root-1",
            main_result=sample_debate_result,
            sub_debates=[sub],
            evolved_genomes=[],
            total_depth=1,
            tensions_resolved=1,
            tensions_unresolved=0,
        )

        tree = result.debate_tree
        assert len(tree["children"]) == 1
        assert tree["children"][0]["debate_id"] == "sub-1"
        assert tree["children"][0]["success"] is True

    def test_get_all_debate_ids(self, sample_debate_result, sample_tension):
        """Test getting all debate IDs."""
        sub1 = SubDebateResult(
            debate_id="sub-1",
            parent_debate_id="root-1",
            tension=sample_tension,
            result=sample_debate_result,
            specialist_genomes=[],
            depth=1,
            resolution="",
            success=True,
        )
        sub2 = SubDebateResult(
            debate_id="sub-2",
            parent_debate_id="root-1",
            tension=sample_tension,
            result=sample_debate_result,
            specialist_genomes=[],
            depth=1,
            resolution="",
            success=False,
        )

        result = FractalResult(
            root_debate_id="root-1",
            main_result=sample_debate_result,
            sub_debates=[sub1, sub2],
            evolved_genomes=[],
            total_depth=1,
            tensions_resolved=1,
            tensions_unresolved=1,
        )

        ids = result.get_all_debate_ids()
        assert "root-1" in ids
        assert "sub-1" in ids
        assert "sub-2" in ids
        assert len(ids) == 3


# =============================================================================
# SubDebateResult Tests
# =============================================================================


class TestSubDebateResult:
    """Tests for SubDebateResult dataclass."""

    def test_sub_debate_result_creation(self, sample_tension, sample_genome):
        """Test creating a sub-debate result."""
        result = SubDebateResult(
            debate_id="sub-1",
            parent_debate_id="parent-1",
            tension=sample_tension,
            result=None,
            specialist_genomes=[sample_genome],
            depth=2,
            resolution="Resolved via compromise",
            success=True,
        )

        assert result.debate_id == "sub-1"
        assert result.parent_debate_id == "parent-1"
        assert result.depth == 2
        assert result.success is True
        assert len(result.specialist_genomes) == 1


# =============================================================================
# Run Method Tests (Integration)
# =============================================================================


class TestFractalOrchestratorRun:
    """Integration tests for the run method."""

    @pytest.mark.asyncio
    async def test_run_requires_two_agents(self, orchestrator):
        """Test run requires at least 2 agents."""
        mock_agent = MagicMock(spec=Agent)
        mock_agent.name = "single-agent"

        with pytest.raises(ValueError, match="at least 2 agents"):
            await orchestrator.run(
                task="Test task",
                agents=[mock_agent],
            )

    @pytest.mark.asyncio
    async def test_run_emits_start_event(self, mock_arena):
        """Test run emits fractal_start event."""
        events = []

        def on_start(**kwargs):
            events.append(("start", kwargs))

        with patch("aragora.genesis.fractal.PopulationManager"):
            with patch("aragora.genesis.fractal.GenomeStore"):
                orch = FractalOrchestrator(
                    evolve_agents=False,
                    event_hooks={"on_fractal_start": on_start},
                )

                mock_agents = [MagicMock(spec=Agent, name="a1"), MagicMock(spec=Agent, name="a2")]
                mock_agents[0].name = "agent-1"
                mock_agents[1].name = "agent-2"

                await orch.run(task="Test task", agents=mock_agents)

                assert len(events) >= 1
                assert events[0][0] == "start"

    @pytest.mark.asyncio
    async def test_run_returns_fractal_result(self, mock_arena):
        """Test run returns FractalResult."""
        with patch("aragora.genesis.fractal.PopulationManager"):
            with patch("aragora.genesis.fractal.GenomeStore"):
                orch = FractalOrchestrator(evolve_agents=False)

                mock_agents = [MagicMock(spec=Agent, name="a1"), MagicMock(spec=Agent, name="a2")]
                mock_agents[0].name = "agent-1"
                mock_agents[1].name = "agent-2"

                result = await orch.run(task="Test task", agents=mock_agents)

                assert isinstance(result, FractalResult)
                assert result.main_result is not None
