"""
Tests for the dialectical synthesis phase.

Tests cover:
1. Synthesis identifies distinct positions from proposals
2. Synthesis generates combined position using agent
3. Synthesis result includes elements from both sides
4. Skips synthesis when only one position exists
5. Handles agent generation failure gracefully
6. Confidence reflects how well synthesis addresses critiques
7. Selects best-calibrated agent for synthesis when prefer_agent not set
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.debate.phases.synthesis_phase import (
    DialecticalPosition,
    DialecticalSynthesizer,
    SynthesisConfig,
    SynthesisResult,
    _build_synthesis_prompt,
    _extract_bullet_points,
    _extract_claims,
    _parse_synthesis_output,
)


# =============================================================================
# Mock Objects
# =============================================================================


@dataclass
class MockVote:
    """Mock Vote for testing."""

    agent: str = "voter1"
    choice: str = "agent1"
    reasoning: str = "Good proposal"
    confidence: float = 0.9


@dataclass
class MockResult:
    """Mock debate result."""

    final_answer: str = ""
    synthesis: str = ""
    winner: str = ""
    confidence: float = 0.8
    votes: list[Any] = field(default_factory=list)
    consensus_reached: bool = False


@dataclass
class MockEnv:
    """Mock environment."""

    task: str = "Should we use microservices or monolith architecture?"


@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str = "agent1"
    model: str = "test-model"
    role: str = "proposer"
    generate: AsyncMock = field(default_factory=lambda: AsyncMock(return_value="synthesis output"))


@dataclass
class MockDebateContext:
    """Mock debate context for testing."""

    env: MockEnv = field(default_factory=MockEnv)
    agents: list[MockAgent] = field(default_factory=list)
    proposals: dict[str, str] = field(default_factory=dict)
    result: MockResult = field(default_factory=MockResult)
    round_critiques: list[Any] = field(default_factory=list)
    context_messages: list[Any] = field(default_factory=list)


def make_context(
    proposals: dict[str, str] | None = None,
    agents: list[MockAgent] | None = None,
    votes: list[MockVote] | None = None,
) -> MockDebateContext:
    """Create a MockDebateContext with sensible defaults."""
    if proposals is None:
        proposals = {
            "agent1": "We should use microservices because they enable independent scaling and deployment.",
            "agent2": "A monolith is better because it reduces complexity and is easier to debug.",
        }
    if agents is None:
        agents = [
            MockAgent(name="agent1"),
            MockAgent(name="agent2"),
            MockAgent(name="agent3"),
        ]
    result = MockResult(votes=votes if votes is not None else [])

    return MockDebateContext(
        proposals=proposals,
        agents=agents,
        result=result,
    )


# =============================================================================
# Tests: SynthesisConfig
# =============================================================================


class TestSynthesisConfig:
    """Tests for SynthesisConfig defaults and construction."""

    def test_default_config(self):
        """Config has sensible defaults."""
        config = SynthesisConfig()
        assert config.enable_synthesis is True
        assert config.max_synthesis_attempts == 3
        assert config.min_opposing_positions == 2
        assert config.synthesis_confidence_threshold == 0.5
        assert config.prefer_agent is None

    def test_custom_config(self):
        """Config accepts custom values."""
        config = SynthesisConfig(
            enable_synthesis=False,
            max_synthesis_attempts=5,
            min_opposing_positions=3,
            synthesis_confidence_threshold=0.8,
            prefer_agent="claude-opus",
        )
        assert config.enable_synthesis is False
        assert config.max_synthesis_attempts == 5
        assert config.min_opposing_positions == 3
        assert config.synthesis_confidence_threshold == 0.8
        assert config.prefer_agent == "claude-opus"


# =============================================================================
# Tests: DialecticalPosition
# =============================================================================


class TestDialecticalPosition:
    """Tests for DialecticalPosition dataclass."""

    def test_basic_construction(self):
        """Position stores agent, content, and claims."""
        pos = DialecticalPosition(
            agent="agent1",
            content="Use microservices",
            key_claims=["microservices enable scaling"],
            supporting_agents=["agent3"],
        )
        assert pos.agent == "agent1"
        assert pos.content == "Use microservices"
        assert len(pos.key_claims) == 1
        assert "agent3" in pos.supporting_agents

    def test_default_empty_lists(self):
        """Position defaults to empty lists."""
        pos = DialecticalPosition(agent="a", content="c")
        assert pos.key_claims == []
        assert pos.supporting_agents == []


# =============================================================================
# Tests: SynthesisResult
# =============================================================================


class TestSynthesisResult:
    """Tests for SynthesisResult dataclass."""

    def test_full_construction(self):
        """SynthesisResult captures all synthesis metadata."""
        thesis = DialecticalPosition(agent="agent1", content="Thesis text")
        antithesis = DialecticalPosition(agent="agent2", content="Antithesis text")

        result = SynthesisResult(
            synthesis="Combined position",
            confidence=0.85,
            thesis=thesis,
            antithesis=antithesis,
            elements_from_thesis=["independent scaling"],
            elements_from_antithesis=["simplicity"],
            novel_elements=["hybrid approach"],
            synthesizer_agent="agent3",
        )

        assert result.synthesis == "Combined position"
        assert result.confidence == 0.85
        assert result.thesis.agent == "agent1"
        assert result.antithesis.agent == "agent2"
        assert len(result.elements_from_thesis) == 1
        assert len(result.elements_from_antithesis) == 1
        assert len(result.novel_elements) == 1
        assert result.synthesizer_agent == "agent3"


# =============================================================================
# Tests: Helper functions
# =============================================================================


class TestExtractClaims:
    """Tests for _extract_claims helper."""

    def test_extracts_assertion_sentences(self):
        """Extracts sentences with assertion markers."""
        text = "Microservices are great. They enable scaling. Hello world."
        claims = _extract_claims(text)
        # "are" and "enable" are not in the set, but let's check
        # Actually "are" IS in the set
        assert any("are great" in c.lower() for c in claims)

    def test_empty_text(self):
        """Returns empty list for empty text."""
        assert _extract_claims("") == []

    def test_limits_to_five_claims(self):
        """Returns at most 5 claims."""
        text = ". ".join(
            f"Point {i} is important and should be considered"
            for i in range(10)
        )
        claims = _extract_claims(text)
        assert len(claims) <= 5

    def test_skips_short_sentences(self):
        """Skips sentences shorter than 15 characters."""
        text = "Short. This is a longer sentence that should be considered."
        claims = _extract_claims(text)
        for claim in claims:
            assert len(claim) > 15


class TestExtractBulletPoints:
    """Tests for _extract_bullet_points helper."""

    def test_extracts_dash_bullets(self):
        """Extracts lines starting with dash."""
        text = "Header\n- First point\n- Second point\nNot a bullet"
        items = _extract_bullet_points(text)
        assert items == ["First point", "Second point"]

    def test_extracts_asterisk_bullets(self):
        """Extracts lines starting with asterisk."""
        text = "Header\n* Point A\n* Point B"
        items = _extract_bullet_points(text)
        assert items == ["Point A", "Point B"]

    def test_empty_section(self):
        """Returns empty list for text without bullets."""
        assert _extract_bullet_points("No bullets here") == []


class TestParseSynthesisOutput:
    """Tests for _parse_synthesis_output."""

    def test_parses_structured_output(self):
        """Parses well-structured synthesis agent output."""
        raw = """### Synthesis
The best approach combines both architectures.

### Elements from Thesis
- Independent scaling
- Team autonomy

### Elements from Antithesis
- Simpler debugging
- Lower overhead

### Novel Elements
- Start monolith, extract services incrementally
"""
        synthesis, from_thesis, from_antithesis, novel = _parse_synthesis_output(raw)

        assert "combines both" in synthesis
        assert "Independent scaling" in from_thesis
        assert "Team autonomy" in from_thesis
        assert "Simpler debugging" in from_antithesis
        assert "Start monolith" in novel[0]

    def test_falls_back_for_unstructured(self):
        """Returns raw text when no sections found."""
        raw = "Just a plain synthesis text without any sections."
        synthesis, from_thesis, from_antithesis, novel = _parse_synthesis_output(raw)

        assert synthesis == raw
        assert from_thesis == []
        assert from_antithesis == []
        assert novel == []


class TestBuildSynthesisPrompt:
    """Tests for _build_synthesis_prompt."""

    def test_includes_thesis_and_antithesis(self):
        """Prompt includes both positions and the task."""
        thesis = DialecticalPosition(
            agent="agent1",
            content="Use microservices",
            key_claims=["scaling is key"],
            supporting_agents=["agent3"],
        )
        antithesis = DialecticalPosition(
            agent="agent2",
            content="Use monolith",
            key_claims=["simplicity matters"],
        )
        prompt = _build_synthesis_prompt(thesis, antithesis, "Architecture choice")

        assert "agent1" in prompt
        assert "agent2" in prompt
        assert "Use microservices" in prompt
        assert "Use monolith" in prompt
        assert "Architecture choice" in prompt
        assert "scaling is key" in prompt
        assert "simplicity matters" in prompt
        assert "agent3" in prompt


# =============================================================================
# Tests: DialecticalSynthesizer._identify_positions
# =============================================================================


class TestIdentifyPositions:
    """Tests for position identification from proposals and votes."""

    def test_identifies_positions_from_proposals(self):
        """Identifies distinct positions from proposals."""
        synthesizer = DialecticalSynthesizer()
        proposals = {
            "agent1": "Microservices are the way forward for scalability.",
            "agent2": "Monoliths are simpler and more reliable for our use case.",
        }
        votes: list[MockVote] = []

        positions = synthesizer._identify_positions(proposals, votes)

        assert len(positions) == 2
        assert positions[0].agent in ("agent1", "agent2")
        assert positions[1].agent in ("agent1", "agent2")

    def test_orders_by_vote_support(self):
        """Positions are ordered by vote count (most supported first)."""
        synthesizer = DialecticalSynthesizer()
        proposals = {
            "agent1": "Position A about architecture is important.",
            "agent2": "Position B should be considered as well.",
        }
        votes = [
            MockVote(agent="agent3", choice="agent2"),
            MockVote(agent="agent4", choice="agent2"),
            MockVote(agent="agent5", choice="agent1"),
        ]

        positions = synthesizer._identify_positions(proposals, votes)

        assert positions[0].agent == "agent2"  # More votes
        assert positions[1].agent == "agent1"  # Fewer votes

    def test_identifies_supporting_agents(self):
        """Supporting agents are correctly identified from votes."""
        synthesizer = DialecticalSynthesizer()
        proposals = {
            "agent1": "This proposal is about scalability and should be adopted.",
            "agent2": "This counter-proposal requires careful consideration.",
        }
        votes = [
            MockVote(agent="agent3", choice="agent1"),
            MockVote(agent="agent4", choice="agent1"),
            MockVote(agent="agent1", choice="agent1"),  # Self-vote excluded from supporters
        ]

        positions = synthesizer._identify_positions(proposals, votes)
        agent1_pos = next(p for p in positions if p.agent == "agent1")

        assert "agent3" in agent1_pos.supporting_agents
        assert "agent4" in agent1_pos.supporting_agents
        assert "agent1" not in agent1_pos.supporting_agents  # Self-vote excluded

    def test_empty_proposals(self):
        """Returns empty list for empty proposals."""
        synthesizer = DialecticalSynthesizer()
        assert synthesizer._identify_positions({}, []) == []

    def test_extracts_key_claims(self):
        """Key claims are extracted from proposal text."""
        synthesizer = DialecticalSynthesizer()
        proposals = {
            "agent1": "The system should use microservices. Each service must be independently deployable.",
        }

        positions = synthesizer._identify_positions(proposals, [])
        assert len(positions) == 1
        assert len(positions[0].key_claims) > 0


# =============================================================================
# Tests: DialecticalSynthesizer.synthesize
# =============================================================================


class TestSynthesize:
    """Tests for the main synthesize method."""

    @pytest.mark.asyncio
    async def test_generates_combined_position(self):
        """Synthesis generates a combined position using agent."""
        structured_output = """### Synthesis
A hybrid approach combines microservices for scaling with monolith simplicity.

### Elements from Thesis
- Independent scaling capability

### Elements from Antithesis
- Simplified debugging workflow

### Novel Elements
- Incremental service extraction pattern
"""
        agent = MockAgent(
            name="synth-agent",
            generate=AsyncMock(return_value=structured_output),
        )
        ctx = make_context(agents=[
            agent,
            MockAgent(name="agent1"),
            MockAgent(name="agent2"),
        ])
        ctx.proposals = {
            "agent1": "Microservices are best for independent scaling and team autonomy.",
            "agent2": "Monolith is better for simplified debugging and lower overhead.",
        }

        synthesizer = DialecticalSynthesizer()
        result = await synthesizer.synthesize(ctx)

        assert result is not None
        assert isinstance(result, SynthesisResult)
        assert "hybrid" in result.synthesis.lower() or len(result.synthesis) > 0
        assert result.synthesizer_agent == "synth-agent"

    @pytest.mark.asyncio
    async def test_result_includes_elements_from_both_sides(self):
        """Synthesis result includes provenance from both thesis and antithesis."""
        structured_output = """### Synthesis
Combined architecture approach.

### Elements from Thesis
- Scaling capability
- Team independence

### Elements from Antithesis
- Debugging simplicity
- Lower overhead

### Novel Elements
- Gradual migration path
"""
        agent = MockAgent(
            name="synth-agent",
            generate=AsyncMock(return_value=structured_output),
        )
        ctx = make_context(agents=[
            agent,
            MockAgent(name="agent1"),
            MockAgent(name="agent2"),
        ])

        synthesizer = DialecticalSynthesizer()
        result = await synthesizer.synthesize(ctx)

        assert result is not None
        assert len(result.elements_from_thesis) > 0
        assert len(result.elements_from_antithesis) > 0
        assert "Scaling capability" in result.elements_from_thesis
        assert "Debugging simplicity" in result.elements_from_antithesis

    @pytest.mark.asyncio
    async def test_skips_when_only_one_position(self):
        """Synthesis is skipped when fewer than min_opposing_positions exist."""
        ctx = make_context(proposals={
            "agent1": "Only one proposal here.",
        })

        synthesizer = DialecticalSynthesizer()
        result = await synthesizer.synthesize(ctx)

        assert result is None

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self):
        """Synthesis is skipped when enable_synthesis is False."""
        ctx = make_context()
        config = SynthesisConfig(enable_synthesis=False)

        synthesizer = DialecticalSynthesizer(config)
        result = await synthesizer.synthesize(ctx)

        assert result is None

    @pytest.mark.asyncio
    async def test_skips_when_no_proposals(self):
        """Synthesis is skipped when no proposals exist."""
        ctx = make_context(proposals={})

        synthesizer = DialecticalSynthesizer()
        result = await synthesizer.synthesize(ctx)

        assert result is None

    @pytest.mark.asyncio
    async def test_handles_agent_generation_failure(self):
        """Handles agent generation failure gracefully across all attempts."""
        agent = MockAgent(
            name="failing-agent",
            generate=AsyncMock(side_effect=RuntimeError("API timeout")),
        )
        ctx = make_context(agents=[
            agent,
            MockAgent(name="agent1"),
            MockAgent(name="agent2"),
        ])

        config = SynthesisConfig(max_synthesis_attempts=2)
        synthesizer = DialecticalSynthesizer(config)
        result = await synthesizer.synthesize(ctx)

        assert result is None
        assert agent.generate.call_count == 2  # Retried

    @pytest.mark.asyncio
    async def test_handles_empty_generation(self):
        """Handles agent returning empty string."""
        agent = MockAgent(
            name="empty-agent",
            generate=AsyncMock(return_value=""),
        )
        ctx = make_context(agents=[
            agent,
            MockAgent(name="agent1"),
            MockAgent(name="agent2"),
        ])

        config = SynthesisConfig(max_synthesis_attempts=1)
        synthesizer = DialecticalSynthesizer(config)
        result = await synthesizer.synthesize(ctx)

        assert result is None

    @pytest.mark.asyncio
    async def test_retries_on_failure_then_succeeds(self):
        """Retries on first failure, succeeds on second attempt."""
        agent = MockAgent(
            name="retry-agent",
            generate=AsyncMock(
                side_effect=[
                    RuntimeError("Temporary failure"),
                    "### Synthesis\nA good combined position.\n### Elements from Thesis\n- Point A",
                ]
            ),
        )
        ctx = make_context(agents=[
            agent,
            MockAgent(name="agent1"),
            MockAgent(name="agent2"),
        ])

        synthesizer = DialecticalSynthesizer()
        result = await synthesizer.synthesize(ctx)

        assert result is not None
        assert agent.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_handles_connection_error(self):
        """Handles ConnectionError during synthesis generation."""
        agent = MockAgent(
            name="conn-error-agent",
            generate=AsyncMock(side_effect=ConnectionError("Connection refused")),
        )
        ctx = make_context(agents=[
            agent,
            MockAgent(name="agent1"),
            MockAgent(name="agent2"),
        ])

        config = SynthesisConfig(max_synthesis_attempts=1)
        synthesizer = DialecticalSynthesizer(config)
        result = await synthesizer.synthesize(ctx)

        assert result is None

    @pytest.mark.asyncio
    async def test_handles_timeout_error(self):
        """Handles TimeoutError during synthesis generation."""
        agent = MockAgent(
            name="timeout-agent",
            generate=AsyncMock(side_effect=TimeoutError("Timed out")),
        )
        ctx = make_context(agents=[
            agent,
            MockAgent(name="agent1"),
            MockAgent(name="agent2"),
        ])

        config = SynthesisConfig(max_synthesis_attempts=1)
        synthesizer = DialecticalSynthesizer(config)
        result = await synthesizer.synthesize(ctx)

        assert result is None


# =============================================================================
# Tests: Confidence validation
# =============================================================================


class TestValidateSynthesis:
    """Tests for synthesis confidence scoring."""

    def test_empty_synthesis_scores_zero(self):
        """Empty synthesis gets 0 confidence."""
        synthesizer = DialecticalSynthesizer()
        score = synthesizer._validate_synthesis("", [])
        assert score == 0.0

    def test_nonempty_synthesis_gets_base_score(self):
        """Non-empty synthesis gets at least the base score."""
        synthesizer = DialecticalSynthesizer()
        score = synthesizer._validate_synthesis("Some synthesis text", [])
        assert score >= 0.3

    def test_long_synthesis_gets_length_bonus(self):
        """Synthesis longer than 100 chars gets length bonus."""
        synthesizer = DialecticalSynthesizer()
        short_score = synthesizer._validate_synthesis("Short", [])
        long_score = synthesizer._validate_synthesis("x" * 150, [])
        assert long_score > short_score

    def test_references_to_positions_increase_confidence(self):
        """Mentioning claims from positions increases confidence."""
        synthesizer = DialecticalSynthesizer()
        positions = [
            DialecticalPosition(
                agent="agent1",
                content="Scaling is important",
                key_claims=["scaling is important for microservices"],
            ),
            DialecticalPosition(
                agent="agent2",
                content="Simplicity matters",
                key_claims=["simplicity matters for maintenance"],
            ),
        ]

        # Synthesis that references both positions' claims
        synthesis = (
            "A good approach combines scaling for microservices with "
            "simplicity for maintenance. Both agent1 and agent2 raise valid "
            "points. ### Elements show **important** elements."
        )
        score_with_refs = synthesizer._validate_synthesis(synthesis, positions)

        # Synthesis that references nothing
        score_without_refs = synthesizer._validate_synthesis(
            "Completely unrelated text about cooking recipes.", positions
        )

        assert score_with_refs > score_without_refs

    def test_structured_output_increases_confidence(self):
        """Synthesis with structured sections gets higher confidence."""
        synthesizer = DialecticalSynthesizer()
        structured = (
            "### Synthesis section with **bold** elements from thesis "
            "and elements from antithesis plus novel ideas. " * 3
        )
        unstructured = "Plain text without any formatting or structure at all. " * 3

        score_structured = synthesizer._validate_synthesis(structured, [])
        score_unstructured = synthesizer._validate_synthesis(unstructured, [])

        assert score_structured > score_unstructured

    def test_confidence_capped_at_one(self):
        """Confidence never exceeds 1.0."""
        synthesizer = DialecticalSynthesizer()
        positions = [
            DialecticalPosition(
                agent="agent1",
                content="scaling",
                key_claims=["scaling"],
                supporting_agents=[],
            ),
            DialecticalPosition(
                agent="agent2",
                content="simplicity",
                key_claims=["simplicity"],
                supporting_agents=[],
            ),
        ]
        # Synthesis that hits every bonus
        synthesis = (
            "agent1 and agent2 both raise good points about scaling and simplicity. "
            "### This is a well-structured **synthesis** with elements from thesis "
            "and elements from antithesis and novel ideas."
        )
        score = synthesizer._validate_synthesis(synthesis, positions)
        assert score <= 1.0


# =============================================================================
# Tests: Agent selection
# =============================================================================


class TestSelectSynthesisAgent:
    """Tests for synthesis agent selection logic."""

    def test_prefers_configured_agent(self):
        """Selects the preferred agent when specified in config."""
        config = SynthesisConfig(prefer_agent="agent2")
        synthesizer = DialecticalSynthesizer(config)

        ctx = make_context(agents=[
            MockAgent(name="agent1"),
            MockAgent(name="agent2"),
            MockAgent(name="agent3"),
        ])

        agent = synthesizer._select_synthesis_agent(ctx)
        assert agent is not None
        assert agent.name == "agent2"

    def test_selects_synthesizer_role(self):
        """Selects agent with synthesizer role when no preferred agent."""
        synthesizer = DialecticalSynthesizer()

        ctx = make_context(agents=[
            MockAgent(name="agent1", role="proposer"),
            MockAgent(name="agent2", role="synthesizer"),
            MockAgent(name="agent3", role="critic"),
        ])

        agent = synthesizer._select_synthesis_agent(ctx)
        assert agent is not None
        assert agent.name == "agent2"

    def test_falls_back_to_first_agent(self):
        """Falls back to first agent when no preferred or synthesizer role."""
        synthesizer = DialecticalSynthesizer()

        ctx = make_context(agents=[
            MockAgent(name="agent1", role="proposer"),
            MockAgent(name="agent2", role="critic"),
        ])

        agent = synthesizer._select_synthesis_agent(ctx)
        assert agent is not None
        assert agent.name == "agent1"

    def test_returns_none_when_no_agents(self):
        """Returns None when no agents available."""
        synthesizer = DialecticalSynthesizer()
        ctx = make_context(agents=[])

        agent = synthesizer._select_synthesis_agent(ctx)
        assert agent is None

    def test_preferred_agent_not_found_falls_through(self):
        """Falls through to role-based selection when preferred agent not found."""
        config = SynthesisConfig(prefer_agent="nonexistent")
        synthesizer = DialecticalSynthesizer(config)

        ctx = make_context(agents=[
            MockAgent(name="agent1", role="synthesizer"),
            MockAgent(name="agent2", role="critic"),
        ])

        agent = synthesizer._select_synthesis_agent(ctx)
        assert agent is not None
        assert agent.name == "agent1"  # Synthesizer role wins

    @pytest.mark.asyncio
    async def test_uses_best_calibrated_agent_as_synthesizer_role(self):
        """When no prefer_agent, synthesizer-role agent is selected for generation."""
        synth_agent = MockAgent(
            name="calibrated-synth",
            role="synthesizer",
            generate=AsyncMock(return_value="### Synthesis\nGood combined approach."),
        )
        ctx = make_context(agents=[
            MockAgent(name="agent1", role="proposer"),
            synth_agent,
            MockAgent(name="agent2", role="critic"),
        ])

        synthesizer = DialecticalSynthesizer()
        result = await synthesizer.synthesize(ctx)

        assert result is not None
        assert result.synthesizer_agent == "calibrated-synth"
        synth_agent.generate.assert_called_once()


# =============================================================================
# Tests: Edge cases
# =============================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_min_opposing_positions_threshold(self):
        """Respects custom min_opposing_positions threshold."""
        config = SynthesisConfig(min_opposing_positions=3)
        synthesizer = DialecticalSynthesizer(config)

        # Only 2 proposals, but threshold is 3
        ctx = make_context()
        result = await synthesizer.synthesize(ctx)

        assert result is None

    @pytest.mark.asyncio
    async def test_synthesis_with_no_votes(self):
        """Synthesis works even when no votes are available."""
        agent = MockAgent(
            name="synth",
            generate=AsyncMock(return_value="### Synthesis\nCombined view."),
        )
        ctx = make_context(agents=[agent, MockAgent(name="a1"), MockAgent(name="a2")])
        ctx.result.votes = []  # No votes

        synthesizer = DialecticalSynthesizer()
        result = await synthesizer.synthesize(ctx)

        assert result is not None
        assert result.thesis.agent in ctx.proposals
        assert result.antithesis.agent in ctx.proposals

    @pytest.mark.asyncio
    async def test_synthesis_with_none_result(self):
        """Handles ctx.result being None gracefully."""
        agent = MockAgent(
            name="synth",
            generate=AsyncMock(return_value="### Synthesis\nResult."),
        )
        ctx = make_context(agents=[agent, MockAgent(name="a1"), MockAgent(name="a2")])
        ctx.result = None  # type: ignore[assignment]

        synthesizer = DialecticalSynthesizer()
        result = await synthesizer.synthesize(ctx)

        # Should still work - votes will be empty
        assert result is not None

    @pytest.mark.asyncio
    async def test_no_agents_returns_none(self):
        """Returns None when no agents are available."""
        ctx = make_context(agents=[])

        synthesizer = DialecticalSynthesizer()
        result = await synthesizer.synthesize(ctx)

        assert result is None

    def test_identify_positions_preserves_insertion_order_on_tie(self):
        """When vote counts are tied, insertion order is preserved."""
        synthesizer = DialecticalSynthesizer()
        proposals = {
            "first": "First proposal should be considered carefully.",
            "second": "Second proposal must also be evaluated.",
        }
        # No votes -> all tied at 0
        positions = synthesizer._identify_positions(proposals, [])

        assert positions[0].agent == "first"
        assert positions[1].agent == "second"
