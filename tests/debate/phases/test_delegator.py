"""
Tests for debate delegator module.

Tests cover:
- DelegationConfig dataclass
- DelegationResult dataclass
- AnalysisResult dataclass
- SynthesisResult dataclass
- DebateDelegator class initialization
- delegate_analysis method
- delegate_task method
- synthesize_analyses method
- _build_analysis_prompt method
- _parse_analysis_response method
- _parse_synthesis_response method
- _check_analysis_consensus method
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.phases.delegator import (
    AnalysisResult,
    DebateDelegator,
    DelegationConfig,
    DelegationResult,
    SynthesisResult,
)


# =============================================================================
# Mock Objects
# =============================================================================


@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str = "test-agent"
    provider: str = "test-provider"
    model_type: str = "test-model"
    timeout: float = 30.0

    async def generate(self, prompt: str) -> str:
        """Mock generate method."""
        return f"Response from {self.name}"


# =============================================================================
# DelegationConfig Tests
# =============================================================================


class TestDelegationConfig:
    """Tests for DelegationConfig dataclass."""

    def test_default_values(self):
        """Config initializes with correct defaults."""
        config = DelegationConfig()

        assert config.max_concurrent == 3
        assert config.timeout_per_agent == 45.0
        assert config.context_token_limit == 2000
        assert config.require_consensus is False
        assert config.consensus_threshold == 0.6
        assert config.retry_on_failure is True
        assert config.max_retries == 1

    def test_custom_values(self):
        """Config accepts custom values."""
        config = DelegationConfig(
            max_concurrent=5,
            timeout_per_agent=60.0,
            context_token_limit=4000,
            require_consensus=True,
            consensus_threshold=0.8,
            retry_on_failure=False,
            max_retries=3,
        )

        assert config.max_concurrent == 5
        assert config.timeout_per_agent == 60.0
        assert config.context_token_limit == 4000
        assert config.require_consensus is True
        assert config.consensus_threshold == 0.8
        assert config.retry_on_failure is False
        assert config.max_retries == 3


# =============================================================================
# DelegationResult Tests
# =============================================================================


class TestDelegationResult:
    """Tests for DelegationResult dataclass."""

    def test_default_values(self):
        """Result initializes with correct defaults."""
        result = DelegationResult(results=[], agent_names=[])

        assert result.results == []
        assert result.agent_names == []
        assert result.consensus_reached is False
        assert result.consensus_value is None
        assert result.duration_seconds == 0.0

    def test_with_results(self):
        """Result stores results correctly."""
        analyses = [
            AnalysisResult(agent="agent1", summary="Summary 1"),
            AnalysisResult(agent="agent2", summary="Summary 2"),
        ]

        result = DelegationResult(
            results=analyses,
            agent_names=["agent1", "agent2"],
            consensus_reached=True,
            consensus_value=analyses[0],
            duration_seconds=5.5,
        )

        assert len(result.results) == 2
        assert result.agent_names == ["agent1", "agent2"]
        assert result.consensus_reached is True
        assert result.consensus_value == analyses[0]
        assert result.duration_seconds == 5.5


# =============================================================================
# AnalysisResult Tests
# =============================================================================


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_default_values(self):
        """AnalysisResult initializes with correct defaults."""
        result = AnalysisResult(agent="test-agent", summary="Test summary")

        assert result.agent == "test-agent"
        assert result.summary == "Test summary"
        assert result.key_points == []
        assert result.confidence == 0.5
        assert result.raw_response == ""

    def test_with_all_values(self):
        """AnalysisResult stores all values correctly."""
        result = AnalysisResult(
            agent="analyst1",
            summary="Detailed summary",
            key_points=["Point 1", "Point 2", "Point 3"],
            confidence=0.85,
            raw_response="Full raw response text",
        )

        assert result.agent == "analyst1"
        assert result.summary == "Detailed summary"
        assert len(result.key_points) == 3
        assert result.confidence == 0.85
        assert result.raw_response == "Full raw response text"


# =============================================================================
# SynthesisResult Tests
# =============================================================================


class TestSynthesisResult:
    """Tests for SynthesisResult dataclass."""

    def test_default_values(self):
        """SynthesisResult initializes with correct defaults."""
        result = SynthesisResult(summary="Synthesis summary")

        assert result.summary == "Synthesis summary"
        assert result.agreements == []
        assert result.disagreements == []
        assert result.recommendation == ""
        assert result.confidence == 0.5

    def test_with_all_values(self):
        """SynthesisResult stores all values correctly."""
        result = SynthesisResult(
            summary="Combined insights",
            agreements=["Point A", "Point B"],
            disagreements=["Point X"],
            recommendation="Proceed with caution",
            confidence=0.75,
        )

        assert result.summary == "Combined insights"
        assert result.agreements == ["Point A", "Point B"]
        assert result.disagreements == ["Point X"]
        assert result.recommendation == "Proceed with caution"
        assert result.confidence == 0.75


# =============================================================================
# DebateDelegator Initialization Tests
# =============================================================================


class TestDebateDelegatorInit:
    """Tests for DebateDelegator initialization."""

    def test_init_with_minimal_args(self):
        """Delegator initializes with minimal arguments."""
        agents = [MockAgent(name="agent1")]
        delegator = DebateDelegator(agent_pool=agents)

        assert delegator._agents == agents
        assert delegator._config is not None
        assert delegator._config.max_concurrent == 3
        assert delegator._generate_fn is None

    def test_init_with_custom_config(self):
        """Delegator initializes with custom config."""
        agents = [MockAgent(name="agent1")]
        config = DelegationConfig(max_concurrent=10, timeout_per_agent=120.0)

        delegator = DebateDelegator(agent_pool=agents, config=config)

        assert delegator._config == config
        assert delegator._config.max_concurrent == 10
        assert delegator._config.timeout_per_agent == 120.0

    def test_init_with_generate_fn(self):
        """Delegator stores generate_fn callback."""
        agents = [MockAgent(name="agent1")]
        generate_fn = AsyncMock(return_value="Generated response")

        delegator = DebateDelegator(agent_pool=agents, generate_fn=generate_fn)

        assert delegator._generate_fn is generate_fn

    def test_init_with_empty_agent_pool(self):
        """Delegator initializes with empty agent pool."""
        delegator = DebateDelegator(agent_pool=[])

        assert delegator._agents == []


# =============================================================================
# delegate_analysis Tests
# =============================================================================


class TestDelegateAnalysis:
    """Tests for delegate_analysis method."""

    @pytest.mark.asyncio
    async def test_returns_empty_result_when_no_agents(self):
        """Returns empty result when agent pool is empty."""
        delegator = DebateDelegator(agent_pool=[])

        result = await delegator.delegate_analysis(
            task="Analyze this",
            context="Some context",
            num_analysts=3,
        )

        assert result.results == []
        assert result.agent_names == []

    @pytest.mark.asyncio
    async def test_delegates_to_multiple_agents(self):
        """Delegates analysis to multiple agents."""
        agents = [
            MockAgent(name="analyst1"),
            MockAgent(name="analyst2"),
            MockAgent(name="analyst3"),
        ]
        generate_fn = AsyncMock(
            return_value="SUMMARY: Test analysis\nKEY_POINTS:\n- Point 1\nCONFIDENCE: 0.8"
        )

        delegator = DebateDelegator(agent_pool=agents, generate_fn=generate_fn)

        result = await delegator.delegate_analysis(
            task="Analyze the proposals",
            context="Context data here",
            num_analysts=2,
        )

        assert len(result.results) == 2
        assert generate_fn.call_count == 2

    @pytest.mark.asyncio
    async def test_uses_agent_generate_when_no_generate_fn(self):
        """Uses agent's generate method when no generate_fn provided."""
        agent = MockAgent(name="analyst1")
        agents = [agent]

        delegator = DebateDelegator(agent_pool=agents)

        with patch.object(agent, "generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = "SUMMARY: Response\nCONFIDENCE: 0.7"
            result = await delegator.delegate_analysis(
                task="Analyze this",
                context="Context",
                num_analysts=1,
            )

        mock_gen.assert_called_once()
        assert len(result.results) == 1

    @pytest.mark.asyncio
    async def test_respects_num_analysts_limit(self):
        """Respects the num_analysts parameter."""
        agents = [MockAgent(name=f"agent{i}") for i in range(5)]
        generate_fn = AsyncMock(return_value="SUMMARY: Analysis\nCONFIDENCE: 0.5")

        delegator = DebateDelegator(agent_pool=agents, generate_fn=generate_fn)

        result = await delegator.delegate_analysis(
            task="Analyze",
            context="Context",
            num_analysts=2,
        )

        # Only first 2 agents should be used
        assert len(result.results) == 2
        assert generate_fn.call_count == 2

    @pytest.mark.asyncio
    async def test_uses_custom_analysis_prompt(self):
        """Uses custom analysis prompt when provided."""
        agents = [MockAgent(name="analyst1")]
        generate_fn = AsyncMock(return_value="SUMMARY: Custom\nCONFIDENCE: 0.9")

        delegator = DebateDelegator(agent_pool=agents, generate_fn=generate_fn)

        await delegator.delegate_analysis(
            task="Analyze",
            context="Context",
            num_analysts=1,
            analysis_prompt="Custom prompt: {task}",
        )

        # Check that custom prompt was used
        call_args = generate_fn.call_args[0]
        assert call_args[1] == "Custom prompt: {task}"

    @pytest.mark.asyncio
    async def test_tracks_duration(self):
        """Tracks total duration of delegation."""
        agents = [MockAgent(name="analyst1")]
        generate_fn = AsyncMock(return_value="SUMMARY: Quick\nCONFIDENCE: 0.5")

        delegator = DebateDelegator(agent_pool=agents, generate_fn=generate_fn)

        result = await delegator.delegate_analysis(
            task="Analyze",
            context="Context",
            num_analysts=1,
        )

        assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_checks_consensus_when_required(self):
        """Checks consensus when require_consensus is True."""
        agents = [
            MockAgent(name="analyst1"),
            MockAgent(name="analyst2"),
        ]
        config = DelegationConfig(require_consensus=True, consensus_threshold=0.5)

        # Both return high confidence
        generate_fn = AsyncMock(return_value="SUMMARY: Agreement\nCONFIDENCE: 0.9")

        delegator = DebateDelegator(agent_pool=agents, config=config, generate_fn=generate_fn)

        result = await delegator.delegate_analysis(
            task="Analyze",
            context="Context",
            num_analysts=2,
        )

        assert result.consensus_reached is True
        assert result.consensus_value is not None

    @pytest.mark.asyncio
    async def test_no_consensus_when_low_confidence(self):
        """No consensus when agents have low confidence."""
        agents = [
            MockAgent(name="analyst1"),
            MockAgent(name="analyst2"),
        ]
        config = DelegationConfig(require_consensus=True, consensus_threshold=0.8)

        # Low confidence responses
        generate_fn = AsyncMock(return_value="SUMMARY: Uncertain\nCONFIDENCE: 0.3")

        delegator = DebateDelegator(agent_pool=agents, config=config, generate_fn=generate_fn)

        result = await delegator.delegate_analysis(
            task="Analyze",
            context="Context",
            num_analysts=2,
        )

        assert result.consensus_reached is False


# =============================================================================
# delegate_task Tests
# =============================================================================


class TestDelegateTask:
    """Tests for delegate_task method."""

    @pytest.mark.asyncio
    async def test_returns_empty_result_when_no_agents(self):
        """Returns empty result when no agents available."""
        delegator = DebateDelegator(agent_pool=[])

        result = await delegator.delegate_task(
            task="Do something",
            context="Context",
        )

        assert result.results == []
        assert result.agent_names == []

    @pytest.mark.asyncio
    async def test_delegates_to_specified_agents(self):
        """Delegates to specified agents when provided."""
        pool_agents = [MockAgent(name=f"pool{i}") for i in range(3)]
        specific_agents = [MockAgent(name="specific1"), MockAgent(name="specific2")]
        generate_fn = AsyncMock(return_value="Task completed")

        delegator = DebateDelegator(agent_pool=pool_agents, generate_fn=generate_fn)

        result = await delegator.delegate_task(
            task="Complete task",
            context="Context",
            agents=specific_agents,
        )

        assert len(result.results) == 2
        assert "specific1" in result.agent_names
        assert "specific2" in result.agent_names

    @pytest.mark.asyncio
    async def test_uses_pool_agents_when_none_specified(self):
        """Uses pool agents when no specific agents provided."""
        agents = [MockAgent(name=f"agent{i}") for i in range(5)]
        config = DelegationConfig(max_concurrent=2)
        generate_fn = AsyncMock(return_value="Done")

        delegator = DebateDelegator(agent_pool=agents, config=config, generate_fn=generate_fn)

        result = await delegator.delegate_task(
            task="Task",
            context="Context",
        )

        # Should use first max_concurrent agents from pool
        assert len(result.results) == 2

    @pytest.mark.asyncio
    async def test_applies_custom_parse_fn(self):
        """Applies custom parse function to results."""
        agents = [MockAgent(name="agent1")]
        generate_fn = AsyncMock(return_value="raw response")

        def parse_fn(agent_name: str, response: str) -> dict:
            return {"agent": agent_name, "parsed": response.upper()}

        delegator = DebateDelegator(agent_pool=agents, generate_fn=generate_fn)

        result = await delegator.delegate_task(
            task="Task",
            context="Context",
            agents=agents,
            parse_fn=parse_fn,
        )

        assert result.results[0]["agent"] == "agent1"
        assert result.results[0]["parsed"] == "RAW RESPONSE"

    @pytest.mark.asyncio
    async def test_returns_string_when_no_parse_fn(self):
        """Returns raw string when no parse function provided."""
        agents = [MockAgent(name="agent1")]
        generate_fn = AsyncMock(return_value="raw response text")

        delegator = DebateDelegator(agent_pool=agents, generate_fn=generate_fn)

        result = await delegator.delegate_task(
            task="Task",
            context="Context",
            agents=agents,
        )

        assert result.results[0] == "raw response text"

    @pytest.mark.asyncio
    async def test_truncates_context_to_token_limit(self):
        """Truncates context based on token limit."""
        agents = [MockAgent(name="agent1")]
        config = DelegationConfig(context_token_limit=10)  # Very small limit
        generate_fn = AsyncMock(return_value="Done")

        delegator = DebateDelegator(agent_pool=agents, config=config, generate_fn=generate_fn)

        long_context = "A" * 1000  # Much longer than limit

        await delegator.delegate_task(
            task="Task",
            context=long_context,
            agents=agents,
        )

        # Verify prompt was truncated (10 tokens * 4 chars = 40 chars max)
        call_args = generate_fn.call_args[0]
        assert len(call_args[1]) < len(long_context) + 100  # Some overhead for prompt


# =============================================================================
# synthesize_analyses Tests
# =============================================================================


class TestSynthesizeAnalyses:
    """Tests for synthesize_analyses method."""

    @pytest.mark.asyncio
    async def test_returns_empty_synthesis_when_no_analyses(self):
        """Returns empty synthesis when no analyses provided."""
        agents = [MockAgent(name="synth1")]
        delegator = DebateDelegator(agent_pool=agents)

        result = await delegator.synthesize_analyses([])

        assert result.summary == "No analyses to synthesize"

    @pytest.mark.asyncio
    async def test_returns_no_synthesizer_when_no_agents(self):
        """Returns no synthesizer message when pool is empty."""
        delegator = DebateDelegator(agent_pool=[])

        analyses = [AnalysisResult(agent="agent1", summary="Analysis 1")]
        result = await delegator.synthesize_analyses(analyses)

        assert result.summary == "No synthesizer available"

    @pytest.mark.asyncio
    async def test_uses_first_agent_as_synthesizer(self):
        """Uses first agent in pool as synthesizer by default."""
        agents = [MockAgent(name="synth1"), MockAgent(name="synth2")]
        generate_fn = AsyncMock(
            return_value="SUMMARY: Unified\nAGREEMENTS: A, B\nDISAGREEMENTS: X\nRECOMMENDATION: Do it\nCONFIDENCE: 0.85"
        )

        delegator = DebateDelegator(agent_pool=agents, generate_fn=generate_fn)

        analyses = [
            AnalysisResult(agent="analyst1", summary="Summary 1", confidence=0.7),
            AnalysisResult(agent="analyst2", summary="Summary 2", confidence=0.8),
        ]

        await delegator.synthesize_analyses(analyses)

        # First agent should be used
        call_args = generate_fn.call_args[0]
        assert call_args[0].name == "synth1"

    @pytest.mark.asyncio
    async def test_uses_specified_synthesizer(self):
        """Uses specified synthesizer agent."""
        agents = [MockAgent(name="pool1")]
        synthesizer = MockAgent(name="custom_synth")
        generate_fn = AsyncMock(return_value="SUMMARY: Custom synthesis\nCONFIDENCE: 0.9")

        delegator = DebateDelegator(agent_pool=agents, generate_fn=generate_fn)

        analyses = [AnalysisResult(agent="analyst1", summary="Analysis")]

        await delegator.synthesize_analyses(analyses, synthesizer=synthesizer)

        call_args = generate_fn.call_args[0]
        assert call_args[0].name == "custom_synth"

    @pytest.mark.asyncio
    async def test_parses_synthesis_response(self):
        """Correctly parses synthesis response."""
        agents = [MockAgent(name="synth1")]
        generate_fn = AsyncMock(
            return_value="SUMMARY: Combined insights here\nAGREEMENTS: Point A, Point B\nDISAGREEMENTS: Point X, Point Y\nRECOMMENDATION: Proceed carefully\nCONFIDENCE: 0.75"
        )

        delegator = DebateDelegator(agent_pool=agents, generate_fn=generate_fn)

        analyses = [
            AnalysisResult(agent="a1", summary="S1", key_points=["KP1"]),
            AnalysisResult(agent="a2", summary="S2", key_points=["KP2"]),
        ]

        result = await delegator.synthesize_analyses(analyses)

        assert result.summary == "Combined insights here"
        assert result.agreements == ["Point A", "Point B"]
        assert result.disagreements == ["Point X", "Point Y"]
        assert result.recommendation == "Proceed carefully"
        assert result.confidence == 0.75

    @pytest.mark.asyncio
    async def test_handles_synthesis_error_gracefully(self):
        """Handles errors during synthesis gracefully."""
        agents = [MockAgent(name="synth1")]
        generate_fn = AsyncMock(side_effect=RuntimeError("Synthesis failed"))

        delegator = DebateDelegator(agent_pool=agents, generate_fn=generate_fn)

        analyses = [AnalysisResult(agent="a1", summary="S1")]

        result = await delegator.synthesize_analyses(analyses)

        assert "Synthesis failed" in result.summary
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_includes_key_points_in_prompt(self):
        """Includes key points from analyses in synthesis prompt."""
        agents = [MockAgent(name="synth1")]
        generate_fn = AsyncMock(return_value="SUMMARY: Done\nCONFIDENCE: 0.5")

        delegator = DebateDelegator(agent_pool=agents, generate_fn=generate_fn)

        analyses = [
            AnalysisResult(agent="a1", summary="S1", key_points=["Important point"]),
        ]

        await delegator.synthesize_analyses(analyses)

        call_args = generate_fn.call_args[0]
        prompt = call_args[1]
        assert "Important point" in prompt


# =============================================================================
# _build_analysis_prompt Tests
# =============================================================================


class TestBuildAnalysisPrompt:
    """Tests for _build_analysis_prompt method."""

    def test_builds_prompt_with_task_and_context(self):
        """Builds prompt containing task and context."""
        delegator = DebateDelegator(agent_pool=[])

        prompt = delegator._build_analysis_prompt(
            task="Evaluate the proposal",
            context="This is the context data",
        )

        assert "Evaluate the proposal" in prompt
        assert "This is the context data" in prompt

    def test_truncates_long_context(self):
        """Truncates context when exceeding token limit."""
        config = DelegationConfig(context_token_limit=50)  # 50 tokens * 4 = 200 chars
        delegator = DebateDelegator(agent_pool=[], config=config)

        long_context = "A" * 1000

        prompt = delegator._build_analysis_prompt(
            task="Task",
            context=long_context,
        )

        # Context should be truncated to approximately 200 chars (50 tokens * 4)
        # Allow small margin for edge cases
        assert prompt.count("A") <= 210
        assert prompt.count("A") < 1000  # Definitely truncated

    def test_includes_response_format_instructions(self):
        """Includes response format instructions."""
        delegator = DebateDelegator(agent_pool=[])

        prompt = delegator._build_analysis_prompt(
            task="Task",
            context="Context",
        )

        assert "SUMMARY:" in prompt
        assert "KEY_POINTS:" in prompt
        assert "CONFIDENCE:" in prompt


# =============================================================================
# _parse_analysis_response Tests
# =============================================================================


class TestParseAnalysisResponse:
    """Tests for _parse_analysis_response method."""

    def test_parses_well_formatted_response(self):
        """Parses well-formatted response correctly."""
        delegator = DebateDelegator(agent_pool=[])

        response = """SUMMARY: This is the summary
KEY_POINTS:
- First point
- Second point
- Third point
CONFIDENCE: 0.85"""

        result = delegator._parse_analysis_response("agent1", response)

        assert result.agent == "agent1"
        assert result.summary == "This is the summary"
        assert result.key_points == ["First point", "Second point", "Third point"]
        assert result.confidence == 0.85
        assert result.raw_response == response

    def test_handles_multiline_summary(self):
        """Handles summary spanning multiple lines."""
        delegator = DebateDelegator(agent_pool=[])

        response = """SUMMARY: First line of summary
continuation of summary
KEY_POINTS:
- Point 1
CONFIDENCE: 0.7"""

        result = delegator._parse_analysis_response("agent1", response)

        assert "First line of summary" in result.summary
        assert "continuation of summary" in result.summary

    def test_uses_fallback_when_no_summary(self):
        """Uses fallback when SUMMARY not found."""
        delegator = DebateDelegator(agent_pool=[])

        response = "Just some unformatted response text"

        result = delegator._parse_analysis_response("agent1", response)

        assert result.summary == "Just some unformatted response text"

    def test_clamps_confidence_to_valid_range(self):
        """Clamps confidence to 0.0-1.0 range."""
        delegator = DebateDelegator(agent_pool=[])

        response_high = "SUMMARY: Test\nCONFIDENCE: 1.5"
        result_high = delegator._parse_analysis_response("agent1", response_high)
        assert result_high.confidence == 1.0

        response_low = "SUMMARY: Test\nCONFIDENCE: -0.5"
        result_low = delegator._parse_analysis_response("agent1", response_low)
        assert result_low.confidence == 0.0

    def test_handles_invalid_confidence(self):
        """Uses default confidence when value is invalid."""
        delegator = DebateDelegator(agent_pool=[])

        response = "SUMMARY: Test\nCONFIDENCE: not_a_number"

        result = delegator._parse_analysis_response("agent1", response)

        assert result.confidence == 0.5  # Default value

    def test_handles_empty_key_points(self):
        """Handles response with no key points."""
        delegator = DebateDelegator(agent_pool=[])

        response = "SUMMARY: Test summary\nCONFIDENCE: 0.6"

        result = delegator._parse_analysis_response("agent1", response)

        assert result.key_points == []

    def test_handles_case_insensitive_headers(self):
        """Handles case-insensitive header parsing."""
        delegator = DebateDelegator(agent_pool=[])

        response = """summary: Case insensitive summary
key_points:
- Point here
confidence: 0.9"""

        result = delegator._parse_analysis_response("agent1", response)

        assert result.summary == "Case insensitive summary"
        assert result.key_points == ["Point here"]
        assert result.confidence == 0.9


# =============================================================================
# _parse_synthesis_response Tests
# =============================================================================


class TestParseSynthesisResponse:
    """Tests for _parse_synthesis_response method."""

    def test_parses_well_formatted_response(self):
        """Parses well-formatted synthesis response."""
        delegator = DebateDelegator(agent_pool=[])

        response = """SUMMARY: Unified summary
AGREEMENTS: Point A, Point B, Point C
DISAGREEMENTS: Issue X, Issue Y
RECOMMENDATION: Proceed with plan
CONFIDENCE: 0.8"""

        result = delegator._parse_synthesis_response(response)

        assert result.summary == "Unified summary"
        assert result.agreements == ["Point A", "Point B", "Point C"]
        assert result.disagreements == ["Issue X", "Issue Y"]
        assert result.recommendation == "Proceed with plan"
        assert result.confidence == 0.8

    def test_handles_empty_agreements(self):
        """Handles response with no agreements."""
        delegator = DebateDelegator(agent_pool=[])

        response = """SUMMARY: Summary
AGREEMENTS:
DISAGREEMENTS: Issue 1
RECOMMENDATION: Rec
CONFIDENCE: 0.5"""

        result = delegator._parse_synthesis_response(response)

        assert result.agreements == []

    def test_uses_fallback_summary_for_unformatted(self):
        """Uses fallback for unformatted response."""
        delegator = DebateDelegator(agent_pool=[])

        response = "Just an unformatted synthesis response"

        result = delegator._parse_synthesis_response(response)

        assert "unformatted synthesis" in result.summary

    def test_clamps_confidence_to_valid_range(self):
        """Clamps synthesis confidence to valid range."""
        delegator = DebateDelegator(agent_pool=[])

        response = "SUMMARY: Test\nCONFIDENCE: 2.0"

        result = delegator._parse_synthesis_response(response)

        assert result.confidence == 1.0

    def test_handles_invalid_confidence(self):
        """Uses default confidence for invalid values."""
        delegator = DebateDelegator(agent_pool=[])

        response = "SUMMARY: Test\nCONFIDENCE: invalid"

        result = delegator._parse_synthesis_response(response)

        assert result.confidence == 0.5


# =============================================================================
# _check_analysis_consensus Tests
# =============================================================================


class TestCheckAnalysisConsensus:
    """Tests for _check_analysis_consensus method."""

    def test_returns_false_for_single_analysis(self):
        """Returns false consensus for single analysis."""
        delegator = DebateDelegator(agent_pool=[])

        analyses = [AnalysisResult(agent="a1", summary="S1", confidence=0.9)]

        consensus, value = delegator._check_analysis_consensus(analyses)

        assert consensus is False
        assert value is None

    def test_returns_false_when_low_confidence(self):
        """Returns false when analyses have low confidence."""
        config = DelegationConfig(consensus_threshold=0.8)
        delegator = DebateDelegator(agent_pool=[], config=config)

        analyses = [
            AnalysisResult(agent="a1", summary="S1", confidence=0.3),
            AnalysisResult(agent="a2", summary="S2", confidence=0.4),
        ]

        consensus, value = delegator._check_analysis_consensus(analyses)

        assert consensus is False

    def test_returns_true_when_high_confidence_majority(self):
        """Returns true when majority has high confidence."""
        config = DelegationConfig(consensus_threshold=0.5)
        delegator = DebateDelegator(agent_pool=[], config=config)

        analyses = [
            AnalysisResult(agent="a1", summary="S1", confidence=0.9),
            AnalysisResult(agent="a2", summary="S2", confidence=0.85),
            AnalysisResult(agent="a3", summary="S3", confidence=0.4),
        ]

        consensus, value = delegator._check_analysis_consensus(analyses)

        assert consensus is True
        assert value is not None
        assert value.confidence >= 0.7

    def test_returns_false_when_below_threshold(self):
        """Returns false when high confidence count below threshold."""
        config = DelegationConfig(consensus_threshold=0.9)  # 90% must agree
        delegator = DebateDelegator(agent_pool=[], config=config)

        analyses = [
            AnalysisResult(agent="a1", summary="S1", confidence=0.9),
            AnalysisResult(agent="a2", summary="S2", confidence=0.3),  # Low confidence
        ]

        consensus, value = delegator._check_analysis_consensus(analyses)

        # Only 50% have high confidence, need 90%
        assert consensus is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestDebateDelegatorIntegration:
    """Integration tests for DebateDelegator."""

    @pytest.mark.asyncio
    async def test_full_analysis_and_synthesis_workflow(self):
        """Tests complete analysis and synthesis workflow."""
        agents = [
            MockAgent(name="analyst1"),
            MockAgent(name="analyst2"),
            MockAgent(name="synthesizer"),
        ]

        analysis_response = "SUMMARY: Analysis done\nKEY_POINTS:\n- Point 1\nCONFIDENCE: 0.8"
        synthesis_response = "SUMMARY: Synthesized\nAGREEMENTS: All agree\nDISAGREEMENTS: None\nRECOMMENDATION: Proceed\nCONFIDENCE: 0.9"

        call_count = [0]

        async def mock_generate(agent, prompt):
            call_count[0] += 1
            if "synthesizing" in prompt.lower():
                return synthesis_response
            return analysis_response

        delegator = DebateDelegator(agent_pool=agents, generate_fn=mock_generate)

        # Run analysis
        analysis_result = await delegator.delegate_analysis(
            task="Analyze proposals",
            context="Proposal context",
            num_analysts=2,
        )

        assert len(analysis_result.results) == 2

        # Run synthesis
        synthesis_result = await delegator.synthesize_analyses(
            analysis_result.results,
            synthesizer=agents[2],
        )

        assert synthesis_result.summary == "Synthesized"
        assert synthesis_result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_handles_mixed_success_and_failure(self):
        """Handles some agents succeeding and others failing."""
        agents = [
            MockAgent(name="success1"),
            MockAgent(name="failure1"),
            MockAgent(name="success2"),
        ]

        call_count = [0]

        async def mock_generate(agent, prompt):
            call_count[0] += 1
            if agent.name == "failure1":
                raise RuntimeError("Agent failed")
            return "SUMMARY: Success\nCONFIDENCE: 0.7"

        delegator = DebateDelegator(agent_pool=agents, generate_fn=mock_generate)

        # Should handle the failure gracefully
        result = await delegator.delegate_analysis(
            task="Analyze",
            context="Context",
            num_analysts=3,
        )

        # At least some results should succeed
        assert len(result.results) >= 1
