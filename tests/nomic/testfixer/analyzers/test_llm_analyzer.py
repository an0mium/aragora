"""
Comprehensive tests for LLMFailureAnalyzer.

Tests the LLM-powered failure analyzer including:
- Configuration defaults and custom settings
- Text truncation with middle ellipsis
- Tag extraction from structured responses
- Confidence parsing (decimal and percentage)
- Prompt building for analysis and validation
- Agent analysis with mocked responses
- Cross-validation between agents
- Synthesis of multiple analyses
- Full analyze flow integration
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from aragora.nomic.testfixer.analyzers.llm_analyzer import (
    LLMFailureAnalyzer,
    LLMAnalyzerConfig,
    AgentAnalysis,
)
from aragora.nomic.testfixer.runner import TestFailure


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_test_failure() -> TestFailure:
    """Create a sample test failure for testing."""
    return TestFailure(
        test_name="test_user_authentication",
        test_file="tests/test_auth.py",
        error_type="AssertionError",
        error_message="assert user.is_authenticated == True",
        stack_trace="""
Traceback (most recent call last):
  File "tests/test_auth.py", line 42, in test_user_authentication
    assert user.is_authenticated == True
AssertionError: assert False == True
""",
        line_number=42,
        involved_files=["src/auth/user.py", "src/auth/session.py"],
        involved_functions=["authenticate", "create_session"],
    )


@pytest.fixture
def sample_code_context() -> dict[str, str]:
    """Sample code context for tests."""
    return {
        "src/auth/user.py": """
class User:
    def __init__(self, username):
        self.username = username
        self.is_authenticated = False

    def authenticate(self, password):
        # Bug: Not setting is_authenticated
        return self._verify_password(password)
""",
        "tests/test_auth.py": """
def test_user_authentication():
    user = User("testuser")
    user.authenticate("correct_password")
    assert user.is_authenticated == True
""",
    }


@pytest.fixture
def default_config() -> LLMAnalyzerConfig:
    """Create default config."""
    return LLMAnalyzerConfig()


@pytest.fixture
def mock_agent() -> MagicMock:
    """Create a mock agent."""
    agent = MagicMock()
    agent.name = "mock_agent"
    agent.generate = AsyncMock()
    return agent


# ---------------------------------------------------------------------------
# LLMAnalyzerConfig Tests
# ---------------------------------------------------------------------------


class TestLLMAnalyzerConfig:
    """Tests for LLMAnalyzerConfig dataclass."""

    def test_default_agent_types(self):
        """Test default agent types."""
        config = LLMAnalyzerConfig()
        assert "anthropic-api" in config.agent_types
        assert "openai-api" in config.agent_types
        assert len(config.agent_types) == 2

    def test_default_max_context_chars(self):
        """Test default context limit."""
        config = LLMAnalyzerConfig()
        assert config.max_context_chars == 60_000

    def test_default_synthesis_threshold(self):
        """Test default synthesis threshold."""
        config = LLMAnalyzerConfig()
        assert config.synthesis_threshold == 0.6

    def test_default_cross_validate(self):
        """Test cross-validation is enabled by default."""
        config = LLMAnalyzerConfig()
        assert config.cross_validate is True

    def test_default_agent_timeout(self):
        """Test default agent timeout."""
        config = LLMAnalyzerConfig()
        assert config.agent_timeout == 60.0

    def test_default_consensus_settings(self):
        """Test default consensus settings."""
        config = LLMAnalyzerConfig()
        assert config.require_consensus is False
        assert config.consensus_threshold == 0.7

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LLMAnalyzerConfig(
            agent_types=["anthropic-api", "gemini-api", "openai-api"],
            max_context_chars=30_000,
            synthesis_threshold=0.8,
            cross_validate=False,
            agent_timeout=120.0,
            require_consensus=True,
            consensus_threshold=0.9,
        )
        assert len(config.agent_types) == 3
        assert config.max_context_chars == 30_000
        assert config.synthesis_threshold == 0.8
        assert config.cross_validate is False
        assert config.agent_timeout == 120.0
        assert config.require_consensus is True
        assert config.consensus_threshold == 0.9

    def test_models_default_none(self):
        """Test models default to None."""
        config = LLMAnalyzerConfig()
        assert config.models is None

    def test_custom_models(self):
        """Test custom model specification."""
        config = LLMAnalyzerConfig(
            models={
                "anthropic-api": "claude-3-opus-20240229",
                "openai-api": "gpt-4o",
            }
        )
        assert config.models["anthropic-api"] == "claude-3-opus-20240229"
        assert config.models["openai-api"] == "gpt-4o"


# ---------------------------------------------------------------------------
# AgentAnalysis Tests
# ---------------------------------------------------------------------------


class TestAgentAnalysis:
    """Tests for AgentAnalysis dataclass."""

    def test_basic_creation(self):
        """Test creating an AgentAnalysis."""
        analysis = AgentAnalysis(
            agent_name="test_agent",
            root_cause="Missing authentication flag",
            approach="Set is_authenticated = True after verification",
            confidence=0.85,
        )
        assert analysis.agent_name == "test_agent"
        assert analysis.root_cause == "Missing authentication flag"
        assert analysis.approach == "Set is_authenticated = True after verification"
        assert analysis.confidence == 0.85

    def test_optional_fields(self):
        """Test optional fields default values."""
        analysis = AgentAnalysis(
            agent_name="test",
            root_cause="cause",
            approach="fix",
            confidence=0.5,
        )
        assert analysis.category is None
        assert analysis.raw_response == ""

    def test_with_category(self):
        """Test with category specified."""
        analysis = AgentAnalysis(
            agent_name="test",
            root_cause="cause",
            approach="fix",
            confidence=0.5,
            category="assertion_error",
        )
        assert analysis.category == "assertion_error"


# ---------------------------------------------------------------------------
# LLMFailureAnalyzer - Utility Methods Tests
# ---------------------------------------------------------------------------


class TestLLMFailureAnalyzerUtilities:
    """Tests for LLMFailureAnalyzer utility methods."""

    def test_truncate_short_text(self):
        """Test truncate with text under limit."""
        analyzer = LLMFailureAnalyzer()
        text = "Short text"
        result = analyzer._truncate(text, 100)
        assert result == text

    def test_truncate_exact_limit(self):
        """Test truncate with text exactly at limit."""
        analyzer = LLMFailureAnalyzer()
        text = "A" * 100
        result = analyzer._truncate(text, 100)
        assert result == text

    def test_truncate_long_text(self):
        """Test truncate with text over limit."""
        analyzer = LLMFailureAnalyzer()
        text = "A" * 1000
        result = analyzer._truncate(text, 500)

        # Should have head, ellipsis info, and tail
        assert len(result) < len(text)
        assert "truncated" in result
        assert result.startswith("A")
        assert result.endswith("A")

    def test_truncate_preserves_structure(self):
        """Test truncate preserves head and tail."""
        analyzer = LLMFailureAnalyzer()
        text = "START" + "X" * 1000 + "END"
        result = analyzer._truncate(text, 300)

        assert result.startswith("START")
        assert result.endswith("END")

    def test_extract_tag_found(self):
        """Test tag extraction when tag exists."""
        import re
        from aragora.nomic.testfixer.analyzers.llm_analyzer import _ROOT_CAUSE_RE

        analyzer = LLMFailureAnalyzer()
        text = "Some text <root_cause>The real cause</root_cause> more text"
        result = analyzer._extract_tag(_ROOT_CAUSE_RE, text)

        assert result == "The real cause"

    def test_extract_tag_not_found(self):
        """Test tag extraction when tag missing."""
        import re
        from aragora.nomic.testfixer.analyzers.llm_analyzer import _ROOT_CAUSE_RE

        analyzer = LLMFailureAnalyzer()
        text = "Some text without the tag"
        result = analyzer._extract_tag(_ROOT_CAUSE_RE, text)

        assert result is None

    def test_extract_tag_multiline(self):
        """Test tag extraction with multiline content."""
        import re
        from aragora.nomic.testfixer.analyzers.llm_analyzer import _APPROACH_RE

        analyzer = LLMFailureAnalyzer()
        text = """<approach>
Step 1: Do this
Step 2: Do that
Step 3: Done
</approach>"""
        result = analyzer._extract_tag(_APPROACH_RE, text)

        assert "Step 1" in result
        assert "Step 2" in result
        assert "Step 3" in result

    def test_extract_tag_case_insensitive(self):
        """Test tag extraction is case insensitive."""
        import re
        from aragora.nomic.testfixer.analyzers.llm_analyzer import _ROOT_CAUSE_RE

        analyzer = LLMFailureAnalyzer()
        text = "<ROOT_CAUSE>Upper case tag</ROOT_CAUSE>"
        result = analyzer._extract_tag(_ROOT_CAUSE_RE, text)

        assert result == "Upper case tag"


class TestConfidenceParsing:
    """Tests for confidence parsing."""

    def test_parse_decimal_confidence(self):
        """Test parsing decimal confidence."""
        analyzer = LLMFailureAnalyzer()
        text = "<confidence>0.85</confidence>"
        result = analyzer._parse_confidence(text)

        assert result == 0.85

    def test_parse_percentage_confidence(self):
        """Test parsing percentage confidence."""
        analyzer = LLMFailureAnalyzer()
        text = "<confidence>85%</confidence>"
        result = analyzer._parse_confidence(text)

        assert result == 0.85

    def test_parse_confidence_default(self):
        """Test default confidence when not found."""
        analyzer = LLMFailureAnalyzer()
        text = "No confidence tag here"
        result = analyzer._parse_confidence(text)

        assert result == 0.5  # Default

    def test_parse_confidence_custom_default(self):
        """Test custom default confidence."""
        analyzer = LLMFailureAnalyzer()
        result = analyzer._parse_confidence("no tag", default=0.3)

        assert result == 0.3

    def test_parse_confidence_clamp_high(self):
        """Test confidence is clamped to 1.0 max."""
        analyzer = LLMFailureAnalyzer()
        text = "<confidence>1.5</confidence>"
        result = analyzer._parse_confidence(text)

        assert result == 1.0

    def test_parse_confidence_clamp_low(self):
        """Test confidence is clamped to 0.0 min."""
        analyzer = LLMFailureAnalyzer()
        text = "<confidence>-0.5</confidence>"
        result = analyzer._parse_confidence(text)

        assert result == 0.0

    def test_parse_confidence_invalid(self):
        """Test handling invalid confidence value."""
        analyzer = LLMFailureAnalyzer()
        text = "<confidence>invalid</confidence>"
        result = analyzer._parse_confidence(text)

        assert result == 0.5  # Default


# ---------------------------------------------------------------------------
# Prompt Building Tests
# ---------------------------------------------------------------------------


class TestPromptBuilding:
    """Tests for prompt building methods."""

    def test_build_analysis_prompt_basic(self, sample_test_failure):
        """Test basic analysis prompt building."""
        analyzer = LLMFailureAnalyzer()
        prompt = analyzer._build_analysis_prompt(sample_test_failure, {})

        assert "test_user_authentication" in prompt
        assert "tests/test_auth.py" in prompt
        assert "AssertionError" in prompt
        assert "assert user.is_authenticated == True" in prompt

    def test_build_analysis_prompt_with_context(self, sample_test_failure, sample_code_context):
        """Test prompt building with code context."""
        analyzer = LLMFailureAnalyzer()
        prompt = analyzer._build_analysis_prompt(sample_test_failure, sample_code_context)

        assert "src/auth/user.py" in prompt
        assert "class User:" in prompt
        assert "is_authenticated = False" in prompt

    def test_build_analysis_prompt_includes_categories(self, sample_test_failure):
        """Test prompt includes failure categories."""
        analyzer = LLMFailureAnalyzer()
        prompt = analyzer._build_analysis_prompt(sample_test_failure, {})

        # Should include category list
        assert "category" in prompt.lower()

    def test_build_analysis_prompt_includes_format(self, sample_test_failure):
        """Test prompt includes response format instructions."""
        analyzer = LLMFailureAnalyzer()
        prompt = analyzer._build_analysis_prompt(sample_test_failure, {})

        assert "<root_cause>" in prompt
        assert "<approach>" in prompt
        assert "<confidence>" in prompt
        assert "<category>" in prompt

    def test_build_validation_prompt(self, sample_test_failure):
        """Test validation prompt building."""
        analyzer = LLMFailureAnalyzer()

        original = AgentAnalysis(
            agent_name="agent1",
            root_cause="The authenticate method doesn't set is_authenticated",
            approach="Add self.is_authenticated = True after password verification",
            confidence=0.9,
        )

        prompt = analyzer._build_validation_prompt(sample_test_failure, original)

        assert "Review another AI's analysis" in prompt
        assert original.root_cause in prompt
        assert original.approach in prompt
        assert "0.90" in prompt or "90" in prompt  # Confidence


# ---------------------------------------------------------------------------
# Agent Analysis Tests
# ---------------------------------------------------------------------------


class TestAgentAnalysisRetrieval:
    """Tests for getting analysis from agents."""

    @pytest.mark.asyncio
    async def test_get_agent_analysis_success(self, mock_agent):
        """Test successful agent analysis."""
        mock_agent.generate.return_value = """
<category>assertion_error</category>
<root_cause>The authenticate method doesn't set is_authenticated flag</root_cause>
<approach>Add self.is_authenticated = True in authenticate method</approach>
<confidence>0.85</confidence>
"""

        analyzer = LLMFailureAnalyzer()
        result = await analyzer._get_agent_analysis(mock_agent, "test prompt")

        assert result is not None
        assert result.agent_name == "mock_agent"
        assert "is_authenticated" in result.root_cause
        assert result.confidence == 0.85
        assert result.category == "assertion_error"

    @pytest.mark.asyncio
    async def test_get_agent_analysis_no_root_cause(self, mock_agent):
        """Test handling response without root_cause."""
        mock_agent.generate.return_value = """
<approach>Some approach</approach>
<confidence>0.7</confidence>
"""

        analyzer = LLMFailureAnalyzer()
        result = await analyzer._get_agent_analysis(mock_agent, "test prompt")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_agent_analysis_timeout(self, mock_agent):
        """Test handling timeout."""

        async def slow_response(prompt):
            await asyncio.sleep(10)
            return "response"

        mock_agent.generate.side_effect = slow_response

        config = LLMAnalyzerConfig(agent_timeout=0.1)
        analyzer = LLMFailureAnalyzer(config)

        result = await analyzer._get_agent_analysis(mock_agent, "test prompt")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_agent_analysis_exception(self, mock_agent):
        """Test handling exception from agent."""
        mock_agent.generate.side_effect = RuntimeError("API error")

        analyzer = LLMFailureAnalyzer()
        result = await analyzer._get_agent_analysis(mock_agent, "test prompt")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_agent_analysis_default_approach(self, mock_agent):
        """Test default approach when not provided."""
        mock_agent.generate.return_value = """
<root_cause>Some root cause</root_cause>
<confidence>0.7</confidence>
"""

        analyzer = LLMFailureAnalyzer()
        result = await analyzer._get_agent_analysis(mock_agent, "test prompt")

        assert result is not None
        assert "guidance" in result.approach.lower()


# ---------------------------------------------------------------------------
# Synthesis Tests
# ---------------------------------------------------------------------------


class TestSynthesis:
    """Tests for analysis synthesis."""

    def test_synthesize_empty_analyses(self):
        """Test synthesis with no analyses."""
        analyzer = LLMFailureAnalyzer()
        root_cause, approach, confidence = analyzer._synthesize_analyses([])

        assert root_cause == ""
        assert approach == ""
        assert confidence == 0.0

    def test_synthesize_single_analysis(self):
        """Test synthesis with single analysis."""
        analyzer = LLMFailureAnalyzer()

        analysis = AgentAnalysis(
            agent_name="agent1",
            root_cause="Single root cause",
            approach="Single approach",
            confidence=0.8,
        )

        root_cause, approach, confidence = analyzer._synthesize_analyses([analysis])

        assert root_cause == "Single root cause"
        assert approach == "Single approach"
        assert confidence == 0.8

    def test_synthesize_picks_highest_confidence(self):
        """Test synthesis picks highest confidence analysis."""
        analyzer = LLMFailureAnalyzer()

        analyses = [
            AgentAnalysis(
                agent_name="agent1",
                root_cause="Low confidence cause",
                approach="Low approach",
                confidence=0.5,
            ),
            AgentAnalysis(
                agent_name="agent2",
                root_cause="High confidence cause",
                approach="High approach",
                confidence=0.9,
            ),
        ]

        root_cause, approach, confidence = analyzer._synthesize_analyses(analyses)

        assert root_cause == "High confidence cause"
        assert confidence == 0.9

    def test_synthesize_with_validations(self):
        """Test synthesis with cross-validation data."""
        analyzer = LLMFailureAnalyzer()

        analyses = [
            AgentAnalysis(
                agent_name="agent1",
                root_cause="Cause 1",
                approach="Approach 1",
                confidence=0.7,
            ),
            AgentAnalysis(
                agent_name="agent2",
                root_cause="Cause 2",
                approach="Approach 2",
                confidence=0.7,
            ),
        ]

        # agent1 gets full agreement, agent2 gets disagreement
        validations = {
            "agent1": [("Good analysis", True, 0.9)],
            "agent2": [("Incorrect", False, 0.8)],
        }

        root_cause, approach, confidence = analyzer._synthesize_analyses(analyses, validations)

        # agent1 should win due to validation boost
        assert root_cause == "Cause 1"

    def test_synthesize_combines_approaches(self):
        """Test synthesis combines complementary approaches."""
        config = LLMAnalyzerConfig(synthesis_threshold=0.5)
        analyzer = LLMFailureAnalyzer(config)

        analyses = [
            AgentAnalysis(
                agent_name="agent1",
                root_cause="Primary cause",
                approach="Primary approach with many words that is substantial",
                confidence=0.85,
            ),
            AgentAnalysis(
                agent_name="agent2",
                root_cause="Similar cause",
                approach="Secondary approach that adds additional context and considerations for edge cases",
                confidence=0.80,
            ),
        ]

        root_cause, approach, confidence = analyzer._synthesize_analyses(analyses)

        # Should include both approaches
        assert "Primary approach" in approach
        assert "agent2" in approach or "Secondary" in approach

    def test_synthesize_consensus_lowering(self):
        """Test confidence lowered when consensus required but not reached."""
        config = LLMAnalyzerConfig(require_consensus=True)
        analyzer = LLMFailureAnalyzer(config)

        # Two analyses with very different root causes
        analyses = [
            AgentAnalysis(
                agent_name="agent1",
                root_cause="completely different root cause one",
                approach="Approach 1",
                confidence=0.8,
            ),
            AgentAnalysis(
                agent_name="agent2",
                root_cause="totally unrelated root cause two",
                approach="Approach 2",
                confidence=0.79,
            ),
        ]

        root_cause, approach, confidence = analyzer._synthesize_analyses(analyses)

        # Confidence should be lower due to disagreement
        assert confidence < 0.8


# ---------------------------------------------------------------------------
# Full Analysis Flow Tests
# ---------------------------------------------------------------------------


class TestFullAnalysisFlow:
    """Tests for complete analysis workflow."""

    @pytest.mark.asyncio
    async def test_analyze_no_agents_raises(self, sample_test_failure):
        """Test that analyze raises when no agents available."""
        with patch("aragora.nomic.testfixer.analyzers.llm_analyzer.create_agent") as mock_create:
            mock_create.side_effect = RuntimeError("No API key")

            analyzer = LLMFailureAnalyzer()

            with pytest.raises(RuntimeError, match="No agents"):
                await analyzer.analyze(sample_test_failure, {})

    @pytest.mark.asyncio
    async def test_analyze_returns_fallback_on_no_valid_analyses(self, sample_test_failure):
        """Test fallback when all agents fail."""
        with patch("aragora.nomic.testfixer.analyzers.llm_analyzer.create_agent") as mock_create:
            mock_agent = MagicMock()
            mock_agent.name = "test_agent"
            mock_agent.generate = AsyncMock(return_value="No valid tags")
            mock_create.return_value = mock_agent

            analyzer = LLMFailureAnalyzer()
            root_cause, approach, confidence = await analyzer.analyze(sample_test_failure, {})

            assert "Unable to determine" in root_cause
            assert "Manual investigation" in approach
            assert confidence == 0.3

    @pytest.mark.asyncio
    async def test_analyze_success_single_agent(self, sample_test_failure, sample_code_context):
        """Test successful analysis with single agent."""
        with patch("aragora.nomic.testfixer.analyzers.llm_analyzer.create_agent") as mock_create:
            mock_agent = MagicMock()
            mock_agent.name = "test_agent"
            mock_agent.generate = AsyncMock(
                return_value="""
<category>assertion_error</category>
<root_cause>The authenticate method doesn't set is_authenticated</root_cause>
<approach>Add self.is_authenticated = True after verification</approach>
<confidence>0.85</confidence>
"""
            )
            mock_create.return_value = mock_agent

            config = LLMAnalyzerConfig(agent_types=["test-api"])
            analyzer = LLMFailureAnalyzer(config)

            root_cause, approach, confidence = await analyzer.analyze(
                sample_test_failure, sample_code_context
            )

            assert "is_authenticated" in root_cause
            assert "True" in approach
            assert confidence == 0.85

    @pytest.mark.asyncio
    async def test_analyze_with_cross_validation(self, sample_test_failure, sample_code_context):
        """Test analysis with cross-validation enabled."""
        with patch("aragora.nomic.testfixer.analyzers.llm_analyzer.create_agent") as mock_create:
            agent1 = MagicMock()
            agent1.name = "agent1"
            agent1.generate = AsyncMock()

            agent2 = MagicMock()
            agent2.name = "agent2"
            agent2.generate = AsyncMock()

            # First call is analysis, subsequent are validations
            call_count = [0]

            async def generate_response(prompt):
                call_count[0] += 1
                if "Review another AI" in prompt:
                    return "<agreement>agree</agreement><critique>Looks good</critique><confidence>0.9</confidence>"
                return """
<root_cause>Authentication flag not set</root_cause>
<approach>Set flag in authenticate method</approach>
<confidence>0.85</confidence>
"""

            agent1.generate.side_effect = generate_response
            agent2.generate.side_effect = generate_response

            mock_create.side_effect = [agent1, agent2]

            config = LLMAnalyzerConfig(
                agent_types=["api1", "api2"],
                cross_validate=True,
            )
            analyzer = LLMFailureAnalyzer(config)

            root_cause, approach, confidence = await analyzer.analyze(
                sample_test_failure, sample_code_context
            )

            assert root_cause != ""
            assert approach != ""
            assert confidence > 0

    @pytest.mark.asyncio
    async def test_analyze_skips_cross_validation_single_agent(self, sample_test_failure):
        """Test cross-validation is skipped with single agent."""
        with patch("aragora.nomic.testfixer.analyzers.llm_analyzer.create_agent") as mock_create:
            mock_agent = MagicMock()
            mock_agent.name = "agent"
            mock_agent.generate = AsyncMock(
                return_value="<root_cause>cause</root_cause><approach>fix</approach><confidence>0.8</confidence>"
            )
            mock_create.return_value = mock_agent

            config = LLMAnalyzerConfig(
                agent_types=["test"],
                cross_validate=True,
            )
            analyzer = LLMFailureAnalyzer(config)

            await analyzer.analyze(sample_test_failure, {})

            # Should only be called once (no validation calls)
            assert mock_agent.generate.call_count == 1


# ---------------------------------------------------------------------------
# Edge Cases and Error Handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_ensure_agents_only_once(self):
        """Test agents are only initialized once."""
        with patch("aragora.nomic.testfixer.analyzers.llm_analyzer.create_agent") as mock_create:
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            analyzer = LLMFailureAnalyzer()
            analyzer._ensure_agents()
            analyzer._ensure_agents()

            # Should only create agents once
            assert mock_create.call_count == 2  # 2 default agent types

    def test_empty_code_context(self, sample_test_failure):
        """Test prompt building with empty code context."""
        analyzer = LLMFailureAnalyzer()
        prompt = analyzer._build_analysis_prompt(sample_test_failure, {})

        assert "No code context available" in prompt

    def test_many_involved_files(self, sample_test_failure):
        """Test with many involved files."""
        sample_test_failure.involved_files = [f"file{i}.py" for i in range(20)]

        analyzer = LLMFailureAnalyzer()
        prompt = analyzer._build_analysis_prompt(sample_test_failure, {})

        # Should limit files shown
        assert "file0.py" in prompt
        assert "file9.py" in prompt

    @pytest.mark.asyncio
    async def test_cross_validation_timeout(self, sample_test_failure):
        """Test cross-validation handles timeout."""
        with patch("aragora.nomic.testfixer.analyzers.llm_analyzer.create_agent") as mock_create:
            agent1 = MagicMock()
            agent1.name = "agent1"

            agent2 = MagicMock()
            agent2.name = "agent2"

            async def slow_validate(prompt):
                if "Review" in prompt:
                    await asyncio.sleep(10)
                return "<root_cause>cause</root_cause><approach>fix</approach><confidence>0.8</confidence>"

            agent1.generate = AsyncMock(side_effect=slow_validate)
            agent2.generate = AsyncMock(side_effect=slow_validate)

            mock_create.side_effect = [agent1, agent2]

            config = LLMAnalyzerConfig(
                agent_types=["a1", "a2"],
                agent_timeout=0.1,
            )
            analyzer = LLMFailureAnalyzer(config)

            # Should complete without error despite timeout
            root_cause, approach, confidence = await analyzer.analyze(sample_test_failure, {})

            assert root_cause != ""

    def test_large_stack_trace_truncated(self):
        """Test large stack trace is truncated in prompt."""
        failure = TestFailure(
            test_name="test",
            test_file="test.py",
            error_type="Error",
            error_message="msg",
            stack_trace="X" * 50000,
        )

        analyzer = LLMFailureAnalyzer()
        prompt = analyzer._build_analysis_prompt(failure, {})

        # Stack trace should be truncated
        assert len(prompt) < 100000
        assert "truncated" in prompt
