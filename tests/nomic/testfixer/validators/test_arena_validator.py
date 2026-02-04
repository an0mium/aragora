"""
Comprehensive tests for ArenaValidator.

Tests the Arena-based fix validation including:
- Configuration defaults and custom settings
- Validation prompt building
- Response parsing (verdict, confidence, reasoning)
- Agent validation retrieval
- Consensus calculation
- Full validation flow
- Arena debate integration
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from aragora.nomic.testfixer.validators.arena_validator import (
    ArenaValidator,
    ArenaValidatorConfig,
    ValidationResult,
)
from aragora.nomic.testfixer.runner import TestFailure
from aragora.nomic.testfixer.analyzer import FailureCategory


# ---------------------------------------------------------------------------
# Mock Objects for Testing
# ---------------------------------------------------------------------------


@dataclass
class MockPatch:
    """Mock patch for testing."""

    file_path: str
    original_content: str = ""
    patched_content: str = ""


@dataclass
class MockProposal:
    """Mock proposal for testing."""

    id: str = "test-proposal-001"
    description: str = "Fix authentication flag"
    post_debate_confidence: float = 0.8
    patches: list = field(default_factory=list)


@dataclass
class MockAnalysis:
    """Mock analysis for testing."""

    failure: TestFailure = None
    root_cause: str = "Missing authentication flag"
    fix_target: FailureCategory = FailureCategory.ASSERTION_ERROR
    category: FailureCategory = FailureCategory.ASSERTION_ERROR
    root_cause_file: str = "src/auth.py"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_test_failure() -> TestFailure:
    """Create a sample test failure."""
    return TestFailure(
        test_name="test_auth",
        test_file="tests/test_auth.py",
        error_type="AssertionError",
        error_message="Expected True but got False",
        stack_trace="File test_auth.py line 10\nAssertionError",
        line_number=10,
    )


@pytest.fixture
def sample_proposal() -> MockProposal:
    """Create a sample proposal."""
    return MockProposal(
        id="prop-001",
        description="Add is_authenticated = True after password check",
        post_debate_confidence=0.85,
        patches=[
            MockPatch(
                file_path="src/auth.py",
                original_content="def authenticate(self):\n    return True",
                patched_content="def authenticate(self):\n    self.is_authenticated = True\n    return True",
            )
        ],
    )


@pytest.fixture
def sample_analysis(sample_test_failure) -> MockAnalysis:
    """Create a sample analysis."""
    return MockAnalysis(
        failure=sample_test_failure,
        root_cause="authenticate() doesn't set is_authenticated flag",
        fix_target=FailureCategory.IMPLEMENTATION_BUG,
    )


@pytest.fixture
def mock_agent() -> MagicMock:
    """Create a mock agent."""
    agent = MagicMock()
    agent.name = "mock_validator"
    agent.generate = AsyncMock()
    return agent


# ---------------------------------------------------------------------------
# ValidationResult Tests
# ---------------------------------------------------------------------------


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_basic_creation(self):
        """Test creating a ValidationResult."""
        result = ValidationResult(
            is_valid=True,
            confidence=0.85,
            consensus_reached=True,
            agreement_ratio=0.9,
        )
        assert result.is_valid is True
        assert result.confidence == 0.85
        assert result.consensus_reached is True
        assert result.agreement_ratio == 0.9

    def test_default_fields(self):
        """Test default field values."""
        result = ValidationResult(
            is_valid=False,
            confidence=0.5,
            consensus_reached=False,
            agreement_ratio=0.5,
        )
        assert result.supporting_agents == []
        assert result.dissenting_agents == []
        assert result.critiques == []
        assert result.improvements == []
        assert result.debate_rounds == 0
        assert result.raw_responses == {}

    def test_summary_valid(self):
        """Test summary for valid result."""
        result = ValidationResult(
            is_valid=True,
            confidence=0.9,
            consensus_reached=True,
            agreement_ratio=0.8,
            supporting_agents=["agent1", "agent2"],
            dissenting_agents=["agent3"],
        )
        summary = result.summary()

        assert "VALID" in summary
        assert "90%" in summary
        assert "consensus" in summary
        assert "2 support" in summary
        assert "1 dissent" in summary

    def test_summary_invalid(self):
        """Test summary for invalid result."""
        result = ValidationResult(
            is_valid=False,
            confidence=0.3,
            consensus_reached=False,
            agreement_ratio=0.4,
            supporting_agents=["agent1"],
            dissenting_agents=["agent2", "agent3"],
        )
        summary = result.summary()

        assert "INVALID" in summary
        assert "no consensus" in summary


# ---------------------------------------------------------------------------
# ArenaValidatorConfig Tests
# ---------------------------------------------------------------------------


class TestArenaValidatorConfig:
    """Tests for ArenaValidatorConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ArenaValidatorConfig()

        assert "anthropic-api" in config.agent_types
        assert "openai-api" in config.agent_types
        assert "gemini-api" in config.agent_types
        assert config.debate_rounds == 2
        assert config.consensus_threshold == 0.7
        assert config.min_confidence_to_pass == 0.6
        assert config.require_consensus is False
        assert config.agent_timeout == 60.0
        assert config.debate_timeout == 180.0
        assert config.max_context_chars == 50_000

    def test_custom_values(self):
        """Test custom configuration."""
        config = ArenaValidatorConfig(
            agent_types=["anthropic-api", "openai-api"],
            debate_rounds=3,
            consensus_threshold=0.8,
            min_confidence_to_pass=0.7,
            require_consensus=True,
            agent_timeout=120.0,
        )

        assert len(config.agent_types) == 2
        assert config.debate_rounds == 3
        assert config.consensus_threshold == 0.8
        assert config.require_consensus is True


# ---------------------------------------------------------------------------
# ArenaValidator Utility Methods Tests
# ---------------------------------------------------------------------------


class TestArenaValidatorUtilities:
    """Tests for utility methods."""

    def test_truncate_short_text(self):
        """Test truncate with short text."""
        validator = ArenaValidator()
        result = validator._truncate("short", 100)
        assert result == "short"

    def test_truncate_long_text(self):
        """Test truncate with long text."""
        validator = ArenaValidator()
        text = "X" * 1000
        result = validator._truncate(text, 500)

        assert len(result) < len(text)
        assert "truncated" in result


class TestResponseParsing:
    """Tests for response parsing."""

    def test_parse_approve_verdict(self):
        """Test parsing APPROVE verdict."""
        validator = ArenaValidator()
        response = """
VERDICT: APPROVE
CONFIDENCE: 0.9
REASONING: The fix correctly addresses the issue.
CONCERNS: None
IMPROVEMENTS: None
"""
        result = validator._parse_validation_response(response, "test_agent")

        assert result["approves"] is True
        assert result["confidence"] == 0.9
        assert "correctly addresses" in result["reasoning"]

    def test_parse_reject_verdict(self):
        """Test parsing REJECT verdict."""
        validator = ArenaValidator()
        response = """
VERDICT: REJECT
CONFIDENCE: 0.8
REASONING: The fix doesn't handle edge cases.
CONCERNS: Missing null check.
IMPROVEMENTS: Add null validation.
"""
        result = validator._parse_validation_response(response, "test_agent")

        assert result["approves"] is False
        assert result["confidence"] == 0.8

    def test_parse_implicit_approve(self):
        """Test parsing implicit APPROVE (no VERDICT: prefix)."""
        validator = ArenaValidator()
        response = "APPROVE - this looks good. CONFIDENCE: 0.85"

        result = validator._parse_validation_response(response, "agent")

        assert result["approves"] is True

    def test_parse_concerns(self):
        """Test parsing concerns."""
        validator = ArenaValidator()
        response = """
VERDICT: APPROVE
CONFIDENCE: 0.7
REASONING: Looks okay.
CONCERNS: Edge case with empty strings not handled.
IMPROVEMENTS: None
"""
        result = validator._parse_validation_response(response, "agent")

        assert len(result["concerns"]) > 0
        assert "empty strings" in result["concerns"][0]

    def test_parse_improvements(self):
        """Test parsing improvements."""
        validator = ArenaValidator()
        response = """
VERDICT: APPROVE
CONFIDENCE: 0.7
REASONING: Works but could be better.
CONCERNS: None
IMPROVEMENTS: Add logging for debugging.
"""
        result = validator._parse_validation_response(response, "agent")

        assert len(result["improvements"]) > 0
        assert "logging" in result["improvements"][0]

    def test_parse_default_confidence(self):
        """Test default confidence when not specified."""
        validator = ArenaValidator()
        response = "VERDICT: APPROVE"

        result = validator._parse_validation_response(response, "agent")

        assert result["confidence"] == 0.5  # Default

    def test_parse_concerns_none_skipped(self):
        """Test that 'None' concerns are skipped."""
        validator = ArenaValidator()
        response = """
VERDICT: APPROVE
CONCERNS: none
IMPROVEMENTS: n/a
"""
        result = validator._parse_validation_response(response, "agent")

        assert len(result["concerns"]) == 0
        assert len(result["improvements"]) == 0


# ---------------------------------------------------------------------------
# Prompt Building Tests
# ---------------------------------------------------------------------------


class TestPromptBuilding:
    """Tests for validation prompt building."""

    def test_build_validation_prompt(self, sample_proposal, sample_analysis):
        """Test basic validation prompt building."""
        validator = ArenaValidator()
        prompt = validator._build_validation_prompt(sample_proposal, sample_analysis)

        # Should include failure info
        assert "test_auth" in prompt
        assert "AssertionError" in prompt

        # Should include root cause
        assert "is_authenticated" in prompt.lower() or sample_analysis.root_cause in prompt

        # Should include proposal info
        assert sample_proposal.description in prompt
        assert "85%" in prompt or "0.85" in prompt

        # Should include patches
        assert "src/auth.py" in prompt

    def test_prompt_includes_instructions(self, sample_proposal, sample_analysis):
        """Test prompt includes validation instructions."""
        validator = ArenaValidator()
        prompt = validator._build_validation_prompt(sample_proposal, sample_analysis)

        assert "VERDICT" in prompt
        assert "APPROVE" in prompt
        assert "REJECT" in prompt
        assert "CONFIDENCE" in prompt
        assert "REASONING" in prompt


# ---------------------------------------------------------------------------
# Agent Validation Tests
# ---------------------------------------------------------------------------


class TestAgentValidation:
    """Tests for getting validation from agents."""

    @pytest.mark.asyncio
    async def test_get_agent_validation_success(self, mock_agent):
        """Test successful validation retrieval."""
        mock_agent.generate.return_value = """
VERDICT: APPROVE
CONFIDENCE: 0.85
REASONING: The fix is correct.
"""
        validator = ArenaValidator()
        result = await validator._get_agent_validation(mock_agent, "test prompt")

        assert result is not None
        assert result["approves"] is True
        assert result["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_get_agent_validation_timeout(self, mock_agent):
        """Test handling timeout."""

        async def slow_response(prompt):
            await asyncio.sleep(10)
            return "response"

        mock_agent.generate.side_effect = slow_response

        config = ArenaValidatorConfig(agent_timeout=0.1)
        validator = ArenaValidator(config)

        result = await validator._get_agent_validation(mock_agent, "test prompt")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_agent_validation_exception(self, mock_agent):
        """Test handling exception."""
        mock_agent.generate.side_effect = Exception("API error")

        validator = ArenaValidator()
        result = await validator._get_agent_validation(mock_agent, "test prompt")

        assert result is None


# ---------------------------------------------------------------------------
# Full Validation Flow Tests
# ---------------------------------------------------------------------------


class TestValidationFlow:
    """Tests for complete validation workflow."""

    @pytest.mark.asyncio
    async def test_validate_no_agents_raises(self, sample_proposal, sample_analysis):
        """Test validation raises when no agents available."""
        with patch(
            "aragora.nomic.testfixer.validators.arena_validator.create_agent"
        ) as mock_create:
            mock_create.side_effect = Exception("No API key")

            validator = ArenaValidator()

            with pytest.raises(RuntimeError, match="No agents"):
                await validator.validate(sample_proposal, sample_analysis)

    @pytest.mark.asyncio
    async def test_validate_no_responses(self, sample_proposal, sample_analysis):
        """Test validation when all agents fail."""
        with patch(
            "aragora.nomic.testfixer.validators.arena_validator.create_agent"
        ) as mock_create:
            mock_agent = MagicMock()
            mock_agent.name = "agent"
            mock_agent.generate = AsyncMock(side_effect=Exception("Failed"))
            mock_create.return_value = mock_agent

            config = ArenaValidatorConfig(agent_types=["test"])
            validator = ArenaValidator(config)

            result = await validator.validate(sample_proposal, sample_analysis)

            assert result.is_valid is False
            assert result.confidence == 0.0
            assert "No agents provided valid responses" in result.critiques

    @pytest.mark.asyncio
    async def test_validate_unanimous_approval(self, sample_proposal, sample_analysis):
        """Test validation with unanimous approval."""
        with patch(
            "aragora.nomic.testfixer.validators.arena_validator.create_agent"
        ) as mock_create:
            agent1 = MagicMock()
            agent1.name = "agent1"
            agent1.generate = AsyncMock(
                return_value="VERDICT: APPROVE\nCONFIDENCE: 0.9\nREASONING: Good fix"
            )

            agent2 = MagicMock()
            agent2.name = "agent2"
            agent2.generate = AsyncMock(
                return_value="VERDICT: APPROVE\nCONFIDENCE: 0.85\nREASONING: Correct"
            )

            mock_create.side_effect = [agent1, agent2]

            config = ArenaValidatorConfig(agent_types=["a1", "a2"])
            validator = ArenaValidator(config)

            result = await validator.validate(sample_proposal, sample_analysis)

            assert result.is_valid is True
            assert result.consensus_reached is True
            assert result.agreement_ratio == 1.0
            assert len(result.supporting_agents) == 2
            assert len(result.dissenting_agents) == 0

    @pytest.mark.asyncio
    async def test_validate_unanimous_rejection(self, sample_proposal, sample_analysis):
        """Test validation with unanimous rejection."""
        with patch(
            "aragora.nomic.testfixer.validators.arena_validator.create_agent"
        ) as mock_create:
            agent1 = MagicMock()
            agent1.name = "agent1"
            agent1.generate = AsyncMock(
                return_value="VERDICT: REJECT\nCONFIDENCE: 0.8\nREASONING: Wrong approach"
            )

            agent2 = MagicMock()
            agent2.name = "agent2"
            agent2.generate = AsyncMock(
                return_value="VERDICT: REJECT\nCONFIDENCE: 0.75\nREASONING: Broken"
            )

            mock_create.side_effect = [agent1, agent2]

            config = ArenaValidatorConfig(agent_types=["a1", "a2"])
            validator = ArenaValidator(config)

            result = await validator.validate(sample_proposal, sample_analysis)

            assert result.is_valid is False
            assert result.consensus_reached is True
            assert result.agreement_ratio == 0.0
            assert len(result.supporting_agents) == 0
            assert len(result.dissenting_agents) == 2

    @pytest.mark.asyncio
    async def test_validate_mixed_verdicts(self, sample_proposal, sample_analysis):
        """Test validation with mixed verdicts."""
        with patch(
            "aragora.nomic.testfixer.validators.arena_validator.create_agent"
        ) as mock_create:
            agent1 = MagicMock()
            agent1.name = "agent1"
            agent1.generate = AsyncMock(return_value="VERDICT: APPROVE\nCONFIDENCE: 0.7")

            agent2 = MagicMock()
            agent2.name = "agent2"
            agent2.generate = AsyncMock(return_value="VERDICT: REJECT\nCONFIDENCE: 0.6")

            mock_create.side_effect = [agent1, agent2]

            config = ArenaValidatorConfig(agent_types=["a1", "a2"])
            validator = ArenaValidator(config)

            result = await validator.validate(sample_proposal, sample_analysis)

            assert result.agreement_ratio == 0.5
            assert result.consensus_reached is False

    @pytest.mark.asyncio
    async def test_validate_require_consensus(self, sample_proposal, sample_analysis):
        """Test validation fails without consensus when required."""
        with patch(
            "aragora.nomic.testfixer.validators.arena_validator.create_agent"
        ) as mock_create:
            agent1 = MagicMock()
            agent1.name = "agent1"
            agent1.generate = AsyncMock(return_value="VERDICT: APPROVE\nCONFIDENCE: 0.8")

            agent2 = MagicMock()
            agent2.name = "agent2"
            agent2.generate = AsyncMock(return_value="VERDICT: REJECT\nCONFIDENCE: 0.8")

            mock_create.side_effect = [agent1, agent2]

            config = ArenaValidatorConfig(
                agent_types=["a1", "a2"],
                require_consensus=True,
            )
            validator = ArenaValidator(config)

            result = await validator.validate(sample_proposal, sample_analysis)

            # Even though more approvals, require_consensus makes it invalid
            assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_validate_confidence_boost_on_consensus(self, sample_proposal, sample_analysis):
        """Test confidence is boosted when consensus reached."""
        with patch(
            "aragora.nomic.testfixer.validators.arena_validator.create_agent"
        ) as mock_create:
            mock_agent = MagicMock()
            mock_agent.name = "agent"
            mock_agent.generate = AsyncMock(return_value="VERDICT: APPROVE\nCONFIDENCE: 0.8")
            mock_create.return_value = mock_agent

            config = ArenaValidatorConfig(agent_types=["test"])
            validator = ArenaValidator(config)

            result = await validator.validate(sample_proposal, sample_analysis)

            # With single agent, consensus_reached should be True
            assert result.consensus_reached is True
            # Confidence should be boosted
            assert result.confidence > 0.8


# ---------------------------------------------------------------------------
# Arena Debate Integration Tests
# ---------------------------------------------------------------------------


class TestArenaDebateIntegration:
    """Tests for validate_with_debate method."""

    @pytest.mark.asyncio
    async def test_validate_with_debate_timeout(self, sample_proposal, sample_analysis):
        """Test arena debate timeout handling."""
        with patch(
            "aragora.nomic.testfixer.validators.arena_validator.create_agent"
        ) as mock_create:
            mock_agent = MagicMock()
            mock_agent.name = "agent"
            mock_agent.generate = AsyncMock(return_value="VERDICT: APPROVE\nCONFIDENCE: 0.8")
            mock_create.return_value = mock_agent

            with patch("aragora.nomic.testfixer.validators.arena_validator.Arena") as mock_arena:

                async def slow_run():
                    await asyncio.sleep(10)

                arena_instance = MagicMock()
                arena_instance.run = AsyncMock(side_effect=slow_run)
                mock_arena.return_value = arena_instance

                config = ArenaValidatorConfig(
                    agent_types=["test"],
                    debate_timeout=0.1,
                )
                validator = ArenaValidator(config)

                result = await validator.validate_with_debate(sample_proposal, sample_analysis)

                assert result.is_valid is False
                assert "timed out" in result.critiques[0]

    @pytest.mark.asyncio
    async def test_validate_with_debate_fallback_on_error(self, sample_proposal, sample_analysis):
        """Test fallback to simple validation on Arena error."""
        with patch(
            "aragora.nomic.testfixer.validators.arena_validator.create_agent"
        ) as mock_create:
            mock_agent = MagicMock()
            mock_agent.name = "agent"
            mock_agent.generate = AsyncMock(return_value="VERDICT: APPROVE\nCONFIDENCE: 0.85")
            mock_create.return_value = mock_agent

            with patch("aragora.nomic.testfixer.validators.arena_validator.Arena") as mock_arena:
                mock_arena.side_effect = Exception("Arena initialization failed")

                config = ArenaValidatorConfig(agent_types=["test"])
                validator = ArenaValidator(config)

                # Should fall back to simple validate()
                result = await validator.validate_with_debate(sample_proposal, sample_analysis)

                # Should still get a result from fallback
                assert result.is_valid is True
                assert result.confidence == pytest.approx(0.85 * 1.1, rel=0.1)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_debate_timeout_returns_result(self, sample_proposal, sample_analysis):
        """Test that debate timeout returns empty result."""
        with patch(
            "aragora.nomic.testfixer.validators.arena_validator.create_agent"
        ) as mock_create:

            async def slow_response(prompt):
                await asyncio.sleep(10)
                return "response"

            mock_agent = MagicMock()
            mock_agent.name = "agent"
            mock_agent.generate = AsyncMock(side_effect=slow_response)
            mock_create.return_value = mock_agent

            config = ArenaValidatorConfig(
                agent_types=["test"],
                debate_timeout=0.1,
            )
            validator = ArenaValidator(config)

            result = await validator.validate(sample_proposal, sample_analysis)

            # Should return invalid result after timeout
            assert result.is_valid is False

    def test_ensure_agents_only_once(self):
        """Test agents only initialized once."""
        with patch(
            "aragora.nomic.testfixer.validators.arena_validator.create_agent"
        ) as mock_create:
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            config = ArenaValidatorConfig(agent_types=["test"])
            validator = ArenaValidator(config)

            validator._ensure_agents()
            validator._ensure_agents()

            assert mock_create.call_count == 1
