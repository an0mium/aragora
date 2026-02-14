"""Tests for ThinkPRMVerifier."""

import pytest
import asyncio
from aragora.verification.think_prm import (
    ThinkPRMVerifier,
    ThinkPRMConfig,
    StepVerdict,
    StepVerification,
    ProcessVerificationResult,
    create_think_prm_verifier,
)


# Mock query function for testing
async def mock_query_fn(agent_id: str, prompt: str, max_tokens: int = 1000) -> str:
    """Mock LLM query function that returns verification responses."""
    # Check only the step content (between CURRENT STEP and CLAIMED DEPENDENCIES)
    # to avoid matching template text like "factual errors" in the prompt itself
    step_section = ""
    if "CURRENT STEP TO VERIFY:" in prompt:
        after = prompt.split("CURRENT STEP TO VERIFY:", 1)[1]
        if "CLAIMED DEPENDENCIES:" in after:
            step_section = after.split("CLAIMED DEPENDENCIES:", 1)[0]
        else:
            step_section = after
    if "error" in step_section.lower() or "wrong" in step_section.lower():
        return """VERDICT: INCORRECT
CONFIDENCE: 0.85
REASONING: The step contains logical errors
SUGGESTED_FIX: Review the logical chain"""
    else:
        return """VERDICT: CORRECT
CONFIDENCE: 0.9
REASONING: The reasoning is valid and well-supported
SUGGESTED_FIX: None"""


class TestThinkPRMVerifier:
    """Test suite for ThinkPRMVerifier."""

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        verifier = ThinkPRMVerifier()
        assert verifier.config.verifier_agent_id == "claude"
        assert verifier.config.max_context_chars == 2000

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = ThinkPRMConfig(
            verifier_agent_id="custom",
            max_context_chars=1000,
        )
        verifier = ThinkPRMVerifier(config)
        assert verifier.config.verifier_agent_id == "custom"

    @pytest.mark.asyncio
    async def test_verify_step_correct(self) -> None:
        """Test verification of a correct step."""
        verifier = ThinkPRMVerifier()

        result = await verifier.verify_step(
            step_content="The temperature rose because CO2 levels increased.",
            round_number=1,
            agent_id="agent1",
            prior_context="We are discussing climate change.",
            dependencies=["CO2 causes warming"],
            query_fn=mock_query_fn,
        )

        assert result.verdict == StepVerdict.CORRECT
        assert result.confidence > 0.5
        assert result.round_number == 1
        assert result.agent_id == "agent1"

    @pytest.mark.asyncio
    async def test_verify_step_incorrect(self) -> None:
        """Test verification of an incorrect step."""
        verifier = ThinkPRMVerifier()

        result = await verifier.verify_step(
            step_content="This contains an error in reasoning.",
            round_number=2,
            agent_id="agent2",
            prior_context="Prior context.",
            dependencies=[],
            query_fn=mock_query_fn,
        )

        assert result.verdict == StepVerdict.INCORRECT
        assert result.suggested_fix is not None

    @pytest.mark.asyncio
    async def test_verify_debate_process(self) -> None:
        """Test full debate process verification."""
        verifier = ThinkPRMVerifier()

        debate_rounds = [
            {
                "debate_id": "test-debate",
                "contributions": [
                    {"agent_id": "agent1", "content": "First valid point.", "dependencies": []},
                ],
            },
            {
                "debate_id": "test-debate",
                "contributions": [
                    {"agent_id": "agent2", "content": "Second valid point.", "dependencies": []},
                ],
            },
        ]

        result = await verifier.verify_debate_process(
            debate_rounds=debate_rounds,
            query_fn=mock_query_fn,
        )

        assert result.debate_id == "test-debate"
        assert result.total_steps == 2
        assert result.overall_score >= 0.0

    @pytest.mark.asyncio
    async def test_critical_error_detection(self) -> None:
        """Test that late-stage errors are flagged as critical."""
        config = ThinkPRMConfig(critical_round_threshold=0.6)
        verifier = ThinkPRMVerifier(config)

        # 5 rounds, last 2 have errors (>60% threshold)
        debate_rounds = [
            {"debate_id": "test", "contributions": [{"agent_id": "a1", "content": "Round 1 ok"}]},
            {"debate_id": "test", "contributions": [{"agent_id": "a1", "content": "Round 2 ok"}]},
            {"debate_id": "test", "contributions": [{"agent_id": "a1", "content": "Round 3 ok"}]},
            {
                "debate_id": "test",
                "contributions": [{"agent_id": "a1", "content": "Round 4 has error"}],
            },
            {
                "debate_id": "test",
                "contributions": [{"agent_id": "a1", "content": "Round 5 has error"}],
            },
        ]

        result = await verifier.verify_debate_process(
            debate_rounds=debate_rounds,
            query_fn=mock_query_fn,
        )

        # Rounds 4 and 5 are >= 60% threshold (3/5 = 60%)
        # With our mock, "error" triggers INCORRECT
        assert len(result.critical_errors) >= 1

    def test_parse_verification_response(self) -> None:
        """Test response parsing."""
        verifier = ThinkPRMVerifier()

        response = """VERDICT: CORRECT
CONFIDENCE: 0.85
REASONING: The logic is sound
SUGGESTED_FIX: None"""

        result = verifier._parse_verification_response(
            response=response,
            step_content="Test content",
            round_number=1,
            agent_id="agent1",
        )

        assert result.verdict == StepVerdict.CORRECT
        assert result.confidence == 0.85
        assert "logic is sound" in result.reasoning

    def test_parse_needs_revision(self) -> None:
        """Test parsing NEEDS_REVISION verdict."""
        verifier = ThinkPRMVerifier()

        response = """VERDICT: NEEDS_REVISION
CONFIDENCE: 0.6
REASONING: Some parts need clarification
SUGGESTED_FIX: Add more evidence"""

        result = verifier._parse_verification_response(
            response=response,
            step_content="Test",
            round_number=1,
            agent_id="a1",
        )

        assert result.verdict == StepVerdict.NEEDS_REVISION
        assert result.suggested_fix == "Add more evidence"

    def test_cache_functionality(self) -> None:
        """Test that caching works."""
        verifier = ThinkPRMVerifier()

        # Verify caching config
        assert verifier.config.cache_verifications is True

        # Add to cache manually
        verifier._cache["test_key"] = StepVerification(
            step_id="cached",
            round_number=1,
            agent_id="a1",
            content_summary="cached content",
            verdict=StepVerdict.CORRECT,
            confidence=0.9,
            reasoning="cached",
        )

        # Clear cache
        verifier.clear_cache()
        assert len(verifier._cache) == 0

    def test_get_metrics_empty(self) -> None:
        """Test metrics when no verifications done."""
        verifier = ThinkPRMVerifier()
        metrics = verifier.get_metrics()

        assert metrics["total_verifications"] == 0
        assert metrics["avg_confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_get_metrics_with_data(self) -> None:
        """Test metrics after verifications."""
        verifier = ThinkPRMVerifier()

        # Do some verifications
        await verifier.verify_step(
            step_content="Test 1",
            round_number=1,
            agent_id="a1",
            prior_context="",
            dependencies=[],
            query_fn=mock_query_fn,
        )
        await verifier.verify_step(
            step_content="Test 2",
            round_number=2,
            agent_id="a2",
            prior_context="",
            dependencies=[],
            query_fn=mock_query_fn,
        )

        metrics = verifier.get_metrics()

        assert metrics["total_verifications"] == 2
        assert metrics["avg_confidence"] > 0

    def test_reset(self) -> None:
        """Test reset clears state."""
        verifier = ThinkPRMVerifier()

        # Add some state
        verifier._cache["key"] = None
        verifier._verification_history.append(None)

        verifier.reset()

        assert len(verifier._cache) == 0
        assert len(verifier._verification_history) == 0


class TestCreateThinkPRMVerifier:
    """Test the factory function."""

    def test_creates_with_defaults(self) -> None:
        """Test factory creates verifier with defaults."""
        verifier = create_think_prm_verifier()
        assert isinstance(verifier, ThinkPRMVerifier)

    def test_creates_with_custom_config(self) -> None:
        """Test factory accepts custom configuration."""
        verifier = create_think_prm_verifier(
            verifier_agent_id="custom",
            parallel=False,
        )
        assert verifier.config.verifier_agent_id == "custom"
        assert verifier.config.parallel_verification is False
