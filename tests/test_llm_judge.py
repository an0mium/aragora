"""
Tests for the evaluation module (llm_judge).

Tests the LLM-as-Judge evaluation system including:
- EvaluationDimension enum
- Scoring rubrics and weights
- DimensionScore, EvaluationResult, PairwiseResult dataclasses
- JudgeConfig configuration
- LLMJudge core methods (mocked LLM calls)
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.evaluation.llm_judge import (
    DEFAULT_RUBRICS,
    DEFAULT_WEIGHTS,
    WEIGHT_PROFILES,
    DimensionScore,
    EvaluationDimension,
    EvaluationResult,
    EvaluationRubric,
    JudgeConfig,
    LLMJudge,
    PairwiseResult,
)


# =============================================================================
# EvaluationDimension Tests
# =============================================================================


class TestEvaluationDimension:
    """Tests for EvaluationDimension enum."""

    def test_all_dimensions_exist(self):
        """Test that all 8 dimensions are defined."""
        expected = {
            "relevance", "accuracy", "completeness", "clarity",
            "reasoning", "evidence", "creativity", "safety"
        }
        actual = {d.value for d in EvaluationDimension}
        assert actual == expected

    def test_dimension_values(self):
        """Test individual dimension values."""
        assert EvaluationDimension.RELEVANCE.value == "relevance"
        assert EvaluationDimension.ACCURACY.value == "accuracy"
        assert EvaluationDimension.SAFETY.value == "safety"

    def test_dimension_is_string_enum(self):
        """Test that dimensions inherit from str."""
        assert isinstance(EvaluationDimension.RELEVANCE, str)
        assert EvaluationDimension.RELEVANCE == "relevance"


# =============================================================================
# Weight Configuration Tests
# =============================================================================


class TestWeightConfiguration:
    """Tests for weight configuration."""

    def test_default_weights_sum_to_one(self):
        """Test that default weights sum to 1.0."""
        total = sum(DEFAULT_WEIGHTS.values())
        assert total == pytest.approx(1.0)

    def test_default_weights_all_dimensions(self):
        """Test that default weights cover all dimensions."""
        assert len(DEFAULT_WEIGHTS) == len(EvaluationDimension)
        for dim in EvaluationDimension:
            assert dim in DEFAULT_WEIGHTS

    def test_weight_profiles_exist(self):
        """Test that weight profiles are defined."""
        expected_profiles = {
            "factual_qa", "creative_writing", "code_generation",
            "debate", "safety_critical"
        }
        assert set(WEIGHT_PROFILES.keys()) == expected_profiles

    def test_each_profile_sums_to_one(self):
        """Test that each weight profile sums to 1.0."""
        for profile_name, weights in WEIGHT_PROFILES.items():
            total = sum(weights.values())
            assert total == pytest.approx(1.0), f"Profile {profile_name} doesn't sum to 1.0"

    def test_debate_profile_emphasizes_reasoning(self):
        """Test that debate profile weights reasoning highly."""
        debate_weights = WEIGHT_PROFILES["debate"]
        assert debate_weights[EvaluationDimension.REASONING] >= 0.20

    def test_safety_critical_profile_emphasizes_safety(self):
        """Test that safety_critical profile weights safety highly."""
        safety_weights = WEIGHT_PROFILES["safety_critical"]
        assert safety_weights[EvaluationDimension.SAFETY] >= 0.20


# =============================================================================
# EvaluationRubric Tests
# =============================================================================


class TestEvaluationRubric:
    """Tests for EvaluationRubric dataclass."""

    def test_rubric_creation(self):
        """Test creating a rubric."""
        rubric = EvaluationRubric(
            dimension=EvaluationDimension.RELEVANCE,
            description="Test description",
            score_1="Poor",
            score_2="Below Average",
            score_3="Average",
            score_4="Above Average",
            score_5="Excellent",
        )
        assert rubric.dimension == EvaluationDimension.RELEVANCE
        assert rubric.description == "Test description"
        assert rubric.score_5 == "Excellent"

    def test_rubric_to_prompt(self):
        """Test rubric prompt generation."""
        rubric = EvaluationRubric(
            dimension=EvaluationDimension.ACCURACY,
            description="Factual correctness",
            score_1="Many errors",
            score_2="Some errors",
            score_3="Mostly correct",
            score_4="Very accurate",
            score_5="Perfect",
        )
        prompt = rubric.to_prompt()

        assert "ACCURACY" in prompt
        assert "Factual correctness" in prompt
        assert "Score 1" in prompt
        assert "Score 5" in prompt
        assert "Perfect" in prompt

    def test_default_rubrics_exist(self):
        """Test that default rubrics cover all dimensions."""
        assert len(DEFAULT_RUBRICS) == len(EvaluationDimension)
        for dim in EvaluationDimension:
            assert dim in DEFAULT_RUBRICS
            rubric = DEFAULT_RUBRICS[dim]
            assert rubric.dimension == dim


# =============================================================================
# DimensionScore Tests
# =============================================================================


class TestDimensionScore:
    """Tests for DimensionScore dataclass."""

    def test_dimension_score_creation(self):
        """Test creating a dimension score."""
        score = DimensionScore(
            dimension=EvaluationDimension.CLARITY,
            score=4.5,
            confidence=0.9,
            feedback="Very clear response",
        )
        assert score.dimension == EvaluationDimension.CLARITY
        assert score.score == 4.5
        assert score.confidence == 0.9
        assert score.feedback == "Very clear response"
        assert score.examples == []  # default

    def test_dimension_score_with_examples(self):
        """Test dimension score with examples."""
        score = DimensionScore(
            dimension=EvaluationDimension.EVIDENCE,
            score=3.0,
            confidence=0.8,
            feedback="Some evidence provided",
            examples=["Example 1", "Example 2"],
        )
        assert len(score.examples) == 2

    def test_dimension_score_to_dict(self):
        """Test dimension score serialization."""
        score = DimensionScore(
            dimension=EvaluationDimension.REASONING,
            score=4.0,
            confidence=0.85,
            feedback="Good logic",
            examples=["Valid argument"],
        )
        d = score.to_dict()

        assert d["dimension"] == "reasoning"
        assert d["score"] == 4.0
        assert d["confidence"] == 0.85
        assert d["feedback"] == "Good logic"
        assert d["examples"] == ["Valid argument"]


# =============================================================================
# EvaluationResult Tests
# =============================================================================


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_result_creation_defaults(self):
        """Test creating result with defaults."""
        result = EvaluationResult()

        assert result.id  # Should have auto-generated ID
        assert result.response_id == ""
        assert result.dimension_scores == {}
        assert result.overall_score == 0.0
        assert result.strengths == []
        assert result.weaknesses == []

    def test_result_creation_with_values(self):
        """Test creating result with values."""
        result = EvaluationResult(
            response_id="resp-123",
            use_case="debate",
            judge_model="claude-sonnet-4-20250514",
            summary="Good response overall",
            strengths=["Clear", "Well-reasoned"],
            weaknesses=["Lacks evidence"],
        )

        assert result.response_id == "resp-123"
        assert result.use_case == "debate"
        assert len(result.strengths) == 2
        assert len(result.weaknesses) == 1

    def test_calculate_overall_score_empty(self):
        """Test overall score calculation with no dimension scores."""
        result = EvaluationResult()
        score = result.calculate_overall_score()
        assert score == 0.0

    def test_calculate_overall_score_with_scores(self):
        """Test overall score calculation with dimension scores."""
        result = EvaluationResult()
        result.dimension_scores = {
            EvaluationDimension.RELEVANCE: DimensionScore(
                dimension=EvaluationDimension.RELEVANCE,
                score=4.0,
                confidence=0.9,
                feedback="Relevant",
            ),
            EvaluationDimension.ACCURACY: DimensionScore(
                dimension=EvaluationDimension.ACCURACY,
                score=5.0,
                confidence=0.95,
                feedback="Accurate",
            ),
        }

        score = result.calculate_overall_score()

        # Should be weighted average
        assert 4.0 <= score <= 5.0
        assert result.overall_confidence > 0.9

    def test_calculate_overall_score_custom_weights(self):
        """Test overall score with custom weights."""
        result = EvaluationResult()
        result.dimension_scores = {
            EvaluationDimension.RELEVANCE: DimensionScore(
                dimension=EvaluationDimension.RELEVANCE,
                score=5.0,
                confidence=1.0,
                feedback="Perfect",
            ),
            EvaluationDimension.ACCURACY: DimensionScore(
                dimension=EvaluationDimension.ACCURACY,
                score=1.0,
                confidence=1.0,
                feedback="Poor",
            ),
        }

        # Custom weights: all relevance
        custom_weights = {
            EvaluationDimension.RELEVANCE: 1.0,
            EvaluationDimension.ACCURACY: 0.0,
        }

        score = result.calculate_overall_score(custom_weights)
        assert score == 5.0  # Only relevance counts

    def test_result_to_dict(self):
        """Test result serialization."""
        result = EvaluationResult(
            response_id="resp-456",
            use_case="factual_qa",
            summary="Test summary",
        )
        result.dimension_scores = {
            EvaluationDimension.CLARITY: DimensionScore(
                dimension=EvaluationDimension.CLARITY,
                score=4.0,
                confidence=0.8,
                feedback="Clear",
            ),
        }
        result.calculate_overall_score()

        d = result.to_dict()

        assert d["response_id"] == "resp-456"
        assert d["use_case"] == "factual_qa"
        assert "clarity" in d["dimension_scores"]
        assert "timestamp" in d


# =============================================================================
# PairwiseResult Tests
# =============================================================================


class TestPairwiseResult:
    """Tests for PairwiseResult dataclass."""

    def test_pairwise_creation_defaults(self):
        """Test creating pairwise result with defaults."""
        result = PairwiseResult()

        assert result.id  # Auto-generated
        assert result.winner == ""
        assert result.confidence == 0.0
        assert result.dimension_preferences == {}

    def test_pairwise_creation_with_values(self):
        """Test creating pairwise result with values."""
        result = PairwiseResult(
            response_a_id="resp-A",
            response_b_id="resp-B",
            winner="A",
            confidence=0.85,
            dimension_preferences={
                "relevance": "A",
                "accuracy": "B",
                "clarity": "tie",
            },
            explanation="Response A was more relevant",
        )

        assert result.winner == "A"
        assert result.confidence == 0.85
        assert len(result.dimension_preferences) == 3

    def test_pairwise_to_dict(self):
        """Test pairwise result serialization."""
        result = PairwiseResult(
            response_a_id="A1",
            response_b_id="B1",
            winner="B",
            explanation="B was better",
        )

        d = result.to_dict()

        assert d["response_a_id"] == "A1"
        assert d["winner"] == "B"
        assert "timestamp" in d


# =============================================================================
# JudgeConfig Tests
# =============================================================================


class TestJudgeConfig:
    """Tests for JudgeConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration."""
        config = JudgeConfig()

        assert config.model == "claude-sonnet-4-20250514"
        assert config.temperature == 0.0
        assert config.max_tokens == 4000
        assert config.use_case == "default"
        assert config.pass_threshold == 3.5
        assert config.use_multiple_judges is False

    def test_config_custom(self):
        """Test custom configuration."""
        config = JudgeConfig(
            model="gpt-4o",
            use_case="debate",
            pass_threshold=4.0,
            dimensions=[EvaluationDimension.REASONING, EvaluationDimension.EVIDENCE],
        )

        assert config.model == "gpt-4o"
        assert config.use_case == "debate"
        assert config.pass_threshold == 4.0
        assert len(config.dimensions) == 2

    def test_config_custom_weights(self):
        """Test configuration with custom weights."""
        custom = {
            EvaluationDimension.SAFETY: 0.5,
            EvaluationDimension.ACCURACY: 0.5,
        }
        config = JudgeConfig(custom_weights=custom)

        assert config.custom_weights == custom


# =============================================================================
# LLMJudge Core Tests
# =============================================================================


class TestLLMJudge:
    """Tests for LLMJudge class."""

    def test_judge_creation_defaults(self):
        """Test creating judge with defaults."""
        judge = LLMJudge()

        assert judge._config.model == "claude-sonnet-4-20250514"
        assert len(judge._dimensions) == 8
        assert judge._weights == DEFAULT_WEIGHTS

    def test_judge_creation_custom_config(self):
        """Test creating judge with custom config."""
        config = JudgeConfig(
            use_case="debate",
            dimensions=[EvaluationDimension.REASONING],
        )
        judge = LLMJudge(config)

        assert judge._weights == WEIGHT_PROFILES["debate"]
        assert len(judge._dimensions) == 1
        assert EvaluationDimension.REASONING in judge._dimensions

    def test_judge_custom_rubrics(self):
        """Test judge with custom rubrics."""
        custom_rubric = EvaluationRubric(
            dimension=EvaluationDimension.CLARITY,
            description="Custom clarity",
            score_1="Very bad",
            score_2="Bad",
            score_3="Ok",
            score_4="Good",
            score_5="Very good",
        )
        config = JudgeConfig(
            custom_rubrics={EvaluationDimension.CLARITY: custom_rubric}
        )
        judge = LLMJudge(config)

        assert judge._rubrics[EvaluationDimension.CLARITY].description == "Custom clarity"

    def test_judge_uses_weight_profile(self):
        """Test that judge uses correct weight profile."""
        config = JudgeConfig(use_case="safety_critical")
        judge = LLMJudge(config)

        assert judge._weights == WEIGHT_PROFILES["safety_critical"]
        assert judge._weights[EvaluationDimension.SAFETY] >= 0.20


# =============================================================================
# LLMJudge Async Method Tests (Mocked)
# =============================================================================


class TestLLMJudgeAsync:
    """Tests for LLMJudge async methods with mocked LLM calls."""

    @pytest.mark.asyncio
    async def test_evaluate_success(self):
        """Test successful evaluation with mocked LLM."""
        judge = LLMJudge()

        # Mock the LLM call
        mock_response = """
{
    "dimension_scores": {
        "relevance": {"score": 4, "confidence": 0.9, "feedback": "Very relevant"},
        "accuracy": {"score": 5, "confidence": 0.95, "feedback": "Accurate"}
    },
    "summary": "Good response",
    "strengths": ["Clear", "Accurate"],
    "weaknesses": ["Could be more detailed"],
    "suggestions": ["Add more examples"]
}
"""
        with patch.object(judge, "_call_judge", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            with patch.object(judge, "_parse_evaluation") as mock_parse:
                mock_parse.return_value = {
                    EvaluationDimension.RELEVANCE: DimensionScore(
                        dimension=EvaluationDimension.RELEVANCE,
                        score=4.0,
                        confidence=0.9,
                        feedback="Very relevant",
                    ),
                }

                with patch.object(judge, "_extract_feedback") as mock_feedback:
                    mock_feedback.return_value = {
                        "summary": "Good response",
                        "strengths": ["Clear"],
                        "weaknesses": [],
                        "suggestions": [],
                    }

                    result = await judge.evaluate(
                        query="What is Python?",
                        response="Python is a programming language.",
                    )

        assert result.response_id
        assert result.summary == "Good response"
        assert result.overall_score > 0

    @pytest.mark.asyncio
    async def test_evaluate_error_handling(self):
        """Test that evaluation handles errors gracefully."""
        judge = LLMJudge()

        with patch.object(judge, "_call_judge", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = Exception("API Error")

            result = await judge.evaluate(
                query="What is Python?",
                response="Python is great.",
            )

        # Should return partial result with error info
        assert "error" in result.summary.lower() or "Error" in result.summary

    @pytest.mark.asyncio
    async def test_compare_basic(self):
        """Test basic pairwise comparison with mocked LLM."""
        judge = LLMJudge()

        with patch.object(judge, "_call_judge", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = """
{
    "winner": "A",
    "confidence": 0.8,
    "dimension_preferences": {"relevance": "A", "clarity": "B"},
    "explanation": "A was more relevant"
}
"""
            with patch.object(judge, "_parse_comparison") as mock_parse:
                mock_parse.return_value = {
                    "winner": "A",
                    "confidence": 0.8,
                    "dimension_preferences": {"relevance": "A"},
                    "explanation": "A was better",
                }

                result = await judge.compare(
                    query="Explain recursion",
                    response_a="Recursion is when a function calls itself.",
                    response_b="It's complicated.",
                )

        assert result.response_a_id
        assert result.response_b_id


# =============================================================================
# Integration Tests
# =============================================================================


class TestEvaluationIntegration:
    """Integration tests for evaluation module."""

    def test_full_evaluation_flow(self):
        """Test complete evaluation result flow without LLM."""
        # Create dimension scores
        scores = {}
        for dim in EvaluationDimension:
            scores[dim] = DimensionScore(
                dimension=dim,
                score=4.0,
                confidence=0.85,
                feedback=f"Good {dim.value}",
            )

        # Create result
        result = EvaluationResult(
            response_id="test-response",
            dimension_scores=scores,
            use_case="debate",
            summary="Overall good response",
            strengths=["Clear reasoning", "Good evidence"],
            weaknesses=["Could be more creative"],
        )

        # Calculate score
        result.calculate_overall_score(WEIGHT_PROFILES["debate"])

        # Check quality gate
        result.threshold_used = 3.5
        result.passes_threshold = result.overall_score >= result.threshold_used

        assert result.overall_score == 4.0  # All dimensions scored 4.0
        assert result.passes_threshold is True
        assert len(result.to_dict()["dimension_scores"]) == 8

    def test_weight_profile_impact(self):
        """Test that different weight profiles produce different scores."""
        # Score high on creativity, low on accuracy
        scores = {
            EvaluationDimension.RELEVANCE: DimensionScore(
                dimension=EvaluationDimension.RELEVANCE,
                score=3.0,
                confidence=0.8,
                feedback="Ok",
            ),
            EvaluationDimension.ACCURACY: DimensionScore(
                dimension=EvaluationDimension.ACCURACY,
                score=2.0,
                confidence=0.8,
                feedback="Some errors",
            ),
            EvaluationDimension.CREATIVITY: DimensionScore(
                dimension=EvaluationDimension.CREATIVITY,
                score=5.0,
                confidence=0.9,
                feedback="Very creative",
            ),
        }

        result = EvaluationResult(dimension_scores=scores)

        # With creative_writing profile (creativity=0.30)
        creative_score = result.calculate_overall_score(WEIGHT_PROFILES["creative_writing"])

        result_factual = EvaluationResult(dimension_scores=scores.copy())

        # With factual_qa profile (creativity=0.00, accuracy=0.30)
        factual_score = result_factual.calculate_overall_score(WEIGHT_PROFILES["factual_qa"])

        # Creative profile should rate this higher
        assert creative_score > factual_score
