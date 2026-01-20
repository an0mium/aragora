"""Tests for LLM-as-Judge evaluation system."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

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


class TestEvaluationDimension:
    """Tests for EvaluationDimension enum."""

    def test_all_dimensions_exist(self):
        """All 8 evaluation dimensions exist."""
        assert EvaluationDimension.RELEVANCE
        assert EvaluationDimension.ACCURACY
        assert EvaluationDimension.COMPLETENESS
        assert EvaluationDimension.CLARITY
        assert EvaluationDimension.REASONING
        assert EvaluationDimension.EVIDENCE
        assert EvaluationDimension.CREATIVITY
        assert EvaluationDimension.SAFETY

    def test_dimension_count(self):
        """Exactly 8 dimensions defined."""
        assert len(EvaluationDimension) == 8

    def test_dimension_values(self):
        """Dimensions have lowercase string values."""
        assert EvaluationDimension.RELEVANCE.value == "relevance"
        assert EvaluationDimension.ACCURACY.value == "accuracy"
        assert EvaluationDimension.SAFETY.value == "safety"


class TestDefaultWeights:
    """Tests for DEFAULT_WEIGHTS configuration."""

    def test_weights_sum_to_one(self):
        """Default weights sum to 1.0."""
        total = sum(DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_all_dimensions_have_weights(self):
        """All dimensions have default weights."""
        for dim in EvaluationDimension:
            assert dim in DEFAULT_WEIGHTS

    def test_no_negative_weights(self):
        """All weights are non-negative."""
        for weight in DEFAULT_WEIGHTS.values():
            assert weight >= 0


class TestWeightProfiles:
    """Tests for use case specific weight profiles."""

    def test_profiles_exist(self):
        """Expected weight profiles exist."""
        assert "factual_qa" in WEIGHT_PROFILES
        assert "creative_writing" in WEIGHT_PROFILES
        assert "code_generation" in WEIGHT_PROFILES
        assert "debate" in WEIGHT_PROFILES
        assert "safety_critical" in WEIGHT_PROFILES

    def test_profiles_sum_to_one(self):
        """Each profile's weights sum to 1.0."""
        for profile_name, weights in WEIGHT_PROFILES.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.001, f"{profile_name} weights sum to {total}"

    def test_factual_qa_emphasizes_accuracy(self):
        """Factual QA profile emphasizes accuracy."""
        weights = WEIGHT_PROFILES["factual_qa"]
        assert weights[EvaluationDimension.ACCURACY] > weights[EvaluationDimension.CREATIVITY]

    def test_creative_writing_emphasizes_creativity(self):
        """Creative writing profile emphasizes creativity."""
        weights = WEIGHT_PROFILES["creative_writing"]
        assert weights[EvaluationDimension.CREATIVITY] > weights[EvaluationDimension.ACCURACY]

    def test_safety_critical_emphasizes_safety(self):
        """Safety critical profile emphasizes safety."""
        weights = WEIGHT_PROFILES["safety_critical"]
        assert weights[EvaluationDimension.SAFETY] > DEFAULT_WEIGHTS[EvaluationDimension.SAFETY]


class TestEvaluationRubric:
    """Tests for EvaluationRubric dataclass."""

    def test_create_rubric(self):
        """Rubric can be created with all fields."""
        rubric = EvaluationRubric(
            dimension=EvaluationDimension.CLARITY,
            description="Test description",
            score_1="Poor",
            score_2="Below Average",
            score_3="Average",
            score_4="Above Average",
            score_5="Excellent",
        )
        assert rubric.dimension == EvaluationDimension.CLARITY
        assert rubric.description == "Test description"
        assert rubric.score_1 == "Poor"
        assert rubric.score_5 == "Excellent"

    def test_to_prompt(self):
        """to_prompt returns formatted rubric text."""
        rubric = EvaluationRubric(
            dimension=EvaluationDimension.RELEVANCE,
            description="How relevant is it?",
            score_1="Not at all",
            score_2="Slightly",
            score_3="Moderately",
            score_4="Very",
            score_5="Perfectly",
        )
        prompt = rubric.to_prompt()
        assert "RELEVANCE" in prompt
        assert "How relevant is it?" in prompt
        assert "Score 1" in prompt
        assert "Score 5" in prompt
        assert "Perfectly" in prompt

    def test_default_rubrics_complete(self):
        """All dimensions have default rubrics."""
        for dim in EvaluationDimension:
            assert dim in DEFAULT_RUBRICS
            rubric = DEFAULT_RUBRICS[dim]
            assert rubric.dimension == dim
            assert rubric.description
            assert rubric.score_1
            assert rubric.score_5


class TestDimensionScore:
    """Tests for DimensionScore dataclass."""

    def test_create_score(self):
        """Score can be created with required fields."""
        score = DimensionScore(
            dimension=EvaluationDimension.ACCURACY,
            score=4.5,
            confidence=0.9,
            feedback="Good accuracy",
        )
        assert score.dimension == EvaluationDimension.ACCURACY
        assert score.score == 4.5
        assert score.confidence == 0.9
        assert score.feedback == "Good accuracy"

    def test_default_examples(self):
        """Examples default to empty list."""
        score = DimensionScore(
            dimension=EvaluationDimension.CLARITY,
            score=3.0,
            confidence=0.7,
            feedback="Adequate",
        )
        assert score.examples == []

    def test_to_dict(self):
        """to_dict returns dictionary representation."""
        score = DimensionScore(
            dimension=EvaluationDimension.REASONING,
            score=4.0,
            confidence=0.85,
            feedback="Sound logic",
            examples=["Good deduction", "Clear argument"],
        )
        d = score.to_dict()
        assert d["dimension"] == "reasoning"
        assert d["score"] == 4.0
        assert d["confidence"] == 0.85
        assert d["feedback"] == "Sound logic"
        assert len(d["examples"]) == 2


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_create_result(self):
        """Result can be created with defaults."""
        result = EvaluationResult()
        assert result.id  # Should have auto-generated ID
        assert result.dimension_scores == {}
        assert result.overall_score == 0.0
        assert result.overall_confidence == 0.0

    def test_calculate_overall_score_empty(self):
        """calculate_overall_score returns 0 for empty scores."""
        result = EvaluationResult()
        score = result.calculate_overall_score()
        assert score == 0.0

    def test_calculate_overall_score_with_scores(self):
        """calculate_overall_score computes weighted average."""
        result = EvaluationResult()
        result.dimension_scores = {
            EvaluationDimension.RELEVANCE: DimensionScore(
                dimension=EvaluationDimension.RELEVANCE,
                score=5.0,
                confidence=0.9,
                feedback="Excellent",
            ),
            EvaluationDimension.ACCURACY: DimensionScore(
                dimension=EvaluationDimension.ACCURACY,
                score=4.0,
                confidence=0.8,
                feedback="Good",
            ),
        }
        # Use equal weights for simplicity
        weights = {
            EvaluationDimension.RELEVANCE: 0.5,
            EvaluationDimension.ACCURACY: 0.5,
        }
        score = result.calculate_overall_score(weights)
        assert score == 4.5  # (5.0 * 0.5 + 4.0 * 0.5) / 1.0

    def test_calculate_overall_score_updates_confidence(self):
        """calculate_overall_score also computes average confidence."""
        result = EvaluationResult()
        result.dimension_scores = {
            EvaluationDimension.CLARITY: DimensionScore(
                dimension=EvaluationDimension.CLARITY,
                score=4.0,
                confidence=0.8,
                feedback="Clear",
            ),
            EvaluationDimension.REASONING: DimensionScore(
                dimension=EvaluationDimension.REASONING,
                score=3.0,
                confidence=0.6,
                feedback="OK",
            ),
        }
        result.calculate_overall_score()
        assert result.overall_confidence == 0.7  # (0.8 + 0.6) / 2

    def test_to_dict(self):
        """to_dict returns complete dictionary."""
        result = EvaluationResult(
            response_id="test-123",
            overall_score=4.2,
            overall_confidence=0.85,
            judge_model="claude",
            use_case="debate",
            summary="Good response",
            strengths=["Clear", "Accurate"],
            weaknesses=["Could be more creative"],
            passes_threshold=True,
            threshold_used=3.5,
        )
        d = result.to_dict()
        assert d["response_id"] == "test-123"
        assert d["overall_score"] == 4.2
        assert d["judge_model"] == "claude"
        assert d["passes_threshold"] is True
        assert "timestamp" in d


class TestPairwiseResult:
    """Tests for PairwiseResult dataclass."""

    def test_create_result(self):
        """Result can be created with defaults."""
        result = PairwiseResult()
        assert result.id
        assert result.winner == ""
        assert result.confidence == 0.0

    def test_create_result_with_values(self):
        """Result can be created with all values."""
        result = PairwiseResult(
            response_a_id="resp-A",
            response_b_id="resp-B",
            winner="A",
            confidence=0.9,
            dimension_preferences={"accuracy": "A", "clarity": "B"},
            explanation="A was more accurate",
            judge_model="gpt-4o",
        )
        assert result.winner == "A"
        assert result.confidence == 0.9
        assert result.dimension_preferences["accuracy"] == "A"

    def test_to_dict(self):
        """to_dict returns complete dictionary."""
        result = PairwiseResult(
            response_a_id="A",
            response_b_id="B",
            winner="B",
            confidence=0.75,
            explanation="B was better",
        )
        d = result.to_dict()
        assert d["winner"] == "B"
        assert d["confidence"] == 0.75
        assert "timestamp" in d


class TestJudgeConfig:
    """Tests for JudgeConfig dataclass."""

    def test_default_config(self):
        """Config has sensible defaults."""
        config = JudgeConfig()
        assert "claude" in config.model.lower() or "gpt" in config.model.lower()
        assert config.temperature == 0.0
        assert config.pass_threshold == 3.5
        assert config.use_multiple_judges is False

    def test_custom_config(self):
        """Config accepts custom values."""
        config = JudgeConfig(
            model="gpt-4o",
            temperature=0.2,
            use_case="debate",
            pass_threshold=4.0,
            use_multiple_judges=True,
        )
        assert config.model == "gpt-4o"
        assert config.temperature == 0.2
        assert config.use_case == "debate"
        assert config.pass_threshold == 4.0

    def test_custom_weights(self):
        """Config can specify custom weights."""
        custom_weights = {
            EvaluationDimension.ACCURACY: 0.5,
            EvaluationDimension.SAFETY: 0.5,
        }
        config = JudgeConfig(custom_weights=custom_weights)
        assert config.custom_weights == custom_weights

    def test_custom_dimensions(self):
        """Config can specify subset of dimensions."""
        dims = [EvaluationDimension.ACCURACY, EvaluationDimension.RELEVANCE]
        config = JudgeConfig(dimensions=dims)
        assert config.dimensions == dims


class TestLLMJudge:
    """Tests for LLMJudge class."""

    def test_judge_creation_default(self):
        """Judge can be created with defaults."""
        judge = LLMJudge()
        assert judge._config is not None
        assert judge._rubrics == DEFAULT_RUBRICS
        assert judge._weights == DEFAULT_WEIGHTS

    def test_judge_creation_with_config(self):
        """Judge accepts custom config."""
        config = JudgeConfig(use_case="debate")
        judge = LLMJudge(config=config)
        assert judge._config.use_case == "debate"
        assert judge._weights == WEIGHT_PROFILES["debate"]

    def test_judge_custom_rubrics(self):
        """Judge merges custom rubrics."""
        custom_rubric = EvaluationRubric(
            dimension=EvaluationDimension.ACCURACY,
            description="Custom accuracy check",
            score_1="Bad",
            score_2="Poor",
            score_3="OK",
            score_4="Good",
            score_5="Great",
        )
        config = JudgeConfig(custom_rubrics={EvaluationDimension.ACCURACY: custom_rubric})
        judge = LLMJudge(config=config)
        assert judge._rubrics[EvaluationDimension.ACCURACY].description == "Custom accuracy check"

    def test_judge_dimension_subset(self):
        """Judge evaluates only specified dimensions."""
        dims = [EvaluationDimension.ACCURACY, EvaluationDimension.SAFETY]
        config = JudgeConfig(dimensions=dims)
        judge = LLMJudge(config=config)
        assert judge._dimensions == dims

    @pytest.mark.asyncio
    async def test_evaluate_calls_judge(self):
        """evaluate calls LLM and parses result."""
        judge = LLMJudge()

        # Mock the internal methods
        mock_response = """
        RELEVANCE: 4/5 - Good coverage
        ACCURACY: 5/5 - All facts correct

        SUMMARY: Overall good response
        STRENGTHS: Clear, accurate
        WEAKNESSES: Could be more detailed
        """

        with patch.object(judge, "_call_judge", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            with patch.object(judge, "_parse_evaluation") as mock_parse:
                mock_parse.return_value = {
                    EvaluationDimension.RELEVANCE: DimensionScore(
                        dimension=EvaluationDimension.RELEVANCE,
                        score=4.0,
                        confidence=0.8,
                        feedback="Good coverage",
                    ),
                    EvaluationDimension.ACCURACY: DimensionScore(
                        dimension=EvaluationDimension.ACCURACY,
                        score=5.0,
                        confidence=0.9,
                        feedback="All facts correct",
                    ),
                }

                with patch.object(judge, "_extract_feedback") as mock_feedback:
                    mock_feedback.return_value = {
                        "summary": "Overall good response",
                        "strengths": ["Clear", "accurate"],
                        "weaknesses": ["Could be more detailed"],
                        "suggestions": [],
                    }

                    result = await judge.evaluate(
                        query="What is Python?",
                        response="Python is a programming language.",
                    )

                    mock_call.assert_called_once()
                    assert result.dimension_scores
                    assert result.summary == "Overall good response"

    @pytest.mark.asyncio
    async def test_evaluate_handles_error(self):
        """evaluate handles LLM errors gracefully."""
        judge = LLMJudge()

        with patch.object(judge, "_call_judge", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = Exception("API error")

            result = await judge.evaluate(
                query="Test query",
                response="Test response",
            )

            assert "error" in result.summary.lower()

    @pytest.mark.asyncio
    async def test_compare_basic(self):
        """compare returns pairwise result."""
        judge = LLMJudge()

        mock_comparison = """
        WINNER: A
        CONFIDENCE: 0.8
        EXPLANATION: Response A was more accurate
        """

        with patch.object(judge, "_call_judge", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_comparison

            with patch.object(judge, "_parse_comparison") as mock_parse:
                mock_parse.return_value = {
                    "winner": "A",
                    "confidence": 0.8,
                    "explanation": "Response A was more accurate",
                    "dimension_preferences": {},
                }

                result = await judge.compare(
                    query="What is 2+2?",
                    response_a="4",
                    response_b="5",
                )

                assert result.winner == "A"
                assert result.confidence == 0.8

    @pytest.mark.asyncio
    async def test_compare_handles_error(self):
        """compare handles errors gracefully."""
        judge = LLMJudge()

        with patch.object(judge, "_call_judge", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = Exception("API error")

            result = await judge.compare(
                query="Test",
                response_a="A",
                response_b="B",
            )

            assert result.winner == "tie"
            assert "error" in result.explanation.lower()

    @pytest.mark.asyncio
    async def test_evaluate_batch(self):
        """evaluate_batch processes multiple items."""
        judge = LLMJudge()

        items = [
            {"query": "Q1", "response": "R1"},
            {"query": "Q2", "response": "R2"},
        ]

        with patch.object(judge, "evaluate", new_callable=AsyncMock) as mock_eval:
            mock_eval.return_value = EvaluationResult(overall_score=4.0)

            results = await judge.evaluate_batch(items)

            assert len(results) == 2
            assert mock_eval.call_count == 2
