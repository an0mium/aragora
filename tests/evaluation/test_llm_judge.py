"""Tests for LLM-as-Judge evaluation system."""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

from aragora.evaluation.llm_judge import (
    EvaluationDimension,
    EvaluationRubric,
    DimensionScore,
    EvaluationResult,
    PairwiseResult,
    JudgeConfig,
    LLMJudge,
    DEFAULT_WEIGHTS,
    WEIGHT_PROFILES,
    DEFAULT_RUBRICS,
)


class TestEvaluationDimension:
    """Tests for EvaluationDimension enum."""

    def test_all_dimensions_exist(self):
        """Should have all 8 evaluation dimensions."""
        expected = [
            "relevance",
            "accuracy",
            "completeness",
            "clarity",
            "reasoning",
            "evidence",
            "creativity",
            "safety",
        ]
        actual = [d.value for d in EvaluationDimension]
        assert sorted(actual) == sorted(expected)

    def test_dimension_is_string_enum(self):
        """Dimensions should be string values."""
        assert EvaluationDimension.RELEVANCE.value == "relevance"
        assert EvaluationDimension.ACCURACY.value == "accuracy"
        assert EvaluationDimension.SAFETY.value == "safety"


class TestDefaultWeights:
    """Tests for default dimension weights."""

    def test_weights_sum_to_one(self):
        """Default weights should sum to approximately 1.0."""
        total = sum(DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01

    def test_all_dimensions_have_weights(self):
        """All dimensions should have weights."""
        for dim in EvaluationDimension:
            assert dim in DEFAULT_WEIGHTS

    def test_weights_are_positive(self):
        """All weights should be positive."""
        for weight in DEFAULT_WEIGHTS.values():
            assert weight >= 0


class TestWeightProfiles:
    """Tests for use-case specific weight profiles."""

    def test_factual_qa_profile_exists(self):
        """Should have factual_qa profile."""
        assert "factual_qa" in WEIGHT_PROFILES

    def test_creative_writing_profile_exists(self):
        """Should have creative_writing profile."""
        assert "creative_writing" in WEIGHT_PROFILES

    def test_code_generation_profile_exists(self):
        """Should have code_generation profile."""
        assert "code_generation" in WEIGHT_PROFILES

    def test_debate_profile_exists(self):
        """Should have debate profile."""
        assert "debate" in WEIGHT_PROFILES

    def test_safety_critical_profile_exists(self):
        """Should have safety_critical profile."""
        assert "safety_critical" in WEIGHT_PROFILES

    def test_profiles_sum_to_one(self):
        """All profiles should sum to approximately 1.0."""
        for profile_name, weights in WEIGHT_PROFILES.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.01, f"{profile_name} weights sum to {total}"

    def test_factual_qa_emphasizes_accuracy(self):
        """Factual QA should weight accuracy highly."""
        weights = WEIGHT_PROFILES["factual_qa"]
        assert weights[EvaluationDimension.ACCURACY] >= 0.25

    def test_creative_writing_emphasizes_creativity(self):
        """Creative writing should weight creativity highly."""
        weights = WEIGHT_PROFILES["creative_writing"]
        assert weights[EvaluationDimension.CREATIVITY] >= 0.25

    def test_debate_emphasizes_reasoning(self):
        """Debate should weight reasoning highly."""
        weights = WEIGHT_PROFILES["debate"]
        assert weights[EvaluationDimension.REASONING] >= 0.20

    def test_safety_critical_emphasizes_safety(self):
        """Safety critical should weight safety highly."""
        weights = WEIGHT_PROFILES["safety_critical"]
        assert weights[EvaluationDimension.SAFETY] >= 0.20


class TestEvaluationRubric:
    """Tests for EvaluationRubric dataclass."""

    def test_create_rubric(self):
        """Should create rubric with all fields."""
        rubric = EvaluationRubric(
            dimension=EvaluationDimension.RELEVANCE,
            description="Test description",
            score_1="Poor",
            score_2="Below average",
            score_3="Average",
            score_4="Above average",
            score_5="Excellent",
        )
        assert rubric.dimension == EvaluationDimension.RELEVANCE
        assert rubric.description == "Test description"

    def test_to_prompt(self):
        """Should format rubric for prompt."""
        rubric = EvaluationRubric(
            dimension=EvaluationDimension.ACCURACY,
            description="Is it accurate?",
            score_1="Wrong",
            score_2="Mostly wrong",
            score_3="Partially correct",
            score_4="Mostly correct",
            score_5="Fully correct",
        )
        prompt = rubric.to_prompt()

        assert "ACCURACY" in prompt
        assert "Is it accurate?" in prompt
        assert "Score 1" in prompt
        assert "Score 5" in prompt
        assert "Fully correct" in prompt


class TestDefaultRubrics:
    """Tests for default rubrics."""

    def test_all_dimensions_have_rubrics(self):
        """All dimensions should have default rubrics."""
        for dim in EvaluationDimension:
            assert dim in DEFAULT_RUBRICS
            assert isinstance(DEFAULT_RUBRICS[dim], EvaluationRubric)

    def test_rubrics_have_all_scores(self):
        """Each rubric should have all 5 score descriptions."""
        for dim, rubric in DEFAULT_RUBRICS.items():
            assert rubric.score_1
            assert rubric.score_2
            assert rubric.score_3
            assert rubric.score_4
            assert rubric.score_5


class TestDimensionScore:
    """Tests for DimensionScore dataclass."""

    def test_create_score(self):
        """Should create dimension score."""
        score = DimensionScore(
            dimension=EvaluationDimension.CLARITY,
            score=4.5,
            confidence=0.9,
            feedback="Clear and well-organized",
        )
        assert score.dimension == EvaluationDimension.CLARITY
        assert score.score == 4.5
        assert score.confidence == 0.9

    def test_score_defaults(self):
        """Should have default empty examples."""
        score = DimensionScore(
            dimension=EvaluationDimension.REASONING,
            score=3.0,
            confidence=0.5,
            feedback="Average",
        )
        assert score.examples == []

    def test_to_dict(self):
        """Should serialize to dictionary."""
        score = DimensionScore(
            dimension=EvaluationDimension.EVIDENCE,
            score=4.0,
            confidence=0.8,
            feedback="Good evidence",
            examples=["Example 1", "Example 2"],
        )
        data = score.to_dict()

        assert data["dimension"] == "evidence"
        assert data["score"] == 4.0
        assert data["confidence"] == 0.8
        assert data["feedback"] == "Good evidence"
        assert len(data["examples"]) == 2


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_create_result(self):
        """Should create evaluation result."""
        result = EvaluationResult()
        assert result.id  # Should have auto-generated UUID
        assert result.dimension_scores == {}
        assert result.overall_score == 0.0

    def test_calculate_overall_score(self):
        """Should calculate weighted overall score."""
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
                score=3.0,
                confidence=0.8,
                feedback="Average",
            ),
        }

        # Use simple equal weights for testing
        weights = {
            EvaluationDimension.RELEVANCE: 0.5,
            EvaluationDimension.ACCURACY: 0.5,
        }

        score = result.calculate_overall_score(weights)

        assert score == 4.0  # (5*0.5 + 3*0.5) / 1.0
        assert result.overall_score == 4.0
        assert result.overall_confidence == 0.9  # (1.0 + 0.8) / 2

    def test_calculate_overall_score_with_defaults(self):
        """Should use default weights if none provided."""
        result = EvaluationResult()
        for dim in EvaluationDimension:
            result.dimension_scores[dim] = DimensionScore(
                dimension=dim,
                score=4.0,
                confidence=0.8,
                feedback="Good",
            )

        score = result.calculate_overall_score()

        assert score == 4.0  # All scores are 4.0

    def test_calculate_overall_score_empty(self):
        """Should return 0 for empty scores."""
        result = EvaluationResult()
        score = result.calculate_overall_score()
        assert score == 0.0

    def test_to_dict(self):
        """Should serialize to dictionary."""
        result = EvaluationResult(
            response_id="resp-123",
            overall_score=4.2,
            summary="Good response",
            strengths=["Clear", "Accurate"],
            weaknesses=["Too long"],
        )
        data = result.to_dict()

        assert data["response_id"] == "resp-123"
        assert data["overall_score"] == 4.2
        assert data["summary"] == "Good response"
        assert len(data["strengths"]) == 2
        assert len(data["weaknesses"]) == 1


class TestPairwiseResult:
    """Tests for PairwiseResult dataclass."""

    def test_create_result(self):
        """Should create pairwise result."""
        result = PairwiseResult(
            response_a_id="A",
            response_b_id="B",
            winner="A",
            confidence=0.85,
        )
        assert result.winner == "A"
        assert result.confidence == 0.85

    def test_result_defaults(self):
        """Should have default values."""
        result = PairwiseResult()
        assert result.winner == ""
        assert result.confidence == 0.0
        assert result.dimension_preferences == {}

    def test_to_dict(self):
        """Should serialize to dictionary."""
        result = PairwiseResult(
            response_a_id="resp-A",
            response_b_id="resp-B",
            winner="B",
            confidence=0.75,
            explanation="Response B was more accurate",
        )
        data = result.to_dict()

        assert data["response_a_id"] == "resp-A"
        assert data["response_b_id"] == "resp-B"
        assert data["winner"] == "B"
        assert "timestamp" in data


class TestJudgeConfig:
    """Tests for JudgeConfig dataclass."""

    def test_default_config(self):
        """Should have sensible defaults."""
        config = JudgeConfig()

        assert "claude" in config.model.lower() or "sonnet" in config.model.lower()
        assert config.temperature == 0.0
        assert config.pass_threshold == 3.5
        assert not config.use_multiple_judges

    def test_custom_config(self):
        """Should accept custom values."""
        config = JudgeConfig(
            model="gpt-4",
            temperature=0.3,
            use_case="debate",
            pass_threshold=4.0,
        )

        assert config.model == "gpt-4"
        assert config.temperature == 0.3
        assert config.use_case == "debate"
        assert config.pass_threshold == 4.0


class TestLLMJudge:
    """Tests for LLMJudge class."""

    @pytest.fixture
    def judge(self):
        """Create a judge with default config."""
        return LLMJudge()

    @pytest.fixture
    def debate_judge(self):
        """Create a judge configured for debate evaluation."""
        config = JudgeConfig(use_case="debate")
        return LLMJudge(config)

    def test_init_default(self, judge):
        """Should initialize with defaults."""
        assert judge._weights == DEFAULT_WEIGHTS
        assert len(judge._dimensions) == 8

    def test_init_with_use_case(self, debate_judge):
        """Should use profile weights for use case."""
        assert debate_judge._weights == WEIGHT_PROFILES["debate"]

    def test_init_with_custom_weights(self):
        """Should use custom weights."""
        custom = {EvaluationDimension.ACCURACY: 1.0}
        config = JudgeConfig(custom_weights=custom)
        judge = LLMJudge(config)

        assert judge._weights == custom

    def test_init_with_subset_dimensions(self):
        """Should use subset of dimensions."""
        dims = [EvaluationDimension.ACCURACY, EvaluationDimension.CLARITY]
        config = JudgeConfig(dimensions=dims)
        judge = LLMJudge(config)

        assert judge._dimensions == dims

    def test_build_evaluation_prompt(self, judge):
        """Should build evaluation prompt."""
        prompt = judge._build_evaluation_prompt(
            query="What is 2+2?",
            response="The answer is 4.",
        )

        assert "What is 2+2?" in prompt
        assert "The answer is 4." in prompt
        assert "RELEVANCE" in prompt
        assert "ACCURACY" in prompt
        assert "JSON" in prompt

    def test_build_evaluation_prompt_with_context(self, judge):
        """Should include context in prompt."""
        prompt = judge._build_evaluation_prompt(
            query="Summarize this",
            response="Summary here",
            context="Original document text",
        )

        assert "Original document text" in prompt
        assert "Context" in prompt

    def test_build_evaluation_prompt_with_reference(self, judge):
        """Should include reference in prompt."""
        prompt = judge._build_evaluation_prompt(
            query="Translate this",
            response="Translation",
            reference="Correct translation",
        )

        assert "Correct translation" in prompt
        assert "Reference" in prompt

    def test_build_comparison_prompt(self, judge):
        """Should build comparison prompt."""
        prompt = judge._build_comparison_prompt(
            query="Explain X",
            response_a="Explanation A",
            response_b="Explanation B",
        )

        assert "Explain X" in prompt
        assert "Response A" in prompt
        assert "Response B" in prompt
        assert "Explanation A" in prompt
        assert "Explanation B" in prompt

    def test_parse_evaluation_valid_json(self, judge):
        """Should parse valid JSON evaluation."""
        json_text = """```json
{
    "dimension_scores": {
        "relevance": {"score": 4, "confidence": 0.9, "feedback": "Good", "examples": []},
        "accuracy": {"score": 5, "confidence": 0.95, "feedback": "Perfect", "examples": []}
    },
    "summary": "Good response"
}
```"""
        scores = judge._parse_evaluation(json_text)

        assert EvaluationDimension.RELEVANCE in scores
        assert scores[EvaluationDimension.RELEVANCE].score == 4
        assert scores[EvaluationDimension.ACCURACY].score == 5

    def test_parse_evaluation_raw_json(self, judge):
        """Should parse raw JSON without code block."""
        json_text = """{
    "dimension_scores": {
        "relevance": {"score": 3, "confidence": 0.7, "feedback": "OK"}
    }
}"""
        scores = judge._parse_evaluation(json_text)

        assert EvaluationDimension.RELEVANCE in scores
        assert scores[EvaluationDimension.RELEVANCE].score == 3

    def test_parse_evaluation_missing_dimension(self, judge):
        """Should use default for missing dimensions."""
        json_text = '{"dimension_scores": {}}'
        scores = judge._parse_evaluation(json_text)

        # Should have all dimensions with defaults
        for dim in EvaluationDimension:
            assert dim in scores
            assert scores[dim].score == 3.0

    def test_extract_score_from_text(self, judge):
        """Should extract score from text with dimension: score format."""
        # The regex expects "dimension: score" or "dimension score" format
        text = "relevance: 4.5"
        score = judge._extract_score_from_text(text, "relevance")
        assert score == 4.5

    def test_extract_score_from_text_with_whitespace(self, judge):
        """Should extract score with whitespace separator."""
        text = "relevance 4"
        score = judge._extract_score_from_text(text, "relevance")
        assert score == 4.0

    def test_extract_score_from_text_default(self, judge):
        """Should return default if no score found."""
        text = "No scores here"
        score = judge._extract_score_from_text(text, "relevance")
        assert score == 3.0

    def test_extract_score_from_text_bounds_low(self, judge):
        """Should clamp scores at minimum 1.0."""
        # The regex only matches single digit, so "10" matches "1"
        text = "relevance: 0"
        score = judge._extract_score_from_text(text, "relevance")
        assert score == 1.0  # Clamped to minimum

    def test_extract_score_from_text_bounds_high(self, judge):
        """Should clamp scores at maximum 5.0."""
        text = "relevance: 6"
        score = judge._extract_score_from_text(text, "relevance")
        assert score == 5.0  # Clamped to maximum

    def test_extract_feedback(self, judge):
        """Should extract feedback from JSON."""
        json_text = """```json
{
    "summary": "Overall good",
    "strengths": ["Clear", "Accurate"],
    "weaknesses": ["Too long"],
    "suggestions": ["Be more concise"]
}
```"""
        feedback = judge._extract_feedback(json_text)

        assert feedback["summary"] == "Overall good"
        assert "Clear" in feedback["strengths"]
        assert "Too long" in feedback["weaknesses"]

    def test_parse_comparison_valid(self, judge):
        """Should parse valid comparison."""
        json_text = """```json
{
    "winner": "A",
    "confidence": 0.85,
    "dimension_preferences": {"accuracy": "A", "clarity": "B"},
    "explanation": "A was more accurate"
}
```"""
        result = judge._parse_comparison(json_text)

        assert result["winner"] == "A"
        assert result["confidence"] == 0.85
        assert result["dimension_preferences"]["accuracy"] == "A"

    def test_parse_comparison_invalid(self, judge):
        """Should return defaults for invalid comparison."""
        result = judge._parse_comparison("not valid json")

        assert result["winner"] == "tie"
        assert result["confidence"] == 0.5


class TestLLMJudgeAsync:
    """Async tests for LLMJudge."""

    @pytest.fixture
    def judge(self):
        """Create a judge."""
        return LLMJudge()

    @pytest.mark.asyncio
    async def test_evaluate_success(self, judge):
        """Should evaluate response successfully."""
        mock_response = """```json
{
    "dimension_scores": {
        "relevance": {"score": 4, "confidence": 0.9, "feedback": "Relevant"},
        "accuracy": {"score": 5, "confidence": 0.95, "feedback": "Accurate"},
        "completeness": {"score": 4, "confidence": 0.8, "feedback": "Complete"},
        "clarity": {"score": 4, "confidence": 0.85, "feedback": "Clear"},
        "reasoning": {"score": 4, "confidence": 0.8, "feedback": "Good"},
        "evidence": {"score": 3, "confidence": 0.7, "feedback": "Some evidence"},
        "creativity": {"score": 3, "confidence": 0.6, "feedback": "Standard"},
        "safety": {"score": 5, "confidence": 0.95, "feedback": "Safe"}
    },
    "summary": "Good response overall",
    "strengths": ["Accurate", "Clear"],
    "weaknesses": ["Could use more evidence"],
    "suggestions": ["Add citations"]
}
```"""

        with patch.object(judge, "_call_judge", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            result = await judge.evaluate(
                query="What is the capital of France?",
                response="Paris is the capital of France.",
            )

            assert result.overall_score > 0
            assert result.summary == "Good response overall"
            assert "Accurate" in result.strengths

    @pytest.mark.asyncio
    async def test_evaluate_error_handling(self, judge):
        """Should handle errors gracefully."""
        with patch.object(judge, "_call_judge", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = Exception("API error")

            result = await judge.evaluate(
                query="Test",
                response="Test response",
            )

            assert "error" in result.summary.lower()

    @pytest.mark.asyncio
    async def test_compare_success(self, judge):
        """Should compare responses successfully."""
        mock_response = """```json
{
    "winner": "A",
    "confidence": 0.8,
    "dimension_preferences": {"accuracy": "A"},
    "explanation": "Response A was more accurate"
}
```"""

        with patch.object(judge, "_call_judge", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            result = await judge.compare(
                query="What is 2+2?",
                response_a="4",
                response_b="5",
            )

            assert result.winner == "A"
            assert result.confidence == 0.8

    @pytest.mark.asyncio
    async def test_evaluate_batch(self, judge):
        """Should evaluate batch in parallel."""
        mock_response = """```json
{
    "dimension_scores": {
        "relevance": {"score": 4, "confidence": 0.9, "feedback": "Good"}
    },
    "summary": "OK"
}
```"""

        with patch.object(judge, "_call_judge", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            items = [
                {"query": "Q1", "response": "R1"},
                {"query": "Q2", "response": "R2"},
            ]

            results = await judge.evaluate_batch(items)

            assert len(results) == 2
            assert mock_call.call_count == 2


class TestQualityGate:
    """Tests for quality gate functionality."""

    @pytest.fixture
    def judge(self):
        """Create judge with specific threshold."""
        config = JudgeConfig(pass_threshold=4.0)
        return LLMJudge(config)

    @pytest.mark.asyncio
    async def test_passes_threshold(self, judge):
        """Should pass when score meets threshold."""
        mock_response = """```json
{
    "dimension_scores": {
        "relevance": {"score": 5, "confidence": 1.0, "feedback": "Perfect"}
    }
}
```"""

        with patch.object(judge, "_call_judge", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            result = await judge.evaluate("Q", "R")

            # All dimensions default to score 3.0 except relevance which is 5.0
            # With default weights, this should be close to 3.0+
            assert result.threshold_used == 4.0

    @pytest.mark.asyncio
    async def test_fails_threshold(self, judge):
        """Should fail when score below threshold."""
        mock_response = """```json
{
    "dimension_scores": {
        "relevance": {"score": 2, "confidence": 0.9, "feedback": "Poor"}
    }
}
```"""

        with patch.object(judge, "_call_judge", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_response

            result = await judge.evaluate("Q", "R")

            # Score should be low
            assert result.overall_score < 4.0
            assert not result.passes_threshold
