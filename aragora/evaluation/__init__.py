"""
Evaluation Module - LLM-as-Judge for Agent Outputs.

Provides comprehensive evaluation of agent responses across
8 dimensions:
- Relevance: How well the response addresses the query
- Accuracy: Factual correctness of claims
- Completeness: Coverage of all aspects
- Clarity: Readability and understandability
- Reasoning: Quality of logical arguments
- Evidence: Use of supporting evidence
- Creativity: Novel insights and approaches
- Safety: Absence of harmful content

Features:
- Multi-model judging for reliability
- Calibrated scoring with rubrics
- Comparative evaluation (pairwise)
- Dimension weighting by use case
- Detailed feedback generation
"""

from aragora.evaluation.llm_judge import (
    EvaluationDimension,
    EvaluationResult,
    DimensionScore,
    EvaluationRubric,
    LLMJudge,
    JudgeConfig,
    PairwiseResult,
    evaluate_response,
    compare_responses,
)

__all__ = [
    "EvaluationDimension",
    "EvaluationResult",
    "DimensionScore",
    "EvaluationRubric",
    "LLMJudge",
    "JudgeConfig",
    "PairwiseResult",
    "evaluate_response",
    "compare_responses",
]
