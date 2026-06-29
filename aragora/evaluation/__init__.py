"""Evaluation helpers for agent output scoring and structural metrics.

Provides comprehensive evaluation of agent responses across 8 dimensions:
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

Imports from this package are lazy so lightweight metric modules under
``aragora.evaluation`` do not load the LLM judge/config stack unless callers ask
for those judge symbols directly.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

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

_LLM_JUDGE_NAMES = set(__all__)


def __getattr__(name: str) -> Any:
    if name in _LLM_JUDGE_NAMES:
        module = import_module("aragora.evaluation.llm_judge")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'aragora.evaluation' has no attribute {name!r}")
