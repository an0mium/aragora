"""
Prompt building components for debate agents.

This package contains extracted modules from the original PromptBuilder
to improve code organization, testability, and maintainability.

Modules:
- context_formatters: Functions for formatting various context types
- domain_classifier: Question domain detection and classification
"""

from aragora.debate.prompts.context_formatters import (
    format_patterns_for_prompt,
    format_evidence_for_prompt,
    format_trending_for_prompt,
    format_successful_patterns,
    format_elo_ranking_context,
    format_calibration_context,
)

from aragora.debate.prompts.domain_classifier import (
    detect_question_domain_keywords,
    DOMAIN_KEYWORDS,
)

__all__ = [
    # Context formatters
    "format_patterns_for_prompt",
    "format_evidence_for_prompt",
    "format_trending_for_prompt",
    "format_successful_patterns",
    "format_elo_ranking_context",
    "format_calibration_context",
    # Domain classification
    "detect_question_domain_keywords",
    "DOMAIN_KEYWORDS",
]
