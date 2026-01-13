"""
Audience participation module for Aragora debates.

Provides sanitization, clustering, and formatting of audience suggestions
for injection into agent prompts.
"""

from .suggestions import (
    SuggestionCluster,
    cluster_suggestions,
    format_for_prompt,
    sanitize_suggestion,
)

__all__ = [
    "SuggestionCluster",
    "sanitize_suggestion",
    "cluster_suggestions",
    "format_for_prompt",
]
