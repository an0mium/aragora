"""
Aragora Prompts - Specialized prompts for different debate types.
"""

from aragora.prompts.code_review import (
    SECURITY_PROMPT,
    PERFORMANCE_PROMPT,
    QUALITY_PROMPT,
    build_review_prompt,
    get_focus_prompts,
)

__all__ = [
    "SECURITY_PROMPT",
    "PERFORMANCE_PROMPT",
    "QUALITY_PROMPT",
    "build_review_prompt",
    "get_focus_prompts",
]
