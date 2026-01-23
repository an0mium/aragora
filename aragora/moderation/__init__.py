"""
Content Moderation Module.

Provides spam filtering, content validation, and moderation
pipeline integration for the Aragora debate platform.
"""

from __future__ import annotations

from .spam_integration import (
    SpamVerdict,
    SpamCheckResult,
    SpamModerationIntegration,
    ContentModerationError,
    get_spam_moderation,
    check_debate_content,
)

__all__ = [
    "SpamVerdict",
    "SpamCheckResult",
    "SpamModerationIntegration",
    "ContentModerationError",
    "get_spam_moderation",
    "check_debate_content",
]
