"""
Content Moderation Module.

Provides spam filtering, content validation, and moderation
pipeline integration for the Aragora debate platform.
"""

from __future__ import annotations

from .spam_integration import (
    SpamVerdict,
    SpamCheckResult,
    SpamModerationConfig,
    ModerationQueueItem,
    SpamModerationIntegration,
    ContentModerationError,
    get_spam_moderation,
    check_debate_content,
    list_review_queue,
    pop_review_item,
    queue_for_review,
    review_queue_size,
)

__all__ = [
    "SpamVerdict",
    "SpamCheckResult",
    "SpamModerationConfig",
    "ModerationQueueItem",
    "SpamModerationIntegration",
    "ContentModerationError",
    "get_spam_moderation",
    "check_debate_content",
    "list_review_queue",
    "pop_review_item",
    "queue_for_review",
    "review_queue_size",
]
