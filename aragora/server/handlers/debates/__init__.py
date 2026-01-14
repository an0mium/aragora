"""
Debates handlers subpackage.

This package contains debate-related handlers split by domain:
- handler: Main DebatesHandler class for debate endpoints
- batch: Batch debate operations mixin
- fork: Fork and follow-up debate operations mixin
"""

from .batch import BatchOperationsMixin
from .fork import ForkOperationsMixin
from .handler import (
    DebatesHandler,
    STATUS_MAP,
    STATUS_REVERSE_MAP,
    denormalize_status,
    normalize_debate_response,
    normalize_status,
)

__all__ = [
    "DebatesHandler",
    "BatchOperationsMixin",
    "ForkOperationsMixin",
    "normalize_status",
    "denormalize_status",
    "normalize_debate_response",
    "STATUS_MAP",
    "STATUS_REVERSE_MAP",
]
