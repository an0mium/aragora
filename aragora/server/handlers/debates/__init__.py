"""
Debates handlers subpackage.

This package contains debate-related handlers split by domain:
- handler: Main DebatesHandler class for debate endpoints
- batch: Batch debate operations mixin
- fork: Fork and follow-up debate operations mixin
- graph_debates: Graph-based multi-party debates
- matrix_debates: Matrix debate format
- response_formatting: Status normalization and response formatting
"""

from .batch import BatchOperationsMixin
from .fork import ForkOperationsMixin
from .graph_debates import GraphDebatesHandler, _graph_limiter
from .handler import DebatesHandler
from .matrix_debates import MatrixDebatesHandler
from .response_formatting import (
    STATUS_MAP,
    STATUS_REVERSE_MAP,
    denormalize_status,
    normalize_debate_response,
    normalize_status,
)

__all__ = [
    "BatchOperationsMixin",
    "DebatesHandler",
    "ForkOperationsMixin",
    "GraphDebatesHandler",
    "MatrixDebatesHandler",
    "STATUS_MAP",
    "STATUS_REVERSE_MAP",
    "_graph_limiter",
    "denormalize_status",
    "normalize_debate_response",
    "normalize_status",
]
