"""
Debates handlers subpackage.

This package contains debate-related handlers split by domain:
- handler: Main DebatesHandler class for debate endpoints (composes all mixins)
- analysis: Meta-critique and argument graph analysis mixin
- batch: Batch debate operations mixin
- create: Debate creation and cancellation mixin
- crud: CRUD operations mixin (list, get, update, delete)
- evidence: Citations, evidence, and verification reports mixin
- export: Export format operations mixin
- fork: Fork and follow-up debate operations mixin
- graph_debates: Graph-based multi-party debates
- matrix_debates: Matrix debate format
- response_formatting: Status normalization and response formatting
- routing: Route dispatch and authentication helpers
- search: Cross-debate search operations mixin
"""

from .analysis import AnalysisOperationsMixin
from .batch import BatchOperationsMixin
from .create import CreateOperationsMixin
from .crud import CrudOperationsMixin
from .evidence import EvidenceOperationsMixin
from .export import ExportOperationsMixin
from .fork import ForkOperationsMixin
from .graph_debates import GraphDebatesHandler, _graph_limiter
from .handler import DebateHandler, DebatesHandler
from .matrix_debates import MatrixDebatesHandler
from .response_formatting import (
    CACHE_TTL_CONVERGENCE,
    CACHE_TTL_DEBATES_LIST,
    CACHE_TTL_IMPASSE,
    CACHE_TTL_SEARCH,
    STATUS_MAP,
    STATUS_REVERSE_MAP,
    denormalize_status,
    normalize_debate_response,
    normalize_status,
)
from .routing import (
    ALLOWED_EXPORT_FORMATS,
    ALLOWED_EXPORT_TABLES,
    ARTIFACT_ENDPOINTS,
    AUTH_REQUIRED_ENDPOINTS,
    ID_ONLY_METHODS,
    ROUTES,
    SUFFIX_ROUTES,
    RoutingMixin,
    build_suffix_routes,
)
from .search import SearchOperationsMixin

__all__ = [
    # Main handlers
    "DebatesHandler",
    "DebateHandler",  # Backward compatibility alias
    "GraphDebatesHandler",
    "MatrixDebatesHandler",
    # Mixins
    "AnalysisOperationsMixin",
    "BatchOperationsMixin",
    "CreateOperationsMixin",
    "CrudOperationsMixin",
    "EvidenceOperationsMixin",
    "ExportOperationsMixin",
    "ForkOperationsMixin",
    "RoutingMixin",
    "SearchOperationsMixin",
    # Response formatting
    "CACHE_TTL_CONVERGENCE",
    "CACHE_TTL_DEBATES_LIST",
    "CACHE_TTL_IMPASSE",
    "CACHE_TTL_SEARCH",
    "STATUS_MAP",
    "STATUS_REVERSE_MAP",
    "denormalize_status",
    "normalize_debate_response",
    "normalize_status",
    # Routing configuration
    "ALLOWED_EXPORT_FORMATS",
    "ALLOWED_EXPORT_TABLES",
    "ARTIFACT_ENDPOINTS",
    "AUTH_REQUIRED_ENDPOINTS",
    "ID_ONLY_METHODS",
    "ROUTES",
    "SUFFIX_ROUTES",
    "build_suffix_routes",
    # Utilities
    "_graph_limiter",
]
