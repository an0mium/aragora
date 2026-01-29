"""
OpenAPI Schema Definitions.

Contains common schema components used across all API endpoints.

This is the facade module that re-exports all schemas from domain-specific
modules while maintaining backward compatibility with the original
COMMON_SCHEMAS dictionary.
"""

from typing import Any

# Import domain-specific schemas
from .common import COMMON_SCHEMAS as _COMMON_SCHEMAS
from .debate import DEBATE_SCHEMAS
from .belief import BELIEF_SCHEMAS
from .enterprise import ENTERPRISE_SCHEMAS
from .workflow import WORKFLOW_SCHEMAS
from .explainability import EXPLAINABILITY_SCHEMAS
from .control_plane import CONTROL_PLANE_SCHEMAS
from .codebase import CODEBASE_SCHEMAS
from .cost import COST_SCHEMAS
from .inbox import INBOX_SCHEMAS
from .memory import MEMORY_SCHEMAS
from .analytics import ANALYTICS_SCHEMAS

# Import helpers
from .helpers import (
    ok_response,
    array_response,
    error_response,
    STANDARD_ERRORS,
)

# Merge all schemas into unified COMMON_SCHEMAS for backward compatibility
COMMON_SCHEMAS: dict[str, Any] = {
    **_COMMON_SCHEMAS,
    **DEBATE_SCHEMAS,
    **BELIEF_SCHEMAS,
    **ENTERPRISE_SCHEMAS,
    **WORKFLOW_SCHEMAS,
    **EXPLAINABILITY_SCHEMAS,
    **CONTROL_PLANE_SCHEMAS,
    **CODEBASE_SCHEMAS,
    **COST_SCHEMAS,
    **INBOX_SCHEMAS,
    **MEMORY_SCHEMAS,
    **ANALYTICS_SCHEMAS,
}


__all__ = [
    # Main unified schemas dictionary
    "COMMON_SCHEMAS",
    # Standard errors
    "STANDARD_ERRORS",
    # Helper functions
    "ok_response",
    "array_response",
    "error_response",
    # Domain-specific schemas (for direct access if needed)
    "DEBATE_SCHEMAS",
    "BELIEF_SCHEMAS",
    "ENTERPRISE_SCHEMAS",
    "WORKFLOW_SCHEMAS",
    "EXPLAINABILITY_SCHEMAS",
    "CONTROL_PLANE_SCHEMAS",
    "CODEBASE_SCHEMAS",
    "COST_SCHEMAS",
    "INBOX_SCHEMAS",
    "MEMORY_SCHEMAS",
    "ANALYTICS_SCHEMAS",
]
