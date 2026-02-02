"""
Knowledge Base endpoint handlers.

DEPRECATED: This module is now refactored into the knowledge_base/ subdirectory.
Import from knowledge_base instead:

    from aragora.server.handlers.knowledge_base import KnowledgeHandler, KnowledgeMoundHandler

Or via the handlers package:

    from aragora.server.handlers import KnowledgeHandler, KnowledgeMoundHandler

This file maintains backward compatibility by re-exporting from the new location.

The modular structure provides:
- handlers/knowledge_base/
  ├── __init__.py           # Re-exports KnowledgeHandler
  ├── handler.py            # Main KnowledgeHandler
  ├── facts.py              # FactsOperationsMixin
  ├── query.py              # QueryOperationsMixin
  ├── search.py             # SearchOperationsMixin
  └── mound/
      ├── __init__.py       # Re-exports KnowledgeMoundHandler
      ├── handler.py        # Main KnowledgeMoundHandler
      ├── nodes.py          # NodeOperationsMixin
      ├── relationships.py  # RelationshipOperationsMixin
      ├── graph.py          # GraphOperationsMixin
      ├── culture.py        # CultureOperationsMixin
      ├── staleness.py      # StalenessOperationsMixin
      ├── sync.py           # SyncOperationsMixin
      └── export.py         # ExportOperationsMixin
"""

from __future__ import annotations

import warnings

from aragora.rbac.decorators import require_permission

# Re-export from the new modular location for backward compatibility
from .knowledge_base import KnowledgeHandler, KnowledgeMoundHandler

# =============================================================================
# RBAC Permissions
# =============================================================================

KNOWLEDGE_READ_PERMISSION = "knowledge:read"

__all__ = [
    "KnowledgeHandler",
    "KnowledgeMoundHandler",
    "KNOWLEDGE_READ_PERMISSION",
]

# Issue deprecation warning (only once per session)
warnings.warn(
    "aragora.server.handlers.knowledge is deprecated. "
    "Import from aragora.server.handlers.knowledge_base instead.",
    DeprecationWarning,
    stacklevel=2,
)
