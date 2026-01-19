"""
Knowledge Base Handler Module.

This module provides modular endpoint handlers for the knowledge base API.
The handlers are split into mixins for better maintainability:

- FactsOperationsMixin: Fact CRUD operations
- QueryOperationsMixin: Natural language query handling
- SearchOperationsMixin: Chunk/embedding search operations

For Knowledge Mound handlers, see the mound/ submodule.
"""

from .handler import KnowledgeHandler
from .mound import KnowledgeMoundHandler

__all__ = [
    "KnowledgeHandler",
    "KnowledgeMoundHandler",
]
