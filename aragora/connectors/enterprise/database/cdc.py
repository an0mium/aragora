"""
Change Data Capture (CDC) utilities.

Provides a unified model for tracking database changes across
different database systems (PostgreSQL, MongoDB, etc.).

Features:
- Unified ChangeEvent model for all operations
- Resume token management for reliable streaming
- Change event handlers for Knowledge Mound integration
- Operation type classification

Note: Implementation is split across submodules for maintainability:
- cdc_models: Core data models, enums, and configuration
- cdc_handlers: Event handler classes and stream manager
- cdc_databases: Database-specific CDC handler implementations

This module re-exports everything for backward compatibility.
"""

from __future__ import annotations

# Re-export from cdc_models
from .cdc_models import (
    CDCConfig,
    CDCSourceType,
    ChangeEvent,
    ChangeOperation,
    ResumeToken,
    ResumeTokenStore,
)

# Re-export from cdc_handlers
from .cdc_handlers import (
    CallbackHandler,
    CDCStreamManager,
    ChangeEventHandler,
    CompositeHandler,
    KnowledgeMoundHandler,
)

# Re-export from cdc_databases
from .cdc_databases import (
    BaseCDCHandler,
    MongoDBCDCHandler,
    MySQLCDCHandler,
    PostgresCDCHandler,
    create_mongodb_cdc,
    create_mysql_cdc,
    create_postgres_cdc,
)

__all__ = [
    # Models and enums
    "ChangeEvent",
    "ChangeOperation",
    "CDCSourceType",
    "ResumeToken",
    "ResumeTokenStore",
    "CDCConfig",
    # Event handlers
    "ChangeEventHandler",
    "KnowledgeMoundHandler",
    "CallbackHandler",
    "CompositeHandler",
    "CDCStreamManager",
    # Database CDC handlers
    "BaseCDCHandler",
    "PostgresCDCHandler",
    "MySQLCDCHandler",
    "MongoDBCDCHandler",
    # Factory functions
    "create_postgres_cdc",
    "create_mysql_cdc",
    "create_mongodb_cdc",
]
