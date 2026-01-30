"""
Governance storage package.

Provides persistent storage for decision governance artifacts including
approval requests, verification history, and decision records.

Submodules:
- models: Data classes (ApprovalRecord, VerificationRecord, DecisionRecord)
- metrics: Observability metric helpers
- store: GovernanceStore (sync, SQLite/PostgreSQL via DatabaseBackend)
- postgres_store: PostgresGovernanceStore (async, native asyncpg)
- factory: Singleton factory functions
"""

from .factory import get_governance_store, reset_governance_store
from .models import ApprovalRecord, DecisionRecord, VerificationRecord
from .postgres_store import PostgresGovernanceStore
from .store import GovernanceStore

__all__ = [
    "GovernanceStore",
    "PostgresGovernanceStore",
    "ApprovalRecord",
    "VerificationRecord",
    "DecisionRecord",
    "get_governance_store",
    "reset_governance_store",
]
