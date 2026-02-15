"""
Aragora persistence layer.

Provides persistent storage for nomic loop history, debate artifacts,
real-time event streaming, and centralized database configuration.
"""

from aragora.persistence.db_config import (
    DatabaseMode,
    DatabaseType,
    get_db_mode,
    get_db_path,
    get_db_path_str,
    get_elo_db_path,
    get_genesis_db_path,
    get_insights_db_path,
    get_memory_db_path,
    get_nomic_dir,
    get_personas_db_path,
    get_positions_db_path,
)
from aragora.persistence.models import (
    AgentMetrics,
    DebateArtifact,
    NomicCycle,
    StreamEvent,
)
# SupabaseClient is lazily imported via __getattr__ below.

__all__ = [
    # Supabase client
    "SupabaseClient",
    # Models
    "NomicCycle",
    "DebateArtifact",
    "StreamEvent",
    "AgentMetrics",
    # Database configuration
    "DatabaseType",
    "DatabaseMode",
    "get_db_path",
    "get_db_path_str",
    "get_db_mode",
    "get_nomic_dir",
    "get_elo_db_path",
    "get_memory_db_path",
    "get_positions_db_path",
    "get_personas_db_path",
    "get_insights_db_path",
    "get_genesis_db_path",
]


def __getattr__(name: str):
    if name == "SupabaseClient":
        from aragora.persistence.supabase_client import SupabaseClient

        globals()["SupabaseClient"] = SupabaseClient
        return SupabaseClient
    raise AttributeError(f"module 'aragora.persistence' has no attribute {name!r}")
