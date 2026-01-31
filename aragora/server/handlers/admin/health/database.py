"""
Database health check implementations.

Provides health checks for:
- Database schema validation (consolidated databases)
- All database stores (debate storage, ELO, insights, etc.)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol, runtime_checkable

from ...base import HandlerResult, json_response

logger = logging.getLogger(__name__)


@runtime_checkable
class _HealthHandlerProtocol(Protocol):
    """Protocol for handlers used by database health checks."""

    ctx: dict[str, Any]

    def get_storage(self) -> Any: ...
    def get_elo_system(self) -> Any: ...
    def get_nomic_dir(self) -> Any: ...


def database_schema_health(handler: _HealthHandlerProtocol) -> HandlerResult:
    """Check health of consolidated database schema.

    Validates that all required tables exist in consolidated databases:
    - core.db: debates, traces, tournaments, embeddings, positions
    - memory.db: continuum_memory, consensus, critiques, patterns
    - analytics.db: ratings, matches, insights, predictions
    - agents.db: personas, genomes, populations, genesis_events

    This endpoint helps diagnose schema issues after migration.

    Returns:
        JSON with database health status, missing tables, and validation errors.
    """
    try:
        from aragora.persistence.validator import get_database_health

        health = get_database_health()
        status_code = 200 if health["status"] == "healthy" else 503
        return json_response(health, status=status_code)
    except ImportError:
        return json_response(
            {
                "status": "unavailable",
                "error": "Database validator not available",
            },
            status=503,
        )
    except Exception as e:
        logger.exception(f"Database schema health check failed: {e}")
        return json_response(
            {
                "status": "error",
                "error": str(e),
            },
            status=500,
        )


def database_stores_health(handler: _HealthHandlerProtocol) -> HandlerResult:
    """Check health of all database stores.

    Returns detailed status for each database store:
    - debate_storage: Main debate persistence (SQLite/Supabase)
    - elo_system: Agent rankings database
    - insight_store: Debate insights database
    - flip_detector: Flip detection database
    - consensus_memory: Consensus patterns database
    - user_store: User and organization data
    - agent_metadata: Agent metadata from seed script
    - integration_store: Third-party integrations
    - gmail_token_store: Gmail OAuth tokens
    - sync_store: Enterprise sync
    - decision_result_store: Decision persistence

    This endpoint helps diagnose which specific stores are
    initialized, connected, and functioning.
    """
    from .database_utils import (
        check_agent_metadata,
        check_consensus_memory,
        check_debate_storage,
        check_decision_result_store,
        check_elo_system,
        check_flip_detector,
        check_gmail_token_store,
        check_insight_store,
        check_integration_store,
        check_sync_store,
        check_user_store,
        handle_store_check_errors,
    )

    stores: dict[str, dict[str, Any]] = {}
    all_healthy = True
    start_time = time.time()

    # Define all store checks with their names and check functions
    store_checks = [
        ("debate_storage", lambda: check_debate_storage(handler)),
        ("elo_system", lambda: check_elo_system(handler)),
        ("insight_store", lambda: check_insight_store(handler)),
        ("flip_detector", lambda: check_flip_detector(handler)),
        ("user_store", lambda: check_user_store(handler)),
        ("consensus_memory", lambda: check_consensus_memory(handler)),
        ("agent_metadata", lambda: check_agent_metadata(handler)),
        ("integration_store", lambda: check_integration_store(handler)),
        ("gmail_token_store", lambda: check_gmail_token_store(handler)),
        ("sync_store", lambda: check_sync_store(handler)),
        ("decision_result_store", lambda: check_decision_result_store(handler)),
    ]

    # Run all store checks with unified error handling
    for store_name, check_fn in store_checks:
        result, is_healthy = handle_store_check_errors(store_name, check_fn)
        stores[store_name] = result
        if not is_healthy:
            all_healthy = False

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return json_response(
        {
            "status": "healthy" if all_healthy else "degraded",
            "stores": stores,
            "elapsed_ms": elapsed_ms,
            "summary": {
                "total": len(stores),
                "healthy": sum(1 for s in stores.values() if s.get("healthy", False)),
                "connected": sum(1 for s in stores.values() if s.get("status") == "connected"),
                "not_initialized": sum(
                    1 for s in stores.values() if s.get("status") == "not_initialized"
                ),
            },
        }
    )
