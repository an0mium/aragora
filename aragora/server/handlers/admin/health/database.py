# mypy: ignore-errors
"""
Database health check implementations.

Provides health checks for:
- Database schema validation (consolidated databases)
- All database stores (debate storage, ELO, insights, etc.)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ...base import HandlerResult, json_response

logger = logging.getLogger(__name__)


def database_schema_health(handler) -> HandlerResult:
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


def database_stores_health(handler) -> HandlerResult:
    """Check health of all database stores.

    Returns detailed status for each database store:
    - debate_storage: Main debate persistence (SQLite/Supabase)
    - elo_system: Agent rankings database
    - insight_store: Debate insights database
    - flip_detector: Flip detection database
    - consensus_memory: Consensus patterns database
    - user_store: User and organization data

    This endpoint helps diagnose which specific stores are
    initialized, connected, and functioning.
    """
    stores: dict[str, dict[str, Any]] = {}
    all_healthy = True
    start_time = time.time()

    # 1. Debate Storage
    try:
        import sqlite3

        storage = handler.get_storage()
        if storage is not None:
            count = len(storage.list_recent(limit=1))
            stores["debate_storage"] = {
                "healthy": True,
                "status": "connected",
                "type": type(storage).__name__,
            }
        else:
            stores["debate_storage"] = {
                "healthy": True,
                "status": "not_initialized",
                "hint": "Will auto-create on first debate",
            }
    except (sqlite3.Error, OSError, IOError) as e:
        logger.warning(f"Debate storage database error: {type(e).__name__}: {e}")
        stores["debate_storage"] = {
            "healthy": False,
            "error": f"{type(e).__name__}: {str(e)[:100]}",
            "error_type": "database",
        }
        all_healthy = False
    except (KeyError, TypeError, AttributeError) as e:
        logger.debug(f"Debate storage data access error: {type(e).__name__}: {e}")
        stores["debate_storage"] = {
            "healthy": False,
            "error": f"{type(e).__name__}: {str(e)[:100]}",
            "error_type": "data_access",
        }
        all_healthy = False
    except Exception as e:
        stores["debate_storage"] = {
            "healthy": False,
            "error": f"{type(e).__name__}: {str(e)[:100]}",
        }
        all_healthy = False

    # 2. ELO System
    try:
        elo = handler.get_elo_system()
        if elo is not None:
            leaderboard = elo.get_leaderboard(limit=5)
            stores["elo_system"] = {
                "healthy": True,
                "status": "connected",
                "agent_count": len(leaderboard),
            }
        else:
            stores["elo_system"] = {
                "healthy": True,
                "status": "not_initialized",
                "hint": "Run: python scripts/seed_agents.py",
            }
    except (sqlite3.Error, OSError, IOError) as e:
        logger.warning(f"ELO system database error: {type(e).__name__}: {e}")
        stores["elo_system"] = {
            "healthy": False,
            "error": f"{type(e).__name__}: {str(e)[:100]}",
            "error_type": "database",
        }
        all_healthy = False
    except (KeyError, TypeError, AttributeError) as e:
        logger.debug(f"ELO system data access error: {type(e).__name__}: {e}")
        stores["elo_system"] = {
            "healthy": False,
            "error": f"{type(e).__name__}: {str(e)[:100]}",
            "error_type": "data_access",
        }
        all_healthy = False
    except Exception as e:
        stores["elo_system"] = {
            "healthy": False,
            "error": f"{type(e).__name__}: {str(e)[:100]}",
        }
        all_healthy = False

    # 3. Insight Store
    try:
        insight_store = handler.ctx.get("insight_store")
        if insight_store is not None:
            stores["insight_store"] = {
                "healthy": True,
                "status": "connected",
                "type": type(insight_store).__name__,
            }
        else:
            stores["insight_store"] = {
                "healthy": True,
                "status": "not_initialized",
                "hint": "Will auto-create on first insight",
            }
    except Exception as e:
        stores["insight_store"] = {
            "healthy": False,
            "error": f"{type(e).__name__}: {str(e)[:100]}",
        }

    # 4. Flip Detector
    try:
        flip_detector = handler.ctx.get("flip_detector")
        if flip_detector is not None:
            stores["flip_detector"] = {
                "healthy": True,
                "status": "connected",
                "type": type(flip_detector).__name__,
            }
        else:
            stores["flip_detector"] = {
                "healthy": True,
                "status": "not_initialized",
            }
    except Exception as e:
        stores["flip_detector"] = {
            "healthy": False,
            "error": f"{type(e).__name__}: {str(e)[:100]}",
        }

    # 5. User Store
    try:
        user_store = handler.ctx.get("user_store")
        if user_store is not None:
            stores["user_store"] = {
                "healthy": True,
                "status": "connected",
                "type": type(user_store).__name__,
            }
        else:
            stores["user_store"] = {
                "healthy": True,
                "status": "not_initialized",
            }
    except Exception as e:
        stores["user_store"] = {
            "healthy": False,
            "error": f"{type(e).__name__}: {str(e)[:100]}",
        }

    # 6. Consensus Memory
    try:
        from aragora.memory.consensus import ConsensusMemory  # noqa: F401

        nomic_dir = handler.get_nomic_dir()
        if nomic_dir is not None:
            consensus_path = nomic_dir / "consensus_memory.db"
            if consensus_path.exists():
                stores["consensus_memory"] = {
                    "healthy": True,
                    "status": "exists",
                    "path": str(consensus_path),
                }
            else:
                stores["consensus_memory"] = {
                    "healthy": True,
                    "status": "not_initialized",
                    "hint": "Run: python scripts/seed_consensus.py",
                }
        else:
            stores["consensus_memory"] = {
                "healthy": True,
                "status": "nomic_dir_not_set",
            }
    except ImportError:
        stores["consensus_memory"] = {
            "healthy": True,
            "status": "module_not_available",
        }
    except Exception as e:
        stores["consensus_memory"] = {
            "healthy": False,
            "error": f"{type(e).__name__}: {str(e)[:100]}",
        }

    # 7. Agent Metadata (from seed script)
    try:
        nomic_dir = handler.get_nomic_dir()
        if nomic_dir is not None:
            elo_path = nomic_dir / "elo.db"
            if elo_path.exists():
                import sqlite3

                conn = sqlite3.connect(elo_path)
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_metadata'"
                )
                has_metadata = cursor.fetchone() is not None
                if has_metadata:
                    cursor = conn.execute("SELECT COUNT(*) FROM agent_metadata")
                    count = cursor.fetchone()[0]
                    stores["agent_metadata"] = {
                        "healthy": True,
                        "status": "connected",
                        "agent_count": count,
                    }
                else:
                    stores["agent_metadata"] = {
                        "healthy": True,
                        "status": "table_not_exists",
                        "hint": "Run: python scripts/seed_agents.py --with-metadata",
                    }
                conn.close()
            else:
                stores["agent_metadata"] = {
                    "healthy": True,
                    "status": "database_not_exists",
                }
        else:
            stores["agent_metadata"] = {
                "healthy": True,
                "status": "nomic_dir_not_set",
            }
    except Exception as e:
        stores["agent_metadata"] = {
            "healthy": False,
            "error": f"{type(e).__name__}: {str(e)[:100]}",
        }

    # 8. Integration Store
    try:
        from aragora.storage.integration_store import IntegrationStore  # noqa: F401

        integration_store = handler.ctx.get("integration_store")
        if integration_store is not None:
            stores["integration_store"] = {
                "healthy": True,
                "status": "connected",
                "type": type(integration_store).__name__,
            }
        else:
            stores["integration_store"] = {
                "healthy": True,
                "status": "not_initialized",
                "hint": "Will auto-create on first integration",
            }
    except ImportError:
        stores["integration_store"] = {
            "healthy": True,
            "status": "module_not_available",
        }
    except Exception as e:
        stores["integration_store"] = {
            "healthy": False,
            "error": f"{type(e).__name__}: {str(e)[:100]}",
        }

    # 9. Gmail Token Store
    try:
        from aragora.storage.gmail_token_store import GmailTokenStore  # noqa: F401

        gmail_token_store = handler.ctx.get("gmail_token_store")
        if gmail_token_store is not None:
            stores["gmail_token_store"] = {
                "healthy": True,
                "status": "connected",
                "type": type(gmail_token_store).__name__,
            }
        else:
            stores["gmail_token_store"] = {
                "healthy": True,
                "status": "not_initialized",
                "hint": "Configure Gmail OAuth to enable",
            }
    except ImportError:
        stores["gmail_token_store"] = {
            "healthy": True,
            "status": "module_not_available",
        }
    except Exception as e:
        stores["gmail_token_store"] = {
            "healthy": False,
            "error": f"{type(e).__name__}: {str(e)[:100]}",
        }

    # 10. Sync Store
    try:
        from aragora.connectors.enterprise.sync_store import SyncStore  # noqa: F401

        sync_store = handler.ctx.get("sync_store")
        if sync_store is not None:
            stores["sync_store"] = {
                "healthy": True,
                "status": "connected",
                "type": type(sync_store).__name__,
            }
        else:
            stores["sync_store"] = {
                "healthy": True,
                "status": "not_initialized",
                "hint": "Enable enterprise sync to initialize",
            }
    except ImportError:
        stores["sync_store"] = {
            "healthy": True,
            "status": "module_not_available",
        }
    except Exception as e:
        stores["sync_store"] = {
            "healthy": False,
            "error": f"{type(e).__name__}: {str(e)[:100]}",
        }

    # 11. Decision Result Store
    try:
        from aragora.storage.decision_result_store import DecisionResultStore  # noqa: F401

        decision_store = handler.ctx.get("decision_result_store")
        if decision_store is not None:
            stores["decision_result_store"] = {
                "healthy": True,
                "status": "connected",
                "type": type(decision_store).__name__,
            }
        else:
            stores["decision_result_store"] = {
                "healthy": True,
                "status": "not_initialized",
                "hint": "Will auto-create on first decision",
            }
    except ImportError:
        stores["decision_result_store"] = {
            "healthy": True,
            "status": "module_not_available",
        }
    except Exception as e:
        stores["decision_result_store"] = {
            "healthy": False,
            "error": f"{type(e).__name__}: {str(e)[:100]}",
        }

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
