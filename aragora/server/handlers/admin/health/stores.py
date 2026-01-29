# mypy: ignore-errors
"""Database stores health check mixin.

Provides health checking for all database stores in the system.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ...base import HandlerResult, json_response

logger = logging.getLogger(__name__)

class StoresMixin:
    """Mixin for database store health checks.

    Provides:
    - database_stores_health(): Health check for all database stores

    Expects the class to have:
    - self.get_storage(): Get debate storage
    - self.get_elo_system(): Get ELO ranking system
    - self.get_nomic_dir(): Get nomic directory path
    - self.ctx: Context dict for accessing stores
    """

    def get_storage(self) -> Any:
        """Get debate storage. Subclasses must implement."""
        raise NotImplementedError

    def get_elo_system(self) -> Any:
        """Get ELO ranking system. Subclasses must implement."""
        raise NotImplementedError

    def get_nomic_dir(self) -> Any:
        """Get nomic directory path. Subclasses must implement."""
        raise NotImplementedError

    ctx: dict[str, Any]  # Context dict for accessing stores

    def database_stores_health(self) -> HandlerResult:
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
        stores["debate_storage"], healthy = self._check_debate_storage()
        if not healthy:
            all_healthy = False

        # 2. ELO System
        stores["elo_system"], healthy = self._check_elo_system()
        if not healthy:
            all_healthy = False

        # 3. Insight Store
        stores["insight_store"] = self._check_insight_store()

        # 4. Flip Detector
        stores["flip_detector"] = self._check_flip_detector()

        # 5. User Store
        stores["user_store"] = self._check_user_store()

        # 6. Consensus Memory
        stores["consensus_memory"] = self._check_consensus_memory()

        # 7. Agent Metadata
        stores["agent_metadata"] = self._check_agent_metadata()

        # 8. Integration Store
        stores["integration_store"] = self._check_integration_store()

        # 9. Gmail Token Store
        stores["gmail_token_store"] = self._check_gmail_token_store()

        # 10. Sync Store
        stores["sync_store"] = self._check_sync_store()

        # 11. Decision Result Store
        stores["decision_result_store"] = self._check_decision_result_store()

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

    def _check_debate_storage(self) -> tuple[dict[str, Any], bool]:
        """Check debate storage health."""
        try:
            import sqlite3

            storage = self.get_storage()
            if storage is not None:
                storage.list_recent(limit=1)
                return {
                    "healthy": True,
                    "status": "connected",
                    "type": type(storage).__name__,
                }, True
            else:
                return {
                    "healthy": True,
                    "status": "not_initialized",
                    "hint": "Will auto-create on first debate",
                }, True
        except (sqlite3.Error, OSError, IOError) as e:
            logger.warning(f"Debate storage database error: {type(e).__name__}: {e}")
            return {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
                "error_type": "database",
            }, False
        except (KeyError, TypeError, AttributeError) as e:
            logger.debug(f"Debate storage data access error: {type(e).__name__}: {e}")
            return {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
                "error_type": "data_access",
            }, False
        except Exception as e:
            return {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
            }, False

    def _check_elo_system(self) -> tuple[dict[str, Any], bool]:
        """Check ELO system health."""
        try:
            import sqlite3

            elo = self.get_elo_system()
            if elo is not None:
                leaderboard = elo.get_leaderboard(limit=5)
                return {
                    "healthy": True,
                    "status": "connected",
                    "agent_count": len(leaderboard),
                }, True
            else:
                return {
                    "healthy": True,
                    "status": "not_initialized",
                    "hint": "Run: python scripts/seed_agents.py",
                }, True
        except (sqlite3.Error, OSError, IOError) as e:
            logger.warning(f"ELO system database error: {type(e).__name__}: {e}")
            return {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
                "error_type": "database",
            }, False
        except (KeyError, TypeError, AttributeError) as e:
            logger.debug(f"ELO system data access error: {type(e).__name__}: {e}")
            return {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
                "error_type": "data_access",
            }, False
        except Exception as e:
            return {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
            }, False

    def _check_insight_store(self) -> dict[str, Any]:
        """Check insight store health."""
        try:
            insight_store = self.ctx.get("insight_store")
            if insight_store is not None:
                return {
                    "healthy": True,
                    "status": "connected",
                    "type": type(insight_store).__name__,
                }
            else:
                return {
                    "healthy": True,
                    "status": "not_initialized",
                    "hint": "Will auto-create on first insight",
                }
        except Exception as e:
            return {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
            }

    def _check_flip_detector(self) -> dict[str, Any]:
        """Check flip detector health."""
        try:
            flip_detector = self.ctx.get("flip_detector")
            if flip_detector is not None:
                return {
                    "healthy": True,
                    "status": "connected",
                    "type": type(flip_detector).__name__,
                }
            else:
                return {
                    "healthy": True,
                    "status": "not_initialized",
                }
        except Exception as e:
            return {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
            }

    def _check_user_store(self) -> dict[str, Any]:
        """Check user store health."""
        try:
            user_store = self.ctx.get("user_store")
            if user_store is not None:
                return {
                    "healthy": True,
                    "status": "connected",
                    "type": type(user_store).__name__,
                }
            else:
                return {
                    "healthy": True,
                    "status": "not_initialized",
                }
        except Exception as e:
            return {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
            }

    def _check_consensus_memory(self) -> dict[str, Any]:
        """Check consensus memory health."""
        try:
            from aragora.memory.consensus import ConsensusMemory  # noqa: F401

            nomic_dir = self.get_nomic_dir()
            if nomic_dir is not None:
                consensus_path = nomic_dir / "consensus_memory.db"
                if consensus_path.exists():
                    return {
                        "healthy": True,
                        "status": "exists",
                        "path": str(consensus_path),
                    }
                else:
                    return {
                        "healthy": True,
                        "status": "not_initialized",
                        "hint": "Run: python scripts/seed_consensus.py",
                    }
            else:
                return {
                    "healthy": True,
                    "status": "nomic_dir_not_set",
                }
        except ImportError:
            return {
                "healthy": True,
                "status": "module_not_available",
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
            }

    def _check_agent_metadata(self) -> dict[str, Any]:
        """Check agent metadata health."""
        try:
            nomic_dir = self.get_nomic_dir()
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
                        conn.close()
                        return {
                            "healthy": True,
                            "status": "connected",
                            "agent_count": count,
                        }
                    else:
                        conn.close()
                        return {
                            "healthy": True,
                            "status": "table_not_exists",
                            "hint": "Run: python scripts/seed_agents.py --with-metadata",
                        }
                else:
                    return {
                        "healthy": True,
                        "status": "database_not_exists",
                    }
            else:
                return {
                    "healthy": True,
                    "status": "nomic_dir_not_set",
                }
        except Exception as e:
            return {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
            }

    def _check_integration_store(self) -> dict[str, Any]:
        """Check integration store health."""
        try:
            from aragora.storage.integration_store import IntegrationStore  # noqa: F401

            integration_store = self.ctx.get("integration_store")
            if integration_store is not None:
                return {
                    "healthy": True,
                    "status": "connected",
                    "type": type(integration_store).__name__,
                }
            else:
                return {
                    "healthy": True,
                    "status": "not_initialized",
                    "hint": "Will auto-create on first integration",
                }
        except ImportError:
            return {
                "healthy": True,
                "status": "module_not_available",
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
            }

    def _check_gmail_token_store(self) -> dict[str, Any]:
        """Check Gmail token store health."""
        try:
            from aragora.storage.gmail_token_store import GmailTokenStore  # noqa: F401

            gmail_token_store = self.ctx.get("gmail_token_store")
            if gmail_token_store is not None:
                return {
                    "healthy": True,
                    "status": "connected",
                    "type": type(gmail_token_store).__name__,
                }
            else:
                return {
                    "healthy": True,
                    "status": "not_initialized",
                    "hint": "Configure Gmail OAuth to enable",
                }
        except ImportError:
            return {
                "healthy": True,
                "status": "module_not_available",
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
            }

    def _check_sync_store(self) -> dict[str, Any]:
        """Check sync store health."""
        try:
            from aragora.connectors.enterprise.sync_store import SyncStore  # noqa: F401

            sync_store = self.ctx.get("sync_store")
            if sync_store is not None:
                return {
                    "healthy": True,
                    "status": "connected",
                    "type": type(sync_store).__name__,
                }
            else:
                return {
                    "healthy": True,
                    "status": "not_initialized",
                    "hint": "Enable enterprise sync to initialize",
                }
        except ImportError:
            return {
                "healthy": True,
                "status": "module_not_available",
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
            }

    def _check_decision_result_store(self) -> dict[str, Any]:
        """Check decision result store health."""
        try:
            from aragora.storage.decision_result_store import DecisionResultStore  # noqa: F401

            decision_store = self.ctx.get("decision_result_store")
            if decision_store is not None:
                return {
                    "healthy": True,
                    "status": "connected",
                    "type": type(decision_store).__name__,
                }
            else:
                return {
                    "healthy": True,
                    "status": "not_initialized",
                    "hint": "Will auto-create on first decision",
                }
        except ImportError:
            return {
                "healthy": True,
                "status": "module_not_available",
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
            }

__all__ = ["StoresMixin"]
