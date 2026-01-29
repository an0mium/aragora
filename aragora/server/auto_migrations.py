"""
Auto-migration on server startup.

Orchestrates database migrations when ARAGORA_AUTO_MIGRATE_ON_STARTUP=true.
Supports both PostgreSQL (aragora.migrations) and SQLite (aragora.persistence.migrations).

Safety:
- Migrations only run when explicitly enabled via environment variable
- Backup is created before migration (if backup manager available)
- Failures are logged but don't prevent server startup (graceful degradation)
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


async def run_auto_migrations() -> dict[str, Any]:
    """
    Run pending database migrations if enabled.

    Returns:
        dict with migration results:
        - skipped: True if migrations were skipped
        - reason: Why migrations were skipped (if applicable)
        - postgresql: PostgreSQL migration results
        - sqlite: SQLite migration results
        - success: Overall success status
    """
    # Check if auto-migration is enabled
    if not os.environ.get("ARAGORA_AUTO_MIGRATE_ON_STARTUP", "").lower() == "true":
        return {"skipped": True, "reason": "ARAGORA_AUTO_MIGRATE_ON_STARTUP not enabled"}

    logger.info("Auto-migration enabled, checking for pending migrations...")

    results: dict[str, Any] = {
        "skipped": False,
        "postgresql": None,
        "sqlite": None,
        "success": True,
    }

    # Run PostgreSQL migrations (aragora.migrations)
    results["postgresql"] = await _run_postgresql_migrations()
    if results["postgresql"].get("error"):
        results["success"] = False

    # Run SQLite migrations (aragora.persistence.migrations)
    results["sqlite"] = await _run_sqlite_migrations()
    if results["sqlite"].get("error"):
        results["success"] = False

    return results


async def _run_postgresql_migrations() -> dict[str, Any]:
    """Run PostgreSQL migrations using aragora.migrations.runner."""
    try:
        from aragora.migrations.runner import get_migration_runner

        runner = get_migration_runner()
        pending = runner.get_pending_migrations()

        if not pending:
            logger.debug("No pending PostgreSQL migrations")
            return {"applied": 0, "message": "No pending migrations"}

        logger.info(f"Running {len(pending)} pending PostgreSQL migrations...")
        applied = runner.upgrade()

        logger.info(f"Successfully applied {len(applied)} PostgreSQL migrations")
        return {
            "applied": len(applied),
            "versions": [m.version for m in applied],
        }

    except ImportError as e:
        logger.debug(f"PostgreSQL migrations not available: {e}")
        return {"skipped": True, "reason": "PostgreSQL not configured"}
    except Exception as e:
        logger.error(f"PostgreSQL migration failed: {e}")
        return {"error": str(e)}


async def _run_sqlite_migrations() -> dict[str, Any]:
    """Run SQLite migrations using aragora.persistence.migrations.runner."""
    try:
        from aragora.persistence.migrations.runner import MigrationRunner

        runner = MigrationRunner()

        # Get status for all databases
        statuses = runner.get_all_status()

        total_pending = 0
        for status in statuses.values():
            total_pending += len(status.pending)

        if total_pending == 0:
            logger.debug("No pending SQLite migrations")
            return {"applied": 0, "message": "No pending migrations"}

        logger.info(f"Running {total_pending} pending SQLite migrations...")

        # Run migrations (creates backup automatically)
        result = runner.migrate_all(dry_run=False)

        applied_count = sum(len(db_result.get("applied", [])) for db_result in result.values())

        logger.info(f"Successfully applied {applied_count} SQLite migrations")
        return {
            "applied": applied_count,
            "databases": list(result.keys()),
        }

    except ImportError as e:
        logger.debug(f"SQLite migrations not available: {e}")
        return {"skipped": True, "reason": "SQLite migrations not configured"}
    except Exception as e:
        logger.error(f"SQLite migration failed: {e}")
        return {"error": str(e)}


def check_migrations_pending() -> dict[str, Any]:
    """
    Check if there are pending migrations without running them.

    Returns:
        dict with pending migration counts for each backend.
    """
    result: dict[str, Any] = {
        "postgresql_pending": 0,
        "sqlite_pending": 0,
        "total_pending": 0,
    }

    try:
        from aragora.migrations.runner import get_migration_runner

        runner = get_migration_runner()
        pending = runner.get_pending_migrations()
        result["postgresql_pending"] = len(pending)
    except Exception:
        pass

    try:
        from aragora.persistence.migrations.runner import MigrationRunner

        runner = MigrationRunner()
        statuses = runner.get_all_status()
        for status in statuses.values():
            result["sqlite_pending"] += len(status.pending)
    except Exception:
        pass

    result["total_pending"] = result["postgresql_pending"] + result["sqlite_pending"]
    return result
