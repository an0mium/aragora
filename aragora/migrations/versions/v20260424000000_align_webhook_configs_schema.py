"""
Align webhook_configs schema with PostgresWebhookConfigStore.INITIAL_SCHEMA.

Migration created: 2026-04-24

Background
----------
The canonical schema in ``aragora/db/schema/postgres_schema.sql`` had a
``webhook_configs`` table definition that drifted from
``PostgresWebhookConfigStore.INITIAL_SCHEMA`` in
``aragora/storage/webhook_config_store.py``. The store is the source of
truth; the schema file had the old shape (``org_id``, ``is_active``,
``events JSONB``, ``headers JSONB``, ``last_failure_at``, ``last_success_at``),
while the store uses ``user_id``, ``workspace_id``, ``active``, ``events_json``,
``description``, ``last_delivery_at``, ``last_delivery_status``,
``delivery_count``.

When integration tests ran ``psql -f postgres_schema.sql`` first and then
``store.initialize()`` executed, the store's ``CREATE TABLE IF NOT EXISTS``
was a noop against the stale table, and the subsequent
``CREATE INDEX ... ON webhook_configs(user_id)`` failed with
``UndefinedColumnError``.

This migration brings existing production databases up to the canonical
shape without data loss for rows already written with the old schema. It
is idempotent.

Column mapping
--------------
- ``org_id`` → ``workspace_id`` (rename; same concept in the new model)
- ``is_active`` → ``active``       (rename)
- ``events``   → ``events_json``   (rename; same JSONB type)
- ``headers``  → dropped           (no longer tracked by the store)
- ``last_failure_at`` → dropped    (replaced by ``last_delivery_at`` + ``last_delivery_status``)
- ``last_success_at`` → dropped    (replaced by ``last_delivery_at`` + ``last_delivery_status``)

New columns added: ``user_id``, ``description``, ``last_delivery_at``,
``last_delivery_status``, ``delivery_count``.

Zero-Downtime Strategy
----------------------
- RENAME COLUMN takes a brief exclusive lock but finishes instantly
  (PostgreSQL only updates the catalog, no table rewrite).
- ADD COLUMN with a default is fast on modern Postgres (11+).
- DROP COLUMN is metadata-only until next VACUUM.
- CREATE INDEX intentionally avoids CONCURRENTLY because the current
  ``PostgreSQLBackend.execute_write`` helper wraps each statement in a
  transaction, and PostgreSQL rejects ``CREATE INDEX CONCURRENTLY`` inside
  transaction blocks.
"""

import logging

from aragora.migrations.patterns import safe_create_index, safe_drop_index
from aragora.migrations.runner import Migration
from aragora.storage.backends import DatabaseBackend, PostgreSQLBackend

logger = logging.getLogger(__name__)


def _table_exists(backend: DatabaseBackend, table: str) -> bool:
    try:
        if isinstance(backend, PostgreSQLBackend):
            rows = backend.fetch_all(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_name = %s
                """,
                (table,),
            )
        else:
            rows = backend.fetch_all(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
        return len(rows) > 0
    except Exception:  # noqa: BLE001
        return False


def _column_exists(backend: DatabaseBackend, table: str, column: str) -> bool:
    try:
        if isinstance(backend, PostgreSQLBackend):
            rows = backend.fetch_all(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name = %s AND column_name = %s
                """,
                (table, column),
            )
        else:
            rows = backend.fetch_all(f"PRAGMA table_info({table})")
            return any(r[1] == column for r in rows)
        return len(rows) > 0
    except Exception:  # noqa: BLE001
        return False


def up_fn(backend: DatabaseBackend) -> None:
    """Align webhook_configs to the canonical shape."""
    if not _table_exists(backend, "webhook_configs"):
        logger.info("webhook_configs table not present; nothing to migrate")
        return

    is_postgres = isinstance(backend, PostgreSQLBackend)

    if is_postgres:
        # --------------------------------------------------------------
        # Rename legacy columns (preserves data)
        # --------------------------------------------------------------
        if _column_exists(backend, "webhook_configs", "org_id") and not _column_exists(
            backend, "webhook_configs", "workspace_id"
        ):
            backend.execute_write(
                "ALTER TABLE webhook_configs RENAME COLUMN org_id TO workspace_id"
            )
            logger.info("Renamed webhook_configs.org_id -> workspace_id")

        if _column_exists(backend, "webhook_configs", "is_active") and not _column_exists(
            backend, "webhook_configs", "active"
        ):
            backend.execute_write("ALTER TABLE webhook_configs RENAME COLUMN is_active TO active")
            logger.info("Renamed webhook_configs.is_active -> active")

        if _column_exists(backend, "webhook_configs", "events") and not _column_exists(
            backend, "webhook_configs", "events_json"
        ):
            backend.execute_write("ALTER TABLE webhook_configs RENAME COLUMN events TO events_json")
            logger.info("Renamed webhook_configs.events -> events_json")

        # --------------------------------------------------------------
        # Add new columns (idempotent)
        # --------------------------------------------------------------
        for column, ddl in (
            ("user_id", "TEXT"),
            ("description", "TEXT"),
            ("last_delivery_at", "TIMESTAMPTZ"),
            ("last_delivery_status", "INTEGER"),
            ("delivery_count", "INTEGER DEFAULT 0"),
        ):
            backend.execute_write(
                f"ALTER TABLE webhook_configs ADD COLUMN IF NOT EXISTS {column} {ddl}"
            )
            logger.debug("Ensured webhook_configs.%s exists (%s)", column, ddl)

        # --------------------------------------------------------------
        # Drop removed columns (idempotent)
        # --------------------------------------------------------------
        # Drop any indexes on those columns first so DROP COLUMN succeeds.
        safe_drop_index(backend, "idx_webhook_configs_org", concurrently=False)
        for column in ("headers", "last_failure_at", "last_success_at"):
            backend.execute_write(f"ALTER TABLE webhook_configs DROP COLUMN IF EXISTS {column}")
            logger.debug("Ensured webhook_configs.%s is dropped", column)

        # --------------------------------------------------------------
        # Indexes on new columns (idempotent)
        # --------------------------------------------------------------
        safe_create_index(
            backend,
            "idx_webhook_configs_user",
            "webhook_configs",
            ["user_id"],
            concurrently=False,
        )
        safe_create_index(
            backend,
            "idx_webhook_configs_workspace",
            "webhook_configs",
            ["workspace_id"],
            concurrently=False,
        )
        safe_create_index(
            backend,
            "idx_webhook_configs_active",
            "webhook_configs",
            ["active"],
            concurrently=False,
        )

        logger.info("webhook_configs schema aligned with INITIAL_SCHEMA")
    else:
        # SQLite: PostgresWebhookConfigStore is not used for SQLite backends;
        # the SQLite WebhookConfigStore manages its own table. Nothing to do.
        logger.info("SQLite backend: webhook_configs is managed by the sync store; skipping")


def down_fn(backend: DatabaseBackend) -> None:
    """Revert to the legacy webhook_configs shape.

    Note: rollback re-creates the old shape but cannot recover data from
    columns that were dropped (``headers``, ``last_failure_at``,
    ``last_success_at``). Use only for schema compatibility emergencies.
    """
    if not _table_exists(backend, "webhook_configs"):
        return

    is_postgres = isinstance(backend, PostgreSQLBackend)
    if not is_postgres:
        logger.info("SQLite backend: nothing to roll back")
        return

    # Drop new indexes
    safe_drop_index(backend, "idx_webhook_configs_user", concurrently=False)
    safe_drop_index(backend, "idx_webhook_configs_workspace", concurrently=False)
    safe_drop_index(backend, "idx_webhook_configs_active", concurrently=False)

    # Drop columns we added (lossy for user_id / description / delivery_* data)
    for column in (
        "user_id",
        "description",
        "last_delivery_at",
        "last_delivery_status",
        "delivery_count",
    ):
        backend.execute_write(f"ALTER TABLE webhook_configs DROP COLUMN IF EXISTS {column}")

    # Rename back to legacy column names if they exist in new form
    if _column_exists(backend, "webhook_configs", "workspace_id") and not _column_exists(
        backend, "webhook_configs", "org_id"
    ):
        backend.execute_write("ALTER TABLE webhook_configs RENAME COLUMN workspace_id TO org_id")
    if _column_exists(backend, "webhook_configs", "active") and not _column_exists(
        backend, "webhook_configs", "is_active"
    ):
        backend.execute_write("ALTER TABLE webhook_configs RENAME COLUMN active TO is_active")
    if _column_exists(backend, "webhook_configs", "events_json") and not _column_exists(
        backend, "webhook_configs", "events"
    ):
        backend.execute_write("ALTER TABLE webhook_configs RENAME COLUMN events_json TO events")

    # Re-add legacy columns (without data)
    backend.execute_write(
        "ALTER TABLE webhook_configs ADD COLUMN IF NOT EXISTS headers JSONB DEFAULT '{}'"
    )
    backend.execute_write(
        "ALTER TABLE webhook_configs ADD COLUMN IF NOT EXISTS last_failure_at TIMESTAMPTZ"
    )
    backend.execute_write(
        "ALTER TABLE webhook_configs ADD COLUMN IF NOT EXISTS last_success_at TIMESTAMPTZ"
    )

    # Restore legacy index
    safe_create_index(
        backend,
        "idx_webhook_configs_org",
        "webhook_configs",
        ["org_id"],
        concurrently=False,
    )


migration = Migration(
    version=20260424000000,
    name="Align webhook_configs schema with INITIAL_SCHEMA",
    up_fn=up_fn,
    down_fn=down_fn,
)
