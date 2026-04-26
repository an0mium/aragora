from __future__ import annotations

import re

from aragora.migrations.versions import (
    v20260424000000_align_webhook_configs_schema as migration_module,
)


class PostgreSQLBackend:
    """Tiny fake that exercises the migration's PostgreSQL branch."""

    def __init__(self) -> None:
        self.columns = {
            "id",
            "name",
            "url",
            "events",
            "secret",
            "headers",
            "is_active",
            "org_id",
            "created_at",
            "updated_at",
            "failure_count",
            "last_failure_at",
            "last_success_at",
        }
        self.indexes = {"idx_webhook_configs_org"}
        self.statements: list[str] = []

    def fetch_all(self, sql: str, params: tuple = ()) -> list[tuple[str]]:
        if "information_schema.tables" in sql:
            return [("webhook_configs",)] if params == ("webhook_configs",) else []
        if "information_schema.columns" in sql:
            column = params[1]
            return [(column,)] if column in self.columns else []
        return []

    def execute_write(self, sql: str, params: tuple = ()) -> None:
        del params
        if "CONCURRENTLY" in sql:
            raise AssertionError("migration must not use CONCURRENTLY inside execute_write")
        self.statements.append(sql)

        if match := re.search(r"RENAME COLUMN (\w+) TO (\w+)", sql):
            old, new = match.groups()
            self.columns.remove(old)
            self.columns.add(new)
            return

        if match := re.search(r"ADD COLUMN IF NOT EXISTS (\w+)", sql):
            self.columns.add(match.group(1))
            return

        if match := re.search(r"DROP COLUMN IF EXISTS (\w+)", sql):
            self.columns.discard(match.group(1))
            return

        if match := re.search(r'DROP INDEX IF EXISTS "(\w+)"', sql):
            self.indexes.discard(match.group(1))
            return

        if match := re.search(r'CREATE INDEX IF NOT EXISTS "(\w+)"', sql):
            self.indexes.add(match.group(1))


def test_webhook_configs_alignment_migration_avoids_concurrent_indexes(
    monkeypatch,
) -> None:
    monkeypatch.setattr(migration_module, "PostgreSQLBackend", PostgreSQLBackend)
    backend = PostgreSQLBackend()

    migration_module.up_fn(backend)

    assert {
        "workspace_id",
        "active",
        "events_json",
        "user_id",
        "description",
        "last_delivery_at",
        "last_delivery_status",
        "delivery_count",
    } <= backend.columns
    assert (
        not {
            "org_id",
            "is_active",
            "events",
            "headers",
            "last_failure_at",
            "last_success_at",
        }
        & backend.columns
    )
    assert {
        "idx_webhook_configs_user",
        "idx_webhook_configs_workspace",
        "idx_webhook_configs_active",
    } <= backend.indexes
    assert "idx_webhook_configs_org" not in backend.indexes
    assert not any("CONCURRENTLY" in statement for statement in backend.statements)
