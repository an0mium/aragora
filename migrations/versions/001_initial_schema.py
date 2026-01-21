"""Initial Aragora PostgreSQL schema.

Revision ID: 001_initial
Revises:
Create Date: 2026-01-20

This migration establishes the baseline schema for Aragora's PostgreSQL backend.
The schema is defined in migrations/sql/001_initial_schema.sql and executed directly.

All stores use this unified schema:
- webhook_configs
- integrations, user_id_mappings
- gmail_tokens
- finding_workflows
- gauntlet_runs
- job_queue
- governance_artifacts
- marketplace_items
- federation_nodes
- approval_requests
- token_blacklist
- users
- webhooks (legacy)
"""
from pathlib import Path
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Apply initial schema from SQL file."""
    # Load and execute the SQL schema file
    schema_path = Path(__file__).parent.parent / "sql" / "001_initial_schema.sql"
    if schema_path.exists():
        sql = schema_path.read_text()
        # Split by semicolons and execute each statement
        # (Some statements like CREATE FUNCTION span multiple lines)
        statements = []
        current = []
        in_function = False

        for line in sql.split("\n"):
            stripped = line.strip()
            # Skip comments
            if stripped.startswith("--"):
                continue

            current.append(line)

            # Track function blocks (which contain semicolons)
            if "CREATE OR REPLACE FUNCTION" in line or "CREATE FUNCTION" in line:
                in_function = True
            if in_function and stripped.endswith("$$ LANGUAGE plpgsql;"):
                in_function = False
                statements.append("\n".join(current))
                current = []
            elif not in_function and stripped.endswith(";"):
                statements.append("\n".join(current))
                current = []

        # Execute each statement
        for stmt in statements:
            stmt = stmt.strip()
            if stmt and not stmt.startswith("--"):
                op.execute(stmt)
    else:
        # Fallback: Create minimal required tables
        op.execute("""
            CREATE SCHEMA IF NOT EXISTS aragora;
            SET search_path TO aragora, public;

            CREATE TABLE IF NOT EXISTS _schema_versions (
                module TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );

            INSERT INTO _schema_versions (module, version, updated_at)
            VALUES ('aragora_core', 1, NOW())
            ON CONFLICT (module) DO UPDATE SET version = 1, updated_at = NOW();
        """)


def downgrade() -> None:
    """Drop all Aragora tables.

    WARNING: This will delete all data!
    """
    op.execute("""
        DROP SCHEMA IF EXISTS aragora CASCADE;
    """)
