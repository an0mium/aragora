"""
Add decision receipts table for cryptographic audit trails.

Migration created: 2026-02-01

This migration creates the decision_receipts table for:
- Cryptographic proof of debate outcomes (SHA-256)
- Audit trail for compliance (SOC2, GDPR)
- Decision provenance tracking
- Receipt verification support

The table supports the Gauntlet receipts system and enterprise compliance.
"""

import logging

from aragora.migrations.runner import Migration
from aragora.migrations.patterns import safe_create_index, safe_drop_index
from aragora.storage.backends import DatabaseBackend, PostgreSQLBackend

logger = logging.getLogger(__name__)


def _table_exists(backend: DatabaseBackend, table: str) -> bool:
    """Check if a table exists."""
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


def up_fn(backend: DatabaseBackend) -> None:
    """Create decision_receipts table."""
    logger.info("Creating decision_receipts table")

    is_postgres = isinstance(backend, PostgreSQLBackend)

    if not _table_exists(backend, "decision_receipts"):
        if is_postgres:
            backend.execute_write("""
                CREATE TABLE decision_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    debate_id TEXT NOT NULL,
                    gauntlet_id TEXT,
                    receipt_hash TEXT NOT NULL,
                    hash_algorithm TEXT NOT NULL DEFAULT 'sha256',
                    decision_summary TEXT NOT NULL,
                    consensus_type TEXT,
                    confidence_score REAL,
                    participating_agents JSONB NOT NULL DEFAULT '[]',
                    vote_distribution JSONB,
                    input_hash TEXT,
                    output_hash TEXT,
                    chain_hash TEXT,
                    previous_receipt_id TEXT REFERENCES decision_receipts(receipt_id),
                    signed_by TEXT,
                    signature TEXT,
                    verification_status TEXT DEFAULT 'unverified',
                    verified_at TIMESTAMP,
                    verified_by TEXT,
                    issued_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    expires_at TIMESTAMP,
                    revoked_at TIMESTAMP,
                    revocation_reason TEXT,
                    metadata JSONB DEFAULT '{}',
                    workspace_id TEXT,
                    org_id TEXT
                )
            """)
        else:
            backend.execute_write("""
                CREATE TABLE decision_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    debate_id TEXT NOT NULL,
                    gauntlet_id TEXT,
                    receipt_hash TEXT NOT NULL,
                    hash_algorithm TEXT NOT NULL DEFAULT 'sha256',
                    decision_summary TEXT NOT NULL,
                    consensus_type TEXT,
                    confidence_score REAL,
                    participating_agents TEXT NOT NULL DEFAULT '[]',
                    vote_distribution TEXT,
                    input_hash TEXT,
                    output_hash TEXT,
                    chain_hash TEXT,
                    previous_receipt_id TEXT,
                    signed_by TEXT,
                    signature TEXT,
                    verification_status TEXT DEFAULT 'unverified',
                    verified_at TIMESTAMP,
                    verified_by TEXT,
                    issued_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    revoked_at TIMESTAMP,
                    revocation_reason TEXT,
                    metadata TEXT DEFAULT '{}',
                    workspace_id TEXT,
                    org_id TEXT
                )
            """)
        logger.info("Created decision_receipts table")

        # Create indexes for common query patterns
        safe_create_index(
            backend,
            "idx_receipts_debate",
            "decision_receipts",
            ["debate_id"],
            concurrently=True,
        )
        safe_create_index(
            backend,
            "idx_receipts_gauntlet",
            "decision_receipts",
            ["gauntlet_id"],
            concurrently=True,
        )
        safe_create_index(
            backend,
            "idx_receipts_hash",
            "decision_receipts",
            ["receipt_hash"],
            unique=True,
            concurrently=True,
        )
        safe_create_index(
            backend,
            "idx_receipts_chain",
            "decision_receipts",
            ["chain_hash"],
            concurrently=True,
        )
        safe_create_index(
            backend,
            "idx_receipts_issued",
            "decision_receipts",
            ["issued_at"],
            concurrently=True,
        )
        safe_create_index(
            backend,
            "idx_receipts_workspace",
            "decision_receipts",
            ["workspace_id", "issued_at"],
            concurrently=True,
        )
        safe_create_index(
            backend,
            "idx_receipts_verification",
            "decision_receipts",
            ["verification_status"],
            concurrently=True,
        )
        logger.info("Created indexes on decision_receipts")

    logger.info("Migration 20260201000200 applied successfully")


def down_fn(backend: DatabaseBackend) -> None:
    """Drop decision_receipts table."""
    logger.info("Dropping decision_receipts table")

    # Drop indexes first
    safe_drop_index(backend, "idx_receipts_verification", concurrently=True)
    safe_drop_index(backend, "idx_receipts_workspace", concurrently=True)
    safe_drop_index(backend, "idx_receipts_issued", concurrently=True)
    safe_drop_index(backend, "idx_receipts_chain", concurrently=True)
    safe_drop_index(backend, "idx_receipts_hash", concurrently=True)
    safe_drop_index(backend, "idx_receipts_gauntlet", concurrently=True)
    safe_drop_index(backend, "idx_receipts_debate", concurrently=True)

    # Drop table
    backend.execute_write("DROP TABLE IF EXISTS decision_receipts")

    logger.info("Migration 20260201000200 rolled back successfully")


migration = Migration(
    version=20260201000200,
    name="Add decision receipts table",
    up_fn=up_fn,
    down_fn=down_fn,
)
