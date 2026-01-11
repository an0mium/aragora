"""
Initial schema migration.

This migration documents the base schema that already exists in Aragora.
It creates the core tables if they don't exist (idempotent).
"""

from aragora.migrations.runner import Migration

migration = Migration(
    version=20240101000000,
    name="Initial schema",
    up_sql="""
        -- Gauntlet results table
        CREATE TABLE IF NOT EXISTS gauntlet_results (
            gauntlet_id TEXT PRIMARY KEY,
            input_hash TEXT NOT NULL,
            input_summary TEXT,
            result_json TEXT NOT NULL,
            verdict TEXT NOT NULL,
            confidence REAL,
            robustness_score REAL,
            critical_count INTEGER DEFAULT 0,
            high_count INTEGER DEFAULT 0,
            medium_count INTEGER DEFAULT 0,
            low_count INTEGER DEFAULT 0,
            total_findings INTEGER DEFAULT 0,
            agents_used TEXT,
            template_used TEXT,
            duration_seconds REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            org_id TEXT
        );

        -- Indexes for gauntlet_results
        CREATE INDEX IF NOT EXISTS idx_gauntlet_input_hash ON gauntlet_results(input_hash);
        CREATE INDEX IF NOT EXISTS idx_gauntlet_created ON gauntlet_results(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_gauntlet_verdict ON gauntlet_results(verdict);
        CREATE INDEX IF NOT EXISTS idx_gauntlet_org ON gauntlet_results(org_id, created_at DESC)
    """,
    down_sql="""
        -- WARNING: This will delete all gauntlet data!
        DROP INDEX IF EXISTS idx_gauntlet_org;
        DROP INDEX IF EXISTS idx_gauntlet_verdict;
        DROP INDEX IF EXISTS idx_gauntlet_created;
        DROP INDEX IF EXISTS idx_gauntlet_input_hash;
        DROP TABLE IF EXISTS gauntlet_results
    """,
)
