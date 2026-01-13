"""
Database migration utilities.

Provides tools for:
- Creating PostgreSQL schema from SQLite
- Migrating data from SQLite to PostgreSQL
- Verifying migration integrity
"""

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# PostgreSQL schema - converted from SQLite
POSTGRES_SCHEMA = """
-- Debates table
CREATE TABLE IF NOT EXISTS debates (
    id TEXT PRIMARY KEY,
    slug TEXT UNIQUE NOT NULL,
    task TEXT NOT NULL,
    agents JSONB NOT NULL,
    artifact_json JSONB NOT NULL,
    consensus_reached BOOLEAN,
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    view_count INTEGER DEFAULT 0,
    audio_path TEXT,
    audio_generated_at TIMESTAMP,
    audio_duration_seconds INTEGER,
    org_id TEXT,
    is_public BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_debates_slug ON debates(slug);
CREATE INDEX IF NOT EXISTS idx_debates_created ON debates(created_at);
CREATE INDEX IF NOT EXISTS idx_debates_org ON debates(org_id, created_at);

-- Schema versions table
CREATE TABLE IF NOT EXISTS _schema_versions (
    module TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Share settings table
CREATE TABLE IF NOT EXISTS share_settings (
    id TEXT PRIMARY KEY,
    debate_id TEXT NOT NULL REFERENCES debates(id) ON DELETE CASCADE,
    token TEXT UNIQUE NOT NULL,
    visibility TEXT NOT NULL DEFAULT 'authenticated',
    expires_at TIMESTAMP,
    password_hash TEXT,
    allowed_emails TEXT[],
    max_views INTEGER,
    current_views INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_public BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_share_settings_token ON share_settings(token);
CREATE INDEX IF NOT EXISTS idx_share_settings_debate ON share_settings(debate_id);

-- ELO rankings table
CREATE TABLE IF NOT EXISTS elo_rankings (
    agent_name TEXT PRIMARY KEY,
    elo REAL NOT NULL DEFAULT 1500,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    draws INTEGER DEFAULT 0,
    matches_played INTEGER DEFAULT 0,
    last_match_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Match history table
CREATE TABLE IF NOT EXISTS matches (
    id SERIAL PRIMARY KEY,
    debate_id TEXT,
    agent_name TEXT NOT NULL,
    opponent_name TEXT NOT NULL,
    result TEXT NOT NULL,  -- 'win', 'loss', 'draw'
    elo_change REAL,
    new_elo REAL,
    domain TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_matches_agent ON matches(agent_name);
CREATE INDEX IF NOT EXISTS idx_matches_timestamp ON matches(created_at);

-- Memory store table
CREATE TABLE IF NOT EXISTS memory_store (
    id SERIAL PRIMARY KEY,
    agent_name TEXT NOT NULL,
    debate_id TEXT,
    content TEXT NOT NULL,
    importance REAL DEFAULT 0.5,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_memory_agent_debate ON memory_store(agent_name, debate_id);
CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory_store(timestamp);

-- Continuum memory table (multi-tier)
CREATE TABLE IF NOT EXISTS continuum_memory (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    tier TEXT NOT NULL,  -- 'fast', 'medium', 'slow', 'glacial'
    importance REAL DEFAULT 0.5,
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_continuum_timestamp ON continuum_memory(created_at);
CREATE INDEX IF NOT EXISTS idx_continuum_tier ON continuum_memory(tier);

-- Votes table
CREATE TABLE IF NOT EXISTS votes (
    id SERIAL PRIMARY KEY,
    debate_id TEXT NOT NULL,
    round_num INTEGER NOT NULL,
    agent_name TEXT NOT NULL,
    vote_value TEXT,
    confidence REAL,
    reasoning TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_votes_agent_debate ON votes(agent_name, debate_id);
CREATE INDEX IF NOT EXISTS idx_votes_debate_round ON votes(debate_id, round_num);

-- Consensus memory table
CREATE TABLE IF NOT EXISTS consensus_memory (
    id TEXT PRIMARY KEY,
    debate_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    conclusion TEXT NOT NULL,
    confidence REAL,
    agents TEXT[],
    evidence JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_consensus_debate ON consensus_memory(debate_id);

-- Tournaments table
CREATE TABLE IF NOT EXISTS tournaments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT DEFAULT 'pending',  -- 'pending', 'active', 'completed'
    participants TEXT[],
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- Tournament matches table
CREATE TABLE IF NOT EXISTS tournament_matches (
    id SERIAL PRIMARY KEY,
    tournament_id TEXT NOT NULL REFERENCES tournaments(id) ON DELETE CASCADE,
    debate_id TEXT,
    round_num INTEGER NOT NULL,
    agent_a TEXT NOT NULL,
    agent_b TEXT NOT NULL,
    winner TEXT,
    score_a REAL,
    score_b REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tournament_matches ON tournament_matches(tournament_id, round_num);

-- Plugins table
CREATE TABLE IF NOT EXISTS plugins (
    name TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    description TEXT,
    author TEXT,
    capabilities TEXT[],
    requirements TEXT[],
    entry_point TEXT NOT NULL,
    timeout_seconds INTEGER DEFAULT 60,
    max_memory_mb INTEGER DEFAULT 512,
    python_packages TEXT[],
    system_tools TEXT[],
    license TEXT,
    homepage TEXT,
    tags TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    enabled BOOLEAN DEFAULT TRUE
);

-- Pulse topics table (trending)
CREATE TABLE IF NOT EXISTS pulse_topics (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    source TEXT NOT NULL,  -- 'twitter', 'hackernews', 'reddit', etc.
    score REAL DEFAULT 0,
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    debate_count INTEGER DEFAULT 0,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_pulse_topics_score ON pulse_topics(score DESC);
CREATE INDEX IF NOT EXISTS idx_pulse_topics_source ON pulse_topics(source);
"""


def get_sqlite_tables(conn: sqlite3.Connection) -> list[str]:
    """Get list of tables in SQLite database."""
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    return [row[0] for row in cursor.fetchall()]


def get_table_columns(conn: sqlite3.Connection, table: str) -> list[tuple[str, str]]:
    """Get column names and types for a table."""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return [(row[1], row[2]) for row in cursor.fetchall()]


def get_row_count(conn: sqlite3.Connection, table: str) -> int:
    """Get number of rows in a table."""
    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
    return cursor.fetchone()[0]


def migrate_table(
    sqlite_conn: sqlite3.Connection,
    pg_conn: Any,
    table: str,
    batch_size: int = 1000,
) -> int:
    """Migrate a single table from SQLite to PostgreSQL.

    Args:
        sqlite_conn: SQLite connection
        pg_conn: PostgreSQL connection
        table: Table name to migrate
        batch_size: Number of rows to insert at once

    Returns:
        Number of rows migrated
    """
    columns = get_table_columns(sqlite_conn, table)
    column_names = [c[0] for c in columns]

    # Build INSERT statement
    placeholders = ", ".join(["%s"] * len(columns))
    insert_sql = f"INSERT INTO {table} ({', '.join(column_names)}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"

    # Fetch and insert in batches
    cursor = sqlite_conn.execute(f"SELECT * FROM {table}")
    total_migrated = 0

    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break

        # Convert rows for PostgreSQL
        converted_rows = []
        for row in rows:
            converted_row = []
            for i, value in enumerate(row):
                col_type = columns[i][1].upper()
                # Convert JSON strings to JSONB
                if col_type in ("TEXT", "JSON", "JSONB") and isinstance(value, str):
                    if value.startswith("{") or value.startswith("["):
                        try:
                            # Validate it's JSON
                            json.loads(value)
                            # Keep as string, PostgreSQL will cast
                        except json.JSONDecodeError:
                            pass
                converted_row.append(value)
            converted_rows.append(tuple(converted_row))

        # Insert batch
        pg_cursor = pg_conn.cursor()
        pg_cursor.executemany(insert_sql, converted_rows)
        total_migrated += len(converted_rows)

    return total_migrated


def migrate_sqlite_to_postgres(
    sqlite_path: str,
    pg_host: str = "localhost",
    pg_port: int = 5432,
    pg_database: str = "aragora",
    pg_user: str = "aragora",
    pg_password: str = "",
    dry_run: bool = False,
) -> dict[str, int]:
    """Migrate all data from SQLite to PostgreSQL.

    Args:
        sqlite_path: Path to SQLite database
        pg_host: PostgreSQL host
        pg_port: PostgreSQL port
        pg_database: PostgreSQL database name
        pg_user: PostgreSQL user
        pg_password: PostgreSQL password
        dry_run: If True, don't actually migrate data

    Returns:
        Dict mapping table names to rows migrated
    """
    try:
        import psycopg2
    except ImportError:
        logger.error("psycopg2 not installed. Run: pip install psycopg2-binary")
        return {}

    # Connect to databases
    sqlite_conn = sqlite3.connect(sqlite_path)
    pg_conn = psycopg2.connect(
        host=pg_host,
        port=pg_port,
        dbname=pg_database,
        user=pg_user,
        password=pg_password,
    )

    results = {}

    try:
        # Create schema in PostgreSQL
        if not dry_run:
            pg_cursor = pg_conn.cursor()
            pg_cursor.execute(POSTGRES_SCHEMA)
            pg_conn.commit()
            logger.info("PostgreSQL schema created")

        # Get tables to migrate
        tables = get_sqlite_tables(sqlite_conn)
        logger.info(f"Found {len(tables)} tables to migrate: {tables}")

        # Migrate each table
        for table in tables:
            row_count = get_row_count(sqlite_conn, table)
            logger.info(f"Migrating {table}: {row_count} rows")

            if dry_run:
                results[table] = row_count
            else:
                try:
                    migrated = migrate_table(sqlite_conn, pg_conn, table)
                    pg_conn.commit()
                    results[table] = migrated
                    logger.info(f"  Migrated {migrated} rows")
                except Exception as e:
                    logger.error(f"  Error migrating {table}: {e}")
                    pg_conn.rollback()
                    results[table] = -1

    finally:
        sqlite_conn.close()
        pg_conn.close()

    return results


def verify_migration(
    sqlite_path: str,
    pg_host: str = "localhost",
    pg_port: int = 5432,
    pg_database: str = "aragora",
    pg_user: str = "aragora",
    pg_password: str = "",
) -> dict[str, dict]:
    """Verify migration by comparing row counts.

    Returns:
        Dict mapping table names to comparison results
    """
    try:
        import psycopg2
    except ImportError:
        logger.error("psycopg2 not installed")
        return {}

    sqlite_conn = sqlite3.connect(sqlite_path)
    pg_conn = psycopg2.connect(
        host=pg_host,
        port=pg_port,
        dbname=pg_database,
        user=pg_user,
        password=pg_password,
    )

    results = {}

    try:
        tables = get_sqlite_tables(sqlite_conn)

        for table in tables:
            sqlite_count = get_row_count(sqlite_conn, table)

            try:
                pg_cursor = pg_conn.cursor()
                pg_cursor.execute(f"SELECT COUNT(*) FROM {table}")
                pg_count = pg_cursor.fetchone()[0]
            except Exception as e:
                logger.warning(f"Failed to get row count for table {table} in PostgreSQL: {e}")
                pg_count = -1

            results[table] = {
                "sqlite": sqlite_count,
                "postgres": pg_count,
                "match": sqlite_count == pg_count,
            }

    finally:
        sqlite_conn.close()
        pg_conn.close()

    return results


def main():
    """CLI entry point for database migration."""
    parser = argparse.ArgumentParser(description="Aragora database migration tool")
    parser.add_argument(
        "command",
        choices=["migrate", "verify", "schema"],
        help="Command to run",
    )
    parser.add_argument(
        "--sqlite-path",
        default="aragora.db",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--pg-host",
        default="localhost",
        help="PostgreSQL host",
    )
    parser.add_argument(
        "--pg-port",
        type=int,
        default=5432,
        help="PostgreSQL port",
    )
    parser.add_argument(
        "--pg-database",
        default="aragora",
        help="PostgreSQL database name",
    )
    parser.add_argument(
        "--pg-user",
        default="aragora",
        help="PostgreSQL user",
    )
    parser.add_argument(
        "--pg-password",
        default="",
        help="PostgreSQL password",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually migrate data",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.command == "schema":
        print(POSTGRES_SCHEMA)

    elif args.command == "migrate":
        results = migrate_sqlite_to_postgres(
            sqlite_path=args.sqlite_path,
            pg_host=args.pg_host,
            pg_port=args.pg_port,
            pg_database=args.pg_database,
            pg_user=args.pg_user,
            pg_password=args.pg_password,
            dry_run=args.dry_run,
        )
        print("\nMigration results:")
        for table, count in results.items():
            status = "OK" if count >= 0 else "FAILED"
            print(f"  {table}: {count} rows [{status}]")

    elif args.command == "verify":
        results = verify_migration(
            sqlite_path=args.sqlite_path,
            pg_host=args.pg_host,
            pg_port=args.pg_port,
            pg_database=args.pg_database,
            pg_user=args.pg_user,
            pg_password=args.pg_password,
        )
        print("\nVerification results:")
        all_match = True
        for table, comparison in results.items():
            status = "OK" if comparison["match"] else "MISMATCH"
            if not comparison["match"]:
                all_match = False
            print(
                f"  {table}: SQLite={comparison['sqlite']}, "
                f"PostgreSQL={comparison['postgres']} [{status}]"
            )

        sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
