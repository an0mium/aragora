#!/usr/bin/env python3
"""
Database Consolidation Migration Script

Migrates from 10+ SQLite databases to 4 consolidated databases:
- aragora_core.db: Debates, consensus, positions
- aragora_memory.db: Multi-tier learning, critiques, insights
- aragora_agents.db: Agent identity, ELO, personas, genomes
- .nomic/genesis.db: Immutable provenance (unchanged)

Usage:
    python scripts/migrate_databases.py --dry-run
    python scripts/migrate_databases.py --validate
    python scripts/migrate_databases.py --migrate

See docs/DATABASE_CONSOLIDATION.md for full migration strategy.
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DatabaseInventory:
    """Tracks database files and their metadata."""
    path: Path
    tables: List[str] = field(default_factory=list)
    row_counts: Dict[str, int] = field(default_factory=dict)
    checksum: str = ""
    size_bytes: int = 0


@dataclass
class MigrationResult:
    """Result of a migration operation."""
    source_db: str
    target_db: str
    tables_migrated: List[str]
    rows_migrated: int
    success: bool
    error: Optional[str] = None
    duration_seconds: float = 0.0


class DatabaseMigrator:
    """Handles database consolidation migrations."""

    # Source -> Target mapping
    MIGRATION_MAP = {
        # Core database consolidation
        "aragora_debates.db": ("aragora_core.db", ["debates"]),
        "consensus_memory.db": ("aragora_core.db", ["consensus", "dissent"]),
        "position_ledger.db": ("aragora_core.db", ["positions"]),

        # Memory database consolidation
        "aragora_memory.db": ("aragora_memory.db", ["continuum_memory", "tier_transitions", "meta_learning_state"]),
        "agora_memory.db": ("aragora_memory.db", ["critiques", "patterns", "agent_reputation"]),
        "aragora_insights.db": ("aragora_memory.db", ["insights", "pattern_clusters"]),

        # Agent database consolidation
        "aragora_elo.db": ("aragora_agents.db", ["ratings", "matches", "elo_history", "calibration_buckets"]),
        "persona_lab.db": ("aragora_agents.db", ["experiments"]),
    }

    TARGET_DATABASES = [
        "aragora_core.db",
        "aragora_memory.db",
        "aragora_agents.db",
    ]

    def __init__(self, source_dir: Path, target_dir: Path, backup_dir: Path):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.backup_dir = backup_dir
        self.inventory: Dict[str, DatabaseInventory] = {}
        self.results: List[MigrationResult] = []

    def discover_databases(self) -> Dict[str, DatabaseInventory]:
        """Scan for all database files and inventory them."""
        logger.info(f"Discovering databases in {self.source_dir}")

        db_files = list(self.source_dir.glob("*.db"))
        db_files.extend(self.source_dir.glob(".nomic/*.db"))

        for db_path in db_files:
            if not db_path.exists():
                continue

            inv = DatabaseInventory(path=db_path)
            inv.size_bytes = db_path.stat().st_size
            inv.checksum = self._compute_checksum(db_path)

            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Get table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                inv.tables = [row[0] for row in cursor.fetchall()]

                # Get row counts
                for table in inv.tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        inv.row_counts[table] = cursor.fetchone()[0]
                    except sqlite3.Error:
                        inv.row_counts[table] = -1

                conn.close()
            except sqlite3.Error as e:
                logger.warning(f"Could not read {db_path}: {e}")

            self.inventory[db_path.name] = inv

        logger.info(f"Found {len(self.inventory)} database files")
        return self.inventory

    def _compute_checksum(self, path: Path) -> str:
        """Compute MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def backup_all(self) -> bool:
        """Create JSON backups of all databases."""
        logger.info(f"Creating backups in {self.backup_dir}")
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_subdir = self.backup_dir / f"backup_{timestamp}"
        backup_subdir.mkdir()

        manifest = {
            "timestamp": timestamp,
            "databases": {},
        }

        for db_name, inv in self.inventory.items():
            try:
                # Copy the raw database file
                shutil.copy2(inv.path, backup_subdir / db_name)

                # Export to JSON for readability
                json_path = backup_subdir / f"{db_name}.json"
                self._export_to_json(inv.path, json_path)

                manifest["databases"][db_name] = {
                    "checksum": inv.checksum,
                    "size_bytes": inv.size_bytes,
                    "tables": inv.tables,
                    "row_counts": inv.row_counts,
                }
            except Exception as e:
                logger.error(f"Failed to backup {db_name}: {e}")
                return False

        # Write manifest
        manifest_path = backup_subdir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Backup complete: {backup_subdir}")
        return True

    def _export_to_json(self, db_path: Path, json_path: Path) -> None:
        """Export all tables in a database to JSON."""
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        export = {}

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            try:
                cursor.execute(f"SELECT * FROM {table}")
                rows = [dict(row) for row in cursor.fetchall()]
                export[table] = rows
            except sqlite3.Error:
                export[table] = []

        conn.close()

        with open(json_path, "w") as f:
            json.dump(export, f, indent=2, default=str)

    def validate_source(self) -> Tuple[bool, List[str]]:
        """Validate source databases are readable and have expected tables."""
        errors = []

        for source_db, (target_db, expected_tables) in self.MIGRATION_MAP.items():
            if source_db not in self.inventory:
                # Not all databases are required (some are created on-demand)
                logger.warning(f"Source database not found: {source_db}")
                continue

            inv = self.inventory[source_db]
            for table in expected_tables:
                if table not in inv.tables:
                    errors.append(f"Missing table {table} in {source_db}")

        return len(errors) == 0, errors

    def create_target_schemas(self, dry_run: bool = True) -> bool:
        """Create the consolidated target database schemas."""
        if dry_run:
            logger.info("[DRY RUN] Would create target schemas")
            return True

        self.target_dir.mkdir(parents=True, exist_ok=True)

        # Schema definitions for consolidated databases
        schemas = {
            "aragora_core.db": """
                -- Debates table (from DebateStorage)
                CREATE TABLE IF NOT EXISTS debates (
                    id TEXT PRIMARY KEY,
                    slug TEXT UNIQUE,
                    topic TEXT NOT NULL,
                    agents TEXT,  -- JSON array
                    rounds INTEGER,
                    result TEXT,  -- JSON
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_debates_slug ON debates(slug);

                -- Consensus table (from ConsensusMemory)
                CREATE TABLE IF NOT EXISTS consensus (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    debate_id TEXT,
                    topic TEXT NOT NULL,
                    position TEXT NOT NULL,
                    confidence REAL,
                    evidence TEXT,  -- JSON
                    participants TEXT,  -- JSON
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (debate_id) REFERENCES debates(id)
                );
                CREATE INDEX IF NOT EXISTS idx_consensus_topic ON consensus(topic);

                -- Dissent table (from ConsensusMemory)
                CREATE TABLE IF NOT EXISTS dissent (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    consensus_id INTEGER,
                    agent TEXT NOT NULL,
                    position TEXT NOT NULL,
                    reasoning TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (consensus_id) REFERENCES consensus(id)
                );

                -- Positions table (new unified position tracking)
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent TEXT NOT NULL,
                    claim TEXT NOT NULL,
                    stance TEXT,  -- 'support', 'oppose', 'neutral'
                    confidence REAL,
                    citations TEXT,  -- JSON
                    debate_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (debate_id) REFERENCES debates(id)
                );
                CREATE INDEX IF NOT EXISTS idx_positions_agent ON positions(agent);

                -- Schema version tracking
                CREATE TABLE IF NOT EXISTS _schema_versions (
                    module TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """,

            "aragora_memory.db": """
                -- Continuum memory tiers
                CREATE TABLE IF NOT EXISTS continuum_memory (
                    id TEXT PRIMARY KEY,
                    tier TEXT NOT NULL,  -- 'fast', 'medium', 'slow', 'glacial'
                    content TEXT NOT NULL,
                    importance REAL DEFAULT 0.5,
                    surprise_score REAL DEFAULT 0.0,
                    consolidation_score REAL DEFAULT 0.0,
                    access_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    update_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_continuum_tier ON continuum_memory(tier);

                -- Tier transitions log
                CREATE TABLE IF NOT EXISTS tier_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT,
                    from_tier TEXT,
                    to_tier TEXT,
                    reason TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Critiques (from CritiqueStore)
                CREATE TABLE IF NOT EXISTS critiques (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    debate_id TEXT,
                    agent TEXT,
                    target_agent TEXT,
                    critique_type TEXT,
                    content TEXT,
                    impact_score REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Patterns (from CritiqueStore)
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT,
                    description TEXT,
                    frequency INTEGER DEFAULT 1,
                    agents TEXT,  -- JSON
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Agent reputation (from CritiqueStore)
                CREATE TABLE IF NOT EXISTS agent_reputation (
                    agent TEXT PRIMARY KEY,
                    reputation_score REAL DEFAULT 0.0,
                    critique_count INTEGER DEFAULT 0,
                    helpful_count INTEGER DEFAULT 0,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Insights (from InsightStore)
                CREATE TABLE IF NOT EXISTS insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    debate_id TEXT,
                    insight_type TEXT,
                    content TEXT,
                    confidence REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Pattern clusters (from InsightStore)
                CREATE TABLE IF NOT EXISTS pattern_clusters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cluster_id TEXT,
                    patterns TEXT,  -- JSON
                    centroid TEXT,  -- JSON embedding
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Schema version tracking
                CREATE TABLE IF NOT EXISTS _schema_versions (
                    module TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """,

            "aragora_agents.db": """
                -- ELO ratings (from EloSystem)
                CREATE TABLE IF NOT EXISTS ratings (
                    agent TEXT PRIMARY KEY,
                    elo REAL DEFAULT 1500.0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    draws INTEGER DEFAULT 0,
                    peak_elo REAL DEFAULT 1500.0,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Match history (from EloSystem)
                CREATE TABLE IF NOT EXISTS matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    debate_id TEXT,
                    agent_a TEXT,
                    agent_b TEXT,
                    winner TEXT,  -- 'a', 'b', 'draw'
                    elo_delta_a REAL,
                    elo_delta_b REAL,
                    domain TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_matches_agents ON matches(agent_a, agent_b);

                -- ELO history (from EloSystem)
                CREATE TABLE IF NOT EXISTS elo_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent TEXT,
                    elo REAL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_elo_history_agent ON elo_history(agent);

                -- Calibration data (from EloSystem)
                CREATE TABLE IF NOT EXISTS calibration_buckets (
                    bucket_id TEXT PRIMARY KEY,
                    agent TEXT,
                    confidence_low REAL,
                    confidence_high REAL,
                    correct_count INTEGER DEFAULT 0,
                    total_count INTEGER DEFAULT 0,
                    calibration_score REAL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Genomes (from GenomeStore)
                CREATE TABLE IF NOT EXISTS genomes (
                    id TEXT PRIMARY KEY,
                    agent TEXT NOT NULL,
                    traits TEXT,  -- JSON
                    lineage TEXT,  -- JSON
                    fitness_score REAL,
                    generation INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_genomes_agent ON genomes(agent);

                -- Personas (from PersonaManager)
                CREATE TABLE IF NOT EXISTS personas (
                    agent TEXT PRIMARY KEY,
                    traits TEXT,  -- JSON
                    expertise TEXT,  -- JSON
                    specialization TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Experiments (from PersonaLaboratory)
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT UNIQUE,
                    agent TEXT,
                    variant_a TEXT,  -- JSON
                    variant_b TEXT,  -- JSON
                    metrics TEXT,  -- JSON
                    winner TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Schema version tracking
                CREATE TABLE IF NOT EXISTS _schema_versions (
                    module TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """,
        }

        for db_name, schema in schemas.items():
            db_path = self.target_dir / db_name
            try:
                conn = sqlite3.connect(db_path)
                conn.executescript(schema)
                conn.commit()
                conn.close()
                logger.info(f"Created schema for {db_name}")
            except sqlite3.Error as e:
                logger.error(f"Failed to create schema for {db_name}: {e}")
                return False

        return True

    def migrate(self, dry_run: bool = True) -> List[MigrationResult]:
        """Execute the migration."""
        if dry_run:
            logger.info("[DRY RUN] Migration plan:")
            for source_db, (target_db, tables) in self.MIGRATION_MAP.items():
                if source_db in self.inventory:
                    inv = self.inventory[source_db]
                    total_rows = sum(inv.row_counts.get(t, 0) for t in tables if t in inv.row_counts)
                    logger.info(f"  {source_db} -> {target_db}: {tables} ({total_rows} rows)")
            return []

        # Actual migration would happen here
        logger.warning("Full migration not yet implemented. Use --dry-run to see plan.")
        return []

    def report(self) -> Dict:
        """Generate migration report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "source_dir": str(self.source_dir),
            "target_dir": str(self.target_dir),
            "inventory": {},
            "migration_plan": {},
        }

        for db_name, inv in self.inventory.items():
            report["inventory"][db_name] = {
                "path": str(inv.path),
                "size_bytes": inv.size_bytes,
                "tables": inv.tables,
                "row_counts": inv.row_counts,
                "checksum": inv.checksum,
            }

        for source, (target, tables) in self.MIGRATION_MAP.items():
            if source in self.inventory:
                report["migration_plan"][source] = {
                    "target": target,
                    "tables": tables,
                }

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Aragora Database Consolidation Migration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/migrate_databases.py --dry-run
  python scripts/migrate_databases.py --validate
  python scripts/migrate_databases.py --backup
  python scripts/migrate_databases.py --migrate

See docs/DATABASE_CONSOLIDATION.md for full documentation.
        """
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing source databases (default: current directory)"
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path.cwd() / "consolidated",
        help="Directory for consolidated databases (default: ./consolidated)"
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path.cwd() / "backups",
        help="Directory for backups (default: ./backups)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show migration plan without executing"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate source databases"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backups of all databases"
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Execute the migration (requires --backup first)"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate JSON report of current state"
    )

    args = parser.parse_args()

    migrator = DatabaseMigrator(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        backup_dir=args.backup_dir,
    )

    # Always discover databases first
    migrator.discover_databases()

    if args.report:
        report = migrator.report()
        print(json.dumps(report, indent=2))
        return 0

    if args.validate:
        valid, errors = migrator.validate_source()
        if valid:
            logger.info("Validation passed!")
        else:
            logger.error(f"Validation failed with {len(errors)} errors:")
            for err in errors:
                logger.error(f"  - {err}")
            return 1
        return 0

    if args.backup:
        success = migrator.backup_all()
        return 0 if success else 1

    if args.migrate:
        if not args.backup:
            logger.warning("Recommend running --backup before --migrate")
        migrator.create_target_schemas(dry_run=args.dry_run)
        results = migrator.migrate(dry_run=args.dry_run)
        return 0

    if args.dry_run:
        migrator.create_target_schemas(dry_run=True)
        migrator.migrate(dry_run=True)
        return 0

    # Default: show inventory
    print("\nDatabase Inventory:")
    print("-" * 60)
    total_size = 0
    total_rows = 0
    for db_name, inv in sorted(migrator.inventory.items()):
        rows = sum(inv.row_counts.values())
        total_rows += rows
        total_size += inv.size_bytes
        print(f"\n{db_name}")
        print(f"  Size: {inv.size_bytes / 1024:.1f} KB")
        print(f"  Tables: {', '.join(inv.tables) or 'none'}")
        print(f"  Total Rows: {rows}")

    print("\n" + "-" * 60)
    print(f"Total: {len(migrator.inventory)} databases, {total_size / 1024:.1f} KB, {total_rows} rows")
    print("\nRun with --dry-run to see migration plan")
    return 0


if __name__ == "__main__":
    sys.exit(main())
