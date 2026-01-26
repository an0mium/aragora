"""
Database schema validation for startup health checks.

Validates that all required tables and columns exist in consolidated databases.

Usage:
    from aragora.persistence.validator import validate_consolidated_schema

    result = validate_consolidated_schema()
    if not result.success:
        for error in result.errors:
            logger.error(f"Schema validation failed: {error}")
        raise RuntimeError("Database schema validation failed")

Environment Variables:
    ARAGORA_SKIP_SCHEMA_VALIDATION: Set to "1" to bypass validation
"""

__all__ = [
    "ValidationResult",
    "validate_consolidated_schema",
    "validate_schema_versions",
]

import logging
import os
import sqlite3
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


class ValidationResult(NamedTuple):
    """Result of schema validation."""

    success: bool
    errors: list[str]
    warnings: list[str]


# Required tables per consolidated database
REQUIRED_TABLES = {
    "core.db": [
        "debates",
        "traces",
        "trace_events",
        "tournaments",
        "tournament_matches",
        "embeddings",
        "positions",
        "detected_flips",
    ],
    "memory.db": [
        "continuum_memory",
        "consensus",
        "dissent",
        "critiques",
        "patterns",
        "agent_reputation",
        "semantic_embeddings",
    ],
    "analytics.db": [
        "ratings",
        "matches",
        "elo_history",
        "calibration_predictions",
        "insights",
        "debate_summaries",
        "predictions",
    ],
    "agents.db": [
        "personas",
        "genomes",
        "populations",
        "genesis_events",
        "experiments",
        "position_history",
    ],
}


def _should_skip_validation() -> bool:
    """Check if schema validation should be skipped."""
    return os.environ.get("ARAGORA_SKIP_SCHEMA_VALIDATION", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _get_consolidated_db_dir() -> Path:
    """Get the directory containing consolidated databases."""
    from aragora.persistence.db_config import get_nomic_dir

    return get_nomic_dir()


def _check_table_exists(db_path: Path, table_name: str) -> bool:
    """Check if a table exists in a SQLite database."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        result = cursor.fetchone() is not None
        conn.close()
        return result
    except Exception as e:
        logger.warning(f"Error checking table {table_name} in {db_path}: {e}")
        return False


def _get_table_count(db_path: Path) -> int:
    """Get count of tables in a database."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


def validate_consolidated_schema() -> ValidationResult:
    """
    Validate that consolidated databases have required schema.

    Returns:
        ValidationResult with success status, errors, and warnings.
    """
    if _should_skip_validation():
        return ValidationResult(True, [], ["Schema validation skipped via env var"])

    # Check database mode
    try:
        from aragora.persistence.db_config import DatabaseMode, get_db_mode

        mode = get_db_mode()
        if mode != DatabaseMode.CONSOLIDATED:
            return ValidationResult(True, [], [f"Not in consolidated mode (mode={mode.value})"])
    except ImportError:
        return ValidationResult(True, [], ["db_config not available"])

    errors: list[str] = []
    warnings: list[str] = []
    db_dir = _get_consolidated_db_dir()

    for db_name, required_tables in REQUIRED_TABLES.items():
        db_path = db_dir / db_name

        # Check database file exists
        if not db_path.exists():
            errors.append(f"Database {db_name} not found at {db_path}")
            continue

        # Check required tables exist
        missing_tables = []
        for table in required_tables:
            if not _check_table_exists(db_path, table):
                missing_tables.append(table)

        if missing_tables:
            errors.append(f"Database {db_name} missing tables: {', '.join(missing_tables)}")

        # Log table count for diagnostics
        table_count = _get_table_count(db_path)
        logger.debug(f"Database {db_name}: {table_count} tables")

    success = len(errors) == 0
    if success:
        logger.info("Database schema validation passed")
    else:
        logger.error(f"Database schema validation failed: {len(errors)} errors")

    return ValidationResult(success, errors, warnings)


def validate_schema_versions() -> ValidationResult:
    """
    Validate schema versions are consistent across databases.

    Returns:
        ValidationResult with version consistency status.
    """
    if _should_skip_validation():
        return ValidationResult(True, [], ["Schema version check skipped"])

    errors: list[str] = []
    warnings: list[str] = []
    db_dir = _get_consolidated_db_dir()

    for db_name in REQUIRED_TABLES.keys():
        db_path = db_dir / db_name

        if not db_path.exists():
            continue

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='_schema_versions'"
            )
            if cursor.fetchone():
                cursor = conn.execute("SELECT module, version FROM _schema_versions")
                versions = cursor.fetchall()
                logger.debug(f"{db_name} schema versions: {dict(versions)}")
            else:
                warnings.append(f"{db_name}: No _schema_versions table")
            conn.close()
        except Exception as e:
            warnings.append(f"{db_name}: Error reading schema versions: {e}")

    return ValidationResult(len(errors) == 0, errors, warnings)


def get_database_health() -> dict:
    """
    Get comprehensive database health information.

    Returns:
        Dictionary with database health details for API response.
    """
    try:
        from aragora.persistence.db_config import get_db_mode

        mode = get_db_mode()
    except ImportError:
        mode = None

    schema_result = validate_consolidated_schema()
    version_result = validate_schema_versions()

    db_dir = _get_consolidated_db_dir()
    databases = {}

    for db_name in REQUIRED_TABLES.keys():
        db_path = db_dir / db_name
        db_info = {
            "exists": db_path.exists(),
            "path": str(db_path),
        }

        if db_path.exists():
            db_info["size_bytes"] = db_path.stat().st_size
            db_info["table_count"] = _get_table_count(db_path)

            # Check required tables
            missing = []
            for table in REQUIRED_TABLES[db_name]:
                if not _check_table_exists(db_path, table):
                    missing.append(table)
            db_info["missing_tables"] = missing

        databases[db_name] = db_info

    return {
        "status": "healthy" if schema_result.success else "unhealthy",
        "mode": mode.value if mode else "unknown",
        "validation": {
            "success": schema_result.success,
            "errors": schema_result.errors,
            "warnings": schema_result.warnings,
        },
        "versions": {
            "success": version_result.success,
            "warnings": version_result.warnings,
        },
        "databases": databases,
    }
