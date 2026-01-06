"""
Database maintenance utilities for Aragora.

Provides automated maintenance for SQLite databases including:
- VACUUM: Reclaim unused space
- ANALYZE: Update query optimizer statistics
- WAL checkpoint: Flush write-ahead log to main database
- Data retention: Clean up old records
"""

from aragora.maintenance.db_maintenance import (
    DatabaseMaintenance,
    run_startup_maintenance,
    schedule_maintenance,
)

__all__ = [
    "DatabaseMaintenance",
    "run_startup_maintenance",
    "schedule_maintenance",
]
