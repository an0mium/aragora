"""
SQLite-backed Gauntlet result storage.

Provides persistent storage for Gauntlet validation results with
support for listing, filtering, and comparison operations.
"""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Generator, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class GauntletMetadata:
    """Summary metadata for a stored Gauntlet result."""
    gauntlet_id: str
    input_hash: str
    input_summary: str
    verdict: str
    confidence: float
    robustness_score: float
    critical_count: int
    high_count: int
    total_findings: int
    agents_used: list[str]
    template_used: Optional[str]
    created_at: datetime
    duration_seconds: float


class GauntletStorage:
    """
    Persistent storage for Gauntlet validation results.

    Stores complete results in SQLite with support for:
    - Save/load individual results
    - List results with pagination and filters
    - Compare two results
    - Track result history by input hash

    Usage:
        storage = GauntletStorage()
        storage.save(result)

        result = storage.get("gauntlet-abc123")
        recent = storage.list_recent(limit=20)
    """

    def __init__(self, db_path: str = "aragora_gauntlet.db"):
        """Initialize storage with database path."""
        self.db_path = Path(db_path)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with WAL mode for concurrency."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=None,  # Autocommit mode
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
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
                )
            """)

            # Indexes for common queries
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_gauntlet_input_hash "
                "ON gauntlet_results(input_hash)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_gauntlet_created "
                "ON gauntlet_results(created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_gauntlet_verdict "
                "ON gauntlet_results(verdict)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_gauntlet_org "
                "ON gauntlet_results(org_id, created_at DESC)"
            )

    def save(self, result: Any, org_id: Optional[str] = None) -> str:
        """
        Save a GauntletResult to storage.

        Args:
            result: GauntletResult object (from result.py or config.py)
            org_id: Optional organization ID for multi-tenancy

        Returns:
            The gauntlet_id of the saved result
        """
        # Handle both result.py and config.py GauntletResult types
        gauntlet_id = getattr(result, 'gauntlet_id', None) or getattr(result, 'id', f"gauntlet-{id(result)}")
        input_hash = getattr(result, 'input_hash', '')
        input_summary = getattr(result, 'input_summary', '')[:500] if hasattr(result, 'input_summary') else ''
        if not input_hash and input_summary:
            input_hash = hashlib.sha256(input_summary.encode()).hexdigest()

        # Get verdict - handle different result types
        verdict = 'unknown'
        if hasattr(result, 'verdict'):
            verdict = result.verdict.value if hasattr(result.verdict, 'value') else str(result.verdict)
        elif hasattr(result, 'passed'):
            verdict = 'pass' if result.passed else 'fail'

        confidence = getattr(result, 'confidence', 0.5)
        robustness = getattr(result, 'robustness_score', 0.5)

        # Count findings by severity
        critical = high = medium = low = 0
        if hasattr(result, 'critical_findings') and hasattr(result, 'high_findings'):
            critical = len(getattr(result, 'critical_findings', []) or [])
            high = len(getattr(result, 'high_findings', []) or [])
            medium = len(getattr(result, 'medium_findings', []) or [])
            low = len(getattr(result, 'low_findings', []) or [])
        elif hasattr(result, 'risk_summary'):
            critical = getattr(result.risk_summary, 'critical', 0)
            high = getattr(result.risk_summary, 'high', 0)
            medium = getattr(result.risk_summary, 'medium', 0)
            low = getattr(result.risk_summary, 'low', 0)
        elif hasattr(result, 'severity_counts'):
            counts = result.severity_counts
            critical = counts.get('critical', 0)
            high = counts.get('high', 0)
            medium = counts.get('medium', 0)
            low = counts.get('low', 0)

        total = getattr(result, 'total_findings', None)
        if total is None:
            total = critical + high + medium + low
            if hasattr(result, 'vulnerabilities'):
                total = len(result.vulnerabilities)
            elif hasattr(result, 'findings'):
                total = len(result.findings)

        agents = getattr(result, 'agents_used', None) or getattr(result, 'agents_involved', [])
        template = getattr(result, 'template_used', None)
        duration = getattr(result, 'duration_seconds', 0)

        # Serialize result to JSON
        if hasattr(result, 'to_dict'):
            result_dict = result.to_dict()
        else:
            result_dict = {
                'gauntlet_id': gauntlet_id,
                'verdict': verdict,
                'confidence': confidence,
            }

        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO gauntlet_results (
                    gauntlet_id, input_hash, input_summary, result_json,
                    verdict, confidence, robustness_score,
                    critical_count, high_count, medium_count, low_count,
                    total_findings, agents_used, template_used,
                    duration_seconds, org_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                gauntlet_id,
                input_hash,
                input_summary,
                json.dumps(result_dict, default=str),
                verdict,
                confidence,
                robustness,
                critical,
                high,
                medium,
                low,
                total,
                json.dumps(agents),
                template,
                duration,
                org_id,
            ))

        logger.info(f"Saved gauntlet result: {gauntlet_id}")
        return gauntlet_id

    def get(self, gauntlet_id: str, org_id: Optional[str] = None) -> Optional[dict]:
        """
        Get a GauntletResult by ID.

        Args:
            gauntlet_id: The gauntlet result ID
            org_id: Optional org ID for ownership verification

        Returns:
            Result dict or None if not found
        """
        with self._get_connection() as conn:
            if org_id:
                cursor = conn.execute(
                    "SELECT result_json FROM gauntlet_results "
                    "WHERE gauntlet_id = ? AND org_id = ?",
                    (gauntlet_id, org_id)
                )
            else:
                cursor = conn.execute(
                    "SELECT result_json FROM gauntlet_results WHERE gauntlet_id = ?",
                    (gauntlet_id,)
                )
            row = cursor.fetchone()

        return json.loads(row[0]) if row else None

    def list_recent(
        self,
        limit: int = 20,
        offset: int = 0,
        org_id: Optional[str] = None,
        verdict: Optional[str] = None,
        min_severity: Optional[str] = None,
    ) -> list[GauntletMetadata]:
        """
        List recent Gauntlet results with optional filters.

        Args:
            limit: Maximum results to return
            offset: Offset for pagination
            org_id: Filter by organization
            verdict: Filter by verdict (pass, fail, conditional)
            min_severity: Only include results with findings at this severity or higher

        Returns:
            List of GauntletMetadata
        """
        query = """
            SELECT gauntlet_id, input_hash, input_summary, verdict, confidence,
                   robustness_score, critical_count, high_count, total_findings,
                   agents_used, template_used, created_at, duration_seconds
            FROM gauntlet_results
            WHERE 1=1
        """
        params: list = []

        if org_id:
            query += " AND org_id = ?"
            params.append(org_id)

        if verdict:
            query += " AND verdict = ?"
            params.append(verdict)

        if min_severity == 'critical':
            query += " AND critical_count > 0"
        elif min_severity == 'high':
            query += " AND (critical_count > 0 OR high_count > 0)"

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            results = []
            for row in cursor.fetchall():
                try:
                    created = datetime.fromisoformat(row[11])
                except (ValueError, TypeError):
                    created = datetime.now()

                results.append(GauntletMetadata(
                    gauntlet_id=row[0],
                    input_hash=row[1],
                    input_summary=row[2] or '',
                    verdict=row[3],
                    confidence=row[4] or 0,
                    robustness_score=row[5] or 0,
                    critical_count=row[6] or 0,
                    high_count=row[7] or 0,
                    total_findings=row[8] or 0,
                    agents_used=json.loads(row[9]) if row[9] else [],
                    template_used=row[10],
                    created_at=created,
                    duration_seconds=row[12] or 0,
                ))

        return results

    def get_history(
        self,
        input_hash: str,
        limit: int = 10,
        org_id: Optional[str] = None,
    ) -> list[GauntletMetadata]:
        """
        Get validation history for a specific input.

        Useful for tracking improvements over time.

        Args:
            input_hash: SHA-256 hash of the input content
            limit: Maximum results to return
            org_id: Filter by organization

        Returns:
            List of GauntletMetadata ordered by date (newest first)
        """
        query = """
            SELECT gauntlet_id, input_hash, input_summary, verdict, confidence,
                   robustness_score, critical_count, high_count, total_findings,
                   agents_used, template_used, created_at, duration_seconds
            FROM gauntlet_results
            WHERE input_hash = ?
        """
        params: list = [input_hash]

        if org_id:
            query += " AND org_id = ?"
            params.append(org_id)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            results = []
            for row in cursor.fetchall():
                try:
                    created = datetime.fromisoformat(row[11])
                except (ValueError, TypeError):
                    created = datetime.now()

                results.append(GauntletMetadata(
                    gauntlet_id=row[0],
                    input_hash=row[1],
                    input_summary=row[2] or '',
                    verdict=row[3],
                    confidence=row[4] or 0,
                    robustness_score=row[5] or 0,
                    critical_count=row[6] or 0,
                    high_count=row[7] or 0,
                    total_findings=row[8] or 0,
                    agents_used=json.loads(row[9]) if row[9] else [],
                    template_used=row[10],
                    created_at=created,
                    duration_seconds=row[12] or 0,
                ))

        return results

    def compare(
        self,
        id1: str,
        id2: str,
        org_id: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Compare two Gauntlet results.

        Args:
            id1: First gauntlet ID (usually newer)
            id2: Second gauntlet ID (usually older)
            org_id: Optional org ID for ownership verification

        Returns:
            Comparison dict with deltas, or None if either result not found
        """
        result1 = self.get(id1, org_id)
        result2 = self.get(id2, org_id)

        if not result1 or not result2:
            return None

        # Extract comparable metrics
        def extract_metrics(r: dict) -> dict:
            risk = r.get('risk_summary', {})
            return {
                'verdict': r.get('verdict', 'unknown'),
                'confidence': r.get('confidence', 0),
                'robustness_score': r.get('robustness_score', 0) or r.get('attack_summary', {}).get('robustness_score', 0),
                'critical': risk.get('critical', 0),
                'high': risk.get('high', 0),
                'medium': risk.get('medium', 0),
                'low': risk.get('low', 0),
                'total': risk.get('total', 0),
            }

        m1 = extract_metrics(result1)
        m2 = extract_metrics(result2)

        # Calculate deltas (positive = improvement in result1)
        return {
            'result1_id': id1,
            'result2_id': id2,
            'verdict_changed': m1['verdict'] != m2['verdict'],
            'verdict_improved': (
                m1['verdict'] == 'pass' and m2['verdict'] != 'pass'
            ),
            'deltas': {
                'confidence': m1['confidence'] - m2['confidence'],
                'robustness': m1['robustness_score'] - m2['robustness_score'],
                'critical': m2['critical'] - m1['critical'],  # Reduction is good
                'high': m2['high'] - m1['high'],
                'medium': m2['medium'] - m1['medium'],
                'low': m2['low'] - m1['low'],
                'total': m2['total'] - m1['total'],
            },
            'metrics_1': m1,
            'metrics_2': m2,
            'improved': (
                m1['critical'] <= m2['critical'] and
                m1['high'] <= m2['high'] and
                m1['robustness_score'] >= m2['robustness_score']
            ),
        }

    def delete(self, gauntlet_id: str, org_id: Optional[str] = None) -> bool:
        """
        Delete a Gauntlet result.

        Args:
            gauntlet_id: Result ID to delete
            org_id: Optional org ID for ownership verification

        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            if org_id:
                cursor = conn.execute(
                    "DELETE FROM gauntlet_results WHERE gauntlet_id = ? AND org_id = ?",
                    (gauntlet_id, org_id)
                )
            else:
                cursor = conn.execute(
                    "DELETE FROM gauntlet_results WHERE gauntlet_id = ?",
                    (gauntlet_id,)
                )
            deleted = cursor.rowcount > 0

        if deleted:
            logger.info(f"Deleted gauntlet result: {gauntlet_id}")

        return deleted

    def count(self, org_id: Optional[str] = None, verdict: Optional[str] = None) -> int:
        """
        Count total Gauntlet results.

        Args:
            org_id: Filter by organization
            verdict: Filter by verdict

        Returns:
            Total count
        """
        query = "SELECT COUNT(*) FROM gauntlet_results WHERE 1=1"
        params: list = []

        if org_id:
            query += " AND org_id = ?"
            params.append(org_id)

        if verdict:
            query += " AND verdict = ?"
            params.append(verdict)

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            row = cursor.fetchone()

        return row[0] if row else 0


# Module-level singleton for convenience
_default_storage: Optional[GauntletStorage] = None


def get_storage(db_path: str = "aragora_gauntlet.db") -> GauntletStorage:
    """Get or create the default GauntletStorage instance."""
    global _default_storage
    if _default_storage is None:
        _default_storage = GauntletStorage(db_path)
    return _default_storage
