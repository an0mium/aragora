"""Epistemic outcome ledger for debate claims with explicit settlement metadata.

This module stores normalized claim/falsifier/metric tuples and tracks
resolution state over time. It is intentionally lightweight and uses the
existing SQLiteStore pattern used across Aragora.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_EPISTEMIC_OUTCOMES_DB",
    "EpistemicOutcome",
    "EpistemicOutcomeStore",
    "get_epistemic_outcome_store",
]

import json
import sqlite3
import threading
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aragora.config import DB_TIMEOUT_SECONDS
from aragora.persistence.db_config import get_nomic_dir
from aragora.storage.base_store import SQLiteStore

DEFAULT_EPISTEMIC_OUTCOMES_DB = get_nomic_dir() / "epistemic_outcomes.db"


@dataclass
class EpistemicOutcome:
    """Tracked claim emitted from a debate with settlement scaffolding."""

    debate_id: str
    claim: str
    falsifier: str
    metric: str
    review_horizon_days: int = 30
    resolver_type: str = "human"
    status: str = "open"
    initial_confidence: float = 0.0
    confidence_delta: float = 0.0
    resolved_truth: bool | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to DB-ready payload."""
        payload = asdict(self)
        payload["metadata"] = json.dumps(payload.get("metadata") or {}, sort_keys=True)
        value = payload.get("resolved_truth")
        payload["resolved_truth"] = None if value is None else int(bool(value))
        payload["review_horizon_days"] = max(1, int(payload.get("review_horizon_days", 30)))
        return payload

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> EpistemicOutcome:
        """Create instance from sqlite row."""
        data = dict(row)
        try:
            data["metadata"] = json.loads(data.get("metadata") or "{}")
        except json.JSONDecodeError:
            data["metadata"] = {}
        resolved = data.get("resolved_truth")
        data["resolved_truth"] = None if resolved is None else bool(resolved)
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


class _EpistemicOutcomeDB(SQLiteStore):
    SCHEMA_NAME = "epistemic_outcomes"
    SCHEMA_VERSION = 1
    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS epistemic_outcomes (
            debate_id TEXT PRIMARY KEY,
            claim TEXT NOT NULL,
            falsifier TEXT NOT NULL,
            metric TEXT NOT NULL,
            review_horizon_days INTEGER NOT NULL DEFAULT 30,
            resolver_type TEXT NOT NULL DEFAULT 'human',
            status TEXT NOT NULL DEFAULT 'open',
            initial_confidence REAL NOT NULL DEFAULT 0.0,
            confidence_delta REAL NOT NULL DEFAULT 0.0,
            resolved_truth INTEGER,
            created_at TEXT NOT NULL,
            resolved_at TEXT,
            metadata TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_epistemic_outcomes_status
        ON epistemic_outcomes(status);

        CREATE INDEX IF NOT EXISTS idx_epistemic_outcomes_created_at
        ON epistemic_outcomes(created_at);
    """


class EpistemicOutcomeStore:
    """SQLite persistence wrapper for EpistemicOutcome records."""

    def __init__(self, db_path: Path = DEFAULT_EPISTEMIC_OUTCOMES_DB):
        self.db_path = Path(db_path)
        self._db = _EpistemicOutcomeDB(str(self.db_path), timeout=DB_TIMEOUT_SECONDS)

    def record_outcome(self, outcome: EpistemicOutcome) -> None:
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO epistemic_outcomes (
                    debate_id, claim, falsifier, metric, review_horizon_days,
                    resolver_type, status, initial_confidence, confidence_delta,
                    resolved_truth, created_at, resolved_at, metadata
                ) VALUES (
                    :debate_id, :claim, :falsifier, :metric, :review_horizon_days,
                    :resolver_type, :status, :initial_confidence, :confidence_delta,
                    :resolved_truth, :created_at, :resolved_at, :metadata
                )
                """,
                outcome.to_dict(),
            )
            conn.commit()

    def get_outcome(self, debate_id: str) -> EpistemicOutcome | None:
        with self._db.connection() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM epistemic_outcomes WHERE debate_id = ?",
                (debate_id,),
            ).fetchone()
        if row is None:
            return None
        return EpistemicOutcome.from_row(row)

    def list_outcomes(self, *, status: str | None = None, limit: int = 100) -> list[EpistemicOutcome]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)

        query = "SELECT * FROM epistemic_outcomes"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, int(limit)))

        with self._db.connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, tuple(params)).fetchall()
        return [EpistemicOutcome.from_row(row) for row in rows]

    def resolve_outcome(
        self,
        debate_id: str,
        *,
        resolved_truth: bool,
        confidence_delta: float = 0.0,
        resolver_type: str = "human",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        current = self.get_outcome(debate_id)
        if current is None:
            return False

        merged_metadata = dict(current.metadata)
        if metadata:
            merged_metadata.update(metadata)

        current.status = "resolved"
        current.resolved_truth = bool(resolved_truth)
        current.confidence_delta = float(confidence_delta)
        current.resolved_at = datetime.now(timezone.utc).isoformat()
        current.resolver_type = resolver_type
        current.metadata = merged_metadata
        self.record_outcome(current)
        return True


_STORE_LOCK = threading.Lock()
_STORE_SINGLETON: EpistemicOutcomeStore | None = None


def get_epistemic_outcome_store(
    db_path: Path | None = None,
) -> EpistemicOutcomeStore:
    """Get process-wide singleton store unless explicit db_path is provided."""
    global _STORE_SINGLETON
    if db_path is not None:
        return EpistemicOutcomeStore(db_path=db_path)
    with _STORE_LOCK:
        if _STORE_SINGLETON is None:
            _STORE_SINGLETON = EpistemicOutcomeStore()
        return _STORE_SINGLETON
