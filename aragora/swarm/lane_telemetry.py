"""Telemetry for terminal swarm/autonomy lanes.

Records one terminal telemetry row per canonical lane outcome so autonomous
execution can be measured like product infrastructure.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from aragora.persistence.db_config import get_default_data_dir

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LaneTelemetryRecord:
    """Telemetry row for one terminal autonomous lane."""

    lane_kind: str
    lane_id: str
    run_id: str = ""
    task_id: str = ""
    work_order_id: str = ""
    project_id: str = ""
    terminal_outcome: str = ""
    worker_outcome: str = ""
    deliverable_type: str = ""
    receipt_id: str = ""
    human_intervention_required: bool = False
    duration_seconds: float = 0.0
    pr_url: str = ""
    pr_number: int | None = None
    merge_ref: str = ""
    merged_at: str = ""
    time_to_pr_seconds: float | None = None
    time_to_merge_seconds: float | None = None
    false_success_candidate: bool = False
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> LaneTelemetryRecord:
        return cls(
            lane_kind=str(row["lane_kind"] or ""),
            lane_id=str(row["lane_id"] or ""),
            run_id=str(row["run_id"] or ""),
            task_id=str(row["task_id"] or ""),
            work_order_id=str(row["work_order_id"] or ""),
            project_id=str(row["project_id"] or ""),
            terminal_outcome=str(row["terminal_outcome"] or ""),
            worker_outcome=str(row["worker_outcome"] or ""),
            deliverable_type=str(row["deliverable_type"] or ""),
            receipt_id=str(row["receipt_id"] or ""),
            human_intervention_required=bool(row["human_intervention_required"]),
            duration_seconds=float(row["duration_seconds"] or 0.0),
            pr_url=str(row["pr_url"] or ""),
            pr_number=row["pr_number"],
            merge_ref=str(row["merge_ref"] or ""),
            merged_at=str(row["merged_at"] or ""),
            time_to_pr_seconds=row["time_to_pr_seconds"],
            time_to_merge_seconds=row["time_to_merge_seconds"],
            false_success_candidate=bool(row["false_success_candidate"]),
            timestamp=float(row["timestamp"] or 0.0),
            metadata=json.loads(str(row["metadata_json"] or "{}")),
        )


class LaneTelemetryCollector:
    """SQLite-backed collector for terminal swarm lane telemetry."""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            data_dir = get_default_data_dir()
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(data_dir / "swarm_lane_telemetry.db")

        self.db_path = db_path
        self._persistent_conn: sqlite3.Connection | None = None
        if db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(":memory:")
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._persistent_conn is not None:
            return self._persistent_conn
        return sqlite3.connect(self.db_path)

    def _close_conn(self, conn: sqlite3.Connection) -> None:
        if conn is not self._persistent_conn:
            conn.close()

    def _init_schema(self) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS lane_telemetry (
                    lane_kind TEXT NOT NULL,
                    lane_id TEXT NOT NULL,
                    run_id TEXT NOT NULL DEFAULT '',
                    task_id TEXT NOT NULL DEFAULT '',
                    work_order_id TEXT NOT NULL DEFAULT '',
                    project_id TEXT NOT NULL DEFAULT '',
                    terminal_outcome TEXT NOT NULL DEFAULT '',
                    worker_outcome TEXT NOT NULL DEFAULT '',
                    deliverable_type TEXT NOT NULL DEFAULT '',
                    receipt_id TEXT NOT NULL DEFAULT '',
                    human_intervention_required INTEGER NOT NULL DEFAULT 0,
                    duration_seconds REAL NOT NULL DEFAULT 0,
                    pr_url TEXT NOT NULL DEFAULT '',
                    pr_number INTEGER,
                    merge_ref TEXT NOT NULL DEFAULT '',
                    merged_at TEXT NOT NULL DEFAULT '',
                    time_to_pr_seconds REAL,
                    time_to_merge_seconds REAL,
                    false_success_candidate INTEGER NOT NULL DEFAULT 0,
                    timestamp REAL NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (lane_kind, lane_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_lane_telemetry_timestamp
                ON lane_telemetry(timestamp DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_lane_telemetry_outcome
                ON lane_telemetry(terminal_outcome)
                """
            )
            conn.commit()
        finally:
            self._close_conn(conn)

    def record_lane(self, record: LaneTelemetryRecord) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO lane_telemetry (
                    lane_kind,
                    lane_id,
                    run_id,
                    task_id,
                    work_order_id,
                    project_id,
                    terminal_outcome,
                    worker_outcome,
                    deliverable_type,
                    receipt_id,
                    human_intervention_required,
                    duration_seconds,
                    pr_url,
                    pr_number,
                    merge_ref,
                    merged_at,
                    time_to_pr_seconds,
                    time_to_merge_seconds,
                    false_success_candidate,
                    timestamp,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.lane_kind,
                    record.lane_id,
                    record.run_id,
                    record.task_id,
                    record.work_order_id,
                    record.project_id,
                    record.terminal_outcome,
                    record.worker_outcome,
                    record.deliverable_type,
                    record.receipt_id,
                    1 if record.human_intervention_required else 0,
                    record.duration_seconds,
                    record.pr_url,
                    record.pr_number,
                    record.merge_ref,
                    record.merged_at,
                    record.time_to_pr_seconds,
                    record.time_to_merge_seconds,
                    1 if record.false_success_candidate else 0,
                    record.timestamp,
                    json.dumps(record.metadata, sort_keys=True),
                ),
            )
            conn.commit()
        finally:
            self._close_conn(conn)

    def get_recent_lanes(self, n: int = 20) -> list[LaneTelemetryRecord]:
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM lane_telemetry ORDER BY timestamp DESC LIMIT ?",
                (n,),
            ).fetchall()
            return [LaneTelemetryRecord.from_row(row) for row in rows]
        finally:
            self._close_conn(conn)

    def get_lane(self, lane_kind: str, lane_id: str) -> LaneTelemetryRecord | None:
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                """
                SELECT * FROM lane_telemetry
                WHERE lane_kind = ? AND lane_id = ?
                LIMIT 1
                """,
                (lane_kind, lane_id),
            ).fetchone()
            return LaneTelemetryRecord.from_row(row) if row is not None else None
        finally:
            self._close_conn(conn)

    def get_throughput(self, window_days: int = 7) -> int:
        row = self._aggregate_scalar(
            "SELECT COUNT(*) FROM lane_telemetry WHERE timestamp >= ?",
            window_days,
        )
        return int(row or 0)

    def get_success_rate(self, window_days: int = 7) -> float:
        row = self._aggregate_row(
            """
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN terminal_outcome IN ('deliverable_created', 'pr_adopted')
                            AND false_success_candidate = 0
                            THEN 1 ELSE 0 END) AS successes
            FROM lane_telemetry
            WHERE timestamp >= ?
              AND COALESCE(TRIM(terminal_outcome), '') NOT IN ('', 'unknown')
            """,
            window_days,
        )
        total = int(row["total"] or 0)
        successes = int(row["successes"] or 0)
        return successes / total if total else 0.0

    def get_false_success_candidate_count(self, window_days: int = 7) -> int:
        row = self._aggregate_scalar(
            """
            SELECT SUM(false_success_candidate)
            FROM lane_telemetry
            WHERE timestamp >= ?
            """,
            window_days,
        )
        return int(row or 0)

    def get_human_intervention_rate(self, window_days: int = 7) -> float:
        row = self._aggregate_row(
            """
            SELECT COUNT(*) AS total,
                   SUM(human_intervention_required) AS human_required
            FROM lane_telemetry
            WHERE timestamp >= ?
              AND COALESCE(TRIM(terminal_outcome), '') NOT IN ('', 'unknown')
            """,
            window_days,
        )
        total = int(row["total"] or 0)
        human_required = int(row["human_required"] or 0)
        return human_required / total if total else 0.0

    def get_merge_yield(self, window_days: int = 7) -> float:
        row = self._aggregate_row(
            """
            SELECT
                SUM(CASE WHEN deliverable_type != '' THEN 1 ELSE 0 END) AS deliverable_count,
                SUM(CASE WHEN merged_at != '' OR merge_ref != '' THEN 1 ELSE 0 END) AS merged_count
            FROM lane_telemetry
            WHERE timestamp >= ?
            """,
            window_days,
        )
        deliverable_count = int(row["deliverable_count"] or 0)
        merged_count = int(row["merged_count"] or 0)
        return merged_count / deliverable_count if deliverable_count else 0.0

    def get_avg_time_to_pr(self, window_days: int = 7) -> float:
        row = self._aggregate_scalar(
            """
            SELECT AVG(time_to_pr_seconds)
            FROM lane_telemetry
            WHERE timestamp >= ? AND time_to_pr_seconds IS NOT NULL
            """,
            window_days,
        )
        return float(row or 0.0)

    def get_avg_time_to_merge(self, window_days: int = 7) -> float:
        row = self._aggregate_scalar(
            """
            SELECT AVG(time_to_merge_seconds)
            FROM lane_telemetry
            WHERE timestamp >= ? AND time_to_merge_seconds IS NOT NULL
            """,
            window_days,
        )
        return float(row or 0.0)

    def _aggregate_scalar(self, query: str, window_days: int) -> Any:
        row = self._aggregate_row(query, window_days)
        if row is None:
            return 0
        return row[0]

    def _aggregate_row(self, query: str, window_days: int) -> sqlite3.Row | tuple[Any, ...] | None:
        cutoff = time.time() - (window_days * 86400)
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        try:
            return conn.execute(query, (cutoff,)).fetchone()
        finally:
            self._close_conn(conn)


__all__ = ["LaneTelemetryCollector", "LaneTelemetryRecord"]
