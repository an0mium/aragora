"""Calibration gate for auto-handle merge paths (#6372).

This module adds a small SQLite-backed outcome-history layer for the two
auto-handle paths named in ``docs/THESIS.md`` Commitment 1:

  - ``fire_and_forget`` low-risk merge in :mod:`aragora.swarm.tranche_integrate`
  - ``admin_merge_allowed`` review-gate bypass in :mod:`aragora.ralph.supervisor`

It deliberately stays narrow:

  - coarse, stable decision-class fingerprints
  - per-event drift detection (allowed by #6372 acceptance criteria)
  - append/update decision outcomes plus active drift alerts
  - JSON drift receipts under ``.aragora/review-queue/drift`` when a repo root
    is available
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aragora.persistence.db_config import get_default_data_dir

AUTO_HANDLE_PATH_FIRE_AND_FORGET = "fire_and_forget"
AUTO_HANDLE_PATH_ADMIN_MERGE_ALLOWED = "admin_merge_allowed"

DEFAULT_WINDOW_DAYS = 30
DEFAULT_MIN_SAMPLES = 20
DEFAULT_MIN_SUCCESS_RATE = 0.95
DEFAULT_DRIFT_THRESHOLD = 0.05

OUTCOME_SUCCESS = "success"
OUTCOME_HUMAN_OVERRIDE = "merge_then_human_override"
OUTCOME_REVERT = "merge_then_revert"
OUTCOME_INCIDENT = "merge_then_incident"

_FAILURE_OUTCOMES = frozenset(
    {
        OUTCOME_HUMAN_OVERRIDE,
        OUTCOME_REVERT,
        OUTCOME_INCIDENT,
    }
)


@dataclass(frozen=True, slots=True)
class AutoHandleClassSummary:
    auto_handle_path: str
    decision_class: str
    window_days: int
    total_samples: int
    successes: int
    failures: int
    success_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AutoHandleGateDecision:
    allowed: bool
    auto_handle_path: str
    decision_class: str
    reason: str
    summary: AutoHandleClassSummary
    active_drift_alert: bool = False
    warmup_active: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["summary"] = self.summary.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class AutoHandleDriftAlert:
    alert_id: str
    auto_handle_path: str
    decision_class: str
    previous_success_rate: float | None
    current_success_rate: float | None
    window_days: int
    min_samples: int
    min_success_rate: float
    drift_threshold: float
    detected_at: float
    remediation_action: str
    receipt_path: str | None = None
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bucket_count(value: int) -> str:
    if value <= 1:
        return "1"
    if value <= 3:
        return "2-3"
    if value <= 6:
        return "4-6"
    return "7+"


def classify_scope(paths: list[str]) -> str:
    roots: list[str] = []
    for raw in paths:
        text = str(raw or "").strip().strip("/")
        if not text:
            continue
        head = text.split("/", 1)[0]
        roots.append(head or "root")
    unique = sorted(dict.fromkeys(roots))
    if not unique:
        return "unknown"
    return "+".join(unique[:3]) + ("+more" if len(unique) > 3 else "")


def fingerprint_low_risk_class(
    *,
    changed_files: list[str],
    review_tier: int | None,
    lane_count: int,
) -> str:
    return (
        f"tier={review_tier if review_tier is not None else 'unknown'}"
        f"|lanes={bucket_count(max(lane_count, 0))}"
        f"|files={bucket_count(len(changed_files))}"
        f"|scope={classify_scope(changed_files)}"
    )


def fingerprint_admin_merge_class(
    *,
    base_branch: str | None,
    required_checks_count: int,
    target_kind: str | None,
) -> str:
    return (
        f"base={str(base_branch or 'unknown').strip() or 'unknown'}"
        f"|checks={bucket_count(max(required_checks_count, 0))}"
        f"|target={str(target_kind or 'unknown').strip() or 'unknown'}"
    )


def auto_handle_decision_id(*, auto_handle_path: str, pr_url: str, decision_class: str) -> str:
    payload = "\x1f".join(
        (
            str(auto_handle_path or "").strip(),
            str(pr_url or "").strip(),
            str(decision_class or "").strip(),
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"{str(auto_handle_path or '').strip()}:{digest}"


class AutoHandleCalibrationStore:
    """SQLite-backed decision/outcome history for auto-handle paths."""

    _schema_lock = threading.Lock()

    def __init__(
        self,
        *,
        db_path: str | None = None,
        window_days: int = DEFAULT_WINDOW_DAYS,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        min_success_rate: float = DEFAULT_MIN_SUCCESS_RATE,
        drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
    ) -> None:
        if db_path is None:
            data_dir = get_default_data_dir()
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str((data_dir / "auto_handle_calibration.db").resolve())
        self.db_path = db_path
        self.window_days = int(window_days)
        self.min_samples = int(min_samples)
        self.min_success_rate = float(min_success_rate)
        self.drift_threshold = float(drift_threshold)
        self._persistent_conn: sqlite3.Connection | None = None
        self._thread_local = threading.local()
        if db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._configure_conn(self._persistent_conn)
        with self._schema_lock:
            self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._persistent_conn is not None:
            return self._persistent_conn
        conn = getattr(self._thread_local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            self._configure_conn(conn)
            self._thread_local.conn = conn
        return conn

    def _configure_conn(self, conn: sqlite3.Connection) -> None:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        if self.db_path != ":memory:":
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                # Older SQLite builds can reject WAL transitions; keep the store usable.
                pass

    def _close_conn(self, conn: sqlite3.Connection) -> None:
        if conn is self._persistent_conn:
            return
        if conn is getattr(self._thread_local, "conn", None):
            return
        if conn is not None:
            conn.close()

    def _init_schema(self) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS auto_handle_decisions (
                    decision_id TEXT PRIMARY KEY,
                    auto_handle_path TEXT NOT NULL,
                    decision_class TEXT NOT NULL,
                    pr_url TEXT NOT NULL DEFAULT '',
                    pr_number INTEGER,
                    outcome TEXT NOT NULL,
                    decided_at REAL NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_auto_handle_decisions_window
                ON auto_handle_decisions(auto_handle_path, decision_class, decided_at DESC)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS auto_handle_drift_alerts (
                    auto_handle_path TEXT NOT NULL,
                    decision_class TEXT NOT NULL,
                    alert_id TEXT NOT NULL,
                    previous_success_rate REAL,
                    current_success_rate REAL,
                    window_days INTEGER NOT NULL,
                    min_samples INTEGER NOT NULL,
                    min_success_rate REAL NOT NULL,
                    drift_threshold REAL NOT NULL,
                    detected_at REAL NOT NULL,
                    remediation_action TEXT NOT NULL,
                    receipt_path TEXT,
                    active INTEGER NOT NULL DEFAULT 1,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (auto_handle_path, decision_class)
                )
                """
            )
            conn.commit()
        finally:
            self._close_conn(conn)

    def summarize_class(
        self,
        *,
        auto_handle_path: str,
        decision_class: str,
        window_days: int | None = None,
    ) -> AutoHandleClassSummary:
        days = int(window_days or self.window_days)
        cutoff = time.time() - (days * 86400)
        conn = self._get_conn()
        try:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_samples,
                    SUM(CASE WHEN outcome = ? THEN 1 ELSE 0 END) AS successes,
                    SUM(CASE WHEN outcome IN (?, ?, ?) THEN 1 ELSE 0 END) AS failures
                FROM auto_handle_decisions
                WHERE auto_handle_path = ?
                  AND decision_class = ?
                  AND decided_at >= ?
                """,
                (
                    OUTCOME_SUCCESS,
                    OUTCOME_HUMAN_OVERRIDE,
                    OUTCOME_REVERT,
                    OUTCOME_INCIDENT,
                    auto_handle_path,
                    decision_class,
                    cutoff,
                ),
            ).fetchone()
        finally:
            self._close_conn(conn)

        total_samples = int(row["total_samples"] or 0) if row is not None else 0
        successes = int(row["successes"] or 0) if row is not None else 0
        failures = int(row["failures"] or 0) if row is not None else 0
        success_rate = (successes / total_samples) if total_samples else None
        return AutoHandleClassSummary(
            auto_handle_path=auto_handle_path,
            decision_class=decision_class,
            window_days=days,
            total_samples=total_samples,
            successes=successes,
            failures=failures,
            success_rate=success_rate,
        )

    def get_active_alert(
        self,
        *,
        auto_handle_path: str,
        decision_class: str,
    ) -> AutoHandleDriftAlert | None:
        conn = self._get_conn()
        try:
            row = conn.execute(
                """
                SELECT * FROM auto_handle_drift_alerts
                WHERE auto_handle_path = ? AND decision_class = ? AND active = 1
                LIMIT 1
                """,
                (auto_handle_path, decision_class),
            ).fetchone()
        finally:
            self._close_conn(conn)
        return self._alert_from_row(row) if row is not None else None

    def list_active_alerts(self, *, limit: int = 5) -> list[AutoHandleDriftAlert]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT * FROM auto_handle_drift_alerts
                WHERE active = 1
                ORDER BY detected_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        finally:
            self._close_conn(conn)
        return [self._alert_from_row(row) for row in rows]

    def evaluate_gate(
        self,
        *,
        auto_handle_path: str,
        decision_class: str,
    ) -> AutoHandleGateDecision:
        summary = self.summarize_class(
            auto_handle_path=auto_handle_path,
            decision_class=decision_class,
        )
        active_alert = self.get_active_alert(
            auto_handle_path=auto_handle_path,
            decision_class=decision_class,
        )
        if active_alert is not None:
            return AutoHandleGateDecision(
                allowed=False,
                auto_handle_path=auto_handle_path,
                decision_class=decision_class,
                reason=(
                    "class is currently narrowed by drift gating until its "
                    "rolling success rate recovers"
                ),
                summary=summary,
                active_drift_alert=True,
            )
        if summary.total_samples < self.min_samples:
            if summary.failures == 0:
                return AutoHandleGateDecision(
                    allowed=True,
                    auto_handle_path=auto_handle_path,
                    decision_class=decision_class,
                    reason=(
                        f"class is in calibration warm-up: {summary.total_samples} < "
                        f"{self.min_samples} samples in the last {summary.window_days}d; "
                        "allowing auto-handle while outcome history is seeded"
                    ),
                    summary=summary,
                    warmup_active=True,
                )
            return AutoHandleGateDecision(
                allowed=False,
                auto_handle_path=auto_handle_path,
                decision_class=decision_class,
                reason=(
                    f"class remains uncalibrated after {summary.failures} recorded failure(s): "
                    f"{summary.total_samples} < {self.min_samples} in the last "
                    f"{summary.window_days}d"
                ),
                summary=summary,
            )
        if summary.success_rate is None or summary.success_rate < self.min_success_rate:
            rate = summary.success_rate if summary.success_rate is not None else 0.0
            return AutoHandleGateDecision(
                allowed=False,
                auto_handle_path=auto_handle_path,
                decision_class=decision_class,
                reason=(
                    f"class success rate {rate:.1%} is below required {self.min_success_rate:.1%}"
                ),
                summary=summary,
            )
        return AutoHandleGateDecision(
            allowed=True,
            auto_handle_path=auto_handle_path,
            decision_class=decision_class,
            reason="class is calibrated for auto-handle",
            summary=summary,
        )

    def record_outcome(
        self,
        *,
        decision_id: str,
        auto_handle_path: str,
        decision_class: str,
        outcome: str,
        pr_url: str = "",
        pr_number: int | None = None,
        metadata: dict[str, Any] | None = None,
        repo_root: Path | None = None,
    ) -> dict[str, Any]:
        if outcome not in _FAILURE_OUTCOMES | {OUTCOME_SUCCESS}:
            raise ValueError(f"Unsupported auto-handle outcome: {outcome}")

        normalized_pr_url = str(pr_url or "").strip()
        previous_summary = self.summarize_class(
            auto_handle_path=auto_handle_path,
            decision_class=decision_class,
        )
        conn = self._get_conn()
        inserted = False
        existing_outcome: str | None = None
        try:
            cursor = conn.execute(
                """
                INSERT INTO auto_handle_decisions (
                    decision_id,
                    auto_handle_path,
                    decision_class,
                    pr_url,
                    pr_number,
                    outcome,
                    decided_at,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(decision_id) DO NOTHING
                """,
                (
                    decision_id,
                    auto_handle_path,
                    decision_class,
                    normalized_pr_url,
                    pr_number,
                    outcome,
                    time.time(),
                    json.dumps(metadata or {}, sort_keys=True),
                ),
            )
            inserted = cursor.rowcount > 0
            if inserted:
                conn.commit()
            else:
                row = conn.execute(
                    """
                    SELECT auto_handle_path, decision_class, pr_url, outcome
                    FROM auto_handle_decisions
                    WHERE decision_id = ?
                    """,
                    (decision_id,),
                ).fetchone()
                if row is None:
                    raise RuntimeError(
                        f"Failed to load existing auto-handle decision for duplicate id {decision_id!r}"
                    )
                existing_outcome = str(row["outcome"] or "").strip() or None
                if (
                    str(row["auto_handle_path"] or "").strip() != str(auto_handle_path).strip()
                    or str(row["decision_class"] or "").strip() != str(decision_class).strip()
                    or str(row["pr_url"] or "").strip() != normalized_pr_url
                    or str(row["outcome"] or "").strip() != str(outcome).strip()
                ):
                    raise ValueError(
                        "Refusing to overwrite existing auto-handle decision with conflicting outcome "
                        f"for {decision_id!r}"
                    )
        finally:
            self._close_conn(conn)

        if not inserted:
            active_alert = self.get_active_alert(
                auto_handle_path=auto_handle_path,
                decision_class=decision_class,
            )
            return {
                "summary": previous_summary.to_dict(),
                "alert": active_alert.to_dict() if active_alert is not None else None,
                "recovered": False,
                "recorded": False,
                "duplicate": True,
                "existing_outcome": existing_outcome,
            }

        current_summary = self.summarize_class(
            auto_handle_path=auto_handle_path,
            decision_class=decision_class,
        )
        active_alert = self.get_active_alert(
            auto_handle_path=auto_handle_path,
            decision_class=decision_class,
        )
        previous_rate = previous_summary.success_rate
        current_rate = current_summary.success_rate
        drop = (
            (previous_rate - current_rate)
            if previous_rate is not None and current_rate is not None
            else 0.0
        )
        should_block = (
            current_summary.total_samples >= self.min_samples
            and current_rate is not None
            and (current_rate < self.min_success_rate or drop >= self.drift_threshold)
        )
        recovered = (
            active_alert is not None
            and current_summary.total_samples >= self.min_samples
            and current_rate is not None
            and current_rate >= self.min_success_rate
        )

        alert: AutoHandleDriftAlert | None = None
        if should_block and active_alert is None:
            alert = self._upsert_alert(
                auto_handle_path=auto_handle_path,
                decision_class=decision_class,
                previous_success_rate=previous_rate,
                current_success_rate=current_rate,
                repo_root=repo_root,
            )
        elif recovered and active_alert is not None:
            self._clear_alert(
                auto_handle_path=auto_handle_path,
                decision_class=decision_class,
            )
        elif active_alert is not None:
            alert = active_alert

        return {
            "summary": current_summary.to_dict(),
            "alert": alert.to_dict() if alert is not None else None,
            "recovered": bool(recovered),
            "recorded": True,
            "duplicate": False,
            "existing_outcome": None,
        }

    def _upsert_alert(
        self,
        *,
        auto_handle_path: str,
        decision_class: str,
        previous_success_rate: float | None,
        current_success_rate: float | None,
        repo_root: Path | None,
    ) -> AutoHandleDriftAlert:
        alert = AutoHandleDriftAlert(
            alert_id=f"auto-handle-drift-{uuid.uuid4().hex[:12]}",
            auto_handle_path=auto_handle_path,
            decision_class=decision_class,
            previous_success_rate=previous_success_rate,
            current_success_rate=current_success_rate,
            window_days=self.window_days,
            min_samples=self.min_samples,
            min_success_rate=self.min_success_rate,
            drift_threshold=self.drift_threshold,
            detected_at=time.time(),
            remediation_action="require_human_review_for_class",
            receipt_path=None,
            active=True,
        )
        if repo_root is not None:
            alert = AutoHandleDriftAlert(
                **{
                    **alert.to_dict(),
                    "receipt_path": self._write_receipt(alert=alert, repo_root=repo_root),
                }
            )
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO auto_handle_drift_alerts (
                    auto_handle_path,
                    decision_class,
                    alert_id,
                    previous_success_rate,
                    current_success_rate,
                    window_days,
                    min_samples,
                    min_success_rate,
                    drift_threshold,
                    detected_at,
                    remediation_action,
                    receipt_path,
                    active,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    alert.auto_handle_path,
                    alert.decision_class,
                    alert.alert_id,
                    alert.previous_success_rate,
                    alert.current_success_rate,
                    alert.window_days,
                    alert.min_samples,
                    alert.min_success_rate,
                    alert.drift_threshold,
                    alert.detected_at,
                    alert.remediation_action,
                    alert.receipt_path,
                    json.dumps({}, sort_keys=True),
                ),
            )
            conn.commit()
        finally:
            self._close_conn(conn)
        return alert

    def _clear_alert(self, *, auto_handle_path: str, decision_class: str) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                """
                UPDATE auto_handle_drift_alerts
                SET active = 0
                WHERE auto_handle_path = ? AND decision_class = ?
                """,
                (auto_handle_path, decision_class),
            )
            conn.commit()
        finally:
            self._close_conn(conn)

    def _write_receipt(self, *, alert: AutoHandleDriftAlert, repo_root: Path) -> str:
        receipts_dir = repo_root / ".aragora" / "review-queue" / "drift"
        receipts_dir.mkdir(parents=True, exist_ok=True)
        path = receipts_dir / f"{alert.alert_id}.json"
        path.write_text(json.dumps(alert.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return str(path)

    @staticmethod
    def _alert_from_row(row: sqlite3.Row) -> AutoHandleDriftAlert:
        return AutoHandleDriftAlert(
            alert_id=str(row["alert_id"] or ""),
            auto_handle_path=str(row["auto_handle_path"] or ""),
            decision_class=str(row["decision_class"] or ""),
            previous_success_rate=row["previous_success_rate"],
            current_success_rate=row["current_success_rate"],
            window_days=int(row["window_days"] or 0),
            min_samples=int(row["min_samples"] or 0),
            min_success_rate=float(row["min_success_rate"] or 0.0),
            drift_threshold=float(row["drift_threshold"] or 0.0),
            detected_at=float(row["detected_at"] or 0.0),
            remediation_action=str(row["remediation_action"] or ""),
            receipt_path=str(row["receipt_path"] or "") or None,
            active=bool(row["active"]),
        )


__all__ = [
    "AUTO_HANDLE_PATH_ADMIN_MERGE_ALLOWED",
    "AUTO_HANDLE_PATH_FIRE_AND_FORGET",
    "AutoHandleCalibrationStore",
    "AutoHandleClassSummary",
    "AutoHandleDriftAlert",
    "AutoHandleGateDecision",
    "DEFAULT_DRIFT_THRESHOLD",
    "DEFAULT_MIN_SAMPLES",
    "DEFAULT_MIN_SUCCESS_RATE",
    "DEFAULT_WINDOW_DAYS",
    "OUTCOME_HUMAN_OVERRIDE",
    "OUTCOME_INCIDENT",
    "OUTCOME_REVERT",
    "OUTCOME_SUCCESS",
    "auto_handle_decision_id",
    "fingerprint_admin_merge_class",
    "fingerprint_low_risk_class",
]
