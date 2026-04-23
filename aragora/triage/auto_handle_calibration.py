"""SQLite store primitives for auto-handle calibration (#6372, PR A of #6448 split).

This module is the narrow persistence layer that underpins the calibration
gate named in ``docs/THESIS.md`` Commitment 1 for the two auto-handle paths:

  - ``fire_and_forget`` low-risk merge in :mod:`aragora.swarm.tranche_integrate`
  - ``admin_merge_allowed`` review-gate bypass in :mod:`aragora.ralph.supervisor`

Scope of PR A
-------------

PR A ships **only the store primitives**: decision-class fingerprints, the
SQLite schema, outcome recording, drift-alert upsert/clear, and JSON drift
receipts. The gate integration (``evaluate_gate``), the call-sites in
``tranche_integrate`` / ``ralph/supervisor``, the CLI drift-alert surface,
and F4 externally-merged seeding are intentionally deferred to PR B and
PR C so this change stays bounded and reviewable.

Connection lifecycle
--------------------

File-backed stores open and close a fresh connection per operation via
:func:`contextlib.closing`. This avoids the thread-local ownership
ambiguity flagged by the 5th Mode 3 panel on #6448: every call is fully
self-contained, no file descriptors leak across thread boundaries, and
concurrent access is coordinated purely by SQLite's busy-timeout + WAL
journal. ``:memory:`` stores keep a single persistent connection guarded
by a per-instance :class:`threading.Lock` because each
``sqlite3.connect(":memory:")`` opens a *distinct* database and tests
that share state across threads need a single shared handle.

Error surface
-------------

Write paths (``record_outcome``, ``_upsert_alert``, ``_clear_alert``,
schema setup) translate :class:`sqlite3.Error` subclasses into
:class:`AutoHandleStoreError` so callers can explicitly choose whether
to tolerate persistence failures rather than having them silently
swallowed. Read paths let :class:`sqlite3.Error` bubble directly.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aragora.persistence.db_config import get_default_data_dir

__all__ = [
    "AUTO_HANDLE_PATH_ADMIN_MERGE_ALLOWED",
    "AUTO_HANDLE_PATH_FIRE_AND_FORGET",
    "AutoHandleCalibrationStore",
    "AutoHandleClassSummary",
    "AutoHandleDriftAlert",
    "AutoHandleGateDecision",
    "AutoHandleStoreError",
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


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AutoHandleStoreError(RuntimeError):
    """Raised when the auto-handle calibration store fails to persist data.

    Write paths wrap :class:`sqlite3.Error` (and its subclasses) in this
    exception so callers can filter persistence errors from input-validation
    errors (:class:`ValueError`). The caller, not the store, gets to decide
    whether a persistence failure is acceptable — for example, a
    fire-and-forget logger might log-and-continue, whereas a gate-enforcement
    path would propagate.
    """


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


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
    """Return type for the (deferred) ``evaluate_gate`` method.

    The gate evaluator is deferred to PR B (per the #6448 split); the
    dataclass is defined here so callers that import the module see a
    stable surface once PR B lands.
    """

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


# ---------------------------------------------------------------------------
# Pure helpers (no I/O)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class AutoHandleCalibrationStore:
    """SQLite-backed decision/outcome history for auto-handle paths.

    See module docstring for the connection-lifecycle rationale and the
    write-path error contract.
    """

    # Serialises schema creation across ``AutoHandleCalibrationStore``
    # instances — multiple instances pointing at the same file must not
    # race to ``CREATE TABLE IF NOT EXISTS``.
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

        # ``:memory:`` databases cannot be shared across ``sqlite3.connect``
        # calls, so we keep one persistent connection and guard it with a
        # per-instance lock. File-backed stores don't need this — each
        # operation opens and closes a fresh connection.
        self._is_memory = db_path == ":memory:"
        self._persistent_lock = threading.Lock()
        self._persistent_conn: sqlite3.Connection | None = None
        if self._is_memory:
            self._persistent_conn = sqlite3.connect(
                ":memory:",
                check_same_thread=False,
                timeout=30.0,
            )
            self._configure_conn(self._persistent_conn)

        with self._schema_lock:
            self._init_schema()

    # -- Connection plumbing ------------------------------------------------

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a SQLite connection valid for a single operation.

        For ``:memory:`` stores the single persistent connection is shared
        across callers (serialised by ``_persistent_lock``) because each
        ``sqlite3.connect(":memory:")`` opens a distinct database. For
        file-backed stores a fresh connection is opened per call and
        closed when the context exits, eliminating thread-local leaks.
        """
        if self._persistent_conn is not None:
            with self._persistent_lock:
                yield self._persistent_conn
            return
        with closing(sqlite3.connect(self.db_path, timeout=30.0)) as conn:
            self._configure_conn(conn)
            yield conn

    def _configure_conn(self, conn: sqlite3.Connection) -> None:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        if not self._is_memory:
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                # Older SQLite builds can reject WAL transitions; keep the
                # store usable on legacy platforms.
                pass

    # -- Schema -------------------------------------------------------------

    def _init_schema(self) -> None:
        try:
            with self._connection() as conn:
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
        except sqlite3.Error as exc:
            raise AutoHandleStoreError(
                f"Failed to initialise auto-handle calibration schema at {self.db_path!r}"
            ) from exc

    # -- Reads --------------------------------------------------------------

    def summarize_class(
        self,
        *,
        auto_handle_path: str,
        decision_class: str,
        window_days: int | None = None,
    ) -> AutoHandleClassSummary:
        days = int(window_days if window_days is not None else self.window_days)
        cutoff = time.time() - (days * 86400)
        with self._connection() as conn:
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
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM auto_handle_drift_alerts
                WHERE auto_handle_path = ? AND decision_class = ? AND active = 1
                LIMIT 1
                """,
                (auto_handle_path, decision_class),
            ).fetchone()
        return self._alert_from_row(row) if row is not None else None

    def list_active_alerts(self, *, limit: int = 5) -> list[AutoHandleDriftAlert]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM auto_handle_drift_alerts
                WHERE active = 1
                ORDER BY detected_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [self._alert_from_row(row) for row in rows]

    # -- Writes -------------------------------------------------------------

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
        """Record a decision outcome and update drift-alert state.

        Raises:
            ValueError: when ``outcome`` is not a supported value or when
                a duplicate decision id is recorded with a conflicting
                outcome (data-integrity error, not a persistence error).
            AutoHandleStoreError: when any underlying SQLite operation
                fails. Callers should explicitly catch this if they want
                to tolerate persistence failures — the store never
                silently swallows DB errors.
        """
        if outcome not in _FAILURE_OUTCOMES | {OUTCOME_SUCCESS}:
            raise ValueError(f"Unsupported auto-handle outcome: {outcome}")

        normalized_pr_url = str(pr_url or "").strip()

        try:
            previous_summary = self.summarize_class(
                auto_handle_path=auto_handle_path,
                decision_class=decision_class,
            )

            inserted = False
            existing_outcome: str | None = None
            with self._connection() as conn:
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
                        raise AutoHandleStoreError(
                            "Failed to load existing auto-handle decision for "
                            f"duplicate id {decision_id!r}"
                        )
                    existing_outcome = str(row["outcome"] or "").strip() or None
                    if (
                        str(row["auto_handle_path"] or "").strip() != str(auto_handle_path).strip()
                        or str(row["decision_class"] or "").strip() != str(decision_class).strip()
                        or str(row["pr_url"] or "").strip() != normalized_pr_url
                        or str(row["outcome"] or "").strip() != str(outcome).strip()
                    ):
                        raise ValueError(
                            "Refusing to overwrite existing auto-handle decision with "
                            f"conflicting outcome for {decision_id!r}"
                        )

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
                and outcome == OUTCOME_SUCCESS
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
        except sqlite3.Error as exc:
            raise AutoHandleStoreError(
                f"SQLite failure while recording outcome for {decision_id!r}"
            ) from exc

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
        try:
            with self._connection() as conn:
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
        except sqlite3.Error as exc:
            raise AutoHandleStoreError(
                f"Failed to upsert drift alert for {auto_handle_path}/{decision_class}"
            ) from exc
        return alert

    def _clear_alert(self, *, auto_handle_path: str, decision_class: str) -> None:
        try:
            with self._connection() as conn:
                conn.execute(
                    """
                    UPDATE auto_handle_drift_alerts
                    SET active = 0
                    WHERE auto_handle_path = ? AND decision_class = ?
                    """,
                    (auto_handle_path, decision_class),
                )
                conn.commit()
        except sqlite3.Error as exc:
            raise AutoHandleStoreError(
                f"Failed to clear drift alert for {auto_handle_path}/{decision_class}"
            ) from exc

    def _write_receipt(self, *, alert: AutoHandleDriftAlert, repo_root: Path) -> str:
        receipts_dir = repo_root / ".aragora" / "review-queue" / "drift"
        receipts_dir.mkdir(parents=True, exist_ok=True)
        path = receipts_dir / f"{alert.alert_id}.json"
        path.write_text(
            json.dumps(alert.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
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
