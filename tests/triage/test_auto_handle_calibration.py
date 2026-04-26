"""Tests for :mod:`aragora.triage.auto_handle_calibration` store primitives.

Covers PR A of the #6448 split: schema creation, outcome recording,
window-filtered summaries, drift-alert upsert/clear, receipt writes,
thread-safety under concurrent writes, and the write-path error surface
(``AutoHandleStoreError`` wrapping ``sqlite3.Error``).

These tests deliberately use real ``sqlite3`` against ``tmp_path`` — no
mocks. The gate integration (``evaluate_gate``), CLI surface, and
call-site integration are out of scope for PR A.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest

from aragora.triage.auto_handle_calibration import (
    AUTO_HANDLE_PATH_ADMIN_MERGE_ALLOWED,
    AUTO_HANDLE_PATH_FIRE_AND_FORGET,
    OUTCOME_HUMAN_OVERRIDE,
    OUTCOME_SUCCESS,
    SCHEMA_VERSION,
    AutoHandleCalibrationStore,
    AutoHandleStoreError,
    auto_handle_decision_id,
    fingerprint_admin_merge_class,
    fingerprint_low_risk_class,
)

# A stable, non-default decision class used throughout the suite.
TEST_CLASS = "tier=1|lanes=1|files=1|scope=aragora"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> AutoHandleCalibrationStore:
    """Fresh file-backed store rooted at ``tmp_path``."""
    return AutoHandleCalibrationStore(
        db_path=str(tmp_path / "auto_handle_calibration.db"),
        min_samples=2,
        min_success_rate=0.9,
        drift_threshold=0.05,
    )


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------


class TestSchemaCreation:
    def test_schema_tables_exist(self, tmp_path: Path) -> None:
        db_path = tmp_path / "fresh.db"
        AutoHandleCalibrationStore(db_path=str(db_path))
        assert db_path.exists()
        with closing(sqlite3.connect(str(db_path))) as conn:
            names = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert "auto_handle_decisions" in names
        assert "auto_handle_drift_alerts" in names

    def test_schema_is_idempotent(self, tmp_path: Path) -> None:
        """Creating the store twice against the same file must not error."""
        db_path = str(tmp_path / "twice.db")
        first = AutoHandleCalibrationStore(db_path=db_path)
        second = AutoHandleCalibrationStore(db_path=db_path)
        # Recording through the second instance works — shared schema.
        result = second.record_outcome(
            decision_id="idem-1",
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
            outcome=OUTCOME_SUCCESS,
            pr_url="https://example.com/pr/1",
        )
        assert result["recorded"] is True
        # First instance sees the persisted row.
        summary = first.summarize_class(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
        )
        assert summary.total_samples == 1

    def test_in_memory_store_initialises(self) -> None:
        """``:memory:`` stores create the schema via the shared conn."""
        store = AutoHandleCalibrationStore(db_path=":memory:")
        summary = store.summarize_class(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
        )
        assert summary.total_samples == 0
        assert summary.success_rate is None

    def test_schema_version_stamp_is_set_on_fresh_store(self, tmp_path: Path) -> None:
        """Fresh stores land at ``PRAGMA user_version = SCHEMA_VERSION``."""
        db_path = tmp_path / "versioned.db"
        AutoHandleCalibrationStore(db_path=str(db_path))
        with closing(sqlite3.connect(str(db_path))) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == SCHEMA_VERSION
        assert SCHEMA_VERSION > 0  # guard against accidental downgrade

    def test_schema_version_mismatch_raises(self, tmp_path: Path) -> None:
        """A DB with a future user_version refuses to open.

        The 8/8 Mode 3 panel on PR #6468 asked for a schema landmark so
        future migration work has something to pivot on. A store
        stamped at a higher user_version than this code understands
        must raise, not silently run against an unknown schema.
        """
        db_path = tmp_path / "future.db"
        # Seed a valid v1 schema first.
        AutoHandleCalibrationStore(db_path=str(db_path))
        # Then simulate a forward-incompatible stamp on disk.
        with closing(sqlite3.connect(str(db_path))) as conn:
            conn.execute("PRAGMA user_version = 999")
            conn.commit()

        with pytest.raises(AutoHandleStoreError, match="user_version"):
            AutoHandleCalibrationStore(db_path=str(db_path))

    def test_schema_version_zero_is_upgraded_transparently(self, tmp_path: Path) -> None:
        """Legacy DBs without user_version stamp are treated as v1.

        A database written by the pre-versioning iteration of this
        module has ``PRAGMA user_version = 0`` by default and the same
        table shape as v1. Opening it with the new code should stamp
        it to ``SCHEMA_VERSION`` without raising or losing data.
        """
        db_path = tmp_path / "legacy.db"
        # Hand-roll a legacy-shaped DB with user_version=0 and a seed row.
        with closing(sqlite3.connect(str(db_path))) as conn:
            conn.executescript(
                """
                CREATE TABLE auto_handle_decisions (
                    decision_id TEXT PRIMARY KEY,
                    auto_handle_path TEXT NOT NULL,
                    decision_class TEXT NOT NULL,
                    pr_url TEXT NOT NULL DEFAULT '',
                    pr_number INTEGER,
                    outcome TEXT NOT NULL,
                    decided_at REAL NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE auto_handle_drift_alerts (
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
                );
                """
            )
            conn.execute(
                """
                INSERT INTO auto_handle_decisions
                (decision_id, auto_handle_path, decision_class, pr_url,
                 pr_number, outcome, decided_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "legacy-1",
                    AUTO_HANDLE_PATH_FIRE_AND_FORGET,
                    TEST_CLASS,
                    "",
                    None,
                    OUTCOME_SUCCESS,
                    time.time(),
                    "{}",
                ),
            )
            conn.commit()

        # Opening it with the new code stamps user_version and preserves data.
        store = AutoHandleCalibrationStore(db_path=str(db_path))
        summary = store.summarize_class(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
        )
        assert summary.total_samples == 1
        with closing(sqlite3.connect(str(db_path))) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# record_outcome — happy path
# ---------------------------------------------------------------------------


class TestRecordOutcomeHappyPath:
    def test_success_is_persisted_and_summarised(self, store: AutoHandleCalibrationStore) -> None:
        result = store.record_outcome(
            decision_id="dec-1",
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
            outcome=OUTCOME_SUCCESS,
            pr_url="https://github.com/synaptent/aragora/pull/1",
            pr_number=1,
            metadata={"reviewer": "droid"},
        )
        assert result["recorded"] is True
        assert result["duplicate"] is False
        assert result["existing_outcome"] is None

        summary = store.summarize_class(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
        )
        assert summary.total_samples == 1
        assert summary.successes == 1
        assert summary.failures == 0
        assert summary.success_rate == 1.0

    def test_metadata_is_round_tripped_to_db(self, store: AutoHandleCalibrationStore) -> None:
        store.record_outcome(
            decision_id="dec-meta",
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
            outcome=OUTCOME_SUCCESS,
            metadata={"a": 1, "b": "two"},
        )
        with closing(sqlite3.connect(store.db_path)) as conn:
            row = conn.execute(
                "SELECT metadata_json FROM auto_handle_decisions WHERE decision_id = ?",
                ("dec-meta",),
            ).fetchone()
        assert row is not None
        payload = json.loads(row[0])
        assert payload == {"a": 1, "b": "two"}

    def test_unsupported_outcome_raises_value_error(
        self, store: AutoHandleCalibrationStore
    ) -> None:
        with pytest.raises(ValueError, match="Unsupported auto-handle outcome"):
            store.record_outcome(
                decision_id="dec-bad",
                auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
                decision_class=TEST_CLASS,
                outcome="not-a-real-outcome",
            )
        # Nothing persisted for invalid outcome.
        summary = store.summarize_class(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
        )
        assert summary.total_samples == 0

    def test_idempotent_duplicate_returns_existing(self, store: AutoHandleCalibrationStore) -> None:
        store.record_outcome(
            decision_id="dup-1",
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
            outcome=OUTCOME_SUCCESS,
            pr_url="https://example.com/pr/42",
        )
        second = store.record_outcome(
            decision_id="dup-1",
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
            outcome=OUTCOME_SUCCESS,
            pr_url="https://example.com/pr/42",
        )
        assert second["recorded"] is False
        assert second["duplicate"] is True
        assert second["existing_outcome"] == OUTCOME_SUCCESS
        # Count still 1 — no double-insert.
        summary = store.summarize_class(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
        )
        assert summary.total_samples == 1

    def test_conflicting_duplicate_raises_value_error(
        self, store: AutoHandleCalibrationStore
    ) -> None:
        store.record_outcome(
            decision_id="conflict-1",
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
            outcome=OUTCOME_SUCCESS,
        )
        with pytest.raises(ValueError, match="conflicting outcome"):
            store.record_outcome(
                decision_id="conflict-1",
                auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
                decision_class=TEST_CLASS,
                outcome=OUTCOME_HUMAN_OVERRIDE,
            )


# ---------------------------------------------------------------------------
# record_outcome — DB error path (silent-swallow regression from #6448 panel)
# ---------------------------------------------------------------------------


class TestRecordOutcomeDbErrorPath:
    def test_missing_table_raises_store_error(self, store: AutoHandleCalibrationStore) -> None:
        """Dropping the decisions table after init must surface, not silent-swallow.

        The 5th Mode 3 panel on #6448 flagged silent swallowing in the write
        path. This test is the regression guard: a corrupted/locked DB must
        raise ``AutoHandleStoreError`` with the underlying ``sqlite3.Error``
        as ``__cause__`` so callers can filter persistence errors from
        input-validation errors.
        """
        with closing(sqlite3.connect(store.db_path)) as conn:
            conn.execute("DROP TABLE auto_handle_decisions")
            conn.commit()
        with pytest.raises(AutoHandleStoreError) as excinfo:
            store.record_outcome(
                decision_id="err-1",
                auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
                decision_class=TEST_CLASS,
                outcome=OUTCOME_SUCCESS,
            )
        assert isinstance(excinfo.value.__cause__, sqlite3.Error)

    def test_init_against_invalid_path_raises_store_error(self, tmp_path: Path) -> None:
        """Pointing at a directory (invalid DB target) raises at init."""
        # ``sqlite3.connect(<directory-path>)`` raises OperationalError. We
        # route that through ``_init_schema`` so callers see a typed error.
        bad_path = tmp_path / "subdir"
        bad_path.mkdir()
        with pytest.raises(AutoHandleStoreError):
            AutoHandleCalibrationStore(db_path=str(bad_path))


# ---------------------------------------------------------------------------
# WAL mode — required, no silent fallback
# ---------------------------------------------------------------------------


class TestWalModeRequired:
    """The 8/8 Mode 3 panel on PR #6468 flagged the previous silent WAL
    fallback as a deployment bug worth surfacing. A host that cannot
    support WAL (read-only directory, some Docker overlays, exotic VFS)
    changes concurrent reader/writer semantics; we'd rather fail loudly
    at init than discover the behaviour shift in production.
    """

    @pytest.mark.skipif(
        sys.platform.startswith("win"),
        reason="Windows permission model does not behave like POSIX for this check",
    )
    @pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="Running as root bypasses directory-mode restrictions",
    )
    def test_wal_mode_required_raises_on_failure(self, tmp_path: Path) -> None:
        """When WAL cannot be enabled, init raises ``AutoHandleStoreError``.

        We set up a directory we can pre-populate with a DB file, then
        strip write permission. SQLite can still open the existing file
        but cannot create the ``-wal`` / ``-shm`` shared-memory side
        files required for WAL, so ``PRAGMA journal_mode=WAL`` fails.
        """
        db_dir = tmp_path / "wal_blocked"
        db_dir.mkdir()
        db_path = db_dir / "store.db"

        # Pre-create the DB file so SQLite can open it for reading
        # after we chmod the directory to read-only.
        with closing(sqlite3.connect(str(db_path))) as seed:
            seed.execute("CREATE TABLE IF NOT EXISTS touch (x INTEGER)")
            seed.commit()

        original_mode = db_dir.stat().st_mode & 0o777
        os.chmod(db_dir, 0o500)  # r-x: no writes allowed
        try:
            with pytest.raises(AutoHandleStoreError, match="WAL"):
                AutoHandleCalibrationStore(db_path=str(db_path))
        finally:
            os.chmod(db_dir, original_mode)


# ---------------------------------------------------------------------------
# summarize_class — window filtering
# ---------------------------------------------------------------------------


class TestSummarizeClassWindowFiltering:
    def test_pre_window_entries_are_excluded(self, store: AutoHandleCalibrationStore) -> None:
        now = time.time()
        # Insert directly so we can control ``decided_at``.
        with closing(sqlite3.connect(store.db_path)) as conn:
            conn.executemany(
                """
                INSERT INTO auto_handle_decisions
                (decision_id, auto_handle_path, decision_class, pr_url,
                 pr_number, outcome, decided_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        "old-success",
                        AUTO_HANDLE_PATH_FIRE_AND_FORGET,
                        TEST_CLASS,
                        "",
                        None,
                        OUTCOME_SUCCESS,
                        now - 60 * 86400,  # 60 days ago
                        "{}",
                    ),
                    (
                        "old-failure",
                        AUTO_HANDLE_PATH_FIRE_AND_FORGET,
                        TEST_CLASS,
                        "",
                        None,
                        OUTCOME_HUMAN_OVERRIDE,
                        now - 45 * 86400,  # 45 days ago
                        "{}",
                    ),
                    (
                        "recent-success",
                        AUTO_HANDLE_PATH_FIRE_AND_FORGET,
                        TEST_CLASS,
                        "",
                        None,
                        OUTCOME_SUCCESS,
                        now - 1,  # just now
                        "{}",
                    ),
                ],
            )
            conn.commit()

        summary_30d = store.summarize_class(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
            window_days=30,
        )
        assert summary_30d.total_samples == 1
        assert summary_30d.successes == 1
        assert summary_30d.failures == 0

        summary_90d = store.summarize_class(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
            window_days=90,
        )
        assert summary_90d.total_samples == 3
        assert summary_90d.successes == 2
        assert summary_90d.failures == 1

    def test_only_matching_path_and_class_counted(self, store: AutoHandleCalibrationStore) -> None:
        store.record_outcome(
            decision_id="a",
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
            outcome=OUTCOME_SUCCESS,
        )
        store.record_outcome(
            decision_id="b",
            auto_handle_path=AUTO_HANDLE_PATH_ADMIN_MERGE_ALLOWED,
            decision_class=TEST_CLASS,
            outcome=OUTCOME_SUCCESS,
        )
        store.record_outcome(
            decision_id="c",
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class="other-class",
            outcome=OUTCOME_SUCCESS,
        )
        summary = store.summarize_class(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
        )
        assert summary.total_samples == 1


# ---------------------------------------------------------------------------
# Thread-safety — concurrent writes
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_writes_preserve_row_count(self, tmp_path: Path) -> None:
        """Two threads writing distinct decision ids produce exactly N rows.

        File-backed SQLite with ``busy_timeout`` + WAL handles concurrent
        writers; the connection-per-operation pattern prevents cross-thread
        handle sharing. A corrupt DB or swallowed error would surface as a
        row-count mismatch.
        """
        store = AutoHandleCalibrationStore(
            db_path=str(tmp_path / "concurrent.db"),
            min_samples=10_000,  # disable alert-triggering
        )
        total = 40

        def writer(idx: int) -> None:
            store.record_outcome(
                decision_id=f"t-{idx}",
                auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
                decision_class=TEST_CLASS,
                outcome=OUTCOME_SUCCESS,
                pr_number=idx,
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            # ``list`` materialises the iterator so exceptions propagate.
            list(executor.map(writer, range(total)))

        summary = store.summarize_class(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
        )
        assert summary.total_samples == total
        assert summary.successes == total

    def test_thread_writes_from_non_owner_thread_do_not_leak_state(self, tmp_path: Path) -> None:
        """A single write from a fresh thread must not corrupt state.

        Regression guard for the thread-local SQLite connection defect
        flagged on #6448: the previous implementation stored handles on
        ``threading.local`` and leaked across ownership boundaries.
        """
        store = AutoHandleCalibrationStore(
            db_path=str(tmp_path / "non_owner.db"),
            min_samples=10_000,
        )
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                store.record_outcome(
                    decision_id="worker-1",
                    auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
                    decision_class=TEST_CLASS,
                    outcome=OUTCOME_SUCCESS,
                )
            except BaseException as exc:  # noqa: BLE001 — test surface
                errors.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert errors == []
        summary = store.summarize_class(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
        )
        assert summary.total_samples == 1


# ---------------------------------------------------------------------------
# record_outcome — atomicity (TOCTOU guard) and rollback
# ---------------------------------------------------------------------------


class TestRecordOutcomeAtomicity:
    """Regression guards for the 8/8 Mode 3 panel blocker on PR #6468.

    The panel flagged a TOCTOU race in the read-compute-write compound
    inside ``record_outcome`` (the first repair on this branch replaced
    ``threading.local`` with a connection-per-op design, but that made
    the compound straddle multiple connections). These tests pin the
    atomicity contract: the compound now runs inside a single
    ``BEGIN IMMEDIATE`` ... ``COMMIT`` on one connection.
    """

    def test_record_outcome_is_atomic_under_concurrent_callers(self, tmp_path: Path) -> None:
        """Concurrent callers on the same class cannot create split alert state.

        Four threads call ``record_outcome`` simultaneously with
        distinct decision ids but the *same* ``(auto_handle_path,
        decision_class)``. The compound (read → compute → write-outcome
        → upsert/clear alert) must serialise at the SQLite layer so the
        final alert state is consistent with the total outcome history.
        """
        store = AutoHandleCalibrationStore(
            db_path=str(tmp_path / "atomic.db"),
            min_samples=2,
            min_success_rate=0.95,
            drift_threshold=0.05,
        )
        # Seed the class with a baseline success so drift computation
        # has a ``previous_rate`` to compare against.
        store.record_outcome(
            decision_id="seed",
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
            outcome=OUTCOME_SUCCESS,
        )

        # Alternate success/failure so concurrent writers push the class
        # across the alert threshold and back again on every step. That
        # stresses both the upsert and clear branches of the compound.
        outcomes = [
            OUTCOME_HUMAN_OVERRIDE,  # failure → should trigger alert
            OUTCOME_SUCCESS,  # success after failure
            OUTCOME_HUMAN_OVERRIDE,  # failure again
            OUTCOME_SUCCESS,  # success again
        ]
        errors: list[BaseException] = []

        def writer(item: tuple[int, str]) -> None:
            idx, outcome = item
            try:
                store.record_outcome(
                    decision_id=f"concurrent-{idx}",
                    auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
                    decision_class=TEST_CLASS,
                    outcome=outcome,
                    pr_number=idx,
                )
            except BaseException as exc:  # noqa: BLE001 — collect, don't mask
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(writer, list(enumerate(outcomes))))

        assert errors == [], f"unexpected thread errors: {errors!r}"

        # 1. All 4 concurrent writes + 1 seed landed exactly once (no
        #    duplicates, no losses).
        summary = store.summarize_class(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
        )
        assert summary.total_samples == 5

        # 2. Exactly one row per (path, class) survives in the alerts
        #    table — regardless of how many upsert/clear cycles the
        #    threads went through. With the TOCTOU race we were
        #    chasing, concurrent threads could deactivate an alert
        #    another thread had just written, or write two "active"
        #    alerts for the same class (UPSERT + INSERT OR REPLACE
        #    collapses to the second outcome). Both failure modes
        #    would show up as an inconsistency here.
        with closing(sqlite3.connect(store.db_path)) as conn:
            rows = conn.execute(
                """
                SELECT auto_handle_path, decision_class, active
                FROM auto_handle_drift_alerts
                WHERE auto_handle_path = ? AND decision_class = ?
                """,
                (AUTO_HANDLE_PATH_FIRE_AND_FORGET, TEST_CLASS),
            ).fetchall()
        assert len(rows) <= 1  # INSERT OR REPLACE keyed on (path, class)

    def test_record_outcome_rolls_back_on_write_failure(
        self,
        store: AutoHandleCalibrationStore,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A failure inside the compound must leave the store unchanged.

        We monkeypatch ``_upsert_alert_with_conn`` to raise partway
        through, simulating a SQLite failure during the alert write.
        The outer transaction must roll back: neither the outcome row
        nor any alert change may survive.
        """
        # Seed enough failures that the next failure will cross the
        # drift threshold and trigger the upsert path we'll sabotage.
        for idx in range(3):
            store.record_outcome(
                decision_id=f"seed-{idx}",
                auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
                decision_class=TEST_CLASS,
                outcome=OUTCOME_SUCCESS,
            )
        baseline_rows = _count_decision_rows(store.db_path)
        baseline_alerts = _snapshot_alert_rows(store.db_path)
        repo_root = tmp_path / "repo"
        repo_root.mkdir()

        def boom(**kwargs: object) -> None:
            raise sqlite3.OperationalError("synthetic alert upsert failure")

        monkeypatch.setattr(store, "_upsert_alert_with_conn", boom)

        with pytest.raises(AutoHandleStoreError) as excinfo:
            store.record_outcome(
                decision_id="rollback-trigger",
                auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
                decision_class=TEST_CLASS,
                outcome=OUTCOME_HUMAN_OVERRIDE,
                repo_root=repo_root,
            )
        assert isinstance(excinfo.value.__cause__, sqlite3.Error)

        # 1. The outcome row is NOT present — the INSERT was rolled back.
        assert _count_decision_rows(store.db_path) == baseline_rows
        with closing(sqlite3.connect(store.db_path)) as conn:
            row = conn.execute(
                "SELECT 1 FROM auto_handle_decisions WHERE decision_id = ?",
                ("rollback-trigger",),
            ).fetchone()
        assert row is None

        # 2. The alert state is unchanged — no partial upsert survived.
        assert _snapshot_alert_rows(store.db_path) == baseline_alerts

        # 3. Receipts are written only after the DB transaction commits.
        #    A failed alert upsert must not leave an orphan JSON receipt.
        drift_dir = repo_root / ".aragora" / "review-queue" / "drift"
        assert not drift_dir.exists() or list(drift_dir.glob("*.json")) == []


def _count_decision_rows(db_path: str) -> int:
    with closing(sqlite3.connect(db_path)) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM auto_handle_decisions").fetchone()[0])


def _snapshot_alert_rows(db_path: str) -> list[tuple[str, ...]]:
    with closing(sqlite3.connect(db_path)) as conn:
        return [
            tuple(str(cell) for cell in row)
            for row in conn.execute(
                """
                SELECT
                    auto_handle_path,
                    decision_class,
                    alert_id,
                    active,
                    detected_at
                FROM auto_handle_drift_alerts
                ORDER BY auto_handle_path, decision_class, alert_id
                """
            ).fetchall()
        ]


# ---------------------------------------------------------------------------
# Drift alert — upsert and clear round-trip
# ---------------------------------------------------------------------------


class TestDriftAlertRoundTrip:
    def test_upsert_then_get_active_alert(self, store: AutoHandleCalibrationStore) -> None:
        alert = store._upsert_alert(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
            previous_success_rate=0.98,
            current_success_rate=0.50,
            repo_root=None,
        )
        loaded = store.get_active_alert(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
        )
        assert loaded is not None
        assert loaded.alert_id == alert.alert_id
        assert loaded.previous_success_rate == pytest.approx(0.98)
        assert loaded.current_success_rate == pytest.approx(0.50)
        assert loaded.active is True

    def test_clear_marks_alert_inactive(self, store: AutoHandleCalibrationStore) -> None:
        store._upsert_alert(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
            previous_success_rate=0.95,
            current_success_rate=0.40,
            repo_root=None,
        )
        store._clear_alert(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
        )
        assert (
            store.get_active_alert(
                auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
                decision_class=TEST_CLASS,
            )
            is None
        )

    def test_upsert_replaces_previous_alert(self, store: AutoHandleCalibrationStore) -> None:
        first = store._upsert_alert(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
            previous_success_rate=0.99,
            current_success_rate=0.80,
            repo_root=None,
        )
        second = store._upsert_alert(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
            previous_success_rate=0.80,
            current_success_rate=0.60,
            repo_root=None,
        )
        assert first.alert_id != second.alert_id
        loaded = store.get_active_alert(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
        )
        assert loaded is not None
        assert loaded.alert_id == second.alert_id
        assert store.list_active_alerts() == [loaded]


# ---------------------------------------------------------------------------
# Receipt write
# ---------------------------------------------------------------------------


class TestReceiptWrite:
    def test_record_outcome_writes_receipt_after_db_commit(
        self, store: AutoHandleCalibrationStore, tmp_path: Path
    ) -> None:
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        store.record_outcome(
            decision_id="receipt-seed",
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
            outcome=OUTCOME_SUCCESS,
        )

        result = store.record_outcome(
            decision_id="receipt-trigger",
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
            outcome=OUTCOME_HUMAN_OVERRIDE,
            repo_root=repo_root,
        )

        assert result["alert"] is not None
        receipt_path = Path(result["alert"]["receipt_path"])
        assert receipt_path.exists()
        with closing(sqlite3.connect(store.db_path)) as conn:
            row = conn.execute(
                """
                SELECT receipt_path
                FROM auto_handle_drift_alerts
                WHERE alert_id = ?
                """,
                (result["alert"]["alert_id"],),
            ).fetchone()
        assert row is not None
        assert row[0] == str(receipt_path)

    def test_receipt_json_is_written_under_review_queue_drift(
        self, store: AutoHandleCalibrationStore, tmp_path: Path
    ) -> None:
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        alert = store._upsert_alert(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
            previous_success_rate=0.98,
            current_success_rate=0.50,
            repo_root=repo_root,
        )
        assert alert.receipt_path is not None
        receipt_path = Path(alert.receipt_path)
        assert receipt_path.exists()
        assert receipt_path.parent == repo_root / ".aragora" / "review-queue" / "drift"
        assert receipt_path.name == f"{alert.alert_id}.json"
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        # Schema round-trip — receipt carries the full alert payload.
        assert payload["alert_id"] == alert.alert_id
        assert payload["auto_handle_path"] == AUTO_HANDLE_PATH_FIRE_AND_FORGET
        assert payload["decision_class"] == TEST_CLASS
        assert payload["remediation_action"] == "require_human_review_for_class"

    def test_no_receipt_when_repo_root_missing(self, store: AutoHandleCalibrationStore) -> None:
        alert = store._upsert_alert(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
            previous_success_rate=0.97,
            current_success_rate=0.40,
            repo_root=None,
        )
        assert alert.receipt_path is None


# ---------------------------------------------------------------------------
# evaluate_gate — calibration and drift gating
# ---------------------------------------------------------------------------


def _seed_successes(
    store: AutoHandleCalibrationStore,
    *,
    count: int,
    decision_class: str = TEST_CLASS,
    repo_root: Path | None = None,
) -> None:
    for idx in range(count):
        store.record_outcome(
            decision_id=f"seed-success-{idx}",
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=decision_class,
            outcome=OUTCOME_SUCCESS,
            pr_url=f"https://example.com/pr/{idx}",
            repo_root=repo_root,
        )


class TestEvaluateGate:
    def test_gate_allows_warmup_class_without_failures(
        self, store: AutoHandleCalibrationStore
    ) -> None:
        _seed_successes(store, count=1)

        gate = store.evaluate_gate(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
        )

        assert gate.allowed is True
        assert gate.warmup_active is True
        assert gate.summary.total_samples == 1

    def test_gate_rejects_uncalibrated_class_with_failures(
        self, store: AutoHandleCalibrationStore
    ) -> None:
        store.record_outcome(
            decision_id="warmup-failure",
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
            outcome=OUTCOME_HUMAN_OVERRIDE,
            pr_url="https://example.com/pr/bad",
        )

        gate = store.evaluate_gate(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
        )

        assert gate.allowed is False
        assert "uncalibrated" in gate.reason
        assert gate.summary.failures == 1

    def test_gate_rejects_below_threshold_classes(self, tmp_path: Path) -> None:
        store = AutoHandleCalibrationStore(
            db_path=str(tmp_path / "threshold.db"),
            min_samples=2,
            min_success_rate=0.80,
            drift_threshold=0.05,
        )
        _seed_successes(store, count=1)
        store.record_outcome(
            decision_id="threshold-failure",
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
            outcome=OUTCOME_HUMAN_OVERRIDE,
            pr_url="https://example.com/pr/bad",
        )

        gate = store.evaluate_gate(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
        )

        assert gate.allowed is False
        assert gate.active_drift_alert is True
        assert gate.summary.total_samples == 2
        assert gate.summary.failures == 1

    def test_drift_detector_blocks_until_recovery(self, tmp_path: Path) -> None:
        store = AutoHandleCalibrationStore(
            db_path=str(tmp_path / "drift.db"),
            min_samples=2,
            min_success_rate=0.75,
            drift_threshold=0.10,
        )
        _seed_successes(store, count=2, repo_root=tmp_path)

        result = store.record_outcome(
            decision_id="regressed",
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
            outcome=OUTCOME_HUMAN_OVERRIDE,
            pr_url="https://example.com/pr/regressed",
            repo_root=tmp_path,
        )

        assert isinstance(result["alert"], dict)
        receipt_path = Path(str(result["alert"]["receipt_path"]))
        assert receipt_path.exists()

        blocked = store.evaluate_gate(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
        )
        assert blocked.allowed is False
        assert blocked.active_drift_alert is True

        store.record_outcome(
            decision_id="recovery",
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
            outcome=OUTCOME_SUCCESS,
            pr_url="https://example.com/pr/recovery",
            repo_root=tmp_path,
        )

        recovered = store.evaluate_gate(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=TEST_CLASS,
        )
        assert recovered.allowed is True
        assert recovered.active_drift_alert is False


# ---------------------------------------------------------------------------
# Helper sanity — fingerprints + decision ids are pure
# ---------------------------------------------------------------------------


class TestHelpersArePure:
    def test_fingerprint_low_risk_class_is_stable(self) -> None:
        fp = fingerprint_low_risk_class(
            changed_files=["aragora/triage/x.py", "aragora/triage/y.py"],
            review_tier=1,
            lane_count=2,
        )
        assert fp == "tier=1|lanes=2-3|files=2-3|scope=aragora"

    def test_fingerprint_admin_merge_class_uses_defaults(self) -> None:
        fp = fingerprint_admin_merge_class(
            base_branch=None,
            required_checks_count=4,
            target_kind=None,
        )
        assert fp == "base=unknown|checks=4-6|target=unknown"

    def test_decision_id_is_deterministic(self) -> None:
        a = auto_handle_decision_id(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            pr_url="https://example.com/pr/1",
            decision_class=TEST_CLASS,
        )
        b = auto_handle_decision_id(
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            pr_url="https://example.com/pr/1",
            decision_class=TEST_CLASS,
        )
        assert a == b
        assert a.startswith(f"{AUTO_HANDLE_PATH_FIRE_AND_FORGET}:")
