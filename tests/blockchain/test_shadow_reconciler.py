"""Tests for ShadowReconciler — stateless attestation hash reconciliation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import pytest

from aragora.blockchain.shadow_reconciler import (
    DriftRecord,
    ReconciliationReport,
    ShadowReconciler,
    _compute_receipt_hash,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class FakeRun:
    """Minimal stand-in for RunLedger."""

    run_id: str = ""
    receipt_id: str = ""
    attestation: dict[str, Any] = field(default_factory=dict)


class FakePlanStore:
    """In-memory plan store exposing list_runs / get_run."""

    def __init__(self, runs: list[FakeRun] | None = None) -> None:
        self._runs: dict[str, FakeRun] = {}
        for run in runs or []:
            self._runs[run.run_id] = run

    def get_run(self, run_id: str) -> FakeRun | None:
        return self._runs.get(run_id)

    def list_runs(self, *, limit: int = 50, **_kw: Any) -> list[FakeRun]:
        return list(self._runs.values())[:limit]


# ---------------------------------------------------------------------------
# Unit tests for _compute_receipt_hash
# ---------------------------------------------------------------------------


class TestComputeReceiptHash:
    def test_deterministic(self) -> None:
        assert _compute_receipt_hash("hello") == _sha256("hello")

    def test_different_inputs(self) -> None:
        assert _compute_receipt_hash("a") != _compute_receipt_hash("b")


# ---------------------------------------------------------------------------
# DriftRecord / ReconciliationReport dataclass sanity
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_drift_record_defaults(self) -> None:
        rec = DriftRecord(run_id="r1", shadow_hash="a", computed_hash="b", match=False)
        assert rec.detail == ""

    def test_report_defaults(self) -> None:
        rpt = ReconciliationReport()
        assert rpt.total == 0
        assert rpt.matched == 0
        assert rpt.drifted == 0
        assert rpt.skipped == 0
        assert rpt.records == []


# ---------------------------------------------------------------------------
# ShadowReconciler tests
# ---------------------------------------------------------------------------


class TestReconcileMatching:
    """Runs where shadow hash matches the recomputed hash."""

    def test_single_matching_run(self) -> None:
        receipt_data = "receipt-content-1"
        expected_hash = _sha256(receipt_data)
        run = FakeRun(
            run_id="r1",
            attestation={"receipt_hash": expected_hash, "receipt_data": receipt_data},
        )
        store = FakePlanStore([run])
        reconciler = ShadowReconciler(store)
        report = reconciler.reconcile()

        assert report.total == 1
        assert report.matched == 1
        assert report.drifted == 0
        assert report.skipped == 0
        assert len(report.records) == 1
        assert report.records[0].match is True

    def test_multiple_matching_runs(self) -> None:
        runs = []
        for i in range(3):
            data = f"data-{i}"
            runs.append(
                FakeRun(
                    run_id=f"r{i}",
                    attestation={"receipt_hash": _sha256(data), "receipt_data": data},
                )
            )
        report = ShadowReconciler(FakePlanStore(runs)).reconcile()
        assert report.matched == 3
        assert report.drifted == 0


class TestReconcileDrift:
    """Runs where shadow hash does NOT match."""

    def test_single_drifted_run(self) -> None:
        run = FakeRun(
            run_id="r1",
            attestation={"receipt_hash": "wrong-hash", "receipt_data": "actual-data"},
        )
        report = ShadowReconciler(FakePlanStore([run])).reconcile()

        assert report.total == 1
        assert report.drifted == 1
        assert report.matched == 0
        assert report.records[0].match is False
        assert report.records[0].detail == "hash mismatch"

    def test_mixed_match_and_drift(self) -> None:
        good_data = "good"
        runs = [
            FakeRun(
                run_id="match",
                attestation={"receipt_hash": _sha256(good_data), "receipt_data": good_data},
            ),
            FakeRun(
                run_id="drift",
                attestation={"receipt_hash": "bad", "receipt_data": "something"},
            ),
        ]
        report = ShadowReconciler(FakePlanStore(runs)).reconcile()
        assert report.matched == 1
        assert report.drifted == 1
        assert report.total == 2


class TestReconcileSkipped:
    """Runs that should be skipped due to missing data."""

    def test_no_attestation(self) -> None:
        run = FakeRun(run_id="r1", attestation={})
        report = ShadowReconciler(FakePlanStore([run])).reconcile()
        assert report.skipped == 1
        assert report.total == 1
        assert len(report.records) == 0

    def test_missing_receipt_hash(self) -> None:
        run = FakeRun(run_id="r1", attestation={"receipt_data": "data"})
        report = ShadowReconciler(FakePlanStore([run])).reconcile()
        assert report.skipped == 1

    def test_missing_receipt_data_and_receipt_id(self) -> None:
        run = FakeRun(run_id="r1", attestation={"receipt_hash": "abc"})
        report = ShadowReconciler(FakePlanStore([run])).reconcile()
        assert report.skipped == 1


class TestReconcileFallbackToReceiptId:
    """When attestation has no receipt_data, fall back to run.receipt_id."""

    def test_uses_receipt_id(self) -> None:
        receipt_id = "receipt-42"
        run = FakeRun(
            run_id="r1",
            receipt_id=receipt_id,
            attestation={"receipt_hash": _sha256(receipt_id)},
        )
        report = ShadowReconciler(FakePlanStore([run])).reconcile()
        assert report.matched == 1
        assert report.drifted == 0


class TestReconcileByRunIds:
    """Filter reconciliation to specific run IDs."""

    def test_specific_run_ids(self) -> None:
        data = "d"
        runs = [
            FakeRun(run_id="a", attestation={"receipt_hash": _sha256(data), "receipt_data": data}),
            FakeRun(run_id="b", attestation={"receipt_hash": _sha256(data), "receipt_data": data}),
            FakeRun(run_id="c", attestation={"receipt_hash": _sha256(data), "receipt_data": data}),
        ]
        report = ShadowReconciler(FakePlanStore(runs)).reconcile(run_ids=["a", "c"])
        assert report.total == 2
        assert report.matched == 2

    def test_missing_run_id_ignored(self) -> None:
        report = ShadowReconciler(FakePlanStore([])).reconcile(run_ids=["nonexistent"])
        assert report.total == 0


class TestReconcileLimit:
    """Limit parameter caps the number of runs examined."""

    def test_limit(self) -> None:
        data = "x"
        runs = [
            FakeRun(
                run_id=f"r{i}", attestation={"receipt_hash": _sha256(data), "receipt_data": data}
            )
            for i in range(10)
        ]
        report = ShadowReconciler(FakePlanStore(runs)).reconcile(limit=3)
        assert report.total == 3


class TestReconcileEmptyStore:
    """Edge case: empty plan store."""

    def test_empty(self) -> None:
        report = ShadowReconciler(FakePlanStore([])).reconcile()
        assert report.total == 0
        assert report.matched == 0
        assert report.drifted == 0
        assert report.skipped == 0
        assert report.records == []


class TestDictBasedRuns:
    """ShadowReconciler should also work with dict-based run representations."""

    def test_dict_run_matching(self) -> None:
        data = "dict-receipt"

        class DictStore:
            def list_runs(self, *, limit: int = 50, **_kw: Any) -> list[dict[str, Any]]:
                return [
                    {
                        "run_id": "d1",
                        "receipt_id": "",
                        "attestation": {"receipt_hash": _sha256(data), "receipt_data": data},
                    }
                ]

            def get_run(self, run_id: str) -> dict[str, Any] | None:
                return None

        report = ShadowReconciler(DictStore()).reconcile()
        assert report.matched == 1
        assert report.drifted == 0


class TestNoBlockchainWrites:
    """Verify the reconciler is truly query-only."""

    def test_no_write_methods(self) -> None:
        reconciler = ShadowReconciler(FakePlanStore([]))
        # Ensure no public methods suggest mutation
        public = [m for m in dir(reconciler) if not m.startswith("_")]
        write_keywords = {"write", "save", "insert", "update", "delete", "anchor", "submit"}
        for method in public:
            assert not any(kw in method.lower() for kw in write_keywords), (
                f"Unexpected write-like method: {method}"
            )
