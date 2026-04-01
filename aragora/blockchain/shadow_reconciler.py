"""Shadow reconciler for comparing attestation hashes against receipt anchors.

Queries PlanStore for pipeline runs that carry attestation payloads, recomputes
the expected receipt hash via ReceiptAnchor logic, and reports any drift between
the stored (shadow) hash and the freshly computed value.

This reconciler is **stateless and query-only** — it never writes to the
blockchain or mutates plan store data.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DriftRecord:
    """One reconciliation result for a single pipeline run.

    Attributes:
        run_id: The pipeline run identifier.
        shadow_hash: The hash stored in the attestation payload.
        computed_hash: The hash recomputed from receipt data.
        match: Whether shadow and computed hashes agree.
        detail: Optional human-readable detail about the drift.
    """

    run_id: str
    shadow_hash: str
    computed_hash: str
    match: bool
    detail: str = ""


@dataclass
class ReconciliationReport:
    """Aggregated result of a reconciliation pass.

    Attributes:
        total: Number of runs examined.
        matched: Number of runs where hashes matched.
        drifted: Number of runs where hashes diverged.
        skipped: Number of runs skipped (missing data).
        records: Per-run drift records.
    """

    total: int = 0
    matched: int = 0
    drifted: int = 0
    skipped: int = 0
    records: list[DriftRecord] = field(default_factory=list)


def _compute_receipt_hash(receipt_data: str) -> str:
    """Compute a SHA-256 hex digest from receipt data, mirroring ReceiptAnchor."""
    return hashlib.sha256(receipt_data.encode()).hexdigest()


class ShadowReconciler:
    """Compares attestation hashes stored in PlanStore against recomputed values.

    The reconciler is stateless: it receives a PlanStore (or any object exposing
    ``list_runs`` and ``get_run``) and performs read-only queries.  No blockchain
    writes are ever issued.

    Args:
        plan_store: A PlanStore instance (or duck-typed equivalent).
    """

    def __init__(self, plan_store: Any) -> None:
        self._store = plan_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reconcile(
        self,
        *,
        limit: int = 100,
        run_ids: list[str] | None = None,
    ) -> ReconciliationReport:
        """Run a reconciliation pass.

        Args:
            limit: Maximum number of runs to examine when *run_ids* is not
                provided.
            run_ids: If given, reconcile only these specific runs.

        Returns:
            A :class:`ReconciliationReport` summarising the results.
        """
        report = ReconciliationReport()

        runs = self._fetch_runs(limit=limit, run_ids=run_ids)

        for run in runs:
            attestation = self._get_attestation(run)
            if not attestation:
                report.skipped += 1
                report.total += 1
                continue

            shadow_hash = attestation.get("receipt_hash", "")
            receipt_data = self._extract_receipt_data(run, attestation)

            if not shadow_hash or not receipt_data:
                report.skipped += 1
                report.total += 1
                continue

            computed_hash = _compute_receipt_hash(receipt_data)
            match = shadow_hash == computed_hash

            record = DriftRecord(
                run_id=self._get_run_id(run),
                shadow_hash=shadow_hash,
                computed_hash=computed_hash,
                match=match,
                detail="" if match else "hash mismatch",
            )
            report.records.append(record)
            report.total += 1

            if match:
                report.matched += 1
            else:
                report.drifted += 1
                logger.warning(
                    "Drift detected for run %s: shadow=%s computed=%s",
                    record.run_id,
                    shadow_hash[:16],
                    computed_hash[:16],
                )

        logger.info(
            "Reconciliation complete: total=%d matched=%d drifted=%d skipped=%d",
            report.total,
            report.matched,
            report.drifted,
            report.skipped,
        )
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_runs(
        self,
        *,
        limit: int,
        run_ids: list[str] | None,
    ) -> list[Any]:
        """Retrieve runs from the plan store."""
        if run_ids:
            runs = []
            for rid in run_ids:
                run = self._store.get_run(rid)
                if run is not None:
                    runs.append(run)
            return runs
        return list(self._store.list_runs(limit=limit))

    @staticmethod
    def _get_attestation(run: Any) -> dict[str, Any]:
        """Extract the attestation dict from a run object."""
        if isinstance(run, dict):
            return dict(run.get("attestation") or {})
        return dict(getattr(run, "attestation", None) or {})

    @staticmethod
    def _get_run_id(run: Any) -> str:
        """Extract the run ID from a run object."""
        if isinstance(run, dict):
            return str(run.get("run_id", ""))
        return str(getattr(run, "run_id", ""))

    @staticmethod
    def _extract_receipt_data(run: Any, attestation: dict[str, Any]) -> str:
        """Build the canonical receipt data string for hashing.

        Tries ``attestation["receipt_data"]`` first, then falls back to
        the run's ``receipt_id``.
        """
        explicit = attestation.get("receipt_data")
        if explicit:
            return str(explicit)

        if isinstance(run, dict):
            receipt_id = run.get("receipt_id", "")
        else:
            receipt_id = getattr(run, "receipt_id", "")
        return str(receipt_id) if receipt_id else ""


__all__ = [
    "DriftRecord",
    "ReconciliationReport",
    "ShadowReconciler",
]
