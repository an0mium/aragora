"""Auto-settlement workers for tiered epistemic outcomes.

Workers consume pending epistemic outcomes and resolve them when deterministic
or oracle evidence becomes available.
"""

from __future__ import annotations

__all__ = [
    "SettlementResolution",
    "WorkerResult",
    "DeterministicSettlementWorker",
    "OracleSettlementWorker",
    "EpistemicSettlementCoordinator",
]

import logging
from dataclasses import dataclass, field
from typing import Callable

from aragora.debate.epistemic_outcomes import EpistemicOutcome, EpistemicOutcomeStore

logger = logging.getLogger(__name__)


@dataclass
class SettlementResolution:
    """Resolution decision returned by an evaluator."""

    resolved_truth: bool
    confidence_delta: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class WorkerResult:
    """Execution summary for a settlement worker run."""

    scanned: int = 0
    resolved: int = 0
    skipped: int = 0
    errors: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "scanned": self.scanned,
            "resolved": self.resolved,
            "skipped": self.skipped,
            "errors": self.errors,
        }


class DeterministicSettlementWorker:
    """Resolve `pending_deterministic` outcomes using deterministic signals."""

    def __init__(
        self,
        store: EpistemicOutcomeStore,
        evaluator: Callable[[EpistemicOutcome], SettlementResolution | None] | None = None,
    ) -> None:
        self._store = store
        self._evaluator = evaluator or self._default_evaluator

    @staticmethod
    def _default_evaluator(outcome: EpistemicOutcome) -> SettlementResolution | None:
        metadata = outcome.metadata if isinstance(outcome.metadata, dict) else {}

        deterministic_truth = metadata.get("deterministic_truth")
        if isinstance(deterministic_truth, bool):
            delta = metadata.get("deterministic_confidence_delta")
            confidence_delta = float(delta) if isinstance(delta, (int, float)) else (
                0.15 if deterministic_truth else -0.15
            )
            return SettlementResolution(
                resolved_truth=deterministic_truth,
                confidence_delta=confidence_delta,
                metadata={"source": "deterministic_truth_flag"},
            )

        ci_status = str(metadata.get("ci_status") or "").strip().lower()
        if ci_status in {"pass", "passed", "success", "green"}:
            return SettlementResolution(
                resolved_truth=True,
                confidence_delta=0.1,
                metadata={"source": "ci_status", "ci_status": ci_status},
            )
        if ci_status in {"fail", "failed", "error", "red"}:
            return SettlementResolution(
                resolved_truth=False,
                confidence_delta=-0.1,
                metadata={"source": "ci_status", "ci_status": ci_status},
            )

        tests_failed = metadata.get("tests_failed")
        tests_passed = metadata.get("tests_passed")
        if isinstance(tests_failed, int) and tests_failed >= 0:
            if tests_failed == 0 and (not isinstance(tests_passed, int) or tests_passed > 0):
                return SettlementResolution(
                    resolved_truth=True,
                    confidence_delta=0.08,
                    metadata={"source": "test_counts"},
                )
            if tests_failed > 0:
                return SettlementResolution(
                    resolved_truth=False,
                    confidence_delta=-0.08,
                    metadata={"source": "test_counts"},
                )
        return None

    def run_once(self, *, limit: int = 100) -> WorkerResult:
        result = WorkerResult()
        pending = self._store.list_outcomes(status="pending_deterministic", limit=limit)
        result.scanned = len(pending)

        for outcome in pending:
            try:
                resolution = self._evaluator(outcome)
                if resolution is None:
                    result.skipped += 1
                    continue

                applied = self._store.resolve_outcome(
                    outcome.debate_id,
                    resolved_truth=resolution.resolved_truth,
                    confidence_delta=resolution.confidence_delta,
                    resolver_type="deterministic_worker",
                    metadata={"settled_by": "deterministic_worker", **resolution.metadata},
                )
                if applied:
                    result.resolved += 1
                else:
                    result.skipped += 1
            except (TypeError, ValueError, OSError) as e:
                result.errors += 1
                logger.debug("deterministic_worker_error debate=%s err=%s", outcome.debate_id, e)
        return result


class OracleSettlementWorker:
    """Resolve `pending_oracle` outcomes using oracle-fed signals."""

    def __init__(
        self,
        store: EpistemicOutcomeStore,
        evaluator: Callable[[EpistemicOutcome], SettlementResolution | None] | None = None,
    ) -> None:
        self._store = store
        self._evaluator = evaluator or self._default_evaluator

    @staticmethod
    def _default_evaluator(outcome: EpistemicOutcome) -> SettlementResolution | None:
        metadata = outcome.metadata if isinstance(outcome.metadata, dict) else {}

        oracle_truth = metadata.get("oracle_truth")
        if isinstance(oracle_truth, bool):
            delta = metadata.get("oracle_confidence_delta")
            confidence_delta = float(delta) if isinstance(delta, (int, float)) else (
                0.12 if oracle_truth else -0.12
            )
            return SettlementResolution(
                resolved_truth=oracle_truth,
                confidence_delta=confidence_delta,
                metadata={"source": "oracle_truth_flag"},
            )

        oracle_signal = str(metadata.get("oracle_signal") or "").strip().lower()
        if oracle_signal in {"pass", "true", "correct", "aligned"}:
            return SettlementResolution(
                resolved_truth=True,
                confidence_delta=0.1,
                metadata={"source": "oracle_signal", "oracle_signal": oracle_signal},
            )
        if oracle_signal in {"fail", "false", "incorrect", "diverged"}:
            return SettlementResolution(
                resolved_truth=False,
                confidence_delta=-0.1,
                metadata={"source": "oracle_signal", "oracle_signal": oracle_signal},
            )
        return None

    def run_once(self, *, limit: int = 100) -> WorkerResult:
        result = WorkerResult()
        pending = self._store.list_outcomes(status="pending_oracle", limit=limit)
        result.scanned = len(pending)

        for outcome in pending:
            try:
                resolution = self._evaluator(outcome)
                if resolution is None:
                    result.skipped += 1
                    continue

                applied = self._store.resolve_outcome(
                    outcome.debate_id,
                    resolved_truth=resolution.resolved_truth,
                    confidence_delta=resolution.confidence_delta,
                    resolver_type="oracle_worker",
                    metadata={"settled_by": "oracle_worker", **resolution.metadata},
                )
                if applied:
                    result.resolved += 1
                else:
                    result.skipped += 1
            except (TypeError, ValueError, OSError) as e:
                result.errors += 1
                logger.debug("oracle_worker_error debate=%s err=%s", outcome.debate_id, e)
        return result


class EpistemicSettlementCoordinator:
    """Run all epistemic settlement workers in one pass."""

    def __init__(self, store: EpistemicOutcomeStore) -> None:
        self._deterministic = DeterministicSettlementWorker(store)
        self._oracle = OracleSettlementWorker(store)

    def run_once(self, *, limit_per_tier: int = 100) -> dict[str, dict[str, int]]:
        deterministic = self._deterministic.run_once(limit=limit_per_tier).to_dict()
        oracle = self._oracle.run_once(limit=limit_per_tier).to_dict()
        totals = {
            "scanned": deterministic["scanned"] + oracle["scanned"],
            "resolved": deterministic["resolved"] + oracle["resolved"],
            "skipped": deterministic["skipped"] + oracle["skipped"],
            "errors": deterministic["errors"] + oracle["errors"],
        }
        return {
            "deterministic": deterministic,
            "oracle": oracle,
            "total": totals,
        }
