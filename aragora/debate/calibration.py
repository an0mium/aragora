"""Calibration Tracker -- prediction accuracy over time.

Tracks how well agents' stated confidence matches actual outcomes.
Used by the epistemic settlement loop to measure whether agents are
well-calibrated (i.e., when they say 80% confident, they are correct
~80% of the time).

The tracker supports:

1. **Outcome recording**: Record predicted confidence vs actual outcome
   for each debate/agent.
2. **Calibration curves**: Group predictions into confidence buckets and
   compare predicted vs actual rates.
3. **Brier scores**: Compute per-agent and global Brier scores (lower
   is better, 0 = perfect).

Storage is pluggable: in-memory default, with an interface for persistent
backends.

Usage::

    tracker = CalibrationTracker()

    # After a settlement is resolved
    tracker.record_outcome("debate-1", predicted_confidence=0.8, actual_outcome=True)
    tracker.record_outcome("debate-2", predicted_confidence=0.9, actual_outcome=False,
                           agent_id="claude")

    # Get calibration data
    curve = tracker.get_calibration_curve()
    brier = tracker.get_brier_score()
    agent_brier = tracker.get_brier_score(agent_id="claude")
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

_BUCKET_EDGES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
_BUCKET_LABELS = [f"{lo}-{hi}%" for lo, hi in zip(_BUCKET_EDGES[:-1], _BUCKET_EDGES[1:])]


@dataclass
class CalibrationRecord:
    """A single calibration data point.

    Attributes:
        record_id: Unique identifier.
        debate_id: The debate this prediction came from.
        agent_id: The agent that made the prediction (empty for system-level).
        predicted_confidence: The stated confidence level (0.0-1.0).
        actual_outcome: Whether the prediction was correct.
        domain: Problem domain for bucketing.
        recorded_at: ISO timestamp when the record was created.
        metadata: Additional context.
    """

    record_id: str = field(default_factory=lambda: f"cal-{uuid.uuid4().hex[:12]}")
    debate_id: str = ""
    agent_id: str = ""
    predicted_confidence: float = 0.5
    actual_outcome: bool = False
    domain: str = "general"
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationRecord:
        return cls(
            record_id=data.get("record_id", f"cal-{uuid.uuid4().hex[:12]}"),
            debate_id=data.get("debate_id", ""),
            agent_id=data.get("agent_id", ""),
            predicted_confidence=float(data.get("predicted_confidence", 0.5)),
            actual_outcome=bool(data.get("actual_outcome", False)),
            domain=data.get("domain", "general"),
            recorded_at=data.get("recorded_at", datetime.now(timezone.utc).isoformat()),
            metadata=dict(data.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Calibration Store Interface
# ---------------------------------------------------------------------------


class CalibrationStore:
    """Abstract interface for calibration record persistence.

    Subclass this to provide persistent storage. The default
    implementation is in-memory.
    """

    def save(self, record: CalibrationRecord) -> None:
        raise NotImplementedError

    def list_records(
        self,
        agent_id: str | None = None,
        domain: str | None = None,
        debate_id: str | None = None,
    ) -> list[CalibrationRecord]:
        raise NotImplementedError


class InMemoryCalibrationStore(CalibrationStore):
    """In-memory calibration store for development and testing."""

    def __init__(self) -> None:
        self._records: list[CalibrationRecord] = []

    def save(self, record: CalibrationRecord) -> None:
        self._records.append(record)

    def list_records(
        self,
        agent_id: str | None = None,
        domain: str | None = None,
        debate_id: str | None = None,
    ) -> list[CalibrationRecord]:
        results = self._records
        if agent_id is not None:
            results = [r for r in results if r.agent_id == agent_id]
        if domain is not None:
            results = [r for r in results if r.domain == domain]
        if debate_id is not None:
            results = [r for r in results if r.debate_id == debate_id]
        return list(results)


class JsonFileCalibrationStore(CalibrationStore):
    """JSON file-based calibration store for lightweight persistence.

    Stores all records in ``{data_dir}/calibration/records.json``.

    Args:
        data_dir: Root data directory.
    """

    def __init__(self, data_dir: Path) -> None:
        self._records: list[CalibrationRecord] = []
        cal_dir = data_dir / "calibration"
        cal_dir.mkdir(parents=True, exist_ok=True)
        self._json_path = cal_dir / "records.json"
        self._load()

    def save(self, record: CalibrationRecord) -> None:
        self._records.append(record)
        self._persist()

    def list_records(
        self,
        agent_id: str | None = None,
        domain: str | None = None,
        debate_id: str | None = None,
    ) -> list[CalibrationRecord]:
        results = self._records
        if agent_id is not None:
            results = [r for r in results if r.agent_id == agent_id]
        if domain is not None:
            results = [r for r in results if r.domain == domain]
        if debate_id is not None:
            results = [r for r in results if r.debate_id == debate_id]
        return list(results)

    def _persist(self) -> None:
        try:
            payload = [r.to_dict() for r in self._records]
            tmp_path = self._json_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(payload, indent=2))
            tmp_path.replace(self._json_path)
        except OSError as e:
            logger.warning("Failed to persist calibration records: %s", e)

    def _load(self) -> None:
        if not self._json_path.exists():
            return
        try:
            raw = json.loads(self._json_path.read_text())
            for item in raw:
                self._records.append(CalibrationRecord.from_dict(item))
        except (json.JSONDecodeError, OSError, KeyError, TypeError) as e:
            logger.warning(
                "Failed to load calibration records from %s: %s",
                self._json_path,
                e,
            )


# ---------------------------------------------------------------------------
# Calibration Tracker
# ---------------------------------------------------------------------------


def _bucket_label(confidence: float) -> str:
    """Map a 0-1 confidence to its 10-bucket label (e.g. '70-80%')."""
    pct = confidence * 100
    for lo, hi, label in zip(_BUCKET_EDGES[:-1], _BUCKET_EDGES[1:], _BUCKET_LABELS):
        if pct < hi or hi == 100:
            return label
    return _BUCKET_LABELS[-1]  # pragma: no cover


class CalibrationTracker:
    """Tracks prediction accuracy over time.

    Provides the core calibration feedback mechanism for the epistemic
    settlement loop. Records predicted confidence vs actual outcome and
    computes calibration curves and Brier scores.

    Args:
        store: Optional calibration store for persistence. Defaults to
            an in-memory store.
    """

    def __init__(self, store: CalibrationStore | None = None) -> None:
        self._store = store or InMemoryCalibrationStore()

    def record_outcome(
        self,
        debate_id: str,
        predicted_confidence: float,
        actual_outcome: bool,
        *,
        agent_id: str = "",
        domain: str = "general",
        metadata: dict[str, Any] | None = None,
    ) -> CalibrationRecord:
        """Record a prediction outcome for calibration tracking.

        Args:
            debate_id: The debate that produced the prediction.
            predicted_confidence: Stated confidence (0.0-1.0, clamped).
            actual_outcome: Whether the prediction was correct.
            agent_id: Optional agent identifier.
            domain: Problem domain.
            metadata: Additional context.

        Returns:
            The created CalibrationRecord.
        """
        predicted_confidence = max(0.0, min(1.0, predicted_confidence))

        record = CalibrationRecord(
            debate_id=debate_id,
            agent_id=agent_id,
            predicted_confidence=predicted_confidence,
            actual_outcome=actual_outcome,
            domain=domain,
            metadata=metadata or {},
        )

        self._store.save(record)

        logger.debug(
            "Calibration record: debate=%s agent=%s confidence=%.2f outcome=%s",
            debate_id,
            agent_id or "(system)",
            predicted_confidence,
            actual_outcome,
        )

        return record

    def get_calibration_curve(
        self,
        agent_id: str | None = None,
        domain: str | None = None,
    ) -> dict[str, Any]:
        """Get calibration curve data grouped by confidence buckets.

        For each 10%-wide bucket, computes the mean predicted confidence
        and the actual outcome rate.

        Args:
            agent_id: Filter by agent. None for all agents.
            domain: Filter by domain. None for all domains.

        Returns:
            Dictionary with 'buckets' (list of dicts with predicted,
            actual, count) and 'total_records'.
        """
        records = self._store.list_records(agent_id=agent_id, domain=domain)

        # Accumulate per-bucket: [sum_confidence, sum_outcome, count]
        bucket_acc: dict[str, list[float]] = {lbl: [0.0, 0.0, 0.0] for lbl in _BUCKET_LABELS}

        for r in records:
            label = _bucket_label(r.predicted_confidence)
            acc = bucket_acc[label]
            acc[0] += r.predicted_confidence
            acc[1] += 1.0 if r.actual_outcome else 0.0
            acc[2] += 1.0

        buckets = []
        for label in _BUCKET_LABELS:
            acc = bucket_acc[label]
            count = int(acc[2])
            if count > 0:
                buckets.append(
                    {
                        "bucket": label,
                        "predicted": acc[0] / count,
                        "actual": acc[1] / count,
                        "count": count,
                    }
                )
            else:
                buckets.append(
                    {
                        "bucket": label,
                        "predicted": 0.0,
                        "actual": 0.0,
                        "count": 0,
                    }
                )

        return {
            "buckets": buckets,
            "total_records": len(records),
        }

    def get_brier_score(
        self,
        agent_id: str | None = None,
        domain: str | None = None,
    ) -> float:
        """Compute Brier score measuring calibration quality.

        Brier score = mean of (predicted - outcome)^2 over all records.
        Lower is better: 0.0 = perfect calibration, 1.0 = worst possible.

        Args:
            agent_id: Filter by agent. None for all agents.
            domain: Filter by domain. None for all domains.

        Returns:
            The Brier score (0.0-1.0). Returns 0.0 if no records.
        """
        records = self._store.list_records(agent_id=agent_id, domain=domain)
        if not records:
            return 0.0

        brier_sum = 0.0
        for r in records:
            outcome_val = 1.0 if r.actual_outcome else 0.0
            brier_sum += (r.predicted_confidence - outcome_val) ** 2

        return brier_sum / len(records)

    def get_agent_scores(self) -> dict[str, float]:
        """Get Brier scores for all agents that have records.

        Returns:
            Dictionary mapping agent_id to Brier score.
        """
        all_records = self._store.list_records()
        agents: set[str] = set()
        for r in all_records:
            if r.agent_id:
                agents.add(r.agent_id)

        scores: dict[str, float] = {}
        for agent in agents:
            scores[agent] = self.get_brier_score(agent_id=agent)

        return scores

    def get_summary(
        self,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Get a summary of calibration data.

        Args:
            agent_id: Filter by agent. None for all.

        Returns:
            Dictionary with total records, Brier score, and
            per-agent scores if no agent filter.
        """
        records = self._store.list_records(agent_id=agent_id)
        brier = self.get_brier_score(agent_id=agent_id)

        correct = sum(1 for r in records if r.actual_outcome)
        total = len(records)

        summary: dict[str, Any] = {
            "total_records": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "brier_score": brier,
        }

        if agent_id is None and total > 0:
            summary["per_agent"] = self.get_agent_scores()

        return summary


__all__ = [
    "CalibrationRecord",
    "CalibrationStore",
    "CalibrationTracker",
    "InMemoryCalibrationStore",
    "JsonFileCalibrationStore",
]
