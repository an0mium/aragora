"""Threshold recalibration scheduler + receipt schema for #6375 Step B.

This module deliberately does not implement Factory's Step A event-source
adapter. Instead it defines the stable scheduler-facing contract:

1. an event source yields an :class:`InvalidationRecalibrationSample`;
2. the scheduler calls ``compute_baseline()`` and ``derive_threshold()`` from
   :mod:`aragora.review.invalidation`;
3. a deterministic :class:`ThresholdUpdateReceipt` records the inputs,
   derived threshold, and prior-vs-current deltas.

When ``aragora.triage.invalidation_event_source`` lands, it can adapt to the
``InvalidationEventSource`` protocol here without changing the receipt schema.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from aragora.review.invalidation import (
    DEFAULT_BASELINE_WINDOW_DAYS,
    DEFAULT_MIN_BASELINE_SAMPLES,
    DEFAULT_MINIMUM_MEANINGFUL_RATE,
    DEFAULT_SAFETY_MARGIN,
    BaselineMeasurement,
    InvalidatedDecision,
    ThresholdProposal,
    compute_baseline,
    derive_threshold,
)

UTC = timezone.utc

THRESHOLD_UPDATE_RECEIPT_SCHEMA_VERSION = "threshold_update_receipt.v1"
DEFAULT_THRESHOLD_RECEIPT_DIR = Path(".aragora") / "review-queue" / "thresholds"

__all__ = [
    "DEFAULT_THRESHOLD_RECEIPT_DIR",
    "THRESHOLD_UPDATE_RECEIPT_SCHEMA_VERSION",
    "InvalidationEventSource",
    "InvalidationRecalibrationSample",
    "ThresholdRecalibrationScheduler",
    "ThresholdUpdateReceipt",
    "compute_threshold_update_receipt_id",
    "write_threshold_update_receipt",
]


class InvalidationEventSource(Protocol):
    """Protocol implemented by the future Step A event-source adapter."""

    def collect_recalibration_sample(
        self,
        *,
        window_start: datetime,
        window_end: datetime,
    ) -> "InvalidationRecalibrationSample":
        """Return denominators + classified invalidations for one window."""


@dataclass(frozen=True, slots=True)
class InvalidationRecalibrationSample:
    """Scheduler input produced by an invalidation event source.

    ``invalidations`` are the numerator events. ``total_human_settled`` and
    ``total_auto_handled`` are the denominators for the same measurement
    window. This mirrors ``compute_baseline()`` so Step A can remain a pure
    adapter over receipt stores.
    """

    invalidations: tuple[InvalidatedDecision | dict[str, Any], ...] = ()
    total_human_settled: int = 0
    total_auto_handled: int = 0
    per_class_human: Mapping[str, int] = field(default_factory=dict)
    per_class_auto: Mapping[str, int] = field(default_factory=dict)
    source_name: str = "manual"
    source_version: str = ""
    collected_at: datetime | None = None
    notes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total_human_settled < 0 or self.total_auto_handled < 0:
            raise ValueError("decision counts must be non-negative")
        for name, values in (
            ("per_class_human", self.per_class_human),
            ("per_class_auto", self.per_class_auto),
        ):
            for decision_class, count in values.items():
                if count < 0:
                    raise ValueError(
                        f"{name}[{decision_class!r}] must be non-negative; got {count}"
                    )
        object.__setattr__(self, "invalidations", tuple(self.invalidations))
        object.__setattr__(self, "per_class_human", MappingProxyType(dict(self.per_class_human)))
        object.__setattr__(self, "per_class_auto", MappingProxyType(dict(self.per_class_auto)))
        object.__setattr__(self, "notes", MappingProxyType(dict(self.notes)))
        if self.collected_at is not None:
            object.__setattr__(self, "collected_at", _ensure_utc(self.collected_at))

    def to_dict(self) -> dict[str, Any]:
        return {
            "invalidations": [
                item.to_dict() if isinstance(item, InvalidatedDecision) else dict(item)
                for item in self.invalidations
            ],
            "total_human_settled": int(self.total_human_settled),
            "total_auto_handled": int(self.total_auto_handled),
            "per_class_human": dict(self.per_class_human),
            "per_class_auto": dict(self.per_class_auto),
            "source_name": self.source_name,
            "source_version": self.source_version,
            "collected_at": (
                self.collected_at.astimezone(UTC).isoformat()
                if self.collected_at is not None
                else None
            ),
            "notes": dict(self.notes),
        }


@dataclass(frozen=True, slots=True)
class ThresholdUpdateReceipt:
    """Auditable receipt emitted by one threshold recalibration cycle."""

    receipt_id: str
    generated_at: datetime
    source_name: str
    source_version: str
    measurement: BaselineMeasurement
    proposal: ThresholdProposal
    previous_threshold: float | None = None
    previous_baseline_human_rate: float | None = None
    notes: Mapping[str, str] = field(default_factory=dict)
    schema_version: str = THRESHOLD_UPDATE_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "generated_at", _ensure_utc(self.generated_at))
        object.__setattr__(self, "notes", MappingProxyType(dict(self.notes)))
        if self.schema_version != THRESHOLD_UPDATE_RECEIPT_SCHEMA_VERSION:
            raise ValueError(
                "schema_version must be "
                f"{THRESHOLD_UPDATE_RECEIPT_SCHEMA_VERSION!r}; got {self.schema_version!r}"
            )

    @property
    def threshold_delta(self) -> float | None:
        return _delta(self.proposal.threshold, self.previous_threshold)

    @property
    def baseline_human_rate_delta(self) -> float | None:
        return _delta(self.measurement.baseline_human_rate, self.previous_baseline_human_rate)

    @property
    def sample_count(self) -> int:
        return int(self.measurement.total_human_settled)

    def to_dict(self, *, include_receipt_id: bool = True) -> dict[str, Any]:
        measurement = self.measurement.to_dict()
        proposal = self.proposal.to_dict()
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at.astimezone(UTC).isoformat(),
            "source": {
                "name": self.source_name,
                "version": self.source_version,
            },
            "sample_count": int(self.sample_count),
            "window": {
                "start": measurement["window_start"],
                "end": measurement["window_end"],
                "days": measurement["window_days"],
            },
            "baseline": {
                "human_rate": measurement["baseline_human_rate"],
                "human_rate_ci": {
                    "low": measurement["baseline_human_rate_ci_low"],
                    "high": measurement["baseline_human_rate_ci_high"],
                },
                "auto_handle_rate": measurement["auto_handle_rate"],
                "auto_handle_rate_ci": {
                    "low": measurement["auto_handle_rate_ci_low"],
                    "high": measurement["auto_handle_rate_ci_high"],
                },
            },
            "threshold": {
                "derived": proposal["threshold"],
                "is_placeholder": proposal["is_placeholder"],
                "safety_margin": proposal["safety_margin"],
                "minimum_meaningful_rate": proposal["minimum_meaningful_rate"],
                "rationale": proposal["rationale"],
            },
            "prior": {
                "threshold": _float_or_none(self.previous_threshold),
                "baseline_human_rate": _float_or_none(self.previous_baseline_human_rate),
            },
            "delta": {
                "threshold": _float_or_none(self.threshold_delta),
                "baseline_human_rate": _float_or_none(self.baseline_human_rate_delta),
            },
            "measurement": measurement,
            "proposal": proposal,
            "notes": dict(self.notes),
        }
        if include_receipt_id:
            payload["receipt_id"] = self.receipt_id
        return payload


class ThresholdRecalibrationScheduler:
    """Runs one threshold recalibration cycle over a sampled window."""

    def __init__(
        self,
        *,
        window_days: int = DEFAULT_BASELINE_WINDOW_DAYS,
        min_samples: int = DEFAULT_MIN_BASELINE_SAMPLES,
        safety_margin: float = DEFAULT_SAFETY_MARGIN,
        minimum_meaningful_rate: float = DEFAULT_MINIMUM_MEANINGFUL_RATE,
        placeholder_value: float = 0.05,
    ) -> None:
        if window_days <= 0:
            raise ValueError("window_days must be positive")
        if min_samples <= 0:
            raise ValueError("min_samples must be positive")
        self.window_days = int(window_days)
        self.min_samples = int(min_samples)
        self.safety_margin = float(safety_margin)
        self.minimum_meaningful_rate = float(minimum_meaningful_rate)
        self.placeholder_value = float(placeholder_value)

    def run_from_source(
        self,
        source: InvalidationEventSource,
        *,
        previous_receipt: ThresholdUpdateReceipt | None = None,
        previous_threshold: float | None = None,
        previous_baseline_human_rate: float | None = None,
        now: datetime | None = None,
    ) -> ThresholdUpdateReceipt:
        """Collect a sample from ``source`` and emit a receipt.

        This is the wire-up point Factory's Step A adapter should target:
        implement ``collect_recalibration_sample(window_start=..., window_end=...)``.
        """
        now = _ensure_utc(now) if now is not None else datetime.now(UTC)
        window_start = now - timedelta(days=self.window_days)
        sample = source.collect_recalibration_sample(
            window_start=window_start,
            window_end=now,
        )
        return self.run_from_sample(
            sample,
            previous_receipt=previous_receipt,
            previous_threshold=previous_threshold,
            previous_baseline_human_rate=previous_baseline_human_rate,
            now=now,
        )

    def run_from_sample(
        self,
        sample: InvalidationRecalibrationSample,
        *,
        previous_receipt: ThresholdUpdateReceipt | None = None,
        previous_threshold: float | None = None,
        previous_baseline_human_rate: float | None = None,
        now: datetime | None = None,
    ) -> ThresholdUpdateReceipt:
        """Emit a recalibration receipt from a pre-collected sample."""
        now = _ensure_utc(now) if now is not None else datetime.now(UTC)
        previous_threshold, previous_baseline_human_rate = _resolve_previous(
            previous_receipt=previous_receipt,
            previous_threshold=previous_threshold,
            previous_baseline_human_rate=previous_baseline_human_rate,
        )
        measurement = compute_baseline(
            sample.invalidations,
            total_human_settled=sample.total_human_settled,
            total_auto_handled=sample.total_auto_handled,
            window_end=now,
            window_days=self.window_days,
            per_class_human=dict(sample.per_class_human) or None,
            per_class_auto=dict(sample.per_class_auto) or None,
            min_samples=self.min_samples,
        )
        proposal = derive_threshold(
            measurement,
            safety_margin=self.safety_margin,
            minimum_meaningful_rate=self.minimum_meaningful_rate,
            measured_at=now,
            placeholder_value=self.placeholder_value,
        )
        return _build_receipt(
            generated_at=now,
            source_name=sample.source_name,
            source_version=sample.source_version,
            measurement=measurement,
            proposal=proposal,
            previous_threshold=previous_threshold,
            previous_baseline_human_rate=previous_baseline_human_rate,
            notes=sample.notes,
        )


def compute_threshold_update_receipt_id(payload: Mapping[str, Any]) -> str:
    """Return the stable SHA-256 receipt ID for a receipt payload."""
    preimage = dict(payload)
    preimage.pop("receipt_id", None)
    encoded = json.dumps(
        preimage,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_threshold_update_receipt(
    receipt: ThresholdUpdateReceipt,
    *,
    repo_root: Path,
    receipt_dir: Path = DEFAULT_THRESHOLD_RECEIPT_DIR,
) -> Path:
    """Write ``receipt`` to ``repo_root / receipt_dir`` and return the path."""
    directory = repo_root / receipt_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{receipt.receipt_id}.json"
    path.write_text(
        json.dumps(receipt.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _build_receipt(
    *,
    generated_at: datetime,
    source_name: str,
    source_version: str,
    measurement: BaselineMeasurement,
    proposal: ThresholdProposal,
    previous_threshold: float | None,
    previous_baseline_human_rate: float | None,
    notes: Mapping[str, str],
) -> ThresholdUpdateReceipt:
    placeholder = ThresholdUpdateReceipt(
        receipt_id="",
        generated_at=generated_at,
        source_name=source_name,
        source_version=source_version,
        measurement=measurement,
        proposal=proposal,
        previous_threshold=previous_threshold,
        previous_baseline_human_rate=previous_baseline_human_rate,
        notes=notes,
    )
    receipt_id = compute_threshold_update_receipt_id(placeholder.to_dict(include_receipt_id=False))
    return ThresholdUpdateReceipt(
        receipt_id=receipt_id,
        generated_at=generated_at,
        source_name=source_name,
        source_version=source_version,
        measurement=measurement,
        proposal=proposal,
        previous_threshold=previous_threshold,
        previous_baseline_human_rate=previous_baseline_human_rate,
        notes=notes,
    )


def _resolve_previous(
    *,
    previous_receipt: ThresholdUpdateReceipt | None,
    previous_threshold: float | None,
    previous_baseline_human_rate: float | None,
) -> tuple[float | None, float | None]:
    if previous_receipt is None:
        return previous_threshold, previous_baseline_human_rate
    if previous_threshold is not None or previous_baseline_human_rate is not None:
        raise ValueError(
            "pass either previous_receipt or explicit previous_threshold/"
            "previous_baseline_human_rate, not both"
        )
    return previous_receipt.proposal.threshold, previous_receipt.measurement.baseline_human_rate


def _ensure_utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)


def _float_or_none(value: float | None) -> float | None:
    return None if value is None else float(value)


def _delta(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    return float(current - previous)
