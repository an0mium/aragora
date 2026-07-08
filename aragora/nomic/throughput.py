"""
Append-only throughput ledger for the self-improvement loop.

One JSONL file (``.aragora/throughput/ledger.jsonl``) records what the loop
actually retires — merges (classified by :mod:`aragora.nomic.work_mix`),
external artifacts, parks, reverts, and daily SENSE snapshots — so the
work-mix budget, the kill switches, and the weekly founder digest all read
from a single source of truth.

Also owns the substrate-freeze marker (``substrate_freeze.marker``): written
when the mix budget is breached, honored by goal selection while it exists.
Advisory in Phase 1 — this module computes and records; nothing here mutates
GitHub or gate behavior.

Plan: docs/plans/2026-07-08-vision-audit-and-work-mix-governor.md (section 4.4).
Part of epic #9039 (issue #9040).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from aragora.nomic.work_mix import (
    ClassifiedWork,
    MixBudget,
    MixReport,
    WorkClass,
    compute_mix,
    evaluate_budget,
)

THROUGHPUT_DIR = Path(".aragora") / "throughput"
LEDGER_FILENAME = "ledger.jsonl"
FREEZE_FILENAME = "substrate_freeze.marker"

_RECORD_KINDS = frozenset({"merge", "snapshot", "artifact", "park", "revert", "note"})


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class LedgerRecord:
    """One append-only ledger entry."""

    kind: str
    timestamp: str
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in _RECORD_KINDS:
            raise ValueError(f"unknown ledger record kind: {self.kind!r}")

    def to_json(self) -> str:
        return json.dumps(
            {"kind": self.kind, "timestamp": self.timestamp, "data": self.data},
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, line: str) -> LedgerRecord:
        raw = json.loads(line)
        return cls(kind=raw["kind"], timestamp=raw["timestamp"], data=raw.get("data", {}))

    @property
    def when(self) -> datetime:
        return datetime.fromisoformat(self.timestamp)


class ThroughputLedger:
    """Append-only JSONL ledger rooted at a repo checkout."""

    def __init__(self, root: Path | str = ".") -> None:
        self.root = Path(root)
        self.path = self.root / THROUGHPUT_DIR / LEDGER_FILENAME

    def append(self, record: LedgerRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.to_json() + "\n")

    def records(self, since: datetime | None = None) -> list[LedgerRecord]:
        """Read records, skipping corrupt lines (e.g. a write truncated by a
        killed process) so one bad line never bricks the ledger."""
        if not self.path.exists():
            return []
        result: list[LedgerRecord] = []
        skipped = 0
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = LedgerRecord.from_json(line)
                    record.when  # noqa: B018 - validate timestamp eagerly
                except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                    skipped += 1
                    logger.warning("skipping corrupt ledger line %d in %s", line_number, self.path)
                    continue
                if since is None or record.when >= since:
                    result.append(record)
        if skipped:
            logger.warning("ledger %s had %d corrupt line(s)", self.path, skipped)
        return result

    # -- convenience writers -------------------------------------------------

    def record_merge(
        self,
        classified: ClassifiedWork,
        *,
        title: str = "",
        when: datetime | None = None,
    ) -> LedgerRecord:
        record = LedgerRecord(
            kind="merge",
            timestamp=(when or _utcnow()).isoformat(),
            data={
                "identifier": classified.identifier,
                "work_class": classified.work_class.value,
                "exempt": classified.exempt,
                "rationale": classified.rationale,
                "cross_check_disagreement": classified.cross_check_disagreement,
                "title": title,
                "file_counts": {
                    cls.value: count for cls, count in classified.file_counts.items() if count
                },
            },
        )
        self.append(record)
        return record

    def record_artifact(
        self, name: str, *, url: str = "", when: datetime | None = None
    ) -> LedgerRecord:
        record = LedgerRecord(
            kind="artifact",
            timestamp=(when or _utcnow()).isoformat(),
            data={"name": name, "url": url},
        )
        self.append(record)
        return record

    def record_snapshot(
        self, metrics: dict[str, Any], *, when: datetime | None = None
    ) -> LedgerRecord:
        record = LedgerRecord(
            kind="snapshot", timestamp=(when or _utcnow()).isoformat(), data=metrics
        )
        self.append(record)
        return record


@dataclass
class ThroughputMetrics:
    """Rolling loop-health metrics computed from ledger records."""

    window_days: int
    merges_total: int
    merges_by_class: dict[WorkClass, int]
    product_share: float
    substrate_share: float
    self_repair_ratio: float
    external_artifacts: int
    parks: int
    reverts: int
    cross_check_disagreements: int
    # Exempt merges (security/main-red/revert) are excluded from the mix but
    # remain visible here and in the digest — exemption is a gaming surface.
    exempt_merges: int = 0


def _merge_to_classified(record: LedgerRecord) -> ClassifiedWork:
    data = record.data
    counts = {cls: 0 for cls in WorkClass}
    for value, count in data.get("file_counts", {}).items():
        counts[WorkClass(value)] = count
    return ClassifiedWork(
        identifier=str(data.get("identifier", "")),
        work_class=WorkClass(data["work_class"]),
        file_counts=counts,
        rationale=str(data.get("rationale", "")),
        exempt=bool(data.get("exempt", False)),
        cross_check_disagreement=data.get("cross_check_disagreement"),
    )


def compute_metrics(
    records: list[LedgerRecord],
    *,
    now: datetime | None = None,
    window_days: int = 7,
) -> ThroughputMetrics:
    """Compute rolling metrics over the window ``[now - window_days, now)``.

    The window is bounded on BOTH ends so a caller computing a *previous*
    period (``now=real_now - 7d``) does not silently absorb current-period
    records — week-over-week deltas depend on this.
    """
    now = now or _utcnow()
    cutoff = now - timedelta(days=window_days)
    windowed = [r for r in records if cutoff <= r.when < now]

    merges = [r for r in windowed if r.kind == "merge"]
    classified = [_merge_to_classified(r) for r in merges]
    mix = compute_mix(classified, window_days=window_days)

    # Self-repair: fixes whose target is the loop itself (substrate-class
    # merges with a fix-shaped title). High ratios trigger the
    # remove-the-fragile-abstraction review, not more patches.
    self_repair = sum(
        1
        for record, work in zip(merges, classified)
        if work.work_class is WorkClass.SUBSTRATE
        and str(record.data.get("title", "")).lower().startswith(("fix", "repair"))
    )

    return ThroughputMetrics(
        window_days=window_days,
        merges_total=len(merges),
        merges_by_class=dict(mix.counts),
        product_share=mix.product_share,
        substrate_share=mix.substrate_share,
        self_repair_ratio=self_repair / len(merges) if merges else 0.0,
        external_artifacts=sum(1 for r in windowed if r.kind == "artifact"),
        parks=sum(1 for r in windowed if r.kind == "park"),
        reverts=sum(1 for r in windowed if r.kind == "revert"),
        cross_check_disagreements=sum(1 for work in classified if work.cross_check_disagreement),
        exempt_merges=sum(1 for work in classified if work.exempt),
    )


def mix_from_records(
    records: list[LedgerRecord],
    *,
    now: datetime | None = None,
    window_days: int = 7,
) -> MixReport:
    """The windowed work-mix report used for budget evaluation."""
    now = now or _utcnow()
    cutoff = now - timedelta(days=window_days)
    classified = [
        _merge_to_classified(r) for r in records if r.kind == "merge" and cutoff <= r.when < now
    ]
    return compute_mix(classified, window_days=window_days)


# -- substrate-freeze marker -------------------------------------------------


def freeze_marker_path(root: Path | str = ".") -> Path:
    return Path(root) / THROUGHPUT_DIR / FREEZE_FILENAME


def write_freeze_marker(root: Path | str = ".", *, reason: str) -> Path:
    path = freeze_marker_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"timestamp": _utcnow().isoformat(), "reason": reason}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def clear_freeze_marker(root: Path | str = ".") -> bool:
    path = freeze_marker_path(root)
    if path.exists():
        path.unlink()
        return True
    return False


def freeze_active(root: Path | str = ".") -> dict[str, Any] | None:
    """Return the marker payload if the freeze is active, else None."""
    path = freeze_marker_path(root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    return payload if isinstance(payload, dict) else {}


# -- digest ------------------------------------------------------------------


def render_digest(
    metrics: ThroughputMetrics,
    *,
    previous: ThroughputMetrics | None = None,
    budget: MixBudget | None = None,
    freeze: dict[str, Any] | None = None,
    samples: list[dict[str, Any]] | None = None,
) -> str:
    """Render the weekly founder digest as markdown."""
    budget = budget or MixBudget()
    mix = MixReport(
        counts=metrics.merges_by_class,
        total=metrics.merges_total,
        window_days=metrics.window_days,
    )
    verdict = evaluate_budget(mix, budget)

    def delta(current: float, prev: float | None) -> str:
        if prev is None:
            return ""
        change = current - prev
        return f" ({'+' if change >= 0 else ''}{change:.0%} WoW)"

    lines = [
        "# Weekly throughput digest",
        "",
        f"Window: trailing {metrics.window_days} days",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Merges | {metrics.merges_total} |",
        "| Product share | "
        f"{metrics.product_share:.0%}"
        f"{delta(metrics.product_share, previous.product_share if previous else None)} |",
        "| Substrate share | "
        f"{metrics.substrate_share:.0%} (budget ≤ {budget.substrate_max:.0%}) |",
        f"| Self-repair ratio | {metrics.self_repair_ratio:.0%} |",
        f"| External artifacts | {metrics.external_artifacts} |",
        f"| Parks / reverts | {metrics.parks} / {metrics.reverts} |",
        f"| Classifier disagreements | {metrics.cross_check_disagreements} |",
        f"| Exempt merges (excluded from mix) | {metrics.exempt_merges} |",
        "",
        f"**Budget verdict:** {'OK' if verdict.ok else '; '.join(verdict.reasons)}",
        "",
        "**Substrate freeze:** "
        + (
            f"ACTIVE since {freeze.get('timestamp', '?')} — {freeze.get('reason', '')}"
            if freeze
            else "inactive"
        ),
    ]
    if samples:
        lines += ["", "## Spot-audit sample", ""]
        for sample in samples:
            exempt_tag = " [EXEMPT — not counted in mix]" if sample.get("exempt") else ""
            lines.append(
                f"- {sample.get('identifier', '?')}: {sample.get('work_class', '?')}"
                f"{exempt_tag} — {sample.get('title', '')}"
            )
    return "\n".join(lines) + "\n"
