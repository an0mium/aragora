"""Persistent shift ledger for proof-first runtime truth (RS-10).

Accumulates structured entries across shifts so operator status
can be answered from the ledger rather than shell sampling.
Each entry is one JSONL line with a type, timestamp, and payload.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_LEDGER_PATH = ".aragora/proof_first_shift/shift_ledger.jsonl"


@dataclass
class LedgerEntry:
    """One ledger row."""

    entry_type: str
    timestamp: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LedgerEntry:
        return cls(
            entry_type=str(data.get("entry_type", "unknown")),
            timestamp=str(data.get("timestamp", "")),
            payload=dict(data.get("payload") or {}),
        )


class ShiftLedger:
    """Append-only JSONL ledger for shift runtime truth.

    Entry types:
        shift_start     — new shift began
        shift_stop      — shift ended (with reason)
        cycle_tick      — periodic tick with queue/process/benchmark state
        service_restart — boss loop or merge arbiter restarted
        service_failure — restart attempt failed
        benchmark_run   — benchmark publication run observed
        queue_change    — boss-ready queue restocked or pruned
        pr_merged       — automation PR merged
        auth_failure    — Codex/GitHub auth failure detected
        publication_failure — benchmark PR publication failure
    """

    def __init__(self, path: Path | None = None):
        self._path = path or Path(DEFAULT_LEDGER_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def _now_iso(self) -> str:
        return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    def append(self, entry_type: str, **payload: Any) -> LedgerEntry:
        """Append a typed entry to the ledger."""
        entry = LedgerEntry(
            entry_type=entry_type,
            timestamp=self._now_iso(),
            payload=payload,
        )
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")
        except OSError as exc:
            logger.warning("shift_ledger_write_failed: %s", exc)
        return entry

    def read_all(self) -> list[LedgerEntry]:
        """Read all entries from the ledger."""
        if not self._path.exists():
            return []
        entries: list[LedgerEntry] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(LedgerEntry.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError) as exc:
                logger.debug("shift_ledger_parse_skip: %s", exc)
        return entries

    def read_recent(self, *, max_age_hours: float = 24.0) -> list[LedgerEntry]:
        """Read entries from the last N hours."""
        cutoff = time.time() - (max_age_hours * 3600)
        entries = self.read_all()
        recent: list[LedgerEntry] = []
        for entry in entries:
            try:
                ts = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00")).timestamp()
                if ts >= cutoff:
                    recent.append(entry)
            except (ValueError, AttributeError):
                recent.append(entry)  # keep unparseable entries
        return recent

    def read_by_type(self, entry_type: str) -> list[LedgerEntry]:
        """Read all entries of a specific type."""
        return [e for e in self.read_all() if e.entry_type == entry_type]

    # -- Convenience recording methods --

    def record_shift_start(
        self,
        *,
        shift_id: str,
        max_hours: float,
        benchmark_mode: str,
        queue_size: int,
    ) -> LedgerEntry:
        return self.append(
            "shift_start",
            shift_id=shift_id,
            max_hours=max_hours,
            benchmark_mode=benchmark_mode,
            queue_size=queue_size,
        )

    def record_shift_stop(
        self,
        *,
        shift_id: str,
        reason: str,
        cycles: int,
        duration_seconds: float,
    ) -> LedgerEntry:
        return self.append(
            "shift_stop",
            shift_id=shift_id,
            reason=reason,
            cycles=cycles,
            duration_seconds=duration_seconds,
        )

    def record_cycle_tick(
        self,
        *,
        queue_size: int,
        open_prs: int,
        boss_running: bool,
        merge_running: bool,
        benchmark_fresh: bool,
        actions: list[str] | None = None,
    ) -> LedgerEntry:
        return self.append(
            "cycle_tick",
            queue_size=queue_size,
            open_prs=open_prs,
            boss_running=boss_running,
            merge_running=merge_running,
            benchmark_fresh=benchmark_fresh,
            actions=actions or [],
        )

    def record_service_restart(
        self, *, service: str, success: bool, detail: str = ""
    ) -> LedgerEntry:
        return self.append(
            "service_restart",
            service=service,
            success=success,
            detail=detail,
        )

    def record_pr_merged(self, *, pr_number: int, title: str = "") -> LedgerEntry:
        return self.append("pr_merged", pr_number=pr_number, title=title)

    def record_benchmark_run(
        self, *, run_id: int, conclusion: str, created_at: str = ""
    ) -> LedgerEntry:
        return self.append(
            "benchmark_run",
            run_id=run_id,
            conclusion=conclusion,
            created_at=created_at,
        )

    def record_failure(self, *, failure_type: str, detail: str = "") -> LedgerEntry:
        return self.append(failure_type, detail=detail)

    # -- Status summary --

    def get_status_summary(self, *, max_age_hours: float = 24.0) -> dict[str, Any]:
        """Build a status summary from recent ledger entries.

        This is the single truthful view that replaces shell sampling.
        """
        recent = self.read_recent(max_age_hours=max_age_hours)

        shifts = [e for e in recent if e.entry_type == "shift_start"]
        stops = [e for e in recent if e.entry_type == "shift_stop"]
        ticks = [e for e in recent if e.entry_type == "cycle_tick"]
        merges = [e for e in recent if e.entry_type == "pr_merged"]
        benchmarks = [e for e in recent if e.entry_type == "benchmark_run"]
        restarts = [e for e in recent if e.entry_type == "service_restart"]
        auth_failures = [e for e in recent if e.entry_type == "auth_failure"]
        pub_failures = [e for e in recent if e.entry_type == "publication_failure"]

        last_tick = ticks[-1] if ticks else None
        last_stop = stops[-1] if stops else None
        last_benchmark = benchmarks[-1] if benchmarks else None

        return {
            "period_hours": max_age_hours,
            "total_entries": len(recent),
            "shifts_started": len(shifts),
            "shifts_stopped": len(stops),
            "last_stop_reason": last_stop.payload.get("reason", "") if last_stop else "",
            "cycle_ticks": len(ticks),
            "prs_merged": len(merges),
            "pr_numbers_merged": [e.payload.get("pr_number") for e in merges],
            "service_restarts": len(restarts),
            "restart_successes": sum(1 for e in restarts if e.payload.get("success")),
            "restart_failures": sum(1 for e in restarts if not e.payload.get("success")),
            "auth_failures": len(auth_failures),
            "publication_failures": len(pub_failures),
            "benchmark_runs": len(benchmarks),
            "last_benchmark_conclusion": (
                last_benchmark.payload.get("conclusion", "") if last_benchmark else ""
            ),
            "current_queue_size": (last_tick.payload.get("queue_size", 0) if last_tick else None),
            "current_boss_running": (last_tick.payload.get("boss_running") if last_tick else None),
            "current_merge_running": (
                last_tick.payload.get("merge_running") if last_tick else None
            ),
            "current_benchmark_fresh": (
                last_tick.payload.get("benchmark_fresh") if last_tick else None
            ),
        }
