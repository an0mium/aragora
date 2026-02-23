"""Persistent store for self-improvement runs.

Stores run metadata, progress, and results using JSON file backing.
Each run is a JSON object stored in a JSONL file for append-only durability.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RunStatus(str, Enum):
    """Status of a self-improvement run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SelfImproveRun:
    """A single self-improvement run."""

    run_id: str
    goal: str
    status: RunStatus = RunStatus.PENDING
    tracks: list[str] = field(default_factory=list)
    mode: str = "flat"  # "flat" or "hierarchical"
    budget_limit_usd: float | None = None
    max_cycles: int = 5
    dry_run: bool = False
    created_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    total_subtasks: int = 0
    completed_subtasks: int = 0
    failed_subtasks: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    summary: str = ""
    plan: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SelfImproveRun:
        d = dict(d)
        d["status"] = RunStatus(d.get("status", "pending"))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class SelfImproveRunStore:
    """Persistent store for self-improvement runs.

    Uses JSONL file for durable append-only storage with an in-memory
    index for fast lookups.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        if data_dir is None:
            from aragora.persistence.db_config import get_default_data_dir

            data_dir = get_default_data_dir()
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._runs_file = self._data_dir / "self_improve_runs.jsonl"
        self._runs: dict[str, SelfImproveRun] = {}
        self._load()

    def _load(self) -> None:
        """Load runs from JSONL file."""
        if not self._runs_file.exists():
            return
        try:
            for line in self._runs_file.read_text().strip().splitlines():
                if line.strip():
                    d = json.loads(line)
                    run = SelfImproveRun.from_dict(d)
                    self._runs[run.run_id] = run
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Error loading runs: %s", type(e).__name__)

    def _save_run(self, run: SelfImproveRun) -> None:
        """Append or rewrite a run to the JSONL file."""
        # Rewrite full file (safe for small datasets)
        lines = [json.dumps(r.to_dict()) for r in self._runs.values()]
        self._runs_file.write_text("\n".join(lines) + "\n" if lines else "")

    def create_run(self, goal: str, **kwargs: Any) -> SelfImproveRun:
        """Create a new run and persist it."""
        run = SelfImproveRun(
            run_id=str(uuid.uuid4())[:8],
            goal=goal,
            **kwargs,
        )
        self._runs[run.run_id] = run
        self._save_run(run)
        return run

    def get_run(self, run_id: str) -> SelfImproveRun | None:
        """Get a run by ID."""
        return self._runs.get(run_id)

    def update_run(self, run_id: str, **updates: Any) -> SelfImproveRun | None:
        """Update run fields and persist."""
        run = self._runs.get(run_id)
        if not run:
            return None
        for key, value in updates.items():
            if hasattr(run, key):
                if key == "status" and isinstance(value, str):
                    value = RunStatus(value)
                setattr(run, key, value)
        self._save_run(run)
        return run

    def list_runs(
        self,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
    ) -> list[SelfImproveRun]:
        """List runs with optional filtering."""
        runs = sorted(
            self._runs.values(),
            key=lambda r: r.created_at,
            reverse=True,
        )
        if status:
            runs = [r for r in runs if r.status.value == status]
        return runs[offset : offset + limit]

    def cancel_run(self, run_id: str) -> SelfImproveRun | None:
        """Cancel a running or pending run."""
        run = self._runs.get(run_id)
        if not run:
            return None
        if run.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
            return None  # Already terminal
        run.status = RunStatus.CANCELLED
        run.completed_at = datetime.now(timezone.utc).isoformat()
        self._save_run(run)
        return run
