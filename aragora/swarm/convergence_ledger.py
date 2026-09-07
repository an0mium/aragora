"""ConvergenceLedger — a per-PR round budget that survives head drift.

The merge-quorum gate already bounds reruns per *head* (``max_reruns_per_head`` in
:func:`aragora.swarm.merge_quorum_reconcile.plan_rerun`). But every repair commit
creates a *new head*, so that budget resets each round and never bites — which is
why a churning PR can accumulate dozens of "fix" commits with zero merges and zero
posted evidence. The unit of churn is the *repair round* (dissent → repair → new
head → dissent), and it is keyed by the PR, not by any single head.

This ledger bounds rounds **per PR, across head drift**. When the budget is spent,
the loop must stop re-running and instead reach a *decision* (net-value
adjudication: merge-as-is / one bounded round / close / restructure) rather than
churning forever. It also keeps a per-PR head + verdict history so a later
trajectory/churn detector and the net-value adjudicator have the cross-round memory
the stateless gate lacks.

Pure data plus atomic, file-locked persistence. No model calls, no network — the
intelligence layers consume this; this module only remembers.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import cast

logger = logging.getLogger(__name__)

DEFAULT_LEDGER_PATH = Path.home() / ".aragora" / "pr_convergence_ledger.json"
DEFAULT_ROUND_BUDGET = 6  # repair rounds across head drift before adjudication
MAX_TRACKED_PRS = 500  # prune oldest beyond this (by last_round_at)

# POSIX-only file lock; the module imports fine without it (mutations raise a clear
# error rather than corrupting on a platform without flock).
fcntl: ModuleType | None
try:
    import fcntl as _fcntl

    fcntl = _fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PRConvergence:
    """One PR's cross-round convergence record.

    ``rounds`` is the count of *distinct heads* this PR has put through review — the
    churn metric. ``verdicts`` is the per-round verdict history (it may exceed
    ``len(heads)`` if the same head is re-reviewed). ``adjudication`` is set once a
    net-value decision has been recorded for the PR.
    """

    pr_number: int
    heads: list[str] = field(default_factory=list)
    verdicts: list[str] = field(default_factory=list)
    first_seen: str = ""
    last_round_at: str = ""
    adjudication: dict | None = None

    @property
    def rounds(self) -> int:
        return len(self.heads)

    def budget_remaining(self, budget: int) -> int:
        return max(0, budget - self.rounds)

    def is_exhausted(self, budget: int) -> bool:
        return budget > 0 and self.rounds >= budget


class ConvergenceLedger:
    """Atomic, file-locked per-PR convergence store.

    Mutations take an exclusive lock, reload, mutate, persist, release — so
    concurrent agents hitting the same ledger can't clobber each other's rounds.
    Reads are lock-free (the atomic ``os.replace`` on write means a reader never
    sees a torn file).
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_LEDGER_PATH
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    # ---- mutations (locked) --------------------------------------------------

    def record_round(
        self, pr_number: int, head_sha: str, *, verdict: str = "", now: str | None = None
    ) -> int:
        """Record that ``pr_number`` has been through a review round at ``head_sha``.

        A *new* head increments the round count (a repair round); re-reviewing the
        same head does not (it is not new churn), but the verdict is still appended
        to the history. Returns the PR's round count after recording.
        """
        stamp = now or _utcnow_iso()
        with self._locked():
            data = self._load()
            rec = data.get(pr_number) or PRConvergence(pr_number=pr_number, first_seen=stamp)
            if head_sha and head_sha not in rec.heads:
                rec.heads.append(head_sha)
            if verdict:
                rec.verdicts.append(verdict)
            rec.last_round_at = stamp
            data[pr_number] = rec
            self._prune(data)
            self._save(data)
            return rec.rounds

    def record_adjudication(
        self,
        pr_number: int,
        *,
        verdict: str,
        rationale: str = "",
        now: str | None = None,
    ) -> None:
        """Record a net-value adjudication outcome for ``pr_number`` (audit trail)."""
        stamp = now or _utcnow_iso()
        with self._locked():
            data = self._load()
            rec = data.get(pr_number) or PRConvergence(pr_number=pr_number, first_seen=stamp)
            rec.adjudication = {"verdict": verdict, "rationale": rationale, "at": stamp}
            data[pr_number] = rec
            self._save(data)

    # ---- reads (lock-free) ---------------------------------------------------

    def get(self, pr_number: int) -> PRConvergence | None:
        return self._load().get(pr_number)

    def rounds(self, pr_number: int) -> int:
        rec = self.get(pr_number)
        return rec.rounds if rec else 0

    def budget_remaining(self, pr_number: int, budget: int = DEFAULT_ROUND_BUDGET) -> int:
        return max(0, budget - self.rounds(pr_number))

    def is_exhausted(self, pr_number: int, budget: int = DEFAULT_ROUND_BUDGET) -> bool:
        rec = self.get(pr_number)
        return rec.is_exhausted(budget) if rec else False

    def verdict_history(self, pr_number: int) -> list[str]:
        rec = self.get(pr_number)
        return list(rec.verdicts) if rec else []

    def summarize(self, budget: int = DEFAULT_ROUND_BUDGET) -> list[dict]:
        """Compact per-PR rows for a status reader (most-churned first)."""
        rows = [
            {
                "pr": rec.pr_number,
                "rounds": rec.rounds,
                "budget_remaining": rec.budget_remaining(budget),
                "exhausted": rec.is_exhausted(budget),
                "last_verdict": rec.verdicts[-1] if rec.verdicts else "",
                "adjudicated": rec.adjudication is not None,
                "last_round_at": rec.last_round_at,
            }
            for rec in self._load().values()
        ]
        rows.sort(key=lambda r: cast(int, r["rounds"]), reverse=True)
        return rows

    # ---- persistence (atomic) ------------------------------------------------

    def _prune(self, data: dict[int, PRConvergence]) -> None:
        if len(data) <= MAX_TRACKED_PRS:
            return
        # Drop the oldest by last_round_at (un-adjudicated first is not worth the
        # complexity; recency is a fine eviction key for a bounded cache).
        ordered = sorted(data.values(), key=lambda r: r.last_round_at)
        for rec in ordered[: len(data) - MAX_TRACKED_PRS]:
            del data[rec.pr_number]

    @contextlib.contextmanager
    def _locked(self):
        if fcntl is None:  # pragma: no cover - non-POSIX
            raise RuntimeError(
                "ConvergenceLedger mutation requires POSIX fcntl file locking "
                "(not available on this platform)"
            )
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("w") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    def _load(self) -> dict[int, PRConvergence]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("convergence ledger unreadable at %s; starting empty", self.path)
            return {}
        out: dict[int, PRConvergence] = {}
        for key, val in raw.items():
            try:
                pr = int(key)
            except (TypeError, ValueError):
                continue
            out[pr] = PRConvergence(
                pr_number=pr,
                heads=list(val.get("heads", [])),
                verdicts=list(val.get("verdicts", [])),
                first_seen=val.get("first_seen", ""),
                last_round_at=val.get("last_round_at", ""),
                adjudication=val.get("adjudication"),
            )
        return out

    def _save(self, data: dict[int, PRConvergence]) -> None:
        payload = {str(pr): _record_to_dict(rec) for pr, rec in data.items()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=f".{self.path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise


def _record_to_dict(rec: PRConvergence) -> dict:
    d = asdict(rec)
    d.pop("pr_number", None)  # the dict is already keyed by PR number
    return d
