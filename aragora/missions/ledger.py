"""Ledger — the stigmergic shared environment ("pheromone") for a mission swarm.

This is the piece that lets *many simple workers* coordinate **without a foreman**
and without a recursive treadmill. It carries two kinds of trail:

1. **Claims / leases** — a worker atomically claims a unit with a TTL. Others see
   the claim and pick something else (response-threshold division of labor →
   non-overlapping fronts). Stale leases evaporate (pheromone decay), so a dead
   worker never permanently blocks a unit.
2. **Constraints / parks** — when a unit hits the same blocker twice, that fact is
   written to the *environment*, not into any agent's prompt (Reflexion-as-data,
   not prompt-rot). Every worker reads it and avoids the excluded behavior. A
   constraint evaporates on TTL or is invalidated when live state materially
   changes — so the swarm escapes treadmills *and* doesn't ossify.

Correctness hinges on **atomic claims**: two workers must never grab the same unit.
Mutations run under an OS file lock (``fcntl.flock``) with read-modify-write, so
the non-overlap property is real across processes, not aspirational.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .state import MissionState

logger = logging.getLogger(__name__)

DEFAULT_LEASE_TTL = 1800.0  # 30 min — a dead worker's claim evaporates after this


@dataclass
class Lease:
    """A worker's atomic claim on one unit of work."""

    unit: str
    worker_id: str
    claimed_at: float
    ttl: float = DEFAULT_LEASE_TTL

    def is_expired(self, now: float) -> bool:
        return now - self.claimed_at >= self.ttl


@dataclass
class Constraint:
    """A learned exclusion — the pheromone that steers the swarm off a treadmill."""

    key: str
    reason: str
    recorded_at: float
    ttl: float = 0.0  # 0 = active until explicitly invalidated by live-state change

    def is_active(self, now: float) -> bool:
        return self.ttl <= 0 or now - self.recorded_at < self.ttl


@dataclass
class _LedgerData:
    leases: dict[str, Lease] = field(default_factory=dict)
    constraints: dict[str, Constraint] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)


class Ledger:
    """Atomic, file-locked shared state for a worker swarm.

    Mutations acquire an exclusive lock, reload from disk, mutate, persist, and
    release — so concurrent ``claim()`` calls from many processes can never
    double-claim a unit.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    # ---- atomic claim (the load-bearing op) ---------------------------------

    def claim(self, unit: str, worker_id: str, ttl: float = DEFAULT_LEASE_TTL, *, now: float | None = None) -> bool:
        """Claim ``unit`` for ``worker_id``. Returns False if actively held by another.

        Atomic across processes. An expired lease is reclaimable; re-claiming your
        own live lease is idempotent True.
        """
        now = time.time() if now is None else now
        with self._locked():
            data = self._load()
            held = data.leases.get(unit)
            if held and not held.is_expired(now) and held.worker_id != worker_id:
                return False
            data.leases[unit] = Lease(unit=unit, worker_id=worker_id, claimed_at=now, ttl=ttl)
            self._save(data)
            return True

    def release(self, unit: str, worker_id: str) -> None:
        with self._locked():
            data = self._load()
            held = data.leases.get(unit)
            if held and held.worker_id == worker_id:
                del data.leases[unit]
                self._save(data)

    def active_claims(self, *, now: float | None = None) -> dict[str, str]:
        """unit -> worker_id for all non-expired leases (lock-free read)."""
        now = time.time() if now is None else now
        data = self._load()
        return {u: lease.worker_id for u, lease in data.leases.items() if not lease.is_expired(now)}

    # ---- constraints / parks (the pheromone) --------------------------------

    def record_constraint(self, key: str, reason: str, ttl: float = 0.0, *, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._locked():
            data = self._load()
            data.constraints[key] = Constraint(key=key, reason=reason, recorded_at=now, ttl=ttl)
            self._save(data)
            logger.info("recorded constraint %s: %s", key, reason)

    def is_excluded(self, key: str, *, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        c = self._load().constraints.get(key)
        return bool(c and c.is_active(now))

    def invalidate_constraint(self, key: str) -> None:
        """Evaporate a constraint because live state materially changed."""
        with self._locked():
            data = self._load()
            if data.constraints.pop(key, None) is not None:
                self._save(data)

    def bump_attempt(self, key: str) -> int:
        """Increment + return the attempt count for ``key`` (drives park-after-N)."""
        with self._locked():
            data = self._load()
            data.attempts[key] = data.attempts.get(key, 0) + 1
            n = data.attempts[key]
            self._save(data)
            return n

    def attempts(self, key: str) -> int:
        return self._load().attempts.get(key, 0)

    def prune(self, *, now: float | None = None) -> tuple[int, int]:
        """Evaporate expired leases + constraints. Returns (leases, constraints) dropped."""
        now = time.time() if now is None else now
        with self._locked():
            data = self._load()
            dead_l = [u for u, ls in data.leases.items() if ls.is_expired(now)]
            dead_c = [k for k, c in data.constraints.items() if not c.is_active(now)]
            for u in dead_l:
                del data.leases[u]
            for k in dead_c:
                del data.constraints[k]
            if dead_l or dead_c:
                self._save(data)
            return len(dead_l), len(dead_c)

    # ---- persistence (atomic) -----------------------------------------------

    @contextlib.contextmanager
    def _locked(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("w") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    def _load(self) -> _LedgerData:
        if not self.path.exists():
            return _LedgerData()
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return _LedgerData(
            leases={u: Lease(**v) for u, v in raw.get("leases", {}).items()},
            constraints={k: Constraint(**v) for k, v in raw.get("constraints", {}).items()},
            attempts=dict(raw.get("attempts", {})),
        )

    def _save(self, data: _LedgerData) -> None:
        payload = {
            "leases": {u: asdict(ls) for u, ls in data.leases.items()},
            "constraints": {k: asdict(c) for k, c in data.constraints.items()},
            "attempts": data.attempts,
        }
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=f".{self.path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise


def select_for(
    state: MissionState,
    ledger: Ledger,
    worker_id: str,
    *,
    ttl: float = DEFAULT_LEASE_TTL,
    park_threshold: int = 2,
    now: float | None = None,
) -> str | None:
    """Stigmergic pickup: claim the first pending feature that is neither claimed
    by another worker nor under an active exclusion. Returns the claimed feature
    id, or None if nothing is available to *this* worker right now.

    This is how non-overlapping fronts emerge with no central dispatcher: each
    worker scans the same queue, atomic-claims the first free unit, and the
    constraint/attempt trail keeps the swarm off treadmills.
    """
    now = time.time() if now is None else now
    claims = ledger.active_claims(now=now)
    completed = {f.id for f in state.features if f.status == "completed"}

    for feat in state.features:
        if feat.status not in ("pending", "in_progress"):
            continue
        unmet = [p for p in feat.preconditions if p.startswith("feature:") and p[8:] not in completed]
        if unmet:
            continue
        if feat.id in claims and claims[feat.id] != worker_id:
            continue  # another worker owns it
        if ledger.is_excluded(f"feature:{feat.id}", now=now):
            continue  # parked (pheromone says: stay off this treadmill)
        if ledger.attempts(f"feature:{feat.id}") >= park_threshold and not feat.notes:
            ledger.record_constraint(f"feature:{feat.id}", "auto-parked after repeated failure", now=now)
            continue
        if ledger.claim(feat.id, worker_id, ttl=ttl, now=now):
            return feat.id
    return None
