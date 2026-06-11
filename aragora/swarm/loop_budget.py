"""Loop Control Plane v2 - per-loop dollar budgets.

Two small, honest pieces:

* **Policy** (`.aragora/loop_budgets.json`, operator-owned): per-loop dollar
  ceilings with an optional fleet default. The v1 ``ARAGORA_LOOP_BUDGET_USD``
  environment variable remains the lowest-precedence fleet default.
* **Spend ledger** (`.aragora/loop_spend/<loop_id>.json`): a snapshot each loop
  (or its wrapper) writes via :func:`record_loop_spend`. The control plane only
  ever *reads* it; a loop that does not write spend is reported as such rather
  than fabricated.

:func:`resolve_loop_budget` composes both into the raw ``budget`` dict consumed
by ``aragora.swarm.loop_control.classify_loop``:

* ceiling + fresh spend  -> ``ok`` with a real ``remaining_usd`` (a
  non-positive remainder classifies the loop ``budget_exhausted`` -> ``halt``);
* ceiling, but no/stale spend -> ``degraded`` and **no** ``remaining_usd``
  (never halt a loop on unknown or stale spend);
* spend, but no ceiling  -> ``degraded`` (visible spend, nothing to gate on);
* neither                -> ``unavailable``.

A loop's halt-readiness ``budget_ceiling`` guard stays ``False`` until the loop
itself enforces its ceiling in-loop (or is wrapped by an enforcement point);
this module provides the rails, not the retirement.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

POLICY_RELPATH = Path(".aragora") / "loop_budgets.json"
SPEND_DIR_RELPATH = Path(".aragora") / "loop_spend"

# Daily launchd cadence plus slack (matches the docs-drift freshness window):
# spend older than this cannot prove anything about the current window.
DEFAULT_SPEND_FRESH_SECONDS = 26 * 3600.0

ENV_FLEET_DEFAULT = "ARAGORA_LOOP_BUDGET_USD"


def _as_float(value: Any) -> float | None:
    """Parse a finite float; NaN/inf are rejected.

    A NaN spend or ceiling would fail open: ``NaN <= 0`` is False, so an
    exhausted budget would never classify ``budget_exhausted``.
    """
    if isinstance(value, bool):
        return None
    parsed: float
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
    else:
        return None
    return parsed if math.isfinite(parsed) else None


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class BudgetPolicy:
    """Operator-owned per-loop dollar ceilings."""

    default_ceiling_usd: float | None = None
    per_loop_ceiling_usd: dict[str, float] = field(default_factory=dict)
    spend_fresh_seconds: float = DEFAULT_SPEND_FRESH_SECONDS
    source: str = "none"

    @classmethod
    def load(cls, repo_root: Path | str) -> "BudgetPolicy":
        """Read the policy file, falling back to the v1 fleet-default env var.

        Missing or unreadable policy degrades to the env fallback (or an empty
        policy) instead of raising; the budget surface must never take the
        fleet inventory down.
        """
        path = Path(repo_root) / POLICY_RELPATH
        payload: dict[str, Any] | None = None
        try:
            with path.open(encoding="utf-8") as handle:
                loaded = json.load(handle)
            payload = loaded if isinstance(loaded, dict) else None
        except (OSError, json.JSONDecodeError, ValueError):
            payload = None

        env_default = _as_float(os.environ.get(ENV_FLEET_DEFAULT))
        if payload is None:
            if env_default is not None:
                return cls(default_ceiling_usd=env_default, source=f"env:{ENV_FLEET_DEFAULT}")
            return cls()

        per_loop: dict[str, float] = {}
        raw_loops = payload.get("loops")
        if isinstance(raw_loops, dict):
            for loop_id, entry in raw_loops.items():
                ceiling = _as_float(entry.get("ceiling_usd")) if isinstance(entry, dict) else None
                if ceiling is not None and ceiling >= 0:
                    per_loop[str(loop_id)] = ceiling

        default_ceiling = _as_float(payload.get("default_ceiling_usd"))
        source = f"policy:{POLICY_RELPATH}"
        if default_ceiling is None and env_default is not None:
            default_ceiling = env_default
            source = f"policy:{POLICY_RELPATH}+env:{ENV_FLEET_DEFAULT}"

        fresh = _as_float(payload.get("spend_fresh_seconds"))
        return cls(
            default_ceiling_usd=default_ceiling,
            per_loop_ceiling_usd=per_loop,
            spend_fresh_seconds=(
                fresh if fresh is not None and fresh > 0 else DEFAULT_SPEND_FRESH_SECONDS
            ),
            source=source,
        )

    def ceiling_for(self, loop_id: str) -> tuple[float | None, str]:
        """Return ``(ceiling_usd, source)`` for one loop."""
        if loop_id in self.per_loop_ceiling_usd:
            return self.per_loop_ceiling_usd[loop_id], f"{self.source}#loops.{loop_id}"
        if self.default_ceiling_usd is not None:
            return self.default_ceiling_usd, f"{self.source}#default"
        return None, "none"


_LOOP_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def spend_path(repo_root: Path | str, loop_id: str) -> Path:
    """Resolve one loop's snapshot path; the loop id is a filename component.

    Reject anything that is not a plain snake-case identifier so a hostile or
    buggy ``loop_id`` (absolute path, ``..``, separators) can never escape the
    spend directory.
    """
    if not _LOOP_ID_RE.fullmatch(loop_id):
        raise ValueError(f"invalid loop_id: {loop_id!r}")
    return Path(repo_root) / SPEND_DIR_RELPATH / f"{loop_id}.json"


def record_loop_spend(
    repo_root: Path | str,
    loop_id: str,
    spend_usd: float,
    *,
    source: str,
    window_start: str | None = None,
) -> Path:
    """Atomically write one loop's spend snapshot (the loop-side writer API).

    The control-plane collectors never call this; loops (or their wrappers)
    adopt it to make their spend readable.
    """
    if not math.isfinite(spend_usd) or spend_usd < 0:
        raise ValueError(f"spend_usd must be a finite non-negative number, got {spend_usd}")
    path = spend_path(repo_root, loop_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "loop_id": loop_id,
        "spend_usd": float(spend_usd),
        "window_start": window_start,
        "updated_at": _utc_now_iso(),
        "source": source,
    }
    fd, tmp_name = tempfile.mkstemp(prefix=f".{loop_id}-", suffix=".json", dir=str(path.parent))
    try:
        try:
            handle = os.fdopen(fd, "w", encoding="utf-8")
        except BaseException:
            os.close(fd)
            raise
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        # mkstemp creates 0600; widen to a normal data-file mode so a reader
        # running as a different user than the writing loop is not locked out.
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def read_loop_spend(
    repo_root: Path | str, loop_id: str, *, now: float | None = None
) -> dict[str, Any] | None:
    """Read one loop's spend snapshot; ``None`` when absent or unreadable."""
    path = spend_path(repo_root, loop_id)
    try:
        with path.open(encoding="utf-8") as handle:
            # fstat the open descriptor so the freshness mtime belongs to the
            # same snapshot we parsed, even across a concurrent atomic replace.
            mtime = os.fstat(handle.fileno()).st_mtime
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    spend = _as_float(payload.get("spend_usd"))
    if spend is None or spend < 0:
        return None
    now = now if now is not None else time.time()
    return {
        "spend_usd": spend,
        "window_start": (
            payload.get("window_start") if isinstance(payload.get("window_start"), str) else None
        ),
        "updated_at": (
            payload.get("updated_at") if isinstance(payload.get("updated_at"), str) else None
        ),
        "source": str(payload.get("source") or "ledger"),
        "age_s": max(0.0, now - mtime),
    }


def resolve_loop_budget(
    repo_root: Path | str,
    loop_id: str,
    policy: BudgetPolicy | None = None,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    """Compose one loop's raw ``budget`` dict for ``classify_loop``."""
    policy = policy if policy is not None else BudgetPolicy.load(repo_root)
    ceiling, ceiling_source = policy.ceiling_for(loop_id)
    spend_record = read_loop_spend(repo_root, loop_id, now=now)

    if spend_record is None and ceiling is None:
        return {
            "source": "none",
            "source_status": "unavailable",
            "spend_usd": None,
            "ceiling_usd": None,
            "remaining_usd": None,
        }

    spend: float | None = None
    spend_fresh = False
    spend_source = "ledger:absent"
    if spend_record is not None:
        spend = spend_record["spend_usd"]
        spend_fresh = spend_record["age_s"] <= policy.spend_fresh_seconds
        spend_source = f"ledger:{SPEND_DIR_RELPATH / (loop_id + '.json')}" + (
            "" if spend_fresh else " (stale)"
        )

    if ceiling is None:
        return {
            "source": spend_source,
            "source_status": "degraded",
            "spend_usd": spend,
            "ceiling_usd": None,
            "remaining_usd": None,
        }

    if spend is not None and spend_fresh:
        return {
            "source": f"{ceiling_source} + {spend_source}",
            "source_status": "ok",
            "spend_usd": spend,
            "ceiling_usd": ceiling,
            "remaining_usd": ceiling - spend,
        }

    # A ceiling with unknown or stale spend is visible but never halts: a halt
    # on stale spend would be exactly the fabricated-alarm class the control
    # plane exists to avoid.
    return {
        "source": f"{ceiling_source} + {spend_source}",
        "source_status": "degraded",
        "spend_usd": spend,
        "ceiling_usd": ceiling,
        "remaining_usd": None,
    }
