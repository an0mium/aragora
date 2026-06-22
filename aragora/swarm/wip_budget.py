"""Loop Control Plane v2 -- count-based open-PR WIP ceilings.

Companion to :mod:`aragora.swarm.loop_budget` (dollar ceilings). Where the dollar
budget governs a *running* loop's spend, the WIP cap governs whether a generation
lane should create *another* PR given how many it already has open -- the queue-
pressure control that stops one fleet out-generating the merge gate (the
"backlog never drains because generation == settlement" failure mode).

Two honest pieces, mirroring loop_budget:

* **Policy** (``.aragora/wip_budgets.json``, operator-owned): per-fleet open-PR
  ceilings with an optional fleet default. ``ARAGORA_WIP_OPEN_PR_CAP`` is the
  lowest-precedence fleet default.
* **Live count**: the current open-PR count for a fleet, supplied by the caller
  (e.g. from ``gh pr list``). The control plane only ever *reads* a count it is
  handed; it never fabricates one.

:func:`resolve_wip_budget` composes them into a raw ``wip`` dict; :func:`classify_wip`
turns that into a :class:`WipDecision`:

* ceiling + real count   -> ``within_cap`` | ``over_cap`` (the only gating states)
* ceiling, unknown count -> ``degraded``  (visible, never blocks -- fail safe)
* count, no ceiling      -> ``degraded``  (visible count, nothing to gate on)
* neither                -> ``unavailable``

A fabricated/unknown count NEVER classifies ``over_cap``: blocking a legitimate
fleet on a number we cannot trust is exactly the fabricated-alarm class the
control plane exists to avoid (mirrors loop_budget's "never halt on stale spend").

This module provides the rails, not the retirement: a generation lane adopts
:func:`classify_wip` at its create-PR decision point; nothing here mutates a loop.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

POLICY_RELPATH = Path(".aragora") / "wip_budgets.json"
ENV_FLEET_DEFAULT = "ARAGORA_WIP_OPEN_PR_CAP"

# resolve_wip_budget source_status values (human-readable; classify_wip derives
# its verdict from the data, not from this advisory label -- see classify_wip).
WIP_OK = "ok"

# classify_wip verdicts.
WIP_WITHIN_CAP = "within_cap"
WIP_OVER_CAP = "over_cap"
WIP_DEGRADED = "degraded"
WIP_UNAVAILABLE = "unavailable"


def _as_count(value: Any) -> int | None:
    """Parse a finite, non-negative integer count; everything else is ``None``.

    Rejects bools, NaN/inf, negatives, and non-integral floats (a count of
    ``3.7`` open PRs is meaningless). A rejected value reads as "unknown", which
    fails safe (never ``over_cap``) rather than fabricating a gate.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if not math.isfinite(value) or value < 0 or value != int(value):
            return None
        return int(value)
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


@dataclass
class WipPolicy:
    """Operator-owned per-fleet open-PR ceilings."""

    default_cap: int | None = None
    per_fleet_cap: dict[str, int] = field(default_factory=dict)
    source: str = "none"

    @classmethod
    def load(cls, repo_root: Path | str) -> "WipPolicy":
        """Read the policy file, falling back to the env fleet default.

        Missing or unreadable policy degrades to the env fallback (or an empty
        policy) instead of raising; like loop_budget, the WIP surface must never
        take the fleet inventory down.
        """
        path = Path(repo_root) / POLICY_RELPATH
        payload: dict[str, Any] | None = None
        try:
            with path.open(encoding="utf-8") as handle:
                loaded = json.load(handle)
            payload = loaded if isinstance(loaded, dict) else None
        except (OSError, json.JSONDecodeError, ValueError):
            payload = None

        env_default = _as_count(os.environ.get(ENV_FLEET_DEFAULT))
        if payload is None:
            if env_default is not None:
                return cls(default_cap=env_default, source=f"env:{ENV_FLEET_DEFAULT}")
            return cls()

        per_fleet: dict[str, int] = {}
        raw_fleets = payload.get("fleets")
        if isinstance(raw_fleets, dict):
            for fleet_id, entry in raw_fleets.items():
                cap = _as_count(entry.get("cap")) if isinstance(entry, dict) else None
                if cap is not None:
                    per_fleet[str(fleet_id)] = cap

        default_cap = _as_count(payload.get("default_cap"))
        source = f"policy:{POLICY_RELPATH}"
        if default_cap is None and env_default is not None:
            default_cap = env_default
            source = f"policy:{POLICY_RELPATH}+env:{ENV_FLEET_DEFAULT}"

        return cls(default_cap=default_cap, per_fleet_cap=per_fleet, source=source)

    def cap_for(self, fleet_id: str) -> tuple[int | None, str]:
        """Return ``(cap, source)`` for one fleet (per-fleet beats default)."""
        if fleet_id in self.per_fleet_cap:
            return self.per_fleet_cap[fleet_id], f"{self.source}#fleets.{fleet_id}"
        if self.default_cap is not None:
            return self.default_cap, f"{self.source}#default"
        return None, "none"


@dataclass(frozen=True)
class WipDecision:
    """Pure output of :func:`classify_wip`. ``allow_generation`` is the answer to
    'may this lane create another PR right now?' -- ``False`` only on ``over_cap``."""

    verdict: str
    allow_generation: bool
    open_pr_count: int | None
    ceiling: int | None
    remaining: int | None
    source: str


def resolve_wip_budget(
    open_pr_count: int | None,
    policy: WipPolicy,
    fleet_id: str,
) -> dict[str, Any]:
    """Compose one fleet's raw ``wip`` dict for :func:`classify_wip`.

    Unlike :mod:`loop_budget` -- which reads a persisted spend snapshot and tracks
    its mtime freshness -- the open-PR count here is supplied live by the caller
    (e.g. a just-run ``gh pr list``), so there is no freshness/age field: ensuring
    the count is current is the caller's responsibility at the decision point. A
    caller wiring this into a generation lane must measure the count immediately
    before gating, not reuse a stale cached value.
    """
    cap, cap_source = policy.cap_for(fleet_id)
    count = _as_count(open_pr_count)

    if count is None and cap is None:
        return {
            "source": "none",
            "source_status": WIP_UNAVAILABLE,
            "open_pr_count": None,
            "ceiling": None,
            "remaining": None,
        }
    if cap is None:
        # Visible count, nothing to gate on.
        return {
            "source": "count:supplied",
            "source_status": WIP_DEGRADED,
            "open_pr_count": count,
            "ceiling": None,
            "remaining": None,
        }
    if count is None:
        # A ceiling with an unknown count is visible but never blocks: blocking
        # on a number we cannot trust is the fabricated-alarm class to avoid.
        return {
            "source": cap_source,
            "source_status": WIP_DEGRADED,
            "open_pr_count": None,
            "ceiling": cap,
            "remaining": None,
        }
    return {
        "source": f"{cap_source} + count:supplied",
        "source_status": WIP_OK,
        "open_pr_count": count,
        "ceiling": cap,
        "remaining": cap - count,
    }


def classify_wip(wip: dict[str, Any]) -> WipDecision:
    """Classify a ``wip`` dict into a :class:`WipDecision`.

    The verdict is derived from the *data itself* (parsed count + ceiling), NOT
    from the advisory ``source_status`` label, so a caller cannot bypass the
    fail-safe by mislabelling a dict ``source_status="ok"`` with a missing or
    untrusted count. Only a real ceiling AND a real count can produce ``over_cap``
    and withhold generation; every other shape keeps ``allow_generation`` True
    (``cap=0`` is the deliberate "freeze" case: count >= 0 is always over_cap).
    """
    count = _as_count(wip.get("open_pr_count"))
    ceiling = _as_count(wip.get("ceiling"))
    source = str(wip.get("source", "none"))

    if count is not None and ceiling is not None:
        over = count >= ceiling
        return WipDecision(
            verdict=WIP_OVER_CAP if over else WIP_WITHIN_CAP,
            allow_generation=not over,
            open_pr_count=count,
            ceiling=ceiling,
            remaining=ceiling - count,
            source=source,
        )
    verdict = WIP_UNAVAILABLE if (count is None and ceiling is None) else WIP_DEGRADED
    return WipDecision(
        verdict=verdict,
        allow_generation=True,
        open_pr_count=count,
        ceiling=ceiling,
        remaining=None,
        source=source,
    )
