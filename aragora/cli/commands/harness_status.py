"""``aragora swarm harness-status`` — render harness health for the operator.

Pure read-only; never mutates the registry. Takes a snapshot of the
process-wide :class:`HarnessHealthRegistry` and prints either a table
(default) or JSON (``--json``).
"""

from __future__ import annotations

import argparse
import json
from typing import Iterable

from aragora.swarm.harness_fallback import (
    default_implementation_ladder,
    default_review_ladder,
)
from aragora.swarm.harness_health import (
    HarnessHealthSnapshot,
    get_harness_health_registry,
)


# Round 30g: only harnesses with a real shipping implementation in
# aragora.harnesses are listed. Re-add 'aider' here when an actual
# AiderHarness lands; do not list a name we can't dispatch to.
_KNOWN_HARNESSES = ("claude-code", "codex")


def _format_outcome(outcome: str | None) -> str:
    if outcome is None:
        return "never invoked"
    return outcome


def _format_pin(snap: HarnessHealthSnapshot) -> str:
    if snap.permanent_pin_reason is None:
        return ""
    return snap.permanent_pin_reason


def _render_table(snaps: Iterable[HarnessHealthSnapshot]) -> str:
    rows = list(snaps)
    if not rows:
        return "(no harnesses recorded)"
    header = (
        f"{'harness':<14} {'available':<10} {'transient':<10} "
        f"{'last_outcome':<14} {'pin_reason':<40}"
    )
    out = [header, "-" * len(header)]
    for snap in rows:
        out.append(
            f"{snap.harness:<14} "
            f"{('yes' if snap.available else 'no'):<10} "
            f"{snap.transient_failure_count_in_window:<10} "
            f"{_format_outcome(snap.last_outcome):<14} "
            f"{_format_pin(snap):<40}"
        )
    return "\n".join(out)


def _render_ladders(*, registry) -> str:
    ladders = [default_implementation_ladder(), default_review_ladder()]
    out = ["", "Fallback ladders:"]
    for ladder in ladders:
        resolution = ladder.next_available(registry=registry)
        chosen = resolution.chosen or "(none available)"
        skipped = f" skipped={list(resolution.skipped)}" if resolution.skipped else ""
        out.append(f"  {ladder.name}: {' -> '.join(ladder.steps)} -> chosen={chosen}{skipped}")
    return "\n".join(out)


def cmd_harness_status(args: argparse.Namespace) -> None:
    """Handle ``aragora swarm harness-status``.

    Args available on ``args``:
      - ``as_json`` (bool): print JSON instead of table.
      - ``status_limit`` (int): unused, kept for arg-parser uniformity.
    """
    registry = get_harness_health_registry()
    snaps = registry.snapshot_all(harnesses=list(_KNOWN_HARNESSES))
    as_json = bool(getattr(args, "as_json", False))
    if as_json:
        ladders_payload = []
        for ladder in (default_implementation_ladder(), default_review_ladder()):
            resolution = ladder.next_available(registry=registry)
            ladders_payload.append(
                {
                    "ladder": ladder.name,
                    "steps": list(ladder.steps),
                    "chosen": resolution.chosen,
                    "skipped": list(resolution.skipped),
                    "reasons": dict(resolution.reasons),
                }
            )
        payload = {
            "harnesses": [snap.to_dict() for snap in snaps],
            "ladders": ladders_payload,
        }
        print(json.dumps(payload, indent=2))
        return
    print(_render_table(snaps))
    print(_render_ladders(registry=registry))
