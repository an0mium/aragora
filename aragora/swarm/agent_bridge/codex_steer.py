"""Advisory, monotonic-restrictive steer-back channel for a sibling Codex agent.

The companion to ``codex_source`` (observation). Where that module *reads* what
Codex did, this module lets the operator/boss-loop *advise* what Codex should
avoid next -- without ever being able to loosen a gate.

The hard safety invariant, enforced in code rather than by convention: a
steering directive can only ever **add caution**.

* It may append to the *forbidden actions* set (only from a fixed, known-
  restrictive vocabulary -- :data:`STEERABLE_FORBIDDEN_ACTIONS`).
* It may pin specific PRs *off-limits*.
* It may carry an advisory note.

There is deliberately **no field** by which a directive can grant a permission,
remove a forbidden action, authorize a merge / ``--admin`` / Tier-4 settlement,
or relax branch protection. :func:`effective_forbidden_actions` returns a
*superset* of the caller's baseline by construction (set union), so even a
malformed or hostile write to the mailbox can only make Codex more conservative.
The merge-quorum gate remains the sole merge authority.

The mailbox is a local append-only JSONL file (default under the Codex home);
the Codex automation's Phase-0 reads it via :func:`render_phase0_block` and folds
the directives into its own ``forbidden_actions`` before acting.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any

from .codex_source import _iter_jsonl
from .codex_source import _parse_iso
from .codex_source import default_codex_home

# The complete vocabulary a directive may add to a forbidden-actions set. These
# mirror the actions the Codex automation ledgers already gate on; a directive
# referencing anything outside this set is rejected, so the channel cannot be
# used to smuggle in an unknown (and possibly permissive) token.
STEERABLE_FORBIDDEN_ACTIONS: frozenset[str] = frozenset(
    {
        "merge",
        "mark_ready",
        "rerun_required_ci",
        "record_tier4_settlement",
        "mutate_branch_protection",
        "mutate_dirty_root_source_files",
        "create_pr",
        "mutate_pr_branch",
        "touch_shared_root",
    }
)


class SteeringValidationError(ValueError):
    """Raised when a directive would do anything other than add caution."""


def default_mailbox_path(home: Path | None = None) -> Path:
    """Steering mailbox path. Override with ``ARAGORA_CODEX_STEER_MAILBOX``."""
    override = os.environ.get("ARAGORA_CODEX_STEER_MAILBOX")
    if override:
        return Path(override).expanduser()
    home = home or default_codex_home()
    return home / "aragora_steering" / "mailbox.jsonl"


@dataclass(slots=True)
class SteeringDirective:
    """One advisory directive. Every field can only *add* caution.

    ``target_pr`` of ``None`` means the directive applies globally; otherwise it
    applies only when Codex is acting on that PR. ``off_limits_prs`` lists PRs
    Codex must not touch at all.
    """

    issued_by: str
    issued_at: str
    add_forbidden_actions: list[str] = field(default_factory=list)
    off_limits_prs: list[int] = field(default_factory=list)
    target_pr: int | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        unknown = sorted(set(self.add_forbidden_actions) - STEERABLE_FORBIDDEN_ACTIONS)
        if unknown:
            raise SteeringValidationError(
                f"add_forbidden_actions contains non-steerable tokens: {unknown}; "
                f"the channel may only add caution from {sorted(STEERABLE_FORBIDDEN_ACTIONS)}"
            )
        if any(pr <= 0 for pr in self.off_limits_prs):
            raise SteeringValidationError("off_limits_prs must be positive PR numbers")
        if self.target_pr is not None and self.target_pr <= 0:
            raise SteeringValidationError("target_pr must be a positive PR number or null")

    def to_dict(self) -> dict[str, Any]:
        return {
            "issued_by": self.issued_by,
            "issued_at": self.issued_at,
            "add_forbidden_actions": sorted(set(self.add_forbidden_actions)),
            "off_limits_prs": sorted(set(self.off_limits_prs)),
            "target_pr": self.target_pr,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SteeringDirective":
        issued_by = payload.get("issued_by")
        issued_at = payload.get("issued_at")
        if not isinstance(issued_by, str) or not isinstance(issued_at, str):
            raise SteeringValidationError("issued_by and issued_at must be strings")
        raw_actions = payload.get("add_forbidden_actions", [])
        if not isinstance(raw_actions, list):
            raise SteeringValidationError("add_forbidden_actions must be a list")
        raw_prs = payload.get("off_limits_prs", [])
        if not isinstance(raw_prs, list):
            raise SteeringValidationError("off_limits_prs must be a list")
        target_pr = payload.get("target_pr")
        if target_pr is not None and not isinstance(target_pr, int):
            raise SteeringValidationError("target_pr must be an int or null")
        note = payload.get("note")
        if note is not None and not isinstance(note, str):
            raise SteeringValidationError("note must be a string or null")
        return cls(
            issued_by=issued_by,
            issued_at=issued_at,
            add_forbidden_actions=[a for a in raw_actions if isinstance(a, str)],
            off_limits_prs=[p for p in raw_prs if isinstance(p, int) and not isinstance(p, bool)],
            target_pr=target_pr,
            note=note,
        )


def write_directive(directive: SteeringDirective, *, mailbox_path: Path | None = None) -> Path:
    """Append a directive to the mailbox (creating it if needed). Returns the path.

    A directive serializes to well under ``PIPE_BUF`` (4 KB on macOS/Linux), so a
    single ``O_APPEND`` write is atomic and concurrent writers cannot interleave;
    no explicit lock is needed at these sizes.
    """
    path = mailbox_path or default_mailbox_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(directive.to_dict(), sort_keys=True) + "\n")
    return path


def read_directives(
    *,
    mailbox_path: Path | None = None,
    hours: float | None = 24.0,
    now: datetime | None = None,
) -> list[SteeringDirective]:
    """Read directives from the mailbox, oldest first; malformed lines are skipped.

    Sorted by ``issued_at`` so "oldest first" holds even if a shared mailbox is
    appended to by multiple writers with skewed clocks (unparseable timestamps
    sort first, as the most conservative position).
    """
    path = mailbox_path or default_mailbox_path()
    if not path.is_file():
        return []
    now = now or datetime.now(UTC)
    cutoff = None if hours is None else now - timedelta(hours=hours)
    _epoch = datetime.min.replace(tzinfo=UTC)
    dated: list[tuple[datetime, SteeringDirective]] = []
    for record in _iter_jsonl(path):
        issued = _parse_iso(record.get("issued_at"))
        if cutoff is not None and (issued is None or issued < cutoff):
            continue
        try:
            dated.append((issued or _epoch, SteeringDirective.from_dict(record)))
        except SteeringValidationError:
            # A malformed/hostile directive is dropped entirely. Because the only
            # effect a valid directive can have is additive caution, dropping one
            # can never *loosen* the effective posture.
            continue
    dated.sort(key=lambda item: item[0])
    return [directive for _, directive in dated]


def effective_forbidden_actions(
    base_forbidden_actions: list[str],
    directives: list[SteeringDirective],
    *,
    pr: int | None = None,
) -> list[str]:
    """The caller's baseline forbidden set, UNIONed with applicable directives.

    Guaranteed superset of ``base_forbidden_actions`` -- this function can only
    grow the set, never shrink it. A directive applies when it is global
    (``target_pr is None``) or its ``target_pr`` matches ``pr``. A PR on any
    directive's ``off_limits_prs`` gets the whole steerable set added.
    """
    effective: set[str] = set(base_forbidden_actions)
    for directive in directives:
        if directive.target_pr is None or directive.target_pr == pr:
            effective.update(directive.add_forbidden_actions)
        if pr is not None and pr in directive.off_limits_prs:
            effective.update(STEERABLE_FORBIDDEN_ACTIONS)
    return sorted(effective)


def off_limits_prs(directives: list[SteeringDirective]) -> list[int]:
    """All PRs pinned off-limits across the given directives."""
    pinned: set[int] = set()
    for directive in directives:
        pinned.update(directive.off_limits_prs)
    return sorted(pinned)


def render_phase0_block(directives: list[SteeringDirective], *, pr: int | None = None) -> str:
    """Human-readable steering block for a Codex automation prompt's Phase-0.

    Returns ``""`` when there are no applicable directives so the caller can
    omit the section entirely.
    """
    if not directives:
        return ""
    pinned = off_limits_prs(directives)
    added = effective_forbidden_actions([], directives, pr=pr)
    notes = [d.note for d in directives if d.note]
    lines = ["## Operator steering (advisory; additive caution only)"]
    if pinned:
        lines.append(f"- PRs OFF-LIMITS (do not touch): {pinned}")
    if added:
        lines.append(f"- Additional forbidden actions this cycle: {added}")
    for note in notes:
        lines.append(f"- Note: {note}")
    lines.append(
        "- These directives can only ADD caution; the merge-quorum gate remains the "
        "sole merge authority. They never authorize merge, --admin, or Tier-4 settlement."
    )
    return "\n".join(lines)
