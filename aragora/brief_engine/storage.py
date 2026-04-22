"""Flat-file storage layer for brief lifecycle artifacts.

On-disk layout (rooted at ``.aragora/{namespace}/briefs/`` — default
namespace is ``review-queue`` for Mode 3 PDB back-compat; future brief
variants pass their own namespace such as ``security-report``)::

    briefs/
    ├── pr-{n}-{sha12}.json           (ready)
    ├── queued/pr-{n}-{sha12}.json    (queued)
    ├── running/pr-{n}-{sha12}.json   (running)
    ├── failed/pr-{n}-{sha12}.json    (failed)
    ├── invalidated/pr-{n}-{sha12}.json (stale)
    └── index.jsonl                   (append-only event log)

The disk layout *is* the state — no separate database. ``get_state``
infers the lifecycle state from file presence alone.

Atomicity
---------

All artifact writes use the ``write-tmp → os.replace`` dance.
``os.replace`` is POSIX-standard ``rename(2)`` semantics, which is
atomic as long as the source and destination are on the same
filesystem. The in-directory move between lifecycle subdirs
(``queued/`` → ``running/``, etc.) is also a single ``os.replace`` and
therefore atomic.

No cross-filesystem operations are performed. The storage root is a
single directory tree; a caller pointing the env var at a path that
spans filesystems breaks the atomicity contract.

Append-only events
------------------

``index.jsonl`` is a line-delimited event log. Each event is a single
JSON object on one line; writes open in append mode and flush before
returning. A crash mid-write can at worst leave a partial trailing line
that downstream parsers should skip — this mirrors the
``shift_ledger.jsonl`` pattern used elsewhere in the repo.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_NAMESPACE",
    "briefs_root",
    "QUEUED_SUBDIR",
    "RUNNING_SUBDIR",
    "FAILED_SUBDIR",
    "INVALIDATED_SUBDIR",
    "INDEX_FILENAME",
    "get_state",
    "load_ready_brief",
    "load_latest_ready_brief",
    "find_ready_briefs_for_pr",
    "queue_generation",
    "mark_running",
    "write_running_phase",
    "mark_ready",
    "mark_failed",
    "invalidate_if_head_changed",
    "cancel_generation",
    "append_index_event",
]

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from aragora.brief_engine.lifecycle import BriefLifecycleState, validate_transition

logger = logging.getLogger(__name__)

UTC = timezone.utc

DEFAULT_NAMESPACE = "review-queue"
"""Default namespace under ``.aragora/`` for brief storage.

Mode 3 PDB pins this to ``review-queue`` for back-compat with the
landed on-disk layout. Future brief variants (SecurityReportBrief,
etc.) pass their own namespace.
"""

QUEUED_SUBDIR = "queued"
RUNNING_SUBDIR = "running"
FAILED_SUBDIR = "failed"
INVALIDATED_SUBDIR = "invalidated"
INDEX_FILENAME = "index.jsonl"

# Short form used in filenames, matching addendum §6 and the existing
# review-queue handler convention.
_SHA_SHORT_LEN = 12


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _discover_review_queue_root(namespace: str = DEFAULT_NAMESPACE) -> Path:
    """Mirror the root-discovery logic of ``review_queue._review_queue_root``.

    Resolution order:

    1. ``ARAGORA_REVIEW_QUEUE_ROOT`` env var if set (back-compat; only
       honored when ``namespace == DEFAULT_NAMESPACE``).
    2. Walk up from cwd looking for a ``.aragora`` or ``.git`` directory;
       return ``{found}/.aragora/{namespace}``
    3. Fallback: ``cwd/.aragora/{namespace}``
    """
    if namespace == DEFAULT_NAMESPACE:
        override = os.environ.get("ARAGORA_REVIEW_QUEUE_ROOT")
        if override:
            return Path(override)
    cwd = Path.cwd()
    for candidate in (cwd, *cwd.parents):
        if (candidate / ".aragora").exists() or (candidate / ".git").exists():
            return candidate / ".aragora" / namespace
    return cwd / ".aragora" / namespace


def briefs_root(namespace: str = DEFAULT_NAMESPACE) -> Path:
    """Return the ``briefs/`` directory (parent of all lifecycle subdirs).

    ``namespace`` controls the enclosing directory — the default
    ``"review-queue"`` matches the landed Mode 3 PDB on-disk layout.
    Future brief variants pass their own namespace (e.g.,
    ``"security-report"``) to keep their artifacts separated.
    """
    return _discover_review_queue_root(namespace) / "briefs"


def _short_sha(head_sha: str) -> str:
    if not isinstance(head_sha, str) or not head_sha:
        raise ValueError("head_sha must be a non-empty string")
    return head_sha[:_SHA_SHORT_LEN]


def _filename(pr_number: int, head_sha: str) -> str:
    if not isinstance(pr_number, int) or pr_number <= 0:
        raise ValueError("pr_number must be a positive integer")
    return f"pr-{pr_number}-{_short_sha(head_sha)}.json"


def _ready_path(pr_number: int, head_sha: str) -> Path:
    return briefs_root() / _filename(pr_number, head_sha)


def _subdir_path(subdir: str, pr_number: int, head_sha: str) -> Path:
    return briefs_root() / subdir / _filename(pr_number, head_sha)


def _index_path() -> Path:
    return briefs_root() / INDEX_FILENAME


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Atomic I/O primitives
# ---------------------------------------------------------------------------


def _atomic_write_json(target: Path, data: Mapping[str, Any]) -> None:
    """Write ``data`` as JSON to ``target`` atomically.

    The write is performed to a sibling ``.tmp`` file which is then
    renamed via :func:`os.replace`. A crash between the write and the
    rename leaves only the ``.tmp`` visible; the ``target`` path either
    holds the previous contents (if any) or does not exist.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    payload = json.dumps(data, indent=2, sort_keys=True, default=str) + "\n"
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, target)


def _atomic_move(src: Path, dst: Path) -> None:
    """Atomically move ``src`` to ``dst``.

    Parent directories are created as needed. ``os.replace`` is atomic
    on a single filesystem (POSIX ``rename(2)`` semantics).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("brief_engine.storage: could not read %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Index (append-only event log)
# ---------------------------------------------------------------------------


def append_index_event(
    pr_number: int,
    head_sha: str,
    event_type: str,
    fields: Mapping[str, Any] | None = None,
) -> None:
    """Append a single lifecycle event to ``index.jsonl``.

    The event shape is a single-line JSON object with keys:

    - ``timestamp`` (UTC ISO-8601)
    - ``pr_number``
    - ``head_sha`` (full SHA as provided)
    - ``event`` (string tag, e.g., ``queued``, ``running``, ``ready``,
      ``failed``, ``stale``, ``cancelled``)
    - plus any caller-supplied ``fields``

    Writes are append-only and open/close per call so concurrent
    lifecycle transitions don't contend on a long-lived handle.
    """
    event = {
        "timestamp": _now_iso(),
        "pr_number": int(pr_number),
        "head_sha": str(head_sha),
        "event": str(event_type),
    }
    if fields:
        for key, value in fields.items():
            if key in event:
                continue
            event[key] = value
    index = _index_path()
    index.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, sort_keys=True, default=str) + "\n"
    # Use 'a' mode — append is atomic for small writes on POSIX.
    with index.open("a", encoding="utf-8") as handle:
        handle.write(line)


# ---------------------------------------------------------------------------
# State inference
# ---------------------------------------------------------------------------


def get_state(pr_number: int, head_sha: str) -> BriefLifecycleState:
    """Return the current lifecycle state for ``(pr_number, head_sha)``.

    Precedence (in order): READY > FAILED > RUNNING > QUEUED > STALE.
    ABSENT is returned when no artifact is found.

    The ordering matters when a crash interleaves two states on disk
    (e.g., a completed ``mark_ready`` before the ``running/`` record
    could be removed). Terminal states (ready, failed) win over
    in-progress states so the caller observes the most-advanced
    outcome.
    """
    filename = _filename(pr_number, head_sha)
    root = briefs_root()
    if (root / filename).exists():
        return BriefLifecycleState.READY
    if (root / FAILED_SUBDIR / filename).exists():
        return BriefLifecycleState.FAILED
    if (root / RUNNING_SUBDIR / filename).exists():
        return BriefLifecycleState.RUNNING
    if (root / QUEUED_SUBDIR / filename).exists():
        return BriefLifecycleState.QUEUED
    if (root / INVALIDATED_SUBDIR / filename).exists():
        return BriefLifecycleState.STALE
    return BriefLifecycleState.ABSENT


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def load_ready_brief(pr_number: int, head_sha: str) -> dict[str, Any] | None:
    """Return the ready brief JSON for ``(pr_number, head_sha)``, or None.

    Only returns the brief when state is ``ready``. Running / queued /
    failed / invalidated records are NOT returned by this function —
    callers that want those must read the subdir artifact explicitly.
    """
    path = _ready_path(pr_number, head_sha)
    if not path.exists():
        return None
    return _safe_read_json(path)


def find_ready_briefs_for_pr(pr_number: int) -> list[Path]:
    """Return all on-disk ``ready`` briefs for a PR across SHAs.

    Sorted newest-first by mtime. Used by :func:`load_latest_ready_brief`
    and :func:`invalidate_if_head_changed`.
    """
    root = briefs_root()
    if not root.exists():
        return []
    try:
        candidates = [p for p in root.glob(f"pr-{pr_number}-*.json") if p.is_file()]
    except OSError:
        return []
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates


def load_latest_ready_brief(pr_number: int) -> dict[str, Any] | None:
    """Return the most-recently-modified ready brief for ``pr_number``.

    This is the compatibility helper used by the legacy
    ``GET /api/v1/review-queue/prs/{n}/brief`` handler, which doesn't
    yet know the head SHA on the request. Prefer
    :func:`load_ready_brief` for SHA-aware reads.
    """
    for path in find_ready_briefs_for_pr(pr_number):
        data = _safe_read_json(path)
        if data is not None:
            return data
    return None


# ---------------------------------------------------------------------------
# Lifecycle writes
# ---------------------------------------------------------------------------


def queue_generation(
    pr_number: int,
    head_sha: str,
    panel_models: list[str] | tuple[str, ...],
    *,
    requested_at: str | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> None:
    """Transition ``absent → queued`` (or ``failed → queued``, ``stale → queued``).

    Writes ``queued/pr-{n}-{sha12}.json`` atomically and appends a
    ``queued`` event to the index. If a prior ``failed`` record exists
    for this ``(pr, sha)``, it is cleared as part of the transition so
    :func:`get_state` consistently reports QUEUED.
    """
    source = get_state(pr_number, head_sha)
    validate_transition(source, BriefLifecycleState.QUEUED)

    record = {
        "pr_number": int(pr_number),
        "head_sha": str(head_sha),
        "state": BriefLifecycleState.QUEUED.value,
        "panel_models": list(panel_models),
        "requested_at": requested_at or _now_iso(),
    }
    if extra_fields:
        for key, value in extra_fields.items():
            record.setdefault(key, value)

    target = _subdir_path(QUEUED_SUBDIR, pr_number, head_sha)
    _atomic_write_json(target, record)

    # Clean up a superseded failed record (failed → queued retry).
    if source == BriefLifecycleState.FAILED:
        failed_path = _subdir_path(FAILED_SUBDIR, pr_number, head_sha)
        if failed_path.exists():
            try:
                failed_path.unlink()
            except OSError:
                logger.warning(
                    "brief_engine.storage: could not clear failed record %s on retry",
                    failed_path,
                )

    append_index_event(
        pr_number,
        head_sha,
        "queued",
        {"panel_models": list(panel_models), "previous_state": source.value},
    )


def mark_running(pr_number: int, head_sha: str, phase: str) -> None:
    """Transition ``queued → running``.

    Atomically moves the queued record to ``running/`` and annotates it
    with ``started_at`` + ``current_phase``.
    """
    source = get_state(pr_number, head_sha)
    validate_transition(source, BriefLifecycleState.RUNNING)

    queued_path = _subdir_path(QUEUED_SUBDIR, pr_number, head_sha)
    running_path = _subdir_path(RUNNING_SUBDIR, pr_number, head_sha)

    # Pull the queued record to preserve panel_models + requested_at.
    prior = _safe_read_json(queued_path) or {}
    record: dict[str, Any] = dict(prior)
    record.update(
        {
            "pr_number": int(pr_number),
            "head_sha": str(head_sha),
            "state": BriefLifecycleState.RUNNING.value,
            "current_phase": str(phase),
            "started_at": _now_iso(),
        }
    )

    # Write new running record atomically, then remove the queued record.
    # Ordering: write-then-remove means an interrupted transition leaves
    # BOTH records visible — get_state resolves to RUNNING (higher
    # precedence) so the caller's view is correct.
    _atomic_write_json(running_path, record)
    try:
        queued_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning(
            "brief_engine.storage: could not remove queued record %s after start",
            queued_path,
        )

    append_index_event(
        pr_number,
        head_sha,
        "running",
        {"current_phase": str(phase)},
    )


def write_running_phase(
    pr_number: int,
    head_sha: str,
    phase: str,
    cost_usd_so_far: float,
) -> None:
    """Update the running record with current phase + cost.

    Does NOT transition state — the state remains RUNNING. Used by the
    executor between phases so UI polls see progress.
    """
    running_path = _subdir_path(RUNNING_SUBDIR, pr_number, head_sha)
    prior = _safe_read_json(running_path) or {}
    record: dict[str, Any] = dict(prior)
    record.update(
        {
            "pr_number": int(pr_number),
            "head_sha": str(head_sha),
            "state": BriefLifecycleState.RUNNING.value,
            "current_phase": str(phase),
            "cost_usd_so_far": float(cost_usd_so_far),
            "updated_at": _now_iso(),
        }
    )
    _atomic_write_json(running_path, record)
    append_index_event(
        pr_number,
        head_sha,
        "running_phase",
        {"current_phase": str(phase), "cost_usd_so_far": float(cost_usd_so_far)},
    )


def mark_ready(
    pr_number: int,
    head_sha: str,
    brief_json: Mapping[str, Any],
    signature: str | None = None,
) -> None:
    """Transition ``running → ready``.

    Writes the final signed brief JSON at the flat filename, removes the
    running record, and appends a ``ready`` index event.

    If ``signature`` is provided it is merged into the stored brief as a
    top-level ``signature`` field (matching addendum §5). If the brief
    already carries a signature and a different value is passed, the
    caller's explicit signature wins.
    """
    source = get_state(pr_number, head_sha)
    validate_transition(source, BriefLifecycleState.READY)

    payload: dict[str, Any] = dict(brief_json)
    payload.setdefault("pr_number", int(pr_number))
    payload.setdefault("head_sha", str(head_sha))
    if signature is not None:
        payload["signature"] = signature

    ready_path = _ready_path(pr_number, head_sha)
    _atomic_write_json(ready_path, payload)

    running_path = _subdir_path(RUNNING_SUBDIR, pr_number, head_sha)
    if running_path.exists():
        try:
            running_path.unlink()
        except OSError:
            logger.warning(
                "brief_engine.storage: could not remove running record %s after ready",
                running_path,
            )

    append_index_event(
        pr_number,
        head_sha,
        "ready",
        {"signature_present": signature is not None},
    )


def mark_failed(
    pr_number: int,
    head_sha: str,
    error_message: str,
    failed_phase: str,
    cost_usd_so_far: float = 0.0,
) -> None:
    """Transition ``queued|running → failed``.

    Writes a failure record to ``failed/`` with the error message,
    phase, and cost-so-far. Removes the queued/running record. Appends
    a ``failed`` index event.
    """
    source = get_state(pr_number, head_sha)
    validate_transition(source, BriefLifecycleState.FAILED)

    record = {
        "pr_number": int(pr_number),
        "head_sha": str(head_sha),
        "state": BriefLifecycleState.FAILED.value,
        "error_message": str(error_message),
        "failed_phase": str(failed_phase),
        "cost_usd_so_far": float(cost_usd_so_far),
        "failed_at": _now_iso(),
    }
    failed_path = _subdir_path(FAILED_SUBDIR, pr_number, head_sha)
    _atomic_write_json(failed_path, record)

    # Remove the pre-failure record of the prior state.
    for subdir in (QUEUED_SUBDIR, RUNNING_SUBDIR):
        prior = _subdir_path(subdir, pr_number, head_sha)
        if prior.exists():
            try:
                prior.unlink()
            except OSError:
                logger.warning(
                    "brief_engine.storage: could not remove %s record %s after failure",
                    subdir,
                    prior,
                )

    append_index_event(
        pr_number,
        head_sha,
        "failed",
        {
            "error_message": str(error_message),
            "failed_phase": str(failed_phase),
            "cost_usd_so_far": float(cost_usd_so_far),
            "previous_state": source.value,
        },
    )


def invalidate_if_head_changed(pr_number: int, current_head_sha: str) -> bool:
    """Move any ready briefs under a different SHA to ``invalidated/``.

    Returns ``True`` iff at least one ready brief was moved. A ``stale``
    index event is appended for each moved brief.

    The current-SHA brief (if any) is left in place — the intent is to
    detect SHA advancement, not to clear the fresh brief.
    """
    current_short = _short_sha(current_head_sha)
    current_filename = _filename(pr_number, current_head_sha)
    moved_any = False
    for path in find_ready_briefs_for_pr(pr_number):
        if path.name == current_filename:
            continue
        prior_payload = _safe_read_json(path)
        # Extract the SHA-short from the filename for the event record.
        # Filename shape: pr-{n}-{sha12}.json
        stem = path.stem  # e.g., "pr-6328-6a7dfc5e5135"
        prior_sha_short = stem.split("-", 2)[-1] if stem.count("-") >= 2 else ""
        prior_head_sha = str((prior_payload or {}).get("head_sha") or "")
        if not prior_head_sha:
            logger.warning(
                "brief_engine.storage: ready brief %s missing full head_sha; "
                "using filename short SHA in stale event",
                path,
            )
            prior_head_sha = prior_sha_short
        destination = path.parent / INVALIDATED_SUBDIR / path.name
        try:
            _atomic_move(path, destination)
        except OSError as exc:
            logger.warning(
                "brief_engine.storage: could not invalidate ready brief %s: %s",
                path,
                exc,
            )
            continue
        moved_any = True
        append_index_event(
            pr_number,
            prior_head_sha,
            "stale",
            {
                "reason": "head_advanced",
                "new_head_sha_short": current_short,
                "invalidated_path": str(destination),
            },
        )
    return moved_any


def cancel_generation(pr_number: int, head_sha: str) -> BriefLifecycleState:
    """Cancel a queued or running generation.

    Returns the post-cancel state. No-op for terminal / absent states;
    the original state is returned in that case.

    Semantics:

    - QUEUED → ABSENT (queued record removed; no artifact left behind)
    - RUNNING → ABSENT (running record removed; caller may subsequently
      call :func:`mark_failed` to preserve partial cost, but cancel
      itself is trace-free beyond the index event)
    - READY / FAILED / ABSENT / STALE → no-op (original state returned)
    """
    source = get_state(pr_number, head_sha)
    if source not in (BriefLifecycleState.QUEUED, BriefLifecycleState.RUNNING):
        return source

    validate_transition(source, BriefLifecycleState.ABSENT)

    subdir = QUEUED_SUBDIR if source == BriefLifecycleState.QUEUED else RUNNING_SUBDIR
    path = _subdir_path(subdir, pr_number, head_sha)
    if path.exists():
        try:
            path.unlink()
        except OSError:
            logger.warning(
                "brief_engine.storage: could not remove %s record %s on cancel",
                subdir,
                path,
            )

    append_index_event(
        pr_number,
        head_sha,
        "cancelled",
        {"previous_state": source.value},
    )
    return BriefLifecycleState.ABSENT
