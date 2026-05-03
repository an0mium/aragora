"""``aragora review-queue observe-outcomes`` — bounded receipt observation.

Round 30g phase A (continues #6921 / δ #6375).

Iterates settled receipts in a bounded window, fetches GitHub timeline
events for each PR with bounded fanout, calls
:func:`aragora.review.settlement_outcome.observe_outcome`, and either
*previews* the proposed v2 outcome fields (the default; ``--dry-run``
implied) or *writes* them back into the receipt JSON in place
(``--write``).

Safety contract
---------------

  - **Default mode is read-only.** No filesystem mutation, no GitHub
    write, nothing committed. ``--write`` is the only flag that
    permits in-place mutation.
  - **Bounded fan-out.** ``--max-receipts`` (default 20) caps the
    number of receipts examined; ``--per-receipt-event-cap`` (default
    100) caps the number of timeline events fetched per PR. Both are
    enforced before the GitHub fetch.
  - **No network on dry-run when fixtures are provided.** Tests pass
    ``timeline_provider`` to inject synthetic events so the CLI is
    fully testable offline.
  - **No commits.** This module never invokes ``git`` or ``gh pr``
    write operations. Receipt files are JSON on disk and may be
    rewritten by the operator's local tooling, but the *only* mutation
    this CLI performs under ``--write`` is rewriting the receipt JSON
    payload with the new ``outcome_*`` fields.
  - **Sharper insufficiency receipt.** When the post-observation
    baseline still lacks data, a structured insufficiency receipt is
    proposed in dry-run mode and written under
    ``.aragora/evolve-round/<round-id>/`` only when ``--write`` is
    passed. It describes *what* would unblock the next measurement
    (sample-size shortfall, no v2-eligible receipts, GitHub-fetch
    errors, etc.).

Out of scope
------------

  - This CLI does **not** edit ``docs/THESIS.md``. Replacing the
    placeholder 5% in Commitment 3 is gated on a measured baseline
    *and* CI safety margin, neither of which this CLI produces on its
    own.
  - This CLI does **not** invent synthetic outcome data. If
    ``observe_outcome`` returns "no signals fired" for every receipt
    in the window, that is recorded honestly as
    ``schema_v2_observed_no_signals``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from aragora.review.invalidation_event_source import (
    RECEIPTS_SUBDIR,
    _V2_OUTCOME_FIELD_NAMES,
    _any_receipt_has_v2_outcome_fields,
    resolve_review_queue_root,
)
from aragora.review.settlement_outcome import observe_outcome

logger = logging.getLogger(__name__)

UTC = timezone.utc

DEFAULT_WINDOW_DAYS = 14
DEFAULT_MAX_RECEIPTS = 20
DEFAULT_PER_RECEIPT_EVENT_CAP = 100


@dataclass(frozen=True, slots=True)
class _ReceiptOnDisk:
    path: Path
    payload: dict[str, Any]


@dataclass(slots=True)
class _ObserveResult:
    receipt_path: Path
    pr_number: int
    head_sha: str
    fetched_event_count: int
    fetch_error: str | None
    signals_before: dict[str, Any]
    signals_after: dict[str, Any]
    written: bool
    skipped_reason: str | None = None


# ---------------------------------------------------------------------------
# Receipt iteration (bounded, read-only)
# ---------------------------------------------------------------------------


def _parse_iso(raw: str) -> datetime:
    cleaned = raw.replace("Z", "+00:00") if raw.endswith("Z") else raw
    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _iter_eligible_receipts(
    *,
    store_root: str | Path | None,
    window_end: datetime,
    window_days: int,
    max_receipts: int,
) -> list[_ReceiptOnDisk]:
    receipts_dir = resolve_review_queue_root(store_root) / RECEIPTS_SUBDIR
    if not receipts_dir.exists():
        return []
    window_start = window_end - timedelta(days=window_days)
    out: list[_ReceiptOnDisk] = []
    try:
        candidates = sorted(
            (p for p in receipts_dir.iterdir() if p.is_file() and p.suffix == ".json"),
            key=lambda p: p.name,
        )
    except OSError as exc:
        logger.warning("review.observe_outcomes_cli: cannot list %s: %s", receipts_dir, exc)
        return []
    for path in candidates:
        if len(out) >= max_receipts:
            break
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("could not read %s: %s", path, exc)
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("malformed JSON in %s: %s", path, exc)
            continue
        if not isinstance(payload, dict):
            continue
        reviewed_raw = payload.get("reviewed_at")
        if not reviewed_raw:
            continue
        try:
            reviewed_at = _parse_iso(str(reviewed_raw))
        except (TypeError, ValueError):
            continue
        if not (window_start <= reviewed_at <= window_end):
            continue
        out.append(_ReceiptOnDisk(path=path, payload=payload))
    return out


# ---------------------------------------------------------------------------
# GitHub timeline fetch (bounded; pluggable for tests)
# ---------------------------------------------------------------------------


TimelineProvider = Callable[[int, str, int], tuple[list[Mapping[str, Any]], str | None]]
"""Signature: (pr_number, head_sha, event_cap) -> (events, fetch_error_or_None)."""


def _gh_timeline_provider(
    pr_number: int, head_sha: str, event_cap: int
) -> tuple[list[Mapping[str, Any]], str | None]:
    """Default timeline provider using ``gh api``.

    This is intentionally *minimal* — we fetch only the small set of
    events that ``observe_outcome`` actually inspects, and we cap the
    fetch with ``per_page`` plus a single page request to avoid
    runaway pagination.

    Returns (events, error). On any error returns ([], error_message).
    """
    if shutil.which("gh") is None:
        return [], "gh CLI not found on PATH"
    events: list[Mapping[str, Any]] = []
    # Issue/PR timeline (covers reopen events and labeled issues that
    # mention the merge sha). Bounded to one page of `event_cap` items.
    try:
        proc = subprocess.run(
            [
                "gh",
                "api",
                "-H",
                "Accept: application/vnd.github+json",
                f"/repos/:owner/:repo/issues/{pr_number}/timeline?per_page={min(event_cap, 100)}",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return [], f"gh timeline fetch raised {type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return [], f"gh timeline returned {proc.returncode}: {proc.stderr.strip()[:200]}"
    try:
        data = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        return [], f"timeline JSON parse error: {exc}"
    for entry in data[:event_cap]:
        if not isinstance(entry, dict):
            continue
        normalized = _normalize_timeline_entry(entry, pr_number=pr_number)
        if normalized is not None:
            events.append(normalized)
    return events, None


def _normalize_timeline_entry(
    entry: Mapping[str, Any], *, pr_number: int
) -> Mapping[str, Any] | None:
    """Map a GitHub timeline entry into the shape ``observe_outcome`` expects."""
    event_kind = entry.get("event")
    created_at = entry.get("created_at") or entry.get("submitted_at")
    if not isinstance(created_at, str):
        return None
    if event_kind == "reopened":
        return {"type": "pr_reopened", "at": created_at, "pr_number": pr_number}
    if event_kind == "committed":
        message = ""
        commit = entry.get("commit") if isinstance(entry.get("commit"), dict) else None
        if commit:
            message = str(commit.get("message", ""))
        else:
            message = str(entry.get("message", ""))
        return {"type": "commit", "at": created_at, "message": message}
    if event_kind == "labeled":
        label_obj = entry.get("label") if isinstance(entry.get("label"), dict) else {}
        labels = [str(label_obj.get("name", ""))] if label_obj else []
        return {
            "type": "issue_opened",
            "at": created_at,
            "labels": labels,
            "title": str(entry.get("source", {}).get("issue", {}).get("title", ""))
            if isinstance(entry.get("source"), dict)
            else "",
            "body": "",
        }
    return None


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically via tempfile + rename, preserving permissions."""
    target_dir = path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(target_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Insufficiency receipt
# ---------------------------------------------------------------------------


def _resolve_insufficiency_receipt_path(
    *,
    repo_root: Path,
    round_id: str,
    dry_run: bool = False,
    observed_at: datetime | None = None,
) -> Path:
    """Resolve the insufficiency-receipt destination.

    Non-dry-run (write=True) routes to the round's canonical evolve
    directory. Dry-run isolates the proposed artifact under an
    observe-outcomes/<utc-iso> subtree so dry-run runs cannot pollute
    a round's receipts and cannot ever land under docs/ or tests/.
    """
    if dry_run:
        ts = (observed_at or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
        return (
            repo_root
            / ".aragora"
            / "evolve-round"
            / "observe-outcomes"
            / ts
            / "insufficiency-receipt.json"
        )
    return (
        repo_root
        / ".aragora"
        / "evolve-round"
        / round_id
        / "phase-a-observe-outcomes-insufficiency-receipt.json"
    )


def _build_insufficiency_receipt(
    *,
    window_end: datetime,
    window_days: int,
    max_receipts: int,
    receipts_examined: int,
    receipts_with_signals: int,
    receipts_written: int,
    fetch_errors: int,
    v2_now_present: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    if receipts_examined == 0:
        reasons.append(
            "no_receipts_in_window: 0 settlement receipts found inside the "
            f"{window_days}d window ending {window_end.isoformat()}"
        )
    if receipts_examined > 0 and receipts_with_signals == 0:
        reasons.append(
            "no_signals_fired: every receipt observed produced 0 invalidation "
            "signals; either the window is genuinely clean or the timeline "
            "fetch did not surface revert/incident/redo events"
        )
    if fetch_errors > 0:
        reasons.append(
            f"github_fetch_errors: {fetch_errors} of {receipts_examined} "
            "receipts had timeline fetch errors; rerun with network "
            "connectivity verified"
        )
    if max_receipts <= receipts_examined:
        reasons.append(
            f"max_receipts_cap_hit: --max-receipts={max_receipts} reached, "
            "increase the cap or narrow the window to inspect more"
        )
    return {
        "kind": "phase-a-observe-outcomes-insufficiency-receipt",
        "round": "2026-04-30g",
        "window_end_utc": window_end.isoformat(),
        "window_days": window_days,
        "max_receipts": max_receipts,
        "receipts_examined": receipts_examined,
        "receipts_with_signals_fired": receipts_with_signals,
        "receipts_written": receipts_written,
        "github_fetch_errors": fetch_errors,
        "v2_outcome_fields_now_present_in_window": v2_now_present,
        "remaining_blockers": reasons,
        "next_action_advice": (
            "If v2_outcome_fields_now_present_in_window is True but no signals "
            "fired, the empirical baseline can be measured but the human-side "
            "numerator is genuinely zero in this window — do not invent a "
            "threshold; rerun on a wider window or a later cohort."
            if receipts_examined > 0 and receipts_with_signals == 0
            else "Run with --window-days adjusted, ensure gh CLI is "
            "authenticated, and re-attempt. Do NOT manually populate v2 "
            "outcome fields without observing real timeline evidence."
        ),
    }


# ---------------------------------------------------------------------------
# Core observation loop
# ---------------------------------------------------------------------------


def _materialize_receipt(payload: Mapping[str, Any]) -> Any:
    """Reconstruct a ``SettlementReceipt`` from a JSON payload.

    Imported here lazily so this module does not pull the full
    review-queue CLI graph at top level.
    """
    from aragora.cli.commands.review_queue import SettlementReceipt

    field_names = {f.name for f in SettlementReceipt.__dataclass_fields__.values()}
    kwargs = {k: v for k, v in payload.items() if k in field_names}
    return SettlementReceipt(**kwargs)


def _signal_extract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project the v2 outcome-field subset for before/after comparison."""
    return {name: payload.get(name) for name in sorted(_V2_OUTCOME_FIELD_NAMES)}


def _observe_one(
    receipt: _ReceiptOnDisk,
    *,
    window_days: int,
    timeline_provider: TimelineProvider,
    per_receipt_event_cap: int,
    write: bool,
    observed_at: datetime,
) -> _ObserveResult:
    payload = receipt.payload
    pr_number = int(payload.get("pr_number", 0))
    head_sha = str(payload.get("head_sha", ""))
    if pr_number <= 0 or not head_sha:
        return _ObserveResult(
            receipt_path=receipt.path,
            pr_number=pr_number,
            head_sha=head_sha,
            fetched_event_count=0,
            fetch_error="malformed receipt: missing pr_number or head_sha",
            signals_before=_signal_extract(payload),
            signals_after=_signal_extract(payload),
            written=False,
            skipped_reason="malformed receipt",
        )
    events, fetch_error = timeline_provider(pr_number, head_sha, per_receipt_event_cap)
    if fetch_error is not None:
        return _ObserveResult(
            receipt_path=receipt.path,
            pr_number=pr_number,
            head_sha=head_sha,
            fetched_event_count=len(events),
            fetch_error=fetch_error,
            signals_before=_signal_extract(payload),
            signals_after=_signal_extract(payload),
            written=False,
            skipped_reason="github fetch error",
        )
    receipt_obj = _materialize_receipt(payload)
    observed = observe_outcome(
        receipt_obj,
        github_timeline=events,
        window_days=window_days,
        observed_at=observed_at,
    )
    new_payload = dict(payload)
    for name in _V2_OUTCOME_FIELD_NAMES:
        new_payload[name] = getattr(observed, name)
    written = False
    if write:
        try:
            _atomic_write_json(receipt.path, new_payload)
            written = True
        except OSError as exc:
            return _ObserveResult(
                receipt_path=receipt.path,
                pr_number=pr_number,
                head_sha=head_sha,
                fetched_event_count=len(events),
                fetch_error=None,
                signals_before=_signal_extract(payload),
                signals_after=_signal_extract(new_payload),
                written=False,
                skipped_reason=f"atomic write failed: {exc}",
            )
    return _ObserveResult(
        receipt_path=receipt.path,
        pr_number=pr_number,
        head_sha=head_sha,
        fetched_event_count=len(events),
        fetch_error=None,
        signals_before=_signal_extract(payload),
        signals_after=_signal_extract(new_payload),
        written=written,
    )


def run_observe_outcomes(
    *,
    store_root: str | Path | None,
    repo_root: Path,
    window_end: datetime,
    window_days: int,
    max_receipts: int,
    per_receipt_event_cap: int,
    write: bool,
    timeline_provider: TimelineProvider | None = None,
    insufficiency_receipt_path: Path | None = None,
    round_id: str = "2026-04-30g",
) -> dict[str, Any]:
    """Run the observation pipeline. Pure-function-shaped; testable.

    Returns a structured summary dict that the CLI prints. The
    summary includes a ``"results"`` list (one per receipt examined),
    aggregate counters, the ``v2_outcome_fields_now_present`` boolean,
    and (when applicable) the proposed or written insufficiency
    receipt payload and path.
    """
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    if max_receipts <= 0:
        raise ValueError("max_receipts must be positive")
    if per_receipt_event_cap <= 0:
        raise ValueError("per_receipt_event_cap must be positive")
    provider = timeline_provider or _gh_timeline_provider
    observed_at = datetime.now(UTC)

    receipts = _iter_eligible_receipts(
        store_root=store_root,
        window_end=window_end,
        window_days=window_days,
        max_receipts=max_receipts,
    )
    results: list[_ObserveResult] = []
    for receipt in receipts:
        results.append(
            _observe_one(
                receipt,
                window_days=window_days,
                timeline_provider=provider,
                per_receipt_event_cap=per_receipt_event_cap,
                write=write,
                observed_at=observed_at,
            )
        )

    receipts_with_signals = sum(
        1
        for r in results
        if any(v is True for v in r.signals_after.values() if isinstance(v, bool))
    )
    receipts_written = sum(1 for r in results if r.written)
    fetch_errors = sum(1 for r in results if r.fetch_error is not None)

    # If we wrote any v2 fields, the next baseline measurement should
    # see them. Probe defensively.
    try:
        v2_now_present = _any_receipt_has_v2_outcome_fields(
            store_root=store_root,
            window_end=window_end,
            window_days=window_days,
        )
    except (OSError, ValueError) as exc:
        logger.warning("post-observation probe failed: %s", exc)
        v2_now_present = False

    insufficiency_path: Path | None = None
    insufficiency_payload: dict[str, Any] | None = None
    if len(results) == 0 or receipts_with_signals == 0 or fetch_errors > 0:
        insufficiency_payload = _build_insufficiency_receipt(
            window_end=window_end,
            window_days=window_days,
            max_receipts=max_receipts,
            receipts_examined=len(results),
            receipts_with_signals=receipts_with_signals,
            receipts_written=receipts_written,
            fetch_errors=fetch_errors,
            v2_now_present=v2_now_present,
        )
        insufficiency_path = insufficiency_receipt_path or _resolve_insufficiency_receipt_path(
            repo_root=repo_root,
            round_id=round_id,
            dry_run=not write,
            observed_at=observed_at,
        )
        if write:
            try:
                _atomic_write_json(insufficiency_path, insufficiency_payload)
            except OSError as exc:
                logger.warning(
                    "could not write insufficiency receipt %s: %s",
                    insufficiency_path,
                    exc,
                )

    return {
        "mode": "write" if write else "dry-run",
        "window_end_utc": window_end.isoformat(),
        "window_days": window_days,
        "max_receipts": max_receipts,
        "per_receipt_event_cap": per_receipt_event_cap,
        "receipts_examined": len(results),
        "receipts_with_signals_fired": receipts_with_signals,
        "receipts_written": receipts_written,
        "github_fetch_errors": fetch_errors,
        "v2_outcome_fields_now_present_in_window": v2_now_present,
        "insufficiency_receipt_path": (str(insufficiency_path) if insufficiency_path else None),
        "insufficiency_receipt": insufficiency_payload,
        "results": [
            {
                "receipt_path": str(r.receipt_path),
                "pr_number": r.pr_number,
                "head_sha": r.head_sha,
                "fetched_event_count": r.fetched_event_count,
                "fetch_error": r.fetch_error,
                "signals_before": r.signals_before,
                "signals_after": r.signals_after,
                "written": r.written,
                "skipped_reason": r.skipped_reason,
            }
            for r in results
        ],
    }


# ---------------------------------------------------------------------------
# argparse subparser registration
# ---------------------------------------------------------------------------


def add_observe_outcomes_subparser(sub: argparse._SubParsersAction) -> None:
    """Register ``observe-outcomes`` under the ``review-queue`` subparser."""
    p = sub.add_parser(
        "observe-outcomes",
        help=(
            "Observe post-settlement invalidation signals from GitHub "
            "timeline events and (optionally) write them back into v2 "
            "outcome fields on settlement receipts. Dry-run by default."
        ),
        description=(
            "Round 30g phase A. Iterates settled receipts in a bounded\n"
            "window, fetches GitHub timeline events for each PR with\n"
            "bounded fanout, and computes the five canonical v2 outcome\n"
            "signals via aragora.review.settlement_outcome.observe_outcome.\n\n"
            "Default mode is read-only: nothing is written. Pass --write\n"
            "to mutate receipt JSON files in place. The CLI never invokes\n"
            "git or gh write operations and never edits docs/THESIS.md."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help=f"Observation window in days (default: {DEFAULT_WINDOW_DAYS}).",
    )
    p.add_argument(
        "--max-receipts",
        type=int,
        default=DEFAULT_MAX_RECEIPTS,
        help=(
            "Maximum receipts to inspect in this run "
            f"(default: {DEFAULT_MAX_RECEIPTS}). Bounds the GitHub fanout."
        ),
    )
    p.add_argument(
        "--per-receipt-event-cap",
        type=int,
        default=DEFAULT_PER_RECEIPT_EVENT_CAP,
        help=(
            "Maximum timeline events fetched per receipt "
            f"(default: {DEFAULT_PER_RECEIPT_EVENT_CAP})."
        ),
    )
    p.add_argument(
        "--review-queue-root",
        default=None,
        help=(
            "Override the review-queue store root used for settlement "
            "receipts. Defaults to <repo>/.aragora/review-queue."
        ),
    )
    p.add_argument(
        "--write",
        action="store_true",
        help=(
            "OPT-IN: actually write v2 outcome fields back into receipt "
            "JSON files. Default is dry-run preview only."
        ),
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Output the run summary as JSON.",
    )


# ---------------------------------------------------------------------------
# CLI dispatcher entry point lives in aragora.cli.commands.observe_outcomes_cmd
# (T201 per-file-ignore covers print-rendering there). This module keeps the
# pipeline pure-function-shaped.
# ---------------------------------------------------------------------------
