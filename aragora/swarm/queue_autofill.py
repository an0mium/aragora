"""Automatic restock of the boss-loop queue when it goes empty.

When the boss loop picks up no eligible issue for ``N`` consecutive ticks
(default 3) we call back into the issue scanner to surface a small batch of
well-proven candidates. This keeps the autonomous loop from idling while
waiting for a human to re-label work.

Design constraints (from the mission brief)
-------------------------------------------
* **Default off** — the feature is gated on
  :data:`QUEUE_AUTOFILL_FLAG_ENV` (``ARAGORA_QUEUE_AUTOFILL``).  When the
  flag is falsy the loop calls :func:`maybe_autofill_queue` but it returns a
  ``skipped`` result without touching GitHub.
* **Well-proven categories only** — we only scan ``test_coverage`` and
  ``broad_exception`` (high historical success).  Anything else is ignored.
* **Small batches** — never surface more than ``max_issues`` (default 3)
  advisory candidates in one invocation.
* **Filter-aligned** — every candidate is routed through
  :func:`classify_proof_first_queue_issue` with the current roadmap policy
  *before* it is surfaced. If the filter would reject it, we drop it.
* **Rate limited** — at most one autofill per ``min_interval_seconds``
  (default 3600).  The last-run timestamp is persisted in a small JSON
  sentinel inside ``.aragora/overnight/`` so it survives process restarts.
* **Auditable** — every invocation writes a single ``event: queue_autofill``
  row to the boss metrics JSONL so operators can see exactly what happened.

The module is intentionally thin: it composes pieces that already live in
the codebase (``issue_scanner.scan_all``, ``classify_proof_first_queue_issue``,
``boss_validation.assess_issue_body_sanitation``) and contributes only the
policy wrapper.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


QUEUE_AUTOFILL_FLAG_ENV = "ARAGORA_QUEUE_AUTOFILL"

_TRUTHY_VALUES = frozenset({"1", "true", "t", "yes", "y", "on"})

DEFAULT_EMPTY_TICK_THRESHOLD = 3
DEFAULT_MAX_ISSUES = 3
DEFAULT_MIN_INTERVAL_SECONDS = 3600.0
DEFAULT_SENTINEL_PATH = Path(".aragora/overnight/queue_autofill.json")
DEFAULT_METRICS_PATH = Path(".aragora/overnight/boss_metrics.jsonl")

# Only well-proven scanner categories are eligible for autofill.
ALLOWED_CATEGORIES: frozenset[str] = frozenset({"test_coverage", "broad_exception"})


@dataclass(frozen=True)
class AutofillCandidate:
    """Lightweight summary of a candidate that passed the autofill filters."""

    title: str
    category: str
    fingerprint: str
    file_scope: tuple[str, ...]
    lane: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "category": self.category,
            "fingerprint": self.fingerprint,
            "file_scope": list(self.file_scope),
            "lane": self.lane,
        }


@dataclass(frozen=True)
class AutofillResult:
    """Outcome of one autofill invocation."""

    attempted: bool
    reason: str
    consecutive_empty_ticks: int
    threshold: int
    rate_limited: bool = False
    seconds_since_last: float | None = None
    scanned_count: int = 0
    eligible_count: int = 0
    filtered_out: int = 0
    duplicate_count: int = 0
    created: tuple[AutofillCandidate, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def created_count(self) -> int:
        return len(self.created)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": "queue_autofill",
            "attempted": self.attempted,
            "reason": self.reason,
            "consecutive_empty_ticks": self.consecutive_empty_ticks,
            "threshold": self.threshold,
            "rate_limited": self.rate_limited,
            "seconds_since_last": self.seconds_since_last,
            "scanned_count": self.scanned_count,
            "eligible_count": self.eligible_count,
            "filtered_out": self.filtered_out,
            "duplicate_count": self.duplicate_count,
            "created_count": self.created_count,
            "created": [candidate.to_dict() for candidate in self.created],
            "errors": list(self.errors),
        }


# ---------------------------------------------------------------------------
# Feature-flag helper
# ---------------------------------------------------------------------------


def queue_autofill_enabled(
    env: dict[str, str] | None = None,
    *,
    flag: str = QUEUE_AUTOFILL_FLAG_ENV,
) -> bool:
    """Return ``True`` when the autofill feature flag is truthy."""
    source = env if env is not None else os.environ
    value = str(source.get(flag, "")).strip().lower()
    return value in _TRUTHY_VALUES


# ---------------------------------------------------------------------------
# Rate-limit sentinel helpers
# ---------------------------------------------------------------------------


def _read_last_run(sentinel_path: Path) -> float | None:
    try:
        payload = json.loads(sentinel_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("last_run_ts") if isinstance(payload, dict) else None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _write_last_run(sentinel_path: Path, timestamp: float) -> None:
    try:
        sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        sentinel_path.write_text(
            json.dumps({"last_run_ts": float(timestamp)}, sort_keys=True),
            encoding="utf-8",
        )
    except OSError as exc:  # pragma: no cover - best effort
        logger.warning("queue_autofill: failed to persist sentinel at %s: %s", sentinel_path, exc)


# ---------------------------------------------------------------------------
# Core decision function
# ---------------------------------------------------------------------------


def maybe_autofill_queue(
    *,
    repo_root: Path,
    consecutive_empty_ticks: int,
    existing_candidates: Iterable[Any] = (),
    create_issue: Callable[[AutofillCandidate], bool] | None = None,
    env: dict[str, str] | None = None,
    now: float | None = None,
    threshold: int = DEFAULT_EMPTY_TICK_THRESHOLD,
    max_issues: int = DEFAULT_MAX_ISSUES,
    min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS,
    categories: Sequence[str] = ("test_coverage", "broad_exception"),
    sentinel_path: Path | None = None,
    metrics_jsonl_path: Path | None = None,
    scan_fn: Callable[..., list[Any]] | None = None,
    classify_fn: Callable[..., Any] | None = None,
    validate_body_fn: Callable[[str], tuple[bool, str]] | None = None,
    format_body_fn: Callable[[Any], str] | None = None,
) -> AutofillResult:
    """Decide whether to surface advisory restock candidates.

    ``existing_candidates`` is the list of boss-loop-visible candidates the
    caller already rejected (because they didn't meet the filter).  We use
    that list only to decide whether restocking is justified — if any of the
    existing candidates *would* pass the canonical filter today we skip the
    autofill entirely to avoid duplicating tracked work.

    ``create_issue`` is retained for backward compatibility but is ignored.
    Queue autofill is advisory-only and does not create GitHub issues; any
    actual queue widening must flow through explicit human settlement.
    """
    threshold = max(1, int(threshold))
    max_issues = max(0, int(max_issues))
    min_interval_seconds = max(0.0, float(min_interval_seconds))
    timestamp = float(now) if now is not None else time.time()
    repo_root = Path(repo_root).resolve()
    sentinel_path = (sentinel_path or (repo_root / DEFAULT_SENTINEL_PATH)).resolve()
    metrics_jsonl_path = (
        metrics_jsonl_path.resolve()
        if metrics_jsonl_path is not None
        else (repo_root / DEFAULT_METRICS_PATH).resolve()
    )

    if not queue_autofill_enabled(env):
        return _finalize(
            AutofillResult(
                attempted=False,
                reason="flag_disabled",
                consecutive_empty_ticks=consecutive_empty_ticks,
                threshold=threshold,
            ),
            metrics_jsonl_path,
        )

    if consecutive_empty_ticks < threshold:
        return _finalize(
            AutofillResult(
                attempted=False,
                reason="below_threshold",
                consecutive_empty_ticks=consecutive_empty_ticks,
                threshold=threshold,
            ),
            metrics_jsonl_path,
        )

    if max_issues <= 0:
        return _finalize(
            AutofillResult(
                attempted=False,
                reason="max_issues_zero",
                consecutive_empty_ticks=consecutive_empty_ticks,
                threshold=threshold,
            ),
            metrics_jsonl_path,
        )

    # Rate limit: one run per min_interval_seconds.
    last_run_ts = _read_last_run(sentinel_path)
    seconds_since_last = (timestamp - last_run_ts) if last_run_ts is not None else None
    if (
        last_run_ts is not None
        and seconds_since_last is not None
        and seconds_since_last < min_interval_seconds
    ):
        return _finalize(
            AutofillResult(
                attempted=False,
                reason="rate_limited",
                consecutive_empty_ticks=consecutive_empty_ticks,
                threshold=threshold,
                rate_limited=True,
                seconds_since_last=seconds_since_last,
            ),
            metrics_jsonl_path,
        )

    # If any already-known candidate would pass the canonical filter, there
    # is already suitable work — don't duplicate.
    classify_fn = classify_fn or _default_classify
    pre_existing_passes = sum(
        1
        for candidate in existing_candidates
        if _candidate_would_pass_filter(candidate, classify_fn, repo_root)
    )
    if pre_existing_passes > 0:
        return _finalize(
            AutofillResult(
                attempted=False,
                reason="existing_queue_has_work",
                consecutive_empty_ticks=consecutive_empty_ticks,
                threshold=threshold,
                seconds_since_last=seconds_since_last,
                duplicate_count=pre_existing_passes,
            ),
            metrics_jsonl_path,
        )

    scan_fn = scan_fn or _default_scan
    validate_body_fn = validate_body_fn or _default_validate_body
    format_body_fn = format_body_fn or _default_format_body

    allowed_cats = tuple(cat for cat in categories if cat in ALLOWED_CATEGORIES)
    if not allowed_cats:
        return _finalize(
            AutofillResult(
                attempted=False,
                reason="no_allowed_categories",
                consecutive_empty_ticks=consecutive_empty_ticks,
                threshold=threshold,
            ),
            metrics_jsonl_path,
        )

    errors: list[str] = []
    try:
        scanned = list(scan_fn(repo_root, categories=list(allowed_cats)))
    except Exception as exc:
        logger.warning("queue_autofill: scanner raised: %s", exc)
        return _finalize(
            AutofillResult(
                attempted=True,
                reason="scan_failed",
                consecutive_empty_ticks=consecutive_empty_ticks,
                threshold=threshold,
                seconds_since_last=seconds_since_last,
                errors=(str(exc),),
            ),
            metrics_jsonl_path,
        )

    eligible: list[tuple[Any, str, str]] = []
    filtered_out = 0
    for candidate in scanned:
        category = getattr(candidate, "category", "")
        if category not in ALLOWED_CATEGORIES:
            filtered_out += 1
            continue
        body = format_body_fn(candidate)
        ok, reason = validate_body_fn(body)
        if not ok:
            filtered_out += 1
            logger.debug(
                "queue_autofill: validation rejected candidate %r: %s",
                getattr(candidate, "title", ""),
                reason,
            )
            continue
        decision = classify_fn(
            getattr(candidate, "title", ""),
            body,
            labels=("boss-ready",),
            repo_root=repo_root,
        )
        if not getattr(decision, "allowed", False):
            filtered_out += 1
            continue
        lane = getattr(decision, "lane", "")
        eligible.append((candidate, body, lane))

    if not eligible:
        result = AutofillResult(
            attempted=True,
            reason="no_eligible_candidates",
            consecutive_empty_ticks=consecutive_empty_ticks,
            threshold=threshold,
            seconds_since_last=seconds_since_last,
            scanned_count=len(scanned),
            eligible_count=0,
            filtered_out=filtered_out,
        )
        _write_last_run(sentinel_path, timestamp)
        return _finalize(result, metrics_jsonl_path)

    trimmed = eligible[:max_issues]
    created = tuple(_as_autofill_candidate(candidate, lane) for candidate, _body, lane in trimmed)
    if create_issue is not None:
        errors.append("create_issue callback ignored; queue autofill is advisory-only")

    result = AutofillResult(
        attempted=True,
        reason="created_dry_run",
        consecutive_empty_ticks=consecutive_empty_ticks,
        threshold=threshold,
        seconds_since_last=seconds_since_last,
        scanned_count=len(scanned),
        eligible_count=len(eligible),
        filtered_out=filtered_out,
        created=created,
        errors=tuple(errors),
    )
    _write_last_run(sentinel_path, timestamp)
    return _finalize(result, metrics_jsonl_path)


# ---------------------------------------------------------------------------
# Defaults that lazily pull the real implementations
# ---------------------------------------------------------------------------


def _default_scan(repo_root: Path, *, categories: list[str]) -> list[Any]:
    from aragora.swarm.issue_scanner import scan_all

    return scan_all(repo_root, categories=categories)


def _default_classify(
    title: str,
    body: str,
    *,
    labels: Iterable[str] = (),
    repo_root: Path | None = None,
) -> Any:
    from aragora.swarm.proof_first_queue import classify_proof_first_queue_issue

    return classify_proof_first_queue_issue(
        title,
        body,
        labels=tuple(labels),
        repo_root=repo_root,
    )


def _default_validate_body(body: str) -> tuple[bool, str]:
    try:
        from aragora.swarm.boss_validation import assess_issue_body_sanitation

        ok, reason = assess_issue_body_sanitation(body)
        return ok, reason or ""
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("queue_autofill: validation fallback triggered: %s", exc)
        if "## Task" not in body or len(body.strip()) < 50:
            return False, "fallback_validation_failed"
        return True, ""


def _default_format_body(candidate: Any) -> str:
    """Minimal boss-ready body used when the caller did not supply a formatter.

    The real generator script builds a richer body via the issue upgrader.
    For the autofill guardrails we only need enough structure to let the
    sanitizer recognise the shape of a proper boss issue.
    """
    title = getattr(candidate, "title", "autofill candidate")
    description = getattr(candidate, "description", title)
    file_scope = list(getattr(candidate, "file_scope", []) or [])
    new_files = list(getattr(candidate, "new_files", []) or [])
    validation = getattr(candidate, "validation_command", "")
    criteria = list(getattr(candidate, "acceptance_criteria", []) or [])
    fingerprint = getattr(candidate, "fingerprint", "")

    parts: list[str] = [f"## Task\n\n{description}"]
    scope_lines: list[str] = [f"- `{path}`" for path in file_scope]
    scope_lines.extend(f"- `{path}` (create)" for path in new_files)
    if scope_lines:
        parts.append("### File Scope\n" + "\n".join(scope_lines))
    if validation:
        parts.append(f"### Validation\n```bash\n{validation}\n```")
    if criteria:
        parts.append("### Acceptance Criteria\n" + "\n".join(f"- {item}" for item in criteria))
    if fingerprint:
        parts.append(f"<!-- fingerprint:{fingerprint} -->")
    return "\n\n".join(parts)


def _as_autofill_candidate(candidate: Any, lane: str) -> AutofillCandidate:
    file_scope = tuple(getattr(candidate, "file_scope", ()) or ())
    return AutofillCandidate(
        title=str(getattr(candidate, "title", "")),
        category=str(getattr(candidate, "category", "")),
        fingerprint=str(getattr(candidate, "fingerprint", "")),
        file_scope=file_scope,
        lane=str(lane or ""),
    )


def _candidate_would_pass_filter(
    candidate: Any,
    classify_fn: Callable[..., Any],
    repo_root: Path,
) -> bool:
    title = getattr(candidate, "title", "")
    body = getattr(candidate, "body", "") or ""
    labels = getattr(candidate, "labels", ()) or ()
    try:
        decision = classify_fn(title, body, labels=tuple(labels), repo_root=repo_root)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("queue_autofill: classify probe failed: %s", exc)
        return False
    return bool(getattr(decision, "allowed", False))


def _finalize(result: AutofillResult, metrics_jsonl_path: Path) -> AutofillResult:
    _emit_metrics(result, metrics_jsonl_path)
    return result


def _emit_metrics(result: AutofillResult, metrics_jsonl_path: Path) -> None:
    try:
        metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        payload = result.to_dict()
        payload["timestamp"] = time.time()
        with metrics_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")
    except OSError as exc:  # pragma: no cover - best effort
        logger.debug("queue_autofill: metrics emit skipped: %s", exc)


__all__ = [
    "ALLOWED_CATEGORIES",
    "AutofillCandidate",
    "AutofillResult",
    "DEFAULT_EMPTY_TICK_THRESHOLD",
    "DEFAULT_MAX_ISSUES",
    "DEFAULT_MIN_INTERVAL_SECONDS",
    "QUEUE_AUTOFILL_FLAG_ENV",
    "maybe_autofill_queue",
    "queue_autofill_enabled",
]
