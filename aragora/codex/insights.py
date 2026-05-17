"""Analysis layer over the read-only Codex Desktop inspector.

Aggregates inspector data into operator-actionable signals: patterns,
anomalies, work-board cross-references, and signed daily digests. All
operations are read-only; the only writes this module performs go under
``.aragora/codex_insights/`` (aragora workspace) for digest receipts.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .desktop_inspector import (
    SessionSummary,
    ThreadSummary,
    list_active_threads,
    redact_display,
    summarize_session,
)
from .desktop_paths import CodexDesktopPaths

DEFAULT_RECEIPT_ROOT = Path(".aragora/codex_insights")
DEFAULT_TOKEN_ANOMALY_THRESHOLD = 100_000
DEFAULT_REPEATED_TOOL_CALL_THRESHOLD = 5
DEFAULT_STUCK_TURN_MINUTES = 5
DEFAULT_RUNAWAY_TOOL_CALLS = 200
INSPECTOR_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class SessionPattern:
    """Aggregate patterns across the analyzed window."""

    window_seconds: int
    thread_count: int
    archived_excluded: bool
    model_distribution: dict[str, int]
    tool_call_distribution: dict[str, int]
    total_tokens_used: int
    distinct_cwds: int
    duration_seconds_p50: float
    duration_seconds_p95: float
    abandoned_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Anomaly:
    """One flagged anomaly in a session."""

    thread_id: str
    rollout_path: str
    kind: str  # runaway_tool_calls | stuck_turn | token_cap_exceeded | repeated_tool_call
    severity: str  # low | medium | high
    detail: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WorkCrossref:
    """Map of one session to PRs/issues/branches it appears to touch."""

    thread_id: str
    rollout_path: str
    cwd: str
    git_branch: str | None
    pr_references: tuple[str, ...]
    issue_references: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "rollout_path": self.rollout_path,
            "cwd": self.cwd,
            "git_branch": self.git_branch,
            "pr_references": list(self.pr_references),
            "issue_references": list(self.issue_references),
        }


@dataclass(frozen=True, slots=True)
class Digest:
    """A complete intelligence digest covering a window."""

    schema_version: str
    window_since: str
    window_until: str
    thread_count: int
    patterns: SessionPattern
    anomalies: tuple[Anomaly, ...]
    crossref: tuple[WorkCrossref, ...]
    inspector_summaries: tuple[dict[str, Any], ...]
    sha256: str
    hmac_sha256: str | None = None
    signed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "window_since": self.window_since,
            "window_until": self.window_until,
            "thread_count": self.thread_count,
            "patterns": self.patterns.to_dict(),
            "anomalies": [a.to_dict() for a in self.anomalies],
            "crossref": [c.to_dict() for c in self.crossref],
            "inspector_summaries": list(self.inspector_summaries),
            "sha256": self.sha256,
            "hmac_sha256": self.hmac_sha256,
            "signed_at": self.signed_at,
        }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = max(0, min(len(sorted_vals) - 1, int(round(pct * (len(sorted_vals) - 1)))))
    return float(sorted_vals[idx])


def _session_duration_seconds(summary: SessionSummary) -> float:
    if summary.started_at is None or summary.last_event_at is None:
        return 0.0
    return max(0.0, (summary.last_event_at - summary.started_at).total_seconds())


def _is_abandoned(thread: ThreadSummary, summary: SessionSummary, *, now: datetime) -> bool:
    """A thread is 'abandoned' if its last event is far behind its updated_at marker.

    Codex Desktop bumps ``updated_at`` on UI focus changes; if the rollout file
    has been silent for >10 minutes while the thread is still ticking via UI,
    it suggests the agent stopped emitting work.
    """
    if summary.last_event_at is None:
        return True
    silence_seconds = (now - summary.last_event_at).total_seconds()
    return silence_seconds > 600


def _redacted_path(value: Path) -> str:
    return redact_display(value) or ""


def summarize_patterns(
    *,
    since: timedelta,
    include_archived: bool = False,
    paths: CodexDesktopPaths | None = None,
    max_events_per_summary: int = 1500,
) -> tuple[SessionPattern, list[tuple[ThreadSummary, SessionSummary]]]:
    """Return aggregate patterns + the per-thread summaries used to build them.

    The second return value is exposed so callers (anomalies, digest) can
    re-use the same summaries without re-walking each rollout file.
    """
    threads = list_active_threads(
        since=since,
        include_archived=include_archived,
        paths=paths,
    )
    now = datetime.now(UTC)
    per_thread: list[tuple[ThreadSummary, SessionSummary]] = []
    tool_call_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    durations: list[float] = []
    abandoned = 0
    total_tokens = 0
    cwds: set[str] = set()

    for thread in threads:
        if not thread.rollout_path.exists():
            continue
        summary = summarize_session(thread.rollout_path, max_events=max_events_per_summary)
        per_thread.append((thread, summary))
        for name, count in summary.tool_call_counts.items():
            tool_call_counts[name] += count
        if thread.model:
            model_counts[thread.model] += 1
        durations.append(_session_duration_seconds(summary))
        if _is_abandoned(thread, summary, now=now):
            abandoned += 1
        total_tokens += thread.tokens_used
        if thread.cwd:
            cwds.add(thread.cwd)

    pattern = SessionPattern(
        window_seconds=int(since.total_seconds()),
        thread_count=len(threads),
        archived_excluded=not include_archived,
        model_distribution=dict(model_counts),
        tool_call_distribution=dict(tool_call_counts),
        total_tokens_used=total_tokens,
        distinct_cwds=len(cwds),
        duration_seconds_p50=_percentile(durations, 0.5),
        duration_seconds_p95=_percentile(durations, 0.95),
        abandoned_count=abandoned,
    )
    return pattern, per_thread


def detect_anomalies(
    pairs: list[tuple[ThreadSummary, SessionSummary]],
    *,
    token_cap: int = DEFAULT_TOKEN_ANOMALY_THRESHOLD,
    runaway_tool_calls: int = DEFAULT_RUNAWAY_TOOL_CALLS,
    stuck_turn_minutes: int = DEFAULT_STUCK_TURN_MINUTES,
) -> list[Anomaly]:
    """Return a list of anomalies detected across the (thread, summary) pairs.

    Anomalies in order of typical severity:
    - ``runaway_tool_calls``: aggregate tool calls in window > ``runaway_tool_calls``
      with no recent user turn
    - ``stuck_turn``: last event is ``turn_start`` (or similar) with no
      matching completion and silence exceeds ``stuck_turn_minutes``
    - ``token_cap_exceeded``: thread.tokens_used > ``token_cap``
    """
    now = datetime.now(UTC)
    anomalies: list[Anomaly] = []
    stuck_window = timedelta(minutes=stuck_turn_minutes)

    for thread, summary in pairs:
        total_tool_calls = sum(summary.tool_call_counts.values())
        if total_tool_calls >= runaway_tool_calls:
            anomalies.append(
                Anomaly(
                    thread_id=thread.id,
                    rollout_path=_redacted_path(summary.rollout_path),
                    kind="runaway_tool_calls",
                    severity="high",
                    detail=(
                        f"{total_tool_calls} tool calls in scanned window "
                        f"({summary.events_scanned} events, "
                        f"{'truncated' if summary.truncated else 'complete'})"
                    ),
                    evidence={
                        "tool_call_counts": dict(summary.tool_call_counts),
                        "events_scanned": summary.events_scanned,
                        "truncated": summary.truncated,
                    },
                )
            )

        if thread.tokens_used >= token_cap:
            anomalies.append(
                Anomaly(
                    thread_id=thread.id,
                    rollout_path=_redacted_path(summary.rollout_path),
                    kind="token_cap_exceeded",
                    severity="medium",
                    detail=(f"tokens_used={thread.tokens_used} >= cap={token_cap}"),
                    evidence={"tokens_used": thread.tokens_used, "cap": token_cap},
                )
            )

        # Stuck-turn heuristic: more turn_starts than turn_completes AND the
        # session has been silent past the threshold. Using the count delta is
        # more robust than trying to read "last event type" from aggregate
        # counts (which only tell us frequency, not order).
        turn_starts = sum(
            c for k, c in summary.event_type_counts.items() if k.endswith("turn_start")
        )
        turn_completes = sum(
            c for k, c in summary.event_type_counts.items() if k.endswith("turn_complete")
        )
        if (
            summary.last_event_at is not None
            and (now - summary.last_event_at) > stuck_window
            and turn_starts > turn_completes
        ):
            silence_min = int((now - summary.last_event_at).total_seconds() // 60)
            anomalies.append(
                Anomaly(
                    thread_id=thread.id,
                    rollout_path=_redacted_path(summary.rollout_path),
                    kind="stuck_turn",
                    severity="medium",
                    detail=(
                        f"last event silence={silence_min}m (threshold={stuck_turn_minutes}m), "
                        f"turn_starts={turn_starts} > turn_completes={turn_completes}"
                    ),
                    evidence={
                        "last_event_at": summary.last_event_at.isoformat(),
                        "stuck_threshold_minutes": stuck_turn_minutes,
                        "turn_starts": turn_starts,
                        "turn_completes": turn_completes,
                    },
                )
            )

    # Order high → medium → low so operators see the worst first.
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    anomalies.sort(key=lambda a: (severity_rank.get(a.severity, 9), a.thread_id))
    return anomalies


_PR_PATTERN = "#"


def _scan_pr_issue_refs(text: str) -> tuple[list[str], list[str]]:
    """Extract simple ``#N`` references from arbitrary text.

    We don't try to distinguish PR vs issue from the bare ``#N`` form since
    that requires a GitHub round-trip; the caller can disambiguate later via
    ``gh pr view`` if needed. We return all matches in both lists for now.
    """
    refs: list[str] = []
    if not text:
        return [], []
    i = 0
    while i < len(text):
        if text[i] == _PR_PATTERN and i + 1 < len(text) and text[i + 1].isdigit():
            j = i + 1
            while j < len(text) and text[j].isdigit():
                j += 1
            number = text[i + 1 : j]
            if number and len(number) <= 8:
                refs.append(f"#{number}")
            i = j
        else:
            i += 1
    return refs, refs


def crossref_work_board(
    pairs: list[tuple[ThreadSummary, SessionSummary]],
) -> list[WorkCrossref]:
    """Cross-reference each session to PR/issue references found in metadata."""
    crossrefs: list[WorkCrossref] = []
    for thread, summary in pairs:
        text = " ".join(
            [
                thread.title or "",
                thread.first_user_message or "",
                summary.first_user_message or "",
                summary.last_user_message or "",
                thread.git_branch or "",
            ]
        )
        pr_refs, issue_refs = _scan_pr_issue_refs(text)
        crossrefs.append(
            WorkCrossref(
                thread_id=thread.id,
                rollout_path=_redacted_path(summary.rollout_path),
                cwd=thread.cwd,
                git_branch=thread.git_branch,
                pr_references=tuple(sorted(set(pr_refs))),
                issue_references=tuple(sorted(set(issue_refs))),
            )
        )
    return crossrefs


def _sha256_of_payload(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _try_hmac_sign(payload_sha: str) -> str | None:
    """Best-effort HMAC sign via the existing context_signing key."""
    try:
        from aragora.security.context_signing import get_signing_key

        key = get_signing_key()
    except Exception:
        return None
    if not key:
        return None
    import hmac

    return hmac.new(key, payload_sha.encode("utf-8"), hashlib.sha256).hexdigest()


def build_digest(
    *,
    since: timedelta,
    include_archived: bool = False,
    paths: CodexDesktopPaths | None = None,
    max_events_per_summary: int = 1500,
) -> Digest:
    """Build a complete signed digest covering the analyzed window."""
    pattern, pairs = summarize_patterns(
        since=since,
        include_archived=include_archived,
        paths=paths,
        max_events_per_summary=max_events_per_summary,
    )
    anomalies = detect_anomalies(pairs)
    crossref = crossref_work_board(pairs)
    inspector_summaries = tuple(
        {
            "thread_id": t.id,
            "rollout_path": _redacted_path(s.rollout_path),
            "events_scanned": s.events_scanned,
            "truncated": s.truncated,
            "event_type_counts": dict(s.event_type_counts),
            "tool_call_counts": dict(s.tool_call_counts),
            "model_provider": s.model_provider,
        }
        for t, s in pairs
    )

    now = datetime.now(UTC)
    window_since = (now - since).isoformat()
    window_until = now.isoformat()
    payload: dict[str, Any] = {
        "schema_version": INSPECTOR_SCHEMA_VERSION,
        "window_since": window_since,
        "window_until": window_until,
        "thread_count": pattern.thread_count,
        "patterns": pattern.to_dict(),
        "anomalies": [a.to_dict() for a in anomalies],
        "crossref": [c.to_dict() for c in crossref],
        "inspector_summaries": list(inspector_summaries),
    }
    sha = _sha256_of_payload(payload)
    hmac_signature = _try_hmac_sign(sha)

    return Digest(
        schema_version=INSPECTOR_SCHEMA_VERSION,
        window_since=window_since,
        window_until=window_until,
        thread_count=pattern.thread_count,
        patterns=pattern,
        anomalies=tuple(anomalies),
        crossref=tuple(crossref),
        inspector_summaries=inspector_summaries,
        sha256=sha,
        hmac_sha256=hmac_signature,
        signed_at=now.isoformat() if hmac_signature else None,
    )


def persist_digest(
    digest: Digest,
    *,
    root: Path | None = None,
    timestamp: datetime | None = None,
) -> Path:
    """Write the digest as a JSON file under ``.aragora/codex_insights/`` (or override)."""
    output_root = root or DEFAULT_RECEIPT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    when = timestamp or datetime.now(UTC)
    suffix = when.strftime("%Y%m%dT%H%M%SZ")
    target = output_root / f"digest-{suffix}.json"
    target.write_text(
        json.dumps(digest.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return target


def ingest_digest_into_km(
    digest_path: Path,
    *,
    label: str = "codex-insights-digest",
    timeout_seconds: float = 30.0,
) -> tuple[bool, str]:
    """Best-effort: ingest a digest via the existing ``aragora km store`` CLI.

    Returns ``(ok, detail)`` rather than raising — KM may be offline locally
    and we don't want to fail the whole insights pipeline on that.
    """
    if not digest_path.exists():
        return False, f"digest file not found: {digest_path}"
    cmd = [
        "aragora",
        "km",
        "store",
        "--source",
        label,
        "--text",
        f"@{digest_path}",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return False, f"km ingest invocation failed: {exc}"
    if result.returncode != 0:
        return False, (result.stderr or result.stdout or "km store returned non-zero").strip()
    return True, (result.stdout or "ok").strip()
