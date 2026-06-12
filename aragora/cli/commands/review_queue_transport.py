"""GitHub CLI transport helpers for review-queue commands."""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any


GH_COMMAND_TIMEOUT_SECONDS = 30
GITHUB_TRANSPORT_ERROR_KIND = "github_transport"
GITHUB_TRANSPORT_BLOCKED_STATUS = "transport_blocked"
_GITHUB_TRANSPORT_ERROR_MARKERS = (
    "api rate limit already exceeded",
    "check your internet connection",
    "client.timeout exceeded",
    "connection refused",
    "connection reset",
    "connection timed out",
    "context deadline exceeded",
    "could not resolve host",
    "error connecting",
    "failed to start",
    "http 502",
    "http 503",
    "http 504",
    "i/o timeout",
    "net/http",
    "no such host",
    "operation timed out",
    "rate limit exceeded",
    "temporary failure in name resolution",
    "timeout awaiting response headers",
    "tls handshake timeout",
)


class _GhError(RuntimeError):
    """Raised when a 'gh' invocation fails or returns malformed JSON."""


def _command_timeout_message(cmd: list[str], timeout_seconds: int) -> str:
    return f"{' '.join(cmd)} timed out after {timeout_seconds}s"


def _gh_text(args: list[str]) -> str:
    """Run a 'gh' command and return plain stdout."""
    cmd = ["gh", *args]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=GH_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise _GhError(
            _command_timeout_message(cmd, int(exc.timeout or GH_COMMAND_TIMEOUT_SECONDS))
        ) from exc
    except OSError as exc:
        raise _GhError(f"{' '.join(cmd)} failed to start: {exc}") from exc
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or "no stderr"
        raise _GhError(f"gh {' '.join(args)} failed: {stderr}")
    return proc.stdout.strip()


def _gh_json(args: list[str]) -> Any:
    """Run a 'gh' command and parse JSON output. Returns None for empty stdout."""
    cmd = ["gh", *args]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=GH_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise _GhError(
            _command_timeout_message(cmd, int(exc.timeout or GH_COMMAND_TIMEOUT_SECONDS))
        ) from exc
    except OSError as exc:
        raise _GhError(f"{' '.join(cmd)} failed to start: {exc}") from exc
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or "no stderr"
        raise _GhError(f"gh {' '.join(args)} failed: {stderr}")
    out = proc.stdout.strip()
    if not out:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError as exc:
        raise _GhError(f"gh {' '.join(args)} returned malformed JSON: {exc}") from exc


def _gh_error_kind(error: object) -> str:
    """Return a stable machine-readable kind for GitHub helper failures."""
    text = str(error or "").lower()
    if any(marker in text for marker in _GITHUB_TRANSPORT_ERROR_MARKERS):
        return GITHUB_TRANSPORT_ERROR_KIND
    return "github_error"


def _is_github_transport_error(error: object) -> bool:
    return _gh_error_kind(error) == GITHUB_TRANSPORT_ERROR_KIND


def _gh_json_with_transport_retries(
    args: list[str],
    *,
    gh_json: Callable[[list[str]], Any] = _gh_json,
    attempts: int = 2,
    delay_seconds: float = 0.0,
) -> Any:
    """Run a bounded gh JSON call, retrying only transient transport failures."""
    bounded_attempts = max(1, attempts)
    last_error: Exception | None = None
    for attempt in range(1, bounded_attempts + 1):
        try:
            return gh_json(args)
        except Exception as exc:
            last_error = exc
            if not _is_github_transport_error(exc) or attempt >= bounded_attempts:
                raise
            if delay_seconds > 0:
                time.sleep(delay_seconds)
    if last_error is not None:
        raise last_error
    raise _GhError(f"gh {' '.join(args)} failed without an error")


def _numeric_pr_refs(pr_refs: list[str]) -> list[int]:
    numbers: list[int] = []
    for ref in pr_refs:
        match = re.search(r"\d+", str(ref))
        if match:
            numbers.append(int(match.group(0)))
    return numbers


def _transport_blocked_next_prompt(command_name: str, pr_refs: list[str]) -> str:
    pr_text = ", ".join(f"#{number}" for number in _numeric_pr_refs(pr_refs)) or "the target PR"
    return (
        "Do not rely on transcript state. Re-check live GitHub/local state first. "
        f"Primary task: retry {command_name} for {pr_text} only after GitHub API transport "
        "recovers. Treat this packet as preserve/no-mutate; do not mark ready, collect "
        "evidence, record settlement, merge, close, label, rerun workflows, or touch branch "
        "protection while transport is blocked. If the prompt above accomplishes no "
        "incremental progress, make the next prompt one that does. If any work can be better "
        "automated by improving Aragora tooling at a meta level, include the concrete tooling "
        "improvement plan instead of repeating manual queue checks. Always include a final "
        "summary section with the best next recursive prompt."
    )


def _transport_blocked_fields(
    *,
    error: str,
    command_name: str,
    pr_refs: list[str],
    repo_override: str | None,
) -> dict[str, Any]:
    return {
        "status": GITHUB_TRANSPORT_BLOCKED_STATUS,
        "transport_blocked": True,
        "preserve_no_mutate": True,
        "retryable": _is_github_transport_error(error),
        "error_kind": _gh_error_kind(error),
        "error": error,
        "command": command_name,
        "repo": repo_override or "",
        "pr_refs": [str(ref) for ref in pr_refs],
        "not_ready": _numeric_pr_refs(pr_refs),
        "next_prompt": _transport_blocked_next_prompt(command_name, pr_refs),
    }


def _merge_packet_transport_blocked_envelope(
    *,
    error: str,
    pr_refs: list[str],
    repo_override: str | None,
    limit: int,
    queue_cap: int = 6,
) -> dict[str, Any]:
    return {
        "version": "merge_authorization_packet.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "queue_pressure": {
            "current_open_prs": len(pr_refs),
            "cap": queue_cap,
            "active": False,
            "scope": "explicit_pr_refs" if pr_refs else "open_pr_queue",
        },
        "limit": limit,
        "entries": [],
        "admin_squash_order": [],
        "human_risk_settlement_required": [],
        "admin_squash_allowed": False,
        **_transport_blocked_fields(
            error=error,
            command_name="review-queue merge-packet",
            pr_refs=pr_refs,
            repo_override=repo_override,
        ),
    }


def _queue_conductor_transport_blocked_envelope(
    *,
    error: str,
    pr_refs: list[str],
    repo_override: str | None,
    limit: int,
    mode: str,
) -> dict[str, Any]:
    return {
        "version": "queue_conductor.v1",
        "mode": mode,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "origin_main_sha": "",
        "limit": limit,
        "initial_heads": {},
        "candidates": [],
        "tooling_notes": [
            "read_only_packet",
            "transport_failure_is_preserve_no_mutate",
        ],
        **_transport_blocked_fields(
            error=error,
            command_name="review-queue conductor",
            pr_refs=pr_refs,
            repo_override=repo_override,
        ),
    }
