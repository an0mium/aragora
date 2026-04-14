"""Issue dispatch claim lock for parallel boss loop safety.

Prevents two boss loop instances from dispatching the same issue by
writing a claim file to .aragora/issue_claims/{issue_number}.lock.
Claims auto-expire after TTL or when the owning process dies.

Extracted from boss_loop.py to keep it under the LOC ratchet.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
UTC = timezone.utc

ISSUE_CLAIM_TTL_SECONDS = 30 * 60


def issue_claims_dir() -> Path:
    return Path.cwd() / ".aragora" / "issue_claims"


def issue_claim_path(issue_number: int) -> Path:
    return issue_claims_dir() / f"{int(issue_number)}.lock"


def read_issue_claim_payload(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def issue_claim_owner_pid(payload: dict[str, Any] | None) -> int | None:
    if not isinstance(payload, dict):
        return None
    try:
        pid = int(payload.get("pid", 0) or 0)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def claim_owned_by(payload: dict[str, Any] | None, run_id: str) -> bool:
    return (
        isinstance(payload, dict)
        and str(payload.get("run_id", "")).strip() == run_id
        and issue_claim_owner_pid(payload) == os.getpid()
    )


def reap_stale_claim(
    issue_number: int,
    path: Path,
    payload: dict[str, Any] | None,
    run_id: str,
) -> bool:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return True
    except OSError:
        return False

    stale_reason: str | None = None
    age_seconds = max(0.0, time.time() - stat.st_mtime)
    if age_seconds > ISSUE_CLAIM_TTL_SECONDS:
        stale_reason = "expired"
    else:
        owner_pid = issue_claim_owner_pid(payload)
        owner_host = str((payload or {}).get("host", "")).strip()
        if owner_pid is not None and (not owner_host or owner_host == socket.gethostname()):
            try:
                os.kill(owner_pid, 0)
            except ProcessLookupError:
                stale_reason = "owner_dead"
            except PermissionError:
                stale_reason = None

    if stale_reason is None:
        return False

    try:
        path.unlink()
    except FileNotFoundError:
        return True
    except OSError:
        logger.debug("Failed to remove stale boss issue claim %s", path, exc_info=True)
        return False

    logger.info("boss_loop_reaped_issue_claim issue=#%s reason=%s", issue_number, stale_reason)
    return True


def has_active_foreign_claim(issue_number: int, run_id: str) -> bool:
    path = issue_claim_path(issue_number)
    if not path.exists():
        return False
    payload = read_issue_claim_payload(path)
    if claim_owned_by(payload, run_id):
        return False
    if reap_stale_claim(issue_number, path, payload, run_id):
        return False
    return True


def filter_claimed_issues(
    issues: list[Any],
    run_id: str,
) -> list[Any]:
    filtered: list[Any] = []
    for issue in issues:
        if has_active_foreign_claim(issue.number, run_id):
            logger.info("boss_loop_skip_claimed_issue issue=#%s", issue.number)
            continue
        filtered.append(issue)
    return filtered


def claim_issue(issue_number: int, run_id: str) -> tuple[bool, str | None]:
    path = issue_claim_path(issue_number)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "issue_number": int(issue_number),
        "run_id": run_id,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "claimed_at": datetime.now(UTC).isoformat(),
    }
    serialized = json.dumps(payload, sort_keys=True)

    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            existing = read_issue_claim_payload(path)
            if claim_owned_by(existing, run_id):
                return True, None
            if reap_stale_claim(issue_number, path, existing, run_id):
                continue
            owner_parts: list[str] = []
            if isinstance(existing, dict):
                existing_run_id = str(existing.get("run_id", "")).strip()
                if existing_run_id:
                    owner_parts.append(existing_run_id)
                existing_pid = issue_claim_owner_pid(existing)
                if existing_pid is not None:
                    owner_parts.append(f"pid {existing_pid}")
                existing_host = str(existing.get("host", "")).strip()
                if existing_host:
                    owner_parts.append(existing_host)
            owner_text = ", ".join(owner_parts) if owner_parts else "another boss loop"
            return False, f"Issue #{issue_number} is already claimed by {owner_text}."
        except OSError as exc:
            return False, f"Failed to claim issue #{issue_number}: {exc}"

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.write("\n")
        except OSError as exc:
            try:
                path.unlink()
            except OSError:
                logger.debug("Failed to clean up partial boss issue claim %s", path, exc_info=True)
            return False, f"Failed to persist issue claim for #{issue_number}: {exc}"
        return True, None


def release_claim(issue_number: int, run_id: str) -> None:
    path = issue_claim_path(issue_number)
    payload = read_issue_claim_payload(path)
    if payload is not None and not claim_owned_by(payload, run_id):
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        logger.debug("Failed to release boss issue claim %s", path, exc_info=True)
