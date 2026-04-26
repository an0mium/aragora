#!/usr/bin/env python3
"""Publish structured automation handoffs as GitHub issues.

Local Codex automations can verify bounded work while running in contexts where
GitHub writes are unreliable. This bridge runs from a normal shell, reads the
structured handoffs those automations leave in memory or in the local automation
outbox, deduplicates against existing GitHub issues, and creates the missing
issue records with ``gh``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.github_cli_health import check_github_cli_health

UTC = timezone.utc
DEFAULT_CODEX_HOME = Path("/Users/armand/.codex")
DEFAULT_REPO = "synaptent/aragora"
DEFAULT_LABELS = ("boss-ready",)
DEFAULT_LIMIT = 2
DEFAULT_MAX_OPEN_ISSUES = 12
DEFAULT_COMMAND_TIMEOUT_SECONDS = 45
MAX_ISSUE_BODY_CHARS = 60_000
DEFAULT_OUTBOX_DIR = Path(".aragora/automation-outbox")
DEFAULT_RECEIPT_DIR = Path(".aragora/automation-receipts")
DEFAULT_BASE_REF = "origin/main"
PR_OPEN_REQUEST_ACTIONS = {"open_pr", "push_branch_and_open_pr"}
REQUIRED_OUTBOX_KEYS = (
    "task",
    "requires_github",
    "requested_action",
    "repo",
    "local_evidence",
    "validation",
    "idempotency_key",
    "created_at",
)
DEFAULT_AUTOMATION_IDS = (
    "founder-review",
    "founder-triage",
    "engineering-automation-2",
    "aragora-overnight-steward",
)
REQUIRED_LABELS = (
    "Handoff Source",
    "Priority",
    "Task Title",
    "Why Now",
    "Repo Evidence",
    "Acceptance Criteria",
    "Validation",
    "Expiration Hours",
)
OPTIONAL_LABELS = ("Backup Task",)
ALL_LABELS = REQUIRED_LABELS + OPTIONAL_LABELS
BLOCK_TIMESTAMP_KEY = "__block_timestamp"
BLOCK_POSITION_KEY = "__block_position"
BLOCK_TIMESTAMP_PATTERN = re.compile(
    r"(?m)(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
)
PR_REFERENCE_PATTERN = re.compile(r"(?i)\b(?:PR|pull request)\s*#(\d+)\b")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

try:
    from aragora.swarm.github_app_auth import gh_subprocess_run, github_cli_env
except Exception:  # pragma: no cover - fallback for partially bootstrapped script contexts

    def github_cli_env(
        base_env: Mapping[str, str] | None = None,
        *,
        prefer_app: bool = True,
    ) -> dict[str, str]:
        return dict(os.environ if base_env is None else base_env)

    def gh_subprocess_run(
        args: Sequence[str],
        *,
        timeout: float = 30.0,
        prefer_app: bool = True,
        write_op: bool = False,
        env: Mapping[str, str] | None = None,
        max_retries: int = 0,
        base_backoff: float = 5.0,
        max_backoff: float = 600.0,
        sleep: Callable[[float], None] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del prefer_app, write_op, max_retries, base_backoff, max_backoff, sleep
        return subprocess.run(
            ["gh", *list(args)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=dict(os.environ if env is None else env),
            check=False,
        )


@dataclass(frozen=True)
class Handoff:
    source_file: str
    task_title: str
    priority: str
    body: str
    labels: dict[str, str]
    expires_at: str | None
    idempotency_key: str | None = None
    source_kind: str = "memory"


@dataclass(frozen=True)
class PublishDecision:
    task_title: str
    source_file: str
    eligible: bool
    reason: str
    existing_issue_url: str | None = None
    existing_pr_url: str | None = None
    created_issue_url: str | None = None


def _gh_write_op(args: list[str]) -> bool:
    return len(args) >= 2 and (args[0], args[1]) in {
        ("issue", "create"),
        ("issue", "edit"),
        ("issue", "comment"),
    }


def _run(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = github_cli_env(os.environ) if args and args[0] == "gh" else None
    if args and args[0] == "gh":
        return gh_subprocess_run(
            args[1:],
            timeout=DEFAULT_COMMAND_TIMEOUT_SECONDS,
            prefer_app=True,
            write_op=_gh_write_op(args[1:]),
            env=dict(os.environ if env is None else env),
            max_retries=0,
        )
    try:
        return subprocess.run(
            args,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=DEFAULT_COMMAND_TIMEOUT_SECONDS,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        message = (
            stderr
            or f"command timed out after {DEFAULT_COMMAND_TIMEOUT_SECONDS}s: {' '.join(args)}"
        )
        return subprocess.CompletedProcess(args=args, returncode=124, stdout=stdout, stderr=message)


def _codex_home(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    env_value = os.environ.get("CODEX_HOME")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return DEFAULT_CODEX_HOME


def _repo_root(path: Path) -> Path:
    proc = _run(["git", "rev-parse", "--show-toplevel"], cwd=path)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "not a git repository")
    return Path(proc.stdout.strip()).resolve()


def _repo_relative(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def _same_git_origin(left: Path, right: Path) -> bool:
    left_proc = _run(["git", "config", "--get", "remote.origin.url"], cwd=left)
    right_proc = _run(["git", "config", "--get", "remote.origin.url"], cwd=right)
    if left_proc.returncode != 0 or right_proc.returncode != 0:
        return False
    return bool(left_proc.stdout.strip()) and left_proc.stdout.strip() == right_proc.stdout.strip()


def _automation_state_root(root: Path) -> Path:
    """Return the checkout whose shared .aragora state should back handoff publishing."""

    if (root / ".aragora").is_dir():
        return root

    configured = os.environ.get("ARAGORA_AUTOMATION_STATE_ROOT")
    candidates: list[tuple[Path, bool]] = []
    if configured:
        candidates.append((Path(configured).expanduser(), True))
    candidates.append((Path.home() / "Development" / "aragora", False))

    for candidate, explicit in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if not (resolved / ".aragora").is_dir():
            continue
        if explicit or _same_git_origin(root, resolved):
            return resolved
    return root


def _automation_state_path(root: Path, path: Path | None, default_relative: Path) -> Path:
    if path is not None:
        return _repo_relative(root, path)
    return _automation_state_root(root) / default_relative


def _memory_files(codex_home: Path, automation_ids: set[str] | None = None) -> list[Path]:
    automations = codex_home / "automations"
    if not automations.exists():
        return []
    if automation_ids is None:
        return sorted(automations.glob("*/memory.md"))
    return sorted(
        path
        for automation_id in automation_ids
        if (path := automations / automation_id / "memory.md").exists()
    )


def _outbox_files(outbox_dir: Path) -> list[Path]:
    if not outbox_dir.exists():
        return []
    return sorted(path for path in outbox_dir.glob("*.json") if path.is_file())


def _source_mtime(source_file: str) -> float:
    try:
        return Path(source_file).stat().st_mtime
    except OSError:
        return 0.0


def _receipt_path(receipt_dir: Path, idempotency_key: str) -> Path:
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "-", idempotency_key).strip("-")
    if not safe_key:
        safe_key = "handoff"
    return receipt_dir / f"{safe_key}.json"


def _terminal_receipt_exists(receipt_dir: Path, idempotency_key: str) -> bool:
    path = _receipt_path(receipt_dir, idempotency_key)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return str(payload.get("status") or "") in {"published", "already_satisfied"}


def _terminal_receipt_keys(receipt_dir: Path) -> set[str]:
    if not receipt_dir.exists():
        return set()
    terminal: set[str] = set()
    for receipt_file in sorted(receipt_dir.glob("*.json")):
        try:
            payload = json.loads(receipt_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if str(payload.get("status") or "") not in {"published", "already_satisfied"}:
            continue
        idempotency_key = str(payload.get("idempotency_key") or "").strip()
        if idempotency_key:
            terminal.add(idempotency_key)
    return terminal


def _write_receipt(
    receipt_dir: Path,
    handoff: Handoff,
    decision: PublishDecision,
    *,
    repo: str,
) -> None:
    if handoff.source_kind != "outbox" or not handoff.idempotency_key:
        return
    terminal_reasons = {"published", "existing_issue", "existing_pr", "target_open_pr"}
    if decision.reason not in terminal_reasons:
        return
    status = "published" if decision.reason == "published" else "already_satisfied"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "idempotency_key": handoff.idempotency_key,
        "task": handoff.task_title,
        "source_file": handoff.source_file,
        "repo": repo,
        "status": status,
        "reason": decision.reason,
        "created_issue_url": decision.created_issue_url,
        "existing_issue_url": decision.existing_issue_url,
        "existing_pr_url": decision.existing_pr_url,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    _receipt_path(receipt_dir, handoff.idempotency_key).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _label_matches(text: str) -> list[re.Match[str]]:
    label_pattern = "|".join(re.escape(label) for label in ALL_LABELS)
    return list(re.finditer(rf"(?m)^({label_pattern}):\s*", text))


def _timestamp_before(text: str, position: int) -> str | None:
    """Return the closest preceding full ISO timestamp for a memory handoff block."""

    matches = list(BLOCK_TIMESTAMP_PATTERN.finditer(text[:position]))
    if not matches:
        return None
    return matches[-1].group(1)


def _parse_block_datetime(values: dict[str, str]) -> datetime | None:
    raw_timestamp = values.get(BLOCK_TIMESTAMP_KEY)
    if not raw_timestamp:
        return None
    normalized = raw_timestamp.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_blocks(text: str) -> list[dict[str, str]]:
    matches = _label_matches(text)
    blocks: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        if match.group(1) != "Handoff Source":
            continue

        block_matches = [match]
        next_index = index + 1
        while next_index < len(matches) and matches[next_index].group(1) != "Handoff Source":
            block_matches.append(matches[next_index])
            next_index += 1

        values: dict[str, str] = {}
        for item_index, item in enumerate(block_matches):
            next_start = (
                block_matches[item_index + 1].start()
                if item_index + 1 < len(block_matches)
                else len(text)
            )
            values[item.group(1)] = text[item.end() : next_start].strip()

        if all(label in values and values[label] for label in REQUIRED_LABELS):
            values[BLOCK_TIMESTAMP_KEY] = _timestamp_before(text, match.start()) or ""
            values[BLOCK_POSITION_KEY] = str(match.start())
            blocks.append(values)
    return blocks


def _expiration(values: dict[str, str], source_file: Path) -> str | None:
    raw_hours = values.get("Expiration Hours", "").strip()
    try:
        hours = float(raw_hours)
    except ValueError:
        return None
    if hours <= 0:
        return None
    reference_time = _parse_block_datetime(values) or datetime.fromtimestamp(
        source_file.stat().st_mtime, tz=UTC
    )
    expires_at = reference_time + timedelta(hours=hours)
    return expires_at.isoformat()


def _is_expired(expires_at: str | None, *, now: datetime) -> bool:
    if not expires_at:
        return False
    try:
        parsed = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError:
        return True
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC) < now


def _format_body(values: dict[str, str], source_file: Path) -> str:
    lines: list[str] = []
    for label in ALL_LABELS:
        value = values.get(label)
        if value:
            lines.append(f"{label}: {value}")
            lines.append("")
    lines.append("---")
    lines.append(f"Published from automation memory: `{source_file}`")
    return "\n".join(lines).strip()


def _format_json_block(value: Any) -> str:
    if value is None or value == "":
        return "NONE"
    if isinstance(value, str):
        return value.strip() or "NONE"
    return json.dumps(value, indent=2, sort_keys=True)


def _format_outbox_body(payload: dict[str, Any], source_file: Path) -> str:
    fields = [
        ("Task", payload.get("task")),
        ("Requested Action", payload.get("requested_action")),
        ("Requires GitHub", payload.get("requires_github")),
        ("Repo", payload.get("repo")),
        ("Created At", payload.get("created_at")),
        ("Idempotency Key", payload.get("idempotency_key")),
        ("Local Evidence", payload.get("local_evidence")),
        ("Validation", payload.get("validation")),
    ]
    lines: list[str] = []
    for label, value in fields:
        formatted = _format_json_block(value)
        lines.append(f"{label}:")
        if "\n" in formatted or formatted.startswith("{") or formatted.startswith("["):
            lines.append("```json" if formatted.startswith(("{", "[")) else "```")
            lines.append(formatted)
            lines.append("```")
        else:
            lines.append(formatted)
        lines.append("")
    lines.append("---")
    lines.append(f"Published from automation outbox: `{source_file}`")
    return "\n".join(lines).strip()


def _has_required_outbox_contract(payload: dict[str, Any]) -> bool:
    for key in REQUIRED_OUTBOX_KEYS:
        if key not in payload:
            return False
        value = payload[key]
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
    return True


def _outbox_evidence_value(payload: dict[str, Any], key: str) -> str:
    local_evidence = payload.get("local_evidence")
    if isinstance(local_evidence, dict):
        value = str(local_evidence.get(key) or "").strip()
        if value:
            return value
    return str(payload.get(key) or "").strip()


def _outbox_branch_fingerprint(payload: dict[str, Any]) -> str | None:
    requested_action = str(payload.get("requested_action") or "").strip()
    repo = str(payload.get("repo") or "").strip()
    branch = _outbox_evidence_value(payload, "branch")
    if not requested_action or not repo or not branch:
        return None
    return "\0".join((requested_action, repo, branch))


def _git_is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    proc = _run(["git", "merge-base", "--is-ancestor", ancestor, descendant], cwd=repo_root)
    return proc.returncode == 0


def _git_patch_equivalent(repo_root: Path, base: str, candidate: str) -> bool:
    diff_proc = _run(["git", "diff", "--quiet", f"{base}...{candidate}"], cwd=repo_root)
    if diff_proc.returncode == 0:
        return True
    if diff_proc.returncode != 1:
        return False

    proc = _run(["git", "cherry", base, candidate], cwd=repo_root)
    if proc.returncode != 0:
        return False
    statuses = [line[:1] for line in proc.stdout.splitlines() if line.strip()]
    return bool(statuses) and all(status == "-" for status in statuses)


def _outbox_branch_already_merged(repo_root: Path, payload: dict[str, Any]) -> bool:
    requested_action = str(payload.get("requested_action") or "").strip()
    if requested_action not in PR_OPEN_REQUEST_ACTIONS:
        return False

    base_ref = _outbox_evidence_value(payload, "base") or DEFAULT_BASE_REF
    if not base_ref:
        return False

    candidates = [
        _outbox_evidence_value(payload, "head"),
        _outbox_evidence_value(payload, "head_sha"),
        _outbox_evidence_value(payload, "commit"),
        _outbox_evidence_value(payload, "branch"),
    ]
    for candidate in dict.fromkeys(item for item in candidates if item):
        if _git_is_ancestor(repo_root, candidate, base_ref):
            return True
    return False


def _outbox_branch_patch_equivalent(repo_root: Path, payload: dict[str, Any]) -> bool:
    requested_action = str(payload.get("requested_action") or "").strip()
    if requested_action not in PR_OPEN_REQUEST_ACTIONS:
        return False

    base_ref = _outbox_evidence_value(payload, "base") or DEFAULT_BASE_REF
    if not base_ref:
        return False

    candidates = [
        _outbox_evidence_value(payload, "branch"),
        _outbox_evidence_value(payload, "head"),
        _outbox_evidence_value(payload, "head_sha"),
        _outbox_evidence_value(payload, "commit"),
    ]
    for candidate in dict.fromkeys(item for item in candidates if item):
        if _git_patch_equivalent(repo_root, base_ref, candidate):
            return True
    return False


def _terminal_outbox_fingerprints(outbox_dir: Path, receipt_dir: Path) -> set[str]:
    terminal_keys = _terminal_receipt_keys(receipt_dir)
    if not terminal_keys:
        return set()
    fingerprints: set[str] = set()
    for source_file in _outbox_files(outbox_dir):
        try:
            payload = json.loads(source_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not _has_required_outbox_contract(payload):
            continue
        idempotency_key = str(payload.get("idempotency_key") or "").strip()
        if idempotency_key not in terminal_keys:
            continue
        fingerprint = _outbox_branch_fingerprint(payload)
        if fingerprint:
            fingerprints.add(fingerprint)
    return fingerprints


def _latest_block(parsed_blocks: list[dict[str, str]]) -> dict[str, str]:
    def key(values: dict[str, str]) -> tuple[datetime, int]:
        block_time = _parse_block_datetime(values) or datetime.min.replace(tzinfo=UTC)
        try:
            position = int(values.get(BLOCK_POSITION_KEY, "0"))
        except ValueError:
            position = 0
        return (block_time, position)

    return max(parsed_blocks, key=key)


def load_handoffs(
    codex_home: Path,
    *,
    automation_ids: set[str] | None = None,
    now: datetime | None = None,
) -> list[Handoff]:
    current_time = now or datetime.now(UTC)
    handoffs: list[Handoff] = []
    for memory_file in _memory_files(codex_home, automation_ids):
        try:
            text = memory_file.read_text(encoding="utf-8")
        except OSError:
            continue
        parsed_blocks = _parse_blocks(text)
        if not parsed_blocks:
            continue
        values = _latest_block(parsed_blocks)
        priority = values["Priority"].strip()
        task_title = values["Task Title"].strip()
        expires_at = _expiration(values, memory_file)
        if priority.upper() == "NONE" or task_title.upper() == "NONE":
            continue
        if _is_expired(expires_at, now=current_time):
            continue
        handoffs.append(
            Handoff(
                source_file=str(memory_file),
                task_title=task_title,
                priority=priority,
                body=_format_body(values, memory_file),
                labels=values,
                expires_at=expires_at,
            )
        )

    return sorted(
        handoffs,
        key=lambda item: (_source_mtime(item.source_file), item.priority),
        reverse=True,
    )


def load_outbox_handoffs(
    repo_root: Path,
    *,
    outbox_dir: Path | None = None,
    receipt_dir: Path | None = None,
    now: datetime | None = None,
) -> list[Handoff]:
    outbox_root = _automation_state_path(repo_root, outbox_dir, DEFAULT_OUTBOX_DIR).resolve()
    receipt_root = _automation_state_path(repo_root, receipt_dir, DEFAULT_RECEIPT_DIR).resolve()
    current_time = now or datetime.now(UTC)
    terminal_keys = _terminal_receipt_keys(receipt_root)
    terminal_fingerprints = _terminal_outbox_fingerprints(outbox_root, receipt_root)
    handoffs_by_identity: dict[tuple[str, str], Handoff] = {}
    for source_file in _outbox_files(outbox_root):
        try:
            payload = json.loads(source_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if not _has_required_outbox_contract(payload):
            continue
        task_title = str(payload.get("task") or payload.get("title") or "").strip()
        requested_action = str(payload.get("requested_action") or "").strip()
        idempotency_key = str(payload.get("idempotency_key") or "").strip()
        requires_github = payload.get("requires_github", True)
        if isinstance(requires_github, str):
            requires_github = requires_github.strip().lower() not in {"0", "false", "no"}
        if requires_github is False:
            continue
        if not task_title or not requested_action or not idempotency_key:
            continue
        expires_at = str(payload.get("expires_at") or "").strip() or None
        if _is_expired(expires_at, now=current_time):
            continue
        branch_fingerprint = _outbox_branch_fingerprint(payload)
        if idempotency_key in terminal_keys:
            continue
        if branch_fingerprint and branch_fingerprint in terminal_fingerprints:
            continue
        if _outbox_branch_already_merged(repo_root, payload):
            continue
        if _outbox_branch_patch_equivalent(repo_root, payload):
            continue
        handoff = Handoff(
            source_file=str(source_file),
            task_title=task_title,
            priority=str(payload.get("priority") or "MEDIUM").strip() or "MEDIUM",
            body=_format_outbox_body(payload, source_file),
            labels={key: _format_json_block(value) for key, value in payload.items()},
            expires_at=expires_at,
            idempotency_key=idempotency_key,
            source_kind="outbox",
        )
        identity = (
            ("branch", branch_fingerprint)
            if branch_fingerprint
            else ("idempotency", idempotency_key)
        )
        existing = handoffs_by_identity.get(identity)
        if existing is None or (
            _source_mtime(handoff.source_file),
            handoff.source_file,
        ) > (
            _source_mtime(existing.source_file),
            existing.source_file,
        ):
            handoffs_by_identity[identity] = handoff
    return sorted(
        handoffs_by_identity.values(),
        key=lambda item: (_source_mtime(item.source_file), item.priority),
        reverse=True,
    )


def _ensure_gh_auth(repo_root: Path) -> None:
    health = check_github_cli_health(repo_root)
    if not health.ready:
        raise RuntimeError(health.error or health.mode)


def _title_tokens(title: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", title.lower())
        if len(token) >= 3 and token not in STOPWORDS
    }


def _looks_duplicate(candidate: str, existing: str) -> bool:
    candidate_tokens = _title_tokens(candidate)
    existing_tokens = _title_tokens(existing)
    if not candidate_tokens or not existing_tokens:
        return candidate.strip().lower() == existing.strip().lower()
    candidate_handlers = {token for token in candidate_tokens if token.endswith("handler")}
    existing_handlers = {token for token in existing_tokens if token.endswith("handler")}
    if (
        candidate_handlers
        and existing_handlers
        and candidate_handlers.isdisjoint(existing_handlers)
    ):
        return False
    overlap = candidate_tokens & existing_tokens
    return len(overlap) >= min(3, len(candidate_tokens))


def _existing_issue(repo_root: Path, repo: str, title: str) -> dict[str, Any] | None:
    proc = _run(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "all",
            "--search",
            title,
            "--json",
            "number,title,url,state",
            "--limit",
            "50",
        ],
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "failed to list issues")
    payload = json.loads(proc.stdout or "[]")
    if not isinstance(payload, list):
        return None
    for item in payload:
        if not isinstance(item, dict):
            continue
        existing_title = str(item.get("title") or "")
        if _looks_duplicate(title, existing_title):
            return item
    return None


def _existing_pr(repo_root: Path, repo: str, title: str) -> dict[str, Any] | None:
    proc = _run(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "all",
            "--search",
            title,
            "--json",
            "number,title,url,state",
            "--limit",
            "50",
        ],
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "failed to list PRs")
    payload = json.loads(proc.stdout or "[]")
    if not isinstance(payload, list):
        return None
    for item in payload:
        if not isinstance(item, dict):
            continue
        existing_title = str(item.get("title") or "")
        if _looks_duplicate(title, existing_title):
            return item
    return None


def _referenced_pr_numbers(handoff: Handoff) -> list[int]:
    seen: set[int] = set()
    numbers: list[int] = []
    for match in PR_REFERENCE_PATTERN.finditer(f"{handoff.task_title}\n{handoff.body}"):
        try:
            number = int(match.group(1))
        except (TypeError, ValueError):
            continue
        if number in seen:
            continue
        seen.add(number)
        numbers.append(number)
    return numbers


def _pr_by_number(repo_root: Path, repo: str, number: int) -> dict[str, Any] | None:
    proc = _run(
        [
            "gh",
            "pr",
            "view",
            str(number),
            "--repo",
            repo,
            "--json",
            "number,title,url,state",
        ],
        cwd=repo_root,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or proc.stdout or "").lower()
        if "could not resolve to a pull request" in stderr or "not found" in stderr:
            return None
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "failed to view PR")
    payload = json.loads(proc.stdout or "{}")
    return payload if isinstance(payload, dict) else None


def _target_open_pr(repo_root: Path, repo: str, handoff: Handoff) -> dict[str, Any] | None:
    for number in _referenced_pr_numbers(handoff):
        pr = _pr_by_number(repo_root, repo, number)
        if isinstance(pr, dict) and str(pr.get("state") or "").upper() == "OPEN":
            return pr
    return None


def _open_boss_ready_count(repo_root: Path, repo: str, labels: list[str]) -> int:
    args = [
        "gh",
        "issue",
        "list",
        "--repo",
        repo,
        "--state",
        "open",
        "--json",
        "number",
        "--limit",
        "200",
    ]
    for label in labels:
        args.extend(["--label", label])
    proc = _run(args, cwd=repo_root)
    if proc.returncode != 0:
        raise RuntimeError(
            proc.stderr.strip() or proc.stdout.strip() or "failed to count open issues"
        )
    payload = json.loads(proc.stdout or "[]")
    return len(payload) if isinstance(payload, list) else 0


def _create_issue(
    repo_root: Path,
    repo: str,
    handoff: Handoff,
    *,
    labels: list[str],
) -> str:
    args = [
        "gh",
        "issue",
        "create",
        "--repo",
        repo,
        "--title",
        handoff.task_title,
        "--body",
        _fit_issue_body(handoff.body),
    ]
    for label in labels:
        args.extend(["--label", label])
    proc = _run(args, cwd=repo_root)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "gh issue create failed")
    url = str(proc.stdout or "").strip().splitlines()[-1].strip()
    _add_issue_labels(repo_root, repo, url, labels)
    return url


def _fit_issue_body(body: str) -> str:
    if len(body) <= MAX_ISSUE_BODY_CHARS:
        return body
    suffix = (
        "\n\n---\n"
        "Automation publisher truncated this issue body because it exceeded "
        "GitHub's size limit. See the source automation memory path above for full evidence."
    )
    return body[: MAX_ISSUE_BODY_CHARS - len(suffix)].rstrip() + suffix


def _issue_number_from_url(url: str) -> str | None:
    match = re.search(r"/issues/(\d+)(?:$|[/?#])", url)
    return match.group(1) if match else None


def _add_issue_labels(repo_root: Path, repo: str, issue_url: str, labels: list[str]) -> None:
    if not labels:
        return
    number = _issue_number_from_url(issue_url)
    if not number:
        return
    proc = _run(
        ["gh", "issue", "edit", number, "--repo", repo, "--add-label", ",".join(labels)],
        cwd=repo_root,
    )
    if proc.returncode != 0:
        return


def decide_handoffs(
    handoffs: list[Handoff],
    *,
    repo_root: Path,
    repo: str,
    labels: list[str],
    max_open_issues: int,
) -> list[PublishDecision]:
    open_issue_count = _open_boss_ready_count(repo_root, repo, labels)
    decisions: list[PublishDecision] = []
    for handoff in handoffs:
        existing = _existing_issue(repo_root, repo, handoff.task_title)
        if existing:
            decisions.append(
                PublishDecision(
                    task_title=handoff.task_title,
                    source_file=handoff.source_file,
                    eligible=False,
                    reason="existing_issue",
                    existing_issue_url=str(existing.get("url") or ""),
                )
            )
            continue
        target_pr = _target_open_pr(repo_root, repo, handoff)
        if target_pr:
            decisions.append(
                PublishDecision(
                    task_title=handoff.task_title,
                    source_file=handoff.source_file,
                    eligible=False,
                    reason="target_open_pr",
                    existing_pr_url=str(target_pr.get("url") or ""),
                )
            )
            continue
        existing_pr = _existing_pr(repo_root, repo, handoff.task_title)
        if existing_pr:
            decisions.append(
                PublishDecision(
                    task_title=handoff.task_title,
                    source_file=handoff.source_file,
                    eligible=False,
                    reason="existing_pr",
                    existing_pr_url=str(existing_pr.get("url") or ""),
                )
            )
            continue
        if open_issue_count >= max_open_issues:
            decisions.append(
                PublishDecision(
                    task_title=handoff.task_title,
                    source_file=handoff.source_file,
                    eligible=False,
                    reason="open_issue_cap",
                )
            )
            continue
        decisions.append(
            PublishDecision(
                task_title=handoff.task_title,
                source_file=handoff.source_file,
                eligible=True,
                reason="eligible",
            )
        )
    return decisions


def publish_handoffs(
    handoffs: list[Handoff],
    decisions: list[PublishDecision],
    *,
    repo_root: Path,
    repo: str,
    labels: list[str],
    limit: int,
    receipt_dir: Path | None = None,
) -> list[PublishDecision]:
    by_key = {(item.task_title, item.source_file): item for item in handoffs}
    published: list[PublishDecision] = []
    count = 0
    for decision in decisions:
        if not decision.eligible:
            published.append(decision)
            continue
        if count >= limit:
            published.append(
                PublishDecision(
                    task_title=decision.task_title,
                    source_file=decision.source_file,
                    eligible=False,
                    reason="publish_limit",
                )
            )
            continue
        handoff = by_key[(decision.task_title, decision.source_file)]
        url = _create_issue(repo_root, repo, handoff, labels=labels)
        count += 1
        published_decision = PublishDecision(
            task_title=decision.task_title,
            source_file=decision.source_file,
            eligible=False,
            reason="published",
            created_issue_url=url,
        )
        if receipt_dir is not None:
            _write_receipt(receipt_dir, handoff, published_decision, repo=repo)
        published.append(published_decision)
    return published


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish structured automation memory handoffs as GitHub issues."
    )
    parser.add_argument("--repo", default=".", help="Path inside the target repository")
    parser.add_argument(
        "--github-repo",
        default=DEFAULT_REPO,
        help="GitHub repository slug for gh issue operations",
    )
    parser.add_argument(
        "--codex-home",
        default=None,
        help="Codex home containing automations; defaults to $CODEX_HOME or /Users/armand/.codex",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum number of issues to create in one apply run",
    )
    parser.add_argument(
        "--max-open-issues",
        type=int,
        default=DEFAULT_MAX_OPEN_ISSUES,
        help="Maximum open issues with the selected labels before publishing pauses",
    )
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        default=list(DEFAULT_LABELS),
        help="Issue label to add; may be passed multiple times",
    )
    parser.add_argument(
        "--automation-id",
        action="append",
        dest="automation_ids",
        default=[],
        help=(
            "Automation memory id to scan. Defaults to scout/support automations: "
            + ", ".join(DEFAULT_AUTOMATION_IDS)
        ),
    )
    parser.add_argument(
        "--outbox-dir",
        default=None,
        help=(
            "Directory containing JSON automation outbox handoffs. Defaults to "
            ".aragora/automation-outbox under the shared automation state root."
        ),
    )
    parser.add_argument(
        "--receipt-dir",
        default=None,
        help=(
            "Directory for JSON automation publish receipts. Defaults to "
            ".aragora/automation-receipts under the shared automation state root."
        ),
    )
    parser.add_argument(
        "--no-outbox",
        action="store_true",
        help="Disable loading JSON automation outbox handoffs.",
    )
    parser.add_argument("--apply", action="store_true", help="Create eligible GitHub issues")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    repo_root = _repo_root(Path(args.repo))
    codex_home = _codex_home(args.codex_home)
    outbox_dir = _automation_state_path(
        repo_root,
        Path(args.outbox_dir).expanduser() if args.outbox_dir else None,
        DEFAULT_OUTBOX_DIR,
    ).resolve()
    receipt_dir = _automation_state_path(
        repo_root,
        Path(args.receipt_dir).expanduser() if args.receipt_dir else None,
        DEFAULT_RECEIPT_DIR,
    ).resolve()
    labels = list(dict.fromkeys(args.labels))
    automation_ids = set(args.automation_ids or DEFAULT_AUTOMATION_IDS)
    memory_handoffs = load_handoffs(codex_home, automation_ids=automation_ids)
    outbox_handoffs = (
        []
        if args.no_outbox
        else load_outbox_handoffs(repo_root, outbox_dir=outbox_dir, receipt_dir=receipt_dir)
    )
    handoffs = sorted(
        memory_handoffs + outbox_handoffs,
        key=lambda item: (_source_mtime(item.source_file), item.priority),
        reverse=True,
    )
    github_health = check_github_cli_health(repo_root)
    if not github_health.ready:
        payload = {
            "repo": str(repo_root),
            "codex_home": str(codex_home),
            "github_repo": args.github_repo,
            "labels": labels,
            "automation_ids": sorted(automation_ids),
            "outbox_dir": str(outbox_dir),
            "receipt_dir": str(receipt_dir),
            "memory_handoff_count": len(memory_handoffs),
            "outbox_handoff_count": len(outbox_handoffs),
            "handoff_count": len(handoffs),
            "github_health": github_health.to_dict(),
            "decisions": [
                asdict(
                    PublishDecision(
                        task_title=handoff.task_title,
                        source_file=handoff.source_file,
                        eligible=False,
                        reason="github_unavailable",
                    )
                )
                for handoff in handoffs
            ],
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"github_unavailable: {github_health.mode} {github_health.error}".strip())
        return 1

    decisions = decide_handoffs(
        handoffs,
        repo_root=repo_root,
        repo=args.github_repo,
        labels=labels,
        max_open_issues=args.max_open_issues,
    )
    results = (
        publish_handoffs(
            handoffs,
            decisions,
            repo_root=repo_root,
            repo=args.github_repo,
            labels=labels,
            limit=args.limit,
            receipt_dir=receipt_dir,
        )
        if args.apply
        else decisions
    )
    by_key = {(item.task_title, item.source_file): item for item in handoffs}
    if args.apply:
        for item in results:
            handoff = by_key.get((item.task_title, item.source_file))
            if handoff is not None:
                _write_receipt(receipt_dir, handoff, item, repo=args.github_repo)

    payload = {
        "repo": str(repo_root),
        "codex_home": str(codex_home),
        "github_repo": args.github_repo,
        "labels": labels,
        "automation_ids": sorted(automation_ids),
        "outbox_dir": str(outbox_dir),
        "receipt_dir": str(receipt_dir),
        "memory_handoff_count": len(memory_handoffs),
        "outbox_handoff_count": len(outbox_handoffs),
        "handoff_count": len(handoffs),
        "github_health": github_health.to_dict(),
        "decisions": [asdict(item) for item in results],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for item in results:
            marker = (
                "issue"
                if item.reason == "published"
                else "skip"
                if not item.eligible
                else "publish"
            )
            target = item.created_issue_url or item.existing_issue_url or item.existing_pr_url or ""
            print(f"{marker}: {item.task_title} [{item.reason}] {target}".strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
