#!/usr/bin/env python3
"""Publish structured automation handoffs as GitHub issues.

Local Codex automations can verify bounded work while running in contexts where
GitHub writes are unreliable. This bridge runs from a normal shell, reads the
structured handoffs those automations leave in memory, deduplicates against
existing GitHub issues, and creates the missing issue records with ``gh``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc
DEFAULT_CODEX_HOME = Path("/Users/armand/.codex")
DEFAULT_REPO = "synaptent/aragora"
DEFAULT_LABELS = ("boss-ready",)
DEFAULT_LIMIT = 2
DEFAULT_MAX_OPEN_ISSUES = 12
DEFAULT_COMMAND_TIMEOUT_SECONDS = 45
MAX_ISSUE_BODY_CHARS = 60_000
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
    from aragora.swarm.github_app_auth import github_cli_env
except Exception:  # pragma: no cover - fallback for partially bootstrapped script contexts

    def github_cli_env(
        base_env: dict[str, str] | None = None,
        *,
        prefer_app: bool = True,
    ) -> dict[str, str]:
        return dict(os.environ if base_env is None else base_env)


@dataclass(frozen=True)
class Handoff:
    source_file: str
    task_title: str
    priority: str
    body: str
    labels: dict[str, str]
    expires_at: str | None


@dataclass(frozen=True)
class PublishDecision:
    task_title: str
    source_file: str
    eligible: bool
    reason: str
    existing_issue_url: str | None = None
    existing_pr_url: str | None = None
    created_issue_url: str | None = None


def _run(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = github_cli_env(os.environ) if args and args[0] == "gh" else None
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
    return datetime.fromisoformat(expires_at) < now


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
        key=lambda item: (Path(item.source_file).stat().st_mtime, item.priority),
        reverse=True,
    )


def _ensure_gh_auth(repo_root: Path) -> None:
    proc = _run(["gh", "auth", "status"], cwd=repo_root)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "gh auth failed")


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
        published.append(
            PublishDecision(
                task_title=decision.task_title,
                source_file=decision.source_file,
                eligible=False,
                reason="published",
                created_issue_url=url,
            )
        )
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
    parser.add_argument("--apply", action="store_true", help="Create eligible GitHub issues")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    repo_root = _repo_root(Path(args.repo))
    codex_home = _codex_home(args.codex_home)
    labels = list(dict.fromkeys(args.labels))
    automation_ids = set(args.automation_ids or DEFAULT_AUTOMATION_IDS)

    _ensure_gh_auth(repo_root)
    handoffs = load_handoffs(codex_home, automation_ids=automation_ids)
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
        )
        if args.apply
        else decisions
    )

    payload = {
        "repo": str(repo_root),
        "codex_home": str(codex_home),
        "github_repo": args.github_repo,
        "labels": labels,
        "automation_ids": sorted(automation_ids),
        "handoff_count": len(handoffs),
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
