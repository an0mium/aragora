"""GitHub issue feed and eligibility selection for the Boss loop.

Pulls candidate work from GitHub issues via the ``gh`` CLI, parses them into
structured ``GitHubIssue`` objects, and provides eligibility filtering with
optional value-per-cost ranking.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any

logger = logging.getLogger(__name__)
_SCOPE_ROOT_PREFIXES = (
    "aragora/",
    "tests/",
    "scripts/",
    "docs/",
    "docs-site/",
    "sdk/",
    "contracts/",
    ".github/",
)


@dataclass(slots=True)
class GitHubIssue:
    """Minimal representation of a GitHub issue suitable for task selection."""

    number: int
    title: str
    body: str
    labels: list[str]
    url: str
    state: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "title": self.title,
            "body": self.body,
            "labels": list(self.labels),
            "url": self.url,
            "state": self.state,
            "created_at": self.created_at,
        }


class GitHubIssueFeed:
    """Pull open issues from a GitHub repo via the ``gh`` CLI.

    Only issues in the ``open`` state are considered.  The feed is intentionally
    simple: it does not cache, does not paginate beyond the configured limit,
    and does not filter on anything except state and optional label match.
    """

    def __init__(
        self,
        *,
        repo: str | None = None,
        label_filter: str | None = None,
        issue_numbers: list[int] | None = None,
        limit: int = 25,
    ) -> None:
        self.repo = repo  # "owner/repo" or None for current repo
        self.label_filter = label_filter
        self.issue_numbers = [int(item) for item in issue_numbers or [] if int(item) > 0]
        self.limit = max(1, min(limit, 100))

    def fetch(self) -> list[GitHubIssue]:
        """Fetch open issues from GitHub. Returns empty list on failure."""
        if self.issue_numbers:
            issues: list[GitHubIssue] = []
            for number in self.issue_numbers:
                issue = self._fetch_issue(number)
                if issue is not None:
                    issues.append(issue)
            return issues

        cmd = [
            "gh",
            "issue",
            "list",
            "--state",
            "open",
            "--limit",
            str(self.limit),
            "--json",
            "number,title,body,labels,url,state,createdAt",
        ]
        if self.repo:
            cmd.extend(["--repo", self.repo])
        if self.label_filter:
            cmd.extend(["--label", self.label_filter])

        try:
            proc = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("gh issue list failed: %s", exc)
            return []

        if proc.returncode != 0:
            logger.warning("gh issue list returned %d: %s", proc.returncode, proc.stderr.strip())
            return []

        try:
            raw_issues = json.loads(proc.stdout)
        except (json.JSONDecodeError, ValueError):
            logger.warning("gh issue list produced invalid JSON")
            return []

        if not isinstance(raw_issues, list):
            return []

        issues: list[GitHubIssue] = []
        for item in raw_issues:
            if not isinstance(item, dict):
                continue
            labels_raw = item.get("labels") or []
            labels = [
                str(lbl.get("name", "") if isinstance(lbl, dict) else lbl).strip()
                for lbl in labels_raw
                if str(lbl.get("name", "") if isinstance(lbl, dict) else lbl).strip()
            ]
            issues.append(
                GitHubIssue(
                    number=int(item.get("number", 0)),
                    title=str(item.get("title", "")).strip(),
                    body=str(item.get("body", "")).strip(),
                    labels=labels,
                    url=str(item.get("url", "")).strip(),
                    state=str(item.get("state", "OPEN")).strip(),
                    created_at=str(item.get("createdAt", "")).strip(),
                )
            )
        return issues

    def _fetch_issue(self, number: int, *, allow_closed: bool = False) -> GitHubIssue | None:
        cmd = [
            "gh",
            "issue",
            "view",
            str(number),
            "--json",
            "number,title,body,labels,url,state,createdAt",
        ]
        if self.repo:
            cmd.extend(["--repo", self.repo])

        try:
            proc = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
            logger.warning("gh issue view failed: %s", exc)
            return None

        if proc.returncode != 0:
            logger.warning("gh issue view returned %d: %s", proc.returncode, proc.stderr.strip())
            return None

        try:
            item = json.loads(proc.stdout)
        except (json.JSONDecodeError, ValueError):
            logger.warning("gh issue view produced invalid JSON")
            return None

        if not isinstance(item, dict):
            return None

        state = str(item.get("state", "OPEN")).strip().lower()
        if not allow_closed and state != "open":
            return None

        labels_raw = item.get("labels") or []
        labels = [
            str(lbl.get("name", "") if isinstance(lbl, dict) else lbl).strip()
            for lbl in labels_raw
            if str(lbl.get("name", "") if isinstance(lbl, dict) else lbl).strip()
        ]
        if self.label_filter and self.label_filter not in labels:
            return None

        return GitHubIssue(
            number=int(item.get("number", number)),
            title=str(item.get("title", "")).strip(),
            body=str(item.get("body", "")).strip(),
            labels=labels,
            url=str(item.get("url", "")).strip(),
            state=state,
            created_at=str(item.get("createdAt", "")).strip(),
        )


def _normalize_scope_entry(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = text.replace("\\`", "`").strip("`").strip()
    text = text.strip("'\".,;:()[]{}<>").removeprefix("./")
    if not text:
        return None
    if text.endswith("/**"):
        base = text[:-3].rstrip("/")
        text = f"{base}/**" if base else ""
    else:
        text = text.rstrip("/")
    if not text or text.startswith(("http://", "https://")) or "/" not in text:
        return None
    if not any(text == prefix[:-1] or text.startswith(prefix) for prefix in _SCOPE_ROOT_PREFIXES):
        return None
    return text


def _normalize_scope_entries(values: list[str] | set[str] | tuple[str, ...]) -> list[str]:
    normalized: list[str] = []
    for item in values:
        value = _normalize_scope_entry(item)
        if value:
            normalized.append(value)
    return list(dict.fromkeys(normalized))


def _scope_entry_matches(scope_entry: str, path: str) -> bool:
    if "*" in scope_entry:
        return fnmatch(path, scope_entry)
    basename = scope_entry.rsplit("/", 1)[-1]
    if "." not in basename:
        return path == scope_entry or path.startswith(f"{scope_entry}/")
    return path == scope_entry


def scope_entries_overlap(left: str, right: str) -> bool:
    return _scope_entry_matches(left, right) or _scope_entry_matches(right, left)


def infer_issue_scope_entries(issue: GitHubIssue) -> list[str]:
    from aragora.swarm.spec import SwarmSpec

    combined = "\n".join(
        part for part in (str(issue.title or "").strip(), str(issue.body or "").strip()) if part
    )
    return _normalize_scope_entries(SwarmSpec.infer_file_scope_hints(combined))


def issue_overlaps_blocked_scopes(issue: GitHubIssue, blocked_scopes: set[str] | None) -> bool:
    if not blocked_scopes:
        return False
    issue_scopes = infer_issue_scope_entries(issue)
    if not issue_scopes:
        return False
    normalized_blocked = _normalize_scope_entries(blocked_scopes)
    return any(
        scope_entries_overlap(issue_scope, blocked_scope)
        for issue_scope in issue_scopes
        for blocked_scope in normalized_blocked
    )


def fetch_open_pr_changed_paths(*, repo: str | None = None, limit: int = 100) -> set[str]:
    cmd = [
        "gh",
        "pr",
        "list",
        "--state",
        "open",
        "--limit",
        str(max(1, min(int(limit), 100))),
        "--json",
        "files",
    ]
    if repo:
        cmd.extend(["--repo", repo])

    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("gh pr list failed: %s", exc)
        return set()

    if proc.returncode != 0:
        logger.warning("gh pr list returned %d: %s", proc.returncode, proc.stderr.strip())
        return set()

    try:
        raw_prs = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        logger.warning("gh pr list produced invalid JSON")
        return set()

    if not isinstance(raw_prs, list):
        return set()

    blocked: set[str] = set()
    for item in raw_prs:
        if not isinstance(item, dict):
            continue
        files = item.get("files") or []
        if not isinstance(files, list):
            continue
        for changed in files:
            if not isinstance(changed, dict):
                continue
            path = _normalize_scope_entry(changed.get("path"))
            if path:
                blocked.add(path)
    return blocked


def select_eligible_issue(
    issues: list[GitHubIssue],
    *,
    skip_labels: set[str] | None = None,
    require_labels: set[str] | None = None,
    blocked_scopes: set[str] | None = None,
    use_value_ranking: bool = False,
) -> GitHubIssue | None:
    """Select the best open issue that passes eligibility filters.

    Selection rules:
    - Must be in ``open`` state
    - Must have a non-empty title
    - Must not carry any label in ``skip_labels``
    - If ``require_labels`` is set, must carry ALL of them
    - Must not infer file scope that overlaps ``blocked_scopes``

    When ``use_value_ranking`` is True, eligible issues are scored by
    expected value-per-cost and the highest-scored issue is returned.
    Otherwise returns the first eligible issue (GitHub order).

    Returns ``None`` with no improvisation if nothing qualifies.
    """
    _skip = skip_labels or set()
    eligible: list[GitHubIssue] = []
    for issue in issues:
        if issue.state.upper() != "OPEN":
            continue
        if not issue.title:
            continue
        if _skip & set(issue.labels):
            continue
        if require_labels and not require_labels.issubset(set(issue.labels)):
            continue
        if issue_overlaps_blocked_scopes(issue, blocked_scopes):
            continue
        eligible.append(issue)

    if not eligible:
        return None

    if not use_value_ranking:
        return eligible[0]

    try:
        from aragora.swarm.value_estimator import (
            load_outcomes,
            log_prediction,
            rank_issues,
        )

        history = load_outcomes()
        issue_dicts = [i.to_dict() for i in eligible]
        ranked = rank_issues(issue_dicts, historical_outcomes=history)
        if ranked:
            best_estimate, best_dict = ranked[0]
            log_prediction(best_estimate)
            best_number = best_dict.get("number")
            logger.info(
                "value_ranking: #%s score=%.3f (value=%.2f p_success=%.2f proof=%.2f) — %s",
                best_number,
                best_estimate.priority_score,
                best_estimate.expected_value,
                best_estimate.p_success,
                best_estimate.proof_weight,
                best_estimate.reasoning[:80],
            )
            # Return the original GitHubIssue object
            for issue in eligible:
                if issue.number == best_number:
                    return issue
    except Exception as exc:
        logger.debug("Value ranking failed, falling back to first eligible: %s", exc)

    return eligible[0]
