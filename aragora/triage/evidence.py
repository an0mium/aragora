"""Evidence gathering for issue triage.

Per the calibration-only v1 contract, evidence is collected from GitHub +
repo state BEFORE any model is invoked. The panel judges against this
evidence, not just the title.

Evidence includes:
- Raw issue body, labels, author, created/updated timestamps
- Referenced file paths and whether they exist in HEAD
- Related PR/issue numbers found in the body, with their state
- Duplicate-candidate suggestions (title similarity over open issues)

This module is intentionally side-effect free: it reads from the local
repo and from the GitHub CLI / a callable injected at test time.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

ISSUE_REF_PATTERN = re.compile(r"#(\d{2,6})\b")
FILE_PATH_PATTERN = re.compile(
    r"(?:^|[\s`(])([a-zA-Z0-9_./\\-]+\.(?:py|md|yml|yaml|toml|cfg|sh|ts|tsx|js|jsx|json))(?=[\s`):,.]|$)"
)
AUTOMATION_AUTHOR_HINTS = (
    "[bot]",
    "an0mium",
    "boss-loop",
    "stage-gate",
    "swarm-",
    "factory-droid",
    "codex-automation",
    "github-actions",
)


@dataclass(frozen=True)
class IssueRecord:
    """Minimal projection of a GitHub issue for triage."""

    number: int
    title: str
    body: str
    author: str
    labels: tuple[str, ...]
    state: str
    url: str
    created_at: str
    updated_at: str
    comments_count: int = 0
    assignees: tuple[str, ...] = ()


@dataclass
class IssueEvidence:
    """Pre-model evidence assembled for an issue."""

    issue: IssueRecord
    is_automation_generated: bool
    referenced_files: list[dict[str, Any]] = field(default_factory=list)
    referenced_issues: list[dict[str, Any]] = field(default_factory=list)
    duplicate_candidates: list[dict[str, Any]] = field(default_factory=list)
    repo_head_sha: str | None = None
    gathered_at: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue": {
                "number": self.issue.number,
                "title": self.issue.title,
                "body": self.issue.body,
                "author": self.issue.author,
                "labels": list(self.issue.labels),
                "state": self.issue.state,
                "url": self.issue.url,
                "created_at": self.issue.created_at,
                "updated_at": self.issue.updated_at,
                "comments_count": self.issue.comments_count,
                "assignees": list(self.issue.assignees),
            },
            "is_automation_generated": self.is_automation_generated,
            "referenced_files": list(self.referenced_files),
            "referenced_issues": list(self.referenced_issues),
            "duplicate_candidates": list(self.duplicate_candidates),
            "repo_head_sha": self.repo_head_sha,
            "gathered_at": self.gathered_at,
            "notes": list(self.notes),
        }


def is_automation_generated(
    *,
    author: str,
    labels: Sequence[str] = (),
    body: str = "",
) -> bool:
    """Heuristic detector for automation-origin issues.

    Authorship is one signal, not the verdict. The triage rubric must still
    judge substantive value regardless of this flag.
    """
    author_lower = author.lower()
    if any(hint in author_lower for hint in AUTOMATION_AUTHOR_HINTS):
        return True
    automation_labels = {"automation", "automated", "stage-gate-drift", "boss-stuck"}
    if any(label.lower() in automation_labels for label in labels):
        return True
    body_lower = body.lower()
    machine_markers = (
        "this issue was opened by",
        "auto-generated",
        "automation report",
        "boss-loop dispatched",
    )
    return any(marker in body_lower for marker in machine_markers)


def extract_file_references(body: str) -> list[str]:
    """Return file-path-like substrings mentioned in the issue body."""
    if not body:
        return []
    matches = FILE_PATH_PATTERN.findall(body)
    seen: set[str] = set()
    ordered: list[str] = []
    for match in matches:
        cleaned = match.strip().rstrip(".,)`")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            ordered.append(cleaned)
    return ordered


def extract_issue_references(body: str, *, exclude: int | None = None) -> list[int]:
    """Return #NNN issue/PR references in the body, excluding ``exclude``."""
    if not body:
        return []
    refs: list[int] = []
    seen: set[int] = set()
    for match in ISSUE_REF_PATTERN.findall(body):
        try:
            num = int(match)
        except ValueError:
            continue
        if num == exclude or num in seen:
            continue
        seen.add(num)
        refs.append(num)
    return refs


def _run_gh_json(args: Sequence[str]) -> Any:
    """Run a `gh` command expecting JSON output. Returns ``None`` on failure."""
    try:
        proc = subprocess.run(
            ["gh", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def _check_file_exists(repo_root: Path, path: str) -> bool:
    """Return True iff a path resolves to an existing file inside ``repo_root``."""
    try:
        normalized_path = path.replace("\\", "/")
        candidate = (repo_root / normalized_path).resolve()
        repo_resolved = repo_root.resolve()
        try:
            candidate.relative_to(repo_resolved)
        except ValueError:
            return False
        return candidate.is_file()
    except (OSError, ValueError):
        return False


def gather_evidence(
    issue: IssueRecord,
    *,
    repo: str = "synaptent/aragora",
    repo_root: Path | None = None,
    open_issue_index: Sequence[IssueRecord] | None = None,
    gh_runner: Callable[[Sequence[str]], Any] | None = None,
    now_iso: str = "",
) -> IssueEvidence:
    """Collect pre-model evidence for ``issue``.

    Args:
        issue: The issue to evaluate.
        repo: Owner/name slug for ``gh`` lookups.
        repo_root: Local checkout path. Used to verify file references.
        open_issue_index: Optional in-memory list of other open issues used
            to suggest duplicate candidates without hitting the network.
        gh_runner: Injection seam for tests; defaults to subprocess gh.
        now_iso: Timestamp for the receipt; tests pass a fixed value.

    Returns:
        ``IssueEvidence`` populated with everything we know prior to any
        model invocation. Failures degrade to empty fields plus a note so
        the audit trail is honest about what was actually checked.
    """
    runner = gh_runner or _run_gh_json
    repo_root = repo_root or Path.cwd()
    evidence = IssueEvidence(
        issue=issue,
        is_automation_generated=is_automation_generated(
            author=issue.author,
            labels=issue.labels,
            body=issue.body,
        ),
        gathered_at=now_iso,
    )

    file_refs = extract_file_references(issue.body)
    for path in file_refs[:20]:
        exists = _check_file_exists(repo_root, path)
        evidence.referenced_files.append({"path": path, "exists_in_head": exists})

    issue_refs = extract_issue_references(issue.body, exclude=issue.number)
    for ref in issue_refs[:10]:
        data = runner(
            [
                "issue",
                "view",
                str(ref),
                "--repo",
                repo,
                "--json",
                "number,title,state,url,closedAt",
            ]
        )
        if data is None:
            data = runner(
                [
                    "pr",
                    "view",
                    str(ref),
                    "--repo",
                    repo,
                    "--json",
                    "number,title,state,url,closedAt",
                ]
            )
        if data is None:
            evidence.referenced_issues.append(
                {"number": ref, "state": "unknown", "lookup": "failed"}
            )
            continue
        evidence.referenced_issues.append(
            {
                "number": data.get("number", ref),
                "title": data.get("title", ""),
                "state": data.get("state", "unknown"),
                "url": data.get("url", ""),
                "closed_at": data.get("closedAt"),
            }
        )

    if open_issue_index:
        evidence.duplicate_candidates = _shortlist_duplicates(issue, open_issue_index, limit=5)

    head = _read_repo_head(repo_root)
    if head:
        evidence.repo_head_sha = head

    if not evidence.referenced_issues and issue_refs:
        evidence.notes.append("issue references present but gh lookups returned nothing")
    return evidence


def _read_repo_head(repo_root: Path) -> str | None:
    try:
        head_file = repo_root / ".git" / "HEAD"
        if not head_file.exists():
            return None
        head_text = head_file.read_text().strip()
        if head_text.startswith("ref: "):
            ref_path = repo_root / ".git" / head_text.split(" ", 1)[1].strip()
            if ref_path.exists():
                return ref_path.read_text().strip()[:12]
            return None
        return head_text[:12]
    except OSError:
        return None


def _shortlist_duplicates(
    issue: IssueRecord,
    candidates: Sequence[IssueRecord],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return likely-duplicate issues using title-shingle Jaccard similarity."""
    target_shingles = _title_shingles(issue.title)
    if not target_shingles:
        return []
    scored: list[tuple[float, IssueRecord]] = []
    for candidate in candidates:
        if candidate.number == issue.number:
            continue
        candidate_shingles = _title_shingles(candidate.title)
        if not candidate_shingles:
            continue
        intersect = len(target_shingles & candidate_shingles)
        union = len(target_shingles | candidate_shingles)
        if union == 0:
            continue
        similarity = intersect / union
        if similarity >= 0.4:
            scored.append((similarity, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "number": rec.number,
            "title": rec.title,
            "url": rec.url,
            "similarity": round(score, 3),
        }
        for score, rec in scored[:limit]
    ]


def _title_shingles(title: str, *, size: int = 3) -> set[str]:
    tokens = [token.lower() for token in re.split(r"[^a-zA-Z0-9]+", title) if len(token) >= 2]
    if len(tokens) < size:
        return set(tokens)
    return {" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}
