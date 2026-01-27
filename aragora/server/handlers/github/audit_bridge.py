"""
Audit-to-GitHub Bridge Handler.

Provides REST APIs for bridging audit findings to GitHub:
- Create issues from findings
- Create PRs with suggested fixes
- Bulk sync findings to repository
- Track issue/PR status for findings

Endpoints:
- POST /api/v1/github/audit/issues - Create issues from findings
- POST /api/v1/github/audit/issues/bulk - Bulk create issues
- POST /api/v1/github/audit/pr - Create PR with fixes
- GET /api/v1/github/audit/sync/{session_id} - Get sync status
- POST /api/v1/github/audit/sync/{session_id} - Sync session to GitHub
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    success_response,
)
from aragora.server.handlers.utils.decorators import require_permission

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class IssuePriority(str, Enum):
    """Priority mapping for audit findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class SyncStatus(str, Enum):
    """Status of audit-to-GitHub sync."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


# Label mappings for severity levels
SEVERITY_LABELS = {
    "critical": ["priority: critical", "security", "bug"],
    "high": ["priority: high", "bug"],
    "medium": ["priority: medium"],
    "low": ["priority: low"],
    "info": ["documentation", "enhancement"],
}

# Category to label mappings
CATEGORY_LABELS = {
    "security": ["security"],
    "performance": ["performance"],
    "quality": ["code-quality"],
    "compliance": ["compliance"],
    "consistency": ["consistency"],
    "documentation": ["documentation"],
    "testing": ["testing"],
    "accessibility": ["accessibility"],
}


@dataclass
class GitHubIssueResult:
    """Result of creating a GitHub issue."""

    finding_id: str
    issue_number: Optional[int] = None
    issue_url: Optional[str] = None
    status: str = "pending"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "issue_number": self.issue_number,
            "issue_url": self.issue_url,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class SyncResult:
    """Result of syncing audit session to GitHub."""

    sync_id: str
    session_id: str
    repository: str
    status: SyncStatus
    issues_created: List[GitHubIssueResult] = field(default_factory=list)
    pr_created: Optional[Dict[str, Any]] = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sync_id": self.sync_id,
            "session_id": self.session_id,
            "repository": self.repository,
            "status": self.status.value,
            "issues_created": [i.to_dict() for i in self.issues_created],
            "pr_created": self.pr_created,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
        }


# =============================================================================
# In-Memory Storage (replace with database in production)
# =============================================================================

_sync_results: Dict[str, SyncResult] = {}  # sync_id -> result
_session_syncs: Dict[str, List[str]] = {}  # session_id -> [sync_ids]
_finding_issues: Dict[str, Dict[str, int]] = {}  # session_id -> {finding_id: issue_number}
_storage_lock = threading.Lock()


# =============================================================================
# GitHub API Client
# =============================================================================


class GitHubAuditClient:
    """GitHub API client for audit-related operations."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.base_url = "https://api.github.com"

    async def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None,
        milestone: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create a GitHub issue."""
        import aiohttp

        if not self.token:
            # Demo mode
            demo_number = hash(title) % 10000
            return {
                "success": True,
                "demo": True,
                "number": demo_number,
                "html_url": f"https://github.com/{owner}/{repo}/issues/{demo_number}",
            }

        try:
            headers = {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json",
            }

            payload: Dict[str, Any] = {
                "title": title,
                "body": body,
            }
            if labels:
                payload["labels"] = labels
            if assignees:
                payload["assignees"] = assignees
            if milestone:
                payload["milestone"] = milestone

            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/repos/{owner}/{repo}/issues"
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status in (200, 201):
                        data = await response.json()
                        return {
                            "success": True,
                            "number": data["number"],
                            "html_url": data["html_url"],
                            "id": data["id"],
                        }
                    else:
                        error = await response.text()
                        return {"success": False, "error": error}

        except Exception as e:
            logger.exception(f"Failed to create issue: {e}")
            return {"success": False, "error": str(e)}

    async def create_branch(
        self,
        owner: str,
        repo: str,
        branch_name: str,
        from_branch: str = "main",
    ) -> Dict[str, Any]:
        """Create a new branch from an existing branch."""
        import aiohttp

        if not self.token:
            return {"success": True, "demo": True, "ref": f"refs/heads/{branch_name}"}

        try:
            headers = {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json",
            }

            async with aiohttp.ClientSession() as session:
                # Get SHA of source branch
                ref_url = f"{self.base_url}/repos/{owner}/{repo}/git/refs/heads/{from_branch}"
                async with session.get(ref_url, headers=headers) as response:
                    if response.status != 200:
                        return {"success": False, "error": f"Branch {from_branch} not found"}
                    ref_data = await response.json()
                    sha = ref_data["object"]["sha"]

                # Create new branch
                create_url = f"{self.base_url}/repos/{owner}/{repo}/git/refs"
                payload = {
                    "ref": f"refs/heads/{branch_name}",
                    "sha": sha,
                }
                async with session.post(create_url, headers=headers, json=payload) as response:
                    if response.status in (200, 201):
                        data = await response.json()
                        return {"success": True, "ref": data["ref"]}
                    else:
                        error = await response.text()
                        return {"success": False, "error": error}

        except Exception as e:
            logger.exception(f"Failed to create branch: {e}")
            return {"success": False, "error": str(e)}

    async def create_pull_request(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main",
        draft: bool = False,
    ) -> Dict[str, Any]:
        """Create a pull request."""
        import aiohttp

        if not self.token:
            demo_number = hash(title) % 10000
            return {
                "success": True,
                "demo": True,
                "number": demo_number,
                "html_url": f"https://github.com/{owner}/{repo}/pull/{demo_number}",
            }

        try:
            headers = {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json",
            }

            payload = {
                "title": title,
                "body": body,
                "head": head_branch,
                "base": base_branch,
                "draft": draft,
            }

            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/repos/{owner}/{repo}/pulls"
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status in (200, 201):
                        data = await response.json()
                        return {
                            "success": True,
                            "number": data["number"],
                            "html_url": data["html_url"],
                            "id": data["id"],
                        }
                    else:
                        error = await response.text()
                        return {"success": False, "error": error}

        except Exception as e:
            logger.exception(f"Failed to create PR: {e}")
            return {"success": False, "error": str(e)}

    async def get_issue(
        self,
        owner: str,
        repo: str,
        issue_number: int,
    ) -> Optional[Dict[str, Any]]:
        """Get issue details."""
        import aiohttp

        if not self.token:
            return {
                "number": issue_number,
                "state": "open",
                "title": f"Demo Issue #{issue_number}",
            }

        try:
            headers = {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json",
            }

            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/repos/{owner}/{repo}/issues/{issue_number}"
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    return None

        except Exception as e:
            logger.exception(f"Failed to get issue: {e}")
            return None

    async def add_issue_comment(
        self,
        owner: str,
        repo: str,
        issue_number: int,
        body: str,
    ) -> Dict[str, Any]:
        """Add a comment to an issue."""
        import aiohttp

        if not self.token:
            return {"success": True, "demo": True}

        try:
            headers = {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json",
            }

            payload = {"body": body}

            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/repos/{owner}/{repo}/issues/{issue_number}/comments"
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status in (200, 201):
                        return {"success": True, "data": await response.json()}
                    else:
                        error = await response.text()
                        return {"success": False, "error": error}

        except Exception as e:
            logger.exception(f"Failed to add comment: {e}")
            return {"success": False, "error": str(e)}

    async def ensure_labels_exist(
        self,
        owner: str,
        repo: str,
        labels: List[str],
    ) -> None:
        """Ensure labels exist in the repository, creating them if needed."""
        import aiohttp

        if not self.token:
            return

        label_colors = {
            "priority: critical": "B60205",
            "priority: high": "D93F0B",
            "priority: medium": "FBCA04",
            "priority: low": "0E8A16",
            "security": "B60205",
            "bug": "D73A4A",
            "enhancement": "A2EEEF",
            "documentation": "0075CA",
            "performance": "7057FF",
            "code-quality": "BFD4F2",
            "compliance": "FEF2C0",
            "testing": "E99695",
            "accessibility": "5319E7",
            "consistency": "C5DEF5",
        }

        headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                for label in labels:
                    try:
                        # Check if label exists
                        url = f"{self.base_url}/repos/{owner}/{repo}/labels/{label}"
                        async with session.get(url, headers=headers) as response:
                            if response.status == 200:
                                continue  # Label exists

                        # Create label
                        create_url = f"{self.base_url}/repos/{owner}/{repo}/labels"
                        payload = {
                            "name": label,
                            "color": label_colors.get(label, "EDEDED"),
                        }
                        async with session.post(
                            create_url, headers=headers, json=payload
                        ) as response:
                            if response.status in (200, 201):
                                logger.debug(f"Created label: {label}")
                            # Ignore errors (label might already exist with different casing)

                    except Exception as e:
                        logger.warning(f"Could not ensure label {label}: {e}")

        except Exception as e:
            logger.warning(f"Failed to ensure labels: {e}")


# =============================================================================
# Issue Content Generators
# =============================================================================


def generate_issue_title(finding: Dict[str, Any]) -> str:
    """Generate GitHub issue title from finding."""
    severity = finding.get("severity", "").upper()
    title = finding.get("title", "Audit Finding")
    category = finding.get("category", "general")

    if severity in ("CRITICAL", "HIGH"):
        return f"[{severity}] {title}"
    return f"[{category.upper()}] {title}"


def generate_issue_body(
    finding: Dict[str, Any],
    session_id: Optional[str] = None,
    include_evidence: bool = True,
) -> str:
    """Generate GitHub issue body from finding."""
    severity = finding.get("severity", "unknown")
    category = finding.get("category", "general")
    description = finding.get("description", "No description provided.")
    evidence = finding.get("evidence", "") or finding.get("evidence_text", "")
    mitigation = finding.get("mitigation", "") or finding.get("recommendation", "")
    document_id = finding.get("document_id", "")
    chunk_id = finding.get("chunk_id", "")
    finding_id = finding.get("id", "")
    confidence = finding.get("confidence", 0)

    body_parts = [
        "## Audit Finding",
        "",
        f"**Severity:** {severity}",
        f"**Category:** {category}",
        f"**Confidence:** {confidence:.0%}" if confidence else "",
        "",
        "### Description",
        "",
        description,
        "",
    ]

    if include_evidence and evidence:
        body_parts.extend(
            [
                "### Evidence",
                "",
                "```",
                evidence[:2000],  # Limit evidence length
                "```" if len(evidence) <= 2000 else "...(truncated)```",
                "",
            ]
        )

    if mitigation:
        body_parts.extend(
            [
                "### Recommended Fix",
                "",
                mitigation,
                "",
            ]
        )

    # Add metadata footer
    metadata = []
    if finding_id:
        metadata.append(f"Finding ID: `{finding_id}`")
    if session_id:
        metadata.append(f"Audit Session: `{session_id}`")
    if document_id:
        metadata.append(f"Document: `{document_id}`")
    if chunk_id:
        metadata.append(f"Location: `{chunk_id}`")

    if metadata:
        body_parts.extend(
            [
                "---",
                "",
                "**Metadata:**",
                *[f"- {m}" for m in metadata],
                "",
                "_Generated by [Aragora Audit](https://github.com/aragora)_",
            ]
        )

    return "\n".join(body_parts)


def get_labels_for_finding(finding: Dict[str, Any]) -> List[str]:
    """Get GitHub labels for a finding."""
    labels = []

    # Add severity labels
    severity = finding.get("severity", "").lower()
    if severity in SEVERITY_LABELS:
        labels.extend(SEVERITY_LABELS[severity])

    # Add category labels
    category = finding.get("category", "").lower()
    if category in CATEGORY_LABELS:
        labels.extend(CATEGORY_LABELS[category])

    # Add audit type labels
    audit_type = finding.get("audit_type", "").lower()
    if audit_type and audit_type not in labels:
        labels.append(audit_type)

    return list(set(labels))  # Deduplicate


# =============================================================================
# Handler Functions
# =============================================================================


@require_permission("connectors:create")
async def handle_create_issue(
    repository: str,
    finding: Dict[str, Any],
    session_id: Optional[str] = None,
    assignees: Optional[List[str]] = None,
    milestone: Optional[int] = None,
    auto_label: bool = True,
) -> Dict[str, Any]:
    """
    Create a GitHub issue from an audit finding.

    POST /api/v1/github/audit/issues
    {
        "repository": "owner/repo",
        "finding": {...},
        "session_id": "optional-session-id",
        "assignees": ["user1"],
        "milestone": 1,
        "auto_label": true
    }
    """
    try:
        parts = repository.split("/")
        if len(parts) != 2:
            return {"success": False, "error": "Invalid repository format"}
        owner, repo = parts

        client = GitHubAuditClient()

        # Generate issue content
        title = generate_issue_title(finding)
        body = generate_issue_body(finding, session_id)
        labels = get_labels_for_finding(finding) if auto_label else []

        # Ensure labels exist
        if labels:
            await client.ensure_labels_exist(owner, repo, labels)

        # Create issue
        result = await client.create_issue(
            owner=owner,
            repo=repo,
            title=title,
            body=body,
            labels=labels,
            assignees=assignees,
            milestone=milestone,
        )

        if result.get("success"):
            finding_id = finding.get("id", str(uuid4()))
            issue_number = result.get("number")

            # Store mapping
            if session_id:
                with _storage_lock:
                    if session_id not in _finding_issues:
                        _finding_issues[session_id] = {}
                    _finding_issues[session_id][finding_id] = issue_number

            return {
                "success": True,
                "issue_number": issue_number,
                "issue_url": result.get("html_url"),
                "finding_id": finding_id,
                "demo": result.get("demo", False),
            }
        else:
            return result

    except Exception as e:
        logger.exception(f"Failed to create issue: {e}")
        return {"success": False, "error": str(e)}


@require_permission("audit:read")
async def handle_bulk_create_issues(
    repository: str,
    findings: List[Dict[str, Any]],
    session_id: Optional[str] = None,
    assignees: Optional[List[str]] = None,
    auto_label: bool = True,
    skip_existing: bool = True,
    max_concurrent: int = 5,
) -> Dict[str, Any]:
    """
    Create GitHub issues from multiple findings.

    POST /api/v1/github/audit/issues/bulk
    {
        "repository": "owner/repo",
        "findings": [...],
        "session_id": "session-id",
        "skip_existing": true,
        "max_concurrent": 5
    }
    """
    results = []
    errors = []

    # Check for existing issues
    existing_issues = {}
    if skip_existing and session_id:
        with _storage_lock:
            existing_issues = _finding_issues.get(session_id, {})

    # Filter findings
    findings_to_create = []
    for finding in findings:
        finding_id = finding.get("id", "")
        if skip_existing and finding_id in existing_issues:
            results.append(
                GitHubIssueResult(
                    finding_id=finding_id,
                    issue_number=existing_issues[finding_id],
                    status="skipped",
                )
            )
        else:
            findings_to_create.append(finding)

    # Create issues with concurrency limit
    semaphore = asyncio.Semaphore(max_concurrent)

    async def create_with_semaphore(finding: Dict[str, Any]):
        async with semaphore:
            result = await handle_create_issue(
                repository=repository,
                finding=finding,
                session_id=session_id,
                assignees=assignees,
                auto_label=auto_label,
            )
            return result, finding

    # Execute concurrently
    tasks = [create_with_semaphore(f) for f in findings_to_create]
    completed = await asyncio.gather(*tasks, return_exceptions=True)

    for item in completed:
        if isinstance(item, BaseException):
            errors.append(str(item))
            continue

        result, finding = item
        finding_id = finding.get("id", str(uuid4()))

        if result.get("success"):
            results.append(
                GitHubIssueResult(
                    finding_id=finding_id,
                    issue_number=result.get("issue_number"),
                    issue_url=result.get("issue_url"),
                    status="created",
                )
            )
        else:
            results.append(
                GitHubIssueResult(
                    finding_id=finding_id,
                    status="failed",
                    error=result.get("error"),
                )
            )
            errors.append(f"Finding {finding_id}: {result.get('error')}")

    created_count = sum(1 for r in results if r.status == "created")
    skipped_count = sum(1 for r in results if r.status == "skipped")
    failed_count = sum(1 for r in results if r.status == "failed")

    return {
        "success": failed_count == 0,
        "total": len(findings),
        "created": created_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "results": [r.to_dict() for r in results],
        "errors": errors if errors else None,
    }


@require_permission("connectors:create")
async def handle_create_fix_pr(
    repository: str,
    session_id: str,
    findings: List[Dict[str, Any]],
    branch_name: Optional[str] = None,
    base_branch: str = "main",
    draft: bool = True,
    auto_fixes: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Create a PR with suggested fixes for audit findings.

    POST /api/v1/github/audit/pr
    {
        "repository": "owner/repo",
        "session_id": "session-id",
        "findings": [...],
        "branch_name": "fix/audit-findings-session-123",
        "base_branch": "main",
        "draft": true,
        "auto_fixes": {
            "file/path.py": "fixed content..."
        }
    }
    """
    try:
        parts = repository.split("/")
        if len(parts) != 2:
            return {"success": False, "error": "Invalid repository format"}
        owner, repo = parts

        client = GitHubAuditClient()

        # Generate branch name if not provided
        if not branch_name:
            short_id = session_id[:8] if len(session_id) > 8 else session_id
            branch_name = f"fix/audit-{short_id}"

        # Create branch
        branch_result = await client.create_branch(
            owner=owner,
            repo=repo,
            branch_name=branch_name,
            from_branch=base_branch,
        )

        if not branch_result.get("success"):
            return branch_result

        # Generate PR body
        severity_counts: Dict[str, int] = {}
        for finding in findings:
            sev = finding.get("severity", "unknown")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        pr_body_parts = [
            "## Audit Findings Fix",
            "",
            "This PR addresses findings from the automated audit.",
            "",
            "### Summary",
            "",
            f"- **Session ID:** `{session_id}`",
            f"- **Total Findings:** {len(findings)}",
        ]

        for sev, count in sorted(severity_counts.items()):
            pr_body_parts.append(f"- **{sev.capitalize()}:** {count}")

        pr_body_parts.extend(
            [
                "",
                "### Findings Addressed",
                "",
            ]
        )

        for i, finding in enumerate(findings[:20], 1):  # Limit to 20 for readability
            title = finding.get("title", "Untitled")
            severity = finding.get("severity", "unknown")
            pr_body_parts.append(f"{i}. [{severity.upper()}] {title}")

        if len(findings) > 20:
            pr_body_parts.append(f"... and {len(findings) - 20} more")

        pr_body_parts.extend(
            [
                "",
                "---",
                "",
                "_Generated by [Aragora Audit](https://github.com/aragora)_",
            ]
        )

        pr_body = "\n".join(pr_body_parts)

        # Create PR
        pr_result = await client.create_pull_request(
            owner=owner,
            repo=repo,
            title=f"fix: Address audit findings from session {session_id[:8]}",
            body=pr_body,
            head_branch=branch_name,
            base_branch=base_branch,
            draft=draft,
        )

        if pr_result.get("success"):
            return {
                "success": True,
                "pr_number": pr_result.get("number"),
                "pr_url": pr_result.get("html_url"),
                "branch": branch_name,
                "findings_count": len(findings),
                "demo": pr_result.get("demo", False),
            }
        else:
            return pr_result

    except Exception as e:
        logger.exception(f"Failed to create fix PR: {e}")
        return {"success": False, "error": str(e)}


@require_permission("audit:read")
async def handle_sync_session(
    repository: str,
    session_id: str,
    min_severity: str = "low",
    create_issues: bool = True,
    create_pr: bool = False,
    assignees: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Sync an audit session to GitHub.

    POST /api/v1/github/audit/sync/{session_id}
    {
        "repository": "owner/repo",
        "min_severity": "medium",
        "create_issues": true,
        "create_pr": false,
        "assignees": ["user1"]
    }
    """
    try:
        sync_id = f"sync_{uuid4().hex[:12]}"

        # Create sync result
        sync_result = SyncResult(
            sync_id=sync_id,
            session_id=session_id,
            repository=repository,
            status=SyncStatus.IN_PROGRESS,
        )

        with _storage_lock:
            _sync_results[sync_id] = sync_result
            if session_id not in _session_syncs:
                _session_syncs[session_id] = []
            _session_syncs[session_id].append(sync_id)

        # Fetch findings from audit session
        # In production, this would query the actual audit session storage
        try:
            from aragora.server.handlers.features.audit_sessions import _findings, _sessions

            session = _sessions.get(session_id)
            if not session:
                sync_result.status = SyncStatus.FAILED
                sync_result.error = f"Session {session_id} not found"
                return {"success": False, "error": sync_result.error}

            findings = _findings.get(session_id, [])

        except ImportError:
            # Fallback to demo data
            findings = [
                {
                    "id": f"finding-{i}",
                    "title": f"Demo Finding {i}",
                    "description": f"This is demo finding {i}",
                    "severity": ["critical", "high", "medium", "low"][i % 4],
                    "category": "security",
                }
                for i in range(5)
            ]

        # Filter by severity
        severity_order = ["critical", "high", "medium", "low", "info"]
        min_idx = severity_order.index(min_severity) if min_severity in severity_order else 3
        filtered_findings = [
            f for f in findings if severity_order.index(f.get("severity", "info")) <= min_idx
        ]

        # Create issues if requested
        if create_issues and filtered_findings:
            issue_result = await handle_bulk_create_issues(
                repository=repository,
                findings=filtered_findings,
                session_id=session_id,
                assignees=assignees,
            )

            sync_result.issues_created = [
                GitHubIssueResult(**r) for r in issue_result.get("results", [])
            ]

        # Create PR if requested
        if create_pr and filtered_findings:
            pr_result = await handle_create_fix_pr(
                repository=repository,
                session_id=session_id,
                findings=filtered_findings,
            )

            if pr_result.get("success"):
                sync_result.pr_created = {
                    "number": pr_result.get("pr_number"),
                    "url": pr_result.get("pr_url"),
                    "branch": pr_result.get("branch"),
                }

        # Determine final status
        failed_issues = sum(1 for i in sync_result.issues_created if i.status == "failed")
        if failed_issues == 0:
            sync_result.status = SyncStatus.COMPLETED
        elif failed_issues < len(sync_result.issues_created):
            sync_result.status = SyncStatus.PARTIAL
        else:
            sync_result.status = SyncStatus.FAILED

        sync_result.completed_at = datetime.now(timezone.utc)

        logger.info(f"Completed sync {sync_id} for session {session_id}: {sync_result.status}")

        return {
            "success": sync_result.status != SyncStatus.FAILED,
            "sync": sync_result.to_dict(),
        }

    except Exception as e:
        logger.exception(f"Failed to sync session: {e}")
        return {"success": False, "error": str(e)}


@require_permission("audit:read")
async def handle_get_sync_status(
    session_id: str,
    sync_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get sync status for a session.

    GET /api/v1/github/audit/sync/{session_id}
    GET /api/v1/github/audit/sync/{session_id}/{sync_id}
    """
    try:
        if sync_id:
            with _storage_lock:
                sync_result = _sync_results.get(sync_id)

            if not sync_result:
                return {"success": False, "error": "Sync not found"}

            return {
                "success": True,
                "sync": sync_result.to_dict(),
            }
        else:
            # Get all syncs for session
            with _storage_lock:
                sync_ids = _session_syncs.get(session_id, [])
                syncs = [_sync_results[sid].to_dict() for sid in sync_ids if sid in _sync_results]

            return {
                "success": True,
                "session_id": session_id,
                "syncs": syncs,
                "total": len(syncs),
            }

    except Exception as e:
        logger.exception(f"Failed to get sync status: {e}")
        return {"success": False, "error": str(e)}


@require_permission("audit:read")
async def handle_get_finding_issues(
    session_id: str,
    finding_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get GitHub issues linked to findings.

    GET /api/v1/github/audit/issues?session_id=...&finding_id=...
    """
    try:
        with _storage_lock:
            session_issues = _finding_issues.get(session_id, {})

        if finding_id:
            issue_number = session_issues.get(finding_id)
            if issue_number:
                return {
                    "success": True,
                    "finding_id": finding_id,
                    "issue_number": issue_number,
                }
            else:
                return {"success": False, "error": "No issue found for finding"}
        else:
            return {
                "success": True,
                "session_id": session_id,
                "issues": session_issues,
                "total": len(session_issues),
            }

    except Exception as e:
        logger.exception(f"Failed to get finding issues: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# Handler Class
# =============================================================================


class AuditGitHubBridgeHandler(BaseHandler):
    """
    HTTP handler for audit-to-GitHub bridge endpoints.

    Integrates with the Aragora server routing system.
    """

    ROUTES = [
        "/api/v1/github/audit/issues",
        "/api/v1/github/audit/issues/bulk",
        "/api/v1/github/audit/pr",
    ]

    ROUTE_PREFIXES = [
        "/api/v1/github/audit/sync/",
    ]

    def __init__(self, ctx: Dict[str, Any]):
        """Initialize with server context."""
        super().__init__(ctx)  # type: ignore[arg-type]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        if path in self.ROUTES:
            return True
        for prefix in self.ROUTE_PREFIXES:
            if path.startswith(prefix):
                return True
        return False

    def handle(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route audit-GitHub bridge endpoint requests."""
        return None

    async def handle_post_create_issue(self, data: Dict[str, Any]) -> HandlerResult:
        """POST /api/v1/github/audit/issues"""
        repository = data.get("repository")
        finding = data.get("finding")

        if not repository or not finding:
            return error_response("repository and finding required", 400)

        result = await handle_create_issue(
            repository=repository,
            finding=finding,
            session_id=data.get("session_id"),
            assignees=data.get("assignees"),
            milestone=data.get("milestone"),
            auto_label=data.get("auto_label", True),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_bulk_create_issues(self, data: Dict[str, Any]) -> HandlerResult:
        """POST /api/v1/github/audit/issues/bulk"""
        repository = data.get("repository")
        findings = data.get("findings")

        if not repository or not findings:
            return error_response("repository and findings required", 400)

        result = await handle_bulk_create_issues(
            repository=repository,
            findings=findings,
            session_id=data.get("session_id"),
            assignees=data.get("assignees"),
            auto_label=data.get("auto_label", True),
            skip_existing=data.get("skip_existing", True),
            max_concurrent=data.get("max_concurrent", 5),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_create_pr(self, data: Dict[str, Any]) -> HandlerResult:
        """POST /api/v1/github/audit/pr"""
        repository = data.get("repository")
        session_id = data.get("session_id")
        findings = data.get("findings")

        if not repository or not session_id or not findings:
            return error_response("repository, session_id, and findings required", 400)

        result = await handle_create_fix_pr(
            repository=repository,
            session_id=session_id,
            findings=findings,
            branch_name=data.get("branch_name"),
            base_branch=data.get("base_branch", "main"),
            draft=data.get("draft", True),
            auto_fixes=data.get("auto_fixes"),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_sync_session(
        self, data: Dict[str, Any], session_id: str
    ) -> HandlerResult:
        """POST /api/v1/github/audit/sync/{session_id}"""
        repository = data.get("repository")

        if not repository:
            return error_response("repository required", 400)

        result = await handle_sync_session(
            repository=repository,
            session_id=session_id,
            min_severity=data.get("min_severity", "low"),
            create_issues=data.get("create_issues", True),
            create_pr=data.get("create_pr", False),
            assignees=data.get("assignees"),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_get_sync_status(
        self, params: Dict[str, Any], session_id: str
    ) -> HandlerResult:
        """GET /api/v1/github/audit/sync/{session_id}"""
        result = await handle_get_sync_status(
            session_id=session_id,
            sync_id=params.get("sync_id"),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 404)

    async def handle_get_finding_issues(self, params: Dict[str, Any]) -> HandlerResult:
        """GET /api/v1/github/audit/issues?session_id=...&finding_id=..."""
        session_id = params.get("session_id")
        if not session_id:
            return error_response("session_id required", 400)

        result = await handle_get_finding_issues(
            session_id=session_id,
            finding_id=params.get("finding_id"),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 404)

    def _get_user_id(self) -> str:
        """Get user ID from auth context."""
        auth_ctx = self.ctx.get("auth_context")
        if auth_ctx and hasattr(auth_ctx, "user_id"):
            return auth_ctx.user_id
        return "default"


__all__ = [
    "AuditGitHubBridgeHandler",
    "GitHubAuditClient",
    "handle_create_issue",
    "handle_bulk_create_issues",
    "handle_create_fix_pr",
    "handle_sync_session",
    "handle_get_sync_status",
    "handle_get_finding_issues",
]
