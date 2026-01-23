"""
HTTP API Handlers for GitHub Pull Request Review.

Provides REST APIs for automated PR review:
- Trigger multi-agent PR review
- Get PR details and diff
- Submit review comments
- Manage review state

Endpoints:
- POST /api/v1/github/pr/review - Trigger PR review
- GET /api/v1/github/pr/{pr_number} - Get PR details
- POST /api/v1/github/pr/{pr_number}/review - Submit review
- GET /api/v1/github/pr/{pr_number}/reviews - List reviews
- POST /api/v1/github/pr/{pr_number}/comment - Add comment
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class ReviewVerdict(str, Enum):
    """PR review verdict."""

    APPROVE = "APPROVE"
    REQUEST_CHANGES = "REQUEST_CHANGES"
    COMMENT = "COMMENT"


class ReviewStatus(str, Enum):
    """Status of automated review."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReviewComment:
    """A single review comment."""

    id: str
    file_path: str
    line: int
    body: str
    side: str = "RIGHT"  # LEFT or RIGHT
    suggestion: Optional[str] = None
    severity: str = "info"  # info, warning, error
    category: str = "general"  # quality, security, performance, etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "file_path": self.file_path,
            "line": self.line,
            "body": self.body,
            "side": self.side,
            "suggestion": self.suggestion,
            "severity": self.severity,
            "category": self.category,
        }


@dataclass
class PRReviewResult:
    """Result of an automated PR review."""

    review_id: str
    pr_number: int
    repository: str
    status: ReviewStatus
    verdict: Optional[ReviewVerdict] = None
    summary: Optional[str] = None
    comments: List[ReviewComment] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "review_id": self.review_id,
            "pr_number": self.pr_number,
            "repository": self.repository,
            "status": self.status.value,
            "verdict": self.verdict.value if self.verdict else None,
            "summary": self.summary,
            "comments": [c.to_dict() for c in self.comments],
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "metrics": self.metrics,
        }


@dataclass
class PRDetails:
    """Pull request details."""

    number: int
    title: str
    body: str
    state: str
    author: str
    base_branch: str
    head_branch: str
    diff: Optional[str] = None
    changed_files: List[Dict[str, Any]] = field(default_factory=list)
    commits: List[Dict[str, Any]] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "number": self.number,
            "title": self.title,
            "body": self.body,
            "state": self.state,
            "author": self.author,
            "base_branch": self.base_branch,
            "head_branch": self.head_branch,
            "changed_files": self.changed_files,
            "commits": self.commits,
            "labels": self.labels,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# =============================================================================
# In-Memory Storage (replace with database in production)
# =============================================================================

_review_results: Dict[str, PRReviewResult] = {}  # review_id -> result
_pr_reviews: Dict[str, List[str]] = {}  # "repo/pr_number" -> [review_ids]
_storage_lock = threading.Lock()
_running_reviews: Dict[str, asyncio.Task] = {}


# =============================================================================
# GitHub API Client (simplified)
# =============================================================================


class GitHubClient:
    """Simplified GitHub API client."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.base_url = "https://api.github.com"

    async def get_pr(self, owner: str, repo: str, pr_number: int) -> Optional[PRDetails]:
        """Get PR details from GitHub API."""
        import aiohttp

        if not self.token:
            # Return demo data
            return self._demo_pr(pr_number)

        try:
            headers = {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json",
            }

            async with aiohttp.ClientSession() as session:
                # Get PR details
                url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}"
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        return self._demo_pr(pr_number)
                    data = await response.json()

                # Get files
                files_url = f"{url}/files"
                async with session.get(files_url, headers=headers) as files_response:
                    files = await files_response.json() if files_response.status == 200 else []

                return PRDetails(
                    number=data["number"],
                    title=data["title"],
                    body=data.get("body") or "",
                    state=data["state"],
                    author=data["user"]["login"],
                    base_branch=data["base"]["ref"],
                    head_branch=data["head"]["ref"],
                    changed_files=[
                        {
                            "filename": f["filename"],
                            "status": f["status"],
                            "additions": f["additions"],
                            "deletions": f["deletions"],
                            "patch": f.get("patch", ""),
                        }
                        for f in files
                    ],
                    labels=[label["name"] for label in data.get("labels", [])],
                    created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
                    updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")),
                )

        except Exception as e:
            logger.exception(f"Failed to fetch PR: {e}")
            return self._demo_pr(pr_number)

    async def submit_review(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        event: ReviewVerdict,
        body: str,
        comments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Submit a review to GitHub."""
        import aiohttp

        if not self.token:
            return {"success": True, "demo": True}

        try:
            headers = {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json",
            }

            payload = {
                "event": event.value,
                "body": body,
            }
            if comments:
                payload["comments"] = comments

            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status in (200, 201):
                        return {"success": True, "data": await response.json()}
                    else:
                        error = await response.text()
                        return {"success": False, "error": error}

        except Exception as e:
            logger.exception(f"Failed to submit review: {e}")
            return {"success": False, "error": str(e)}

    def _demo_pr(self, pr_number: int) -> PRDetails:
        """Return demo PR data."""
        return PRDetails(
            number=pr_number,
            title=f"Demo PR #{pr_number}: Add feature implementation",
            body="This PR adds a new feature to the codebase.\n\n## Changes\n- Added new module\n- Updated tests\n- Fixed lint errors",
            state="open",
            author="demo-user",
            base_branch="main",
            head_branch=f"feature/demo-{pr_number}",
            changed_files=[
                {
                    "filename": "src/feature.py",
                    "status": "added",
                    "additions": 50,
                    "deletions": 0,
                    "patch": "@@ -0,0 +1,50 @@\n+def new_feature():\n+    pass",
                },
                {
                    "filename": "tests/test_feature.py",
                    "status": "added",
                    "additions": 30,
                    "deletions": 0,
                    "patch": "@@ -0,0 +1,30 @@\n+import pytest",
                },
                {
                    "filename": "README.md",
                    "status": "modified",
                    "additions": 5,
                    "deletions": 2,
                    "patch": "@@ -10,2 +10,5 @@\n-Old content\n+New content",
                },
            ],
            labels=["enhancement", "review-needed"],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )


# =============================================================================
# Review Handlers
# =============================================================================


async def handle_trigger_pr_review(
    repository: str,
    pr_number: int,
    review_type: str = "comprehensive",
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Trigger an automated PR review.

    POST /api/v1/github/pr/review
    {
        "repository": "owner/repo",
        "pr_number": 123,
        "review_type": "comprehensive"  # or "quick", "security"
    }
    """
    try:
        review_id = f"review_{uuid.uuid4().hex[:12]}"
        pr_key = f"{repository}/{pr_number}"

        # Check if review already running
        if pr_key in _running_reviews:
            task = _running_reviews[pr_key]
            if not task.done():
                return {
                    "success": False,
                    "error": "Review already in progress for this PR",
                }

        # Create initial result
        result = PRReviewResult(
            review_id=review_id,
            pr_number=pr_number,
            repository=repository,
            status=ReviewStatus.IN_PROGRESS,
        )

        with _storage_lock:
            _review_results[review_id] = result
            if pr_key not in _pr_reviews:
                _pr_reviews[pr_key] = []
            _pr_reviews[pr_key].append(review_id)

        # Start async review
        async def run_review():
            try:
                # Parse repository
                parts = repository.split("/")
                if len(parts) != 2:
                    raise ValueError("Invalid repository format")
                owner, repo = parts

                # Fetch PR details
                client = GitHubClient()
                pr_details = await client.get_pr(owner, repo, pr_number)
                if not pr_details:
                    raise ValueError("Could not fetch PR details")

                # Perform review (simplified - in real implementation would use debate)
                comments, verdict, summary = await _perform_review(pr_details, review_type)

                # Update result
                with _storage_lock:
                    result.status = ReviewStatus.COMPLETED
                    result.verdict = verdict
                    result.summary = summary
                    result.comments = comments
                    result.completed_at = datetime.now(timezone.utc)
                    result.metrics = {
                        "files_reviewed": len(pr_details.changed_files),
                        "comments_generated": len(comments),
                        "review_type": review_type,
                    }

                logger.info(
                    f"[PRReview] Completed review {review_id} for {repository}#{pr_number}: {verdict.value}"
                )

            except Exception as e:
                logger.exception(f"Review {review_id} failed: {e}")
                with _storage_lock:
                    result.status = ReviewStatus.FAILED
                    result.error = str(e)
                    result.completed_at = datetime.now(timezone.utc)

            finally:
                if pr_key in _running_reviews:
                    del _running_reviews[pr_key]

        # Create and store task
        task = asyncio.create_task(run_review())
        _running_reviews[pr_key] = task

        logger.info(f"[PRReview] Started review {review_id} for {repository}#{pr_number}")

        return {
            "success": True,
            "review_id": review_id,
            "status": "in_progress",
            "pr_number": pr_number,
            "repository": repository,
        }

    except Exception as e:
        logger.exception(f"Failed to trigger PR review: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def _perform_review(
    pr_details: PRDetails,
    review_type: str,
) -> tuple[List[ReviewComment], ReviewVerdict, str]:
    """
    Perform the actual review analysis.

    In a real implementation, this would trigger multi-agent debate
    using the PR_REVIEW_TEMPLATE workflow.
    """
    comments = []
    issues_found = []

    # Analyze each changed file
    for file in pr_details.changed_files:
        filename = file["filename"]
        patch = file.get("patch", "")

        # Check for common issues (simplified analysis)
        if "TODO" in patch or "FIXME" in patch:
            comments.append(
                ReviewComment(
                    id=f"comment_{uuid.uuid4().hex[:8]}",
                    file_path=filename,
                    line=1,
                    body="Consider addressing TODO/FIXME comments before merging.",
                    severity="info",
                    category="quality",
                )
            )

        if "console.log" in patch or "print(" in patch:
            comments.append(
                ReviewComment(
                    id=f"comment_{uuid.uuid4().hex[:8]}",
                    file_path=filename,
                    line=1,
                    body="Debug logging detected. Consider removing before merging.",
                    severity="warning",
                    category="quality",
                )
            )
            issues_found.append("debug_logging")

        if "password" in patch.lower() or "secret" in patch.lower():
            if "test" not in filename.lower():
                comments.append(
                    ReviewComment(
                        id=f"comment_{uuid.uuid4().hex[:8]}",
                        file_path=filename,
                        line=1,
                        body="Potential sensitive data detected. Please verify no secrets are hardcoded.",
                        severity="error",
                        category="security",
                    )
                )
                issues_found.append("potential_secrets")

        # Check for missing tests
        if filename.endswith(".py") and "test_" not in filename:
            has_test = any(
                f["filename"].startswith("test_") or "/tests/" in f["filename"]
                for f in pr_details.changed_files
            )
            if not has_test:
                comments.append(
                    ReviewComment(
                        id=f"comment_{uuid.uuid4().hex[:8]}",
                        file_path=filename,
                        line=1,
                        body="Consider adding tests for the changes in this file.",
                        severity="info",
                        category="quality",
                    )
                )

    # Determine verdict
    if "potential_secrets" in issues_found:
        verdict = ReviewVerdict.REQUEST_CHANGES
        summary = (
            "Security concerns detected. Please address the highlighted issues before merging."
        )
    elif "debug_logging" in issues_found:
        verdict = ReviewVerdict.COMMENT
        summary = "Minor issues detected. Consider addressing the suggestions."
    elif len(comments) > 0:
        verdict = ReviewVerdict.APPROVE
        summary = "LGTM with minor suggestions. Good work!"
    else:
        verdict = ReviewVerdict.APPROVE
        summary = "Looks good to me! No issues detected."

    return comments, verdict, summary


async def handle_get_pr_details(
    repository: str,
    pr_number: int,
) -> Dict[str, Any]:
    """
    Get PR details.

    GET /api/v1/github/pr/{pr_number}?repository=owner/repo
    """
    try:
        parts = repository.split("/")
        if len(parts) != 2:
            return {"success": False, "error": "Invalid repository format"}
        owner, repo = parts

        client = GitHubClient()
        pr_details = await client.get_pr(owner, repo, pr_number)

        if not pr_details:
            return {"success": False, "error": "PR not found"}

        return {
            "success": True,
            "pr": pr_details.to_dict(),
        }

    except Exception as e:
        logger.exception(f"Failed to get PR details: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_get_review_status(
    review_id: str,
) -> Dict[str, Any]:
    """
    Get review status/result.

    GET /api/v1/github/pr/review/{review_id}
    """
    try:
        with _storage_lock:
            result = _review_results.get(review_id)

        if not result:
            return {"success": False, "error": "Review not found"}

        return {
            "success": True,
            "review": result.to_dict(),
        }

    except Exception as e:
        logger.exception(f"Failed to get review status: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_list_pr_reviews(
    repository: str,
    pr_number: int,
) -> Dict[str, Any]:
    """
    List reviews for a PR.

    GET /api/v1/github/pr/{pr_number}/reviews?repository=owner/repo
    """
    try:
        pr_key = f"{repository}/{pr_number}"

        with _storage_lock:
            review_ids = _pr_reviews.get(pr_key, [])
            reviews = [
                _review_results[rid].to_dict() for rid in review_ids if rid in _review_results
            ]

        return {
            "success": True,
            "reviews": reviews,
            "total": len(reviews),
        }

    except Exception as e:
        logger.exception(f"Failed to list PR reviews: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def handle_submit_review(
    repository: str,
    pr_number: int,
    event: str,
    body: str,
    comments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Submit a review to GitHub.

    POST /api/v1/github/pr/{pr_number}/review
    {
        "repository": "owner/repo",
        "event": "APPROVE",
        "body": "LGTM!",
        "comments": []
    }
    """
    try:
        parts = repository.split("/")
        if len(parts) != 2:
            return {"success": False, "error": "Invalid repository format"}
        owner, repo = parts

        try:
            verdict = ReviewVerdict(event)
        except ValueError:
            return {"success": False, "error": f"Invalid event: {event}"}

        client = GitHubClient()
        result = await client.submit_review(
            owner=owner,
            repo=repo,
            pr_number=pr_number,
            event=verdict,
            body=body,
            comments=comments,
        )

        return result

    except Exception as e:
        logger.exception(f"Failed to submit review: {e}")
        return {
            "success": False,
            "error": str(e),
        }


# =============================================================================
# Handler Class
# =============================================================================


class PRReviewHandler(BaseHandler):
    """
    HTTP handler for PR review endpoints.

    Integrates with the Aragora server routing system.
    """

    ROUTES = [
        "/api/v1/github/pr/review",
    ]

    ROUTE_PREFIXES = [
        "/api/v1/github/pr/",
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
        """Route PR review endpoint requests."""
        return None

    async def handle_post_trigger_review(self, data: Dict[str, Any]) -> HandlerResult:
        """POST /api/v1/github/pr/review"""
        repository = data.get("repository")
        pr_number = data.get("pr_number")

        if not repository or not pr_number:
            return error_response("repository and pr_number required", 400)

        result = await handle_trigger_pr_review(
            repository=repository,
            pr_number=pr_number,
            review_type=data.get("review_type", "comprehensive"),
            workspace_id=data.get("workspace_id"),
            user_id=self._get_user_id(),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_get_pr(self, params: Dict[str, Any], pr_number: int) -> HandlerResult:
        """GET /api/v1/github/pr/{pr_number}"""
        repository = params.get("repository")
        if not repository:
            return error_response("repository required", 400)

        result = await handle_get_pr_details(
            repository=repository,
            pr_number=pr_number,
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 404)

    async def handle_get_review_status(
        self, params: Dict[str, Any], review_id: str
    ) -> HandlerResult:
        """GET /api/v1/github/pr/review/{review_id}"""
        result = await handle_get_review_status(review_id=review_id)

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 404)

    async def handle_list_reviews(self, params: Dict[str, Any], pr_number: int) -> HandlerResult:
        """GET /api/v1/github/pr/{pr_number}/reviews"""
        repository = params.get("repository")
        if not repository:
            return error_response("repository required", 400)

        result = await handle_list_pr_reviews(
            repository=repository,
            pr_number=pr_number,
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    async def handle_post_submit_review(
        self, data: Dict[str, Any], pr_number: int
    ) -> HandlerResult:
        """POST /api/v1/github/pr/{pr_number}/review"""
        repository = data.get("repository")
        event = data.get("event")
        body = data.get("body", "")

        if not repository or not event:
            return error_response("repository and event required", 400)

        result = await handle_submit_review(
            repository=repository,
            pr_number=pr_number,
            event=event,
            body=body,
            comments=data.get("comments"),
        )

        if result.get("success"):
            return success_response(result)
        else:
            return error_response(result.get("error", "Unknown error"), 400)

    def _get_user_id(self) -> str:
        """Get user ID from auth context."""
        auth_ctx = self.ctx.get("auth_context")
        if auth_ctx and hasattr(auth_ctx, "user_id"):
            return auth_ctx.user_id
        return "default"
