"""
HTTP API Handlers for Code Review.

Provides REST APIs for multi-agent code review:
- Code review requests
- Diff review
- PR review integration
- Review result management

Endpoints:
- POST /api/v1/code-review/review - Review code snippet
- POST /api/v1/code-review/diff - Review diff/patch
- POST /api/v1/code-review/pr - Review GitHub PR
- GET /api/v1/code-review/results/{id} - Get review results
- GET /api/v1/code-review/history - Get review history
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any, Dict, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    success_response,
)
from aragora.rbac.decorators import require_permission

logger = logging.getLogger(__name__)

# RBAC permissions for code review
CODE_REVIEW_READ_PERMISSION = "code_review:read"
CODE_REVIEW_WRITE_PERMISSION = "code_review:write"

# Thread-safe service instance
_code_reviewer: Optional[Any] = None
_code_reviewer_lock = threading.Lock()

# In-memory storage for review results
_review_results: Dict[str, Any] = {}
_review_results_lock = threading.Lock()


def get_code_reviewer():
    """Get or create code reviewer (thread-safe singleton)."""
    global _code_reviewer
    if _code_reviewer is not None:
        return _code_reviewer

    with _code_reviewer_lock:
        if _code_reviewer is None:
            from aragora.agents.code_reviewer import CodeReviewOrchestrator

            _code_reviewer = CodeReviewOrchestrator()
        return _code_reviewer


def store_review_result(result: Any) -> str:
    """Store review result and return ID."""
    with _review_results_lock:
        result_dict = result.to_dict() if hasattr(result, "to_dict") else result
        result_id = result_dict.get("id", f"review_{datetime.now().timestamp()}")
        _review_results[result_id] = {
            **result_dict,
            "stored_at": datetime.now().isoformat(),
        }
        return result_id


# =============================================================================
# Code Review Endpoints
# =============================================================================


@require_permission(CODE_REVIEW_WRITE_PERMISSION)
async def handle_review_code(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Review a code snippet.

    POST /api/v1/code-review/review
    Body: {
        code: str (required),
        language: str (optional, auto-detected if not provided),
        file_path: str (optional, for context),
        review_types: list[str] (optional - security, performance, maintainability, test_coverage),
        context: str (optional, additional context)
    }
    """
    try:
        reviewer = get_code_reviewer()

        code = data.get("code")
        if not code:
            return error_response("code is required", status=400)

        language = data.get("language")
        file_path = data.get("file_path")
        review_types = data.get("review_types")
        context = data.get("context")

        result = await reviewer.review_code(
            code=code,
            language=language,
            file_path=file_path,
            review_types=review_types,
            context=context,
        )

        result_id = store_review_result(result)

        return success_response(
            {
                "result": result.to_dict(),
                "result_id": result_id,
                "message": "Code review completed",
            }
        )

    except Exception as e:
        logger.exception("Error reviewing code")
        return error_response(f"Failed to review code: {e}", status=500)


@require_permission(CODE_REVIEW_WRITE_PERMISSION)
async def handle_review_diff(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Review a diff/patch.

    POST /api/v1/code-review/diff
    Body: {
        diff: str (required, unified diff format),
        base_branch: str (optional),
        head_branch: str (optional),
        review_types: list[str] (optional),
        context: str (optional)
    }
    """
    try:
        reviewer = get_code_reviewer()

        diff = data.get("diff")
        if not diff:
            return error_response("diff is required", status=400)

        result = await reviewer.review_diff(
            diff=diff,
            base_branch=data.get("base_branch"),
            head_branch=data.get("head_branch"),
            review_types=data.get("review_types"),
            context=data.get("context"),
        )

        result_id = store_review_result(result)

        return success_response(
            {
                "result": result.to_dict(),
                "result_id": result_id,
                "message": "Diff review completed",
            }
        )

    except Exception as e:
        logger.exception("Error reviewing diff")
        return error_response(f"Failed to review diff: {e}", status=500)


@require_permission(CODE_REVIEW_WRITE_PERMISSION)
async def handle_review_pr(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Review a GitHub pull request.

    POST /api/v1/code-review/pr
    Body: {
        pr_url: str (required, GitHub PR URL),
        review_types: list[str] (optional),
        post_comments: bool (optional, default false - requires GitHub token)
    }
    """
    try:
        reviewer = get_code_reviewer()

        pr_url = data.get("pr_url")
        if not pr_url:
            return error_response("pr_url is required", status=400)

        # Validate PR URL format
        if "github.com" not in pr_url or "/pull/" not in pr_url:
            return error_response(
                "Invalid PR URL. Expected format: https://github.com/owner/repo/pull/123",
                status=400,
            )

        result = await reviewer.review_pr(
            pr_url=pr_url,
            review_types=data.get("review_types"),
            post_comments=data.get("post_comments", False),
        )

        result_id = store_review_result(result)

        return success_response(
            {
                "result": result.to_dict(),
                "result_id": result_id,
                "message": "PR review completed",
            }
        )

    except Exception as e:
        logger.exception(f"Error reviewing PR: {data.get('pr_url')}")
        return error_response(f"Failed to review PR: {e}", status=500)


@require_permission(CODE_REVIEW_READ_PERMISSION)
async def handle_get_review_result(
    data: Dict[str, Any],
    result_id: str,
    user_id: str = "default",
) -> HandlerResult:
    """
    Get a review result by ID.

    GET /api/v1/code-review/results/{result_id}
    """
    try:
        with _review_results_lock:
            result = _review_results.get(result_id)

        if not result:
            return error_response(f"Review result {result_id} not found", status=404)

        return success_response({"result": result})

    except Exception as e:
        logger.exception(f"Error getting review result {result_id}")
        return error_response(f"Failed to get result: {e}", status=500)


@require_permission(CODE_REVIEW_READ_PERMISSION)
async def handle_get_review_history(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Get review history.

    GET /api/v1/code-review/history
    Query params: {
        limit: int (optional, default 50),
        offset: int (optional, default 0)
    }
    """
    try:
        limit = int(data.get("limit", 50))
        offset = int(data.get("offset", 0))

        with _review_results_lock:
            # Sort by stored_at descending
            sorted_results = sorted(
                _review_results.values(),
                key=lambda x: x.get("stored_at", ""),
                reverse=True,
            )
            paginated = sorted_results[offset : offset + limit]

        return success_response(
            {
                "reviews": paginated,
                "total": len(_review_results),
                "limit": limit,
                "offset": offset,
            }
        )

    except Exception as e:
        logger.exception("Error getting review history")
        return error_response(f"Failed to get history: {e}", status=500)


# =============================================================================
# Quick Review Endpoints
# =============================================================================


@require_permission(CODE_REVIEW_WRITE_PERMISSION)
async def handle_quick_security_scan(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Quick security-focused code scan.

    POST /api/v1/code-review/security-scan
    Body: {
        code: str (required),
        language: str (optional)
    }
    """
    try:
        reviewer = get_code_reviewer()

        code = data.get("code")
        if not code:
            return error_response("code is required", status=400)

        result = await reviewer.review_code(
            code=code,
            language=data.get("language"),
            review_types=["security"],
        )

        # Extract only security findings
        security_findings = [f for f in result.findings if f.category == "security"]

        return success_response(
            {
                "findings": [f.to_dict() for f in security_findings],
                "total": len(security_findings),
                "severity_summary": {
                    "critical": sum(1 for f in security_findings if f.severity == "critical"),
                    "high": sum(1 for f in security_findings if f.severity == "high"),
                    "medium": sum(1 for f in security_findings if f.severity == "medium"),
                    "low": sum(1 for f in security_findings if f.severity == "low"),
                },
            }
        )

    except Exception as e:
        logger.exception("Error in security scan")
        return error_response(f"Failed to scan code: {e}", status=500)


# =============================================================================
# Handler Registration
# =============================================================================


class CodeReviewHandler(BaseHandler):
    """Handler class for code review endpoints."""

    ROUTES: Dict[str, Any] = {
        "POST /api/v1/code-review/review": handle_review_code,
        "POST /api/v1/code-review/diff": handle_review_diff,
        "POST /api/v1/code-review/pr": handle_review_pr,
        "GET /api/v1/code-review/history": handle_get_review_history,
        "POST /api/v1/code-review/security-scan": handle_quick_security_scan,
    }

    DYNAMIC_ROUTES: Dict[str, Any] = {
        "GET /api/v1/code-review/results/{result_id}": handle_get_review_result,
    }
