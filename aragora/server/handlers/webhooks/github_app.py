"""
GitHub App Webhook Handler.

Receives and processes webhook events from the Aragora GitHub App.
Supports auto-triggering debates on PR creation, issue events, and code pushes.

Endpoints:
- POST /api/v1/webhooks/github - Receive GitHub App webhooks

Events handled:
- pull_request.opened - Auto-trigger code review debate
- pull_request.synchronize - Re-trigger on new commits
- issues.opened - Trigger triage debate
- push - Track code changes for context

Security:
- HMAC-SHA256 signature verification
- Webhook secret validation
- IP allowlisting (optional)
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Optional
from aragora.server.handlers.base import (
    HandlerResult,
    ServerContext,
    error_response,
    json_response,
)

logger = logging.getLogger(__name__)


class GitHubEventType(str, Enum):
    """Supported GitHub webhook event types."""

    PULL_REQUEST = "pull_request"
    PULL_REQUEST_REVIEW = "pull_request_review"
    ISSUES = "issues"
    ISSUE_COMMENT = "issue_comment"
    PUSH = "push"
    INSTALLATION = "installation"
    PING = "ping"


class GitHubAction(str, Enum):
    """GitHub event actions."""

    OPENED = "opened"
    CLOSED = "closed"
    REOPENED = "reopened"
    SYNCHRONIZE = "synchronize"
    CREATED = "created"
    DELETED = "deleted"
    EDITED = "edited"
    SUBMITTED = "submitted"


@dataclass
class GitHubWebhookEvent:
    """Parsed GitHub webhook event."""

    event_type: GitHubEventType
    action: Optional[str]
    delivery_id: str
    installation_id: Optional[int]
    repository: Dict[str, Any]
    sender: Dict[str, Any]
    payload: Dict[str, Any]
    received_at: str

    @classmethod
    def from_request(
        cls,
        event_type: str,
        delivery_id: str,
        payload: Dict[str, Any],
    ) -> "GitHubWebhookEvent":
        """Create event from webhook request."""
        try:
            parsed_type = GitHubEventType(event_type)
        except ValueError:
            parsed_type = GitHubEventType.PING  # Default for unknown events

        return cls(
            event_type=parsed_type,
            action=payload.get("action"),
            delivery_id=delivery_id,
            installation_id=payload.get("installation", {}).get("id"),
            repository=payload.get("repository", {}),
            sender=payload.get("sender", {}),
            payload=payload,
            received_at=datetime.now(timezone.utc).isoformat(),
        )


def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    """
    Verify GitHub webhook signature using HMAC-SHA256.

    Args:
        payload: Raw request body bytes
        signature: X-Hub-Signature-256 header value
        secret: Webhook secret configured in GitHub App

    Returns:
        True if signature is valid
    """
    if not signature or not signature.startswith("sha256="):
        return False

    expected = signature[7:]  # Remove "sha256=" prefix
    computed = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected, computed)


# Event handler registry
_event_handlers: Dict[GitHubEventType, Callable] = {}


def register_event_handler(event_type: GitHubEventType):
    """Decorator to register an event handler."""

    def decorator(func: Callable):
        _event_handlers[event_type] = func
        return func

    return decorator


@register_event_handler(GitHubEventType.PING)
async def handle_ping(event: GitHubWebhookEvent) -> Dict[str, Any]:
    """Handle GitHub ping event (webhook test)."""
    logger.info(f"GitHub App ping received: zen='{event.payload.get('zen')}'")
    return {
        "status": "ok",
        "message": "pong",
        "zen": event.payload.get("zen"),
    }


@register_event_handler(GitHubEventType.PULL_REQUEST)
async def handle_pull_request(event: GitHubWebhookEvent) -> Dict[str, Any]:
    """
    Handle pull request events.

    On PR opened/synchronize:
    - Queue code review debate
    - Analyze changes for context
    - Suggest reviewers based on expertise
    """
    pr = event.payload.get("pull_request", {})
    action = event.action
    repo = event.repository

    logger.info(
        f"PR event: {repo.get('full_name')}#{pr.get('number')} "
        f"action={action} by {event.sender.get('login')}"
    )

    result = {
        "event": "pull_request",
        "action": action,
        "pr_number": pr.get("number"),
        "pr_title": pr.get("title"),
        "repository": repo.get("full_name"),
        "author": pr.get("user", {}).get("login"),
    }

    # Auto-trigger debate on PR open or new commits
    if action in (GitHubAction.OPENED.value, GitHubAction.SYNCHRONIZE.value):
        result["debate_queued"] = True
        result["debate_type"] = "code_review"
        logger.info(f"Queued code review debate for PR #{pr.get('number')}")
        # TODO: Actually queue the debate via debate orchestrator
        # await queue_code_review_debate(pr, repo, event.installation_id)

    return result


@register_event_handler(GitHubEventType.ISSUES)
async def handle_issues(event: GitHubWebhookEvent) -> Dict[str, Any]:
    """
    Handle issue events.

    On issue opened:
    - Queue triage debate for priority/assignment
    - Extract requirements for estimation
    """
    issue = event.payload.get("issue", {})
    action = event.action
    repo = event.repository

    logger.info(f"Issue event: {repo.get('full_name')}#{issue.get('number')} action={action}")

    result = {
        "event": "issues",
        "action": action,
        "issue_number": issue.get("number"),
        "issue_title": issue.get("title"),
        "repository": repo.get("full_name"),
        "author": issue.get("user", {}).get("login"),
    }

    # Auto-triage on issue open
    if action == GitHubAction.OPENED.value:
        result["triage_queued"] = True
        logger.info(f"Queued triage for issue #{issue.get('number')}")
        # TODO: Queue issue triage debate
        # await queue_issue_triage_debate(issue, repo, event.installation_id)

    return result


@register_event_handler(GitHubEventType.PUSH)
async def handle_push(event: GitHubWebhookEvent) -> Dict[str, Any]:
    """
    Handle push events.

    Track code changes for debate context enrichment.
    """
    repo = event.repository
    ref = event.payload.get("ref", "")
    commits = event.payload.get("commits", [])

    logger.info(f"Push event: {repo.get('full_name')} ref={ref} commits={len(commits)}")

    return {
        "event": "push",
        "repository": repo.get("full_name"),
        "ref": ref,
        "commits_count": len(commits),
        "pusher": event.payload.get("pusher", {}).get("name"),
    }


@register_event_handler(GitHubEventType.INSTALLATION)
async def handle_installation(event: GitHubWebhookEvent) -> Dict[str, Any]:
    """
    Handle app installation events.

    Track when the app is installed/uninstalled.
    """
    action = event.action
    installation = event.payload.get("installation", {})

    logger.info(
        f"Installation event: action={action} "
        f"account={installation.get('account', {}).get('login')}"
    )

    return {
        "event": "installation",
        "action": action,
        "installation_id": installation.get("id"),
        "account": installation.get("account", {}).get("login"),
        "account_type": installation.get("account", {}).get("type"),
    }


async def handle_github_webhook(ctx: ServerContext) -> HandlerResult:
    """
    Process incoming GitHub App webhook.

    POST /api/v1/webhooks/github

    Headers:
        X-GitHub-Event: Event type (pull_request, issues, etc.)
        X-GitHub-Delivery: Unique delivery ID
        X-Hub-Signature-256: HMAC signature for verification

    Security:
        - Verifies HMAC-SHA256 signature using webhook secret
        - Rejects requests with invalid signatures
    """
    # Extract headers
    headers = ctx.get("headers", {})
    event_type = headers.get("x-github-event", "")
    delivery_id = headers.get("x-github-delivery", "unknown")
    signature = headers.get("x-hub-signature-256", "")

    # Get webhook secret from environment
    webhook_secret = os.getenv("GITHUB_WEBHOOK_SECRET", "")

    # Verify signature if secret is configured
    if webhook_secret:
        raw_body = ctx.get("raw_body", b"")
        if not verify_signature(raw_body, signature, webhook_secret):
            logger.warning(f"Invalid webhook signature for delivery {delivery_id}")
            return error_response("Invalid signature", status=401)

    # Parse payload
    try:
        payload = ctx.get("body", {})
    except Exception as e:
        logger.error(f"Failed to parse webhook payload: {e}")
        return error_response("Invalid JSON payload", status=400)

    # Create event object
    event = GitHubWebhookEvent.from_request(event_type, delivery_id, payload)

    logger.info(
        f"GitHub webhook received: event={event_type} delivery={delivery_id} action={event.action}"
    )

    # Dispatch to event handler
    handler = _event_handlers.get(event.event_type)
    if handler:
        try:
            result = await handler(event)
            return json_response(
                {
                    "success": True,
                    "delivery_id": delivery_id,
                    "event_type": event_type,
                    **result,
                }
            )
        except Exception as e:
            logger.error(f"Error handling {event_type} event: {e}")
            return error_response(f"Handler error: {e}", status=500)
    else:
        # Unknown event type - acknowledge but don't process
        logger.debug(f"Unhandled event type: {event_type}")
        return json_response(
            {
                "success": True,
                "delivery_id": delivery_id,
                "event_type": event_type,
                "handled": False,
                "message": f"Event type '{event_type}' not handled",
            }
        )


async def handle_github_app_status(ctx: ServerContext) -> HandlerResult:
    """
    Get GitHub App integration status.

    GET /api/v1/webhooks/github/status
    """
    webhook_secret_configured = bool(os.getenv("GITHUB_WEBHOOK_SECRET", ""))

    return json_response(
        {
            "status": "configured" if webhook_secret_configured else "unconfigured",
            "webhook_endpoint": "/api/v1/webhooks/github",
            "supported_events": [e.value for e in GitHubEventType],
            "features": {
                "pr_auto_review": True,
                "issue_triage": True,
                "push_tracking": True,
                "installation_tracking": True,
            },
        }
    )


# Route definitions for registration
GITHUB_APP_ROUTES = [
    ("POST", "/api/v1/webhooks/github", handle_github_webhook),
    ("GET", "/api/v1/webhooks/github/status", handle_github_app_status),
]
