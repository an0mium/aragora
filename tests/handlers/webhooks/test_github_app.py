"""
Tests for GitHub App Webhook Handler.

Covers:
- HMAC-SHA256 signature verification
- Webhook event routing and dispatch
- Pull request event handling (opened, synchronize, closed, etc.)
- Issue event handling (opened, closed, etc.)
- Push event handling
- Installation event handling
- Ping event handling
- Unknown event types
- Status endpoint
- Security: missing/invalid signature, missing secret in production
- GitHubWebhookEvent.from_request construction
- GitHubEventType and GitHubAction enums
- Event handler registry
- queue_code_review_debate and queue_issue_triage_debate
- Error handling in event handlers
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.webhooks.github_app import (
    GitHubAction,
    GitHubEventType,
    GitHubWebhookEvent,
    GITHUB_APP_ROUTES,
    _event_handlers,
    handle_github_app_status,
    handle_github_webhook,
    handle_installation,
    handle_issues,
    handle_ping,
    handle_pull_request,
    handle_push,
    queue_code_review_debate,
    queue_issue_triage_debate,
    register_event_handler,
    verify_signature,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_signature(payload_bytes: bytes, secret: str) -> str:
    """Compute a valid HMAC-SHA256 signature for testing."""
    mac = hmac_mod.new(secret.encode(), payload_bytes, hashlib.sha256)
    return f"sha256={mac.hexdigest()}"


def _make_ctx(
    payload: dict[str, Any] | None = None,
    event_type: str = "ping",
    delivery_id: str = "delivery-001",
    signature: str = "",
    raw_body: bytes | None = None,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a handler context dict that mimics what the server provides."""
    body = payload or {}
    raw = raw_body if raw_body is not None else json.dumps(body).encode()
    headers = {
        "x-github-event": event_type,
        "x-github-delivery": delivery_id,
        "x-hub-signature-256": signature,
    }
    if extra_headers:
        headers.update(extra_headers)
    return {"headers": headers, "body": body, "raw_body": raw}


def _make_pr_payload(
    action: str = "opened",
    number: int = 42,
    title: str = "Fix the flux capacitor",
    body: str = "This PR fixes the flux capacitor.",
    repo_full_name: str = "octocat/Hello-World",
    login: str = "octocat",
    installation_id: int | None = 12345,
) -> dict[str, Any]:
    """Build a pull_request webhook payload."""
    payload: dict[str, Any] = {
        "action": action,
        "pull_request": {
            "number": number,
            "title": title,
            "body": body,
            "html_url": f"https://github.com/{repo_full_name}/pull/{number}",
            "user": {"login": login},
        },
        "repository": {"full_name": repo_full_name},
        "sender": {"login": login},
    }
    if installation_id is not None:
        payload["installation"] = {"id": installation_id}
    return payload


def _make_issue_payload(
    action: str = "opened",
    number: int = 7,
    title: str = "Bug: widget crashes",
    body: str = "Steps to reproduce ...",
    labels: list[str] | None = None,
    repo_full_name: str = "octocat/Hello-World",
    login: str = "contributor",
    installation_id: int | None = 12345,
) -> dict[str, Any]:
    """Build an issues webhook payload."""
    payload: dict[str, Any] = {
        "action": action,
        "issue": {
            "number": number,
            "title": title,
            "body": body,
            "html_url": f"https://github.com/{repo_full_name}/issues/{number}",
            "user": {"login": login},
            "labels": [{"name": l} for l in (labels or [])],
        },
        "repository": {"full_name": repo_full_name},
        "sender": {"login": login},
    }
    if installation_id is not None:
        payload["installation"] = {"id": installation_id}
    return payload


def _make_push_payload(
    ref: str = "refs/heads/main",
    commits_count: int = 3,
    repo_full_name: str = "octocat/Hello-World",
    pusher_name: str = "octocat",
) -> dict[str, Any]:
    """Build a push webhook payload."""
    commits = [{"id": f"abc{i}", "message": f"commit {i}"} for i in range(commits_count)]
    return {
        "ref": ref,
        "commits": commits,
        "repository": {"full_name": repo_full_name},
        "sender": {"login": pusher_name},
        "pusher": {"name": pusher_name},
    }


def _make_installation_payload(
    action: str = "created",
    installation_id: int = 99,
    account_login: str = "my-org",
    account_type: str = "Organization",
) -> dict[str, Any]:
    """Build an installation webhook payload."""
    return {
        "action": action,
        "installation": {
            "id": installation_id,
            "account": {"login": account_login, "type": account_type},
        },
        "sender": {"login": account_login},
    }


# ===========================================================================
# Enum tests
# ===========================================================================

class TestGitHubEventType:
    """Tests for the GitHubEventType enum."""

    def test_all_expected_values(self):
        values = {e.value for e in GitHubEventType}
        assert "pull_request" in values
        assert "pull_request_review" in values
        assert "issues" in values
        assert "issue_comment" in values
        assert "push" in values
        assert "installation" in values
        assert "ping" in values

    def test_string_enum_behaviour(self):
        assert GitHubEventType.PUSH == "push"
        assert str(GitHubEventType.PING) == "GitHubEventType.PING"

    def test_from_string_value(self):
        assert GitHubEventType("pull_request") is GitHubEventType.PULL_REQUEST

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            GitHubEventType("nonexistent_event")


class TestGitHubAction:
    """Tests for the GitHubAction enum."""

    def test_all_expected_values(self):
        values = {a.value for a in GitHubAction}
        expected = {"opened", "closed", "reopened", "synchronize", "created",
                    "deleted", "edited", "submitted"}
        assert values == expected

    def test_string_enum_behaviour(self):
        assert GitHubAction.OPENED == "opened"
        assert GitHubAction.SYNCHRONIZE == "synchronize"


# ===========================================================================
# GitHubWebhookEvent.from_request tests
# ===========================================================================

class TestGitHubWebhookEvent:
    """Tests for GitHubWebhookEvent dataclass construction."""

    def test_from_request_known_event_type(self):
        payload = _make_pr_payload()
        event = GitHubWebhookEvent.from_request("pull_request", "d-1", payload)
        assert event.event_type is GitHubEventType.PULL_REQUEST
        assert event.delivery_id == "d-1"
        assert event.action == "opened"
        assert event.repository == payload["repository"]
        assert event.sender == payload["sender"]
        assert event.installation_id == 12345

    def test_from_request_unknown_event_type_defaults_to_ping(self):
        event = GitHubWebhookEvent.from_request("some_unknown", "d-2", {})
        assert event.event_type is GitHubEventType.PING

    def test_from_request_no_action(self):
        event = GitHubWebhookEvent.from_request("push", "d-3", {"ref": "refs/heads/main"})
        assert event.action is None

    def test_from_request_no_installation(self):
        event = GitHubWebhookEvent.from_request("ping", "d-4", {})
        assert event.installation_id is None

    def test_from_request_empty_payload(self):
        event = GitHubWebhookEvent.from_request("ping", "d-5", {})
        assert event.repository == {}
        assert event.sender == {}
        assert event.payload == {}

    def test_received_at_is_iso_string(self):
        event = GitHubWebhookEvent.from_request("ping", "d-6", {})
        # Should be a valid ISO 8601 timestamp with UTC offset
        assert "T" in event.received_at


# ===========================================================================
# verify_signature tests
# ===========================================================================

class TestVerifySignature:
    """Tests for HMAC-SHA256 webhook signature verification."""

    def test_valid_signature(self):
        secret = "my-webhook-secret"
        payload = b'{"zen": "test"}'
        sig = _compute_signature(payload, secret)
        assert verify_signature(payload, sig, secret) is True

    def test_invalid_signature(self):
        payload = b'{"zen": "test"}'
        assert verify_signature(payload, "sha256=badhex", "secret") is False

    def test_missing_signature(self):
        assert verify_signature(b"body", "", "secret") is False

    def test_signature_without_prefix(self):
        # Missing the sha256= prefix
        mac = hmac_mod.new(b"secret", b"body", hashlib.sha256).hexdigest()
        assert verify_signature(b"body", mac, "secret") is False

    def test_wrong_secret(self):
        payload = b'{"data": 1}'
        sig = _compute_signature(payload, "correct-secret")
        assert verify_signature(payload, sig, "wrong-secret") is False

    def test_tampered_payload(self):
        secret = "s"
        original = b'{"ok": true}'
        sig = _compute_signature(original, secret)
        tampered = b'{"ok": false}'
        assert verify_signature(tampered, sig, secret) is False

    def test_empty_payload(self):
        secret = "s"
        payload = b""
        sig = _compute_signature(payload, secret)
        assert verify_signature(payload, sig, secret) is True

    def test_none_signature(self):
        assert verify_signature(b"x", None, "s") is False


# ===========================================================================
# Event handler registry tests
# ===========================================================================

class TestEventHandlerRegistry:
    """Tests for the register_event_handler decorator."""

    def test_all_event_types_registered(self):
        # The module registers handlers for PING, PULL_REQUEST, ISSUES, PUSH, INSTALLATION
        registered = set(_event_handlers.keys())
        assert GitHubEventType.PING in registered
        assert GitHubEventType.PULL_REQUEST in registered
        assert GitHubEventType.ISSUES in registered
        assert GitHubEventType.PUSH in registered
        assert GitHubEventType.INSTALLATION in registered

    def test_register_event_handler_decorator(self):
        """register_event_handler should add functions to the registry."""
        original_handlers = dict(_event_handlers)

        @register_event_handler(GitHubEventType.ISSUE_COMMENT)
        async def _test_handler(event):
            return {"test": True}

        assert _event_handlers[GitHubEventType.ISSUE_COMMENT] is _test_handler

        # Clean up
        if GitHubEventType.ISSUE_COMMENT not in original_handlers:
            del _event_handlers[GitHubEventType.ISSUE_COMMENT]


# ===========================================================================
# Individual event handler tests (called directly)
# ===========================================================================

class TestHandlePing:
    """Tests for the ping event handler."""

    @pytest.mark.asyncio
    async def test_ping_returns_pong(self):
        event = GitHubWebhookEvent.from_request("ping", "d-1", {"zen": "Keep it logically awesome."})
        result = await handle_ping(event)
        assert result["status"] == "ok"
        assert result["message"] == "pong"
        assert result["zen"] == "Keep it logically awesome."

    @pytest.mark.asyncio
    async def test_ping_without_zen(self):
        event = GitHubWebhookEvent.from_request("ping", "d-2", {})
        result = await handle_ping(event)
        assert result["zen"] is None


class TestHandlePullRequest:
    """Tests for pull_request event handler."""

    @pytest.mark.asyncio
    async def test_pr_opened_queues_debate(self):
        payload = _make_pr_payload(action="opened")
        event = GitHubWebhookEvent.from_request("pull_request", "d-1", payload)

        with patch(
            "aragora.server.handlers.webhooks.github_app.queue_code_review_debate",
            new_callable=AsyncMock,
            return_value="debate-123",
        ):
            result = await handle_pull_request(event)

        assert result["event"] == "pull_request"
        assert result["action"] == "opened"
        assert result["pr_number"] == 42
        assert result["pr_title"] == "Fix the flux capacitor"
        assert result["repository"] == "octocat/Hello-World"
        assert result["debate_queued"] is True
        assert result["debate_id"] == "debate-123"
        assert result["debate_type"] == "code_review"

    @pytest.mark.asyncio
    async def test_pr_synchronize_queues_debate(self):
        payload = _make_pr_payload(action="synchronize")
        event = GitHubWebhookEvent.from_request("pull_request", "d-2", payload)

        with patch(
            "aragora.server.handlers.webhooks.github_app.queue_code_review_debate",
            new_callable=AsyncMock,
            return_value="debate-456",
        ):
            result = await handle_pull_request(event)

        assert result["debate_queued"] is True
        assert result["debate_id"] == "debate-456"

    @pytest.mark.asyncio
    async def test_pr_opened_debate_queue_fails(self):
        payload = _make_pr_payload(action="opened")
        event = GitHubWebhookEvent.from_request("pull_request", "d-3", payload)

        with patch(
            "aragora.server.handlers.webhooks.github_app.queue_code_review_debate",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handle_pull_request(event)

        assert result["debate_queued"] is False
        assert "debate_id" not in result

    @pytest.mark.asyncio
    async def test_pr_closed_does_not_queue_debate(self):
        payload = _make_pr_payload(action="closed")
        event = GitHubWebhookEvent.from_request("pull_request", "d-4", payload)
        result = await handle_pull_request(event)

        assert result["action"] == "closed"
        assert "debate_queued" not in result
        assert "debate_id" not in result

    @pytest.mark.asyncio
    async def test_pr_edited_does_not_queue_debate(self):
        payload = _make_pr_payload(action="edited")
        event = GitHubWebhookEvent.from_request("pull_request", "d-5", payload)
        result = await handle_pull_request(event)
        assert "debate_queued" not in result

    @pytest.mark.asyncio
    async def test_pr_author_extracted(self):
        payload = _make_pr_payload(login="alice")
        event = GitHubWebhookEvent.from_request("pull_request", "d-6", payload)
        with patch(
            "aragora.server.handlers.webhooks.github_app.queue_code_review_debate",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handle_pull_request(event)
        assert result["author"] == "alice"

    @pytest.mark.asyncio
    async def test_pr_missing_pull_request_key(self):
        """Payload with no pull_request key should not crash."""
        payload = {"action": "opened", "repository": {"full_name": "r"}, "sender": {"login": "u"}}
        event = GitHubWebhookEvent.from_request("pull_request", "d-7", payload)
        with patch(
            "aragora.server.handlers.webhooks.github_app.queue_code_review_debate",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handle_pull_request(event)
        assert result["pr_number"] is None


class TestHandleIssues:
    """Tests for issues event handler."""

    @pytest.mark.asyncio
    async def test_issue_opened_queues_triage(self):
        payload = _make_issue_payload(action="opened")
        event = GitHubWebhookEvent.from_request("issues", "d-1", payload)

        with patch(
            "aragora.server.handlers.webhooks.github_app.queue_issue_triage_debate",
            new_callable=AsyncMock,
            return_value="triage-001",
        ):
            result = await handle_issues(event)

        assert result["event"] == "issues"
        assert result["action"] == "opened"
        assert result["issue_number"] == 7
        assert result["triage_queued"] is True
        assert result["debate_id"] == "triage-001"

    @pytest.mark.asyncio
    async def test_issue_opened_triage_fails(self):
        payload = _make_issue_payload(action="opened")
        event = GitHubWebhookEvent.from_request("issues", "d-2", payload)

        with patch(
            "aragora.server.handlers.webhooks.github_app.queue_issue_triage_debate",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handle_issues(event)

        assert result["triage_queued"] is False
        assert "debate_id" not in result

    @pytest.mark.asyncio
    async def test_issue_closed_does_not_queue_triage(self):
        payload = _make_issue_payload(action="closed")
        event = GitHubWebhookEvent.from_request("issues", "d-3", payload)
        result = await handle_issues(event)
        assert result["action"] == "closed"
        assert "triage_queued" not in result

    @pytest.mark.asyncio
    async def test_issue_reopened_does_not_queue_triage(self):
        payload = _make_issue_payload(action="reopened")
        event = GitHubWebhookEvent.from_request("issues", "d-4", payload)
        result = await handle_issues(event)
        assert "triage_queued" not in result

    @pytest.mark.asyncio
    async def test_issue_author_extracted(self):
        payload = _make_issue_payload(login="bob")
        event = GitHubWebhookEvent.from_request("issues", "d-5", payload)
        with patch(
            "aragora.server.handlers.webhooks.github_app.queue_issue_triage_debate",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handle_issues(event)
        assert result["author"] == "bob"

    @pytest.mark.asyncio
    async def test_issue_missing_issue_key(self):
        payload = {"action": "opened", "repository": {"full_name": "r"}, "sender": {"login": "u"}}
        event = GitHubWebhookEvent.from_request("issues", "d-6", payload)
        with patch(
            "aragora.server.handlers.webhooks.github_app.queue_issue_triage_debate",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handle_issues(event)
        assert result["issue_number"] is None


class TestHandlePush:
    """Tests for push event handler."""

    @pytest.mark.asyncio
    async def test_push_returns_expected_fields(self):
        payload = _make_push_payload(ref="refs/heads/main", commits_count=5, pusher_name="alice")
        event = GitHubWebhookEvent.from_request("push", "d-1", payload)
        result = await handle_push(event)

        assert result["event"] == "push"
        assert result["repository"] == "octocat/Hello-World"
        assert result["ref"] == "refs/heads/main"
        assert result["commits_count"] == 5
        assert result["pusher"] == "alice"

    @pytest.mark.asyncio
    async def test_push_zero_commits(self):
        payload = _make_push_payload(commits_count=0)
        event = GitHubWebhookEvent.from_request("push", "d-2", payload)
        result = await handle_push(event)
        assert result["commits_count"] == 0

    @pytest.mark.asyncio
    async def test_push_missing_pusher(self):
        payload = {"ref": "refs/heads/dev", "commits": [], "repository": {"full_name": "x/y"}, "sender": {"login": "z"}}
        event = GitHubWebhookEvent.from_request("push", "d-3", payload)
        result = await handle_push(event)
        assert result["pusher"] is None

    @pytest.mark.asyncio
    async def test_push_tag_ref(self):
        payload = _make_push_payload(ref="refs/tags/v1.0.0", commits_count=1)
        event = GitHubWebhookEvent.from_request("push", "d-4", payload)
        result = await handle_push(event)
        assert result["ref"] == "refs/tags/v1.0.0"


class TestHandleInstallation:
    """Tests for installation event handler."""

    @pytest.mark.asyncio
    async def test_installation_created(self):
        payload = _make_installation_payload(action="created", installation_id=99, account_login="my-org")
        event = GitHubWebhookEvent.from_request("installation", "d-1", payload)
        result = await handle_installation(event)

        assert result["event"] == "installation"
        assert result["action"] == "created"
        assert result["installation_id"] == 99
        assert result["account"] == "my-org"
        assert result["account_type"] == "Organization"

    @pytest.mark.asyncio
    async def test_installation_deleted(self):
        payload = _make_installation_payload(action="deleted")
        event = GitHubWebhookEvent.from_request("installation", "d-2", payload)
        result = await handle_installation(event)
        assert result["action"] == "deleted"

    @pytest.mark.asyncio
    async def test_installation_missing_account(self):
        payload = {"action": "created", "installation": {"id": 1}, "sender": {"login": "x"}}
        event = GitHubWebhookEvent.from_request("installation", "d-3", payload)
        result = await handle_installation(event)
        assert result["account"] is None
        assert result["account_type"] is None


# ===========================================================================
# handle_github_webhook (main dispatcher) tests
# ===========================================================================

class TestHandleGitHubWebhook:
    """Tests for the main webhook dispatcher."""

    @pytest.mark.asyncio
    async def test_ping_in_dev_mode_no_secret(self, monkeypatch):
        """Without a secret in dev mode, verification is skipped."""
        monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "development")

        ctx = _make_ctx(payload={"zen": "Hello!"}, event_type="ping")
        result = await handle_github_webhook(ctx)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["success"] is True
        assert body["message"] == "pong"

    @pytest.mark.asyncio
    async def test_ping_with_valid_signature(self, monkeypatch):
        """With a configured secret, valid signature is accepted."""
        secret = "webhook-secret-123"
        monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", secret)

        payload = {"zen": "Valid!"}
        raw = json.dumps(payload).encode()
        sig = _compute_signature(raw, secret)

        ctx = _make_ctx(payload=payload, event_type="ping", signature=sig, raw_body=raw)
        result = await handle_github_webhook(ctx)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["success"] is True
        assert body["message"] == "pong"

    @pytest.mark.asyncio
    async def test_invalid_signature_returns_401(self, monkeypatch):
        secret = "real-secret"
        monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", secret)

        ctx = _make_ctx(
            payload={"zen": "test"},
            event_type="ping",
            signature="sha256=bad",
            raw_body=b'{"zen": "test"}',
        )
        result = await handle_github_webhook(ctx)
        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_secret_in_production_returns_503(self, monkeypatch):
        monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "production")

        ctx = _make_ctx(event_type="ping")
        result = await handle_github_webhook(ctx)
        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_missing_secret_staging_returns_503(self, monkeypatch):
        monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "staging")

        ctx = _make_ctx(event_type="ping")
        result = await handle_github_webhook(ctx)
        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_missing_secret_in_test_env_skips_verification(self, monkeypatch):
        monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "test")

        ctx = _make_ctx(payload={"zen": "test-env"}, event_type="ping")
        result = await handle_github_webhook(ctx)
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_missing_secret_in_local_env_skips_verification(self, monkeypatch):
        monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "local")

        ctx = _make_ctx(payload={"zen": "local-env"}, event_type="ping")
        result = await handle_github_webhook(ctx)
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_missing_secret_in_dev_env_skips_verification(self, monkeypatch):
        monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "dev")

        ctx = _make_ctx(payload={"zen": "dev-env"}, event_type="ping")
        result = await handle_github_webhook(ctx)
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_unknown_event_acknowledged_but_not_handled(self, monkeypatch):
        """An event type that maps to a GitHubEventType without a registered handler."""
        monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "test")

        # pull_request_review is a valid GitHubEventType but has no registered handler
        ctx = _make_ctx(event_type="pull_request_review", payload={"action": "submitted"})
        result = await handle_github_webhook(ctx)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["success"] is True
        assert body["handled"] is False
        assert "not handled" in body["message"]

    @pytest.mark.asyncio
    async def test_completely_unknown_event_treated_as_ping(self, monkeypatch):
        """A completely unknown event type (not in enum) gets mapped to PING."""
        monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "test")

        ctx = _make_ctx(event_type="star", payload={})
        result = await handle_github_webhook(ctx)

        # Unknown events default to PING via from_request, so ping handler fires
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["success"] is True
        assert body["message"] == "pong"

    @pytest.mark.asyncio
    async def test_delivery_id_in_response(self, monkeypatch):
        monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "test")

        ctx = _make_ctx(event_type="ping", delivery_id="abc-xyz-123", payload={})
        result = await handle_github_webhook(ctx)

        body = json.loads(result.body)
        assert body["delivery_id"] == "abc-xyz-123"

    @pytest.mark.asyncio
    async def test_event_type_in_response(self, monkeypatch):
        monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "test")

        ctx = _make_ctx(event_type="push", payload=_make_push_payload())
        result = await handle_github_webhook(ctx)

        body = json.loads(result.body)
        assert body["event_type"] == "push"

    @pytest.mark.asyncio
    async def test_pull_request_event_dispatched(self, monkeypatch):
        monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "test")

        payload = _make_pr_payload(action="closed")
        ctx = _make_ctx(event_type="pull_request", payload=payload)
        result = await handle_github_webhook(ctx)

        body = json.loads(result.body)
        assert body["success"] is True
        assert body["event"] == "pull_request"
        assert body["action"] == "closed"

    @pytest.mark.asyncio
    async def test_issues_event_dispatched(self, monkeypatch):
        monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "test")

        payload = _make_issue_payload(action="closed")
        ctx = _make_ctx(event_type="issues", payload=payload)
        result = await handle_github_webhook(ctx)

        body = json.loads(result.body)
        assert body["success"] is True
        assert body["event"] == "issues"

    @pytest.mark.asyncio
    async def test_installation_event_dispatched(self, monkeypatch):
        monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "test")

        payload = _make_installation_payload()
        ctx = _make_ctx(event_type="installation", payload=payload)
        result = await handle_github_webhook(ctx)

        body = json.loads(result.body)
        assert body["success"] is True
        assert body["event"] == "installation"

    @pytest.mark.asyncio
    async def test_handler_error_returns_500(self, monkeypatch):
        """If an event handler raises, 500 is returned."""
        monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "test")

        original = _event_handlers.get(GitHubEventType.PING)
        try:
            async def _bad_handler(event):
                raise RuntimeError("boom")

            _event_handlers[GitHubEventType.PING] = _bad_handler

            ctx = _make_ctx(event_type="ping", payload={})
            result = await handle_github_webhook(ctx)
            assert result.status_code == 500
        finally:
            if original:
                _event_handlers[GitHubEventType.PING] = original

    @pytest.mark.asyncio
    async def test_handler_value_error_returns_500(self, monkeypatch):
        monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "test")

        original = _event_handlers.get(GitHubEventType.PING)
        try:
            async def _bad_handler(event):
                raise ValueError("invalid data")

            _event_handlers[GitHubEventType.PING] = _bad_handler

            ctx = _make_ctx(event_type="ping", payload={})
            result = await handle_github_webhook(ctx)
            assert result.status_code == 500
        finally:
            if original:
                _event_handlers[GitHubEventType.PING] = original

    @pytest.mark.asyncio
    async def test_missing_headers_key(self, monkeypatch):
        """ctx with no headers key should still work (uses defaults)."""
        monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "test")

        ctx = {"body": {}, "raw_body": b"{}"}
        result = await handle_github_webhook(ctx)
        # Empty event_type -> parsed as unknown -> unhandled
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_default_delivery_id(self, monkeypatch):
        """When x-github-delivery is missing, defaults to 'unknown'."""
        monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "test")

        ctx = {"headers": {"x-github-event": "ping"}, "body": {}, "raw_body": b"{}"}
        result = await handle_github_webhook(ctx)
        body = json.loads(result.body)
        assert body["delivery_id"] == "unknown"


# ===========================================================================
# handle_github_app_status tests
# ===========================================================================

class TestHandleGitHubAppStatus:
    """Tests for the status endpoint."""

    @pytest.mark.asyncio
    async def test_status_configured(self, monkeypatch):
        monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "some-secret")
        result = await handle_github_app_status({})

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "configured"
        assert body["webhook_endpoint"] == "/api/v1/webhooks/github"
        assert "pull_request" in body["supported_events"]
        assert body["features"]["pr_auto_review"] is True
        assert body["features"]["issue_triage"] is True
        assert body["features"]["push_tracking"] is True
        assert body["features"]["installation_tracking"] is True

    @pytest.mark.asyncio
    async def test_status_unconfigured(self, monkeypatch):
        monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
        result = await handle_github_app_status({})

        body = json.loads(result.body)
        assert body["status"] == "unconfigured"

    @pytest.mark.asyncio
    async def test_status_supported_events_complete(self, monkeypatch):
        monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
        result = await handle_github_app_status({})

        body = json.loads(result.body)
        expected = [e.value for e in GitHubEventType]
        assert body["supported_events"] == expected


# ===========================================================================
# queue_code_review_debate tests
# ===========================================================================

def _mock_decision_modules():
    """Create mocked decision module objects for patching internal imports."""
    mock_decision = MagicMock()
    mock_decision.DecisionRequest = MagicMock()
    mock_decision.DecisionType = MagicMock()
    mock_decision.DecisionType.DEBATE = "debate"
    mock_decision.InputSource = MagicMock()
    mock_decision.InputSource.GITHUB = "github"
    mock_decision.ResponseChannel = MagicMock()
    mock_decision.RequestContext = MagicMock()
    return mock_decision


class TestQueueCodeReviewDebate:
    """Tests for queue_code_review_debate helper."""

    @pytest.mark.asyncio
    async def test_returns_none_on_import_error(self):
        """When decision module is missing, returns None gracefully."""
        # Patch at the point of use inside the function: the local import statement
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def failing_import(name, *args, **kwargs):
            if name == "aragora.core.decision":
                raise ImportError("no module")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=failing_import):
            result = await queue_code_review_debate({}, {}, None)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_router_failure(self):
        """When router.route returns a non-success result, returns None."""
        mock_decision = _mock_decision_modules()
        mock_result = MagicMock()
        mock_result.success = False
        mock_router = MagicMock()
        mock_router.route = AsyncMock(return_value=mock_result)
        mock_decision.get_decision_router = MagicMock(return_value=mock_router)

        mock_origin = MagicMock()
        mock_origin.register_debate_origin = MagicMock()

        with patch.dict("sys.modules", {
            "aragora.core.decision": mock_decision,
            "aragora.server.debate_origin": mock_origin,
        }):
            result = await queue_code_review_debate(
                {"number": 1, "title": "T", "html_url": "u", "body": "b", "user": {"login": "l"}},
                {"full_name": "o/r"},
                123,
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_successful_debate_queue(self):
        """When the decision router succeeds, returns a debate ID string."""
        mock_decision = _mock_decision_modules()
        mock_result = MagicMock()
        mock_result.success = True
        mock_router = MagicMock()
        mock_router.route = AsyncMock(return_value=mock_result)
        mock_decision.get_decision_router = MagicMock(return_value=mock_router)

        mock_origin = MagicMock()
        mock_origin.register_debate_origin = MagicMock()

        with patch.dict("sys.modules", {
            "aragora.core.decision": mock_decision,
            "aragora.server.debate_origin": mock_origin,
        }):
            result = await queue_code_review_debate(
                {"number": 1, "title": "T", "html_url": "u", "body": "b", "user": {"login": "l"}},
                {"full_name": "o/r"},
                123,
            )
        assert result is not None
        # Result should be a UUID string
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_pr_body_none_handled(self):
        """PR body can be None without crashing."""
        mock_decision = _mock_decision_modules()
        mock_result = MagicMock()
        mock_result.success = True
        mock_router = MagicMock()
        mock_router.route = AsyncMock(return_value=mock_result)
        mock_decision.get_decision_router = MagicMock(return_value=mock_router)

        mock_origin = MagicMock()
        mock_origin.register_debate_origin = MagicMock()

        with patch.dict("sys.modules", {
            "aragora.core.decision": mock_decision,
            "aragora.server.debate_origin": mock_origin,
        }):
            result = await queue_code_review_debate(
                {"number": 1, "title": "T", "html_url": "u", "body": None, "user": {"login": "l"}},
                {"full_name": "o/r"},
                None,
            )
        # Should not crash; should succeed since router returns success
        assert result is not None

    @pytest.mark.asyncio
    async def test_runtime_error_returns_none(self):
        """RuntimeError in queue function returns None."""
        mock_decision = _mock_decision_modules()
        mock_decision.get_decision_router = MagicMock(side_effect=RuntimeError("broken"))

        mock_origin = MagicMock()
        mock_origin.register_debate_origin = MagicMock()

        with patch.dict("sys.modules", {
            "aragora.core.decision": mock_decision,
            "aragora.server.debate_origin": mock_origin,
        }):
            result = await queue_code_review_debate(
                {"number": 1, "title": "T", "html_url": "u", "body": "b", "user": {"login": "l"}},
                {"full_name": "o/r"},
                123,
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_registers_debate_origin(self):
        """Verifies that register_debate_origin is called with correct args."""
        mock_decision = _mock_decision_modules()
        mock_result = MagicMock()
        mock_result.success = True
        mock_router = MagicMock()
        mock_router.route = AsyncMock(return_value=mock_result)
        mock_decision.get_decision_router = MagicMock(return_value=mock_router)

        mock_origin = MagicMock()
        mock_origin.register_debate_origin = MagicMock()

        with patch.dict("sys.modules", {
            "aragora.core.decision": mock_decision,
            "aragora.server.debate_origin": mock_origin,
        }):
            await queue_code_review_debate(
                {"number": 42, "title": "Fix", "html_url": "u", "body": "b", "user": {"login": "alice"}},
                {"full_name": "octocat/Hello-World"},
                999,
            )

        mock_origin.register_debate_origin.assert_called_once()
        call_kwargs = mock_origin.register_debate_origin.call_args
        assert call_kwargs[1]["platform"] == "github"
        assert "octocat/Hello-World/pull/42" in call_kwargs[1]["channel_id"]


class TestQueueIssueTriageDebate:
    """Tests for queue_issue_triage_debate helper."""

    @pytest.mark.asyncio
    async def test_returns_none_on_import_error(self):
        """When decision module is missing, returns None gracefully."""
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def failing_import(name, *args, **kwargs):
            if name == "aragora.core.decision":
                raise ImportError("no module")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=failing_import):
            result = await queue_issue_triage_debate({}, {}, None)
        assert result is None

    @pytest.mark.asyncio
    async def test_successful_triage_queue(self):
        """When the decision router succeeds, returns a debate ID string."""
        mock_decision = _mock_decision_modules()
        mock_result = MagicMock()
        mock_result.success = True
        mock_router = MagicMock()
        mock_router.route = AsyncMock(return_value=mock_result)
        mock_decision.get_decision_router = MagicMock(return_value=mock_router)

        mock_origin = MagicMock()
        mock_origin.register_debate_origin = MagicMock()

        with patch.dict("sys.modules", {
            "aragora.core.decision": mock_decision,
            "aragora.server.debate_origin": mock_origin,
        }):
            result = await queue_issue_triage_debate(
                {"number": 5, "title": "Bug", "html_url": "u", "body": "b", "user": {"login": "c"}, "labels": []},
                {"full_name": "o/r"},
                456,
            )
        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_issue_body_none_handled(self):
        """Issue body can be None without crashing."""
        mock_decision = _mock_decision_modules()
        mock_result = MagicMock()
        mock_result.success = True
        mock_router = MagicMock()
        mock_router.route = AsyncMock(return_value=mock_result)
        mock_decision.get_decision_router = MagicMock(return_value=mock_router)

        mock_origin = MagicMock()
        mock_origin.register_debate_origin = MagicMock()

        with patch.dict("sys.modules", {
            "aragora.core.decision": mock_decision,
            "aragora.server.debate_origin": mock_origin,
        }):
            result = await queue_issue_triage_debate(
                {"number": 1, "title": "T", "html_url": "u", "body": None, "user": {"login": "l"}, "labels": []},
                {"full_name": "o/r"},
                None,
            )
        assert result is not None

    @pytest.mark.asyncio
    async def test_issue_with_labels(self):
        """Labels are extracted from issue payload."""
        mock_decision = _mock_decision_modules()
        mock_result = MagicMock()
        mock_result.success = True
        mock_router = MagicMock()
        mock_router.route = AsyncMock(return_value=mock_result)
        mock_decision.get_decision_router = MagicMock(return_value=mock_router)

        mock_origin = MagicMock()
        mock_origin.register_debate_origin = MagicMock()

        with patch.dict("sys.modules", {
            "aragora.core.decision": mock_decision,
            "aragora.server.debate_origin": mock_origin,
        }):
            result = await queue_issue_triage_debate(
                {
                    "number": 2,
                    "title": "Feature",
                    "html_url": "u",
                    "body": "Add feature",
                    "user": {"login": "l"},
                    "labels": [{"name": "bug"}, {"name": "urgent"}],
                },
                {"full_name": "o/r"},
                789,
            )
        assert result is not None

        # Verify origin was registered with labels in metadata
        call_kwargs = mock_origin.register_debate_origin.call_args[1]
        assert "bug" in call_kwargs["metadata"]["labels"]
        assert "urgent" in call_kwargs["metadata"]["labels"]

    @pytest.mark.asyncio
    async def test_router_failure_returns_none(self):
        """When router returns non-success, returns None."""
        mock_decision = _mock_decision_modules()
        mock_result = MagicMock()
        mock_result.success = False
        mock_router = MagicMock()
        mock_router.route = AsyncMock(return_value=mock_result)
        mock_decision.get_decision_router = MagicMock(return_value=mock_router)

        mock_origin = MagicMock()
        mock_origin.register_debate_origin = MagicMock()

        with patch.dict("sys.modules", {
            "aragora.core.decision": mock_decision,
            "aragora.server.debate_origin": mock_origin,
        }):
            result = await queue_issue_triage_debate(
                {"number": 1, "title": "T", "html_url": "u", "body": "b", "user": {"login": "l"}, "labels": []},
                {"full_name": "o/r"},
                123,
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_runtime_error_returns_none(self):
        """RuntimeError in queue function returns None."""
        mock_decision = _mock_decision_modules()
        mock_decision.get_decision_router = MagicMock(side_effect=RuntimeError("broken"))

        mock_origin = MagicMock()
        mock_origin.register_debate_origin = MagicMock()

        with patch.dict("sys.modules", {
            "aragora.core.decision": mock_decision,
            "aragora.server.debate_origin": mock_origin,
        }):
            result = await queue_issue_triage_debate(
                {"number": 1, "title": "T", "html_url": "u", "body": "b", "user": {"login": "l"}, "labels": []},
                {"full_name": "o/r"},
                123,
            )
        assert result is None


# ===========================================================================
# Route definitions
# ===========================================================================

class TestRouteDefinitions:
    """Tests for the GITHUB_APP_ROUTES constant."""

    def test_routes_is_list(self):
        assert isinstance(GITHUB_APP_ROUTES, list)

    def test_routes_count(self):
        assert len(GITHUB_APP_ROUTES) == 2

    def test_post_webhook_route_exists(self):
        methods = [(m, p) for m, p, _ in GITHUB_APP_ROUTES]
        assert ("POST", "/api/v1/webhooks/github") in methods

    def test_get_status_route_exists(self):
        methods = [(m, p) for m, p, _ in GITHUB_APP_ROUTES]
        assert ("GET", "/api/v1/webhooks/github/status") in methods

    def test_post_webhook_handler_is_handle_github_webhook(self):
        for method, path, handler in GITHUB_APP_ROUTES:
            if method == "POST" and path == "/api/v1/webhooks/github":
                assert handler is handle_github_webhook

    def test_get_status_handler_is_handle_github_app_status(self):
        for method, path, handler in GITHUB_APP_ROUTES:
            if method == "GET" and path == "/api/v1/webhooks/github/status":
                assert handler is handle_github_app_status


# ===========================================================================
# Edge cases and security
# ===========================================================================

class TestSecurityEdgeCases:
    """Additional security and edge-case tests."""

    @pytest.mark.asyncio
    async def test_signature_verified_before_dispatch(self, monkeypatch):
        """Even a valid event_type is rejected if signature is bad."""
        monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "s")
        ctx = _make_ctx(
            event_type="push",
            payload=_make_push_payload(),
            signature="sha256=wrong",
            raw_body=json.dumps(_make_push_payload()).encode(),
        )
        result = await handle_github_webhook(ctx)
        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_empty_event_type_header(self, monkeypatch):
        monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "test")

        ctx = _make_ctx(event_type="", payload={})
        result = await handle_github_webhook(ctx)
        # Empty string is unknown -> from_request defaults to PING -> ping handler fires
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["success"] is True
        assert body["message"] == "pong"

    @pytest.mark.asyncio
    async def test_no_aragora_env_and_no_secret_returns_503(self, monkeypatch):
        """Without ARAGORA_ENV set and no secret, production default -> 503."""
        monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
        monkeypatch.delenv("ARAGORA_ENV", raising=False)

        ctx = _make_ctx(event_type="ping")
        result = await handle_github_webhook(ctx)
        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_empty_raw_body_with_valid_secret(self, monkeypatch):
        secret = "s"
        monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", secret)

        raw = b""
        sig = _compute_signature(raw, secret)
        ctx = _make_ctx(event_type="ping", payload={}, signature=sig, raw_body=raw)
        result = await handle_github_webhook(ctx)
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_case_insensitive_env_check(self, monkeypatch):
        """ARAGORA_ENV comparison is case insensitive (via .lower())."""
        monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "Development")  # Capital D

        ctx = _make_ctx(event_type="ping", payload={})
        result = await handle_github_webhook(ctx)
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_large_payload_accepted(self, monkeypatch):
        monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "test")

        # Use action="edited" to avoid triggering the real queue_issue_triage_debate
        large_payload = {"data": "x" * 50_000, "action": "edited",
                         "issue": {"number": 1, "title": "big", "user": {"login": "u"}},
                         "repository": {"full_name": "r"}, "sender": {"login": "u"}}
        ctx = _make_ctx(event_type="issues", payload=large_payload)
        result = await handle_github_webhook(ctx)
        assert result.status_code == 200


class TestSignatureEdgeCases:
    """Detailed signature verification edge cases."""

    def test_signature_with_uppercase_sha256_prefix(self):
        # GitHub always uses lowercase, but our function should reject uppercase
        payload = b"test"
        secret = "s"
        mac = hmac_mod.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        assert verify_signature(payload, f"SHA256={mac}", secret) is False

    def test_signature_with_extra_prefix(self):
        assert verify_signature(b"test", "sha256=sha256=abc", "s") is False

    def test_signature_with_only_prefix(self):
        assert verify_signature(b"test", "sha256=", "s") is False

    def test_unicode_payload(self):
        secret = "s"
        payload = json.dumps({"emoji": "hello"}).encode("utf-8")
        sig = _compute_signature(payload, secret)
        assert verify_signature(payload, sig, secret) is True

    def test_binary_payload(self):
        secret = "key"
        payload = bytes(range(256))
        sig = _compute_signature(payload, secret)
        assert verify_signature(payload, sig, secret) is True


# ===========================================================================
# Integration-style: full flow with signature
# ===========================================================================

class TestFullWebhookFlow:
    """End-to-end flow tests with proper signature verification."""

    @pytest.mark.asyncio
    async def test_push_event_full_flow(self, monkeypatch):
        secret = "integration-secret"
        monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", secret)

        payload = _make_push_payload(ref="refs/heads/feature", commits_count=2, pusher_name="dev")
        raw = json.dumps(payload).encode()
        sig = _compute_signature(raw, secret)

        ctx = _make_ctx(
            event_type="push",
            delivery_id="int-001",
            payload=payload,
            signature=sig,
            raw_body=raw,
        )
        result = await handle_github_webhook(ctx)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["success"] is True
        assert body["delivery_id"] == "int-001"
        assert body["event_type"] == "push"
        assert body["ref"] == "refs/heads/feature"
        assert body["commits_count"] == 2
        assert body["pusher"] == "dev"

    @pytest.mark.asyncio
    async def test_installation_event_full_flow(self, monkeypatch):
        secret = "inst-secret"
        monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", secret)

        payload = _make_installation_payload(action="created", installation_id=77, account_login="acme")
        raw = json.dumps(payload).encode()
        sig = _compute_signature(raw, secret)

        ctx = _make_ctx(
            event_type="installation",
            delivery_id="int-002",
            payload=payload,
            signature=sig,
            raw_body=raw,
        )
        result = await handle_github_webhook(ctx)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["success"] is True
        assert body["event"] == "installation"
        assert body["installation_id"] == 77
        assert body["account"] == "acme"

    @pytest.mark.asyncio
    async def test_pr_closed_full_flow(self, monkeypatch):
        secret = "pr-secret"
        monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", secret)

        payload = _make_pr_payload(action="closed", number=99, title="Done")
        raw = json.dumps(payload).encode()
        sig = _compute_signature(raw, secret)

        ctx = _make_ctx(
            event_type="pull_request",
            delivery_id="int-003",
            payload=payload,
            signature=sig,
            raw_body=raw,
        )
        result = await handle_github_webhook(ctx)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["pr_number"] == 99
        assert body["action"] == "closed"

    @pytest.mark.asyncio
    async def test_issue_closed_full_flow(self, monkeypatch):
        secret = "issue-secret"
        monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", secret)

        payload = _make_issue_payload(action="closed", number=15)
        raw = json.dumps(payload).encode()
        sig = _compute_signature(raw, secret)

        ctx = _make_ctx(
            event_type="issues",
            delivery_id="int-004",
            payload=payload,
            signature=sig,
            raw_body=raw,
        )
        result = await handle_github_webhook(ctx)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["issue_number"] == 15
        assert body["action"] == "closed"

    @pytest.mark.asyncio
    async def test_unhandled_event_with_signature(self, monkeypatch):
        """An unregistered event type with valid signature is acknowledged but not handled."""
        secret = "sig"
        monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", secret)

        # issue_comment is a valid GitHubEventType but has no registered handler
        payload = {"action": "created"}
        raw = json.dumps(payload).encode()
        sig = _compute_signature(raw, secret)

        ctx = _make_ctx(
            event_type="issue_comment",
            payload=payload,
            signature=sig,
            raw_body=raw,
        )
        result = await handle_github_webhook(ctx)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["handled"] is False
        assert body["success"] is True
