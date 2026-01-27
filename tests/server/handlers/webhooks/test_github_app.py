"""
Tests for GitHub App webhook handler.

Tests signature verification, event handling, and security features.
"""

import hashlib
import hmac
import json
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.webhooks.github_app import (
    GitHubAction,
    GitHubEventType,
    GitHubWebhookEvent,
    handle_github_webhook,
    handle_ping,
    handle_pull_request,
    handle_issues,
    handle_push,
    handle_installation,
    verify_signature,
)


class TestVerifySignature:
    """Tests for HMAC signature verification."""

    def test_valid_signature(self):
        """Valid signature should verify successfully."""
        secret = "test-secret-123"
        payload = b'{"action": "opened"}'
        computed = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        signature = f"sha256={computed}"

        assert verify_signature(payload, signature, secret) is True

    def test_invalid_signature(self):
        """Invalid signature should fail verification."""
        secret = "test-secret-123"
        payload = b'{"action": "opened"}'
        signature = "sha256=invalid_signature_value"

        assert verify_signature(payload, signature, secret) is False

    def test_missing_signature(self):
        """Missing signature should fail verification."""
        assert verify_signature(b"payload", "", "secret") is False
        assert verify_signature(b"payload", None, "secret") is False

    def test_wrong_prefix(self):
        """Signature without sha256= prefix should fail."""
        assert verify_signature(b"payload", "sha1=abc123", "secret") is False

    def test_tampered_payload(self):
        """Tampered payload should fail verification."""
        secret = "test-secret-123"
        original = b'{"action": "opened"}'
        tampered = b'{"action": "closed"}'
        computed = hmac.new(secret.encode(), original, hashlib.sha256).hexdigest()
        signature = f"sha256={computed}"

        # Signature was for original, should fail for tampered
        assert verify_signature(tampered, signature, secret) is False


class TestGitHubWebhookEvent:
    """Tests for GitHubWebhookEvent parsing."""

    def test_from_request_pull_request(self):
        """Parse pull request event."""
        event = GitHubWebhookEvent.from_request(
            event_type="pull_request",
            delivery_id="test-delivery-123",
            payload={
                "action": "opened",
                "installation": {"id": 12345},
                "repository": {"full_name": "owner/repo"},
                "sender": {"login": "testuser"},
                "pull_request": {"number": 42},
            },
        )

        assert event.event_type == GitHubEventType.PULL_REQUEST
        assert event.action == "opened"
        assert event.delivery_id == "test-delivery-123"
        assert event.installation_id == 12345
        assert event.repository["full_name"] == "owner/repo"
        assert event.sender["login"] == "testuser"

    def test_from_request_unknown_type(self):
        """Unknown event type defaults to PING."""
        event = GitHubWebhookEvent.from_request(
            event_type="unknown_event",
            delivery_id="test-123",
            payload={},
        )

        assert event.event_type == GitHubEventType.PING

    def test_from_request_missing_fields(self):
        """Handles missing optional fields gracefully."""
        event = GitHubWebhookEvent.from_request(
            event_type="issues",
            delivery_id="test-456",
            payload={"action": "opened"},
        )

        assert event.event_type == GitHubEventType.ISSUES
        assert event.installation_id is None
        assert event.repository == {}
        assert event.sender == {}


class TestEventHandlers:
    """Tests for individual event handlers."""

    @pytest.mark.asyncio
    async def test_handle_ping(self):
        """Ping handler returns pong with zen."""
        event = GitHubWebhookEvent.from_request(
            event_type="ping",
            delivery_id="ping-123",
            payload={"zen": "Speak like a human."},
        )

        result = await handle_ping(event)

        assert result["status"] == "ok"
        assert result["message"] == "pong"
        assert result["zen"] == "Speak like a human."

    @pytest.mark.asyncio
    async def test_handle_pull_request_opened(self):
        """PR opened triggers debate queue."""
        event = GitHubWebhookEvent.from_request(
            event_type="pull_request",
            delivery_id="pr-123",
            payload={
                "action": "opened",
                "pull_request": {
                    "number": 42,
                    "title": "Add new feature",
                    "user": {"login": "author"},
                },
                "repository": {"full_name": "owner/repo"},
                "sender": {"login": "author"},
            },
        )

        result = await handle_pull_request(event)

        assert result["event"] == "pull_request"
        assert result["action"] == "opened"
        assert result["pr_number"] == 42
        assert result["debate_queued"] is True
        assert result["debate_type"] == "code_review"

    @pytest.mark.asyncio
    async def test_handle_pull_request_synchronize(self):
        """PR synchronize (new commits) triggers debate queue."""
        event = GitHubWebhookEvent.from_request(
            event_type="pull_request",
            delivery_id="pr-456",
            payload={
                "action": "synchronize",
                "pull_request": {"number": 42, "title": "Update", "user": {}},
                "repository": {"full_name": "owner/repo"},
                "sender": {"login": "author"},
            },
        )

        result = await handle_pull_request(event)

        assert result["debate_queued"] is True

    @pytest.mark.asyncio
    async def test_handle_pull_request_closed(self):
        """PR closed does not queue debate."""
        event = GitHubWebhookEvent.from_request(
            event_type="pull_request",
            delivery_id="pr-789",
            payload={
                "action": "closed",
                "pull_request": {"number": 42, "title": "Done", "user": {}},
                "repository": {"full_name": "owner/repo"},
                "sender": {"login": "author"},
            },
        )

        result = await handle_pull_request(event)

        assert "debate_queued" not in result

    @pytest.mark.asyncio
    async def test_handle_issues_opened(self):
        """Issue opened triggers triage queue."""
        event = GitHubWebhookEvent.from_request(
            event_type="issues",
            delivery_id="issue-123",
            payload={
                "action": "opened",
                "issue": {
                    "number": 99,
                    "title": "Bug report",
                    "user": {"login": "reporter"},
                },
                "repository": {"full_name": "owner/repo"},
                "sender": {"login": "reporter"},
            },
        )

        result = await handle_issues(event)

        assert result["event"] == "issues"
        assert result["issue_number"] == 99
        assert result["triage_queued"] is True

    @pytest.mark.asyncio
    async def test_handle_push(self):
        """Push event tracks commits."""
        event = GitHubWebhookEvent.from_request(
            event_type="push",
            delivery_id="push-123",
            payload={
                "ref": "refs/heads/main",
                "commits": [{"id": "abc"}, {"id": "def"}],
                "repository": {"full_name": "owner/repo"},
                "pusher": {"name": "developer"},
                "sender": {"login": "developer"},
            },
        )

        result = await handle_push(event)

        assert result["event"] == "push"
        assert result["ref"] == "refs/heads/main"
        assert result["commits_count"] == 2
        assert result["pusher"] == "developer"

    @pytest.mark.asyncio
    async def test_handle_installation(self):
        """Installation event tracks app installs."""
        event = GitHubWebhookEvent.from_request(
            event_type="installation",
            delivery_id="install-123",
            payload={
                "action": "created",
                "installation": {
                    "id": 12345,
                    "account": {"login": "myorg", "type": "Organization"},
                },
                "sender": {"login": "admin"},
            },
        )

        result = await handle_installation(event)

        assert result["event"] == "installation"
        assert result["action"] == "created"
        assert result["installation_id"] == 12345
        assert result["account"] == "myorg"


class TestHandleGitHubWebhook:
    """Integration tests for the main webhook handler."""

    def create_mock_context(
        self,
        event_type: str = "ping",
        payload: dict = None,
        secret: str = None,
    ) -> MagicMock:
        """Create a mock server context for testing."""
        ctx = MagicMock()
        ctx.headers = {
            "x-github-event": event_type,
            "x-github-delivery": "test-delivery-id",
        }

        payload = payload or {"zen": "Test zen"}
        raw_body = json.dumps(payload).encode()
        ctx.body = payload
        ctx.raw_body = raw_body

        if secret:
            computed = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
            ctx.headers["x-hub-signature-256"] = f"sha256={computed}"

        return ctx

    @pytest.mark.asyncio
    async def test_ping_event_success(self):
        """Ping event is handled successfully."""
        ctx = self.create_mock_context(
            event_type="ping",
            payload={"zen": "Speak like a human."},
        )

        with patch.dict("os.environ", {"GITHUB_WEBHOOK_SECRET": ""}, clear=False):
            result = await handle_github_webhook(ctx)

        body = json.loads(result.body.decode("utf-8"))
        assert body["success"] is True
        assert body["event_type"] == "ping"
        assert body["message"] == "pong"

    @pytest.mark.asyncio
    async def test_valid_signature_passes(self):
        """Valid signature allows request through."""
        secret = "my-webhook-secret"
        ctx = self.create_mock_context(
            event_type="ping",
            payload={"zen": "Test"},
            secret=secret,
        )

        with patch.dict("os.environ", {"GITHUB_WEBHOOK_SECRET": secret}, clear=False):
            result = await handle_github_webhook(ctx)

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_invalid_signature_rejected(self):
        """Invalid signature returns 401."""
        ctx = self.create_mock_context(event_type="ping")
        ctx.headers["x-hub-signature-256"] = "sha256=invalid"

        with patch.dict("os.environ", {"GITHUB_WEBHOOK_SECRET": "real-secret"}, clear=False):
            result = await handle_github_webhook(ctx)

        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_unhandled_event_acknowledged(self):
        """Unknown event types are acknowledged but not processed."""
        ctx = self.create_mock_context(event_type="unknown_event")

        with patch.dict("os.environ", {"GITHUB_WEBHOOK_SECRET": ""}, clear=False):
            result = await handle_github_webhook(ctx)

        body = json.loads(result.body.decode("utf-8"))
        assert body["success"] is True
        assert body["handled"] is False

    @pytest.mark.asyncio
    async def test_pr_event_queues_debate(self):
        """Pull request event queues code review debate."""
        ctx = self.create_mock_context(
            event_type="pull_request",
            payload={
                "action": "opened",
                "pull_request": {"number": 1, "title": "Test PR", "user": {}},
                "repository": {"full_name": "owner/repo"},
                "sender": {"login": "user"},
            },
        )

        with patch.dict("os.environ", {"GITHUB_WEBHOOK_SECRET": ""}, clear=False):
            result = await handle_github_webhook(ctx)

        body = json.loads(result.body.decode("utf-8"))
        assert body["success"] is True
        assert body["debate_queued"] is True
