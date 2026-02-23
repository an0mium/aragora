"""Tests for audit-to-GitHub bridge handler.

Covers all routes and behavior of the AuditGitHubBridgeHandler class
and its supporting standalone functions:

- can_handle() route matching (all routes + rejection)
- POST /api/v1/github/audit/issues       - Create issue from finding
- POST /api/v1/github/audit/issues/bulk   - Bulk create issues
- GET  /api/v1/github/audit/issues        - Get finding issues
- POST /api/v1/github/audit/pr            - Create PR with fixes
- GET  /api/v1/github/audit/sync/{id}     - Get sync status
- POST /api/v1/github/audit/sync/{id}     - Sync session to GitHub
- Handler initialization and data models
- Standalone helpers (generate_issue_title, generate_issue_body, get_labels_for_finding)
- GitHubAuditClient demo mode
- Edge cases (missing params, invalid repo format, empty body)
- Error paths (400, 404, 500)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.github.audit_bridge import (
    AuditGitHubBridgeHandler,
    GitHubAuditClient,
    GitHubIssueResult,
    IssuePriority,
    SEVERITY_LABELS,
    CATEGORY_LABELS,
    SyncResult,
    SyncStatus,
    generate_issue_body,
    generate_issue_title,
    get_labels_for_finding,
    handle_bulk_create_issues,
    handle_create_fix_pr,
    handle_create_issue,
    handle_get_finding_issues,
    handle_get_sync_status,
    handle_sync_session,
    _finding_issues,
    _session_syncs,
    _sync_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create an AuditGitHubBridgeHandler with minimal context."""
    return AuditGitHubBridgeHandler({})


@pytest.fixture(autouse=True)
def _clear_storage():
    """Reset module-level in-memory storage between tests."""
    _sync_results.clear()
    _session_syncs.clear()
    _finding_issues.clear()
    yield
    _sync_results.clear()
    _session_syncs.clear()
    _finding_issues.clear()


@pytest.fixture
def sample_finding():
    """A single audit finding dict."""
    return {
        "id": "finding-001",
        "title": "SQL Injection Vulnerability",
        "description": "Unsanitized input in query builder.",
        "severity": "critical",
        "category": "security",
        "evidence": "SELECT * FROM users WHERE id = '$input'",
        "mitigation": "Use parameterized queries.",
        "confidence": 0.95,
        "document_id": "doc-abc",
        "chunk_id": "chunk-42",
    }


@pytest.fixture
def sample_findings():
    """A list of audit findings."""
    return [
        {
            "id": f"finding-{i:03d}",
            "title": f"Finding {i}",
            "description": f"Description for finding {i}",
            "severity": ["critical", "high", "medium", "low"][i % 4],
            "category": ["security", "performance", "quality", "compliance"][i % 4],
        }
        for i in range(5)
    ]


# ---------------------------------------------------------------------------
# can_handle() routing
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Verify route matching in can_handle()."""

    def test_issues_route(self, handler):
        assert handler.can_handle("/api/v1/github/audit/issues")

    def test_issues_bulk_route(self, handler):
        assert handler.can_handle("/api/v1/github/audit/issues/bulk")

    def test_pr_route(self, handler):
        assert handler.can_handle("/api/v1/github/audit/pr")

    def test_sync_prefix_route(self, handler):
        assert handler.can_handle("/api/v1/github/audit/sync/session-123")

    def test_sync_prefix_with_long_id(self, handler):
        assert handler.can_handle("/api/v1/github/audit/sync/very-long-session-id-abc-def-012345")

    def test_rejects_unrelated_path(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    def test_rejects_partial_match(self, handler):
        assert not handler.can_handle("/api/v1/github/audit")

    def test_rejects_v2_path(self, handler):
        assert not handler.can_handle("/api/v2/github/audit/issues")

    def test_rejects_wrong_prefix(self, handler):
        assert not handler.can_handle("/api/v1/github/webhooks")

    def test_rejects_empty_path(self, handler):
        assert not handler.can_handle("")

    def test_rejects_root_path(self, handler):
        assert not handler.can_handle("/")

    def test_sync_prefix_trailing_slash(self, handler):
        # "/api/v1/github/audit/sync/" starts with the prefix, so it should match
        assert handler.can_handle("/api/v1/github/audit/sync/")


# ---------------------------------------------------------------------------
# Handler initialization
# ---------------------------------------------------------------------------


class TestHandlerInit:
    """Tests for handler initialization."""

    def test_creates_with_empty_context(self):
        h = AuditGitHubBridgeHandler({})
        assert h is not None

    def test_creates_with_context(self):
        ctx = {"auth_context": MagicMock(user_id="user-1")}
        h = AuditGitHubBridgeHandler(ctx)
        assert h._get_user_id() == "user-1"

    def test_get_user_id_default(self):
        h = AuditGitHubBridgeHandler({})
        assert h._get_user_id() == "default"

    def test_get_user_id_with_auth_no_user_id(self):
        ctx = {"auth_context": MagicMock(spec=[])}
        h = AuditGitHubBridgeHandler(ctx)
        assert h._get_user_id() == "default"

    def test_routes_defined(self):
        h = AuditGitHubBridgeHandler({})
        assert len(h.ROUTES) == 3
        assert "/api/v1/github/audit/issues" in h.ROUTES
        assert "/api/v1/github/audit/issues/bulk" in h.ROUTES
        assert "/api/v1/github/audit/pr" in h.ROUTES

    def test_route_prefixes_defined(self):
        h = AuditGitHubBridgeHandler({})
        assert len(h.ROUTE_PREFIXES) == 1
        assert "/api/v1/github/audit/sync/" in h.ROUTE_PREFIXES

    def test_handle_returns_none(self):
        """The synchronous handle() always returns None (async dispatch only)."""
        h = AuditGitHubBridgeHandler({})
        assert h.handle("/api/v1/github/audit/issues", {}, MagicMock()) is None


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class TestDataModels:
    """Tests for data model classes."""

    def test_issue_priority_values(self):
        assert IssuePriority.CRITICAL.value == "critical"
        assert IssuePriority.HIGH.value == "high"
        assert IssuePriority.MEDIUM.value == "medium"
        assert IssuePriority.LOW.value == "low"
        assert IssuePriority.INFO.value == "info"

    def test_sync_status_values(self):
        assert SyncStatus.PENDING.value == "pending"
        assert SyncStatus.IN_PROGRESS.value == "in_progress"
        assert SyncStatus.COMPLETED.value == "completed"
        assert SyncStatus.PARTIAL.value == "partial"
        assert SyncStatus.FAILED.value == "failed"

    def test_github_issue_result_to_dict(self):
        r = GitHubIssueResult(
            finding_id="f-1",
            issue_number=42,
            issue_url="https://github.com/owner/repo/issues/42",
            status="created",
        )
        d = r.to_dict()
        assert d["finding_id"] == "f-1"
        assert d["issue_number"] == 42
        assert d["issue_url"] == "https://github.com/owner/repo/issues/42"
        assert d["status"] == "created"
        assert d["error"] is None

    def test_github_issue_result_defaults(self):
        r = GitHubIssueResult(finding_id="f-1")
        d = r.to_dict()
        assert d["issue_number"] is None
        assert d["status"] == "pending"

    def test_sync_result_to_dict(self):
        sr = SyncResult(
            sync_id="sync-1",
            session_id="sess-1",
            repository="owner/repo",
            status=SyncStatus.COMPLETED,
        )
        d = sr.to_dict()
        assert d["sync_id"] == "sync-1"
        assert d["session_id"] == "sess-1"
        assert d["repository"] == "owner/repo"
        assert d["status"] == "completed"
        assert d["issues_created"] == []
        assert d["pr_created"] is None
        assert d["started_at"] is not None
        assert d["completed_at"] is None
        assert d["error"] is None


# ---------------------------------------------------------------------------
# generate_issue_title
# ---------------------------------------------------------------------------


class TestGenerateIssueTitle:
    """Tests for issue title generation."""

    def test_critical_severity(self, sample_finding):
        sample_finding["severity"] = "critical"
        title = generate_issue_title(sample_finding)
        assert title.startswith("[CRITICAL]")
        assert sample_finding["title"] in title

    def test_high_severity(self, sample_finding):
        sample_finding["severity"] = "high"
        title = generate_issue_title(sample_finding)
        assert title.startswith("[HIGH]")

    def test_medium_severity_uses_category(self, sample_finding):
        sample_finding["severity"] = "medium"
        sample_finding["category"] = "security"
        title = generate_issue_title(sample_finding)
        assert title.startswith("[SECURITY]")

    def test_low_severity_uses_category(self, sample_finding):
        sample_finding["severity"] = "low"
        sample_finding["category"] = "performance"
        title = generate_issue_title(sample_finding)
        assert title.startswith("[PERFORMANCE]")

    def test_missing_severity_uses_category(self):
        title = generate_issue_title({"title": "Test", "category": "quality"})
        assert title.startswith("[QUALITY]")

    def test_missing_title_defaults(self):
        title = generate_issue_title({"severity": "critical"})
        assert "Audit Finding" in title

    def test_missing_category_defaults(self):
        title = generate_issue_title({"title": "Test"})
        assert "[GENERAL]" in title


# ---------------------------------------------------------------------------
# generate_issue_body
# ---------------------------------------------------------------------------


class TestGenerateIssueBody:
    """Tests for issue body generation."""

    def test_includes_severity_and_category(self, sample_finding):
        body = generate_issue_body(sample_finding)
        assert "**Severity:** critical" in body
        assert "**Category:** security" in body

    def test_includes_description(self, sample_finding):
        body = generate_issue_body(sample_finding)
        assert "Unsanitized input in query builder." in body

    def test_includes_evidence(self, sample_finding):
        body = generate_issue_body(sample_finding)
        assert "SELECT * FROM users" in body

    def test_excludes_evidence_when_flag_false(self, sample_finding):
        body = generate_issue_body(sample_finding, include_evidence=False)
        assert "### Evidence" not in body

    def test_includes_mitigation(self, sample_finding):
        body = generate_issue_body(sample_finding)
        assert "Use parameterized queries." in body

    def test_includes_session_id(self, sample_finding):
        body = generate_issue_body(sample_finding, session_id="sess-abc")
        assert "sess-abc" in body

    def test_includes_metadata_footer(self, sample_finding):
        body = generate_issue_body(sample_finding, session_id="sess-1")
        assert "finding-001" in body
        assert "doc-abc" in body
        assert "chunk-42" in body

    def test_includes_confidence(self, sample_finding):
        body = generate_issue_body(sample_finding)
        assert "95%" in body

    def test_truncates_long_evidence(self, sample_finding):
        sample_finding["evidence"] = "x" * 3000
        body = generate_issue_body(sample_finding)
        assert "...(truncated)" in body

    def test_evidence_not_truncated_when_short(self, sample_finding):
        sample_finding["evidence"] = "short evidence"
        body = generate_issue_body(sample_finding)
        assert "...(truncated)" not in body

    def test_no_metadata_when_empty(self):
        body = generate_issue_body({"description": "desc"})
        # no finding_id, session_id, document_id, chunk_id -> no metadata section
        assert "**Metadata:**" not in body

    def test_recommendation_field(self):
        finding = {
            "description": "desc",
            "recommendation": "Do something",
        }
        body = generate_issue_body(finding)
        assert "Do something" in body

    def test_evidence_text_field(self):
        finding = {
            "description": "desc",
            "evidence_text": "some evidence text",
        }
        body = generate_issue_body(finding)
        assert "some evidence text" in body


# ---------------------------------------------------------------------------
# get_labels_for_finding
# ---------------------------------------------------------------------------


class TestGetLabelsForFinding:
    """Tests for label generation from findings."""

    def test_critical_severity_labels(self):
        labels = get_labels_for_finding({"severity": "critical"})
        assert "priority: critical" in labels
        assert "security" in labels
        assert "bug" in labels

    def test_high_severity_labels(self):
        labels = get_labels_for_finding({"severity": "high"})
        assert "priority: high" in labels
        assert "bug" in labels

    def test_medium_severity_labels(self):
        labels = get_labels_for_finding({"severity": "medium"})
        assert "priority: medium" in labels

    def test_info_severity_labels(self):
        labels = get_labels_for_finding({"severity": "info"})
        assert "documentation" in labels
        assert "enhancement" in labels

    def test_security_category_labels(self):
        labels = get_labels_for_finding({"category": "security"})
        assert "security" in labels

    def test_performance_category_labels(self):
        labels = get_labels_for_finding({"category": "performance"})
        assert "performance" in labels

    def test_audit_type_added(self):
        labels = get_labels_for_finding({"audit_type": "codebase"})
        assert "codebase" in labels

    def test_audit_type_not_duplicated(self):
        labels = get_labels_for_finding(
            {"severity": "critical", "category": "security", "audit_type": "security"}
        )
        assert labels.count("security") == 1

    def test_combined_labels_deduplicated(self):
        labels = get_labels_for_finding({"severity": "critical", "category": "security"})
        # "security" appears in both SEVERITY_LABELS["critical"] and CATEGORY_LABELS["security"]
        assert labels.count("security") == 1

    def test_empty_finding(self):
        labels = get_labels_for_finding({})
        assert labels == []

    def test_unknown_severity(self):
        labels = get_labels_for_finding({"severity": "unknown"})
        assert "priority: critical" not in labels


# ---------------------------------------------------------------------------
# GitHubAuditClient (demo mode)
# ---------------------------------------------------------------------------


class TestGitHubAuditClientDemo:
    """Tests for GitHubAuditClient in demo mode (no token)."""

    @pytest.mark.asyncio
    async def test_create_issue_demo(self):
        client = GitHubAuditClient(token=None)
        result = await client.create_issue("owner", "repo", "Test Issue", "body")
        assert result["success"] is True
        assert result["demo"] is True
        assert "number" in result
        assert "html_url" in result

    @pytest.mark.asyncio
    async def test_create_branch_demo(self):
        client = GitHubAuditClient(token=None)
        result = await client.create_branch("owner", "repo", "fix/test")
        assert result["success"] is True
        assert result["demo"] is True
        assert "refs/heads/fix/test" in result["ref"]

    @pytest.mark.asyncio
    async def test_create_pull_request_demo(self):
        client = GitHubAuditClient(token=None)
        result = await client.create_pull_request(
            "owner", "repo", "Fix things", "body", "fix/branch"
        )
        assert result["success"] is True
        assert result["demo"] is True
        assert "number" in result
        assert "html_url" in result

    @pytest.mark.asyncio
    async def test_get_issue_demo(self):
        client = GitHubAuditClient(token=None)
        result = await client.get_issue("owner", "repo", 42)
        assert result["number"] == 42
        assert result["state"] == "open"

    @pytest.mark.asyncio
    async def test_add_issue_comment_demo(self):
        client = GitHubAuditClient(token=None)
        result = await client.add_issue_comment("owner", "repo", 42, "Nice work")
        assert result["success"] is True
        assert result["demo"] is True

    @pytest.mark.asyncio
    async def test_ensure_labels_exist_demo(self):
        client = GitHubAuditClient(token=None)
        # Should return None without error
        result = await client.ensure_labels_exist("owner", "repo", ["bug", "security"])
        assert result is None

    def test_token_from_env(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        client = GitHubAuditClient()
        assert client.token is None

    def test_token_from_env_set(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test123")
        client = GitHubAuditClient()
        assert client.token == "ghp_test123"


# ---------------------------------------------------------------------------
# POST /api/v1/github/audit/issues (handle_post_create_issue)
# ---------------------------------------------------------------------------


class TestHandlePostCreateIssue:
    """Tests for creating a single issue via the handler method."""

    @pytest.mark.asyncio
    async def test_create_issue_success(self, handler, sample_finding, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        result = await handler.handle_post_create_issue(
            {
                "repository": "owner/repo",
                "finding": sample_finding,
                "session_id": "sess-001",
            }
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["data"]["success"] is True
        assert body["data"]["demo"] is True
        assert "issue_number" in body["data"]

    @pytest.mark.asyncio
    async def test_create_issue_missing_repository(self, handler, sample_finding):
        result = await handler.handle_post_create_issue({"finding": sample_finding})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_issue_missing_finding(self, handler):
        result = await handler.handle_post_create_issue({"repository": "owner/repo"})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_issue_empty_body(self, handler):
        result = await handler.handle_post_create_issue({})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_issue_invalid_repo_format(self, handler, sample_finding, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        result = await handler.handle_post_create_issue(
            {
                "repository": "invalid-no-slash",
                "finding": sample_finding,
            }
        )
        body = _body(result)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_issue_with_assignees(self, handler, sample_finding, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        result = await handler.handle_post_create_issue(
            {
                "repository": "owner/repo",
                "finding": sample_finding,
                "assignees": ["user1", "user2"],
            }
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["data"]["success"] is True

    @pytest.mark.asyncio
    async def test_create_issue_auto_label_false(self, handler, sample_finding, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        result = await handler.handle_post_create_issue(
            {
                "repository": "owner/repo",
                "finding": sample_finding,
                "auto_label": False,
            }
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["data"]["success"] is True

    @pytest.mark.asyncio
    async def test_create_issue_stores_mapping(self, handler, sample_finding, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        await handler.handle_post_create_issue(
            {
                "repository": "owner/repo",
                "finding": sample_finding,
                "session_id": "sess-map-test",
            }
        )
        assert "sess-map-test" in _finding_issues
        assert "finding-001" in _finding_issues["sess-map-test"]


# ---------------------------------------------------------------------------
# POST /api/v1/github/audit/issues/bulk (handle_post_bulk_create_issues)
# ---------------------------------------------------------------------------


class TestHandlePostBulkCreateIssues:
    """Tests for bulk creating issues via the handler method."""

    @pytest.mark.asyncio
    async def test_bulk_create_success(self, handler, sample_findings, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        result = await handler.handle_post_bulk_create_issues(
            {
                "repository": "owner/repo",
                "findings": sample_findings,
                "session_id": "sess-bulk",
            }
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["data"]["success"] is True
        assert body["data"]["total"] == 5
        assert body["data"]["created"] == 5

    @pytest.mark.asyncio
    async def test_bulk_create_missing_repository(self, handler, sample_findings):
        result = await handler.handle_post_bulk_create_issues({"findings": sample_findings})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_bulk_create_missing_findings(self, handler):
        result = await handler.handle_post_bulk_create_issues({"repository": "owner/repo"})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_bulk_create_empty_body(self, handler):
        result = await handler.handle_post_bulk_create_issues({})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_bulk_create_skip_existing(self, handler, sample_findings, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        # Pre-populate an existing mapping
        _finding_issues["sess-skip"] = {"finding-000": 999}

        result = await handler.handle_post_bulk_create_issues(
            {
                "repository": "owner/repo",
                "findings": sample_findings,
                "session_id": "sess-skip",
                "skip_existing": True,
            }
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["data"]["skipped"] == 1
        assert body["data"]["created"] == 4

    @pytest.mark.asyncio
    async def test_bulk_create_no_skip(self, handler, sample_findings, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        _finding_issues["sess-noskip"] = {"finding-000": 999}

        result = await handler.handle_post_bulk_create_issues(
            {
                "repository": "owner/repo",
                "findings": sample_findings,
                "session_id": "sess-noskip",
                "skip_existing": False,
            }
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["data"]["skipped"] == 0
        assert body["data"]["created"] == 5

    @pytest.mark.asyncio
    async def test_bulk_create_empty_findings_list(self, handler, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        result = await handler.handle_post_bulk_create_issues(
            {"repository": "owner/repo", "findings": []}
        )
        # Empty list is falsy, so "findings required" check fails
        assert _status(result) == 400


# ---------------------------------------------------------------------------
# POST /api/v1/github/audit/pr (handle_post_create_pr)
# ---------------------------------------------------------------------------


class TestHandlePostCreatePR:
    """Tests for creating a PR with fixes via the handler method."""

    @pytest.mark.asyncio
    async def test_create_pr_success(self, handler, sample_findings, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        result = await handler.handle_post_create_pr(
            {
                "repository": "owner/repo",
                "session_id": "sess-pr-001",
                "findings": sample_findings,
            }
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["data"]["success"] is True
        assert body["data"]["demo"] is True
        assert "pr_number" in body["data"]
        assert "branch" in body["data"]

    @pytest.mark.asyncio
    async def test_create_pr_missing_repository(self, handler, sample_findings):
        result = await handler.handle_post_create_pr(
            {"session_id": "s", "findings": sample_findings}
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_pr_missing_session_id(self, handler, sample_findings):
        result = await handler.handle_post_create_pr(
            {"repository": "owner/repo", "findings": sample_findings}
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_pr_missing_findings(self, handler):
        result = await handler.handle_post_create_pr(
            {"repository": "owner/repo", "session_id": "sess-1"}
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_pr_empty_body(self, handler):
        result = await handler.handle_post_create_pr({})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_pr_invalid_repo_format(self, handler, sample_findings, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        result = await handler.handle_post_create_pr(
            {
                "repository": "no-slash",
                "session_id": "sess-1",
                "findings": sample_findings,
            }
        )
        body = _body(result)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_pr_custom_branch(self, handler, sample_findings, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        result = await handler.handle_post_create_pr(
            {
                "repository": "owner/repo",
                "session_id": "sess-1",
                "findings": sample_findings,
                "branch_name": "fix/custom-branch",
                "base_branch": "develop",
                "draft": False,
            }
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["data"]["branch"] == "fix/custom-branch"

    @pytest.mark.asyncio
    async def test_create_pr_auto_branch_name(self, handler, sample_findings, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        result = await handler.handle_post_create_pr(
            {
                "repository": "owner/repo",
                "session_id": "abcdef1234567890",
                "findings": sample_findings,
            }
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["data"]["branch"] == "fix/audit-abcdef12"

    @pytest.mark.asyncio
    async def test_create_pr_short_session_id(self, handler, sample_findings, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        result = await handler.handle_post_create_pr(
            {
                "repository": "owner/repo",
                "session_id": "short",
                "findings": sample_findings,
            }
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["data"]["branch"] == "fix/audit-short"


# ---------------------------------------------------------------------------
# GET /api/v1/github/audit/sync/{session_id} (handle_get_sync_status)
# ---------------------------------------------------------------------------


class TestHandleGetSyncStatus:
    """Tests for getting sync status via the handler method."""

    @pytest.mark.asyncio
    async def test_get_sync_empty_session(self, handler):
        result = await handler.handle_get_sync_status({}, session_id="nonexistent-session")
        body = _body(result)
        assert _status(result) == 200
        assert body["data"]["syncs"] == []
        assert body["data"]["total"] == 0

    @pytest.mark.asyncio
    async def test_get_sync_with_results(self, handler):
        # Pre-populate storage
        sr = SyncResult(
            sync_id="sync-test-1",
            session_id="sess-1",
            repository="owner/repo",
            status=SyncStatus.COMPLETED,
        )
        _sync_results["sync-test-1"] = sr
        _session_syncs["sess-1"] = ["sync-test-1"]

        result = await handler.handle_get_sync_status({}, session_id="sess-1")
        body = _body(result)
        assert _status(result) == 200
        assert body["data"]["total"] == 1
        assert body["data"]["syncs"][0]["sync_id"] == "sync-test-1"

    @pytest.mark.asyncio
    async def test_get_sync_by_id(self, handler):
        sr = SyncResult(
            sync_id="sync-specific",
            session_id="sess-1",
            repository="owner/repo",
            status=SyncStatus.IN_PROGRESS,
        )
        _sync_results["sync-specific"] = sr

        result = await handler.handle_get_sync_status(
            {"sync_id": "sync-specific"}, session_id="sess-1"
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["data"]["sync"]["sync_id"] == "sync-specific"
        assert body["data"]["sync"]["status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_get_sync_by_id_not_found(self, handler):
        result = await handler.handle_get_sync_status(
            {"sync_id": "nonexistent"}, session_id="sess-1"
        )
        assert _status(result) == 404


# ---------------------------------------------------------------------------
# POST /api/v1/github/audit/sync/{session_id} (handle_post_sync_session)
# ---------------------------------------------------------------------------


class TestHandlePostSyncSession:
    """Tests for syncing a session to GitHub via the handler method."""

    @pytest.mark.asyncio
    async def test_sync_missing_repository(self, handler):
        result = await handler.handle_post_sync_session({}, session_id="sess-1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_sync_success_with_demo_findings(self, handler, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        # Provide a session in the audit_sessions storage so it's found
        try:
            from aragora.server.handlers.features import audit_sessions

            audit_sessions._sessions["sess-sync-1"] = {"id": "sess-sync-1", "status": "completed"}
            audit_sessions._findings["sess-sync-1"] = [
                {
                    "id": f"demo-finding-{i}",
                    "title": f"Demo Finding {i}",
                    "description": f"Description {i}",
                    "severity": ["critical", "high", "medium", "low"][i % 4],
                    "category": "security",
                }
                for i in range(3)
            ]
        except ImportError:
            pytest.skip("audit_sessions not available")

        try:
            result = await handler.handle_post_sync_session(
                {
                    "repository": "owner/repo",
                    "min_severity": "low",
                    "create_issues": True,
                    "create_pr": False,
                },
                session_id="sess-sync-1",
            )
            body = _body(result)
            assert _status(result) == 200
        finally:
            audit_sessions._sessions.pop("sess-sync-1", None)
            audit_sessions._findings.pop("sess-sync-1", None)


# ---------------------------------------------------------------------------
# GET /api/v1/github/audit/issues (handle_get_finding_issues)
# ---------------------------------------------------------------------------


class TestHandleGetFindingIssues:
    """Tests for getting linked finding issues via the handler method."""

    @pytest.mark.asyncio
    async def test_get_finding_issues_missing_session_id(self, handler):
        result = await handler.handle_get_finding_issues({})
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_get_finding_issues_empty_session(self, handler):
        result = await handler.handle_get_finding_issues({"session_id": "nonexistent"})
        body = _body(result)
        assert _status(result) == 200
        assert body["data"]["issues"] == {}
        assert body["data"]["total"] == 0

    @pytest.mark.asyncio
    async def test_get_finding_issues_with_data(self, handler):
        _finding_issues["sess-linked"] = {"finding-1": 42, "finding-2": 43}
        result = await handler.handle_get_finding_issues({"session_id": "sess-linked"})
        body = _body(result)
        assert _status(result) == 200
        assert body["data"]["total"] == 2
        assert body["data"]["issues"]["finding-1"] == 42

    @pytest.mark.asyncio
    async def test_get_finding_issues_specific_finding(self, handler):
        _finding_issues["sess-specific"] = {"finding-x": 100}
        result = await handler.handle_get_finding_issues(
            {"session_id": "sess-specific", "finding_id": "finding-x"}
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["data"]["finding_id"] == "finding-x"
        assert body["data"]["issue_number"] == 100

    @pytest.mark.asyncio
    async def test_get_finding_issues_specific_not_found(self, handler):
        _finding_issues["sess-miss"] = {"finding-a": 1}
        result = await handler.handle_get_finding_issues(
            {"session_id": "sess-miss", "finding_id": "finding-z"}
        )
        assert _status(result) == 404


# ---------------------------------------------------------------------------
# Standalone function tests: handle_create_issue
# ---------------------------------------------------------------------------


class TestHandleCreateIssueFn:
    """Tests for the standalone handle_create_issue function."""

    @pytest.mark.asyncio
    async def test_success_demo(self, sample_finding, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        result = await handle_create_issue(
            repository="owner/repo",
            finding=sample_finding,
            session_id="sess-fn",
        )
        assert result["success"] is True
        assert result["demo"] is True
        assert "issue_number" in result
        assert "finding_id" in result

    @pytest.mark.asyncio
    async def test_invalid_repository(self, sample_finding, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        result = await handle_create_issue(
            repository="bad-format",
            finding=sample_finding,
        )
        assert result["success"] is False
        assert "Invalid repository format" in result["error"]

    @pytest.mark.asyncio
    async def test_too_many_slashes(self, sample_finding, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        result = await handle_create_issue(
            repository="a/b/c",
            finding=sample_finding,
        )
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_stores_session_mapping(self, sample_finding, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        await handle_create_issue(
            repository="owner/repo",
            finding=sample_finding,
            session_id="sess-store-test",
        )
        assert "sess-store-test" in _finding_issues

    @pytest.mark.asyncio
    async def test_no_session_no_mapping(self, sample_finding, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        await handle_create_issue(
            repository="owner/repo",
            finding=sample_finding,
        )
        # No session_id -> no mapping stored
        assert len(_finding_issues) == 0


# ---------------------------------------------------------------------------
# Standalone function tests: handle_bulk_create_issues
# ---------------------------------------------------------------------------


class TestHandleBulkCreateIssuesFn:
    """Tests for the standalone handle_bulk_create_issues function."""

    @pytest.mark.asyncio
    async def test_bulk_success(self, sample_findings, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        result = await handle_bulk_create_issues(
            repository="owner/repo",
            findings=sample_findings,
        )
        assert result["success"] is True
        assert result["total"] == 5
        assert result["created"] == 5
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_bulk_skip_existing(self, sample_findings, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        _finding_issues["sess-bulk-fn"] = {"finding-000": 1}
        result = await handle_bulk_create_issues(
            repository="owner/repo",
            findings=sample_findings,
            session_id="sess-bulk-fn",
            skip_existing=True,
        )
        assert result["skipped"] == 1
        assert result["created"] == 4

    @pytest.mark.asyncio
    async def test_bulk_with_max_concurrent(self, sample_findings, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        result = await handle_bulk_create_issues(
            repository="owner/repo",
            findings=sample_findings,
            max_concurrent=1,
        )
        assert result["success"] is True
        assert result["created"] == 5


# ---------------------------------------------------------------------------
# Standalone function tests: handle_create_fix_pr
# ---------------------------------------------------------------------------


class TestHandleCreateFixPRFn:
    """Tests for the standalone handle_create_fix_pr function."""

    @pytest.mark.asyncio
    async def test_pr_success(self, sample_findings, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        result = await handle_create_fix_pr(
            repository="owner/repo",
            session_id="sess-pr-fn",
            findings=sample_findings,
        )
        assert result["success"] is True
        assert "pr_number" in result
        assert "pr_url" in result
        assert result["findings_count"] == 5

    @pytest.mark.asyncio
    async def test_pr_invalid_repo(self, sample_findings, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        result = await handle_create_fix_pr(
            repository="bad",
            session_id="s",
            findings=sample_findings,
        )
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_pr_custom_branch(self, sample_findings, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        result = await handle_create_fix_pr(
            repository="owner/repo",
            session_id="sess-1",
            findings=sample_findings,
            branch_name="fix/my-custom",
        )
        assert result["success"] is True
        assert result["branch"] == "fix/my-custom"

    @pytest.mark.asyncio
    async def test_pr_many_findings_truncated_in_body(self, monkeypatch):
        """PR body truncates findings list at 20 with a '... and N more' note."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        findings = [{"id": f"f-{i}", "title": f"Finding {i}", "severity": "low"} for i in range(25)]
        result = await handle_create_fix_pr(
            repository="owner/repo",
            session_id="sess-many",
            findings=findings,
        )
        assert result["success"] is True
        assert result["findings_count"] == 25


# ---------------------------------------------------------------------------
# Standalone function tests: handle_sync_session
# ---------------------------------------------------------------------------


class TestHandleSyncSessionFn:
    """Tests for the standalone handle_sync_session function."""

    @pytest.mark.asyncio
    async def test_sync_with_session_data(self, monkeypatch):
        """Sync succeeds when session exists in audit_sessions storage."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        try:
            from aragora.server.handlers.features import audit_sessions

            audit_sessions._sessions["demo-session"] = {"id": "demo-session"}
            audit_sessions._findings["demo-session"] = [
                {
                    "id": "f-0",
                    "title": "Demo",
                    "description": "desc",
                    "severity": "medium",
                    "category": "quality",
                }
            ]
        except ImportError:
            pytest.skip("audit_sessions not available")

        try:
            result = await handle_sync_session(
                repository="owner/repo",
                session_id="demo-session",
            )
            assert result["success"] is True
            assert "sync" in result
        finally:
            audit_sessions._sessions.pop("demo-session", None)
            audit_sessions._findings.pop("demo-session", None)

    @pytest.mark.asyncio
    async def test_sync_session_not_found(self, monkeypatch):
        """Sync fails when session does not exist."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        result = await handle_sync_session(
            repository="owner/repo",
            session_id="nonexistent-session-xyz",
        )
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_sync_stores_result(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        try:
            from aragora.server.handlers.features import audit_sessions

            audit_sessions._sessions["sess-store"] = {"id": "sess-store"}
            audit_sessions._findings["sess-store"] = []
        except ImportError:
            pytest.skip("audit_sessions not available")

        try:
            await handle_sync_session(
                repository="owner/repo",
                session_id="sess-store",
            )
            assert len(_sync_results) == 1
            assert "sess-store" in _session_syncs
        finally:
            audit_sessions._sessions.pop("sess-store", None)
            audit_sessions._findings.pop("sess-store", None)


# ---------------------------------------------------------------------------
# Standalone function tests: handle_get_sync_status
# ---------------------------------------------------------------------------


class TestHandleGetSyncStatusFn:
    """Tests for the standalone handle_get_sync_status function."""

    @pytest.mark.asyncio
    async def test_get_all_syncs_for_session(self):
        _session_syncs["sess-1"] = ["sync-a"]
        _sync_results["sync-a"] = SyncResult(
            sync_id="sync-a",
            session_id="sess-1",
            repository="o/r",
            status=SyncStatus.COMPLETED,
        )
        result = await handle_get_sync_status(session_id="sess-1")
        assert result["success"] is True
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_get_specific_sync(self):
        _sync_results["sync-b"] = SyncResult(
            sync_id="sync-b",
            session_id="sess-2",
            repository="o/r",
            status=SyncStatus.PARTIAL,
        )
        result = await handle_get_sync_status(session_id="sess-2", sync_id="sync-b")
        assert result["success"] is True
        assert result["sync"]["status"] == "partial"

    @pytest.mark.asyncio
    async def test_sync_not_found(self):
        result = await handle_get_sync_status(session_id="s", sync_id="nonexistent")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_session(self):
        result = await handle_get_sync_status(session_id="empty-sess")
        assert result["success"] is True
        assert result["syncs"] == []
        assert result["total"] == 0


# ---------------------------------------------------------------------------
# Standalone function tests: handle_get_finding_issues
# ---------------------------------------------------------------------------


class TestHandleGetFindingIssuesFn:
    """Tests for the standalone handle_get_finding_issues function."""

    @pytest.mark.asyncio
    async def test_get_all_issues_for_session(self):
        _finding_issues["sess-fi"] = {"f1": 10, "f2": 20}
        result = await handle_get_finding_issues(session_id="sess-fi")
        assert result["success"] is True
        assert result["total"] == 2

    @pytest.mark.asyncio
    async def test_get_specific_finding_issue(self):
        _finding_issues["sess-fi2"] = {"f1": 10}
        result = await handle_get_finding_issues(session_id="sess-fi2", finding_id="f1")
        assert result["success"] is True
        assert result["issue_number"] == 10

    @pytest.mark.asyncio
    async def test_finding_not_found(self):
        _finding_issues["sess-fi3"] = {}
        result = await handle_get_finding_issues(session_id="sess-fi3", finding_id="missing")
        assert result["success"] is False
        assert "no issue found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_session(self):
        result = await handle_get_finding_issues(session_id="no-such-session")
        assert result["success"] is True
        assert result["issues"] == {}
        assert result["total"] == 0


# ---------------------------------------------------------------------------
# SEVERITY_LABELS and CATEGORY_LABELS constants
# ---------------------------------------------------------------------------


class TestLabelConstants:
    """Verify label mapping constants are correct."""

    def test_severity_labels_has_all_levels(self):
        assert set(SEVERITY_LABELS.keys()) == {"critical", "high", "medium", "low", "info"}

    def test_category_labels_has_expected_keys(self):
        expected = {
            "security",
            "performance",
            "quality",
            "compliance",
            "consistency",
            "documentation",
            "testing",
            "accessibility",
        }
        assert set(CATEGORY_LABELS.keys()) == expected

    def test_critical_has_security_label(self):
        assert "security" in SEVERITY_LABELS["critical"]

    def test_info_has_enhancement_label(self):
        assert "enhancement" in SEVERITY_LABELS["info"]
