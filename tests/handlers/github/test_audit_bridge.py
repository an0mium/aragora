"""Tests for the AuditGitHubBridgeHandler REST endpoints.

Covers all routes and behavior of the AuditGitHubBridgeHandler class:
- can_handle() routing for all ROUTES and ROUTE_PREFIXES
- POST /api/v1/github/audit/issues - Create issue from finding
- POST /api/v1/github/audit/issues/bulk - Bulk create issues
- POST /api/v1/github/audit/pr - Create PR with fixes
- POST /api/v1/github/audit/sync/{session_id} - Sync session to GitHub
- GET /api/v1/github/audit/sync/{session_id} - Get sync status
- GET /api/v1/github/audit/issues?session_id=... - Get finding issues
- Validation errors (missing fields)
- Path parameter extraction
- Error handling (handler_error patterns)
- Data model helpers (generate_issue_title, generate_issue_body, get_labels_for_finding)
- GitHubAuditClient demo mode
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from aragora.server.handlers.github.audit_bridge import (
    AuditGitHubBridgeHandler,
    CATEGORY_LABELS,
    GitHubAuditClient,
    GitHubIssueResult,
    IssuePriority,
    SEVERITY_LABELS,
    SyncResult,
    SyncStatus,
    _finding_issues,
    _session_syncs,
    _sync_results,
    generate_issue_body,
    generate_issue_title,
    get_labels_for_finding,
    handle_bulk_create_issues,
    handle_create_fix_pr,
    handle_create_issue,
    handle_get_finding_issues,
    handle_get_sync_status,
    handle_sync_session,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


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
    """Create an AuditGitHubBridgeHandler instance."""
    ctx: dict[str, Any] = {}
    return AuditGitHubBridgeHandler(ctx)


@pytest.fixture(autouse=True)
def clear_global_state():
    """Clear in-memory storage before and after each test."""
    _sync_results.clear()
    _session_syncs.clear()
    _finding_issues.clear()
    yield
    _sync_results.clear()
    _session_syncs.clear()
    _finding_issues.clear()


@pytest.fixture
def sample_finding():
    """Return a sample audit finding dict."""
    return {
        "id": "finding-001",
        "title": "SQL Injection Vulnerability",
        "description": "User input is not sanitized before database query.",
        "severity": "critical",
        "category": "security",
        "confidence": 0.95,
        "evidence": "cursor.execute(f'SELECT * FROM users WHERE id={user_id}')",
        "mitigation": "Use parameterized queries instead of string formatting.",
        "document_id": "doc-abc",
        "chunk_id": "chunk-42",
        "audit_type": "static-analysis",
    }


@pytest.fixture
def sample_findings():
    """Return a list of sample findings."""
    return [
        {
            "id": f"finding-{i:03d}",
            "title": f"Finding {i}",
            "description": f"Description for finding {i}",
            "severity": ["critical", "high", "medium", "low"][i % 4],
            "category": ["security", "performance", "quality", "compliance"][i % 4],
        }
        for i in range(6)
    ]


@pytest.fixture
def mock_github_client():
    """Create a mock GitHubAuditClient that returns demo-like responses."""
    with patch(
        "aragora.server.handlers.github.audit_bridge.GitHubAuditClient"
    ) as MockClientClass:
        client_instance = AsyncMock()
        client_instance.create_issue = AsyncMock(
            return_value={
                "success": True,
                "number": 42,
                "html_url": "https://github.com/owner/repo/issues/42",
            }
        )
        client_instance.create_branch = AsyncMock(
            return_value={"success": True, "ref": "refs/heads/fix/audit-abc123"}
        )
        client_instance.create_pull_request = AsyncMock(
            return_value={
                "success": True,
                "number": 99,
                "html_url": "https://github.com/owner/repo/pull/99",
                "id": 12345,
            }
        )
        client_instance.ensure_labels_exist = AsyncMock(return_value=None)
        MockClientClass.return_value = client_instance
        yield client_instance


# ---------------------------------------------------------------------------
# Data Model Tests
# ---------------------------------------------------------------------------


class TestIssuePriority:
    """Tests for IssuePriority enum."""

    def test_all_values(self):
        assert IssuePriority.CRITICAL == "critical"
        assert IssuePriority.HIGH == "high"
        assert IssuePriority.MEDIUM == "medium"
        assert IssuePriority.LOW == "low"
        assert IssuePriority.INFO == "info"


class TestSyncStatus:
    """Tests for SyncStatus enum."""

    def test_all_values(self):
        assert SyncStatus.PENDING == "pending"
        assert SyncStatus.IN_PROGRESS == "in_progress"
        assert SyncStatus.COMPLETED == "completed"
        assert SyncStatus.PARTIAL == "partial"
        assert SyncStatus.FAILED == "failed"


class TestGitHubIssueResult:
    """Tests for GitHubIssueResult dataclass."""

    def test_default_values(self):
        result = GitHubIssueResult(finding_id="f-1")
        assert result.finding_id == "f-1"
        assert result.issue_number is None
        assert result.issue_url is None
        assert result.status == "pending"
        assert result.error is None

    def test_to_dict(self):
        result = GitHubIssueResult(
            finding_id="f-1",
            issue_number=42,
            issue_url="https://github.com/owner/repo/issues/42",
            status="created",
        )
        d = result.to_dict()
        assert d["finding_id"] == "f-1"
        assert d["issue_number"] == 42
        assert d["issue_url"] == "https://github.com/owner/repo/issues/42"
        assert d["status"] == "created"
        assert d["error"] is None

    def test_to_dict_with_error(self):
        result = GitHubIssueResult(
            finding_id="f-2",
            status="failed",
            error="API rate limited",
        )
        d = result.to_dict()
        assert d["status"] == "failed"
        assert d["error"] == "API rate limited"


class TestSyncResult:
    """Tests for SyncResult dataclass."""

    def test_default_values(self):
        result = SyncResult(
            sync_id="sync-1",
            session_id="sess-1",
            repository="owner/repo",
            status=SyncStatus.PENDING,
        )
        assert result.sync_id == "sync-1"
        assert result.issues_created == []
        assert result.pr_created is None
        assert result.completed_at is None
        assert result.error is None

    def test_to_dict(self):
        result = SyncResult(
            sync_id="sync-1",
            session_id="sess-1",
            repository="owner/repo",
            status=SyncStatus.COMPLETED,
            completed_at=datetime(2026, 2, 23, 12, 0, tzinfo=timezone.utc),
        )
        d = result.to_dict()
        assert d["sync_id"] == "sync-1"
        assert d["session_id"] == "sess-1"
        assert d["repository"] == "owner/repo"
        assert d["status"] == "completed"
        assert d["completed_at"] == "2026-02-23T12:00:00+00:00"
        assert d["issues_created"] == []
        assert d["pr_created"] is None

    def test_to_dict_with_issues(self):
        issue = GitHubIssueResult(finding_id="f-1", issue_number=10, status="created")
        result = SyncResult(
            sync_id="sync-2",
            session_id="sess-2",
            repository="owner/repo",
            status=SyncStatus.COMPLETED,
            issues_created=[issue],
        )
        d = result.to_dict()
        assert len(d["issues_created"]) == 1
        assert d["issues_created"][0]["finding_id"] == "f-1"


# ---------------------------------------------------------------------------
# Content Generator Tests
# ---------------------------------------------------------------------------


class TestGenerateIssueTitle:
    """Tests for generate_issue_title()."""

    def test_critical_severity(self, sample_finding):
        title = generate_issue_title(sample_finding)
        assert title == "[CRITICAL] SQL Injection Vulnerability"

    def test_high_severity(self):
        finding = {"title": "Memory Leak", "severity": "high", "category": "performance"}
        title = generate_issue_title(finding)
        assert title == "[HIGH] Memory Leak"

    def test_medium_severity_uses_category(self):
        finding = {"title": "Unused Import", "severity": "medium", "category": "quality"}
        title = generate_issue_title(finding)
        assert title == "[QUALITY] Unused Import"

    def test_low_severity_uses_category(self):
        finding = {"title": "Docs Missing", "severity": "low", "category": "documentation"}
        title = generate_issue_title(finding)
        assert title == "[DOCUMENTATION] Docs Missing"

    def test_info_severity_uses_category(self):
        finding = {"title": "Suggestion", "severity": "info", "category": "enhancement"}
        title = generate_issue_title(finding)
        assert title == "[ENHANCEMENT] Suggestion"

    def test_missing_title_uses_default(self):
        finding = {"severity": "critical"}
        title = generate_issue_title(finding)
        assert "[CRITICAL]" in title
        assert "Audit Finding" in title

    def test_missing_severity_uses_category(self):
        finding = {"title": "Test", "category": "testing"}
        title = generate_issue_title(finding)
        assert "[TESTING]" in title

    def test_missing_all_fields(self):
        title = generate_issue_title({})
        assert "[GENERAL]" in title
        assert "Audit Finding" in title


class TestGenerateIssueBody:
    """Tests for generate_issue_body()."""

    def test_basic_body_structure(self, sample_finding):
        body = generate_issue_body(sample_finding)
        assert "## Audit Finding" in body
        assert "**Severity:** critical" in body
        assert "**Category:** security" in body
        assert "### Description" in body
        assert "User input is not sanitized" in body

    def test_includes_evidence(self, sample_finding):
        body = generate_issue_body(sample_finding, include_evidence=True)
        assert "### Evidence" in body
        assert "cursor.execute" in body

    def test_excludes_evidence_when_disabled(self, sample_finding):
        body = generate_issue_body(sample_finding, include_evidence=False)
        assert "### Evidence" not in body

    def test_includes_mitigation(self, sample_finding):
        body = generate_issue_body(sample_finding)
        assert "### Recommended Fix" in body
        assert "parameterized queries" in body

    def test_includes_metadata_footer(self, sample_finding):
        body = generate_issue_body(sample_finding, session_id="sess-123")
        assert "Finding ID: `finding-001`" in body
        assert "Audit Session: `sess-123`" in body
        assert "Document: `doc-abc`" in body
        assert "Location: `chunk-42`" in body

    def test_no_metadata_when_fields_empty(self):
        body = generate_issue_body({"description": "Simple finding"})
        assert "**Metadata:**" not in body

    def test_confidence_formatted_as_percent(self, sample_finding):
        body = generate_issue_body(sample_finding)
        assert "**Confidence:** 95%" in body

    def test_evidence_text_fallback(self):
        finding = {
            "description": "Test",
            "evidence_text": "evidence from text field",
        }
        body = generate_issue_body(finding, include_evidence=True)
        assert "evidence from text field" in body

    def test_recommendation_fallback(self):
        finding = {
            "description": "Test",
            "recommendation": "Fix by doing X",
        }
        body = generate_issue_body(finding)
        assert "Fix by doing X" in body

    def test_long_evidence_truncated(self):
        finding = {
            "description": "Test",
            "evidence": "x" * 3000,
        }
        body = generate_issue_body(finding, include_evidence=True)
        assert "(truncated)" in body


class TestGetLabelsForFinding:
    """Tests for get_labels_for_finding()."""

    def test_critical_severity_labels(self):
        finding = {"severity": "critical"}
        labels = get_labels_for_finding(finding)
        assert "priority: critical" in labels
        assert "security" in labels
        assert "bug" in labels

    def test_high_severity_labels(self):
        finding = {"severity": "high"}
        labels = get_labels_for_finding(finding)
        assert "priority: high" in labels
        assert "bug" in labels

    def test_medium_severity_labels(self):
        finding = {"severity": "medium"}
        labels = get_labels_for_finding(finding)
        assert "priority: medium" in labels

    def test_info_severity_labels(self):
        finding = {"severity": "info"}
        labels = get_labels_for_finding(finding)
        assert "documentation" in labels
        assert "enhancement" in labels

    def test_category_labels_security(self):
        finding = {"severity": "low", "category": "security"}
        labels = get_labels_for_finding(finding)
        assert "security" in labels

    def test_category_labels_performance(self):
        finding = {"category": "performance"}
        labels = get_labels_for_finding(finding)
        assert "performance" in labels

    def test_audit_type_label_added(self):
        finding = {"severity": "low", "audit_type": "static-analysis"}
        labels = get_labels_for_finding(finding)
        assert "static-analysis" in labels

    def test_labels_deduplicated(self):
        # security is both a severity label for critical and a category label
        finding = {"severity": "critical", "category": "security"}
        labels = get_labels_for_finding(finding)
        assert labels.count("security") == 1

    def test_empty_finding_returns_empty(self):
        labels = get_labels_for_finding({})
        assert labels == []

    def test_unknown_severity_ignored(self):
        finding = {"severity": "unknown_value"}
        labels = get_labels_for_finding(finding)
        # Should not crash, labels from severity are just empty
        assert isinstance(labels, list)


# ---------------------------------------------------------------------------
# can_handle() routing
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for route matching via can_handle()."""

    def test_issues_route(self, handler):
        assert handler.can_handle("/api/v1/github/audit/issues")

    def test_issues_bulk_route(self, handler):
        assert handler.can_handle("/api/v1/github/audit/issues/bulk")

    def test_pr_route(self, handler):
        assert handler.can_handle("/api/v1/github/audit/pr")

    def test_sync_session_prefix(self, handler):
        assert handler.can_handle("/api/v1/github/audit/sync/sess-123")

    def test_sync_session_with_sync_id(self, handler):
        assert handler.can_handle("/api/v1/github/audit/sync/sess-123/sync-456")

    def test_unrelated_route_rejected(self, handler):
        assert not handler.can_handle("/api/v1/debates/list")

    def test_partial_route_rejected(self, handler):
        assert not handler.can_handle("/api/v1/github/audit")

    def test_different_prefix_rejected(self, handler):
        assert not handler.can_handle("/api/v2/github/audit/issues")

    def test_empty_path_rejected(self, handler):
        assert not handler.can_handle("")

    def test_trailing_slash_on_exact_route_rejected(self, handler):
        # Exact routes do not match with trailing slash
        assert not handler.can_handle("/api/v1/github/audit/issues/")

    def test_sync_prefix_bare_matches(self, handler):
        # The prefix /api/v1/github/audit/sync/ requires at least one segment after
        assert handler.can_handle("/api/v1/github/audit/sync/any-session-id")


# ---------------------------------------------------------------------------
# Handler Class Method Tests
# ---------------------------------------------------------------------------


class TestHandleSync:
    """Tests for the synchronous handle() method."""

    def test_handle_returns_none(self, handler):
        """handle() always returns None (async methods handle actual requests)."""
        result = handler.handle("/api/v1/github/audit/issues", {}, MagicMock())
        assert result is None


class TestGetUserId:
    """Tests for _get_user_id()."""

    def test_default_user_id(self, handler):
        assert handler._get_user_id() == "default"

    def test_user_id_from_auth_context(self):
        auth_ctx = MagicMock()
        auth_ctx.user_id = "user-42"
        ctx = {"auth_context": auth_ctx}
        h = AuditGitHubBridgeHandler(ctx)
        assert h._get_user_id() == "user-42"

    def test_user_id_auth_context_without_user_id(self):
        auth_ctx = MagicMock(spec=[])  # No user_id attribute
        ctx = {"auth_context": auth_ctx}
        h = AuditGitHubBridgeHandler(ctx)
        assert h._get_user_id() == "default"


# ---------------------------------------------------------------------------
# POST /api/v1/github/audit/issues
# ---------------------------------------------------------------------------


class TestHandlePostCreateIssue:
    """Tests for handle_post_create_issue()."""

    @pytest.mark.asyncio
    async def test_success(self, handler, sample_finding, mock_github_client):
        data = {
            "repository": "owner/repo",
            "finding": sample_finding,
            "session_id": "sess-001",
        }
        result = await handler.handle_post_create_issue(data)
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["data"]["success"] is True
        assert body["data"]["issue_number"] == 42

    @pytest.mark.asyncio
    async def test_missing_repository(self, handler, sample_finding):
        data = {"finding": sample_finding}
        result = await handler.handle_post_create_issue(data)
        assert _status(result) == 400
        body = _body(result)
        assert "repository" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_missing_finding(self, handler):
        data = {"repository": "owner/repo"}
        result = await handler.handle_post_create_issue(data)
        assert _status(result) == 400
        body = _body(result)
        assert "finding" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_missing_both_fields(self, handler):
        data = {}
        result = await handler.handle_post_create_issue(data)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_with_assignees(self, handler, sample_finding, mock_github_client):
        data = {
            "repository": "owner/repo",
            "finding": sample_finding,
            "assignees": ["dev1", "dev2"],
        }
        result = await handler.handle_post_create_issue(data)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_with_milestone(self, handler, sample_finding, mock_github_client):
        data = {
            "repository": "owner/repo",
            "finding": sample_finding,
            "milestone": 3,
        }
        result = await handler.handle_post_create_issue(data)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_auto_label_disabled(self, handler, sample_finding, mock_github_client):
        data = {
            "repository": "owner/repo",
            "finding": sample_finding,
            "auto_label": False,
        }
        result = await handler.handle_post_create_issue(data)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_github_api_failure(self, handler, sample_finding, mock_github_client):
        mock_github_client.create_issue.return_value = {
            "success": False,
            "error": "API rate limited",
        }
        data = {
            "repository": "owner/repo",
            "finding": sample_finding,
        }
        result = await handler.handle_post_create_issue(data)
        assert _status(result) == 400
        body = _body(result)
        assert "error" in body


# ---------------------------------------------------------------------------
# POST /api/v1/github/audit/issues/bulk
# ---------------------------------------------------------------------------


class TestHandlePostBulkCreateIssues:
    """Tests for handle_post_bulk_create_issues()."""

    @pytest.mark.asyncio
    async def test_success(self, handler, sample_findings, mock_github_client):
        data = {
            "repository": "owner/repo",
            "findings": sample_findings,
            "session_id": "sess-bulk",
        }
        result = await handler.handle_post_bulk_create_issues(data)
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["data"]["total"] == 6

    @pytest.mark.asyncio
    async def test_missing_repository(self, handler, sample_findings):
        data = {"findings": sample_findings}
        result = await handler.handle_post_bulk_create_issues(data)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_missing_findings(self, handler):
        data = {"repository": "owner/repo"}
        result = await handler.handle_post_bulk_create_issues(data)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_findings_treated_as_falsy(self, handler):
        data = {"repository": "owner/repo", "findings": []}
        result = await handler.handle_post_bulk_create_issues(data)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_skip_existing(self, handler, sample_findings, mock_github_client):
        # Pre-populate an existing issue mapping
        _finding_issues["sess-skip"] = {"finding-000": 100}
        data = {
            "repository": "owner/repo",
            "findings": sample_findings,
            "session_id": "sess-skip",
            "skip_existing": True,
        }
        result = await handler.handle_post_bulk_create_issues(data)
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["skipped"] >= 1

    @pytest.mark.asyncio
    async def test_custom_max_concurrent(self, handler, sample_findings, mock_github_client):
        data = {
            "repository": "owner/repo",
            "findings": sample_findings,
            "max_concurrent": 2,
        }
        result = await handler.handle_post_bulk_create_issues(data)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_partial_failure(self, handler, mock_github_client):
        call_count = 0

        async def alternating_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                return {"success": False, "error": "Rate limit"}
            return {
                "success": True,
                "number": call_count,
                "html_url": f"https://github.com/owner/repo/issues/{call_count}",
            }

        mock_github_client.create_issue.side_effect = alternating_create
        findings = [
            {"id": f"f-{i}", "title": f"F{i}", "severity": "low"}
            for i in range(4)
        ]
        data = {
            "repository": "owner/repo",
            "findings": findings,
        }
        result = await handler.handle_post_bulk_create_issues(data)
        body = _body(result)
        # Some may succeed, some may fail
        assert body["data"]["total"] == 4


# ---------------------------------------------------------------------------
# POST /api/v1/github/audit/pr
# ---------------------------------------------------------------------------


class TestHandlePostCreatePR:
    """Tests for handle_post_create_pr()."""

    @pytest.mark.asyncio
    async def test_success(self, handler, sample_findings, mock_github_client):
        data = {
            "repository": "owner/repo",
            "session_id": "sess-pr-001",
            "findings": sample_findings,
        }
        result = await handler.handle_post_create_pr(data)
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["data"]["pr_number"] == 99
        assert body["data"]["branch"] is not None

    @pytest.mark.asyncio
    async def test_missing_repository(self, handler, sample_findings):
        data = {"session_id": "sess-pr", "findings": sample_findings}
        result = await handler.handle_post_create_pr(data)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_missing_session_id(self, handler, sample_findings):
        data = {"repository": "owner/repo", "findings": sample_findings}
        result = await handler.handle_post_create_pr(data)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_missing_findings(self, handler):
        data = {"repository": "owner/repo", "session_id": "sess-pr"}
        result = await handler.handle_post_create_pr(data)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_custom_branch_name(self, handler, sample_findings, mock_github_client):
        data = {
            "repository": "owner/repo",
            "session_id": "sess-pr-002",
            "findings": sample_findings,
            "branch_name": "fix/custom-branch",
        }
        result = await handler.handle_post_create_pr(data)
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["branch"] == "fix/custom-branch"

    @pytest.mark.asyncio
    async def test_custom_base_branch(self, handler, sample_findings, mock_github_client):
        data = {
            "repository": "owner/repo",
            "session_id": "sess-pr-003",
            "findings": sample_findings,
            "base_branch": "develop",
        }
        result = await handler.handle_post_create_pr(data)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_draft_false(self, handler, sample_findings, mock_github_client):
        data = {
            "repository": "owner/repo",
            "session_id": "sess-pr-004",
            "findings": sample_findings,
            "draft": False,
        }
        result = await handler.handle_post_create_pr(data)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_branch_creation_failure(self, handler, sample_findings, mock_github_client):
        mock_github_client.create_branch.return_value = {
            "success": False,
            "error": "Branch main not found",
        }
        data = {
            "repository": "owner/repo",
            "session_id": "sess-pr-005",
            "findings": sample_findings,
        }
        result = await handler.handle_post_create_pr(data)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_pr_creation_failure(self, handler, sample_findings, mock_github_client):
        mock_github_client.create_pull_request.return_value = {
            "success": False,
            "error": "Validation Failed",
        }
        data = {
            "repository": "owner/repo",
            "session_id": "sess-pr-006",
            "findings": sample_findings,
        }
        result = await handler.handle_post_create_pr(data)
        assert _status(result) == 400


# ---------------------------------------------------------------------------
# POST /api/v1/github/audit/sync/{session_id}
# ---------------------------------------------------------------------------


class TestHandlePostSyncSession:
    """Tests for handle_post_sync_session()."""

    @pytest.mark.asyncio
    async def test_success(self, handler, mock_github_client):
        data = {"repository": "owner/repo"}
        result = await handler.handle_post_sync_session(data, session_id="sess-sync-001")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_missing_repository(self, handler):
        data = {}
        result = await handler.handle_post_sync_session(data, session_id="sess-sync-002")
        assert _status(result) == 400
        body = _body(result)
        assert "repository" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_with_min_severity(self, handler, mock_github_client):
        data = {
            "repository": "owner/repo",
            "min_severity": "high",
        }
        result = await handler.handle_post_sync_session(data, session_id="sess-sync-003")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_create_pr_flag(self, handler, mock_github_client):
        data = {
            "repository": "owner/repo",
            "create_pr": True,
        }
        result = await handler.handle_post_sync_session(data, session_id="sess-sync-004")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_create_issues_false(self, handler, mock_github_client):
        data = {
            "repository": "owner/repo",
            "create_issues": False,
        }
        result = await handler.handle_post_sync_session(data, session_id="sess-sync-005")
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# GET /api/v1/github/audit/sync/{session_id}
# ---------------------------------------------------------------------------


class TestHandleGetSyncStatus:
    """Tests for handle_get_sync_status() handler method."""

    @pytest.mark.asyncio
    async def test_empty_session_returns_empty_list(self, handler):
        result = await handler.handle_get_sync_status(
            params={}, session_id="nonexistent-session"
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["syncs"] == []
        assert body["data"]["total"] == 0

    @pytest.mark.asyncio
    async def test_with_existing_syncs(self, handler):
        # Pre-populate storage
        sync_result = SyncResult(
            sync_id="sync-abc",
            session_id="sess-get-001",
            repository="owner/repo",
            status=SyncStatus.COMPLETED,
        )
        _sync_results["sync-abc"] = sync_result
        _session_syncs["sess-get-001"] = ["sync-abc"]

        result = await handler.handle_get_sync_status(
            params={}, session_id="sess-get-001"
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["total"] == 1
        assert body["data"]["syncs"][0]["sync_id"] == "sync-abc"

    @pytest.mark.asyncio
    async def test_with_specific_sync_id(self, handler):
        sync_result = SyncResult(
            sync_id="sync-xyz",
            session_id="sess-get-002",
            repository="owner/repo",
            status=SyncStatus.IN_PROGRESS,
        )
        _sync_results["sync-xyz"] = sync_result

        result = await handler.handle_get_sync_status(
            params={"sync_id": "sync-xyz"}, session_id="sess-get-002"
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["sync"]["sync_id"] == "sync-xyz"
        assert body["data"]["sync"]["status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_nonexistent_sync_id(self, handler):
        result = await handler.handle_get_sync_status(
            params={"sync_id": "nonexistent"}, session_id="sess-get-003"
        )
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body.get("error", "").lower()


# ---------------------------------------------------------------------------
# GET /api/v1/github/audit/issues?session_id=...
# ---------------------------------------------------------------------------


class TestHandleGetFindingIssues:
    """Tests for handle_get_finding_issues() handler method."""

    @pytest.mark.asyncio
    async def test_missing_session_id(self, handler):
        result = await handler.handle_get_finding_issues(params={})
        assert _status(result) == 400
        body = _body(result)
        assert "session_id" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_empty_session(self, handler):
        result = await handler.handle_get_finding_issues(
            params={"session_id": "sess-issues-001"}
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["total"] == 0
        assert body["data"]["issues"] == {}

    @pytest.mark.asyncio
    async def test_with_existing_issues(self, handler):
        _finding_issues["sess-issues-002"] = {
            "finding-a": 10,
            "finding-b": 20,
        }
        result = await handler.handle_get_finding_issues(
            params={"session_id": "sess-issues-002"}
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["total"] == 2
        assert body["data"]["issues"]["finding-a"] == 10
        assert body["data"]["issues"]["finding-b"] == 20

    @pytest.mark.asyncio
    async def test_with_specific_finding_id_found(self, handler):
        _finding_issues["sess-issues-003"] = {"finding-x": 55}
        result = await handler.handle_get_finding_issues(
            params={"session_id": "sess-issues-003", "finding_id": "finding-x"}
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["finding_id"] == "finding-x"
        assert body["data"]["issue_number"] == 55

    @pytest.mark.asyncio
    async def test_with_specific_finding_id_not_found(self, handler):
        _finding_issues["sess-issues-004"] = {}
        result = await handler.handle_get_finding_issues(
            params={"session_id": "sess-issues-004", "finding_id": "nonexistent"}
        )
        assert _status(result) == 404
        body = _body(result)
        assert "no issue found" in body.get("error", "").lower()


# ---------------------------------------------------------------------------
# Standalone function: handle_create_issue
# ---------------------------------------------------------------------------


class TestStandaloneCreateIssue:
    """Tests for the handle_create_issue() standalone function."""

    @pytest.mark.asyncio
    async def test_invalid_repository_format(self, mock_github_client):
        result = await handle_create_issue(
            repository="invalid-no-slash",
            finding={"title": "Test"},
        )
        assert result["success"] is False
        assert "invalid repository" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_valid_repository_success(self, mock_github_client):
        result = await handle_create_issue(
            repository="owner/repo",
            finding={"id": "f-1", "title": "Test", "severity": "low"},
        )
        assert result["success"] is True
        assert result["issue_number"] == 42

    @pytest.mark.asyncio
    async def test_stores_finding_mapping_with_session(self, mock_github_client):
        result = await handle_create_issue(
            repository="owner/repo",
            finding={"id": "f-mapped", "title": "Test", "severity": "low"},
            session_id="sess-map",
        )
        assert result["success"] is True
        assert _finding_issues["sess-map"]["f-mapped"] == 42

    @pytest.mark.asyncio
    async def test_no_mapping_without_session(self, mock_github_client):
        result = await handle_create_issue(
            repository="owner/repo",
            finding={"id": "f-nomap", "title": "Test"},
        )
        assert result["success"] is True
        # No session_id means no mapping stored
        assert "sess-none" not in _finding_issues

    @pytest.mark.asyncio
    async def test_demo_mode_without_token(self):
        """GitHubAuditClient with no token enters demo mode."""
        with patch.dict("os.environ", {}, clear=False):
            # Ensure no GITHUB_TOKEN
            import os
            env_backup = os.environ.pop("GITHUB_TOKEN", None)
            try:
                result = await handle_create_issue(
                    repository="owner/repo",
                    finding={"id": "f-demo", "title": "Demo", "severity": "medium"},
                )
                assert result["success"] is True
                assert result.get("demo") is True
            finally:
                if env_backup is not None:
                    os.environ["GITHUB_TOKEN"] = env_backup

    @pytest.mark.asyncio
    async def test_three_part_repository_invalid(self, mock_github_client):
        result = await handle_create_issue(
            repository="owner/repo/extra",
            finding={"title": "Test"},
        )
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Standalone function: handle_create_fix_pr
# ---------------------------------------------------------------------------


class TestStandaloneCreateFixPR:
    """Tests for the handle_create_fix_pr() standalone function."""

    @pytest.mark.asyncio
    async def test_invalid_repository(self, mock_github_client):
        result = await handle_create_fix_pr(
            repository="noslash",
            session_id="sess-1",
            findings=[{"title": "F"}],
        )
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_auto_generated_branch_name(self, mock_github_client):
        result = await handle_create_fix_pr(
            repository="owner/repo",
            session_id="abcdef1234567890",
            findings=[{"title": "F", "severity": "high"}],
        )
        assert result["success"] is True
        assert result["branch"].startswith("fix/audit-")
        assert "abcdef12" in result["branch"]

    @pytest.mark.asyncio
    async def test_short_session_id_branch_name(self, mock_github_client):
        result = await handle_create_fix_pr(
            repository="owner/repo",
            session_id="abc",
            findings=[{"title": "F"}],
        )
        assert result["success"] is True
        assert result["branch"] == "fix/audit-abc"

    @pytest.mark.asyncio
    async def test_findings_count_in_result(self, mock_github_client):
        findings = [{"title": f"F{i}"} for i in range(5)]
        result = await handle_create_fix_pr(
            repository="owner/repo",
            session_id="sess-count",
            findings=findings,
        )
        assert result["success"] is True
        assert result["findings_count"] == 5

    @pytest.mark.asyncio
    async def test_many_findings_pr_body_limit(self, mock_github_client):
        """PR body limits listing to 20 findings."""
        findings = [
            {"title": f"Finding {i}", "severity": "low"} for i in range(25)
        ]
        result = await handle_create_fix_pr(
            repository="owner/repo",
            session_id="sess-many",
            findings=findings,
        )
        assert result["success"] is True
        # The PR creation was called -- verify via mock
        mock_github_client.create_pull_request.assert_called_once()
        call_kwargs = mock_github_client.create_pull_request.call_args[1]
        assert "and 5 more" in call_kwargs["body"]


# ---------------------------------------------------------------------------
# Standalone function: handle_get_sync_status
# ---------------------------------------------------------------------------


class TestStandaloneGetSyncStatus:
    """Tests for the handle_get_sync_status() standalone function."""

    @pytest.mark.asyncio
    async def test_specific_sync_found(self):
        sync = SyncResult(
            sync_id="sync-found",
            session_id="sess-1",
            repository="owner/repo",
            status=SyncStatus.COMPLETED,
        )
        _sync_results["sync-found"] = sync

        result = await handle_get_sync_status(
            session_id="sess-1",
            sync_id="sync-found",
        )
        assert result["success"] is True
        assert result["sync"]["sync_id"] == "sync-found"

    @pytest.mark.asyncio
    async def test_specific_sync_not_found(self):
        result = await handle_get_sync_status(
            session_id="sess-1",
            sync_id="nonexistent",
        )
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_all_syncs_for_session(self):
        s1 = SyncResult(
            sync_id="s1", session_id="sess-all", repository="o/r",
            status=SyncStatus.COMPLETED,
        )
        s2 = SyncResult(
            sync_id="s2", session_id="sess-all", repository="o/r",
            status=SyncStatus.FAILED,
        )
        _sync_results["s1"] = s1
        _sync_results["s2"] = s2
        _session_syncs["sess-all"] = ["s1", "s2"]

        result = await handle_get_sync_status(session_id="sess-all")
        assert result["success"] is True
        assert result["total"] == 2

    @pytest.mark.asyncio
    async def test_session_with_no_syncs(self):
        result = await handle_get_sync_status(session_id="empty-sess")
        assert result["success"] is True
        assert result["total"] == 0
        assert result["syncs"] == []


# ---------------------------------------------------------------------------
# Standalone function: handle_get_finding_issues
# ---------------------------------------------------------------------------


class TestStandaloneGetFindingIssues:
    """Tests for the handle_get_finding_issues() standalone function."""

    @pytest.mark.asyncio
    async def test_all_issues_for_session(self):
        _finding_issues["sess-fi"] = {"f1": 10, "f2": 20}
        result = await handle_get_finding_issues(session_id="sess-fi")
        assert result["success"] is True
        assert result["total"] == 2
        assert result["issues"]["f1"] == 10

    @pytest.mark.asyncio
    async def test_specific_finding_found(self):
        _finding_issues["sess-fi2"] = {"f-target": 77}
        result = await handle_get_finding_issues(
            session_id="sess-fi2", finding_id="f-target"
        )
        assert result["success"] is True
        assert result["issue_number"] == 77

    @pytest.mark.asyncio
    async def test_specific_finding_not_found(self):
        _finding_issues["sess-fi3"] = {}
        result = await handle_get_finding_issues(
            session_id="sess-fi3", finding_id="missing"
        )
        assert result["success"] is False
        assert "no issue found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_nonexistent_session(self):
        result = await handle_get_finding_issues(session_id="nope")
        assert result["success"] is True
        assert result["total"] == 0


# ---------------------------------------------------------------------------
# Standalone function: handle_sync_session
# ---------------------------------------------------------------------------


class TestStandaloneSyncSession:
    """Tests for the handle_sync_session() standalone function."""

    @pytest.mark.asyncio
    async def test_session_not_found_uses_demo(self, mock_github_client):
        """When audit_sessions import fails, uses demo data."""
        with patch(
            "aragora.server.handlers.github.audit_bridge.handle_bulk_create_issues",
            new_callable=AsyncMock,
        ) as mock_bulk:
            mock_bulk.return_value = {
                "success": True,
                "results": [],
            }
            # The import of audit_sessions may fail; sync falls back to demo data
            result = await handle_sync_session(
                repository="owner/repo",
                session_id="demo-sess",
            )
            # Should succeed regardless (demo fallback)
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_sync_stores_result(self, mock_github_client):
        with patch(
            "aragora.server.handlers.github.audit_bridge.handle_bulk_create_issues",
            new_callable=AsyncMock,
        ) as mock_bulk:
            mock_bulk.return_value = {
                "success": True,
                "results": [],
            }
            result = await handle_sync_session(
                repository="owner/repo",
                session_id="sess-store",
            )
            # Sync results should be stored in global state
            assert len(_session_syncs.get("sess-store", [])) >= 1


# ---------------------------------------------------------------------------
# GitHubAuditClient Tests
# ---------------------------------------------------------------------------


class TestGitHubAuditClient:
    """Tests for the GitHubAuditClient class."""

    def test_init_with_token(self):
        client = GitHubAuditClient(token="test-token")
        assert client.token == "test-token"

    def test_init_from_env(self):
        with patch.dict("os.environ", {"GITHUB_TOKEN": "env-token"}):
            client = GitHubAuditClient()
            assert client.token == "env-token"

    def test_init_no_token(self):
        with patch.dict("os.environ", {}, clear=True):
            client = GitHubAuditClient()
            assert client.token is None

    @pytest.mark.asyncio
    async def test_create_issue_demo_mode(self):
        client = GitHubAuditClient(token=None)
        # Ensure env var is also cleared
        with patch.dict("os.environ", {}, clear=True):
            client.token = None
            result = await client.create_issue(
                owner="owner",
                repo="repo",
                title="Test Issue",
                body="Test body",
            )
            assert result["success"] is True
            assert result.get("demo") is True
            assert "html_url" in result

    @pytest.mark.asyncio
    async def test_create_branch_demo_mode(self):
        client = GitHubAuditClient(token=None)
        with patch.dict("os.environ", {}, clear=True):
            client.token = None
            result = await client.create_branch(
                owner="owner",
                repo="repo",
                branch_name="fix/test",
            )
            assert result["success"] is True
            assert result.get("demo") is True
            assert "refs/heads/fix/test" in result["ref"]

    @pytest.mark.asyncio
    async def test_create_pull_request_demo_mode(self):
        client = GitHubAuditClient(token=None)
        with patch.dict("os.environ", {}, clear=True):
            client.token = None
            result = await client.create_pull_request(
                owner="owner",
                repo="repo",
                title="Fix things",
                body="PR body",
                head_branch="fix/test",
            )
            assert result["success"] is True
            assert result.get("demo") is True
            assert "html_url" in result

    @pytest.mark.asyncio
    async def test_get_issue_demo_mode(self):
        client = GitHubAuditClient(token=None)
        with patch.dict("os.environ", {}, clear=True):
            client.token = None
            result = await client.get_issue(
                owner="owner",
                repo="repo",
                issue_number=42,
            )
            assert result is not None
            assert result["number"] == 42
            assert result["state"] == "open"

    @pytest.mark.asyncio
    async def test_add_issue_comment_demo_mode(self):
        client = GitHubAuditClient(token=None)
        with patch.dict("os.environ", {}, clear=True):
            client.token = None
            result = await client.add_issue_comment(
                owner="owner",
                repo="repo",
                issue_number=42,
                body="Test comment",
            )
            assert result["success"] is True
            assert result.get("demo") is True

    @pytest.mark.asyncio
    async def test_ensure_labels_demo_mode(self):
        client = GitHubAuditClient(token=None)
        with patch.dict("os.environ", {}, clear=True):
            client.token = None
            # Should return without error (no-op in demo mode)
            await client.ensure_labels_exist(
                owner="owner",
                repo="repo",
                labels=["bug", "security"],
            )


# ---------------------------------------------------------------------------
# Label / Severity Mapping Tests
# ---------------------------------------------------------------------------


class TestLabelMappings:
    """Tests for SEVERITY_LABELS and CATEGORY_LABELS constants."""

    def test_all_severities_have_labels(self):
        for sev in ["critical", "high", "medium", "low", "info"]:
            assert sev in SEVERITY_LABELS
            assert len(SEVERITY_LABELS[sev]) > 0

    def test_all_categories_have_labels(self):
        for cat in [
            "security", "performance", "quality", "compliance",
            "consistency", "documentation", "testing", "accessibility",
        ]:
            assert cat in CATEGORY_LABELS
            assert len(CATEGORY_LABELS[cat]) > 0


# ---------------------------------------------------------------------------
# Edge Cases and Error Handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_finding_without_id_gets_uuid(self, mock_github_client):
        """Finding without an id field should get a generated UUID."""
        result = await handle_create_issue(
            repository="owner/repo",
            finding={"title": "No ID Finding", "severity": "low"},
        )
        assert result["success"] is True
        assert result["finding_id"] is not None
        # Should be a valid string (UUID)
        assert len(result["finding_id"]) > 0

    @pytest.mark.asyncio
    async def test_concurrent_storage_access(self, mock_github_client):
        """Multiple concurrent create_issue calls should not corrupt storage."""
        import asyncio

        tasks = []
        for i in range(10):
            tasks.append(
                handle_create_issue(
                    repository="owner/repo",
                    finding={"id": f"concurrent-{i}", "title": f"F{i}", "severity": "low"},
                    session_id="sess-concurrent",
                )
            )
        results = await asyncio.gather(*tasks)
        successes = [r for r in results if r.get("success")]
        assert len(successes) == 10
        assert len(_finding_issues.get("sess-concurrent", {})) == 10

    def test_handler_routes_attribute(self, handler):
        """Handler ROUTES should contain all exact routes."""
        assert "/api/v1/github/audit/issues" in handler.ROUTES
        assert "/api/v1/github/audit/issues/bulk" in handler.ROUTES
        assert "/api/v1/github/audit/pr" in handler.ROUTES

    def test_handler_route_prefixes_attribute(self, handler):
        """Handler ROUTE_PREFIXES should contain sync prefix."""
        assert "/api/v1/github/audit/sync/" in handler.ROUTE_PREFIXES

    @pytest.mark.asyncio
    async def test_bulk_with_empty_findings_after_skip(self, handler, mock_github_client):
        """All findings already exist -- nothing to create."""
        _finding_issues["sess-all-skip"] = {
            "f-0": 1,
            "f-1": 2,
        }
        findings = [
            {"id": "f-0", "title": "Already exists"},
            {"id": "f-1", "title": "Also exists"},
        ]
        data = {
            "repository": "owner/repo",
            "findings": findings,
            "session_id": "sess-all-skip",
            "skip_existing": True,
        }
        result = await handler.handle_post_bulk_create_issues(data)
        assert _status(result) == 200
        body = _body(result)
        assert body["data"]["skipped"] == 2
        assert body["data"]["created"] == 0

    @pytest.mark.asyncio
    async def test_create_issue_empty_finding_dict(self, handler, mock_github_client):
        """Minimal empty finding should still work."""
        data = {
            "repository": "owner/repo",
            "finding": {},
        }
        result = await handler.handle_post_create_issue(data)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_sync_status_multiple_sessions_isolated(self, handler):
        """Sync results for different sessions do not leak."""
        s1 = SyncResult(
            sync_id="s1", session_id="sess-A", repository="o/r",
            status=SyncStatus.COMPLETED,
        )
        s2 = SyncResult(
            sync_id="s2", session_id="sess-B", repository="o/r",
            status=SyncStatus.FAILED,
        )
        _sync_results["s1"] = s1
        _sync_results["s2"] = s2
        _session_syncs["sess-A"] = ["s1"]
        _session_syncs["sess-B"] = ["s2"]

        result_a = await handler.handle_get_sync_status(params={}, session_id="sess-A")
        result_b = await handler.handle_get_sync_status(params={}, session_id="sess-B")

        body_a = _body(result_a)
        body_b = _body(result_b)
        assert body_a["data"]["total"] == 1
        assert body_a["data"]["syncs"][0]["status"] == "completed"
        assert body_b["data"]["total"] == 1
        assert body_b["data"]["syncs"][0]["status"] == "failed"
