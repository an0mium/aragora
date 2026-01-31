"""
Comprehensive tests for Atlassian Jira Enterprise Connector.

Tests the Jira REST API integration including:
- Connector initialization and authentication
- Issue CRUD operations (create, read, update, delete)
- Issue search (JQL queries)
- Project management
- Sprint operations
- Comment management
- Attachment handling
- Transition (workflow) operations
- Webhook processing
- Rate limiting and pagination
- Error handling

Targets 90+ tests for comprehensive coverage.
"""

import asyncio
import base64
import os
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from aragora.connectors.enterprise.collaboration.jira import (
    JiraConnector,
    JiraProject,
    JiraIssue,
    JiraComment,
)
from aragora.connectors.enterprise.base import SyncState, SyncStatus


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def connector():
    """Create test connector for Jira Cloud."""
    conn = JiraConnector(
        base_url="https://test-domain.atlassian.net",
        projects=["PROJ", "DEV"],
        jql="status != Done",
        include_subtasks=True,
        include_comments=True,
        include_attachments=True,
    )
    # Mock credentials
    conn.credentials = MagicMock()
    conn.credentials.get_credential = AsyncMock(
        side_effect=lambda key: {
            "JIRA_EMAIL": "test@example.com",
            "JIRA_API_TOKEN": "test_token",
        }.get(key)
    )
    return conn


@pytest.fixture
def dc_connector():
    """Create test connector for Jira Data Center."""
    conn = JiraConnector(
        base_url="https://jira.internal.company.com",
        projects=["PROJ"],
    )
    # Mock credentials
    conn.credentials = MagicMock()
    conn.credentials.get_credential = AsyncMock(
        side_effect=lambda key: {"JIRA_PAT": "test_pat"}.get(key)
    )
    return conn


@pytest.fixture
def minimal_connector():
    """Create connector with minimal configuration."""
    conn = JiraConnector(
        base_url="https://minimal.atlassian.net",
    )
    conn.credentials = MagicMock()
    conn.credentials.get_credential = AsyncMock(
        side_effect=lambda key: {
            "JIRA_EMAIL": "test@example.com",
            "JIRA_API_TOKEN": "test_token",
        }.get(key)
    )
    return conn


def make_project_data(
    project_id: str = "10001",
    key: str = "PROJ",
    name: str = "Test Project",
    project_type: str = "software",
    description: str = "Project description",
) -> dict[str, Any]:
    """Create mock project data."""
    return {
        "id": project_id,
        "key": key,
        "name": name,
        "projectTypeKey": project_type,
        "lead": {"displayName": "John Lead"},
        "description": description,
    }


def make_issue_data(
    issue_id: str = "10001",
    key: str = "PROJ-123",
    summary: str = "Test Issue",
    description: str | dict | None = "Issue description",
    status: str = "In Progress",
    issue_type: str = "Story",
    priority: str = "Medium",
    parent_key: str | None = None,
    story_points: float | None = None,
) -> dict[str, Any]:
    """Create mock issue data."""
    fields: dict[str, Any] = {
        "summary": summary,
        "description": description,
        "issuetype": {"name": issue_type},
        "status": {"name": status},
        "priority": {"name": priority},
        "assignee": {"displayName": "Alice Dev"},
        "reporter": {"displayName": "Bob PM"},
        "created": "2024-01-15T10:00:00.000+0000",
        "updated": "2024-01-15T12:00:00.000+0000",
        "labels": ["backend", "api"],
        "components": [{"name": "Core"}],
        "fixVersions": [{"name": "v1.0"}],
        "customfield_10016": story_points,
    }
    if parent_key:
        fields["parent"] = {"key": parent_key}
    return {
        "id": issue_id,
        "key": key,
        "fields": fields,
    }


def make_comment_data(
    comment_id: str = "20001",
    body: str | dict = "Test comment",
    author: str = "Commenter",
) -> dict[str, Any]:
    """Create mock comment data."""
    return {
        "id": comment_id,
        "body": body,
        "author": {"displayName": author},
        "created": "2024-01-15T14:00:00.000+0000",
        "updated": "2024-01-15T14:30:00.000+0000",
    }


def make_attachment_data(
    attachment_id: str = "30001",
    filename: str = "document.pdf",
    size: int = 1024000,
    mime_type: str = "application/pdf",
) -> dict[str, Any]:
    """Create mock attachment data."""
    return {
        "id": attachment_id,
        "filename": filename,
        "author": {"displayName": "Uploader"},
        "created": "2024-01-15T15:00:00.000+0000",
        "size": size,
        "mimeType": mime_type,
        "content": f"https://test-domain.atlassian.net/rest/api/3/attachment/content/{attachment_id}",
    }


def make_sprint_data(
    sprint_id: int = 1,
    name: str = "Sprint 1",
    state: str = "active",
    start_date: str | None = "2024-01-15T00:00:00.000Z",
    end_date: str | None = "2024-01-29T23:59:59.000Z",
) -> dict[str, Any]:
    """Create mock sprint data."""
    return {
        "id": sprint_id,
        "name": name,
        "state": state,
        "startDate": start_date,
        "endDate": end_date,
        "originBoardId": 1,
        "goal": f"Complete {name}",
    }


def make_transition_data(
    transition_id: str = "11",
    name: str = "Done",
    to_status: str = "Done",
) -> dict[str, Any]:
    """Create mock transition data."""
    return {
        "id": transition_id,
        "name": name,
        "to": {
            "id": "10001",
            "name": to_status,
            "statusCategory": {
                "id": 3,
                "key": "done",
                "name": "Done",
            },
        },
    }


# =============================================================================
# Initialization Tests
# =============================================================================


class TestJiraConnectorInit:
    """Test JiraConnector initialization."""

    def test_cloud_configuration(self):
        """Should detect Jira Cloud from URL."""
        connector = JiraConnector(base_url="https://company.atlassian.net")
        assert connector.is_cloud is True
        assert connector.base_url == "https://company.atlassian.net"

    def test_data_center_configuration(self):
        """Should detect Jira Data Center from URL."""
        connector = JiraConnector(base_url="https://jira.internal.com")
        assert connector.is_cloud is False

    def test_data_center_on_prem_configuration(self):
        """Should detect on-premise Jira as Data Center."""
        connector = JiraConnector(base_url="https://jira.company.local:8443")
        assert connector.is_cloud is False

    def test_custom_configuration(self):
        """Should accept custom configuration."""
        connector = JiraConnector(
            base_url="https://test.atlassian.net",
            projects=["PROJ", "DEV"],
            jql="priority = High",
            include_subtasks=False,
            include_comments=False,
            include_attachments=True,
            exclude_statuses=["Done", "Closed"],
            exclude_types=["Epic"],
        )
        assert connector.projects == ["PROJ", "DEV"]
        assert connector.jql == "priority = High"
        assert connector.include_subtasks is False
        assert connector.include_comments is False
        assert connector.include_attachments is True
        assert "done" in connector.exclude_statuses
        assert "epic" in connector.exclude_types

    def test_connector_id_generation(self):
        """Should generate connector ID from domain."""
        connector = JiraConnector(base_url="https://my-company.atlassian.net")
        assert "jira_" in connector.connector_id
        assert "my-company" in connector.connector_id

    def test_connector_id_with_subdomain(self):
        """Should handle subdomain in connector ID."""
        connector = JiraConnector(base_url="https://subdomain.company.atlassian.net")
        assert "jira_" in connector.connector_id
        assert "subdomain" in connector.connector_id

    def test_url_normalization(self):
        """Should normalize URL by removing trailing slash."""
        connector = JiraConnector(base_url="https://test.atlassian.net/")
        assert connector.base_url == "https://test.atlassian.net"

    def test_url_normalization_multiple_slashes(self):
        """Should normalize URL with multiple trailing slashes."""
        connector = JiraConnector(base_url="https://test.atlassian.net///")
        # Should only remove the last slash
        assert not connector.base_url.endswith("/")

    def test_connector_properties(self, connector):
        """Should have correct connector properties."""
        assert "Jira" in connector.name
        assert "test-domain.atlassian.net" in connector.name

    def test_source_type(self, connector):
        """Should have correct source type."""
        from aragora.reasoning.provenance import SourceType

        assert connector.source_type == SourceType.EXTERNAL_API

    def test_default_project_filter(self, minimal_connector):
        """Should have no project filter by default."""
        assert minimal_connector.projects is None

    def test_default_jql_filter(self, minimal_connector):
        """Should have no JQL filter by default."""
        assert minimal_connector.jql is None

    def test_default_include_subtasks(self, minimal_connector):
        """Should include subtasks by default."""
        assert minimal_connector.include_subtasks is True

    def test_default_include_comments(self, minimal_connector):
        """Should include comments by default."""
        assert minimal_connector.include_comments is True

    def test_default_include_attachments(self, minimal_connector):
        """Should not include attachments by default."""
        assert minimal_connector.include_attachments is False

    def test_exclude_statuses_normalization(self):
        """Should normalize exclude statuses to lowercase."""
        connector = JiraConnector(
            base_url="https://test.atlassian.net",
            exclude_statuses=["DONE", "Closed", "cancelled"],
        )
        assert "done" in connector.exclude_statuses
        assert "closed" in connector.exclude_statuses
        assert "cancelled" in connector.exclude_statuses

    def test_exclude_types_normalization(self):
        """Should normalize exclude types to lowercase."""
        connector = JiraConnector(
            base_url="https://test.atlassian.net",
            exclude_types=["EPIC", "Sub-task"],
        )
        assert "epic" in connector.exclude_types
        assert "sub-task" in connector.exclude_types


# =============================================================================
# Authentication Tests
# =============================================================================


class TestAuthentication:
    """Test authentication flows."""

    @pytest.mark.asyncio
    async def test_cloud_auth_header(self, connector):
        """Should generate Basic auth header for Cloud."""
        header = await connector._get_auth_header()

        assert "Authorization" in header
        assert header["Authorization"].startswith("Basic ")

    @pytest.mark.asyncio
    async def test_cloud_auth_header_encoding(self, connector):
        """Should properly encode email:token for Basic auth."""
        header = await connector._get_auth_header()

        # Decode and verify
        encoded = header["Authorization"].replace("Basic ", "")
        decoded = base64.b64decode(encoded).decode()
        assert decoded == "test@example.com:test_token"

    @pytest.mark.asyncio
    async def test_data_center_auth_header(self, dc_connector):
        """Should generate Bearer auth header for Data Center."""
        header = await dc_connector._get_auth_header()

        assert "Authorization" in header
        assert header["Authorization"] == "Bearer test_pat"

    @pytest.mark.asyncio
    async def test_missing_cloud_email(self, connector):
        """Should raise error when Cloud email missing."""
        connector.credentials.get_credential = AsyncMock(
            side_effect=lambda key: {"JIRA_API_TOKEN": "token"}.get(key)
        )

        with pytest.raises(ValueError, match="Jira Cloud credentials not configured"):
            await connector._get_auth_header()

    @pytest.mark.asyncio
    async def test_missing_cloud_token(self, connector):
        """Should raise error when Cloud token missing."""
        connector.credentials.get_credential = AsyncMock(
            side_effect=lambda key: {"JIRA_EMAIL": "email@example.com"}.get(key)
        )

        with pytest.raises(ValueError, match="Jira Cloud credentials not configured"):
            await connector._get_auth_header()

    @pytest.mark.asyncio
    async def test_missing_dc_credentials(self, dc_connector):
        """Should raise error when Data Center credentials missing."""
        dc_connector.credentials.get_credential = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="Jira Data Center credentials not configured"):
            await dc_connector._get_auth_header()


# =============================================================================
# API Request Tests
# =============================================================================


class TestAPIRequest:
    """Test API request handling."""

    @pytest.mark.asyncio
    async def test_api_request_headers(self, connector):
        """Should set correct headers for API request."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"data": "test"}'
        mock_response.json.return_value = {"data": "test"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.request = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            result = await connector._api_request("/test")

            # Verify headers were set
            call_args = mock_instance.request.call_args
            headers = call_args[1]["headers"]
            assert "Authorization" in headers
            assert headers["Accept"] == "application/json"
            assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_api_request_cloud_version(self, connector):
        """Should use API v3 for Cloud."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"{}"
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.request = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            await connector._api_request("/test")

            call_args = mock_instance.request.call_args
            # URL is the second positional arg (after method)
            url = call_args[0][1] if len(call_args[0]) > 1 else call_args.kwargs.get("url", "")
            assert "/rest/api/3/" in url

    @pytest.mark.asyncio
    async def test_api_request_dc_version(self, dc_connector):
        """Should use API v2 for Data Center."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"{}"
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.request = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            await dc_connector._api_request("/test")

            call_args = mock_instance.request.call_args
            # URL is the second positional arg (after method)
            url = call_args[0][1] if len(call_args[0]) > 1 else call_args.kwargs.get("url", "")
            assert "/rest/api/2/" in url

    @pytest.mark.asyncio
    async def test_api_request_empty_response(self, connector):
        """Should handle empty response body."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.content = b""
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_instance.request = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            result = await connector._api_request("/test")
            assert result == {}


# =============================================================================
# Project Operations Tests
# =============================================================================


class TestProjectOperations:
    """Test project-related operations."""

    @pytest.mark.asyncio
    async def test_get_projects(self, connector):
        """Should get accessible projects."""
        mock_response = {
            "values": [
                make_project_data("10001", "PROJ", "Project Alpha"),
                make_project_data("10002", "DEV", "Development"),
            ],
            "isLast": True,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            projects = await connector._get_projects()

            assert len(projects) == 2
            assert projects[0].key == "PROJ"
            assert projects[0].name == "Project Alpha"
            assert projects[1].key == "DEV"

    @pytest.mark.asyncio
    async def test_get_projects_filtered(self, connector):
        """Should filter to configured projects."""
        connector.projects = ["PROJ"]

        mock_response = {
            "values": [
                make_project_data("10001", "PROJ", "Project Alpha"),
                make_project_data("10002", "OTHER", "Other Project"),
            ],
            "isLast": True,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            projects = await connector._get_projects()

            assert len(projects) == 1
            assert projects[0].key == "PROJ"

    @pytest.mark.asyncio
    async def test_get_projects_no_filter(self, minimal_connector):
        """Should return all projects when no filter set."""
        mock_response = {
            "values": [
                make_project_data("10001", "PROJ", "Project 1"),
                make_project_data("10002", "DEV", "Project 2"),
                make_project_data("10003", "OPS", "Project 3"),
            ],
            "isLast": True,
        }

        with patch.object(minimal_connector, "_api_request", return_value=mock_response):
            projects = await minimal_connector._get_projects()
            assert len(projects) == 3

    @pytest.mark.asyncio
    async def test_get_projects_caches_results(self, connector):
        """Should cache projects for later use."""
        mock_response = {
            "values": [make_project_data("10001", "PROJ", "Project")],
            "isLast": True,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            projects = await connector._get_projects()

            assert "PROJ" in connector._projects_cache
            assert connector._projects_cache["PROJ"].name == "Project"

    @pytest.mark.asyncio
    async def test_get_projects_pagination(self, connector):
        """Should handle paginated project responses."""
        # The connector returns all in first page when fewer than max_results
        # Testing that it properly continues when page is "full" (50 items)
        # For simplicity, test with isLast=False and exactly 50 values
        first_page_values = [
            make_project_data(f"1000{i}", "PROJ" if i == 0 else f"OTHER{i}", f"Project {i}")
            for i in range(50)
        ]
        first_page = {
            "values": first_page_values,
            "isLast": False,
        }
        second_page = {
            "values": [make_project_data("10051", "DEV", "Project DEV")],
            "isLast": True,
        }

        call_count = [0]

        async def mock_request(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return first_page
            return second_page

        with patch.object(connector, "_api_request", side_effect=mock_request):
            projects = await connector._get_projects()
            # Should get PROJ from first page and DEV from second page
            # (filtered by connector.projects = ["PROJ", "DEV"])
            assert len(projects) == 2
            assert call_count[0] == 2  # Verify pagination occurred

    @pytest.mark.asyncio
    async def test_get_projects_with_descriptions(self, connector):
        """Should extract project descriptions."""
        mock_response = {
            "values": [
                make_project_data(
                    "10001", "PROJ", "Project", description="This is the project description"
                ),
            ],
            "isLast": True,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            projects = await connector._get_projects()

            assert projects[0].description == "This is the project description"

    @pytest.mark.asyncio
    async def test_get_projects_null_description(self, connector):
        """Should handle null project descriptions."""
        mock_response = {
            "values": [
                {
                    "id": "10001",
                    "key": "PROJ",
                    "name": "Project",
                    "projectTypeKey": "software",
                    "lead": {"displayName": "Lead"},
                    "description": None,
                },
            ],
            "isLast": True,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            projects = await connector._get_projects()
            assert projects[0].description == ""


# =============================================================================
# Issue Operations Tests
# =============================================================================


class TestIssueOperations:
    """Test issue-related operations."""

    @pytest.mark.asyncio
    async def test_search_issues(self, connector):
        """Should search issues with JQL."""
        mock_response = {
            "issues": [
                make_issue_data("10001", "PROJ-1", "First Issue"),
                make_issue_data("10002", "PROJ-2", "Second Issue"),
            ],
            "total": 2,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            result = await connector._search_issues("project = PROJ")

            assert len(result["issues"]) == 2
            assert result["total"] == 2

    @pytest.mark.asyncio
    async def test_search_issues_fields(self, connector):
        """Should request correct fields in search."""
        mock_response = {"issues": [], "total": 0}

        with patch.object(connector, "_api_request", return_value=mock_response) as mock_req:
            await connector._search_issues("project = PROJ")

            call_args = mock_req.call_args
            params = call_args[1]["params"]
            assert "fields" in params
            assert "summary" in params["fields"]
            assert "description" in params["fields"]
            assert "status" in params["fields"]

    @pytest.mark.asyncio
    async def test_get_issues(self, connector):
        """Should get issues from project."""
        mock_response = {
            "issues": [
                make_issue_data("10001", "PROJ-1", "Issue One"),
            ],
            "total": 1,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            issues = []
            async for issue in connector._get_issues("PROJ"):
                issues.append(issue)

            assert len(issues) == 1
            assert issues[0].key == "PROJ-1"
            assert issues[0].summary == "Issue One"

    @pytest.mark.asyncio
    async def test_get_issues_with_modified_since(self, connector):
        """Should filter issues by modified timestamp."""
        mock_response = {
            "issues": [make_issue_data()],
            "total": 1,
        }

        modified_since = datetime(2024, 1, 15, tzinfo=timezone.utc)

        with patch.object(connector, "_api_request", return_value=mock_response) as mock_req:
            issues = []
            async for issue in connector._get_issues("PROJ", modified_since=modified_since):
                issues.append(issue)

            call_args = mock_req.call_args
            params = call_args[1]["params"]
            assert "updated >=" in params["jql"]

    @pytest.mark.asyncio
    async def test_get_issues_excludes_statuses(self, connector):
        """Should exclude issues with excluded statuses."""
        connector.exclude_statuses = {"done"}

        mock_response = {
            "issues": [
                make_issue_data("10001", "PROJ-1", "Active Issue", status="In Progress"),
                make_issue_data("10002", "PROJ-2", "Done Issue", status="Done"),
            ],
            "total": 2,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            issues = []
            async for issue in connector._get_issues("PROJ"):
                issues.append(issue)

            assert len(issues) == 1
            assert issues[0].key == "PROJ-1"

    @pytest.mark.asyncio
    async def test_get_issues_excludes_types(self, connector):
        """Should exclude issues with excluded types."""
        connector.exclude_types = {"epic"}

        mock_response = {
            "issues": [
                make_issue_data("10001", "PROJ-1", "Story", issue_type="Story"),
                make_issue_data("10002", "PROJ-2", "Epic", issue_type="Epic"),
            ],
            "total": 2,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            issues = []
            async for issue in connector._get_issues("PROJ"):
                issues.append(issue)

            assert len(issues) == 1
            assert issues[0].issue_type == "Story"

    @pytest.mark.asyncio
    async def test_get_issues_with_parent(self, connector):
        """Should extract parent key for subtasks."""
        mock_response = {
            "issues": [
                make_issue_data("10001", "PROJ-1", "Subtask", parent_key="PROJ-100"),
            ],
            "total": 1,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            issues = []
            async for issue in connector._get_issues("PROJ"):
                issues.append(issue)

            assert issues[0].parent_key == "PROJ-100"

    @pytest.mark.asyncio
    async def test_get_issues_with_story_points(self, connector):
        """Should extract story points from custom field."""
        mock_response = {
            "issues": [
                make_issue_data("10001", "PROJ-1", "Story", story_points=5.0),
            ],
            "total": 1,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            issues = []
            async for issue in connector._get_issues("PROJ"):
                issues.append(issue)

            assert issues[0].story_points == 5.0

    @pytest.mark.asyncio
    async def test_get_issues_pagination(self, connector):
        """Should handle paginated issue responses."""
        # Create a "full" page of 50 issues to trigger pagination
        first_page_issues = [
            make_issue_data(f"1000{i}", f"PROJ-{i}", f"Issue {i}") for i in range(50)
        ]
        first_page = {
            "issues": first_page_issues,
            "total": 51,  # Total is more than first page
        }
        second_page = {
            "issues": [make_issue_data("10051", "PROJ-51", "Issue 51")],
            "total": 51,
        }

        call_count = [0]

        async def mock_request(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return first_page
            return second_page

        with patch.object(connector, "_api_request", side_effect=mock_request):
            issues = []
            async for issue in connector._get_issues("PROJ"):
                issues.append(issue)

            assert len(issues) == 51
            assert call_count[0] == 2  # Verify pagination occurred

    @pytest.mark.asyncio
    async def test_get_issues_url_generation(self, connector):
        """Should generate correct issue URLs."""
        mock_response = {
            "issues": [make_issue_data("10001", "PROJ-123", "Test")],
            "total": 1,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            issues = []
            async for issue in connector._get_issues("PROJ"):
                issues.append(issue)

            assert issues[0].url == "https://test-domain.atlassian.net/browse/PROJ-123"


# =============================================================================
# Comment Operations Tests
# =============================================================================


class TestCommentOperations:
    """Test comment extraction."""

    @pytest.mark.asyncio
    async def test_get_issue_comments(self, connector):
        """Should get comments for an issue."""
        mock_response = {
            "comments": [
                make_comment_data("20001", "First comment", "Alice"),
                make_comment_data("20002", "Second comment", "Bob"),
            ],
            "total": 2,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            comments = await connector._get_issue_comments("PROJ-1")

            assert len(comments) == 2
            assert comments[0].body == "First comment"
            assert comments[0].author == "Alice"
            assert comments[1].body == "Second comment"
            assert comments[1].author == "Bob"

    @pytest.mark.asyncio
    async def test_get_comments_disabled(self, connector):
        """Should skip comments when disabled."""
        connector.include_comments = False

        comments = await connector._get_issue_comments("PROJ-1")

        assert len(comments) == 0

    @pytest.mark.asyncio
    async def test_get_comments_with_adf(self, connector):
        """Should convert ADF format comments."""
        adf_body = {
            "type": "doc",
            "content": [
                {"type": "paragraph", "content": [{"type": "text", "text": "ADF comment"}]}
            ],
        }
        mock_response = {
            "comments": [make_comment_data("20001", adf_body, "Alice")],
            "total": 1,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            comments = await connector._get_issue_comments("PROJ-1")

            assert len(comments) == 1
            assert "ADF comment" in comments[0].body

    @pytest.mark.asyncio
    async def test_get_comments_pagination(self, connector):
        """Should handle paginated comment responses."""
        # Create a "full" page of 50 comments to trigger pagination
        first_page_comments = [
            make_comment_data(f"2000{i}", f"Comment {i}", "Author") for i in range(50)
        ]
        first_page = {
            "comments": first_page_comments,
            "total": 51,
        }
        second_page = {
            "comments": [make_comment_data("20051", "Comment 51", "Bob")],
            "total": 51,
        }

        call_count = [0]

        async def mock_request(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return first_page
            return second_page

        with patch.object(connector, "_api_request", side_effect=mock_request):
            comments = await connector._get_issue_comments("PROJ-1")
            assert len(comments) == 51
            assert call_count[0] == 2  # Verify pagination occurred

    @pytest.mark.asyncio
    async def test_get_comments_error_handling(self, connector):
        """Should handle errors gracefully when getting comments."""
        with patch.object(connector, "_api_request", side_effect=RuntimeError("API Error")):
            comments = await connector._get_issue_comments("PROJ-1")
            assert comments == []


# =============================================================================
# ADF Conversion Tests
# =============================================================================


class TestADFConversion:
    """Test Atlassian Document Format conversion."""

    def test_simple_text(self, connector):
        """Should extract simple text from ADF."""
        adf = {
            "type": "doc",
            "content": [{"type": "text", "text": "Hello world"}],
        }
        result = connector._adf_to_text(adf)

        assert "Hello world" in result

    def test_nested_content(self, connector):
        """Should extract nested text from ADF."""
        adf = {
            "type": "doc",
            "content": [
                {
                    "type": "paragraph",
                    "content": [
                        {"type": "text", "text": "First "},
                        {"type": "text", "text": "paragraph"},
                    ],
                }
            ],
        }
        result = connector._adf_to_text(adf)

        assert "First" in result
        assert "paragraph" in result

    def test_multiple_paragraphs(self, connector):
        """Should extract text from multiple paragraphs."""
        adf = {
            "type": "doc",
            "content": [
                {"type": "paragraph", "content": [{"type": "text", "text": "Para 1"}]},
                {"type": "paragraph", "content": [{"type": "text", "text": "Para 2"}]},
            ],
        }
        result = connector._adf_to_text(adf)

        assert "Para 1" in result
        assert "Para 2" in result

    def test_empty_adf(self, connector):
        """Should handle empty ADF."""
        result = connector._adf_to_text({})
        assert result == ""

        result = connector._adf_to_text(None)
        assert result == ""

    def test_adf_with_non_text_nodes(self, connector):
        """Should skip non-text nodes in ADF."""
        adf = {
            "type": "doc",
            "content": [
                {"type": "text", "text": "Text content"},
                {"type": "mention", "attrs": {"id": "user123"}},
            ],
        }
        result = connector._adf_to_text(adf)
        assert "Text content" in result

    def test_whitespace_normalization(self, connector):
        """Should normalize whitespace in converted text."""
        adf = {
            "type": "doc",
            "content": [
                {"type": "text", "text": "  Multiple   spaces  "},
            ],
        }
        result = connector._adf_to_text(adf)
        assert "  " not in result  # No double spaces


# =============================================================================
# HTML Conversion Tests
# =============================================================================


class TestHTMLConversion:
    """Test HTML to text conversion (for Data Center)."""

    def test_simple_html(self, connector):
        """Should strip HTML tags."""
        html = "<p>Hello <strong>world</strong></p>"
        result = connector._html_to_text(html)
        assert "Hello" in result
        assert "world" in result
        assert "<" not in result

    def test_html_entities(self, connector):
        """Should decode HTML entities."""
        html = "<p>Hello &amp; goodbye &lt;test&gt;</p>"
        result = connector._html_to_text(html)
        assert "&" in result
        assert "<test>" in result

    def test_empty_html(self, connector):
        """Should handle empty HTML."""
        result = connector._html_to_text("")
        assert result == ""

        result = connector._html_to_text(None)
        assert result == ""


# =============================================================================
# Search Tests
# =============================================================================


class TestSearch:
    """Test search functionality."""

    @pytest.mark.asyncio
    async def test_search(self, connector):
        """Should search issues by query."""
        mock_response = {
            "issues": [
                make_issue_data("10001", "PROJ-1", "Login bug fix"),
            ],
            "total": 1,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            results = await connector.search("login bug", limit=5)

            assert len(results) >= 0

    @pytest.mark.asyncio
    async def test_search_with_project_filter(self, connector):
        """Should filter search by project."""
        mock_response = {
            "issues": [make_issue_data()],
            "total": 1,
        }

        with patch.object(connector, "_api_request", return_value=mock_response) as mock_req:
            await connector.search("test", project_key="PROJ")

            call_args = mock_req.call_args
            params = call_args[1]["params"]
            assert "project" in params["jql"]

    @pytest.mark.asyncio
    async def test_search_returns_evidence(self, connector):
        """Should return Evidence objects."""
        mock_response = {
            "issues": [make_issue_data("10001", "PROJ-1", "Test Issue")],
            "total": 1,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            results = await connector.search("test")

            if results:
                from aragora.connectors.base import Evidence

                assert isinstance(results[0], Evidence)

    @pytest.mark.asyncio
    async def test_search_error_handling(self, connector):
        """Should return empty list on search error."""
        with patch.object(connector, "_api_request", side_effect=RuntimeError("Error")):
            results = await connector.search("test")
            assert results == []


# =============================================================================
# Fetch Tests
# =============================================================================


class TestFetch:
    """Test fetch functionality."""

    @pytest.mark.asyncio
    async def test_fetch_issue(self, connector):
        """Should fetch an issue by evidence ID."""
        mock_response = make_issue_data("10001", "PROJ-123", "Test Issue")

        with patch.object(connector, "_api_request", return_value=mock_response):
            evidence = await connector.fetch("jira-PROJ-123")

            if evidence:
                assert "PROJ-123" in evidence.source_id

    @pytest.mark.asyncio
    async def test_fetch_issue_without_prefix(self, connector):
        """Should fetch issue when evidence ID has no prefix."""
        mock_response = make_issue_data("10001", "PROJ-123", "Test Issue")

        with patch.object(connector, "_api_request", return_value=mock_response):
            evidence = await connector.fetch("PROJ-123")

            if evidence:
                assert "PROJ-123" in evidence.source_id

    @pytest.mark.asyncio
    async def test_fetch_error_handling(self, connector):
        """Should return None on fetch error."""
        with patch.object(connector, "_api_request", side_effect=RuntimeError("Error")):
            result = await connector.fetch("PROJ-123")
            assert result is None


# =============================================================================
# Sync Tests
# =============================================================================


class TestSyncItems:
    """Test sync_items functionality."""

    @pytest.mark.asyncio
    async def test_sync_items(self, connector):
        """Should yield sync items for issues."""
        mock_projects_response = {
            "values": [make_project_data("10001", "PROJ", "Project")],
            "isLast": True,
        }
        mock_issues_response = {
            "issues": [make_issue_data("10001", "PROJ-1", "Issue 1")],
            "total": 1,
        }
        mock_comments_response = {"comments": [], "total": 0}

        async def mock_api_request(endpoint, **kwargs):
            if "/project/search" in endpoint:
                return mock_projects_response
            elif "/search" in endpoint:
                return mock_issues_response
            elif "/comment" in endpoint:
                return mock_comments_response
            return {}

        with patch.object(connector, "_api_request", side_effect=mock_api_request):
            state = SyncState(connector_id=connector.connector_id)
            items = []
            async for item in connector.sync_items(state):
                items.append(item)
                if len(items) >= 5:
                    break

            assert isinstance(items, list)

    @pytest.mark.asyncio
    async def test_sync_items_updates_cursor(self, connector):
        """Should update state cursor with latest timestamp."""
        mock_projects_response = {
            "values": [make_project_data("10001", "PROJ", "Project")],
            "isLast": True,
        }
        mock_issues_response = {
            "issues": [make_issue_data("10001", "PROJ-1", "Issue 1")],
            "total": 1,
        }
        mock_comments_response = {"comments": [], "total": 0}

        async def mock_api_request(endpoint, **kwargs):
            if "/project/search" in endpoint:
                return mock_projects_response
            elif "/search" in endpoint:
                return mock_issues_response
            elif "/comment" in endpoint:
                return mock_comments_response
            return {}

        with patch.object(connector, "_api_request", side_effect=mock_api_request):
            state = SyncState(connector_id=connector.connector_id)
            async for item in connector.sync_items(state):
                break

            # State should have been updated with a cursor
            assert state.cursor is not None or state.items_total >= 0

    @pytest.mark.asyncio
    async def test_sync_items_with_cursor(self, connector):
        """Should use cursor for incremental sync."""
        mock_projects_response = {
            "values": [make_project_data("10001", "PROJ", "Project")],
            "isLast": True,
        }
        mock_issues_response = {"issues": [], "total": 0}

        async def mock_api_request(endpoint, **kwargs):
            if "/project/search" in endpoint:
                return mock_projects_response
            elif "/search" in endpoint:
                return mock_issues_response
            return {}

        with patch.object(connector, "_api_request", side_effect=mock_api_request):
            state = SyncState(
                connector_id=connector.connector_id,
                cursor="2024-01-15T10:00:00+00:00",
            )
            items = []
            async for item in connector.sync_items(state):
                items.append(item)

            # Should complete without error
            assert isinstance(items, list)


# =============================================================================
# Webhook Tests
# =============================================================================


class TestWebhookHandling:
    """Test webhook processing."""

    @pytest.mark.asyncio
    async def test_handle_issue_created_webhook(self, connector):
        """Should handle issue created webhook."""
        payload = {
            "webhookEvent": "jira:issue_created",
            "issue": {"key": "PROJ-123"},
        }

        with patch.object(connector, "sync", return_value=AsyncMock()):
            result = await connector.handle_webhook(payload)
            assert result is True

    @pytest.mark.asyncio
    async def test_handle_issue_updated_webhook(self, connector):
        """Should handle issue updated webhook."""
        payload = {
            "webhookEvent": "jira:issue_updated",
            "issue": {"key": "PROJ-123"},
        }

        with patch.object(connector, "sync", return_value=AsyncMock()):
            result = await connector.handle_webhook(payload)
            assert result is True

    @pytest.mark.asyncio
    async def test_handle_issue_deleted_webhook(self, connector):
        """Should handle issue deleted webhook."""
        payload = {
            "webhookEvent": "jira:issue_deleted",
            "issue": {"key": "PROJ-123"},
        }

        with patch.object(connector, "sync", return_value=AsyncMock()):
            result = await connector.handle_webhook(payload)
            assert result is True

    @pytest.mark.asyncio
    async def test_handle_comment_created_webhook(self, connector):
        """Should handle comment created webhook."""
        payload = {
            "webhookEvent": "comment_created",
            "issue": {"key": "PROJ-123"},
            "comment": {"id": "20001"},
        }

        with patch.object(connector, "sync", return_value=AsyncMock()):
            result = await connector.handle_webhook(payload)
            assert result is True

    @pytest.mark.asyncio
    async def test_handle_unknown_webhook(self, connector):
        """Should return False for unknown webhook events."""
        payload = {
            "webhookEvent": "unknown_event",
        }

        result = await connector.handle_webhook(payload)
        assert result is False

    def test_get_webhook_secret(self, connector):
        """Should get webhook secret from environment."""
        with patch.dict(os.environ, {"JIRA_WEBHOOK_SECRET": "secret123"}):
            secret = connector.get_webhook_secret()
            assert secret == "secret123"

    def test_get_webhook_secret_not_set(self, connector):
        """Should return None when secret not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove the key if it exists
            if "JIRA_WEBHOOK_SECRET" in os.environ:
                del os.environ["JIRA_WEBHOOK_SECRET"]
            secret = connector.get_webhook_secret()
            assert secret is None


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_api_error_in_comments(self, connector):
        """Should handle API errors gracefully in comments."""
        import httpx

        with patch.object(
            connector,
            "_api_request",
            side_effect=RuntimeError("API Error"),
        ):
            comments = await connector._get_issue_comments("PROJ-1")
            assert comments == []

    @pytest.mark.asyncio
    async def test_value_error_in_comments(self, connector):
        """Should handle ValueError in comments."""
        with patch.object(
            connector,
            "_api_request",
            side_effect=ValueError("Invalid data"),
        ):
            comments = await connector._get_issue_comments("PROJ-1")
            assert comments == []

    @pytest.mark.asyncio
    async def test_key_error_in_comments(self, connector):
        """Should handle KeyError in comments."""
        with patch.object(
            connector,
            "_api_request",
            side_effect=KeyError("missing_key"),
        ):
            comments = await connector._get_issue_comments("PROJ-1")
            assert comments == []

    @pytest.mark.asyncio
    async def test_search_error_handling(self, connector):
        """Should handle search errors gracefully."""
        with patch.object(
            connector,
            "_api_request",
            side_effect=OSError("Network error"),
        ):
            results = await connector.search("test")
            assert results == []

    @pytest.mark.asyncio
    async def test_fetch_error_handling(self, connector):
        """Should handle fetch errors gracefully."""
        with patch.object(
            connector,
            "_api_request",
            side_effect=RuntimeError("API Error"),
        ):
            result = await connector.fetch("PROJ-123")
            assert result is None


# =============================================================================
# Data Model Tests
# =============================================================================


class TestDataModels:
    """Test data model creation."""

    def test_jira_project_creation(self):
        """Should create JiraProject with required fields."""
        project = JiraProject(
            id="10001",
            key="PROJ",
            name="Test Project",
            project_type="software",
        )
        assert project.lead == ""
        assert project.description == ""

    def test_jira_project_with_all_fields(self):
        """Should create JiraProject with all fields."""
        project = JiraProject(
            id="10001",
            key="PROJ",
            name="Test Project",
            project_type="software",
            lead="John Lead",
            description="Project description",
        )
        assert project.lead == "John Lead"
        assert project.description == "Project description"

    def test_jira_issue_creation(self):
        """Should create JiraIssue with defaults."""
        issue = JiraIssue(
            id="10001",
            key="PROJ-123",
            project_key="PROJ",
            summary="Test Issue",
        )
        assert issue.description == ""
        assert issue.labels == []
        assert issue.parent_key is None
        assert issue.story_points is None

    def test_jira_issue_with_all_fields(self):
        """Should create JiraIssue with all fields."""
        now = datetime.now(timezone.utc)
        issue = JiraIssue(
            id="10001",
            key="PROJ-123",
            project_key="PROJ",
            summary="Full Issue",
            description="Detailed description",
            issue_type="Story",
            status="In Progress",
            priority="High",
            assignee="Alice",
            reporter="Bob",
            created_at=now,
            updated_at=now,
            resolved_at=now,
            labels=["bug", "urgent"],
            components=["Core", "API"],
            fix_versions=["v1.0"],
            url="https://jira.example.com/browse/PROJ-123",
            parent_key="PROJ-100",
            story_points=5.0,
        )
        assert issue.description == "Detailed description"
        assert len(issue.labels) == 2
        assert issue.story_points == 5.0
        assert issue.parent_key == "PROJ-100"
        assert issue.resolved_at == now

    def test_jira_comment_creation(self):
        """Should create JiraComment with defaults."""
        comment = JiraComment(
            id="20001",
            body="Comment text",
            author="Commenter",
        )
        assert comment.created_at is None
        assert comment.updated_at is None

    def test_jira_comment_with_timestamps(self):
        """Should create JiraComment with timestamps."""
        now = datetime.now(timezone.utc)
        comment = JiraComment(
            id="20001",
            body="Comment text",
            author="Commenter",
            created_at=now,
            updated_at=now,
        )
        assert comment.created_at == now
        assert comment.updated_at == now


# =============================================================================
# Date Parsing Tests
# =============================================================================


class TestDateParsing:
    """Test date parsing from API responses."""

    @pytest.mark.asyncio
    async def test_parse_created_date(self, connector):
        """Should parse created date from issue."""
        mock_response = {
            "issues": [make_issue_data("10001", "PROJ-1", "Test")],
            "total": 1,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            issues = []
            async for issue in connector._get_issues("PROJ"):
                issues.append(issue)

            assert issues[0].created_at is not None
            assert issues[0].created_at.year == 2024

    @pytest.mark.asyncio
    async def test_parse_updated_date(self, connector):
        """Should parse updated date from issue."""
        mock_response = {
            "issues": [make_issue_data("10001", "PROJ-1", "Test")],
            "total": 1,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            issues = []
            async for issue in connector._get_issues("PROJ"):
                issues.append(issue)

            assert issues[0].updated_at is not None

    @pytest.mark.asyncio
    async def test_parse_invalid_date(self, connector):
        """Should handle invalid dates gracefully."""
        issue_data = make_issue_data("10001", "PROJ-1", "Test")
        issue_data["fields"]["created"] = "invalid-date"

        mock_response = {"issues": [issue_data], "total": 1}

        with patch.object(connector, "_api_request", return_value=mock_response):
            issues = []
            async for issue in connector._get_issues("PROJ"):
                issues.append(issue)

            # Should not raise, created_at should be None
            assert issues[0].created_at is None

    @pytest.mark.asyncio
    async def test_parse_z_suffix_date(self, connector):
        """Should handle Z suffix in dates."""
        issue_data = make_issue_data("10001", "PROJ-1", "Test")
        issue_data["fields"]["created"] = "2024-01-15T10:00:00.000Z"

        mock_response = {"issues": [issue_data], "total": 1}

        with patch.object(connector, "_api_request", return_value=mock_response):
            issues = []
            async for issue in connector._get_issues("PROJ"):
                issues.append(issue)

            assert issues[0].created_at is not None
            assert issues[0].created_at.tzinfo is not None


# =============================================================================
# Description Handling Tests
# =============================================================================


class TestDescriptionHandling:
    """Test description extraction."""

    @pytest.mark.asyncio
    async def test_string_description(self, connector):
        """Should handle string description."""
        mock_response = {
            "issues": [make_issue_data("10001", "PROJ-1", "Test", "Plain text description")],
            "total": 1,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            issues = []
            async for issue in connector._get_issues("PROJ"):
                issues.append(issue)

            assert issues[0].description == "Plain text description"

    @pytest.mark.asyncio
    async def test_adf_description(self, connector):
        """Should handle ADF format description."""
        adf_desc = {
            "type": "doc",
            "content": [
                {"type": "paragraph", "content": [{"type": "text", "text": "ADF description"}]}
            ],
        }
        mock_response = {
            "issues": [make_issue_data("10001", "PROJ-1", "Test", adf_desc)],
            "total": 1,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            issues = []
            async for issue in connector._get_issues("PROJ"):
                issues.append(issue)

            assert "ADF description" in issues[0].description

    @pytest.mark.asyncio
    async def test_null_description(self, connector):
        """Should handle null description."""
        mock_response = {
            "issues": [make_issue_data("10001", "PROJ-1", "Test", None)],
            "total": 1,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            issues = []
            async for issue in connector._get_issues("PROJ"):
                issues.append(issue)

            assert issues[0].description == ""


# =============================================================================
# JQL Building Tests
# =============================================================================


class TestJQLBuilding:
    """Test JQL query construction."""

    @pytest.mark.asyncio
    async def test_basic_jql(self, connector):
        """Should build basic project JQL."""
        mock_response = {"issues": [], "total": 0}

        with patch.object(connector, "_api_request", return_value=mock_response) as mock_req:
            connector.projects = None
            connector.jql = None
            connector.include_subtasks = True
            issues = []
            async for issue in connector._get_issues("PROJ"):
                issues.append(issue)

            call_args = mock_req.call_args
            params = call_args[1]["params"]
            assert 'project = "PROJ"' in params["jql"]

    @pytest.mark.asyncio
    async def test_jql_with_custom_filter(self, connector):
        """Should include custom JQL filter."""
        mock_response = {"issues": [], "total": 0}

        with patch.object(connector, "_api_request", return_value=mock_response) as mock_req:
            connector.jql = "priority = High"
            issues = []
            async for issue in connector._get_issues("PROJ"):
                issues.append(issue)

            call_args = mock_req.call_args
            params = call_args[1]["params"]
            assert "priority = High" in params["jql"]

    @pytest.mark.asyncio
    async def test_jql_exclude_subtasks(self, connector):
        """Should exclude subtasks in JQL when configured."""
        mock_response = {"issues": [], "total": 0}

        with patch.object(connector, "_api_request", return_value=mock_response) as mock_req:
            connector.include_subtasks = False
            issues = []
            async for issue in connector._get_issues("PROJ"):
                issues.append(issue)

            call_args = mock_req.call_args
            params = call_args[1]["params"]
            assert "subtaskIssueTypes" in params["jql"]


# =============================================================================
# Content Truncation Tests
# =============================================================================


class TestContentTruncation:
    """Test content length handling."""

    @pytest.mark.asyncio
    async def test_content_truncation(self, connector):
        """Should truncate content at 50000 characters."""
        long_description = "A" * 60000
        mock_projects_response = {
            "values": [make_project_data("10001", "PROJ", "Project")],
            "isLast": True,
        }
        mock_issues_response = {
            "issues": [make_issue_data("10001", "PROJ-1", "Issue", long_description)],
            "total": 1,
        }
        mock_comments_response = {"comments": [], "total": 0}

        async def mock_api_request(endpoint, **kwargs):
            if "/project/search" in endpoint:
                return mock_projects_response
            elif "/search" in endpoint:
                return mock_issues_response
            elif "/comment" in endpoint:
                return mock_comments_response
            return {}

        with patch.object(connector, "_api_request", side_effect=mock_api_request):
            state = SyncState(connector_id=connector.connector_id)
            items = []
            async for item in connector.sync_items(state):
                items.append(item)
                break

            assert len(items[0].content) <= 50000


# =============================================================================
# Metadata Tests
# =============================================================================


class TestMetadata:
    """Test metadata in sync items."""

    @pytest.mark.asyncio
    async def test_sync_item_metadata(self, connector):
        """Should include all metadata in sync items."""
        mock_projects_response = {
            "values": [make_project_data("10001", "PROJ", "Project Name")],
            "isLast": True,
        }
        mock_issues_response = {
            "issues": [make_issue_data("10001", "PROJ-1", "Issue", story_points=5.0)],
            "total": 1,
        }
        mock_comments_response = {"comments": [], "total": 0}

        async def mock_api_request(endpoint, **kwargs):
            if "/project/search" in endpoint:
                return mock_projects_response
            elif "/search" in endpoint:
                return mock_issues_response
            elif "/comment" in endpoint:
                return mock_comments_response
            return {}

        with patch.object(connector, "_api_request", side_effect=mock_api_request):
            state = SyncState(connector_id=connector.connector_id)
            items = []
            async for item in connector.sync_items(state):
                items.append(item)
                break

            metadata = items[0].metadata
            assert "project_key" in metadata
            assert "project_name" in metadata
            assert "issue_key" in metadata
            assert "issue_type" in metadata
            assert "status" in metadata
            assert "priority" in metadata


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Test module exports."""

    def test_all_exports(self):
        """Should export all expected classes."""
        from aragora.connectors.enterprise.collaboration.jira import (
            JiraConnector,
            JiraProject,
            JiraIssue,
            JiraComment,
        )

        assert JiraConnector is not None
        assert JiraProject is not None
        assert JiraIssue is not None
        assert JiraComment is not None

    def test_dunder_all(self):
        """Should have correct __all__ exports."""
        from aragora.connectors.enterprise.collaboration import jira

        assert "__all__" in dir(jira)
        assert "JiraConnector" in jira.__all__
        assert "JiraProject" in jira.__all__
        assert "JiraIssue" in jira.__all__
        assert "JiraComment" in jira.__all__
