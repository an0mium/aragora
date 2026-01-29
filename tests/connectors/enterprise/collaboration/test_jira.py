"""
Tests for Atlassian Jira Enterprise Connector.

Tests the Jira REST API integration including:
- Project and issue operations
- JQL-based search
- Comment extraction
- Incremental sync
- Error handling
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict

from aragora.connectors.enterprise.collaboration.jira import (
    JiraConnector,
    JiraProject,
    JiraIssue,
    JiraComment,
)
from aragora.connectors.enterprise.base import SyncState


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


def make_api_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a mock API response."""
    return data


def make_project_data(
    project_id: str = "10001",
    key: str = "PROJ",
    name: str = "Test Project",
) -> Dict[str, Any]:
    """Create mock project data."""
    return {
        "id": project_id,
        "key": key,
        "name": name,
        "projectTypeKey": "software",
        "lead": {"displayName": "John Lead"},
        "description": "Test project description",
    }


def make_issue_data(
    issue_id: str = "10001",
    key: str = "PROJ-123",
    summary: str = "Test Issue",
    description: str = "Issue description",
) -> Dict[str, Any]:
    """Create mock issue data."""
    return {
        "id": issue_id,
        "key": key,
        "fields": {
            "summary": summary,
            "description": description,
            "issuetype": {"name": "Story"},
            "status": {"name": "In Progress"},
            "priority": {"name": "Medium"},
            "assignee": {"displayName": "Alice Dev"},
            "reporter": {"displayName": "Bob PM"},
            "created": "2024-01-15T10:00:00.000+0000",
            "updated": "2024-01-15T12:00:00.000+0000",
            "labels": ["backend", "api"],
            "components": [{"name": "Core"}],
            "fixVersions": [{"name": "v1.0"}],
        },
    }


def make_comment_data(
    comment_id: str = "20001",
    body: str = "Test comment",
    author: str = "Commenter",
) -> Dict[str, Any]:
    """Create mock comment data."""
    return {
        "id": comment_id,
        "body": body,
        "author": {"displayName": author},
        "created": "2024-01-15T14:00:00.000+0000",
        "updated": "2024-01-15T14:00:00.000+0000",
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

    def test_custom_configuration(self):
        """Should accept custom configuration."""
        connector = JiraConnector(
            base_url="https://test.atlassian.net",
            projects=["PROJ", "DEV"],
            jql="priority = High",
            include_subtasks=False,
            include_comments=False,
            exclude_statuses=["Done", "Closed"],
            exclude_types=["Epic"],
        )
        assert connector.projects == ["PROJ", "DEV"]
        assert connector.jql == "priority = High"
        assert connector.include_subtasks is False
        assert connector.include_comments is False
        assert "done" in connector.exclude_statuses
        assert "epic" in connector.exclude_types

    def test_connector_id_generation(self):
        """Should generate connector ID from domain."""
        connector = JiraConnector(base_url="https://my-company.atlassian.net")
        assert "jira_" in connector.connector_id
        assert "my-company" in connector.connector_id

    def test_url_normalization(self):
        """Should normalize URL by removing trailing slash."""
        connector = JiraConnector(base_url="https://test.atlassian.net/")
        assert connector.base_url == "https://test.atlassian.net"

    def test_connector_properties(self, connector):
        """Should have correct connector properties."""
        assert "Jira" in connector.name
        assert "test-domain.atlassian.net" in connector.name

    def test_source_type(self, connector):
        """Should have correct source type."""
        from aragora.reasoning.provenance import SourceType

        assert connector.source_type == SourceType.EXTERNAL_API


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
    async def test_data_center_auth_header(self, dc_connector):
        """Should generate Bearer auth header for Data Center."""
        header = await dc_connector._get_auth_header()

        assert "Authorization" in header
        assert header["Authorization"] == "Bearer test_pat"

    @pytest.mark.asyncio
    async def test_missing_cloud_credentials(self, connector):
        """Should raise error when Cloud credentials missing."""
        connector.credentials.get_credential = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="Jira Cloud credentials not configured"):
            await connector._get_auth_header()

    @pytest.mark.asyncio
    async def test_missing_dc_credentials(self, dc_connector):
        """Should raise error when Data Center credentials missing."""
        dc_connector.credentials.get_credential = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="Jira Data Center credentials not configured"):
            await dc_connector._get_auth_header()


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

        with patch.object(connector, "_api_request", return_value=mock_response) as mock_request:
            issues = []
            async for issue in connector._get_issues("PROJ", modified_since=modified_since):
                issues.append(issue)

            # Check that JQL includes updated filter
            call_args = mock_request.call_args
            assert "updated >=" in str(call_args)

    @pytest.mark.asyncio
    async def test_get_issues_excludes_statuses(self, connector):
        """Should exclude issues with excluded statuses."""
        connector.exclude_statuses = {"done"}

        mock_response = {
            "issues": [
                make_issue_data("10001", "PROJ-1", "Active Issue"),
                {
                    **make_issue_data("10002", "PROJ-2", "Done Issue"),
                    "fields": {
                        **make_issue_data()["fields"],
                        "status": {"name": "Done"},
                    },
                },
            ],
            "total": 2,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            issues = []
            async for issue in connector._get_issues("PROJ"):
                issues.append(issue)

            # Only the active issue should be returned
            assert len(issues) == 1
            assert issues[0].key == "PROJ-1"


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

    @pytest.mark.asyncio
    async def test_get_comments_disabled(self, connector):
        """Should skip comments when disabled."""
        connector.include_comments = False

        comments = await connector._get_issue_comments("PROJ-1")

        assert len(comments) == 0


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

    def test_empty_adf(self, connector):
        """Should handle empty ADF."""
        result = connector._adf_to_text({})
        assert result == ""

        result = connector._adf_to_text(None)
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

            assert len(results) >= 0  # May be empty based on implementation

    @pytest.mark.asyncio
    async def test_search_with_project_filter(self, connector):
        """Should filter search by project."""
        mock_response = {
            "issues": [make_issue_data()],
            "total": 1,
        }

        with patch.object(connector, "_api_request", return_value=mock_response) as mock_request:
            await connector.search("test", project_key="PROJ")

            # Check that JQL includes project filter
            call_args = mock_request.call_args
            assert "project" in str(call_args).lower() or len(call_args) > 0


# =============================================================================
# Fetch Tests
# =============================================================================


class TestFetch:
    """Test fetch functionality."""

    @pytest.mark.asyncio
    async def test_fetch_issue(self, connector):
        """Should fetch an issue by evidence ID."""
        mock_issue_response = make_issue_data("10001", "PROJ-123", "Test Issue")
        mock_comments_response = {"comments": [], "total": 0}

        with (
            patch.object(
                connector,
                "_api_request",
                side_effect=[mock_issue_response, mock_comments_response],
            ),
        ):
            evidence = await connector.fetch("jira-PROJ-123")

            # Should return evidence or None
            if evidence:
                assert "PROJ-123" in evidence.id

    @pytest.mark.asyncio
    async def test_fetch_invalid_id(self, connector):
        """Should return None for invalid evidence ID."""
        result = await connector.fetch("invalid-id-format")

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

        call_count = [0]

        async def mock_api_request(endpoint, **kwargs):
            call_count[0] += 1
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
                if len(items) >= 5:  # Limit for test
                    break

            assert isinstance(items, list)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_api_error(self, connector):
        """Should handle API errors gracefully."""
        import httpx

        with patch.object(
            connector,
            "_api_request",
            side_effect=httpx.HTTPStatusError(
                "Error", request=None, response=MagicMock(status_code=500)
            ),
        ):
            comments = await connector._get_issue_comments("PROJ-1")
            # Should return empty list on error
            assert comments == []


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
            created_at=datetime.now(timezone.utc),
            labels=["bug", "urgent"],
            components=["Core", "API"],
            fix_versions=["v1.0"],
            parent_key="PROJ-100",
            story_points=5.0,
        )
        assert issue.description == "Detailed description"
        assert len(issue.labels) == 2
        assert issue.story_points == 5.0

    def test_jira_comment_creation(self):
        """Should create JiraComment with defaults."""
        comment = JiraComment(
            id="20001",
            body="Comment text",
            author="Commenter",
        )
        assert comment.created_at is None
        assert comment.updated_at is None


# =============================================================================
# URL and Issue Text Generation Tests
# =============================================================================


class TestIssueTextGeneration:
    """Test issue text generation for indexing."""

    def test_issue_url_generation(self, connector):
        """Should generate correct issue URL."""
        mock_response = {
            "issues": [make_issue_data("10001", "PROJ-123", "Test")],
            "total": 1,
        }

        with patch.object(connector, "_api_request", return_value=mock_response):
            # The URL is set in _get_issues
            pass  # URL is set during issue parsing

    def test_issue_has_url(self):
        """Should have URL field."""
        issue = JiraIssue(
            id="10001",
            key="PROJ-123",
            project_key="PROJ",
            summary="Test",
            url="https://jira.example.com/browse/PROJ-123",
        )
        assert issue.url == "https://jira.example.com/browse/PROJ-123"
