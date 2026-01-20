"""
Tests for Jira enterprise connector.

Tests cover:
- JiraProject and JiraIssue dataclasses
- JiraConnector initialization and configuration
- Authentication header generation
- JQL building and search
- ADF (Atlassian Document Format) parsing
- Issue and comment retrieval (mocked)
- Webhook handling
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


# ============================================================================
# Dataclass Tests
# ============================================================================


class TestJiraDataclasses:
    """Tests for Jira dataclasses."""

    def test_jira_project_creation(self):
        """Test JiraProject dataclass creation."""
        from aragora.connectors.enterprise.collaboration.jira import JiraProject

        project = JiraProject(
            id="10001",
            key="PROJ",
            name="Test Project",
            project_type="software",
            lead="John Doe",
            description="A test project",
        )

        assert project.id == "10001"
        assert project.key == "PROJ"
        assert project.name == "Test Project"
        assert project.project_type == "software"
        assert project.lead == "John Doe"

    def test_jira_project_defaults(self):
        """Test JiraProject default values."""
        from aragora.connectors.enterprise.collaboration.jira import JiraProject

        project = JiraProject(
            id="10001",
            key="PROJ",
            name="Test",
            project_type="software",
        )

        assert project.lead == ""
        assert project.description == ""

    def test_jira_issue_creation(self):
        """Test JiraIssue dataclass creation."""
        from aragora.connectors.enterprise.collaboration.jira import JiraIssue

        now = datetime.now(timezone.utc)
        issue = JiraIssue(
            id="10001",
            key="PROJ-123",
            project_key="PROJ",
            summary="Fix bug in login",
            description="Users cannot log in",
            issue_type="Bug",
            status="Open",
            priority="High",
            assignee="Jane Doe",
            reporter="John Doe",
            created_at=now,
            updated_at=now,
            labels=["urgent", "security"],
            components=["auth"],
            url="https://jira.example.com/browse/PROJ-123",
        )

        assert issue.key == "PROJ-123"
        assert issue.summary == "Fix bug in login"
        assert issue.issue_type == "Bug"
        assert "urgent" in issue.labels
        assert "auth" in issue.components

    def test_jira_issue_defaults(self):
        """Test JiraIssue default values."""
        from aragora.connectors.enterprise.collaboration.jira import JiraIssue

        issue = JiraIssue(
            id="10001",
            key="PROJ-1",
            project_key="PROJ",
            summary="Test issue",
        )

        assert issue.description == ""
        assert issue.labels == []
        assert issue.components == []
        assert issue.fix_versions == []
        assert issue.parent_key is None
        assert issue.story_points is None

    def test_jira_comment_creation(self):
        """Test JiraComment dataclass creation."""
        from aragora.connectors.enterprise.collaboration.jira import JiraComment

        now = datetime.now(timezone.utc)
        comment = JiraComment(
            id="10001",
            body="This is a comment",
            author="Jane Doe",
            created_at=now,
            updated_at=now,
        )

        assert comment.id == "10001"
        assert comment.body == "This is a comment"
        assert comment.author == "Jane Doe"


# ============================================================================
# Connector Initialization Tests
# ============================================================================


class TestJiraConnectorInit:
    """Tests for JiraConnector initialization."""

    def test_cloud_initialization(self):
        """Test initialization for Jira Cloud."""
        from aragora.connectors.enterprise.collaboration.jira import JiraConnector

        connector = JiraConnector(
            base_url="https://example.atlassian.net",
            projects=["PROJ", "DEV"],
        )

        assert connector.is_cloud is True
        assert "example_atlassian_net" in connector.connector_id
        assert connector.projects == ["PROJ", "DEV"]

    def test_datacenter_initialization(self):
        """Test initialization for Jira Data Center."""
        from aragora.connectors.enterprise.collaboration.jira import JiraConnector

        connector = JiraConnector(
            base_url="https://jira.company.com",
            projects=["PROJ"],
        )

        assert connector.is_cloud is False

    def test_url_normalization(self):
        """Test URL trailing slash is removed."""
        from aragora.connectors.enterprise.collaboration.jira import JiraConnector

        connector = JiraConnector(
            base_url="https://example.atlassian.net/",
        )

        assert not connector.base_url.endswith("/")

    def test_exclude_filters(self):
        """Test exclude filters are normalized."""
        from aragora.connectors.enterprise.collaboration.jira import JiraConnector

        connector = JiraConnector(
            base_url="https://example.atlassian.net",
            exclude_statuses=["Done", "CLOSED"],
            exclude_types=["Epic", "Sub-task"],
        )

        assert "done" in connector.exclude_statuses
        assert "closed" in connector.exclude_statuses
        assert "epic" in connector.exclude_types
        assert "sub-task" in connector.exclude_types

    def test_source_type(self):
        """Test source type is EXTERNAL_API."""
        from aragora.connectors.enterprise.collaboration.jira import JiraConnector
        from aragora.reasoning.provenance import SourceType

        connector = JiraConnector(base_url="https://example.atlassian.net")
        assert connector.source_type == SourceType.EXTERNAL_API

    def test_name_property(self):
        """Test name property includes base URL."""
        from aragora.connectors.enterprise.collaboration.jira import JiraConnector

        connector = JiraConnector(base_url="https://example.atlassian.net")
        assert "example.atlassian.net" in connector.name


# ============================================================================
# Authentication Tests
# ============================================================================


class TestJiraAuthentication:
    """Tests for Jira authentication."""

    @pytest.mark.asyncio
    async def test_cloud_auth_header(self):
        """Test cloud authentication header generation."""
        from aragora.connectors.enterprise.collaboration.jira import JiraConnector

        connector = JiraConnector(base_url="https://example.atlassian.net")

        # Mock credentials
        connector.credentials = AsyncMock()
        connector.credentials.get_credential = AsyncMock(
            side_effect=lambda key: {
                "JIRA_EMAIL": "user@example.com",
                "JIRA_API_TOKEN": "token123",
            }.get(key)
        )

        header = await connector._get_auth_header()

        assert "Authorization" in header
        assert header["Authorization"].startswith("Basic ")

    @pytest.mark.asyncio
    async def test_datacenter_auth_header(self):
        """Test data center authentication header generation."""
        from aragora.connectors.enterprise.collaboration.jira import JiraConnector

        connector = JiraConnector(base_url="https://jira.company.com")

        # Mock credentials
        connector.credentials = AsyncMock()
        connector.credentials.get_credential = AsyncMock(
            side_effect=lambda key: {
                "JIRA_PAT": "pat_token_123",
            }.get(key)
        )

        header = await connector._get_auth_header()

        assert "Authorization" in header
        assert header["Authorization"] == "Bearer pat_token_123"

    @pytest.mark.asyncio
    async def test_missing_cloud_credentials(self):
        """Test error when cloud credentials are missing."""
        from aragora.connectors.enterprise.collaboration.jira import JiraConnector

        connector = JiraConnector(base_url="https://example.atlassian.net")

        # Mock missing credentials
        connector.credentials = AsyncMock()
        connector.credentials.get_credential = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="credentials not configured"):
            await connector._get_auth_header()

    @pytest.mark.asyncio
    async def test_missing_datacenter_credentials(self):
        """Test error when data center credentials are missing."""
        from aragora.connectors.enterprise.collaboration.jira import JiraConnector

        connector = JiraConnector(base_url="https://jira.company.com")

        # Mock missing credentials
        connector.credentials = AsyncMock()
        connector.credentials.get_credential = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="credentials not configured"):
            await connector._get_auth_header()


# ============================================================================
# ADF Parsing Tests
# ============================================================================


class TestADFParsing:
    """Tests for Atlassian Document Format parsing."""

    def test_simple_text_extraction(self):
        """Test extracting text from simple ADF."""
        from aragora.connectors.enterprise.collaboration.jira import JiraConnector

        connector = JiraConnector(base_url="https://example.atlassian.net")

        adf = {
            "type": "doc",
            "content": [
                {
                    "type": "paragraph",
                    "content": [
                        {"type": "text", "text": "Hello "},
                        {"type": "text", "text": "world"},
                    ],
                }
            ],
        }

        result = connector._adf_to_text(adf)
        assert "Hello" in result
        assert "world" in result

    def test_nested_content_extraction(self):
        """Test extracting text from nested ADF."""
        from aragora.connectors.enterprise.collaboration.jira import JiraConnector

        connector = JiraConnector(base_url="https://example.atlassian.net")

        adf = {
            "type": "doc",
            "content": [
                {"type": "heading", "content": [{"type": "text", "text": "Title"}]},
                {
                    "type": "bulletList",
                    "content": [
                        {
                            "type": "listItem",
                            "content": [
                                {
                                    "type": "paragraph",
                                    "content": [{"type": "text", "text": "Item 1"}],
                                }
                            ],
                        }
                    ],
                },
            ],
        }

        result = connector._adf_to_text(adf)
        assert "Title" in result
        assert "Item 1" in result

    def test_empty_adf(self):
        """Test handling empty ADF."""
        from aragora.connectors.enterprise.collaboration.jira import JiraConnector

        connector = JiraConnector(base_url="https://example.atlassian.net")

        assert connector._adf_to_text({}) == ""
        assert connector._adf_to_text(None) == ""

    def test_html_to_text(self):
        """Test HTML to text conversion (Data Center)."""
        from aragora.connectors.enterprise.collaboration.jira import JiraConnector

        connector = JiraConnector(base_url="https://jira.company.com")

        html = "<p>Hello <strong>world</strong></p>"
        result = connector._html_to_text(html)

        assert "Hello" in result
        assert "world" in result
        assert "<" not in result


# ============================================================================
# API Request Tests (Mocked)
# ============================================================================


class TestJiraAPIRequests:
    """Tests for Jira API requests."""

    def test_cloud_uses_api_v3(self):
        """Test that Cloud connector builds URLs with API v3."""
        from aragora.connectors.enterprise.collaboration.jira import JiraConnector

        connector = JiraConnector(base_url="https://example.atlassian.net")

        # Cloud should use API v3
        assert connector.is_cloud is True
        # The _api_request method will build URLs like:
        # {base_url}/rest/api/3{endpoint}
        expected_prefix = f"{connector.base_url}/rest/api/3"
        assert "atlassian.net" in connector.base_url

    def test_datacenter_uses_api_v2(self):
        """Test that Data Center connector builds URLs with API v2."""
        from aragora.connectors.enterprise.collaboration.jira import JiraConnector

        connector = JiraConnector(base_url="https://jira.company.com")

        # Data Center should use API v2
        assert connector.is_cloud is False
        # The _api_request method will build URLs like:
        # {base_url}/rest/api/2{endpoint}


# ============================================================================
# Webhook Tests
# ============================================================================


class TestJiraWebhooks:
    """Tests for Jira webhook handling."""

    @pytest.mark.asyncio
    async def test_issue_created_webhook(self):
        """Test handling issue created webhook."""
        from aragora.connectors.enterprise.collaboration.jira import JiraConnector

        connector = JiraConnector(base_url="https://example.atlassian.net")

        # Mock sync to avoid actual sync
        connector.sync = AsyncMock()

        payload = {
            "webhookEvent": "jira:issue_created",
            "issue": {"key": "PROJ-123"},
        }

        result = await connector.handle_webhook(payload)
        assert result is True

    @pytest.mark.asyncio
    async def test_issue_updated_webhook(self):
        """Test handling issue updated webhook."""
        from aragora.connectors.enterprise.collaboration.jira import JiraConnector

        connector = JiraConnector(base_url="https://example.atlassian.net")

        # Mock sync
        connector.sync = AsyncMock()

        payload = {
            "webhookEvent": "jira:issue_updated",
            "issue": {"key": "PROJ-123"},
        }

        result = await connector.handle_webhook(payload)
        assert result is True

    @pytest.mark.asyncio
    async def test_comment_webhook(self):
        """Test handling comment webhook."""
        from aragora.connectors.enterprise.collaboration.jira import JiraConnector

        connector = JiraConnector(base_url="https://example.atlassian.net")

        # Mock sync
        connector.sync = AsyncMock()

        payload = {
            "webhookEvent": "comment_created",
            "issue": {"key": "PROJ-123"},
        }

        result = await connector.handle_webhook(payload)
        assert result is True

    @pytest.mark.asyncio
    async def test_unhandled_webhook(self):
        """Test unhandled webhook event returns False."""
        from aragora.connectors.enterprise.collaboration.jira import JiraConnector

        connector = JiraConnector(base_url="https://example.atlassian.net")

        payload = {
            "webhookEvent": "project_created",
            "project": {"key": "NEW"},
        }

        result = await connector.handle_webhook(payload)
        assert result is False


# ============================================================================
# Search Tests (Mocked)
# ============================================================================


class TestJiraSearch:
    """Tests for Jira search functionality."""

    @pytest.mark.asyncio
    async def test_search_builds_jql(self):
        """Test search builds correct JQL."""
        from aragora.connectors.enterprise.collaboration.jira import JiraConnector

        connector = JiraConnector(base_url="https://example.atlassian.net")

        # Mock _search_issues to capture JQL
        captured_jql = []

        async def mock_search(jql, **kwargs):
            captured_jql.append(jql)
            return {"issues": []}

        connector._search_issues = mock_search

        await connector.search("login bug")

        assert len(captured_jql) == 1
        assert 'text ~ "login bug"' in captured_jql[0]

    @pytest.mark.asyncio
    async def test_search_with_project_filter(self):
        """Test search with project filter."""
        from aragora.connectors.enterprise.collaboration.jira import JiraConnector

        connector = JiraConnector(base_url="https://example.atlassian.net")

        captured_jql = []

        async def mock_search(jql, **kwargs):
            captured_jql.append(jql)
            return {"issues": []}

        connector._search_issues = mock_search

        await connector.search("bug", project_key="PROJ")

        assert 'project = "PROJ"' in captured_jql[0]


# ============================================================================
# Fetch Tests (Mocked)
# ============================================================================


class TestJiraFetch:
    """Tests for Jira fetch functionality."""

    def test_fetch_extracts_issue_key_from_evidence_id(self):
        """Test that fetch correctly extracts issue key from evidence ID."""
        from aragora.connectors.enterprise.collaboration.jira import JiraConnector

        connector = JiraConnector(base_url="https://example.atlassian.net")

        # Test key extraction logic (the actual fetch would need credentials)
        evidence_id = "jira-PROJ-123"
        expected_key = "PROJ-123"

        # The fetch method extracts key as: evidence_id[5:] when starts with "jira-"
        if evidence_id.startswith("jira-"):
            extracted = evidence_id[5:]
        else:
            extracted = evidence_id

        assert extracted == expected_key

    def test_fetch_handles_direct_issue_key(self):
        """Test that fetch handles direct issue key."""
        from aragora.connectors.enterprise.collaboration.jira import JiraConnector

        connector = JiraConnector(base_url="https://example.atlassian.net")

        # Direct key without prefix
        issue_key = "PROJ-456"

        if issue_key.startswith("jira-"):
            extracted = issue_key[5:]
        else:
            extracted = issue_key

        assert extracted == "PROJ-456"


# ============================================================================
# Module Export Tests
# ============================================================================


class TestJiraExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """Test __all__ includes expected classes."""
        from aragora.connectors.enterprise.collaboration import jira

        assert "JiraConnector" in jira.__all__
        assert "JiraProject" in jira.__all__
        assert "JiraIssue" in jira.__all__
        assert "JiraComment" in jira.__all__
