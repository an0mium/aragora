"""
Extended tests for Linear Connector.

Covers functionality NOT already tested in test_linear.py:
- Issue operations: archive, assign, close with all optional params
- Create/update issue with all optional fields (cycle, parent, labels, estimate, due_date)
- Project management: get single project, projects with team_id filter
- Cycle operations: active cycle when none exists
- Comment operations: add_comment
- Enterprise connector interface: connect, disconnect, search (Evidence), fetch, sync
- Pagination (cursor-based)
- GraphQL query construction details
- Error handling edge cases
- Data model edge cases (missing fields, None values, nested structures)
- Helper functions (_parse_datetime)
- Mock helpers (get_mock_issue, get_mock_team)
- Context manager (__aenter__, __aexit__)
- HTTP client lifecycle (_get_client reuse)
- Full sync and incremental sync
- Labels with team_id filter
- Projects with team_id filter
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from typing import Any

from aragora.connectors.enterprise.collaboration.linear import (
    LinearConnector,
    LinearCredentials,
    LinearIssue,
    LinearTeam,
    LinearUser,
    IssueState,
    IssueStateType,
    IssuePriority,
    Label,
    Project,
    Cycle,
    Comment,
    LinearError,
    _parse_datetime,
    get_mock_issue,
    get_mock_team,
)
from aragora.connectors.enterprise.base import SyncState, SyncResult
from aragora.connectors.base import Evidence
from aragora.reasoning.provenance import SourceType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def credentials():
    """Create test credentials."""
    return LinearCredentials(
        api_key="lin_api_test_key_extended",
        base_url="https://api.linear.app/graphql",
    )


@pytest.fixture
def connector(credentials):
    """Create test connector."""
    return LinearConnector(credentials=credentials, tenant_id="test-tenant")


def make_graphql_response(data: dict[str, Any]) -> MagicMock:
    """Create a mock GraphQL response."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"data": data}
    return response


def make_error_response(status_code: int = 400, text: str = "Error") -> MagicMock:
    """Create a mock error response."""
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    return response


def make_graphql_error_response(
    message: str = "GraphQL Error", errors: list | None = None
) -> MagicMock:
    """Create a mock GraphQL error response."""
    response = MagicMock()
    response.status_code = 200
    if errors is None:
        errors = [{"message": message}]
    response.json.return_value = {"errors": errors}
    return response


def _mock_client_for(connector, mock_response):
    """Set up a mock client returning mock_response on connector._graphql calls."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    return mock_client


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestParseDatetime:
    """Test the _parse_datetime helper function."""

    def test_parse_iso_datetime_with_z(self):
        """Should parse ISO datetime ending with Z."""
        result = _parse_datetime("2024-01-15T10:30:00Z")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30

    def test_parse_iso_datetime_with_offset(self):
        """Should parse ISO datetime with timezone offset."""
        result = _parse_datetime("2024-06-20T14:00:00+00:00")
        assert result is not None
        assert result.year == 2024
        assert result.month == 6

    def test_parse_none_value(self):
        """Should return None for None input."""
        assert _parse_datetime(None) is None

    def test_parse_empty_string(self):
        """Should return None for empty string."""
        assert _parse_datetime("") is None

    def test_parse_invalid_string(self):
        """Should return None for invalid datetime string."""
        assert _parse_datetime("not-a-date") is None

    def test_parse_date_only_string(self):
        """Should parse date-only string."""
        result = _parse_datetime("2024-03-31")
        assert result is not None
        assert result.year == 2024
        assert result.month == 3
        assert result.day == 31


# =============================================================================
# Mock Helper Tests
# =============================================================================


class TestMockHelpers:
    """Test the get_mock_issue and get_mock_team helpers."""

    def test_get_mock_issue(self):
        """Should return a valid mock issue."""
        issue = get_mock_issue()
        assert issue.id == "issue-123"
        assert issue.identifier == "ENG-42"
        assert issue.title == "Implement user authentication"
        assert issue.description == "Add OAuth2 support for user login"
        assert issue.priority == IssuePriority.HIGH
        assert issue.state_name == "In Progress"
        assert issue.state_type == IssueStateType.STARTED
        assert issue.team_key == "ENG"
        assert issue.created_at is not None

    def test_get_mock_team(self):
        """Should return a valid mock team."""
        team = get_mock_team()
        assert team.id == "team-123"
        assert team.name == "Engineering"
        assert team.key == "ENG"
        assert team.description == "Core engineering team"


# =============================================================================
# LinearCredentials Tests
# =============================================================================


class TestLinearCredentials:
    """Test LinearCredentials dataclass."""

    def test_default_base_url(self):
        """Should use default base URL if not specified."""
        creds = LinearCredentials(api_key="test-key")
        assert creds.base_url == "https://api.linear.app/graphql"

    def test_custom_base_url(self):
        """Should accept custom base URL."""
        creds = LinearCredentials(api_key="test-key", base_url="https://custom.linear.app/graphql")
        assert creds.base_url == "https://custom.linear.app/graphql"


# =============================================================================
# LinearError Tests
# =============================================================================


class TestLinearError:
    """Test LinearError exception class."""

    def test_error_with_message(self):
        """Should store error message."""
        err = LinearError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.errors == []

    def test_error_with_errors_list(self):
        """Should store error details list."""
        errors = [{"message": "Field error", "locations": [{"line": 1}]}]
        err = LinearError("Query failed", errors=errors)
        assert str(err) == "Query failed"
        assert len(err.errors) == 1
        assert err.errors[0]["message"] == "Field error"

    def test_error_with_none_errors(self):
        """Should default to empty list when errors is None."""
        err = LinearError("Error", errors=None)
        assert err.errors == []


# =============================================================================
# Data Model Edge Case Tests
# =============================================================================


class TestDataModelEdgeCases:
    """Test data model parsing with missing/edge-case fields."""

    def test_linear_issue_from_api_minimal(self):
        """Should parse issue with minimal data (all optional fields missing)."""
        data = {"id": "i1", "identifier": "T-1", "title": "Minimal"}
        issue = LinearIssue.from_api(data)
        assert issue.id == "i1"
        assert issue.identifier == "T-1"
        assert issue.title == "Minimal"
        assert issue.description is None
        assert issue.priority == IssuePriority.NO_PRIORITY
        assert issue.estimate is None
        assert issue.state_id is None
        assert issue.state_name is None
        assert issue.state_type is None
        assert issue.team_id is None
        assert issue.assignee_id is None
        assert issue.creator_id is None
        assert issue.project_id is None
        assert issue.cycle_id is None
        assert issue.parent_id is None
        assert issue.label_ids == []
        assert issue.subscriber_ids == []
        assert issue.url is None
        assert issue.branch_name is None
        assert issue.due_date is None
        assert issue.created_at is None
        assert issue.updated_at is None
        assert issue.completed_at is None
        assert issue.canceled_at is None
        assert issue.archived_at is None

    def test_linear_issue_from_api_full(self):
        """Should parse issue with all fields populated."""
        data = {
            "id": "i1",
            "identifier": "ENG-100",
            "title": "Full Issue",
            "description": "Full description",
            "priority": 1,
            "estimate": 5,
            "url": "https://linear.app/team/issue/ENG-100",
            "branchName": "eng-100-full-issue",
            "dueDate": "2024-06-30",
            "createdAt": "2024-01-01T00:00:00Z",
            "updatedAt": "2024-01-15T12:00:00Z",
            "completedAt": "2024-02-01T00:00:00Z",
            "canceledAt": None,
            "archivedAt": None,
            "state": {"id": "s1", "name": "Done", "type": "completed"},
            "team": {"id": "t1", "key": "ENG"},
            "assignee": {"id": "u1", "name": "Alice"},
            "creator": {"id": "u2"},
            "project": {"id": "p1"},
            "cycle": {"id": "c1"},
            "parent": {"id": "parent-i1"},
            "labels": {"nodes": [{"id": "l1"}, {"id": "l2"}]},
            "subscribers": {"nodes": [{"id": "u1"}, {"id": "u3"}]},
        }
        issue = LinearIssue.from_api(data)
        assert issue.priority == IssuePriority.URGENT
        assert issue.estimate == 5
        assert issue.url == "https://linear.app/team/issue/ENG-100"
        assert issue.branch_name == "eng-100-full-issue"
        assert issue.state_id == "s1"
        assert issue.state_name == "Done"
        assert issue.state_type == IssueStateType.COMPLETED
        assert issue.team_id == "t1"
        assert issue.team_key == "ENG"
        assert issue.assignee_id == "u1"
        assert issue.assignee_name == "Alice"
        assert issue.creator_id == "u2"
        assert issue.project_id == "p1"
        assert issue.cycle_id == "c1"
        assert issue.parent_id == "parent-i1"
        assert issue.label_ids == ["l1", "l2"]
        assert issue.subscriber_ids == ["u1", "u3"]
        assert issue.due_date is not None
        assert issue.completed_at is not None

    def test_linear_issue_empty_nested_objects(self):
        """Should handle empty nested objects (None state, team, etc.)."""
        data = {
            "id": "i1",
            "identifier": "T-1",
            "title": "Test",
            "state": None,
            "team": None,
            "assignee": None,
            "creator": None,
            "project": None,
            "cycle": None,
            "parent": None,
            "labels": {},
            "subscribers": {},
        }
        issue = LinearIssue.from_api(data)
        assert issue.state_id is None
        assert issue.team_id is None
        assert issue.label_ids == []
        assert issue.subscriber_ids == []

    def test_project_from_api_with_lead_and_teams(self):
        """Should parse project with lead and teams."""
        data = {
            "id": "p1",
            "name": "Project Alpha",
            "description": "Alpha project",
            "icon": "rocket",
            "color": "#ff0000",
            "state": "started",
            "progress": 0.75,
            "targetDate": "2024-12-31",
            "startedAt": "2024-01-01T00:00:00Z",
            "completedAt": None,
            "createdAt": "2024-01-01T00:00:00Z",
            "updatedAt": "2024-06-01T00:00:00Z",
            "lead": {"id": "u1"},
            "teams": {"nodes": [{"id": "t1"}, {"id": "t2"}]},
        }
        project = Project.from_api(data)
        assert project.id == "p1"
        assert project.name == "Project Alpha"
        assert project.state == "started"
        assert project.progress == 0.75
        assert project.lead_id == "u1"
        assert project.team_ids == ["t1", "t2"]
        assert project.target_date is not None
        assert project.started_at is not None

    def test_project_from_api_minimal(self):
        """Should parse project with minimal fields."""
        data = {"id": "p1", "name": "Min Project"}
        project = Project.from_api(data)
        assert project.id == "p1"
        assert project.name == "Min Project"
        assert project.lead_id is None
        assert project.team_ids == []
        assert project.state == "planned"
        assert project.progress == 0

    def test_cycle_from_api_full(self):
        """Should parse cycle with all fields."""
        data = {
            "id": "c1",
            "number": 5,
            "name": "Sprint 5",
            "description": "Fifth sprint",
            "startsAt": "2024-03-01T00:00:00Z",
            "endsAt": "2024-03-14T23:59:59Z",
            "completedAt": "2024-03-14T22:00:00Z",
            "progress": 0.95,
            "scopeTarget": 42,
            "team": {"id": "t1"},
        }
        cycle = Cycle.from_api(data)
        assert cycle.id == "c1"
        assert cycle.number == 5
        assert cycle.name == "Sprint 5"
        assert cycle.description == "Fifth sprint"
        assert cycle.starts_at is not None
        assert cycle.ends_at is not None
        assert cycle.completed_at is not None
        assert cycle.progress == 0.95
        assert cycle.scope_target == 42
        assert cycle.team_id == "t1"

    def test_cycle_from_api_minimal(self):
        """Should parse cycle with minimal data."""
        data = {"id": "c1", "number": 1}
        cycle = Cycle.from_api(data)
        assert cycle.id == "c1"
        assert cycle.number == 1
        assert cycle.name is None
        assert cycle.team_id is None
        assert cycle.scope_target is None

    def test_comment_from_api_full(self):
        """Should parse comment with user info."""
        data = {
            "id": "com1",
            "body": "Great progress!",
            "user": {"id": "u1", "name": "Alice"},
            "createdAt": "2024-01-15T10:00:00Z",
            "updatedAt": "2024-01-15T11:00:00Z",
        }
        comment = Comment.from_api(data)
        assert comment.id == "com1"
        assert comment.body == "Great progress!"
        assert comment.user_id == "u1"
        assert comment.user_name == "Alice"
        assert comment.created_at is not None
        assert comment.updated_at is not None

    def test_comment_from_api_no_user(self):
        """Should parse comment without user info."""
        data = {"id": "com1", "body": "Anonymous comment"}
        comment = Comment.from_api(data)
        assert comment.id == "com1"
        assert comment.user_id is None
        assert comment.user_name is None

    def test_linear_user_from_api_missing_display_name(self):
        """Should fall back to name when displayName is missing."""
        data = {
            "id": "u1",
            "name": "Alice Smith",
            "email": "alice@example.com",
        }
        user = LinearUser.from_api(data)
        assert user.display_name == "Alice Smith"

    def test_linear_user_from_api_guest(self):
        """Should parse guest user correctly."""
        data = {
            "id": "u-guest",
            "name": "Guest User",
            "displayName": "Guest",
            "email": "guest@example.com",
            "active": True,
            "admin": False,
            "guest": True,
        }
        user = LinearUser.from_api(data)
        assert user.guest is True
        assert user.admin is False

    def test_linear_team_from_api_minimal(self):
        """Should parse team with minimal data."""
        data = {"id": "t1", "name": "Team", "key": "TM"}
        team = LinearTeam.from_api(data)
        assert team.id == "t1"
        assert team.description is None
        assert team.icon is None
        assert team.color is None
        assert team.private is False
        assert team.timezone is None

    def test_issue_state_from_api_defaults(self):
        """Should use defaults for missing state fields."""
        data = {"id": "s1", "name": "Custom", "type": "started"}
        state = IssueState.from_api(data)
        assert state.color == "#000000"
        assert state.position == 0
        assert state.description is None

    def test_label_from_api_no_parent(self):
        """Should handle label without parent."""
        data = {"id": "l1", "name": "feature", "color": "#00ff00"}
        label = Label.from_api(data)
        assert label.parent_id is None
        assert label.description is None

    def test_label_from_api_empty_parent(self):
        """Should handle label with empty parent dict."""
        data = {"id": "l1", "name": "feature", "color": "#00ff00", "parent": {}}
        label = Label.from_api(data)
        assert label.parent_id is None


# =============================================================================
# HTTP Client Lifecycle Tests
# =============================================================================


class TestHTTPClientLifecycle:
    """Test HTTP client creation and reuse."""

    @pytest.mark.asyncio
    async def test_get_client_creates_new(self, connector):
        """Should create a new client when none exists."""
        assert connector._client is None
        with patch(
            "aragora.connectors.enterprise.collaboration.linear.httpx.AsyncClient"
        ) as mock_cls:
            mock_instance = AsyncMock()
            mock_cls.return_value = mock_instance
            client = await connector._get_client()
            assert client is mock_instance
            mock_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_client_reuses_existing(self, connector):
        """Should reuse existing client on subsequent calls."""
        existing_client = AsyncMock()
        connector._client = existing_client
        client = await connector._get_client()
        assert client is existing_client

    @pytest.mark.asyncio
    async def test_close_client(self, connector):
        """Should close and clear the HTTP client."""
        mock_client = AsyncMock()
        connector._client = mock_client
        await connector.close()
        mock_client.aclose.assert_called_once()
        assert connector._client is None

    @pytest.mark.asyncio
    async def test_close_when_no_client(self, connector):
        """Should handle close when no client exists."""
        connector._client = None
        await connector.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_context_manager(self, connector):
        """Should support async context manager protocol."""
        mock_client = AsyncMock()
        connector._client = mock_client
        async with connector as c:
            assert c is connector
        mock_client.aclose.assert_called_once()
        assert connector._client is None


# =============================================================================
# GraphQL Query Construction Tests
# =============================================================================


class TestGraphQLQueryConstruction:
    """Test that GraphQL queries are properly constructed with variables."""

    @pytest.mark.asyncio
    async def test_graphql_sends_correct_payload(self, connector):
        """Should send query and variables in correct format."""
        mock_response = make_graphql_response({"test": "data"})
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await connector._graphql("query { test }", {"key": "value"})

            call_args = mock_client.post.call_args
            assert call_args[0][0] == "https://api.linear.app/graphql"
            payload = call_args[1]["json"]
            assert payload["query"] == "query { test }"
            assert payload["variables"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_graphql_default_empty_variables(self, connector):
        """Should send empty variables dict by default."""
        mock_response = make_graphql_response({"test": "data"})
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await connector._graphql("query { test }")

            call_args = mock_client.post.call_args
            payload = call_args[1]["json"]
            assert payload["variables"] == {}

    @pytest.mark.asyncio
    async def test_graphql_empty_errors_list(self, connector):
        """Should handle empty errors list in response."""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"errors": []}
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=response)
            mock_get_client.return_value = mock_client

            with pytest.raises(LinearError) as exc_info:
                await connector._graphql("query { test }")
            assert "Unknown error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_graphql_returns_empty_data_when_missing(self, connector):
        """Should return empty dict when data key is missing."""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {}
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=response)
            mock_get_client.return_value = mock_client

            result = await connector._graphql("query { test }")
            assert result == {}


# =============================================================================
# Issue Operations Extended Tests
# =============================================================================


class TestIssueOperationsExtended:
    """Extended tests for issue operations."""

    @pytest.mark.asyncio
    async def test_create_issue_with_all_optional_fields(self, connector):
        """Should include all optional fields in create issue mutation."""
        mock_data = {
            "issueCreate": {
                "success": True,
                "issue": {
                    "id": "new-1",
                    "identifier": "ENG-50",
                    "title": "Full Issue",
                    "description": "Full description",
                    "priority": 2,
                    "estimate": 8,
                    "dueDate": "2024-06-30",
                    "state": {"id": "s1", "name": "Backlog", "type": "backlog"},
                    "team": {"id": "t1", "key": "ENG"},
                    "assignee": {"id": "u1", "name": "Alice"},
                    "creator": {"id": "u2"},
                    "project": {"id": "p1"},
                    "cycle": {"id": "c1"},
                    "parent": {"id": "parent-1"},
                    "labels": {"nodes": [{"id": "l1"}, {"id": "l2"}]},
                },
            }
        }
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            due = datetime(2024, 6, 30, tzinfo=timezone.utc)
            issue = await connector.create_issue(
                team_id="t1",
                title="Full Issue",
                description="Full description",
                priority=IssuePriority.HIGH,
                state_id="s1",
                assignee_id="u1",
                project_id="p1",
                cycle_id="c1",
                parent_id="parent-1",
                label_ids=["l1", "l2"],
                estimate=8,
                due_date=due,
            )

            # Verify the mutation input
            call_args = mock_client.post.call_args
            payload = call_args[1]["json"]
            input_data = payload["variables"]["input"]
            assert input_data["teamId"] == "t1"
            assert input_data["title"] == "Full Issue"
            assert input_data["description"] == "Full description"
            assert input_data["priority"] == 2
            assert input_data["stateId"] == "s1"
            assert input_data["assigneeId"] == "u1"
            assert input_data["projectId"] == "p1"
            assert input_data["cycleId"] == "c1"
            assert input_data["parentId"] == "parent-1"
            assert input_data["labelIds"] == ["l1", "l2"]
            assert input_data["estimate"] == 8
            assert input_data["dueDate"] == "2024-06-30"

            assert issue.id == "new-1"
            assert issue.estimate == 8

    @pytest.mark.asyncio
    async def test_create_issue_minimal(self, connector):
        """Should create issue with only required fields."""
        mock_data = {
            "issueCreate": {
                "success": True,
                "issue": {
                    "id": "new-2",
                    "identifier": "ENG-51",
                    "title": "Minimal Issue",
                    "priority": 0,
                    "state": {"id": "s1", "name": "Backlog", "type": "backlog"},
                    "team": {"id": "t1", "key": "ENG"},
                    "assignee": None,
                    "creator": {"id": "u1"},
                    "project": None,
                    "cycle": None,
                    "parent": None,
                    "labels": {"nodes": []},
                },
            }
        }
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            issue = await connector.create_issue(
                team_id="t1",
                title="Minimal Issue",
            )

            call_args = mock_client.post.call_args
            payload = call_args[1]["json"]
            input_data = payload["variables"]["input"]
            assert input_data == {"teamId": "t1", "title": "Minimal Issue"}
            assert issue.id == "new-2"

    @pytest.mark.asyncio
    async def test_update_issue_with_all_fields(self, connector):
        """Should include all optional update fields."""
        mock_data = {
            "issueUpdate": {
                "success": True,
                "issue": {
                    "id": "i1",
                    "identifier": "ENG-1",
                    "title": "Updated",
                    "description": "New desc",
                    "priority": 4,
                    "estimate": 3,
                    "dueDate": "2024-12-31",
                    "state": {"id": "s2", "name": "In Progress", "type": "started"},
                    "team": {"id": "t1", "key": "ENG"},
                    "assignee": {"id": "u2", "name": "Bob"},
                    "creator": {"id": "u1"},
                    "project": {"id": "p1"},
                    "cycle": {"id": "c2"},
                    "parent": None,
                    "labels": {"nodes": [{"id": "l3"}]},
                },
            }
        }
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            due = datetime(2024, 12, 31, tzinfo=timezone.utc)
            issue = await connector.update_issue(
                issue_id="i1",
                title="Updated",
                description="New desc",
                priority=IssuePriority.LOW,
                state_id="s2",
                assignee_id="u2",
                project_id="p1",
                cycle_id="c2",
                label_ids=["l3"],
                estimate=3,
                due_date=due,
            )

            call_args = mock_client.post.call_args
            payload = call_args[1]["json"]
            assert payload["variables"]["id"] == "i1"
            input_data = payload["variables"]["input"]
            assert input_data["title"] == "Updated"
            assert input_data["description"] == "New desc"
            assert input_data["priority"] == 4
            assert input_data["stateId"] == "s2"
            assert input_data["assigneeId"] == "u2"
            assert input_data["projectId"] == "p1"
            assert input_data["cycleId"] == "c2"
            assert input_data["labelIds"] == ["l3"]
            assert input_data["estimate"] == 3
            assert input_data["dueDate"] == "2024-12-31"

            assert issue.title == "Updated"
            assert issue.priority == IssuePriority.LOW

    @pytest.mark.asyncio
    async def test_update_issue_no_changes(self, connector):
        """Should send empty input when no fields are specified."""
        mock_data = {
            "issueUpdate": {
                "success": True,
                "issue": {
                    "id": "i1",
                    "identifier": "ENG-1",
                    "title": "Unchanged",
                    "priority": 3,
                    "state": {"id": "s1", "name": "Todo", "type": "unstarted"},
                    "team": {"id": "t1", "key": "ENG"},
                    "assignee": None,
                    "creator": {"id": "u1"},
                    "project": None,
                    "cycle": None,
                    "parent": None,
                    "labels": {"nodes": []},
                },
            }
        }
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await connector.update_issue(issue_id="i1")

            call_args = mock_client.post.call_args
            payload = call_args[1]["json"]
            input_data = payload["variables"]["input"]
            assert input_data == {}

    @pytest.mark.asyncio
    async def test_archive_issue_success(self, connector):
        """Should archive an issue and return True on success."""
        mock_data = {"issueArchive": {"success": True}}
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await connector.archive_issue("i1")
            assert result is True

    @pytest.mark.asyncio
    async def test_archive_issue_failure(self, connector):
        """Should return False when archive fails."""
        mock_data = {"issueArchive": {"success": False}}
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await connector.archive_issue("i1")
            assert result is False

    @pytest.mark.asyncio
    async def test_archive_issue_empty_response(self, connector):
        """Should return False when response is empty."""
        mock_data = {}
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await connector.archive_issue("i1")
            assert result is False

    @pytest.mark.asyncio
    async def test_get_issues_no_pagination(self, connector):
        """Should return None cursor when hasNextPage is False."""
        mock_data = {
            "issues": {
                "nodes": [{"id": "i1", "identifier": "T-1", "title": "Issue", "priority": 0}],
                "pageInfo": {"endCursor": "some-cursor", "hasNextPage": False},
            }
        }
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            issues, cursor = await connector.get_issues()
            assert len(issues) == 1
            assert cursor is None

    @pytest.mark.asyncio
    async def test_get_issues_with_project_filter(self, connector):
        """Should filter issues by project_id."""
        mock_data = {"issues": {"nodes": [], "pageInfo": {"endCursor": None, "hasNextPage": False}}}
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await connector.get_issues(project_id="p1")

            call_args = mock_client.post.call_args
            query = call_args[1]["json"]["query"]
            assert "project" in query

    @pytest.mark.asyncio
    async def test_get_issues_with_assignee_filter(self, connector):
        """Should filter issues by assignee_id."""
        mock_data = {"issues": {"nodes": [], "pageInfo": {"endCursor": None, "hasNextPage": False}}}
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await connector.get_issues(assignee_id="u1")

            call_args = mock_client.post.call_args
            query = call_args[1]["json"]["query"]
            assert "assignee" in query

    @pytest.mark.asyncio
    async def test_get_issues_with_all_filters(self, connector):
        """Should combine multiple filters."""
        mock_data = {"issues": {"nodes": [], "pageInfo": {"endCursor": None, "hasNextPage": False}}}
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await connector.get_issues(
                team_id="t1",
                project_id="p1",
                assignee_id="u1",
                state_type=IssueStateType.STARTED,
            )

            call_args = mock_client.post.call_args
            query = call_args[1]["json"]["query"]
            assert "team" in query
            assert "project" in query
            assert "assignee" in query
            assert "state" in query

    @pytest.mark.asyncio
    async def test_get_issues_with_pagination_cursor(self, connector):
        """Should pass after cursor for pagination."""
        mock_data = {"issues": {"nodes": [], "pageInfo": {"endCursor": None, "hasNextPage": False}}}
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await connector.get_issues(first=10, after="cursor-abc")

            call_args = mock_client.post.call_args
            variables = call_args[1]["json"]["variables"]
            assert variables["first"] == 10
            assert variables["after"] == "cursor-abc"

    @pytest.mark.asyncio
    async def test_search_issues_empty_results(self, connector):
        """Should return empty list when no issues match."""
        mock_data = {"issueSearch": {"nodes": []}}
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            issues = await connector.search_issues("nonexistent query")
            assert issues == []

    @pytest.mark.asyncio
    async def test_search_issues_with_limit(self, connector):
        """Should pass first limit to search query."""
        mock_data = {"issueSearch": {"nodes": []}}
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await connector.search_issues("test", first=5)

            call_args = mock_client.post.call_args
            variables = call_args[1]["json"]["variables"]
            assert variables["first"] == 5
            assert variables["query"] == "test"


# =============================================================================
# Comment Extended Tests
# =============================================================================


class TestCommentOperationsExtended:
    """Extended comment operation tests."""

    @pytest.mark.asyncio
    async def test_add_comment(self, connector):
        """Should add a comment to an issue."""
        mock_data = {
            "commentCreate": {
                "success": True,
                "comment": {
                    "id": "com-new",
                    "body": "New comment body",
                    "user": {"id": "u1", "name": "Alice"},
                    "createdAt": "2024-01-15T14:00:00Z",
                    "updatedAt": "2024-01-15T14:00:00Z",
                },
            }
        }
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            comment = await connector.add_comment("issue-1", "New comment body")

            assert comment.id == "com-new"
            assert comment.body == "New comment body"
            assert comment.user_id == "u1"
            assert comment.user_name == "Alice"

            # Verify mutation input
            call_args = mock_client.post.call_args
            payload = call_args[1]["json"]
            input_data = payload["variables"]["input"]
            assert input_data["issueId"] == "issue-1"
            assert input_data["body"] == "New comment body"

    @pytest.mark.asyncio
    async def test_get_issue_comments_empty(self, connector):
        """Should return empty list when issue has no comments."""
        mock_data = {"issue": {"comments": {"nodes": []}}}
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            comments = await connector.get_issue_comments("issue-1")
            assert comments == []


# =============================================================================
# Project Extended Tests
# =============================================================================


class TestProjectOperationsExtended:
    """Extended project operation tests."""

    @pytest.mark.asyncio
    async def test_get_project_single(self, connector):
        """Should retrieve a single project by ID."""
        mock_data = {
            "project": {
                "id": "p1",
                "name": "Alpha Project",
                "description": "The alpha project",
                "icon": "star",
                "color": "#0000ff",
                "state": "started",
                "progress": 0.6,
                "targetDate": "2024-12-31",
                "startedAt": "2024-01-01T00:00:00Z",
                "completedAt": None,
                "createdAt": "2024-01-01T00:00:00Z",
                "updatedAt": "2024-06-15T00:00:00Z",
                "lead": {"id": "u1"},
                "teams": {"nodes": [{"id": "t1"}, {"id": "t2"}]},
            }
        }
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            project = await connector.get_project("p1")

            assert project.id == "p1"
            assert project.name == "Alpha Project"
            assert project.state == "started"
            assert project.progress == 0.6
            assert project.lead_id == "u1"
            assert project.team_ids == ["t1", "t2"]

    @pytest.mark.asyncio
    async def test_get_projects_with_team_filter(self, connector):
        """Should filter projects by team_id."""
        mock_data = {"projects": {"nodes": []}}
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await connector.get_projects(team_id="t1")

            call_args = mock_client.post.call_args
            query = call_args[1]["json"]["query"]
            assert "accessibleTeams" in query
            assert "t1" in query

    @pytest.mark.asyncio
    async def test_get_projects_without_team_filter(self, connector):
        """Should not include filter when team_id is None."""
        mock_data = {"projects": {"nodes": []}}
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await connector.get_projects()

            call_args = mock_client.post.call_args
            query = call_args[1]["json"]["query"]
            assert "accessibleTeams" not in query


# =============================================================================
# Cycle Extended Tests
# =============================================================================


class TestCycleOperationsExtended:
    """Extended cycle operation tests."""

    @pytest.mark.asyncio
    async def test_get_active_cycle_none(self, connector):
        """Should return None when no active cycle exists."""
        mock_data = {"team": {"activeCycle": None}}
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            cycle = await connector.get_active_cycle("t1")
            assert cycle is None

    @pytest.mark.asyncio
    async def test_get_cycles_empty(self, connector):
        """Should return empty list when team has no cycles."""
        mock_data = {"team": {"cycles": {"nodes": []}}}
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            cycles = await connector.get_cycles("t1")
            assert cycles == []

    @pytest.mark.asyncio
    async def test_get_cycles_with_custom_limit(self, connector):
        """Should pass custom first limit to cycles query."""
        mock_data = {"team": {"cycles": {"nodes": []}}}
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await connector.get_cycles("t1", first=5)

            call_args = mock_client.post.call_args
            variables = call_args[1]["json"]["variables"]
            assert variables["first"] == 5
            assert variables["teamId"] == "t1"


# =============================================================================
# Label Extended Tests
# =============================================================================


class TestLabelOperationsExtended:
    """Extended label operation tests."""

    @pytest.mark.asyncio
    async def test_get_labels_with_team_filter(self, connector):
        """Should filter labels by team_id."""
        mock_data = {"issueLabels": {"nodes": []}}
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await connector.get_labels(team_id="t1")

            call_args = mock_client.post.call_args
            query = call_args[1]["json"]["query"]
            assert "team" in query
            assert "t1" in query

    @pytest.mark.asyncio
    async def test_get_labels_without_team_filter(self, connector):
        """Should not include team filter when team_id is None."""
        mock_data = {"issueLabels": {"nodes": []}}
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await connector.get_labels()

            call_args = mock_client.post.call_args
            query = call_args[1]["json"]["query"]
            # Should not contain team filter
            assert "filter:" not in query or "team" not in query


# =============================================================================
# Enterprise Connector Interface Tests
# =============================================================================


class TestEnterpriseConnectorInterface:
    """Test EnterpriseConnector interface implementation."""

    @pytest.mark.asyncio
    async def test_connect_success(self, connector):
        """Should return True on successful connection."""
        mock_data = {
            "viewer": {
                "id": "u1",
                "name": "Test User",
                "displayName": "Test",
                "email": "test@example.com",
            }
        }
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await connector.connect()
            assert result is True

    @pytest.mark.asyncio
    async def test_connect_failure(self, connector):
        """Should return False when connection fails."""
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=LinearError("Connection failed"))
            mock_get_client.return_value = mock_client

            result = await connector.connect()
            assert result is False

    @pytest.mark.asyncio
    async def test_connect_os_error(self, connector):
        """Should return False on OSError."""
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_get_client.side_effect = OSError("Network unreachable")

            result = await connector.connect()
            assert result is False

    @pytest.mark.asyncio
    async def test_disconnect(self, connector):
        """Should close the client on disconnect."""
        mock_client = AsyncMock()
        connector._client = mock_client
        await connector.disconnect()
        mock_client.aclose.assert_called_once()
        assert connector._client is None

    @pytest.mark.asyncio
    async def test_search_evidence(self, connector):
        """Should search and return Evidence objects."""
        mock_data = {
            "issueSearch": {
                "nodes": [
                    {
                        "id": "i1",
                        "identifier": "ENG-1",
                        "title": "Bug Fix",
                        "description": "Fix the login bug",
                        "priority": 2,
                        "url": "https://linear.app/t/ENG-1",
                        "state": {"id": "s1", "name": "In Progress", "type": "started"},
                        "team": {"id": "t1", "key": "ENG"},
                        "assignee": None,
                        "creator": {"id": "u1"},
                        "project": None,
                        "cycle": None,
                        "labels": {"nodes": []},
                    },
                    {
                        "id": "i2",
                        "identifier": "ENG-2",
                        "title": "Feature Request",
                        "description": "Add dark mode",
                        "priority": 3,
                        "url": "https://linear.app/t/ENG-2",
                        "state": {"id": "s1", "name": "Backlog", "type": "backlog"},
                        "team": {"id": "t1", "key": "ENG"},
                        "assignee": None,
                        "creator": {"id": "u1"},
                        "project": None,
                        "cycle": None,
                        "labels": {"nodes": []},
                    },
                ]
            }
        }
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            results = await connector.search("login bug", limit=10)

            assert len(results) == 2
            assert isinstance(results[0], Evidence)
            assert results[0].id == "linear-issue-i1"
            assert results[0].source_id == "ENG-1"
            assert results[0].source_type == SourceType.EXTERNAL_API
            assert results[0].title == "[ENG-1] Bug Fix"
            assert results[0].url == "https://linear.app/t/ENG-1"
            assert results[0].metadata["type"] == "issue"
            assert results[0].metadata["priority"] == "HIGH"
            assert results[0].metadata["state"] == "In Progress"
            assert results[0].metadata["team"] == "ENG"

    @pytest.mark.asyncio
    async def test_search_evidence_with_limit(self, connector):
        """Should respect limit parameter."""
        nodes = [
            {
                "id": f"i{n}",
                "identifier": f"ENG-{n}",
                "title": f"Issue {n}",
                "description": f"Desc {n}",
                "priority": 0,
                "url": None,
                "state": {"id": "s1", "name": "Backlog", "type": "backlog"},
                "team": {"id": "t1", "key": "ENG"},
                "assignee": None,
                "creator": {"id": "u1"},
                "project": None,
                "cycle": None,
                "labels": {"nodes": []},
            }
            for n in range(5)
        ]
        mock_data = {"issueSearch": {"nodes": nodes}}
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            results = await connector.search("issues", limit=3)
            assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_evidence_error_returns_empty(self, connector):
        """Should return empty list on search error."""
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=LinearError("Search failed"))
            mock_get_client.return_value = mock_client

            # The search method catches LinearError and returns empty
            results = await connector.search("query")
            assert results == []

    @pytest.mark.asyncio
    async def test_fetch_issue_evidence(self, connector):
        """Should fetch evidence for an issue."""
        mock_data = {
            "issue": {
                "id": "abc-123",
                "identifier": "ENG-42",
                "title": "Important Bug",
                "description": "Detailed description",
                "priority": 1,
                "url": "https://linear.app/t/ENG-42",
                "state": {"id": "s1", "name": "In Progress", "type": "started"},
                "team": {"id": "t1", "key": "ENG"},
                "assignee": None,
                "creator": {"id": "u1"},
                "project": None,
                "cycle": None,
                "parent": None,
                "labels": {"nodes": []},
                "subscribers": {"nodes": []},
            }
        }
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await connector.fetch("linear-issue-abc-123")

            assert result is not None
            assert isinstance(result, Evidence)
            assert result.id == "linear-issue-abc-123"
            assert result.source_id == "ENG-42"
            assert result.title == "[ENG-42] Important Bug"
            assert result.metadata["type"] == "issue"
            assert result.metadata["priority"] == "URGENT"

    @pytest.mark.asyncio
    async def test_fetch_project_evidence(self, connector):
        """Should fetch evidence for a project."""
        mock_data = {
            "project": {
                "id": "proj-1",
                "name": "Cool Project",
                "description": "A cool project",
                "state": "started",
                "progress": 0.5,
                "lead": None,
                "teams": {"nodes": []},
            }
        }
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await connector.fetch("linear-project-proj-1")

            assert result is not None
            assert isinstance(result, Evidence)
            assert result.id == "linear-project-proj-1"
            assert result.source_id == "proj-1"
            assert result.title == "Cool Project"
            assert result.metadata["type"] == "project"
            assert result.metadata["state"] == "started"

    @pytest.mark.asyncio
    async def test_fetch_invalid_format(self, connector):
        """Should return None for invalid evidence ID format."""
        result = await connector.fetch("invalid-format")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_wrong_prefix(self, connector):
        """Should return None for non-linear evidence ID."""
        result = await connector.fetch("github-issue-123")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_unknown_entity_type(self, connector):
        """Should return None for unknown entity type."""
        result = await connector.fetch("linear-unknown-123")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_error_returns_none(self, connector):
        """Should return None on fetch error."""
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=LinearError("Not found"))
            mock_get_client.return_value = mock_client

            result = await connector.fetch("linear-issue-bad-id")
            assert result is None

    @pytest.mark.asyncio
    async def test_fetch_id_with_dashes(self, connector):
        """Should handle entity IDs that contain dashes."""
        mock_data = {
            "issue": {
                "id": "abc-def-ghi",
                "identifier": "ENG-1",
                "title": "Dashed ID Issue",
                "priority": 0,
                "state": {"id": "s1", "name": "Todo", "type": "unstarted"},
                "team": {"id": "t1", "key": "ENG"},
                "assignee": None,
                "creator": {"id": "u1"},
                "project": None,
                "cycle": None,
                "parent": None,
                "labels": {"nodes": []},
                "subscribers": {"nodes": []},
            }
        }
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await connector.fetch("linear-issue-abc-def-ghi")
            assert result is not None
            assert result.source_id == "ENG-1"


# =============================================================================
# Pagination Tests
# =============================================================================


class TestPagination:
    """Test cursor-based pagination."""

    @pytest.mark.asyncio
    async def test_paginate_issues_single_page(self, connector):
        """Should yield all issues from a single page."""
        mock_data = {
            "issues": {
                "nodes": [
                    {"id": "i1", "identifier": "T-1", "title": "Issue 1", "priority": 0},
                    {"id": "i2", "identifier": "T-2", "title": "Issue 2", "priority": 0},
                ],
                "pageInfo": {"endCursor": None, "hasNextPage": False},
            }
        }
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            issues = []
            async for issue in connector._paginate_issues(limit=10):
                issues.append(issue)

            assert len(issues) == 2

    @pytest.mark.asyncio
    async def test_paginate_issues_multiple_pages(self, connector):
        """Should paginate through multiple pages."""
        page1_data = {
            "issues": {
                "nodes": [{"id": "i1", "identifier": "T-1", "title": "Issue 1", "priority": 0}],
                "pageInfo": {"endCursor": "cursor-1", "hasNextPage": True},
            }
        }
        page2_data = {
            "issues": {
                "nodes": [{"id": "i2", "identifier": "T-2", "title": "Issue 2", "priority": 0}],
                "pageInfo": {"endCursor": None, "hasNextPage": False},
            }
        }
        page1_response = make_graphql_response(page1_data)
        page2_response = make_graphql_response(page2_data)

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[page1_response, page2_response])
            mock_get_client.return_value = mock_client

            issues = []
            async for issue in connector._paginate_issues(limit=1):
                issues.append(issue)

            assert len(issues) == 2
            assert issues[0].id == "i1"
            assert issues[1].id == "i2"
            # Verify two calls were made
            assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_paginate_projects_yields_all(self, connector):
        """Should yield all projects in a single batch."""
        mock_data = {
            "projects": {
                "nodes": [
                    {
                        "id": "p1",
                        "name": "Project 1",
                        "state": "started",
                        "progress": 0.5,
                        "lead": None,
                        "teams": {"nodes": []},
                    },
                    {
                        "id": "p2",
                        "name": "Project 2",
                        "state": "planned",
                        "progress": 0,
                        "lead": None,
                        "teams": {"nodes": []},
                    },
                ]
            }
        }
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            projects = []
            async for project in connector._paginate_projects(limit=50):
                projects.append(project)

            assert len(projects) == 2
            assert projects[0].name == "Project 1"
            assert projects[1].name == "Project 2"


# =============================================================================
# Sync Tests
# =============================================================================


class TestSyncOperations:
    """Test sync-related operations."""

    @pytest.mark.asyncio
    async def test_sync_items_yields_issues_and_projects(self, connector):
        """Should yield SyncItems for both issues and projects."""
        issue_data = {
            "issues": {
                "nodes": [
                    {
                        "id": "i1",
                        "identifier": "ENG-1",
                        "title": "Issue 1",
                        "description": "Desc 1",
                        "priority": 2,
                        "url": "https://linear.app/t/ENG-1",
                        "state": {"id": "s1", "name": "Todo", "type": "unstarted"},
                        "team": {"id": "t1", "key": "ENG"},
                        "assignee": None,
                        "creator": {"id": "u1"},
                        "project": None,
                        "cycle": None,
                        "parent": None,
                        "labels": {"nodes": []},
                        "subscribers": {"nodes": []},
                        "createdAt": "2024-01-01T00:00:00Z",
                        "updatedAt": "2024-01-15T00:00:00Z",
                    }
                ],
                "pageInfo": {"endCursor": None, "hasNextPage": False},
            }
        }
        project_data = {
            "projects": {
                "nodes": [
                    {
                        "id": "p1",
                        "name": "Project 1",
                        "description": "Proj desc",
                        "state": "started",
                        "progress": 0.5,
                        "lead": None,
                        "teams": {"nodes": []},
                        "createdAt": "2024-01-01T00:00:00Z",
                        "updatedAt": "2024-02-01T00:00:00Z",
                    }
                ]
            }
        }
        issue_response = make_graphql_response(issue_data)
        project_response = make_graphql_response(project_data)

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[issue_response, project_response])
            mock_get_client.return_value = mock_client

            state = SyncState(connector_id="Linear")
            items = []
            async for item in connector.sync_items(state, batch_size=100):
                items.append(item)

            assert len(items) == 2
            # First item is an issue
            assert items[0].id == "linear-issue-i1"
            assert items[0].source_type == "linear_issue"
            assert items[0].title == "Issue 1"
            assert items[0].metadata["type"] == "issue"
            assert items[0].metadata["identifier"] == "ENG-1"
            assert items[0].metadata["priority"] == 2
            # Second item is a project
            assert items[1].id == "linear-project-p1"
            assert items[1].source_type == "linear_project"
            assert items[1].title == "Project 1"
            assert items[1].metadata["type"] == "project"

    @pytest.mark.asyncio
    async def test_incremental_sync_yields_dicts(self, connector):
        """Should yield dictionary items from incremental sync."""
        issue_data = {
            "issues": {
                "nodes": [
                    {
                        "id": "i1",
                        "identifier": "ENG-1",
                        "title": "Issue 1",
                        "description": "Desc",
                        "priority": 0,
                        "state": {"id": "s1", "name": "Todo", "type": "unstarted"},
                        "team": {"id": "t1", "key": "ENG"},
                        "assignee": None,
                        "creator": {"id": "u1"},
                        "project": None,
                        "cycle": None,
                        "parent": None,
                        "labels": {"nodes": []},
                        "subscribers": {"nodes": []},
                    }
                ],
                "pageInfo": {"endCursor": None, "hasNextPage": False},
            }
        }
        project_data = {"projects": {"nodes": []}}
        issue_response = make_graphql_response(issue_data)
        project_response = make_graphql_response(project_data)

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[issue_response, project_response])
            mock_get_client.return_value = mock_client

            items = []
            async for item in connector.incremental_sync():
                items.append(item)

            assert len(items) == 1
            assert items[0]["type"] == "issue"
            assert items[0]["id"] == "linear-issue-i1"
            assert "content" in items[0]
            assert "data" in items[0]

    @pytest.mark.asyncio
    async def test_incremental_sync_with_state(self, connector):
        """Should pass provided state to sync."""
        issue_data = {
            "issues": {
                "nodes": [],
                "pageInfo": {"endCursor": None, "hasNextPage": False},
            }
        }
        project_data = {"projects": {"nodes": []}}
        issue_response = make_graphql_response(issue_data)
        project_response = make_graphql_response(project_data)

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[issue_response, project_response])
            mock_get_client.return_value = mock_client

            state = SyncState(
                connector_id="Linear",
                last_sync_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )
            items = []
            async for item in connector.incremental_sync(state=state):
                items.append(item)

            assert items == []

    @pytest.mark.asyncio
    async def test_full_sync_success(self, connector):
        """Should perform full sync and return SyncResult."""
        issue_data = {
            "issues": {
                "nodes": [
                    {
                        "id": "i1",
                        "identifier": "ENG-1",
                        "title": "Issue 1",
                        "priority": 0,
                        "state": {"id": "s1", "name": "Todo", "type": "unstarted"},
                        "team": {"id": "t1", "key": "ENG"},
                        "assignee": None,
                        "creator": {"id": "u1"},
                        "project": None,
                        "cycle": None,
                        "parent": None,
                        "labels": {"nodes": []},
                        "subscribers": {"nodes": []},
                    }
                ],
                "pageInfo": {"endCursor": None, "hasNextPage": False},
            }
        }
        project_data = {"projects": {"nodes": []}}
        issue_response = make_graphql_response(issue_data)
        project_response = make_graphql_response(project_data)

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[issue_response, project_response])
            mock_get_client.return_value = mock_client

            result = await connector.full_sync()

            assert isinstance(result, SyncResult)
            assert result.connector_id == "Linear"
            assert result.success is True
            assert result.items_synced == 1
            assert result.errors == []
            assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_full_sync_with_error(self, connector):
        """Should handle errors during full sync."""
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=LinearError("API error"))
            mock_get_client.return_value = mock_client

            result = await connector.full_sync()

            assert isinstance(result, SyncResult)
            assert result.success is False
            assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_full_sync_os_error(self, connector):
        """Should handle OSError during full sync."""
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=OSError("Connection refused"))
            mock_get_client.return_value = mock_client

            result = await connector.full_sync()

            assert result.success is False
            assert len(result.errors) > 0


# =============================================================================
# Error Handling Extended Tests
# =============================================================================


class TestErrorHandlingExtended:
    """Extended error handling tests."""

    @pytest.mark.asyncio
    async def test_http_500_error(self, connector):
        """Should handle server errors (500)."""
        mock_response = make_error_response(500, "Internal Server Error")
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(LinearError) as exc_info:
                await connector._graphql("query { test }")
            assert "500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_http_403_forbidden(self, connector):
        """Should handle forbidden errors (403)."""
        mock_response = make_error_response(403, "Forbidden")
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(LinearError) as exc_info:
                await connector._graphql("query { test }")
            assert "403" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_graphql_multiple_errors(self, connector):
        """Should use first error message when multiple errors exist."""
        errors = [
            {"message": "First error"},
            {"message": "Second error"},
        ]
        mock_response = make_graphql_error_response(errors=errors)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(LinearError) as exc_info:
                await connector._graphql("query { test }")
            assert "First error" in str(exc_info.value)
            assert len(exc_info.value.errors) == 2

    @pytest.mark.asyncio
    async def test_connection_error_during_request(self, connector):
        """Should propagate httpx connection errors."""
        import httpx

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_get_client.return_value = mock_client

            with pytest.raises(httpx.ConnectError):
                await connector._graphql("query { test }")

    @pytest.mark.asyncio
    async def test_search_evidence_handles_value_error(self, connector):
        """Should handle ValueError in search gracefully."""
        with patch.object(connector, "search_issues", side_effect=ValueError("Bad value")):
            results = await connector.search("query")
            assert results == []

    @pytest.mark.asyncio
    async def test_fetch_handles_key_error(self, connector):
        """Should handle KeyError in fetch gracefully."""
        with patch.object(connector, "get_issue", side_effect=KeyError("missing key")):
            result = await connector.fetch("linear-issue-123")
            assert result is None


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Test enum classes."""

    def test_issue_priority_int_values(self):
        """Should be integer enum with correct values."""
        assert int(IssuePriority.NO_PRIORITY) == 0
        assert int(IssuePriority.URGENT) == 1
        assert int(IssuePriority.HIGH) == 2
        assert int(IssuePriority.MEDIUM) == 3
        assert int(IssuePriority.LOW) == 4

    def test_issue_priority_from_int(self):
        """Should construct from integer value."""
        assert IssuePriority(0) == IssuePriority.NO_PRIORITY
        assert IssuePriority(1) == IssuePriority.URGENT
        assert IssuePriority(4) == IssuePriority.LOW

    def test_issue_state_type_values(self):
        """Should be string enum with correct values."""
        assert IssueStateType.BACKLOG.value == "backlog"
        assert IssueStateType.UNSTARTED.value == "unstarted"
        assert IssueStateType.STARTED.value == "started"
        assert IssueStateType.COMPLETED.value == "completed"
        assert IssueStateType.CANCELED.value == "canceled"

    def test_issue_state_type_from_string(self):
        """Should construct from string value."""
        assert IssueStateType("backlog") == IssueStateType.BACKLOG
        assert IssueStateType("completed") == IssueStateType.COMPLETED

    def test_issue_priority_name_attribute(self):
        """Should have correct name attribute."""
        assert IssuePriority.URGENT.name == "URGENT"
        assert IssuePriority.HIGH.name == "HIGH"
        assert IssuePriority.NO_PRIORITY.name == "NO_PRIORITY"


# =============================================================================
# Authentication Tests
# =============================================================================


class TestAuthentication:
    """Test authentication behavior."""

    @pytest.mark.asyncio
    async def test_api_key_in_headers(self, connector):
        """Should include API key in Authorization header."""
        with patch(
            "aragora.connectors.enterprise.collaboration.linear.httpx.AsyncClient"
        ) as mock_cls:
            mock_instance = AsyncMock()
            mock_cls.return_value = mock_instance
            await connector._get_client()

            call_args = mock_cls.call_args
            headers = call_args[1]["headers"]
            assert headers["Authorization"] == "lin_api_test_key_extended"
            assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_timeout_configuration(self, connector):
        """Should set 30s timeout on client."""
        with patch(
            "aragora.connectors.enterprise.collaboration.linear.httpx.AsyncClient"
        ) as mock_cls:
            mock_instance = AsyncMock()
            mock_cls.return_value = mock_instance
            await connector._get_client()

            call_args = mock_cls.call_args
            assert call_args[1]["timeout"] == 30.0


# =============================================================================
# Connector Properties Tests
# =============================================================================


class TestConnectorProperties:
    """Test connector property methods."""

    def test_name_property(self, connector):
        """Should return 'Linear' as name."""
        assert connector.name == "Linear"

    def test_source_type_property(self, connector):
        """Should return EXTERNAL_API source type."""
        assert connector.source_type == SourceType.EXTERNAL_API

    def test_connector_id(self, connector):
        """Should have 'linear' connector ID."""
        assert connector.connector_id == "linear"

    def test_tenant_id(self, connector):
        """Should store tenant ID."""
        assert connector.tenant_id == "test-tenant"

    def test_default_tenant_id(self, credentials):
        """Should default to 'default' tenant ID."""
        conn = LinearConnector(credentials=credentials)
        assert conn.tenant_id == "default"

    def test_credentials_stored(self, connector, credentials):
        """Should store the credentials."""
        assert connector._linear_credentials.api_key == "lin_api_test_key_extended"
        assert connector._linear_credentials.base_url == "https://api.linear.app/graphql"

    def test_client_initially_none(self, connector):
        """Should start with no HTTP client."""
        assert connector._client is None


# =============================================================================
# User Operations Extended Tests
# =============================================================================


class TestUserOperationsExtended:
    """Extended user operation tests."""

    @pytest.mark.asyncio
    async def test_get_users_empty(self, connector):
        """Should return empty list when no users exist."""
        mock_data = {"users": {"nodes": []}}
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            users = await connector.get_users()
            assert users == []

    @pytest.mark.asyncio
    async def test_get_current_user_full_data(self, connector):
        """Should parse all viewer fields correctly."""
        mock_data = {
            "viewer": {
                "id": "u-me",
                "name": "My Full Name",
                "displayName": "MyName",
                "email": "me@company.com",
                "active": True,
                "admin": True,
                "avatarUrl": "https://example.com/me.png",
                "guest": False,
                "createdAt": "2023-06-15T00:00:00Z",
            }
        }
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            user = await connector.get_current_user()

            assert user.id == "u-me"
            assert user.name == "My Full Name"
            assert user.display_name == "MyName"
            assert user.email == "me@company.com"
            assert user.active is True
            assert user.admin is True
            assert user.avatar_url == "https://example.com/me.png"
            assert user.guest is False
            assert user.created_at is not None


# =============================================================================
# Team Operations Extended Tests
# =============================================================================


class TestTeamOperationsExtended:
    """Extended team operation tests."""

    @pytest.mark.asyncio
    async def test_get_teams_empty(self, connector):
        """Should return empty list when no teams exist."""
        mock_data = {"teams": {"nodes": []}}
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            teams = await connector.get_teams()
            assert teams == []

    @pytest.mark.asyncio
    async def test_get_team_private(self, connector):
        """Should parse private team correctly."""
        mock_data = {
            "team": {
                "id": "t-priv",
                "name": "Secret Team",
                "key": "SEC",
                "private": True,
                "color": "#000000",
                "timezone": "UTC",
            }
        }
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            team = await connector.get_team("t-priv")

            assert team.id == "t-priv"
            assert team.private is True
            assert team.timezone == "UTC"

    @pytest.mark.asyncio
    async def test_get_team_states_empty(self, connector):
        """Should return empty list when team has no states."""
        mock_data = {"team": {"states": {"nodes": []}}}
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            states = await connector.get_team_states("t1")
            assert states == []

    @pytest.mark.asyncio
    async def test_get_team_states_all_types(self, connector):
        """Should parse all state types correctly."""
        mock_data = {
            "team": {
                "states": {
                    "nodes": [
                        {"id": "s1", "name": "Backlog", "type": "backlog", "color": "#aaa"},
                        {"id": "s2", "name": "Todo", "type": "unstarted", "color": "#bbb"},
                        {"id": "s3", "name": "In Progress", "type": "started", "color": "#ccc"},
                        {"id": "s4", "name": "Done", "type": "completed", "color": "#ddd"},
                        {"id": "s5", "name": "Canceled", "type": "canceled", "color": "#eee"},
                    ]
                }
            }
        }
        mock_response = make_graphql_response(mock_data)
        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            states = await connector.get_team_states("t1")

            assert len(states) == 5
            assert states[0].type == IssueStateType.BACKLOG
            assert states[1].type == IssueStateType.UNSTARTED
            assert states[2].type == IssueStateType.STARTED
            assert states[3].type == IssueStateType.COMPLETED
            assert states[4].type == IssueStateType.CANCELED


# =============================================================================
# Integration-style Tests (Multiple Operations)
# =============================================================================


class TestIntegrationScenarios:
    """Test realistic multi-step scenarios."""

    @pytest.mark.asyncio
    async def test_create_then_update_issue(self, connector):
        """Should create an issue then update it."""
        create_data = {
            "issueCreate": {
                "success": True,
                "issue": {
                    "id": "new-1",
                    "identifier": "ENG-100",
                    "title": "Original Title",
                    "priority": 3,
                    "state": {"id": "s1", "name": "Backlog", "type": "backlog"},
                    "team": {"id": "t1", "key": "ENG"},
                    "assignee": None,
                    "creator": {"id": "u1"},
                    "project": None,
                    "cycle": None,
                    "parent": None,
                    "labels": {"nodes": []},
                },
            }
        }
        update_data = {
            "issueUpdate": {
                "success": True,
                "issue": {
                    "id": "new-1",
                    "identifier": "ENG-100",
                    "title": "Updated Title",
                    "priority": 1,
                    "state": {"id": "s2", "name": "In Progress", "type": "started"},
                    "team": {"id": "t1", "key": "ENG"},
                    "assignee": {"id": "u2", "name": "Bob"},
                    "creator": {"id": "u1"},
                    "project": None,
                    "cycle": None,
                    "parent": None,
                    "labels": {"nodes": []},
                },
            }
        }

        create_response = make_graphql_response(create_data)
        update_response = make_graphql_response(update_data)

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[create_response, update_response])
            mock_get_client.return_value = mock_client

            created = await connector.create_issue(team_id="t1", title="Original Title")
            assert created.id == "new-1"
            assert created.title == "Original Title"

            updated = await connector.update_issue(
                issue_id="new-1",
                title="Updated Title",
                priority=IssuePriority.URGENT,
                assignee_id="u2",
            )
            assert updated.title == "Updated Title"
            assert updated.priority == IssuePriority.URGENT
            assert updated.assignee_id == "u2"

    @pytest.mark.asyncio
    async def test_connect_fetch_disconnect(self, connector):
        """Should connect, fetch data, and disconnect cleanly."""
        viewer_data = {
            "viewer": {
                "id": "u1",
                "name": "Test",
                "displayName": "Test",
                "email": "test@example.com",
            }
        }
        teams_data = {
            "teams": {
                "nodes": [
                    {"id": "t1", "name": "Engineering", "key": "ENG"},
                ]
            }
        }

        viewer_response = make_graphql_response(viewer_data)
        teams_response = make_graphql_response(teams_data)

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[viewer_response, teams_response])
            mock_get_client.return_value = mock_client

            # Connect
            connected = await connector.connect()
            assert connected is True

            # Fetch teams
            teams = await connector.get_teams()
            assert len(teams) == 1
            assert teams[0].name == "Engineering"

        # Disconnect
        connector._client = mock_client
        await connector.disconnect()
        mock_client.aclose.assert_called_once()
