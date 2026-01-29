"""
Tests for Linear Connector.

Tests the Linear GraphQL API integration including:
- Issue CRUD operations
- Project and cycle management
- Team and user queries
- Labels and workflow states
- Search functionality
- Error handling
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
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
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def credentials():
    """Create test credentials."""
    return LinearCredentials(
        api_key="lin_api_test_key_12345",
        base_url="https://api.linear.app/graphql",
    )


@pytest.fixture
def connector(credentials):
    """Create test connector."""
    return LinearConnector(credentials=credentials, tenant_id="test-tenant")


@pytest.fixture
def mock_client():
    """Create mock HTTP client."""
    client = AsyncMock()
    return client


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


def make_graphql_error_response(message: str = "GraphQL Error") -> MagicMock:
    """Create a mock GraphQL error response."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"errors": [{"message": message}]}
    return response


# =============================================================================
# Initialization Tests
# =============================================================================


class TestLinearConnectorInit:
    """Test LinearConnector initialization."""

    def test_default_configuration(self, credentials):
        """Should use provided credentials."""
        connector = LinearConnector(credentials=credentials)
        assert connector.credentials.api_key == "lin_api_test_key_12345"
        assert connector.credentials.base_url == "https://api.linear.app/graphql"

    def test_custom_tenant_id(self, credentials):
        """Should accept custom tenant ID."""
        connector = LinearConnector(credentials=credentials, tenant_id="custom-tenant")
        assert connector.tenant_id == "custom-tenant"

    def test_connector_properties(self, connector):
        """Should have correct connector properties."""
        assert connector.name == "Linear"
        assert connector.connector_id == "linear"

    def test_source_type(self, connector):
        """Should have correct source type."""
        from aragora.reasoning.provenance import SourceType

        assert connector.source_type == SourceType.EXTERNAL_API


# =============================================================================
# GraphQL Execution Tests
# =============================================================================


class TestGraphQLExecution:
    """Test GraphQL query execution."""

    @pytest.mark.asyncio
    async def test_successful_graphql_query(self, connector):
        """Should execute GraphQL query and return data."""
        mock_response = make_graphql_response({"viewer": {"id": "user-1", "name": "Test User"}})

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await connector._graphql("query { viewer { id name } }")

            assert result == {"viewer": {"id": "user-1", "name": "Test User"}}
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_graphql_with_variables(self, connector):
        """Should pass variables to GraphQL query."""
        mock_response = make_graphql_response({"issue": {"id": "issue-1"}})

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await connector._graphql(
                "query GetIssue($id: String!) { issue(id: $id) { id } }",
                variables={"id": "issue-1"},
            )

            call_args = mock_client.post.call_args
            assert call_args[1]["json"]["variables"] == {"id": "issue-1"}

    @pytest.mark.asyncio
    async def test_graphql_http_error(self, connector):
        """Should raise LinearError on HTTP error."""
        mock_response = make_error_response(401, "Unauthorized")

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(LinearError) as exc_info:
                await connector._graphql("query { viewer { id } }")

            assert "401" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_graphql_error_response(self, connector):
        """Should raise LinearError on GraphQL error."""
        mock_response = make_graphql_error_response("Invalid query")

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(LinearError) as exc_info:
                await connector._graphql("query { invalid }")

            assert "Invalid query" in str(exc_info.value)


# =============================================================================
# Issue Operations Tests
# =============================================================================


class TestIssueOperations:
    """Test issue CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_issues(self, connector):
        """Should retrieve list of issues."""
        mock_data = {
            "issues": {
                "nodes": [
                    {
                        "id": "issue-1",
                        "identifier": "LIN-1",
                        "title": "Test Issue",
                        "description": "Description",
                        "priority": 2,
                        "state": {"id": "state-1", "name": "In Progress", "type": "started"},
                        "team": {"id": "team-1", "key": "LIN"},
                        "assignee": None,
                        "creator": {"id": "user-1"},
                        "project": None,
                        "cycle": None,
                        "parent": None,
                        "labels": {"nodes": []},
                        "subscribers": {"nodes": []},
                        "createdAt": "2024-01-15T10:00:00Z",
                        "updatedAt": "2024-01-15T12:00:00Z",
                    }
                ],
                "pageInfo": {"endCursor": "cursor-1", "hasNextPage": True},
            }
        }
        mock_response = make_graphql_response(mock_data)

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            issues, cursor = await connector.get_issues(first=10)

            assert len(issues) == 1
            assert issues[0].id == "issue-1"
            assert issues[0].identifier == "LIN-1"
            assert issues[0].title == "Test Issue"
            assert issues[0].priority == IssuePriority.HIGH
            assert cursor == "cursor-1"

    @pytest.mark.asyncio
    async def test_get_issues_with_filters(self, connector):
        """Should filter issues by team and state."""
        mock_data = {"issues": {"nodes": [], "pageInfo": {"endCursor": None, "hasNextPage": False}}}
        mock_response = make_graphql_response(mock_data)

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await connector.get_issues(
                team_id="team-1", state_type=IssueStateType.STARTED, first=20
            )

            call_args = mock_client.post.call_args
            query = call_args[1]["json"]["query"]
            assert "team" in query
            assert "state" in query

    @pytest.mark.asyncio
    async def test_get_single_issue(self, connector):
        """Should retrieve single issue by ID."""
        mock_data = {
            "issue": {
                "id": "issue-1",
                "identifier": "LIN-1",
                "title": "Single Issue",
                "description": None,
                "priority": 3,
                "state": {"id": "state-1", "name": "Todo", "type": "unstarted"},
                "team": {"id": "team-1", "key": "LIN"},
                "assignee": {"id": "user-1", "name": "Test User"},
                "creator": {"id": "user-1"},
                "project": None,
                "cycle": None,
                "parent": None,
                "labels": {"nodes": []},
                "subscribers": {"nodes": []},
                "createdAt": "2024-01-15T10:00:00Z",
                "updatedAt": "2024-01-15T10:00:00Z",
            }
        }
        mock_response = make_graphql_response(mock_data)

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            issue = await connector.get_issue("issue-1")

            assert issue.id == "issue-1"
            assert issue.title == "Single Issue"
            assert issue.assignee_id == "user-1"
            assert issue.priority == IssuePriority.MEDIUM

    @pytest.mark.asyncio
    async def test_create_issue(self, connector):
        """Should create a new issue."""
        mock_data = {
            "issueCreate": {
                "success": True,
                "issue": {
                    "id": "new-issue-1",
                    "identifier": "LIN-10",
                    "title": "New Issue",
                    "description": "New description",
                    "priority": 2,
                    "state": {"id": "state-1", "name": "Backlog", "type": "backlog"},
                    "team": {"id": "team-1", "key": "LIN"},
                    "assignee": None,
                    "creator": {"id": "user-1"},
                    "project": None,
                    "cycle": None,
                    "parent": None,
                    "labels": {"nodes": []},
                    "subscribers": {"nodes": []},
                    "createdAt": "2024-01-15T10:00:00Z",
                    "updatedAt": "2024-01-15T10:00:00Z",
                },
            }
        }
        mock_response = make_graphql_response(mock_data)

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            issue = await connector.create_issue(
                title="New Issue",
                team_id="team-1",
                description="New description",
                priority=IssuePriority.HIGH,
            )

            assert issue.id == "new-issue-1"
            assert issue.identifier == "LIN-10"
            assert issue.title == "New Issue"

    @pytest.mark.asyncio
    async def test_update_issue(self, connector):
        """Should update an existing issue."""
        mock_data = {
            "issueUpdate": {
                "success": True,
                "issue": {
                    "id": "issue-1",
                    "identifier": "LIN-1",
                    "title": "Updated Title",
                    "description": "Updated description",
                    "priority": 1,
                    "state": {"id": "state-2", "name": "Done", "type": "completed"},
                    "team": {"id": "team-1", "key": "LIN"},
                    "assignee": {"id": "user-2", "name": "New Assignee"},
                    "creator": {"id": "user-1"},
                    "project": None,
                    "cycle": None,
                    "parent": None,
                    "labels": {"nodes": []},
                    "subscribers": {"nodes": []},
                    "createdAt": "2024-01-15T10:00:00Z",
                    "updatedAt": "2024-01-15T14:00:00Z",
                },
            }
        }
        mock_response = make_graphql_response(mock_data)

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            issue = await connector.update_issue(
                issue_id="issue-1",
                title="Updated Title",
                description="Updated description",
                priority=IssuePriority.URGENT,
                assignee_id="user-2",
            )

            assert issue.title == "Updated Title"
            assert issue.assignee_id == "user-2"

    @pytest.mark.asyncio
    async def test_search_issues(self, connector):
        """Should search issues by query."""
        mock_data = {
            "issueSearch": {
                "nodes": [
                    {
                        "id": "issue-1",
                        "identifier": "LIN-1",
                        "title": "Bug: Login fails",
                        "description": "Users cannot login",
                        "priority": 1,
                        "state": {"id": "state-1", "name": "In Progress", "type": "started"},
                        "team": {"id": "team-1", "key": "LIN"},
                        "assignee": None,
                        "creator": {"id": "user-1"},
                        "project": None,
                        "cycle": None,
                        "parent": None,
                        "labels": {"nodes": []},
                        "subscribers": {"nodes": []},
                        "createdAt": "2024-01-15T10:00:00Z",
                        "updatedAt": "2024-01-15T10:00:00Z",
                    }
                ]
            }
        }
        mock_response = make_graphql_response(mock_data)

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            issues = await connector.search_issues("login bug")

            assert len(issues) == 1
            assert "login" in issues[0].title.lower() or "bug" in issues[0].title.lower()


# =============================================================================
# Team Operations Tests
# =============================================================================


class TestTeamOperations:
    """Test team-related operations."""

    @pytest.mark.asyncio
    async def test_get_teams(self, connector):
        """Should retrieve list of teams."""
        mock_data = {
            "teams": {
                "nodes": [
                    {
                        "id": "team-1",
                        "name": "Engineering",
                        "key": "ENG",
                        "description": "Engineering team",
                        "private": False,
                    },
                    {
                        "id": "team-2",
                        "name": "Product",
                        "key": "PROD",
                        "description": "Product team",
                        "private": False,
                    },
                ]
            }
        }
        mock_response = make_graphql_response(mock_data)

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            teams = await connector.get_teams()

            assert len(teams) == 2
            assert teams[0].name == "Engineering"
            assert teams[1].key == "PROD"

    @pytest.mark.asyncio
    async def test_get_single_team(self, connector):
        """Should retrieve single team by ID."""
        mock_data = {
            "team": {
                "id": "team-1",
                "name": "Engineering",
                "key": "ENG",
                "description": "The engineering team",
                "private": False,
                "color": "#4299e1",
            }
        }
        mock_response = make_graphql_response(mock_data)

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            team = await connector.get_team("team-1")

            assert team.id == "team-1"
            assert team.name == "Engineering"
            assert team.color == "#4299e1"

    @pytest.mark.asyncio
    async def test_get_team_states(self, connector):
        """Should retrieve workflow states for a team."""
        mock_data = {
            "team": {
                "states": {
                    "nodes": [
                        {
                            "id": "state-1",
                            "name": "Backlog",
                            "type": "backlog",
                            "color": "#bfbfbf",
                            "position": 0,
                        },
                        {
                            "id": "state-2",
                            "name": "Todo",
                            "type": "unstarted",
                            "color": "#e2e2e2",
                            "position": 1,
                        },
                        {
                            "id": "state-3",
                            "name": "In Progress",
                            "type": "started",
                            "color": "#f2c94c",
                            "position": 2,
                        },
                        {
                            "id": "state-4",
                            "name": "Done",
                            "type": "completed",
                            "color": "#5e6ad2",
                            "position": 3,
                        },
                    ]
                }
            }
        }
        mock_response = make_graphql_response(mock_data)

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            states = await connector.get_team_states("team-1")

            assert len(states) == 4
            assert states[0].type == IssueStateType.BACKLOG
            assert states[2].type == IssueStateType.STARTED
            assert states[3].name == "Done"


# =============================================================================
# Project and Cycle Tests
# =============================================================================


class TestProjectCycleOperations:
    """Test project and cycle operations."""

    @pytest.mark.asyncio
    async def test_get_projects(self, connector):
        """Should retrieve list of projects."""
        mock_data = {
            "projects": {
                "nodes": [
                    {
                        "id": "project-1",
                        "name": "Q1 Roadmap",
                        "description": "Q1 2024 features",
                        "state": "started",
                        "progress": 0.45,
                        "targetDate": "2024-03-31",
                        "lead": None,
                        "teams": {"nodes": []},
                    },
                    {
                        "id": "project-2",
                        "name": "Tech Debt",
                        "description": "Technical debt reduction",
                        "state": "planned",
                        "progress": 0.0,
                        "targetDate": None,
                        "lead": None,
                        "teams": {"nodes": []},
                    },
                ],
            }
        }
        mock_response = make_graphql_response(mock_data)

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            projects = await connector.get_projects(first=10)

            assert len(projects) == 2
            assert projects[0].name == "Q1 Roadmap"
            assert projects[0].progress == 0.45

    @pytest.mark.asyncio
    async def test_get_cycles(self, connector):
        """Should retrieve cycles for a team."""
        mock_data = {
            "team": {
                "cycles": {
                    "nodes": [
                        {
                            "id": "cycle-1",
                            "name": "Sprint 1",
                            "number": 1,
                            "startsAt": "2024-01-01T00:00:00Z",
                            "endsAt": "2024-01-14T23:59:59Z",
                            "progress": 1.0,
                        },
                        {
                            "id": "cycle-2",
                            "name": "Sprint 2",
                            "number": 2,
                            "startsAt": "2024-01-15T00:00:00Z",
                            "endsAt": "2024-01-28T23:59:59Z",
                            "progress": 0.3,
                        },
                    ]
                }
            }
        }
        mock_response = make_graphql_response(mock_data)

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            cycles = await connector.get_cycles("team-1")

            assert len(cycles) == 2
            assert cycles[0].name == "Sprint 1"
            assert cycles[1].progress == 0.3

    @pytest.mark.asyncio
    async def test_get_active_cycle(self, connector):
        """Should retrieve active cycle for a team."""
        mock_data = {
            "team": {
                "activeCycle": {
                    "id": "cycle-2",
                    "name": "Sprint 2",
                    "number": 2,
                    "startsAt": "2024-01-15T00:00:00Z",
                    "endsAt": "2024-01-28T23:59:59Z",
                    "progress": 0.3,
                }
            }
        }
        mock_response = make_graphql_response(mock_data)

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            cycle = await connector.get_active_cycle("team-1")

            assert cycle is not None
            assert cycle.name == "Sprint 2"
            assert cycle.number == 2


# =============================================================================
# User and Label Tests
# =============================================================================


class TestUserLabelOperations:
    """Test user and label operations."""

    @pytest.mark.asyncio
    async def test_get_users(self, connector):
        """Should retrieve list of users."""
        mock_data = {
            "users": {
                "nodes": [
                    {
                        "id": "user-1",
                        "name": "Alice Smith",
                        "displayName": "Alice",
                        "email": "alice@example.com",
                        "active": True,
                        "admin": False,
                    },
                    {
                        "id": "user-2",
                        "name": "Bob Jones",
                        "displayName": "Bob",
                        "email": "bob@example.com",
                        "active": True,
                        "admin": True,
                    },
                ]
            }
        }
        mock_response = make_graphql_response(mock_data)

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            users = await connector.get_users()

            assert len(users) == 2
            assert users[0].email == "alice@example.com"
            assert users[1].admin is True

    @pytest.mark.asyncio
    async def test_get_current_user(self, connector):
        """Should retrieve current authenticated user."""
        mock_data = {
            "viewer": {
                "id": "user-1",
                "name": "Current User",
                "displayName": "Me",
                "email": "me@example.com",
                "active": True,
                "admin": False,
            }
        }
        mock_response = make_graphql_response(mock_data)

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            user = await connector.get_current_user()

            assert user.id == "user-1"
            assert user.email == "me@example.com"

    @pytest.mark.asyncio
    async def test_get_labels(self, connector):
        """Should retrieve labels for organization or team."""
        mock_data = {
            "issueLabels": {
                "nodes": [
                    {
                        "id": "label-1",
                        "name": "bug",
                        "color": "#eb5757",
                        "description": "Bug reports",
                    },
                    {
                        "id": "label-2",
                        "name": "feature",
                        "color": "#6fcf97",
                        "description": "New features",
                    },
                    {
                        "id": "label-3",
                        "name": "urgent",
                        "color": "#f2994a",
                        "description": "High priority",
                    },
                ]
            }
        }
        mock_response = make_graphql_response(mock_data)

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            labels = await connector.get_labels()

            assert len(labels) == 3
            assert labels[0].name == "bug"
            assert labels[1].color == "#6fcf97"


# =============================================================================
# Comments Tests
# =============================================================================


class TestCommentOperations:
    """Test comment operations."""

    @pytest.mark.asyncio
    async def test_get_issue_comments(self, connector):
        """Should retrieve comments for an issue."""
        mock_data = {
            "issue": {
                "comments": {
                    "nodes": [
                        {
                            "id": "comment-1",
                            "body": "This is a comment",
                            "user": {"id": "user-1", "name": "Alice"},
                            "createdAt": "2024-01-15T10:00:00Z",
                        },
                        {
                            "id": "comment-2",
                            "body": "Another comment",
                            "user": {"id": "user-2", "name": "Bob"},
                            "createdAt": "2024-01-15T11:00:00Z",
                        },
                    ]
                }
            }
        }
        mock_response = make_graphql_response(mock_data)

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            comments = await connector.get_issue_comments("issue-1")

            assert len(comments) == 2
            assert comments[0].body == "This is a comment"
            assert comments[1].user_name == "Bob"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_network_timeout(self, connector):
        """Should handle network timeout gracefully."""
        import httpx

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Connection timeout"))
            mock_get_client.return_value = mock_client

            with pytest.raises(httpx.TimeoutException):
                await connector.get_teams()

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, connector):
        """Should handle rate limit errors."""
        mock_response = make_error_response(429, "Rate limit exceeded")

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(LinearError) as exc_info:
                await connector.get_issues()

            assert "429" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_authentication_error(self, connector):
        """Should handle authentication errors."""
        mock_response = make_error_response(401, "Invalid API key")

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(LinearError) as exc_info:
                await connector.get_current_user()

            assert "401" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_issue_id(self, connector):
        """Should handle invalid issue ID gracefully."""
        mock_response = make_graphql_error_response("Entity not found: Issue")

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(LinearError) as exc_info:
                await connector.get_issue("invalid-id")

            assert "Entity not found" in str(exc_info.value)


# =============================================================================
# Data Model Tests
# =============================================================================


class TestDataModels:
    """Test data model parsing."""

    def test_linear_team_from_api(self):
        """Should parse team from API response."""
        data = {
            "id": "team-1",
            "name": "Engineering",
            "key": "ENG",
            "description": "Engineering team",
            "icon": "🔧",
            "color": "#4299e1",
            "private": False,
            "timezone": "America/New_York",
        }
        team = LinearTeam.from_api(data)
        assert team.id == "team-1"
        assert team.name == "Engineering"
        assert team.timezone == "America/New_York"

    def test_linear_user_from_api(self):
        """Should parse user from API response."""
        data = {
            "id": "user-1",
            "name": "Alice Smith",
            "displayName": "Alice",
            "email": "alice@example.com",
            "active": True,
            "admin": False,
            "avatarUrl": "https://example.com/avatar.png",
            "guest": False,
            "createdAt": "2024-01-01T00:00:00Z",
        }
        user = LinearUser.from_api(data)
        assert user.id == "user-1"
        assert user.display_name == "Alice"
        assert user.avatar_url == "https://example.com/avatar.png"

    def test_issue_state_from_api(self):
        """Should parse issue state from API response."""
        data = {
            "id": "state-1",
            "name": "In Progress",
            "type": "started",
            "color": "#f2c94c",
            "position": 2,
            "description": "Work in progress",
        }
        state = IssueState.from_api(data)
        assert state.id == "state-1"
        assert state.type == IssueStateType.STARTED
        assert state.position == 2

    def test_label_from_api(self):
        """Should parse label from API response."""
        data = {
            "id": "label-1",
            "name": "bug",
            "color": "#eb5757",
            "description": "Bug reports",
            "parent": {"id": "parent-label-1"},
        }
        label = Label.from_api(data)
        assert label.id == "label-1"
        assert label.name == "bug"
        assert label.parent_id == "parent-label-1"

    def test_issue_priority_enum(self):
        """Should correctly map priority values."""
        assert IssuePriority.NO_PRIORITY == 0
        assert IssuePriority.URGENT == 1
        assert IssuePriority.HIGH == 2
        assert IssuePriority.MEDIUM == 3
        assert IssuePriority.LOW == 4

    def test_issue_state_type_enum(self):
        """Should correctly map state type values."""
        assert IssueStateType.BACKLOG.value == "backlog"
        assert IssueStateType.UNSTARTED.value == "unstarted"
        assert IssueStateType.STARTED.value == "started"
        assert IssueStateType.COMPLETED.value == "completed"
        assert IssueStateType.CANCELED.value == "canceled"
