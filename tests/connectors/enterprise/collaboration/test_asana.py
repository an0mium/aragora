"""
Tests for Asana Enterprise Connector.

Tests the Asana API integration including:
- Workspace and project operations
- Task CRUD operations
- Subtasks and comments
- Custom fields
- Section management
- Error handling
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List, Optional

from aragora.connectors.enterprise.collaboration.asana import (
    AsanaConnector,
    AsanaCredentials,
    AsanaWorkspace,
    AsanaProject,
    AsanaSection,
    AsanaTask,
    AsanaComment,
    AsanaError,
    TaskCreateRequest,
    TaskStatus,
    ResourceType,
)
from aragora.connectors.base import Evidence
from aragora.reasoning.provenance import SourceType


# =============================================================================
# Concrete Test Subclass
# =============================================================================


class ConcreteAsanaConnector(AsanaConnector):
    """Concrete implementation of AsanaConnector for testing.

    Implements the abstract methods from BaseConnector that are required
    for instantiation.
    """

    @property
    def source_type(self) -> SourceType:
        return SourceType.DOCUMENT

    @property
    def name(self) -> str:
        return "Asana (Test)"

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs,
    ) -> List[Evidence]:
        """Implement abstract search method."""
        # For testing - returns empty list
        return []

    async def fetch(self, evidence_id: str) -> Optional[Evidence]:
        """Implement abstract fetch method."""
        return None


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def credentials():
    """Create test credentials."""
    return AsanaCredentials(
        access_token="test_token_12345",
        refresh_token="test_refresh_token",
    )


@pytest.fixture
def connector(credentials):
    """Create test connector."""
    return ConcreteAsanaConnector(
        credentials=credentials,
        workspace_gid="12345",
    )


@pytest.fixture
def mock_client():
    """Create mock HTTP client."""
    client = AsyncMock()
    return client


def make_api_response(data: Any) -> Dict[str, Any]:
    """Create a mock API response."""
    return {"data": data}


def make_workspace_data(
    gid: str = "12345",
    name: str = "Test Workspace",
) -> Dict[str, Any]:
    """Create mock workspace data."""
    return {
        "gid": gid,
        "name": name,
        "is_organization": False,
    }


def make_project_data(
    gid: str = "proj-123",
    name: str = "Test Project",
    workspace_gid: str = "12345",
) -> Dict[str, Any]:
    """Create mock project data."""
    return {
        "gid": gid,
        "name": name,
        "workspace": {"gid": workspace_gid, "name": "Test Workspace"},
        "owner": {"name": "John Owner"},
        "notes": "Project notes",
        "color": "blue",
        "archived": False,
        "public": True,
        "created_at": "2024-01-15T10:00:00.000Z",
        "modified_at": "2024-01-15T12:00:00.000Z",
        "permalink_url": f"https://app.asana.com/0/{gid}",
    }


def make_task_data(
    gid: str = "task-123",
    name: str = "Test Task",
) -> Dict[str, Any]:
    """Create mock task data."""
    return {
        "gid": gid,
        "name": name,
        "notes": "Task notes",
        "completed": False,
        "due_on": "2024-01-20",
        "start_on": "2024-01-15",
        "created_at": "2024-01-15T10:00:00.000Z",
        "modified_at": "2024-01-15T12:00:00.000Z",
        "assignee": {"gid": "user-1", "name": "Alice Dev"},
        "projects": [{"gid": "proj-1", "name": "Project A"}],
        "memberships": [{"section": {"gid": "section-1"}}],
        "tags": [{"name": "urgent"}],
        "followers": [{"name": "Bob PM"}],
        "num_subtasks": 2,
        "permalink_url": f"https://app.asana.com/0/task/{gid}",
        "custom_fields": [],
    }


def make_section_data(
    gid: str = "section-123",
    name: str = "To Do",
) -> Dict[str, Any]:
    """Create mock section data."""
    return {
        "gid": gid,
        "name": name,
        "project": {"gid": "proj-123"},
    }


def make_comment_data(
    gid: str = "comment-123",
    text: str = "Test comment",
) -> Dict[str, Any]:
    """Create mock comment data."""
    return {
        "gid": gid,
        "text": text,
        "created_by": {"gid": "user-1", "name": "Commenter"},
        "created_at": "2024-01-15T14:00:00.000Z",
        "resource_subtype": "comment_added",  # Asana uses "comment_added" for actual comments
    }


# =============================================================================
# Initialization Tests
# =============================================================================


class TestAsanaConnectorInit:
    """Test AsanaConnector initialization."""

    def test_basic_initialization(self, credentials):
        """Should initialize with credentials."""
        connector = ConcreteAsanaConnector(credentials=credentials)
        assert connector.credentials == credentials
        assert connector.default_workspace_gid is None

    def test_with_workspace_gid(self, credentials):
        """Should accept workspace GID."""
        connector = ConcreteAsanaConnector(
            credentials=credentials,
            workspace_gid="12345",
        )
        assert connector.default_workspace_gid == "12345"

    def test_base_url(self, connector):
        """Should have correct base URL."""
        assert connector.BASE_URL == "https://app.asana.com/api/1.0"


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestContextManager:
    """Test async context manager functionality."""

    @pytest.mark.asyncio
    async def test_enter_context(self, connector):
        """Should initialize client on enter."""
        async with connector as conn:
            assert conn._client is not None
            assert conn == connector

    @pytest.mark.asyncio
    async def test_exit_context(self, connector):
        """Should close client on exit."""
        async with connector:
            pass
        assert connector._client is None

    def test_client_without_context(self, connector):
        """Should raise error when accessing client without context."""
        with pytest.raises(AsanaError, match="not initialized"):
            _ = connector.client


# =============================================================================
# Workspace Tests
# =============================================================================


class TestWorkspaceOperations:
    """Test workspace operations."""

    @pytest.mark.asyncio
    async def test_list_workspaces(self, connector):
        """Should list workspaces."""
        mock_response = {
            "data": [
                make_workspace_data("ws-1", "Workspace A"),
                make_workspace_data("ws-2", "Workspace B"),
            ]
        }

        async with connector:
            with patch.object(connector, "_request", return_value=mock_response):
                workspaces = await connector.list_workspaces()

                assert len(workspaces) == 2
                assert workspaces[0].gid == "ws-1"
                assert workspaces[0].name == "Workspace A"


# =============================================================================
# Project Tests
# =============================================================================


class TestProjectOperations:
    """Test project operations."""

    @pytest.mark.asyncio
    async def test_list_projects(self, connector):
        """Should list projects in workspace."""
        mock_response = make_api_response(
            [
                make_project_data("proj-1", "Project Alpha"),
                make_project_data("proj-2", "Project Beta"),
            ]
        )

        async with connector:
            with patch.object(connector, "_paginate") as mock_paginate:
                # Make _paginate return an async iterator
                async def mock_gen():
                    for item in mock_response["data"]:
                        yield item

                mock_paginate.return_value = mock_gen()

                projects = await connector.list_projects("12345")

                assert len(projects) == 2
                assert projects[0].name == "Project Alpha"

    @pytest.mark.asyncio
    async def test_get_project(self, connector):
        """Should get a project by GID."""
        mock_response = make_api_response(make_project_data("proj-1", "My Project"))

        async with connector:
            with patch.object(connector, "_request", return_value=mock_response):
                project = await connector.get_project("proj-1")

                assert project.gid == "proj-1"
                assert project.name == "My Project"


# =============================================================================
# Section Tests
# =============================================================================


class TestSectionOperations:
    """Test section operations."""

    @pytest.mark.asyncio
    async def test_list_sections(self, connector):
        """Should list sections in project."""
        mock_response = {
            "data": [
                make_section_data("sec-1", "To Do"),
                make_section_data("sec-2", "In Progress"),
                make_section_data("sec-3", "Done"),
            ]
        }

        async with connector:
            with patch.object(connector, "_request", return_value=mock_response):
                sections = await connector.list_sections("proj-1")

                assert len(sections) == 3
                assert sections[0].name == "To Do"

    @pytest.mark.asyncio
    async def test_create_section(self, connector):
        """Should create a section."""
        mock_response = make_api_response(make_section_data("sec-new", "New Section"))

        async with connector:
            with patch.object(connector, "_request", return_value=mock_response):
                section = await connector.create_section("proj-1", "New Section")

                assert section.gid == "sec-new"
                assert section.name == "New Section"


# =============================================================================
# Task Tests
# =============================================================================


class TestTaskOperations:
    """Test task CRUD operations."""

    @pytest.mark.asyncio
    async def test_list_tasks(self, connector):
        """Should list tasks in project."""
        async with connector:
            with patch.object(connector, "_paginate") as mock_paginate:

                async def mock_gen():
                    yield make_task_data("task-1", "Task 1")
                    yield make_task_data("task-2", "Task 2")

                mock_paginate.return_value = mock_gen()

                tasks = await connector.list_tasks(project_gid="proj-1")

                assert len(tasks) == 2
                assert tasks[0].name == "Task 1"

    @pytest.mark.asyncio
    async def test_get_task(self, connector):
        """Should get a task by GID."""
        mock_response = make_api_response(make_task_data("task-1", "My Task"))

        async with connector:
            with patch.object(connector, "_request", return_value=mock_response):
                task = await connector.get_task("task-1")

                assert task.gid == "task-1"
                assert task.name == "My Task"
                assert task.assignee == "Alice Dev"

    @pytest.mark.asyncio
    async def test_create_task(self, connector):
        """Should create a task."""
        mock_response = make_api_response(make_task_data("task-new", "New Task"))

        async with connector:
            with patch.object(connector, "_request", return_value=mock_response):
                request = TaskCreateRequest(
                    name="New Task",
                    project_gid="proj-1",
                    notes="Task description",
                    due_on="2024-01-20",
                )
                task = await connector.create_task(request)

                assert task.gid == "task-new"
                assert task.name == "New Task"

    @pytest.mark.asyncio
    async def test_update_task(self, connector):
        """Should update a task."""
        mock_response = make_api_response(
            {
                **make_task_data("task-1", "Updated Task"),
                "notes": "Updated notes",
            }
        )

        async with connector:
            with patch.object(connector, "_request", return_value=mock_response):
                task = await connector.update_task(
                    "task-1",
                    name="Updated Task",
                    notes="Updated notes",
                )

                assert task.name == "Updated Task"

    @pytest.mark.asyncio
    async def test_complete_task(self, connector):
        """Should mark task as complete."""
        mock_response = make_api_response(
            {
                **make_task_data("task-1", "Completed Task"),
                "completed": True,
                "completed_at": "2024-01-16T10:00:00.000Z",
            }
        )

        async with connector:
            with patch.object(connector, "_request", return_value=mock_response):
                task = await connector.complete_task("task-1")

                assert task.completed is True


# =============================================================================
# Subtask Tests
# =============================================================================


class TestSubtaskOperations:
    """Test subtask operations."""

    @pytest.mark.asyncio
    async def test_list_subtasks(self, connector):
        """Should list subtasks."""
        mock_response = {
            "data": [
                make_task_data("sub-1", "Subtask 1"),
                make_task_data("sub-2", "Subtask 2"),
            ]
        }

        async with connector:
            with patch.object(connector, "_request", return_value=mock_response):
                subtasks = await connector.list_subtasks("task-1")

                assert len(subtasks) == 2
                assert subtasks[0].name == "Subtask 1"

    @pytest.mark.asyncio
    async def test_create_subtask(self, connector):
        """Should create a subtask."""
        mock_response = make_api_response(
            {
                **make_task_data("sub-new", "New Subtask"),
                "parent": {"gid": "task-1"},
            }
        )

        async with connector:
            with patch.object(connector, "_request", return_value=mock_response):
                subtask = await connector.create_subtask(
                    "task-1",
                    "New Subtask",
                    notes="Subtask notes",
                )

                assert subtask.name == "New Subtask"


# =============================================================================
# Comment Tests
# =============================================================================


class TestCommentOperations:
    """Test comment operations."""

    @pytest.mark.asyncio
    async def test_list_comments(self, connector):
        """Should list comments on task."""
        mock_response = {
            "data": [
                make_comment_data("com-1", "First comment"),
                make_comment_data("com-2", "Second comment"),
            ]
        }

        async with connector:
            with patch.object(connector, "_request", return_value=mock_response):
                comments = await connector.list_comments("task-1")

                assert len(comments) == 2
                assert comments[0].text == "First comment"

    @pytest.mark.asyncio
    async def test_add_comment(self, connector):
        """Should add a comment to task."""
        mock_response = make_api_response(make_comment_data("com-new", "New comment"))

        async with connector:
            with patch.object(connector, "_request", return_value=mock_response):
                comment = await connector.add_comment("task-1", "New comment")

                assert comment.text == "New comment"


# =============================================================================
# Tag Tests
# =============================================================================


class TestTagOperations:
    """Test tag operations."""

    @pytest.mark.asyncio
    async def test_list_tags(self, connector):
        """Should list tags in workspace."""
        mock_response = {
            "data": [
                {"gid": "tag-1", "name": "urgent"},
                {"gid": "tag-2", "name": "backend"},
            ]
        }

        async with connector:
            with patch.object(connector, "_request", return_value=mock_response):
                tags = await connector.list_tags("12345")

                assert len(tags) == 2
                assert tags[0]["name"] == "urgent"

    @pytest.mark.asyncio
    async def test_add_tag_to_task(self, connector):
        """Should add tag to task."""
        async with connector:
            with patch.object(connector, "_request", return_value={}) as mock_request:
                await connector.add_tag_to_task("task-1", "tag-1")

                mock_request.assert_called_once()


# =============================================================================
# User Tests
# =============================================================================


class TestUserOperations:
    """Test user operations."""

    @pytest.mark.asyncio
    async def test_get_me(self, connector):
        """Should get current user."""
        mock_response = {
            "data": {
                "gid": "user-me",
                "name": "Current User",
                "email": "user@example.com",
            }
        }

        async with connector:
            with patch.object(connector, "_request", return_value=mock_response):
                user = await connector.get_me()

                assert user["gid"] == "user-me"
                assert user["name"] == "Current User"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling."""

    def test_asana_error_creation(self):
        """Should create AsanaError with details."""
        error = AsanaError(
            message="Not found",
            status_code=404,
            errors=[{"message": "Project not found"}],
        )
        assert str(error) == "Not found"
        assert error.status_code == 404
        assert len(error.errors) == 1

    @pytest.mark.asyncio
    async def test_api_error_handling(self, connector):
        """Should handle API errors."""
        import httpx

        async with connector:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.json.return_value = {"errors": [{"message": "Not found"}]}

            connector._client.request = AsyncMock(return_value=mock_response)

            with pytest.raises(AsanaError, match="Not found"):
                await connector._request("GET", "/projects/invalid")

    @pytest.mark.asyncio
    async def test_http_error_handling(self, connector):
        """Should handle HTTP errors."""
        import httpx

        async with connector:
            connector._client.request = AsyncMock(side_effect=httpx.HTTPError("Connection failed"))

            with pytest.raises(AsanaError, match="HTTP error"):
                await connector._request("GET", "/workspaces")


# =============================================================================
# Data Model Tests
# =============================================================================


class TestDataModels:
    """Test data model creation and parsing."""

    def test_credentials_creation(self):
        """Should create credentials with defaults."""
        creds = AsanaCredentials(access_token="token")
        assert creds.access_token == "token"
        assert creds.refresh_token is None
        assert creds.token_type == "bearer"

    def test_workspace_creation(self):
        """Should create workspace."""
        ws = AsanaWorkspace(gid="123", name="Test", is_organization=True)
        assert ws.gid == "123"
        assert ws.is_organization is True

    def test_project_from_api(self):
        """Should parse project from API response."""
        data = make_project_data("proj-1", "My Project")
        project = AsanaProject.from_api(data, "ws-1")

        assert project.gid == "proj-1"
        assert project.name == "My Project"
        assert project.owner == "John Owner"
        assert project.archived is False

    def test_task_from_api(self):
        """Should parse task from API response."""
        data = make_task_data("task-1", "My Task")
        task = AsanaTask.from_api(data)

        assert task.gid == "task-1"
        assert task.name == "My Task"
        assert task.assignee == "Alice Dev"
        assert "urgent" in task.tags
        assert task.num_subtasks == 2

    def test_task_with_custom_fields(self):
        """Should parse custom fields."""
        data = {
            **make_task_data("task-1", "Task"),
            "custom_fields": [
                {"name": "Priority", "display_value": "High"},
                {"name": "Points", "number_value": 5},
                {"name": "Status", "enum_value": {"name": "Active"}},
            ],
        }
        task = AsanaTask.from_api(data)

        assert task.custom_fields["Priority"] == "High"
        assert task.custom_fields["Points"] == 5
        assert task.custom_fields["Status"] == "Active"

    def test_section_from_api(self):
        """Should parse section from API response."""
        data = make_section_data("sec-1", "Done")
        section = AsanaSection.from_api(data, "proj-1")

        assert section.gid == "sec-1"
        assert section.name == "Done"

    def test_comment_from_api(self):
        """Should parse comment from API response."""
        data = make_comment_data("com-1", "My comment")
        comment = AsanaComment.from_api(data)

        assert comment.gid == "com-1"
        assert comment.text == "My comment"
        assert comment.author == "Commenter"

    def test_task_create_request(self):
        """Should create task request with defaults."""
        request = TaskCreateRequest(name="New Task")
        assert request.name == "New Task"
        assert request.notes == ""
        assert request.workspace_gid is None


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Test enum values."""

    def test_task_status(self):
        """Should have correct task status values."""
        assert TaskStatus.INCOMPLETE.value == "incomplete"
        assert TaskStatus.COMPLETE.value == "complete"

    def test_resource_type(self):
        """Should have correct resource type values."""
        assert ResourceType.WORKSPACE.value == "workspace"
        assert ResourceType.PROJECT.value == "project"
        assert ResourceType.TASK.value == "task"
        assert ResourceType.SECTION.value == "section"
        assert ResourceType.TAG.value == "tag"


# =============================================================================
# Pagination Tests
# =============================================================================


class TestPagination:
    """Test pagination functionality."""

    @pytest.mark.asyncio
    async def test_paginate(self, connector):
        """Should paginate through results."""
        page1 = {
            "data": [{"gid": "1"}, {"gid": "2"}],
            "next_page": {"offset": "offset_123"},
        }
        page2 = {
            "data": [{"gid": "3"}],
            "next_page": None,
        }

        call_count = [0]

        async def mock_request(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return page1
            return page2

        async with connector:
            with patch.object(connector, "_request", side_effect=mock_request):
                results = []
                async for item in connector._paginate("/test"):
                    results.append(item)

                assert len(results) == 3
                assert results[-1]["gid"] == "3"
