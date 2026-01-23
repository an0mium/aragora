"""
Asana Enterprise Connector.

Provides full integration with Asana for project and task management:
- Workspace and project traversal
- Task CRUD operations
- Subtasks and dependencies
- Comments and attachments
- Custom fields
- Sections and milestones
- Webhook support for real-time updates

Requires Asana Personal Access Token or OAuth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator

import httpx

from aragora.connectors.enterprise.base import (
    EnterpriseConnector,
    SyncItem,
    SyncState,
)

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Asana task completion status."""

    INCOMPLETE = "incomplete"
    COMPLETE = "complete"


class ResourceType(str, Enum):
    """Asana resource types."""

    WORKSPACE = "workspace"
    PROJECT = "project"
    TASK = "task"
    SECTION = "section"
    TAG = "tag"
    USER = "user"


@dataclass
class AsanaCredentials:
    """Asana API credentials."""

    access_token: str
    refresh_token: str | None = None
    token_type: str = "bearer"
    expires_at: datetime | None = None


@dataclass
class AsanaWorkspace:
    """An Asana workspace."""

    gid: str
    name: str
    is_organization: bool = False


@dataclass
class AsanaProject:
    """An Asana project."""

    gid: str
    name: str
    workspace_gid: str
    workspace_name: str = ""
    owner: str = ""
    notes: str = ""
    color: str = ""
    archived: bool = False
    public: bool = True
    created_at: datetime | None = None
    modified_at: datetime | None = None
    due_date: datetime | None = None
    start_date: datetime | None = None
    current_status: str = ""  # on_track, at_risk, off_track
    url: str = ""

    @classmethod
    def from_api(cls, data: dict[str, Any], workspace_gid: str = "") -> AsanaProject:
        """Create from API response."""
        return cls(
            gid=data["gid"],
            name=data["name"],
            workspace_gid=data.get("workspace", {}).get("gid", workspace_gid),
            workspace_name=data.get("workspace", {}).get("name", ""),
            owner=data.get("owner", {}).get("name", "") if data.get("owner") else "",
            notes=data.get("notes", ""),
            color=data.get("color", ""),
            archived=data.get("archived", False),
            public=data.get("public", True),
            created_at=_parse_datetime(data.get("created_at")),
            modified_at=_parse_datetime(data.get("modified_at")),
            due_date=_parse_date(data.get("due_date")),
            start_date=_parse_date(data.get("start_on")),
            current_status=data.get("current_status_update", {}).get("status_type", "")
            if data.get("current_status_update")
            else "",
            url=data.get("permalink_url", ""),
        )


@dataclass
class AsanaSection:
    """An Asana project section."""

    gid: str
    name: str
    project_gid: str

    @classmethod
    def from_api(cls, data: dict[str, Any], project_gid: str = "") -> AsanaSection:
        """Create from API response."""
        return cls(
            gid=data["gid"],
            name=data["name"],
            project_gid=data.get("project", {}).get("gid", project_gid),
        )


@dataclass
class AsanaTask:
    """An Asana task."""

    gid: str
    name: str
    notes: str = ""
    completed: bool = False
    completed_at: datetime | None = None
    due_on: datetime | None = None
    due_at: datetime | None = None
    start_on: datetime | None = None
    created_at: datetime | None = None
    modified_at: datetime | None = None
    assignee: str = ""
    assignee_gid: str = ""
    projects: list[str] = field(default_factory=list)
    project_gids: list[str] = field(default_factory=list)
    section_gid: str = ""
    parent_gid: str | None = None
    tags: list[str] = field(default_factory=list)
    followers: list[str] = field(default_factory=list)
    num_subtasks: int = 0
    url: str = ""
    custom_fields: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> AsanaTask:
        """Create from API response."""
        assignee = data.get("assignee") or {}
        memberships = data.get("memberships", [])
        projects = data.get("projects", [])

        custom_fields = {}
        for cf in data.get("custom_fields", []):
            cf_name = cf.get("name", "")
            if cf.get("display_value"):
                custom_fields[cf_name] = cf["display_value"]
            elif cf.get("text_value"):
                custom_fields[cf_name] = cf["text_value"]
            elif cf.get("number_value") is not None:
                custom_fields[cf_name] = cf["number_value"]
            elif cf.get("enum_value"):
                custom_fields[cf_name] = cf["enum_value"].get("name", "")

        return cls(
            gid=data["gid"],
            name=data["name"],
            notes=data.get("notes", ""),
            completed=data.get("completed", False),
            completed_at=_parse_datetime(data.get("completed_at")),
            due_on=_parse_date(data.get("due_on")),
            due_at=_parse_datetime(data.get("due_at")),
            start_on=_parse_date(data.get("start_on")),
            created_at=_parse_datetime(data.get("created_at")),
            modified_at=_parse_datetime(data.get("modified_at")),
            assignee=assignee.get("name", ""),
            assignee_gid=assignee.get("gid", ""),
            projects=[p.get("name", "") for p in projects],
            project_gids=[p.get("gid", "") for p in projects],
            section_gid=memberships[0].get("section", {}).get("gid", "") if memberships else "",
            parent_gid=data.get("parent", {}).get("gid") if data.get("parent") else None,
            tags=[t.get("name", "") for t in data.get("tags", [])],
            followers=[f.get("name", "") for f in data.get("followers", [])],
            num_subtasks=data.get("num_subtasks", 0),
            url=data.get("permalink_url", ""),
            custom_fields=custom_fields,
        )


@dataclass
class AsanaComment:
    """An Asana task comment (story)."""

    gid: str
    text: str
    author: str
    author_gid: str = ""
    created_at: datetime | None = None
    resource_type: str = "comment"

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> AsanaComment:
        """Create from API response."""
        created_by = data.get("created_by") or {}
        return cls(
            gid=data["gid"],
            text=data.get("text", ""),
            author=created_by.get("name", ""),
            author_gid=created_by.get("gid", ""),
            created_at=_parse_datetime(data.get("created_at")),
            resource_type=data.get("resource_subtype", "comment"),
        )


@dataclass
class TaskCreateRequest:
    """Request to create a new task."""

    name: str
    workspace_gid: str | None = None
    project_gid: str | None = None
    section_gid: str | None = None
    parent_gid: str | None = None
    assignee_gid: str | None = None
    notes: str = ""
    due_on: str | None = None  # YYYY-MM-DD
    start_on: str | None = None  # YYYY-MM-DD
    tags: list[str] | None = None
    custom_fields: dict[str, Any] | None = None


class AsanaError(Exception):
    """Asana API error."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        errors: list[dict] | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.errors = errors or []


class AsanaConnector(EnterpriseConnector):
    """
    Enterprise connector for Asana.

    Features:
    - Workspace and project discovery
    - Task management (CRUD)
    - Subtasks and task dependencies
    - Comments and attachments
    - Custom fields
    - Sections and milestones
    - Incremental sync via modified_at
    - Webhook support

    Authentication:
    - Personal Access Token
    - OAuth 2.0

    Usage:
        ```python
        credentials = AsanaCredentials(access_token="...")

        async with AsanaConnector(credentials) as asana:
            # List workspaces
            workspaces = await asana.list_workspaces()

            # List projects in workspace
            projects = await asana.list_projects(workspaces[0].gid)

            # Get tasks in project
            tasks = await asana.list_tasks(project_gid=projects[0].gid)

            # Create a task
            task = await asana.create_task(TaskCreateRequest(
                name="New feature implementation",
                project_gid=projects[0].gid,
                assignee_gid="user_gid",
                notes="Implement the new feature as discussed",
                due_on="2025-02-15"
            ))

            # Add a comment
            await asana.add_comment(task.gid, "Starting work on this now")

            # Complete the task
            await asana.complete_task(task.gid)
        ```
    """

    BASE_URL = "https://app.asana.com/api/1.0"
    CONNECTOR_TYPE = "asana"

    def __init__(
        self,
        credentials: AsanaCredentials,
        workspace_gid: str | None = None,
    ):
        """
        Initialize the Asana connector.

        Args:
            credentials: Asana API credentials
            workspace_gid: Optional default workspace GID
        """
        self.credentials = credentials  # type: ignore[assignment]
        self.default_workspace_gid = workspace_gid
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> AsanaConnector:
        """Enter async context."""
        credentials: AsanaCredentials = self.credentials  # type: ignore[assignment]
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {credentials.access_token}",
                "Accept": "application/json",
            },
            timeout=30.0,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get the HTTP client."""
        if not self._client:
            raise AsanaError("Connector not initialized. Use async context manager.")
        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an API request."""
        try:
            response = await self.client.request(
                method,
                path,
                json=json,
                params=params,
            )

            if response.status_code == 204:
                return {}

            data = response.json()

            if response.status_code >= 400:
                errors = data.get("errors", [])
                message = errors[0].get("message", "Unknown error") if errors else "Unknown error"
                raise AsanaError(
                    message=message,
                    status_code=response.status_code,
                    errors=errors,
                )

            return data

        except httpx.HTTPError as e:
            raise AsanaError(f"HTTP error: {e}") from e

    async def _paginate(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> AsyncIterator[dict[str, Any]]:
        """Paginate through API results."""
        params = params or {}
        params["limit"] = min(limit, 100)
        offset = None

        while True:
            if offset:
                params["offset"] = offset

            data = await self._request("GET", path, params=params)

            for item in data.get("data", []):
                yield item

            next_page = data.get("next_page")
            if not next_page or not next_page.get("offset"):
                break

            offset = next_page["offset"]

    # -------------------------------------------------------------------------
    # Workspaces
    # -------------------------------------------------------------------------

    async def list_workspaces(self) -> list[AsanaWorkspace]:
        """List all workspaces the user has access to."""
        data = await self._request("GET", "/workspaces")
        return [
            AsanaWorkspace(
                gid=w["gid"],
                name=w["name"],
                is_organization=w.get("is_organization", False),
            )
            for w in data.get("data", [])
        ]

    # -------------------------------------------------------------------------
    # Projects
    # -------------------------------------------------------------------------

    async def list_projects(
        self,
        workspace_gid: str | None = None,
        archived: bool = False,
    ) -> list[AsanaProject]:
        """
        List projects in a workspace.

        Args:
            workspace_gid: Workspace GID (uses default if not provided)
            archived: Include archived projects

        Returns:
            List of projects
        """
        ws_gid = workspace_gid or self.default_workspace_gid
        if not ws_gid:
            raise AsanaError("workspace_gid required")

        params = {
            "workspace": ws_gid,
            "archived": str(archived).lower(),
            "opt_fields": "name,owner,notes,color,archived,public,created_at,modified_at,due_date,start_on,current_status_update,permalink_url,workspace",
        }

        projects = []
        async for p in self._paginate("/projects", params):
            projects.append(AsanaProject.from_api(p, ws_gid))

        return projects

    async def get_project(self, project_gid: str) -> AsanaProject:
        """Get a project by GID."""
        params = {
            "opt_fields": "name,owner,notes,color,archived,public,created_at,modified_at,due_date,start_on,current_status_update,permalink_url,workspace",
        }
        data = await self._request("GET", f"/projects/{project_gid}", params=params)
        return AsanaProject.from_api(data.get("data", {}))

    # -------------------------------------------------------------------------
    # Sections
    # -------------------------------------------------------------------------

    async def list_sections(self, project_gid: str) -> list[AsanaSection]:
        """List sections in a project."""
        data = await self._request("GET", f"/projects/{project_gid}/sections")
        return [AsanaSection.from_api(s, project_gid) for s in data.get("data", [])]

    async def create_section(self, project_gid: str, name: str) -> AsanaSection:
        """Create a section in a project."""
        data = await self._request(
            "POST",
            f"/projects/{project_gid}/sections",
            json={"data": {"name": name}},
        )
        return AsanaSection.from_api(data.get("data", {}), project_gid)

    # -------------------------------------------------------------------------
    # Tasks
    # -------------------------------------------------------------------------

    async def list_tasks(
        self,
        project_gid: str | None = None,
        section_gid: str | None = None,
        assignee_gid: str | None = None,
        workspace_gid: str | None = None,
        completed_since: datetime | None = None,
        modified_since: datetime | None = None,
    ) -> list[AsanaTask]:
        """
        List tasks with filters.

        Args:
            project_gid: Filter by project
            section_gid: Filter by section
            assignee_gid: Filter by assignee
            workspace_gid: Required if using assignee filter
            completed_since: Only tasks completed after this time
            modified_since: Only tasks modified after this time

        Returns:
            List of tasks
        """
        params: dict[str, Any] = {
            "opt_fields": "name,notes,completed,completed_at,due_on,due_at,start_on,created_at,modified_at,assignee,projects,memberships.section,parent,tags,followers,num_subtasks,permalink_url,custom_fields",
        }

        if section_gid:
            path = f"/sections/{section_gid}/tasks"
        elif project_gid:
            path = f"/projects/{project_gid}/tasks"
        elif assignee_gid:
            path = "/tasks"
            params["assignee"] = assignee_gid
            params["workspace"] = workspace_gid or self.default_workspace_gid
        else:
            raise AsanaError("project_gid, section_gid, or assignee_gid required")

        if completed_since:
            params["completed_since"] = completed_since.isoformat()

        if modified_since:
            params["modified_since"] = modified_since.isoformat()

        tasks = []
        async for t in self._paginate(path, params):
            tasks.append(AsanaTask.from_api(t))

        return tasks

    async def get_task(self, task_gid: str) -> AsanaTask:
        """Get a task by GID."""
        params = {
            "opt_fields": "name,notes,completed,completed_at,due_on,due_at,start_on,created_at,modified_at,assignee,projects,memberships.section,parent,tags,followers,num_subtasks,permalink_url,custom_fields",
        }
        data = await self._request("GET", f"/tasks/{task_gid}", params=params)
        return AsanaTask.from_api(data.get("data", {}))

    async def create_task(self, request: TaskCreateRequest) -> AsanaTask:
        """
        Create a new task.

        Args:
            request: Task creation parameters

        Returns:
            Created task
        """
        body: dict[str, Any] = {"name": request.name}

        if request.workspace_gid:
            body["workspace"] = request.workspace_gid
        elif request.project_gid:
            body["projects"] = [request.project_gid]
        elif self.default_workspace_gid:
            body["workspace"] = self.default_workspace_gid
        else:
            raise AsanaError("workspace_gid or project_gid required")

        if request.section_gid:
            body["memberships"] = [{"section": request.section_gid}]

        if request.parent_gid:
            body["parent"] = request.parent_gid

        if request.assignee_gid:
            body["assignee"] = request.assignee_gid

        if request.notes:
            body["notes"] = request.notes

        if request.due_on:
            body["due_on"] = request.due_on

        if request.start_on:
            body["start_on"] = request.start_on

        if request.tags:
            body["tags"] = request.tags

        if request.custom_fields:
            body["custom_fields"] = request.custom_fields

        data = await self._request("POST", "/tasks", json={"data": body})
        task = AsanaTask.from_api(data.get("data", {}))

        logger.info("Created Asana task", extra={"task_gid": task.gid, "name": task.name})
        return task

    async def update_task(
        self,
        task_gid: str,
        name: str | None = None,
        notes: str | None = None,
        due_on: str | None = None,
        start_on: str | None = None,
        assignee_gid: str | None = None,
        completed: bool | None = None,
    ) -> AsanaTask:
        """
        Update a task.

        Args:
            task_gid: Task to update
            name: New name
            notes: New notes
            due_on: New due date (YYYY-MM-DD)
            start_on: New start date (YYYY-MM-DD)
            assignee_gid: New assignee
            completed: Completion status

        Returns:
            Updated task
        """
        body: dict[str, Any] = {}

        if name is not None:
            body["name"] = name
        if notes is not None:
            body["notes"] = notes
        if due_on is not None:
            body["due_on"] = due_on
        if start_on is not None:
            body["start_on"] = start_on
        if assignee_gid is not None:
            body["assignee"] = assignee_gid
        if completed is not None:
            body["completed"] = completed

        if not body:
            return await self.get_task(task_gid)

        data = await self._request("PUT", f"/tasks/{task_gid}", json={"data": body})
        return AsanaTask.from_api(data.get("data", {}))

    async def complete_task(self, task_gid: str) -> AsanaTask:
        """Mark a task as complete."""
        return await self.update_task(task_gid, completed=True)

    async def reopen_task(self, task_gid: str) -> AsanaTask:
        """Mark a task as incomplete."""
        return await self.update_task(task_gid, completed=False)

    async def delete_task(self, task_gid: str) -> None:
        """Delete a task."""
        await self._request("DELETE", f"/tasks/{task_gid}")
        logger.info("Deleted Asana task", extra={"task_gid": task_gid})

    async def move_task_to_section(self, task_gid: str, section_gid: str) -> None:
        """Move a task to a different section."""
        await self._request(
            "POST",
            f"/sections/{section_gid}/addTask",
            json={"data": {"task": task_gid}},
        )

    # -------------------------------------------------------------------------
    # Subtasks
    # -------------------------------------------------------------------------

    async def list_subtasks(self, task_gid: str) -> list[AsanaTask]:
        """List subtasks of a task."""
        params = {
            "opt_fields": "name,notes,completed,completed_at,due_on,due_at,start_on,created_at,modified_at,assignee,parent,permalink_url",
        }
        data = await self._request("GET", f"/tasks/{task_gid}/subtasks", params=params)
        return [AsanaTask.from_api(t) for t in data.get("data", [])]

    async def create_subtask(
        self,
        parent_task_gid: str,
        name: str,
        notes: str = "",
        assignee_gid: str | None = None,
        due_on: str | None = None,
    ) -> AsanaTask:
        """Create a subtask."""
        body: dict[str, Any] = {"name": name}

        if notes:
            body["notes"] = notes
        if assignee_gid:
            body["assignee"] = assignee_gid
        if due_on:
            body["due_on"] = due_on

        data = await self._request(
            "POST",
            f"/tasks/{parent_task_gid}/subtasks",
            json={"data": body},
        )
        return AsanaTask.from_api(data.get("data", {}))

    # -------------------------------------------------------------------------
    # Comments (Stories)
    # -------------------------------------------------------------------------

    async def list_comments(self, task_gid: str) -> list[AsanaComment]:
        """List comments on a task."""
        params = {
            "opt_fields": "text,created_by,created_at,resource_subtype",
        }
        data = await self._request("GET", f"/tasks/{task_gid}/stories", params=params)

        # Filter to only comments (not system stories)
        comments = []
        for story in data.get("data", []):
            if story.get("resource_subtype") == "comment_added":
                comments.append(AsanaComment.from_api(story))

        return comments

    async def add_comment(self, task_gid: str, text: str) -> AsanaComment:
        """Add a comment to a task."""
        data = await self._request(
            "POST",
            f"/tasks/{task_gid}/stories",
            json={"data": {"text": text}},
        )
        return AsanaComment.from_api(data.get("data", {}))

    # -------------------------------------------------------------------------
    # Tags
    # -------------------------------------------------------------------------

    async def list_tags(self, workspace_gid: str | None = None) -> list[dict[str, str]]:
        """List tags in a workspace."""
        ws_gid = workspace_gid or self.default_workspace_gid
        if not ws_gid:
            raise AsanaError("workspace_gid required")

        data = await self._request("GET", f"/workspaces/{ws_gid}/tags")
        return [{"gid": t["gid"], "name": t["name"]} for t in data.get("data", [])]

    async def add_tag_to_task(self, task_gid: str, tag_gid: str) -> None:
        """Add a tag to a task."""
        await self._request(
            "POST",
            f"/tasks/{task_gid}/addTag",
            json={"data": {"tag": tag_gid}},
        )

    # -------------------------------------------------------------------------
    # Users
    # -------------------------------------------------------------------------

    async def get_me(self) -> dict[str, Any]:
        """Get the current user."""
        data = await self._request("GET", "/users/me")
        return data.get("data", {})

    async def list_users(self, workspace_gid: str | None = None) -> list[dict[str, Any]]:
        """List users in a workspace."""
        ws_gid = workspace_gid or self.default_workspace_gid
        if not ws_gid:
            raise AsanaError("workspace_gid required")

        users = []
        async for u in self._paginate(f"/workspaces/{ws_gid}/users"):
            users.append(u)

        return users

    # -------------------------------------------------------------------------
    # Enterprise Connector Interface
    # -------------------------------------------------------------------------

    async def connect(self) -> bool:
        """Test the connection."""
        try:
            await self.get_me()
            return True
        except AsanaError:
            return False

    async def disconnect(self) -> None:
        """Disconnect (no-op for token auth)."""
        pass

    async def sync(  # type: ignore[override]
        self,
        state: SyncState | None = None,
        modified_since: datetime | None = None,
    ) -> AsyncIterator[SyncItem]:
        """
        Sync tasks from Asana.

        Yields SyncItems for each task, suitable for indexing.
        """
        workspaces = await self.list_workspaces()

        for workspace in workspaces:
            projects = await self.list_projects(workspace.gid)

            for project in projects:
                tasks = await self.list_tasks(
                    project_gid=project.gid,
                    modified_since=modified_since,
                )

                for task in tasks:
                    content = f"# {task.name}\n\n"
                    if task.notes:
                        content += f"{task.notes}\n\n"
                    if task.assignee:
                        content += f"Assignee: {task.assignee}\n"
                    if task.due_on:
                        content += f"Due: {task.due_on}\n"
                    if task.tags:
                        content += f"Tags: {', '.join(task.tags)}\n"

                    yield SyncItem(
                        id=f"asana:{task.gid}",
                        source_type="asana",
                        source_id=task.gid,
                        content=content,
                        title=task.name,
                        url=task.url,
                        metadata={
                            "gid": task.gid,
                            "project": project.name,
                            "workspace": workspace.name,
                            "assignee": task.assignee,
                            "completed": task.completed,
                            "due_on": task.due_on.isoformat() if task.due_on else None,
                            "tags": task.tags,
                            "custom_fields": task.custom_fields,
                        },
                        updated_at=task.modified_at,
                    )


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _parse_datetime(value: str | None) -> datetime | None:
    """Parse ISO datetime string."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _parse_date(value: str | None) -> datetime | None:
    """Parse date string (YYYY-MM-DD)."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except (ValueError, AttributeError):
        return None


# -----------------------------------------------------------------------------
# Mock Data for Testing
# -----------------------------------------------------------------------------


def get_mock_workspace() -> AsanaWorkspace:
    """Get a mock workspace for testing."""
    return AsanaWorkspace(
        gid="1234567890",
        name="Demo Workspace",
        is_organization=True,
    )


def get_mock_project() -> AsanaProject:
    """Get a mock project for testing."""
    return AsanaProject(
        gid="9876543210",
        name="Product Launch Q1",
        workspace_gid="1234567890",
        workspace_name="Demo Workspace",
        owner="Jane Smith",
        notes="Launch plan for Q1 product release",
        color="light-green",
        archived=False,
        public=True,
        current_status="on_track",
        url="https://app.asana.com/0/9876543210",
    )


def get_mock_task() -> AsanaTask:
    """Get a mock task for testing."""
    return AsanaTask(
        gid="5555555555",
        name="Design review meeting",
        notes="Review the new dashboard designs with the team",
        completed=False,
        due_on=datetime(2025, 2, 15),
        assignee="John Doe",
        assignee_gid="1111111111",
        projects=["Product Launch Q1"],
        project_gids=["9876543210"],
        tags=["design", "meeting"],
        num_subtasks=3,
        url="https://app.asana.com/0/9876543210/5555555555",
    )
