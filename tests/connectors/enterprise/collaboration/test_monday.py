"""
Tests for Monday.com Enterprise Connector.

Tests the Monday.com GraphQL API integration including:
- Board and workspace operations
- Item (task) CRUD operations
- Column value handling
- Groups (sections)
- Subitems and updates
- Search functionality
- Error handling
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List, Optional

from aragora.connectors.enterprise.collaboration.monday import (
    MondayConnector,
    MondayCredentials,
    MondayBoard,
    MondayItem,
    MondayColumn,
    MondayGroup,
    MondayWorkspace,
    MondayUpdate,
    ColumnType,
    BoardKind,
)
from aragora.connectors.base import Evidence


# =============================================================================
# Concrete Test Subclass
# =============================================================================


class ConcreteMondayConnector(MondayConnector):
    """Concrete implementation of MondayConnector for testing.

    Implements the abstract methods from BaseConnector that are required
    for instantiation but not directly tested here.
    """

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs,
    ) -> List[Evidence]:
        """Implement abstract search method."""
        # For testing - delegates to search_items
        items = await self.search_items(query, limit=limit)
        return [
            Evidence(
                id=f"monday-item-{item.id}",
                content=f"{item.name}: {item.group_title}",
                source_type=self.source_type,
                source_id=str(item.board_id) if item.board_id else "unknown",
                metadata={"item_id": item.id, "board_id": item.board_id},
            )
            for item in items
        ]

    async def fetch(self, evidence_id: str) -> Optional[Evidence]:
        """Implement abstract fetch method."""
        # Extract item ID from evidence_id
        if evidence_id.startswith("monday-item-"):
            item_id = int(evidence_id.replace("monday-item-", ""))
            item = await self.get_item(item_id)
            if item:
                return Evidence(
                    id=evidence_id,
                    content=f"{item.name}: {item.group_title}",
                    source_type=self.source_type,
                    source_id=str(item.board_id) if item.board_id else "unknown",
                    metadata={"item_id": item.id, "board_id": item.board_id},
                )
        return None


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def connector():
    """Create test connector."""
    return ConcreteMondayConnector(
        workspace_ids=[1, 2],
        board_ids=[100, 200],
        max_results=50,
    )


@pytest.fixture
def mock_client():
    """Create mock HTTP client."""
    client = AsyncMock()
    return client


def make_graphql_response(data: Dict[str, Any]) -> MagicMock:
    """Create a mock GraphQL response."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"data": data}
    return response


def make_error_response(status_code: int = 400, errors: list = None) -> MagicMock:
    """Create a mock error response."""
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {"errors": errors or [{"message": "Error"}]}
    return response


# =============================================================================
# Initialization Tests
# =============================================================================


class TestMondayConnectorInit:
    """Test MondayConnector initialization."""

    def test_default_configuration(self):
        """Should use default configuration."""
        connector = ConcreteMondayConnector()
        assert connector.workspace_ids is None
        assert connector.board_ids is None
        assert connector.max_results == 100

    def test_custom_configuration(self):
        """Should accept custom configuration."""
        connector = ConcreteMondayConnector(
            workspace_ids=[1, 2],
            board_ids=[100, 200, 300],
            max_results=50,
        )
        assert connector.workspace_ids == [1, 2]
        assert connector.board_ids == [100, 200, 300]
        assert connector.max_results == 50

    def test_connector_properties(self, connector):
        """Should have correct connector properties."""
        assert connector.name == "Monday.com"
        assert connector.connector_id == "monday"

    def test_source_type(self, connector):
        """Should have correct source type."""
        from aragora.reasoning.provenance import SourceType

        assert connector.source_type == SourceType.DOCUMENT


# =============================================================================
# Authentication Tests
# =============================================================================


class TestAuthentication:
    """Test authentication flows."""

    @pytest.mark.asyncio
    async def test_authenticate_with_token(self, connector):
        """Should authenticate with API token."""
        mock_response = make_graphql_response(
            {"me": {"id": "123", "name": "Test User", "email": "test@example.com"}}
        )

        with patch.object(connector, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await connector.authenticate(api_token="test_token")

            assert result is True
            assert connector._api_token == "test_token"

    @pytest.mark.asyncio
    async def test_get_token_from_env(self, connector):
        """Should get token from environment variable."""
        with patch.dict("os.environ", {"MONDAY_API_TOKEN": "env_token"}):
            token = await connector._get_token()
            assert token == "env_token"

    @pytest.mark.asyncio
    async def test_missing_token_raises_error(self, connector):
        """Should raise error when token not configured."""
        with patch.dict("os.environ", {}, clear=True):
            connector._api_token = None
            with pytest.raises(ValueError, match="API token not configured"):
                await connector._get_token()


# =============================================================================
# Board Operations Tests
# =============================================================================


class TestBoardOperations:
    """Test board-related operations."""

    @pytest.mark.asyncio
    async def test_list_boards(self, connector):
        """Should retrieve list of boards."""
        mock_data = {
            "boards": [
                {
                    "id": "100",
                    "name": "Sprint Board",
                    "workspace": {"id": "1", "name": "Engineering"},
                    "description": "Sprint planning board",
                    "board_kind": "public",
                    "state": "active",
                    "items_count": 25,
                    "permissions": "everyone",
                    "owner": {"id": "123"},
                    "url": "https://monday.com/boards/100",
                },
                {
                    "id": "200",
                    "name": "Bug Tracker",
                    "workspace": {"id": "1", "name": "Engineering"},
                    "description": "Bug tracking",
                    "board_kind": "private",
                    "state": "active",
                    "items_count": 50,
                    "permissions": "owners",
                    "owner": {"id": "123"},
                    "url": "https://monday.com/boards/200",
                },
            ]
        }
        mock_response = make_graphql_response(mock_data)

        with (
            patch.object(connector, "_get_client") as mock_get_client,
            patch.object(connector, "_get_token", return_value="test_token"),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            boards = await connector.list_boards()

            assert len(boards) == 2
            assert boards[0].name == "Sprint Board"
            assert boards[0].item_count == 25
            assert boards[1].board_kind == "private"

    @pytest.mark.asyncio
    async def test_get_board(self, connector):
        """Should retrieve single board by ID."""
        mock_data = {
            "boards": [
                {
                    "id": "100",
                    "name": "Sprint Board",
                    "workspace": {"id": "1", "name": "Engineering"},
                    "description": "Sprint planning",
                    "board_kind": "public",
                    "state": "active",
                    "items_count": 25,
                }
            ]
        }
        mock_response = make_graphql_response(mock_data)

        with (
            patch.object(connector, "_get_client") as mock_get_client,
            patch.object(connector, "_get_token", return_value="test_token"),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            board = await connector.get_board(100)

            assert board is not None
            assert board.id == 100
            assert board.name == "Sprint Board"

    @pytest.mark.asyncio
    async def test_get_board_columns(self, connector):
        """Should retrieve board columns."""
        mock_data = {
            "boards": [
                {
                    "columns": [
                        {"id": "name", "title": "Name", "type": "name"},
                        {"id": "status", "title": "Status", "type": "status"},
                        {"id": "person", "title": "Assignee", "type": "people"},
                        {"id": "date", "title": "Due Date", "type": "date"},
                    ]
                }
            ]
        }
        mock_response = make_graphql_response(mock_data)

        with (
            patch.object(connector, "_get_client") as mock_get_client,
            patch.object(connector, "_get_token", return_value="test_token"),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            columns = await connector.get_board_columns(100)

            assert len(columns) == 4
            assert columns[1].title == "Status"
            assert columns[2].column_type == "people"

    @pytest.mark.asyncio
    async def test_get_board_groups(self, connector):
        """Should retrieve board groups."""
        mock_data = {
            "boards": [
                {
                    "groups": [
                        {"id": "new_group", "title": "New", "color": "#579bfc", "position": "1"},
                        {
                            "id": "in_progress",
                            "title": "In Progress",
                            "color": "#fdab3d",
                            "position": "2",
                        },
                        {"id": "done", "title": "Done", "color": "#00c875", "position": "3"},
                    ]
                }
            ]
        }
        mock_response = make_graphql_response(mock_data)

        with (
            patch.object(connector, "_get_client") as mock_get_client,
            patch.object(connector, "_get_token", return_value="test_token"),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            groups = await connector.get_board_groups(100)

            assert len(groups) == 3
            assert groups[0].title == "New"
            assert groups[1].color == "#fdab3d"


# =============================================================================
# Item Operations Tests
# =============================================================================


class TestItemOperations:
    """Test item (task) CRUD operations."""

    @pytest.mark.asyncio
    async def test_list_items(self, connector):
        """Should retrieve items from board."""
        mock_data = {
            "boards": [
                {
                    "items_page": {
                        "cursor": "next_cursor",
                        "items": [
                            {
                                "id": "1001",
                                "name": "Implement feature X",
                                "state": "active",
                                "group": {"id": "in_progress", "title": "In Progress"},
                                "column_values": [
                                    {
                                        "id": "status",
                                        "text": "Working on it",
                                        "value": '{"index":1}',
                                    },
                                    {
                                        "id": "person",
                                        "text": "Alice",
                                        "value": '{"personsAndTeams":[{"id":123}]}',
                                    },
                                ],
                                "created_at": "2024-01-15T10:00:00Z",
                                "updated_at": "2024-01-15T12:00:00Z",
                            },
                            {
                                "id": "1002",
                                "name": "Fix bug Y",
                                "state": "active",
                                "group": {"id": "new_group", "title": "New"},
                                "column_values": [],
                                "created_at": "2024-01-15T11:00:00Z",
                                "updated_at": "2024-01-15T11:00:00Z",
                            },
                        ],
                    }
                }
            ]
        }
        mock_response = make_graphql_response(mock_data)

        with (
            patch.object(connector, "_get_client") as mock_get_client,
            patch.object(connector, "_get_token", return_value="test_token"),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            items, cursor = await connector.list_items(board_id=100)

            assert len(items) == 2
            assert items[0].name == "Implement feature X"
            assert items[0].group_title == "In Progress"
            assert cursor == "next_cursor"

    @pytest.mark.asyncio
    async def test_get_item(self, connector):
        """Should retrieve single item by ID."""
        mock_data = {
            "items": [
                {
                    "id": "1001",
                    "name": "Implement feature X",
                    "state": "active",
                    "group": {"id": "in_progress", "title": "In Progress"},
                    "board": {"id": "100", "name": "Sprint Board"},
                    "column_values": [
                        {"id": "status", "text": "Working on it"},
                    ],
                    "created_at": "2024-01-15T10:00:00Z",
                    "updated_at": "2024-01-15T12:00:00Z",
                }
            ]
        }
        mock_response = make_graphql_response(mock_data)

        with (
            patch.object(connector, "_get_client") as mock_get_client,
            patch.object(connector, "_get_token", return_value="test_token"),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            item = await connector.get_item(1001)

            assert item is not None
            assert item.id == 1001
            assert item.name == "Implement feature X"

    @pytest.mark.asyncio
    async def test_create_item(self, connector):
        """Should create a new item."""
        mock_data = {
            "create_item": {
                "id": "1003",
                "name": "New Task",
                "state": "active",
                "group": {"id": "new_group", "title": "New"},
                "column_values": [],
                "created_at": "2024-01-15T14:00:00Z",
                "updated_at": "2024-01-15T14:00:00Z",
            }
        }
        mock_response = make_graphql_response(mock_data)

        with (
            patch.object(connector, "_get_client") as mock_get_client,
            patch.object(connector, "_get_token", return_value="test_token"),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            item = await connector.create_item(
                board_id=100, item_name="New Task", group_id="new_group"
            )

            assert item.id == 1003
            assert item.name == "New Task"

    @pytest.mark.asyncio
    async def test_create_item_with_column_values(self, connector):
        """Should create item with column values."""
        mock_data = {
            "create_item": {
                "id": "1004",
                "name": "Task with columns",
                "state": "active",
                "group": {"id": "new_group", "title": "New"},
                "column_values": [
                    {"id": "status", "text": "Working on it"},
                    {"id": "date", "text": "2024-01-20"},
                ],
                "created_at": "2024-01-15T14:00:00Z",
                "updated_at": "2024-01-15T14:00:00Z",
            }
        }
        mock_response = make_graphql_response(mock_data)

        with (
            patch.object(connector, "_get_client") as mock_get_client,
            patch.object(connector, "_get_token", return_value="test_token"),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            item = await connector.create_item(
                board_id=100,
                item_name="Task with columns",
                column_values={"status": {"label": "Working on it"}, "date": "2024-01-20"},
            )

            assert item.id == 1004
            # Verify GraphQL mutation was called with column_values
            call_args = mock_client.post.call_args
            assert "column_values" in str(call_args)

    @pytest.mark.asyncio
    async def test_update_item(self, connector):
        """Should update an existing item."""
        mock_data = {
            "change_multiple_column_values": {
                "id": "1001",
                "name": "Updated Task",
                "state": "active",
                "group": {"id": "in_progress", "title": "In Progress"},
                "column_values": [
                    {"id": "status", "text": "Done"},
                ],
                "created_at": "2024-01-15T10:00:00Z",
                "updated_at": "2024-01-15T12:00:00Z",
            }
        }
        mock_response = make_graphql_response(mock_data)

        with (
            patch.object(connector, "_get_client") as mock_get_client,
            patch.object(connector, "_get_token", return_value="test_token"),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            item = await connector.update_item(
                item_id=1001, column_values={"status": {"label": "Done"}}, board_id=100
            )

            assert item.id == 1001

    @pytest.mark.asyncio
    async def test_delete_item(self, connector):
        """Should delete an item."""
        mock_data = {"delete_item": {"id": "1001"}}
        mock_response = make_graphql_response(mock_data)

        with (
            patch.object(connector, "_get_client") as mock_get_client,
            patch.object(connector, "_get_token", return_value="test_token"),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await connector.delete_item(1001)

            assert result is True


# =============================================================================
# Search Tests
# =============================================================================


class TestSearchOperations:
    """Test search functionality."""

    @pytest.mark.asyncio
    async def test_search_items(self, connector):
        """Should search items by query text."""
        mock_data = {
            "items_page_by_column_values": {
                "cursor": None,
                "items": [
                    {
                        "id": "1001",
                        "name": "Login bug fix",
                        "state": "active",
                        "board": {"id": "100", "name": "Bugs"},
                        "group": {"id": "high_priority", "title": "High Priority"},
                        "column_values": [],
                        "created_at": "2024-01-15T10:00:00Z",
                        "updated_at": "2024-01-15T12:00:00Z",
                    },
                    {
                        "id": "1002",
                        "name": "Login page redesign",
                        "state": "active",
                        "board": {"id": "200", "name": "Features"},
                        "group": {"id": "backlog", "title": "Backlog"},
                        "column_values": [],
                        "created_at": "2024-01-15T10:00:00Z",
                        "updated_at": "2024-01-15T12:00:00Z",
                    },
                ],
            }
        }
        mock_response = make_graphql_response(mock_data)

        with (
            patch.object(connector, "_get_client") as mock_get_client,
            patch.object(connector, "_get_token", return_value="test_token"),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            items = await connector.search_items(query_text="login", board_ids=[100, 200])

            assert len(items) == 2
            assert "login" in items[0].name.lower()


# =============================================================================
# Subitem Tests
# =============================================================================


class TestSubitemOperations:
    """Test subitem operations."""

    @pytest.mark.asyncio
    async def test_create_subitem(self, connector):
        """Should create a subitem."""
        mock_data = {
            "create_subitem": {
                "id": "2001",
                "name": "Subtask 1",
                "state": "active",
                "parent_item": {"id": "1001"},
                "column_values": [],
            }
        }
        mock_response = make_graphql_response(mock_data)

        with (
            patch.object(connector, "_get_client") as mock_get_client,
            patch.object(connector, "_get_token", return_value="test_token"),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            subitem = await connector.create_subitem(parent_item_id=1001, subitem_name="Subtask 1")

            assert subitem.id == 2001
            assert subitem.name == "Subtask 1"


# =============================================================================
# Update (Comment) Tests
# =============================================================================


class TestUpdateOperations:
    """Test update (comment) operations."""

    @pytest.mark.asyncio
    async def test_create_update(self, connector):
        """Should create an update (comment) on item."""
        mock_data = {
            "create_update": {
                "id": "3001",
                "body": "This is a comment",
                "creator": {"id": "123", "name": "Test User"},
                "created_at": "2024-01-15T15:00:00Z",
            }
        }
        mock_response = make_graphql_response(mock_data)

        with (
            patch.object(connector, "_get_client") as mock_get_client,
            patch.object(connector, "_get_token", return_value="test_token"),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            update = await connector.create_update(item_id=1001, body="This is a comment")

            assert update.id == 3001
            assert update.body == "This is a comment"


# =============================================================================
# Workspace Tests
# =============================================================================


class TestWorkspaceOperations:
    """Test workspace operations."""

    @pytest.mark.asyncio
    async def test_list_workspaces(self, connector):
        """Should retrieve list of workspaces."""
        mock_data = {
            "workspaces": [
                {"id": "1", "name": "Engineering", "kind": "open", "description": "Eng team"},
                {"id": "2", "name": "Product", "kind": "open", "description": "Product team"},
            ]
        }
        mock_response = make_graphql_response(mock_data)

        with (
            patch.object(connector, "_get_client") as mock_get_client,
            patch.object(connector, "_get_token", return_value="test_token"),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            workspaces = await connector.list_workspaces()

            assert len(workspaces) == 2
            assert workspaces[0].name == "Engineering"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_graphql_error(self, connector):
        """Should handle GraphQL errors."""
        mock_response = make_error_response(
            200, [{"message": "Invalid query", "extensions": {"code": "INVALID_QUERY"}}]
        )

        with (
            patch.object(connector, "_get_client") as mock_get_client,
            patch.object(connector, "_get_token", return_value="test_token"),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(ValueError, match="GraphQL error"):
                await connector.list_boards()

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, connector):
        """Should handle rate limit errors."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"errors": [{"message": "Rate limit exceeded"}]}

        with (
            patch.object(connector, "_get_client") as mock_get_client,
            patch.object(connector, "_get_token", return_value="test_token"),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(ValueError, match="GraphQL error"):
                await connector.list_boards()

    @pytest.mark.asyncio
    async def test_authentication_error(self, connector):
        """Should handle authentication errors."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"errors": [{"message": "Invalid token"}]}

        with (
            patch.object(connector, "_get_client") as mock_get_client,
            patch.object(connector, "_get_token", return_value="test_token"),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(ValueError, match="GraphQL error"):
                await connector.get_board(100)

    @pytest.mark.asyncio
    async def test_network_timeout(self, connector):
        """Should handle network timeouts."""
        import httpx

        with (
            patch.object(connector, "_get_client") as mock_get_client,
            patch.object(connector, "_get_token", return_value="test_token"),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_get_client.return_value = mock_client

            with pytest.raises(httpx.TimeoutException):
                await connector.list_boards()


# =============================================================================
# Data Model Tests
# =============================================================================


class TestDataModels:
    """Test data model parsing."""

    def test_monday_board_from_api(self):
        """Should parse board from API response."""
        data = {
            "id": "100",
            "name": "Sprint Board",
            "workspace": {"id": "1", "name": "Engineering"},
            "description": "Sprint planning",
            "board_kind": "public",
            "state": "active",
            "items_count": 25,
            "permissions": "everyone",
            "owner": {"id": "123"},
            "url": "https://monday.com/boards/100",
        }
        board = MondayBoard.from_api(data)
        assert board.id == 100
        assert board.name == "Sprint Board"
        assert board.workspace_id == 1
        assert board.item_count == 25

    def test_monday_workspace_from_api(self):
        """Should parse workspace from API response."""
        data = {
            "id": "1",
            "name": "Engineering",
            "kind": "open",
            "description": "Engineering team workspace",
        }
        workspace = MondayWorkspace.from_api(data)
        assert workspace.id == 1
        assert workspace.name == "Engineering"
        assert workspace.kind == "open"

    def test_column_type_enum(self):
        """Should correctly map column types."""
        assert ColumnType.TEXT.value == "text"
        assert ColumnType.STATUS.value == "status"
        assert ColumnType.PERSON.value == "people"
        assert ColumnType.DATE.value == "date"
        assert ColumnType.CHECKBOX.value == "checkbox"

    def test_board_kind_enum(self):
        """Should correctly map board kinds."""
        assert BoardKind.PUBLIC.value == "public"
        assert BoardKind.PRIVATE.value == "private"
        assert BoardKind.SHARE.value == "share"
