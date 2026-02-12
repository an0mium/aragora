"""
Extended tests for Monday.com Enterprise Connector.

Covers functionality NOT already tested in test_monday.py:
- Data model serialization (to_dict) and edge cases
- MondayColumnValue, MondayUpdate from_api and to_dict
- MondayItem advanced parsing (subitems, parent_item, get_column_value)
- _parse_datetime edge cases
- Connector is_configured property
- _get_client lifecycle (creation, reuse, recreation after close)
- _get_token priority (instance token vs env var)
- authenticate failure paths (no token, API rejection)
- _graphql_request header construction, variable passing, error handling
- list_boards with filters (workspace_id, limit, state)
- get_board returning None for missing board
- get_board_columns / get_board_groups returning empty for missing board
- list_items with group_id filter, no cursor, empty boards
- get_item returning None
- create_item without group_id, without column_values
- update_item without board_id (auto-fetch), item not found
- move_item_to_group
- archive_item
- create_subitem with column_values
- list_updates, list_updates empty
- create_update body escaping
- search_items without board_ids, search failure handling
- sync_items iteration (with board_ids, without board_ids, pagination)
- _item_to_sync_item mapping
- close() method
- HTTP status code errors (raise_for_status)
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from typing import Any, Optional

import httpx

from aragora.connectors.enterprise.collaboration.monday import (
    MondayConnector,
    MondayCredentials,
    MondayBoard,
    MondayItem,
    MondayColumn,
    MondayGroup,
    MondayWorkspace,
    MondayUpdate,
    MondayColumnValue,
    ColumnType,
    BoardKind,
    MONDAY_API_URL,
    MONDAY_AUTH_URL,
    MONDAY_TOKEN_URL,
    _parse_datetime,
)
from aragora.connectors.enterprise.base import SyncItem, SyncState
from aragora.connectors.base import Evidence
from aragora.reasoning.provenance import SourceType


# =============================================================================
# Concrete Test Subclass (same pattern as existing tests)
# =============================================================================


class ConcreteMondayConnector(MondayConnector):
    """Concrete implementation of MondayConnector for testing."""

    async def search(self, query: str, limit: int = 10, **kwargs) -> list[Evidence]:
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

    async def fetch(self, evidence_id: str) -> Evidence | None:
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
    """Create test connector with default settings."""
    return ConcreteMondayConnector(
        workspace_ids=[1, 2],
        board_ids=[100, 200],
        max_results=50,
    )


@pytest.fixture
def bare_connector():
    """Create connector with no workspace/board filtering."""
    return ConcreteMondayConnector()


def make_graphql_response(data: dict[str, Any]) -> MagicMock:
    """Create a mock GraphQL response."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"data": data}
    response.raise_for_status = MagicMock()
    return response


def make_error_graphql_response(errors: list[dict]) -> MagicMock:
    """Create a mock GraphQL response with errors in body (200 status)."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"errors": errors}
    response.raise_for_status = MagicMock()
    return response


def make_http_error_response(status_code: int) -> MagicMock:
    """Create a mock response that raises on raise_for_status."""
    response = MagicMock()
    response.status_code = status_code
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        f"{status_code} error",
        request=MagicMock(),
        response=MagicMock(status_code=status_code),
    )
    return response


def _patch_graphql(connector, data):
    """Context manager shortcut to patch connector for a successful GraphQL call."""
    mock_response = make_graphql_response(data)
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    return (
        patch.object(connector, "_get_client", return_value=mock_client),
        patch.object(connector, "_get_token", return_value="test_token"),
        mock_client,
    )


# =============================================================================
# API Constants Tests
# =============================================================================


class TestAPIConstants:
    """Test that API constants are correct."""

    def test_api_url(self):
        assert MONDAY_API_URL == "https://api.monday.com/v2"

    def test_auth_url(self):
        assert MONDAY_AUTH_URL == "https://auth.monday.com/oauth2/authorize"

    def test_token_url(self):
        assert MONDAY_TOKEN_URL == "https://auth.monday.com/oauth2/token"


# =============================================================================
# _parse_datetime Tests
# =============================================================================


class TestParseDatetime:
    """Test the _parse_datetime utility function."""

    def test_parse_iso_format(self):
        result = _parse_datetime("2024-01-15T10:30:00+00:00")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30

    def test_parse_z_suffix(self):
        result = _parse_datetime("2024-06-01T08:00:00Z")
        assert result is not None
        assert result.year == 2024
        assert result.month == 6

    def test_parse_none(self):
        assert _parse_datetime(None) is None

    def test_parse_empty_string(self):
        assert _parse_datetime("") is None

    def test_parse_invalid_format(self):
        assert _parse_datetime("not-a-date") is None

    def test_parse_partial_date(self):
        # fromisoformat can parse date-only strings
        result = _parse_datetime("2024-01-15")
        assert result is not None
        assert result.year == 2024

    def test_parse_with_timezone_offset(self):
        result = _parse_datetime("2024-03-20T14:30:00-05:00")
        assert result is not None
        assert result.hour == 14


# =============================================================================
# MondayCredentials Tests
# =============================================================================


class TestMondayCredentials:
    """Test MondayCredentials dataclass."""

    def test_default_values(self):
        creds = MondayCredentials(api_token="abc123")
        assert creds.api_token == "abc123"
        assert creds.refresh_token is None
        assert creds.token_type == "bearer"
        assert creds.expires_at is None

    def test_full_values(self):
        now = datetime.now(timezone.utc)
        creds = MondayCredentials(
            api_token="abc123",
            refresh_token="refresh_xyz",
            token_type="bearer",
            expires_at=now,
        )
        assert creds.refresh_token == "refresh_xyz"
        assert creds.expires_at == now


# =============================================================================
# MondayWorkspace Model Tests
# =============================================================================


class TestMondayWorkspaceModel:
    """Test MondayWorkspace data model."""

    def test_from_api_full(self):
        data = {
            "id": "42",
            "name": "Marketing",
            "kind": "closed",
            "description": "Marketing team workspace",
        }
        ws = MondayWorkspace.from_api(data)
        assert ws.id == 42
        assert ws.name == "Marketing"
        assert ws.kind == "closed"
        assert ws.description == "Marketing team workspace"

    def test_from_api_minimal(self):
        data = {"id": "1", "name": "Default"}
        ws = MondayWorkspace.from_api(data)
        assert ws.kind == "open"
        assert ws.description == ""

    def test_to_dict(self):
        ws = MondayWorkspace(id=10, name="Dev", kind="open", description="Dev workspace")
        d = ws.to_dict()
        assert d["id"] == 10
        assert d["name"] == "Dev"
        assert d["kind"] == "open"
        assert d["description"] == "Dev workspace"


# =============================================================================
# MondayBoard Model Tests
# =============================================================================


class TestMondayBoardModel:
    """Test MondayBoard data model edge cases."""

    def test_from_api_no_workspace(self):
        data = {
            "id": "100",
            "name": "Standalone Board",
            "workspace": None,
            "owner": None,
        }
        board = MondayBoard.from_api(data)
        assert board.workspace_id is None
        assert board.workspace_name == ""
        assert board.owner_id is None

    def test_from_api_with_dates(self):
        data = {
            "id": "100",
            "name": "Board",
            "created_at": "2024-01-10T09:00:00Z",
            "updated_at": "2024-06-15T18:30:00Z",
        }
        board = MondayBoard.from_api(data)
        assert board.created_at is not None
        assert board.created_at.year == 2024
        assert board.updated_at is not None
        assert board.updated_at.month == 6

    def test_to_dict_with_dates(self):
        board = MondayBoard(
            id=100,
            name="Board",
            created_at=datetime(2024, 1, 15, 10, 0, 0),
            updated_at=datetime(2024, 6, 1, 12, 0, 0),
        )
        d = board.to_dict()
        assert d["created_at"] == "2024-01-15T10:00:00"
        assert d["updated_at"] == "2024-06-01T12:00:00"

    def test_to_dict_without_dates(self):
        board = MondayBoard(id=100, name="Board")
        d = board.to_dict()
        assert d["created_at"] is None
        assert d["updated_at"] is None

    def test_to_dict_all_fields(self):
        board = MondayBoard(
            id=100,
            name="Sprint Board",
            workspace_id=1,
            workspace_name="Engineering",
            description="Sprint",
            board_kind="public",
            state="active",
            item_count=25,
            permissions="everyone",
            owner_id=123,
            url="https://monday.com/boards/100",
        )
        d = board.to_dict()
        assert d["id"] == 100
        assert d["workspace_id"] == 1
        assert d["workspace_name"] == "Engineering"
        assert d["description"] == "Sprint"
        assert d["board_kind"] == "public"
        assert d["state"] == "active"
        assert d["item_count"] == 25
        assert d["permissions"] == "everyone"
        assert d["owner_id"] == 123
        assert d["url"] == "https://monday.com/boards/100"

    def test_from_api_defaults(self):
        data = {"id": "200", "name": "Minimal Board"}
        board = MondayBoard.from_api(data)
        assert board.description == ""
        assert board.board_kind == "public"
        assert board.state == "active"
        assert board.item_count == 0
        assert board.permissions == "everyone"
        assert board.url == ""


# =============================================================================
# MondayColumn Model Tests
# =============================================================================


class TestMondayColumnModel:
    """Test MondayColumn data model."""

    def test_from_api_full(self):
        data = {
            "id": "status_col",
            "title": "Status",
            "type": "status",
            "settings_str": '{"labels":{"0":"Done","1":"Working"}}',
            "archived": False,
            "width": 150,
        }
        col = MondayColumn.from_api(data)
        assert col.id == "status_col"
        assert col.title == "Status"
        assert col.column_type == "status"
        assert col.settings_str == '{"labels":{"0":"Done","1":"Working"}}'
        assert col.archived is False
        assert col.width == 150

    def test_from_api_minimal(self):
        data = {"id": "col1", "title": "Notes"}
        col = MondayColumn.from_api(data)
        assert col.column_type == "text"
        assert col.settings_str == ""
        assert col.archived is False
        assert col.width is None

    def test_to_dict(self):
        col = MondayColumn(
            id="date_col",
            title="Due Date",
            column_type="date",
            settings_str="{}",
            archived=True,
            width=120,
        )
        d = col.to_dict()
        assert d["id"] == "date_col"
        assert d["title"] == "Due Date"
        assert d["column_type"] == "date"
        assert d["archived"] is True
        assert d["width"] == 120


# =============================================================================
# MondayGroup Model Tests
# =============================================================================


class TestMondayGroupModel:
    """Test MondayGroup data model."""

    def test_from_api_full(self):
        data = {
            "id": "grp_done",
            "title": "Done",
            "color": "#00c875",
            "archived": False,
            "deleted": False,
            "position": "3",
        }
        grp = MondayGroup.from_api(data)
        assert grp.id == "grp_done"
        assert grp.title == "Done"
        assert grp.color == "#00c875"
        assert grp.position == "3"
        assert grp.archived is False
        assert grp.deleted is False

    def test_from_api_minimal(self):
        data = {"id": "grp_1", "title": "New"}
        grp = MondayGroup.from_api(data)
        assert grp.color == ""
        assert grp.archived is False
        assert grp.deleted is False
        assert grp.position == ""

    def test_to_dict(self):
        grp = MondayGroup(
            id="grp_x",
            title="In Progress",
            color="#fdab3d",
            archived=True,
            deleted=False,
            position="2",
        )
        d = grp.to_dict()
        assert d["id"] == "grp_x"
        assert d["title"] == "In Progress"
        assert d["color"] == "#fdab3d"
        assert d["archived"] is True
        assert d["deleted"] is False
        assert d["position"] == "2"


# =============================================================================
# MondayColumnValue Model Tests
# =============================================================================


class TestMondayColumnValueModel:
    """Test MondayColumnValue data model."""

    def test_from_api_full(self):
        data = {
            "id": "status",
            "type": "status",
            "text": "Working on it",
            "value": '{"index":1}',
        }
        cv = MondayColumnValue.from_api(data)
        assert cv.id == "status"
        assert cv.column_type == "status"
        assert cv.text == "Working on it"
        assert cv.value == '{"index":1}'

    def test_from_api_minimal(self):
        data = {"id": "notes"}
        cv = MondayColumnValue.from_api(data)
        assert cv.column_type == "text"
        assert cv.text == ""
        assert cv.value is None

    def test_to_dict(self):
        cv = MondayColumnValue(
            id="person", column_type="people", text="Alice", value='{"personsAndTeams":[{"id":1}]}'
        )
        d = cv.to_dict()
        assert d["id"] == "person"
        assert d["column_type"] == "people"
        assert d["text"] == "Alice"
        assert d["value"] == '{"personsAndTeams":[{"id":1}]}'


# =============================================================================
# MondayItem Model Tests
# =============================================================================


class TestMondayItemModel:
    """Test MondayItem data model edge cases."""

    def test_from_api_with_subitems(self):
        data = {
            "id": "1001",
            "name": "Parent Task",
            "board": {"id": "100", "name": "Board A"},
            "group": {"id": "grp1", "title": "Group 1"},
            "creator": {"id": "42"},
            "state": "active",
            "url": "https://monday.com/items/1001",
            "column_values": [],
            "subitems": [
                {
                    "id": "2001",
                    "name": "Sub Task 1",
                    "state": "active",
                    "column_values": [
                        {"id": "status", "type": "status", "text": "Done", "value": '{"index":0}'}
                    ],
                },
                {
                    "id": "2002",
                    "name": "Sub Task 2",
                    "state": "active",
                    "column_values": [],
                },
            ],
        }
        item = MondayItem.from_api(data)
        assert item.id == 1001
        assert len(item.subitems) == 2
        assert item.subitems[0].name == "Sub Task 1"
        assert item.subitems[0].id == 2001
        assert len(item.subitems[0].column_values) == 1

    def test_from_api_with_parent_item(self):
        data = {
            "id": "2001",
            "name": "Sub Task",
            "parent_item": {"id": "1001"},
            "column_values": [],
        }
        item = MondayItem.from_api(data)
        assert item.parent_item_id == 1001

    def test_from_api_no_parent_item(self):
        data = {
            "id": "1001",
            "name": "Regular Task",
            "column_values": [],
        }
        item = MondayItem.from_api(data)
        assert item.parent_item_id is None

    def test_from_api_with_board_id_fallback(self):
        data = {
            "id": "1001",
            "name": "Task",
            "column_values": [],
        }
        item = MondayItem.from_api(data, board_id=999)
        assert item.board_id == 999

    def test_from_api_board_in_data_overrides_fallback(self):
        data = {
            "id": "1001",
            "name": "Task",
            "board": {"id": "100", "name": "Board"},
            "column_values": [],
        }
        item = MondayItem.from_api(data, board_id=999)
        assert item.board_id == 100

    def test_get_column_value_found(self):
        item = MondayItem(
            id=1001,
            name="Task",
            board_id=100,
            column_values=[
                MondayColumnValue(id="status", column_type="status", text="Done"),
                MondayColumnValue(id="person", column_type="people", text="Alice"),
            ],
        )
        assert item.get_column_value("status") == "Done"
        assert item.get_column_value("person") == "Alice"

    def test_get_column_value_not_found(self):
        item = MondayItem(
            id=1001,
            name="Task",
            board_id=100,
            column_values=[
                MondayColumnValue(id="status", column_type="status", text="Done"),
            ],
        )
        assert item.get_column_value("nonexistent") is None

    def test_get_column_value_empty_list(self):
        item = MondayItem(id=1001, name="Task", board_id=100)
        assert item.get_column_value("anything") is None

    def test_to_dict_comprehensive(self):
        item = MondayItem(
            id=1001,
            name="Task",
            board_id=100,
            board_name="Board",
            group_id="grp1",
            group_title="Group 1",
            state="active",
            creator_id=42,
            created_at=datetime(2024, 1, 15, 10, 0, 0),
            updated_at=datetime(2024, 1, 15, 12, 0, 0),
            url="https://monday.com/items/1001",
            column_values=[
                MondayColumnValue(id="status", column_type="status", text="Done"),
            ],
            subitems=[
                MondayItem(id=2001, name="Sub", board_id=100),
            ],
            parent_item_id=None,
        )
        d = item.to_dict()
        assert d["id"] == 1001
        assert d["name"] == "Task"
        assert d["board_id"] == 100
        assert d["board_name"] == "Board"
        assert d["group_id"] == "grp1"
        assert d["group_title"] == "Group 1"
        assert d["state"] == "active"
        assert d["creator_id"] == 42
        assert d["created_at"] == "2024-01-15T10:00:00"
        assert d["updated_at"] == "2024-01-15T12:00:00"
        assert d["url"] == "https://monday.com/items/1001"
        assert len(d["column_values"]) == 1
        assert len(d["subitems"]) == 1
        assert d["parent_item_id"] is None

    def test_to_dict_without_dates(self):
        item = MondayItem(id=1001, name="Task", board_id=100)
        d = item.to_dict()
        assert d["created_at"] is None
        assert d["updated_at"] is None


# =============================================================================
# MondayUpdate Model Tests
# =============================================================================


class TestMondayUpdateModel:
    """Test MondayUpdate data model."""

    def test_from_api_full(self):
        data = {
            "id": "5001",
            "item_id": 1001,
            "body": "<p>Great progress!</p>",
            "text_body": "Great progress!",
            "creator": {"id": "42", "name": "Alice"},
            "created_at": "2024-01-15T10:00:00Z",
            "updated_at": "2024-01-15T10:05:00Z",
        }
        update = MondayUpdate.from_api(data)
        assert update.id == 5001
        assert update.item_id == 1001
        assert update.body == "<p>Great progress!</p>"
        assert update.text_body == "Great progress!"
        assert update.creator_id == 42
        assert update.creator_name == "Alice"
        assert update.created_at is not None

    def test_from_api_minimal(self):
        data = {"id": "5002", "item_id": 0}
        update = MondayUpdate.from_api(data)
        assert update.id == 5002
        assert update.body == ""
        assert update.text_body == ""
        assert update.creator_id is None
        assert update.creator_name == ""

    def test_from_api_no_creator(self):
        data = {"id": "5003", "creator": None}
        update = MondayUpdate.from_api(data)
        assert update.creator_id is None
        assert update.creator_name == ""

    def test_to_dict(self):
        update = MondayUpdate(
            id=5001,
            item_id=1001,
            body="<p>Hello</p>",
            text_body="Hello",
            creator_id=42,
            creator_name="Bob",
            created_at=datetime(2024, 3, 20, 14, 0, 0),
            updated_at=datetime(2024, 3, 20, 14, 5, 0),
        )
        d = update.to_dict()
        assert d["id"] == 5001
        assert d["item_id"] == 1001
        assert d["body"] == "<p>Hello</p>"
        assert d["text_body"] == "Hello"
        assert d["creator_id"] == 42
        assert d["creator_name"] == "Bob"
        assert d["created_at"] == "2024-03-20T14:00:00"
        assert d["updated_at"] == "2024-03-20T14:05:00"

    def test_to_dict_without_dates(self):
        update = MondayUpdate(id=5001, item_id=1001, body="Test")
        d = update.to_dict()
        assert d["created_at"] is None
        assert d["updated_at"] is None


# =============================================================================
# ColumnType and BoardKind Enum Tests
# =============================================================================


class TestEnums:
    """Test enum completeness."""

    def test_all_column_types(self):
        assert ColumnType.LONG_TEXT.value == "long-text"
        assert ColumnType.NUMBERS.value == "numbers"
        assert ColumnType.TIMELINE.value == "timeline"
        assert ColumnType.LINK.value == "link"
        assert ColumnType.EMAIL.value == "email"
        assert ColumnType.PHONE.value == "phone"
        assert ColumnType.DROPDOWN.value == "dropdown"
        assert ColumnType.RATING.value == "rating"
        assert ColumnType.HOUR.value == "hour"
        assert ColumnType.FILE.value == "file"
        assert ColumnType.COLOR_PICKER.value == "color-picker"
        assert ColumnType.TAGS.value == "tags"

    def test_column_type_is_str(self):
        # ColumnType inherits from str
        assert isinstance(ColumnType.TEXT, str)
        assert ColumnType.TEXT == "text"

    def test_board_kind_is_str(self):
        assert isinstance(BoardKind.PUBLIC, str)
        assert BoardKind.PUBLIC == "public"


# =============================================================================
# Connector Properties Tests
# =============================================================================


class TestConnectorProperties:
    """Test connector property accessors."""

    def test_source_type(self, connector):
        assert connector.source_type == SourceType.DOCUMENT

    def test_name(self, connector):
        assert connector.name == "Monday.com"

    def test_is_configured_with_instance_token(self, connector):
        connector._api_token = "abc123"
        assert connector.is_configured is True

    def test_is_configured_with_env_api_token(self, bare_connector):
        with patch.dict("os.environ", {"MONDAY_API_TOKEN": "env_token"}, clear=False):
            assert bare_connector.is_configured is True

    def test_is_configured_with_env_access_token(self, bare_connector):
        with patch.dict("os.environ", {"MONDAY_ACCESS_TOKEN": "access_token"}, clear=False):
            assert bare_connector.is_configured is True

    def test_is_not_configured(self, bare_connector):
        with patch.dict("os.environ", {}, clear=True):
            bare_connector._api_token = None
            assert bare_connector.is_configured is False


# =============================================================================
# _get_client Tests
# =============================================================================


class TestGetClient:
    """Test HTTP client lifecycle."""

    @pytest.mark.asyncio
    async def test_creates_new_client(self, connector):
        connector._client = None
        client = await connector._get_client()
        assert client is not None
        assert isinstance(client, httpx.AsyncClient)
        # Clean up
        await client.aclose()

    @pytest.mark.asyncio
    async def test_reuses_existing_client(self, connector):
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        connector._client = mock_client
        client = await connector._get_client()
        assert client is mock_client

    @pytest.mark.asyncio
    async def test_recreates_closed_client(self, connector):
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.is_closed = True
        connector._client = mock_client
        client = await connector._get_client()
        assert client is not mock_client
        assert isinstance(client, httpx.AsyncClient)
        # Clean up
        await client.aclose()


# =============================================================================
# _get_token Tests
# =============================================================================


class TestGetToken:
    """Test token retrieval priority."""

    @pytest.mark.asyncio
    async def test_instance_token_first(self, connector):
        connector._api_token = "instance_token"
        with patch.dict("os.environ", {"MONDAY_API_TOKEN": "env_token"}):
            token = await connector._get_token()
            assert token == "instance_token"

    @pytest.mark.asyncio
    async def test_env_api_token(self, connector):
        connector._api_token = None
        with patch.dict("os.environ", {"MONDAY_API_TOKEN": "env_api_token"}, clear=True):
            token = await connector._get_token()
            assert token == "env_api_token"

    @pytest.mark.asyncio
    async def test_env_access_token_fallback(self, connector):
        connector._api_token = None
        with patch.dict("os.environ", {"MONDAY_ACCESS_TOKEN": "access_tok"}, clear=True):
            token = await connector._get_token()
            assert token == "access_tok"

    @pytest.mark.asyncio
    async def test_no_token_raises(self, connector):
        connector._api_token = None
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API token not configured"):
                await connector._get_token()


# =============================================================================
# Authentication Tests (Extended)
# =============================================================================


class TestAuthenticationExtended:
    """Test authentication edge cases."""

    @pytest.mark.asyncio
    async def test_authenticate_with_oauth_token(self, connector):
        mock_response = make_graphql_response({"me": {"id": "1", "name": "OAuth User"}})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
        ):
            result = await connector.authenticate(oauth_token="oauth_abc")
            assert result is True
            assert connector._api_token == "oauth_abc"

    @pytest.mark.asyncio
    async def test_authenticate_with_env_token(self, connector):
        mock_response = make_graphql_response({"me": {"id": "1", "name": "Env User"}})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.dict("os.environ", {"MONDAY_API_TOKEN": "env_token_auth"}),
            patch.object(connector, "_get_client", return_value=mock_client),
        ):
            connector._api_token = None
            result = await connector.authenticate()
            assert result is True
            assert connector._api_token == "env_token_auth"

    @pytest.mark.asyncio
    async def test_authenticate_no_token_available(self, connector):
        connector._api_token = None
        with patch.dict("os.environ", {}, clear=True):
            result = await connector.authenticate()
            assert result is False

    @pytest.mark.asyncio
    async def test_authenticate_api_rejection(self, connector):
        """API returns no 'me' field."""
        mock_response = make_graphql_response({"me": None})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch.object(connector, "_get_client", return_value=mock_client):
            result = await connector.authenticate(api_token="bad_token")
            assert result is False

    @pytest.mark.asyncio
    async def test_authenticate_network_error(self, connector):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=OSError("Connection refused"))

        with patch.object(connector, "_get_client", return_value=mock_client):
            result = await connector.authenticate(api_token="token")
            assert result is False

    @pytest.mark.asyncio
    async def test_authenticate_graphql_error(self, connector):
        """GraphQL returns errors."""
        mock_response = make_error_graphql_response([{"message": "Invalid token"}])
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch.object(connector, "_get_client", return_value=mock_client):
            result = await connector.authenticate(api_token="invalid")
            assert result is False


# =============================================================================
# _graphql_request Tests
# =============================================================================


class TestGraphQLRequest:
    """Test GraphQL request construction and error handling."""

    @pytest.mark.asyncio
    async def test_headers_contain_auth_and_api_version(self, connector):
        mock_response = make_graphql_response({"boards": []})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="my_token"),
        ):
            await connector._graphql_request("query { boards { id } }")

            call_args = mock_client.post.call_args
            headers = call_args.kwargs.get("headers") or call_args[1].get("headers")
            assert headers["Authorization"] == "my_token"
            assert headers["Content-Type"] == "application/json"
            assert headers["API-Version"] == "2024-01"

    @pytest.mark.asyncio
    async def test_sends_query_in_payload(self, connector):
        mock_response = make_graphql_response({"test": True})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            await connector._graphql_request("query { me { id } }")

            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert payload["query"] == "query { me { id } }"
            assert "variables" not in payload

    @pytest.mark.asyncio
    async def test_sends_variables_when_provided(self, connector):
        mock_response = make_graphql_response({"result": True})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            variables = {"boardId": 100, "limit": 25}
            await connector._graphql_request(
                "query ($boardId: Int!) { boards(ids: [$boardId]) { id } }", variables=variables
            )

            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert payload["variables"] == variables

    @pytest.mark.asyncio
    async def test_posts_to_monday_api_url(self, connector):
        mock_response = make_graphql_response({})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            await connector._graphql_request("query { me { id } }")

            call_args = mock_client.post.call_args
            url = call_args.args[0] if call_args.args else call_args.kwargs.get("url")
            assert url == MONDAY_API_URL

    @pytest.mark.asyncio
    async def test_raises_on_graphql_errors(self, connector):
        mock_response = make_error_graphql_response(
            [{"message": "Field not found"}, {"message": "Syntax error"}]
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            with pytest.raises(ValueError, match="Field not found"):
                await connector._graphql_request("query { invalid }")

    @pytest.mark.asyncio
    async def test_raises_on_http_error(self, connector):
        mock_response = make_http_error_response(500)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            with pytest.raises(httpx.HTTPStatusError):
                await connector._graphql_request("query { boards { id } }")

    @pytest.mark.asyncio
    async def test_returns_data_field(self, connector):
        mock_response = make_graphql_response({"boards": [{"id": "1"}]})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            result = await connector._graphql_request("query { boards { id } }")
            assert result == {"boards": [{"id": "1"}]}

    @pytest.mark.asyncio
    async def test_returns_empty_dict_when_no_data(self, connector):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {}
        response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            result = await connector._graphql_request("query { me { id } }")
            assert result == {}

    @pytest.mark.asyncio
    async def test_multiple_graphql_errors_concatenated(self, connector):
        mock_response = make_error_graphql_response(
            [{"message": "Error A"}, {"message": "Error B"}, {"message": "Error C"}]
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            with pytest.raises(ValueError, match="Error A; Error B; Error C"):
                await connector._graphql_request("bad query")


# =============================================================================
# list_boards Extended Tests
# =============================================================================


class TestListBoardsExtended:
    """Test list_boards with various filter combinations."""

    @pytest.mark.asyncio
    async def test_list_boards_with_workspace_filter(self, connector):
        mock_response = make_graphql_response(
            {"boards": [{"id": "100", "name": "Board A", "workspace": {"id": "5", "name": "WS"}}]}
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            boards = await connector.list_boards(workspace_id=5)
            assert len(boards) == 1
            # Verify query includes workspace_ids
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "workspace_ids: [5]" in payload["query"]

    @pytest.mark.asyncio
    async def test_list_boards_with_limit(self, connector):
        mock_response = make_graphql_response({"boards": []})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            boards = await connector.list_boards(limit=10)
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "limit: 10" in payload["query"]

    @pytest.mark.asyncio
    async def test_list_boards_with_state_all(self, connector):
        mock_response = make_graphql_response({"boards": []})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            boards = await connector.list_boards(state="all")
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            # state "all" should not add state filter
            assert "state:" not in payload["query"]

    @pytest.mark.asyncio
    async def test_list_boards_with_archived_state(self, connector):
        mock_response = make_graphql_response({"boards": []})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            await connector.list_boards(state="archived")
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "state: archived" in payload["query"]

    @pytest.mark.asyncio
    async def test_list_boards_empty_result(self, connector):
        mock_response = make_graphql_response({"boards": []})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            boards = await connector.list_boards()
            assert boards == []

    @pytest.mark.asyncio
    async def test_list_boards_combined_filters(self, connector):
        mock_response = make_graphql_response({"boards": []})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            await connector.list_boards(workspace_id=3, limit=5, state="active")
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "limit: 5" in payload["query"]
            assert "state: active" in payload["query"]
            assert "workspace_ids: [3]" in payload["query"]


# =============================================================================
# get_board Extended Tests
# =============================================================================


class TestGetBoardExtended:
    """Test get_board edge cases."""

    @pytest.mark.asyncio
    async def test_get_board_not_found(self, connector):
        mock_response = make_graphql_response({"boards": []})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            board = await connector.get_board(999999)
            assert board is None

    @pytest.mark.asyncio
    async def test_get_board_query_contains_id(self, connector):
        mock_response = make_graphql_response({"boards": [{"id": "42", "name": "My Board"}]})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            board = await connector.get_board(42)
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "ids: [42]" in payload["query"]
            assert board.id == 42


# =============================================================================
# get_board_columns Extended Tests
# =============================================================================


class TestGetBoardColumnsExtended:
    """Test get_board_columns edge cases."""

    @pytest.mark.asyncio
    async def test_get_board_columns_empty_boards(self, connector):
        mock_response = make_graphql_response({"boards": []})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            columns = await connector.get_board_columns(999)
            assert columns == []

    @pytest.mark.asyncio
    async def test_get_board_columns_no_columns(self, connector):
        mock_response = make_graphql_response({"boards": [{"columns": []}]})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            columns = await connector.get_board_columns(100)
            assert columns == []

    @pytest.mark.asyncio
    async def test_get_board_columns_with_archived(self, connector):
        mock_response = make_graphql_response(
            {
                "boards": [
                    {
                        "columns": [
                            {
                                "id": "col1",
                                "title": "Active Col",
                                "type": "text",
                                "archived": False,
                            },
                            {
                                "id": "col2",
                                "title": "Archived Col",
                                "type": "text",
                                "archived": True,
                            },
                        ]
                    }
                ]
            }
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            columns = await connector.get_board_columns(100)
            assert len(columns) == 2
            assert columns[0].archived is False
            assert columns[1].archived is True


# =============================================================================
# get_board_groups Extended Tests
# =============================================================================


class TestGetBoardGroupsExtended:
    """Test get_board_groups edge cases."""

    @pytest.mark.asyncio
    async def test_get_board_groups_empty_boards(self, connector):
        mock_response = make_graphql_response({"boards": []})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            groups = await connector.get_board_groups(999)
            assert groups == []

    @pytest.mark.asyncio
    async def test_get_board_groups_with_deleted(self, connector):
        mock_response = make_graphql_response(
            {
                "boards": [
                    {
                        "groups": [
                            {"id": "g1", "title": "Active", "deleted": False},
                            {"id": "g2", "title": "Deleted", "deleted": True},
                        ]
                    }
                ]
            }
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            groups = await connector.get_board_groups(100)
            assert len(groups) == 2
            assert groups[1].deleted is True


# =============================================================================
# list_items Extended Tests
# =============================================================================


class TestListItemsExtended:
    """Test list_items edge cases."""

    @pytest.mark.asyncio
    async def test_list_items_empty_boards(self, connector):
        mock_response = make_graphql_response({"boards": []})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            items, cursor = await connector.list_items(board_id=999)
            assert items == []
            assert cursor is None

    @pytest.mark.asyncio
    async def test_list_items_no_cursor(self, connector):
        mock_response = make_graphql_response(
            {
                "boards": [
                    {
                        "items_page": {
                            "cursor": None,
                            "items": [
                                {"id": "1", "name": "Task", "state": "active", "column_values": []},
                            ],
                        }
                    }
                ]
            }
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            items, cursor = await connector.list_items(board_id=100)
            assert len(items) == 1
            assert cursor is None

    @pytest.mark.asyncio
    async def test_list_items_with_group_filter(self, connector):
        mock_response = make_graphql_response(
            {
                "boards": [
                    {
                        "items_page": {
                            "cursor": None,
                            "items": [
                                {
                                    "id": "1",
                                    "name": "Task A",
                                    "state": "active",
                                    "group": {"id": "grp_a", "title": "Group A"},
                                    "column_values": [],
                                },
                                {
                                    "id": "2",
                                    "name": "Task B",
                                    "state": "active",
                                    "group": {"id": "grp_b", "title": "Group B"},
                                    "column_values": [],
                                },
                                {
                                    "id": "3",
                                    "name": "Task C",
                                    "state": "active",
                                    "group": {"id": "grp_a", "title": "Group A"},
                                    "column_values": [],
                                },
                            ],
                        }
                    }
                ]
            }
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            items, cursor = await connector.list_items(board_id=100, group_id="grp_a")
            assert len(items) == 2
            assert all(item.group_id == "grp_a" for item in items)

    @pytest.mark.asyncio
    async def test_list_items_uses_max_results_default(self, connector):
        """Should use connector's max_results when limit is not provided."""
        mock_response = make_graphql_response(
            {"boards": [{"items_page": {"cursor": None, "items": []}}]}
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            await connector.list_items(board_id=100)
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "limit: 50" in payload["query"]  # connector.max_results = 50

    @pytest.mark.asyncio
    async def test_list_items_with_custom_limit(self, connector):
        mock_response = make_graphql_response(
            {"boards": [{"items_page": {"cursor": None, "items": []}}]}
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            await connector.list_items(board_id=100, limit=10)
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "limit: 10" in payload["query"]

    @pytest.mark.asyncio
    async def test_list_items_with_cursor(self, connector):
        mock_response = make_graphql_response(
            {"boards": [{"items_page": {"cursor": "next_page", "items": []}}]}
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            await connector.list_items(board_id=100, cursor="prev_cursor")
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert 'cursor: "prev_cursor"' in payload["query"]


# =============================================================================
# get_item Extended Tests
# =============================================================================


class TestGetItemExtended:
    """Test get_item edge cases."""

    @pytest.mark.asyncio
    async def test_get_item_not_found(self, connector):
        mock_response = make_graphql_response({"items": []})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            item = await connector.get_item(999999)
            assert item is None

    @pytest.mark.asyncio
    async def test_get_item_with_all_fields(self, connector):
        mock_response = make_graphql_response(
            {
                "items": [
                    {
                        "id": "1001",
                        "name": "Full Task",
                        "state": "active",
                        "url": "https://monday.com/items/1001",
                        "created_at": "2024-01-15T10:00:00Z",
                        "updated_at": "2024-06-01T12:00:00Z",
                        "board": {"id": "100", "name": "Sprint Board"},
                        "group": {"id": "grp1", "title": "In Progress"},
                        "creator": {"id": "42"},
                        "parent_item": {"id": "500"},
                        "column_values": [
                            {
                                "id": "status",
                                "type": "status",
                                "text": "Working",
                                "value": '{"index":1}',
                            },
                        ],
                        "subitems": [
                            {"id": "2001", "name": "Sub 1", "state": "active", "column_values": []},
                        ],
                    }
                ]
            }
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            item = await connector.get_item(1001)
            assert item is not None
            assert item.id == 1001
            assert item.board_id == 100
            assert item.board_name == "Sprint Board"
            assert item.group_id == "grp1"
            assert item.creator_id == 42
            assert item.parent_item_id == 500
            assert len(item.column_values) == 1
            assert len(item.subitems) == 1


# =============================================================================
# create_item Extended Tests
# =============================================================================


class TestCreateItemExtended:
    """Test create_item edge cases."""

    @pytest.mark.asyncio
    async def test_create_item_without_group(self, connector):
        mock_response = make_graphql_response(
            {
                "create_item": {
                    "id": "1010",
                    "name": "Ungrouped Task",
                    "state": "active",
                    "column_values": [],
                }
            }
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            item = await connector.create_item(board_id=100, item_name="Ungrouped Task")
            assert item.name == "Ungrouped Task"
            # Verify no group_id in query
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "group_id" not in payload["query"]

    @pytest.mark.asyncio
    async def test_create_item_without_column_values(self, connector):
        mock_response = make_graphql_response(
            {
                "create_item": {
                    "id": "1011",
                    "name": "Simple Task",
                    "state": "active",
                    "column_values": [],
                }
            }
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            item = await connector.create_item(board_id=100, item_name="Simple Task")
            assert item.id == 1011
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "column_values:" not in payload["query"]

    @pytest.mark.asyncio
    async def test_create_item_with_group_and_columns(self, connector):
        mock_response = make_graphql_response(
            {
                "create_item": {
                    "id": "1012",
                    "name": "Full Task",
                    "state": "active",
                    "group": {"id": "grp_new", "title": "New"},
                    "column_values": [{"id": "status", "text": "Done"}],
                }
            }
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            item = await connector.create_item(
                board_id=100,
                item_name="Full Task",
                group_id="grp_new",
                column_values={"status": {"label": "Done"}},
            )
            assert item.id == 1012
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert 'group_id: "grp_new"' in payload["query"]
            assert "column_values:" in payload["query"]


# =============================================================================
# update_item Extended Tests
# =============================================================================


class TestUpdateItemExtended:
    """Test update_item edge cases."""

    @pytest.mark.asyncio
    async def test_update_item_without_board_id_fetches_item(self, connector):
        """When board_id is not provided, it should fetch the item first."""
        # Mock get_item to return an item with board_id
        mock_item = MondayItem(id=1001, name="Task", board_id=200)

        mock_response = make_graphql_response(
            {
                "change_multiple_column_values": {
                    "id": "1001",
                    "name": "Task",
                    "state": "active",
                    "column_values": [{"id": "status", "text": "Done"}],
                }
            }
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
            patch.object(connector, "get_item", return_value=mock_item),
        ):
            item = await connector.update_item(
                item_id=1001,
                column_values={"status": {"label": "Done"}},
            )
            # Verify get_item was called
            connector.get_item.assert_called_once_with(1001)
            assert item.id == 1001

    @pytest.mark.asyncio
    async def test_update_item_not_found_raises(self, connector):
        """When item is not found and no board_id, should raise ValueError."""
        with patch.object(connector, "get_item", return_value=None):
            with pytest.raises(ValueError, match="Item 9999 not found"):
                await connector.update_item(
                    item_id=9999,
                    column_values={"status": {"label": "Done"}},
                )

    @pytest.mark.asyncio
    async def test_update_item_with_board_id(self, connector):
        mock_response = make_graphql_response(
            {
                "change_multiple_column_values": {
                    "id": "1001",
                    "name": "Updated",
                    "state": "active",
                    "column_values": [],
                }
            }
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            item = await connector.update_item(
                item_id=1001,
                column_values={"status": {"label": "Done"}},
                board_id=100,
            )
            # Verify mutation includes board_id
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "board_id: 100" in payload["query"]


# =============================================================================
# move_item_to_group Tests
# =============================================================================


class TestMoveItemToGroup:
    """Test move_item_to_group operation."""

    @pytest.mark.asyncio
    async def test_move_item_to_group(self, connector):
        mock_response = make_graphql_response(
            {
                "move_item_to_group": {
                    "id": "1001",
                    "name": "Task",
                    "state": "active",
                    "group": {"id": "done_grp", "title": "Done"},
                }
            }
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            item = await connector.move_item_to_group(item_id=1001, group_id="done_grp")
            assert item.group_id == "done_grp"
            assert item.group_title == "Done"
            # Verify mutation
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "move_item_to_group" in payload["query"]
            assert 'group_id: "done_grp"' in payload["query"]

    @pytest.mark.asyncio
    async def test_move_item_query_contains_item_id(self, connector):
        mock_response = make_graphql_response(
            {
                "move_item_to_group": {
                    "id": "555",
                    "name": "Task",
                    "state": "active",
                    "group": {"id": "grp2", "title": "Group 2"},
                }
            }
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            await connector.move_item_to_group(item_id=555, group_id="grp2")
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "item_id: 555" in payload["query"]


# =============================================================================
# archive_item Tests
# =============================================================================


class TestArchiveItem:
    """Test archive_item operation."""

    @pytest.mark.asyncio
    async def test_archive_item_success(self, connector):
        mock_response = make_graphql_response({"archive_item": {"id": "1001"}})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            result = await connector.archive_item(1001)
            assert result is True

    @pytest.mark.asyncio
    async def test_archive_item_not_in_response(self, connector):
        mock_response = make_graphql_response({})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            result = await connector.archive_item(9999)
            assert result is False

    @pytest.mark.asyncio
    async def test_archive_item_query(self, connector):
        mock_response = make_graphql_response({"archive_item": {"id": "42"}})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            await connector.archive_item(42)
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "archive_item(item_id: 42)" in payload["query"]


# =============================================================================
# delete_item Extended Tests
# =============================================================================


class TestDeleteItemExtended:
    """Test delete_item edge cases."""

    @pytest.mark.asyncio
    async def test_delete_item_not_in_response(self, connector):
        mock_response = make_graphql_response({})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            result = await connector.delete_item(9999)
            assert result is False

    @pytest.mark.asyncio
    async def test_delete_item_query(self, connector):
        mock_response = make_graphql_response({"delete_item": {"id": "42"}})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            await connector.delete_item(42)
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "delete_item(item_id: 42)" in payload["query"]


# =============================================================================
# create_subitem Extended Tests
# =============================================================================


class TestCreateSubitemExtended:
    """Test create_subitem edge cases."""

    @pytest.mark.asyncio
    async def test_create_subitem_with_column_values(self, connector):
        mock_response = make_graphql_response(
            {
                "create_subitem": {
                    "id": "2010",
                    "name": "Subtask with cols",
                    "state": "active",
                    "parent_item": {"id": "1001"},
                    "column_values": [
                        {
                            "id": "status",
                            "type": "status",
                            "text": "Working",
                            "value": '{"index":1}',
                        },
                    ],
                }
            }
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            subitem = await connector.create_subitem(
                parent_item_id=1001,
                subitem_name="Subtask with cols",
                column_values={"status": {"label": "Working"}},
            )
            assert subitem.id == 2010
            assert subitem.parent_item_id == 1001
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "column_values:" in payload["query"]

    @pytest.mark.asyncio
    async def test_create_subitem_without_column_values(self, connector):
        mock_response = make_graphql_response(
            {
                "create_subitem": {
                    "id": "2011",
                    "name": "Simple Subtask",
                    "state": "active",
                    "column_values": [],
                }
            }
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            subitem = await connector.create_subitem(
                parent_item_id=1001,
                subitem_name="Simple Subtask",
            )
            assert subitem.id == 2011
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            # No column_values param in mutation
            assert "column_values:" not in payload["query"]

    @pytest.mark.asyncio
    async def test_create_subitem_query_structure(self, connector):
        mock_response = make_graphql_response(
            {
                "create_subitem": {
                    "id": "2012",
                    "name": "Sub",
                    "state": "active",
                    "column_values": [],
                }
            }
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            await connector.create_subitem(parent_item_id=777, subitem_name="Sub")
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "parent_item_id: 777" in payload["query"]
            assert 'item_name: "Sub"' in payload["query"]


# =============================================================================
# list_updates Tests
# =============================================================================


class TestListUpdates:
    """Test list_updates operation."""

    @pytest.mark.asyncio
    async def test_list_updates_success(self, connector):
        mock_response = make_graphql_response(
            {
                "items": [
                    {
                        "updates": [
                            {
                                "id": "5001",
                                "body": "<p>First update</p>",
                                "text_body": "First update",
                                "creator": {"id": "42", "name": "Alice"},
                                "created_at": "2024-01-15T10:00:00Z",
                            },
                            {
                                "id": "5002",
                                "body": "<p>Second update</p>",
                                "text_body": "Second update",
                                "creator": {"id": "43", "name": "Bob"},
                                "created_at": "2024-01-15T11:00:00Z",
                            },
                        ]
                    }
                ]
            }
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            updates = await connector.list_updates(item_id=1001)
            assert len(updates) == 2
            assert updates[0].id == 5001
            assert updates[0].item_id == 1001
            assert updates[0].text_body == "First update"
            assert updates[1].creator_name == "Bob"

    @pytest.mark.asyncio
    async def test_list_updates_empty_items(self, connector):
        mock_response = make_graphql_response({"items": []})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            updates = await connector.list_updates(item_id=9999)
            assert updates == []

    @pytest.mark.asyncio
    async def test_list_updates_custom_limit(self, connector):
        mock_response = make_graphql_response({"items": [{"updates": []}]})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            await connector.list_updates(item_id=1001, limit=5)
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "limit: 5" in payload["query"]

    @pytest.mark.asyncio
    async def test_list_updates_no_updates_key(self, connector):
        mock_response = make_graphql_response({"items": [{}]})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            updates = await connector.list_updates(item_id=1001)
            assert updates == []


# =============================================================================
# create_update Extended Tests
# =============================================================================


class TestCreateUpdateExtended:
    """Test create_update edge cases."""

    @pytest.mark.asyncio
    async def test_create_update_escapes_quotes(self, connector):
        mock_response = make_graphql_response(
            {
                "create_update": {
                    "id": "6001",
                    "body": 'He said "hello"',
                    "created_at": "2024-01-15T10:00:00Z",
                }
            }
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            update = await connector.create_update(item_id=1001, body='He said "hello"')
            assert update.id == 6001
            # Verify quotes are escaped in the query
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert '\\"hello\\"' in payload["query"]

    @pytest.mark.asyncio
    async def test_create_update_escapes_newlines(self, connector):
        mock_response = make_graphql_response(
            {
                "create_update": {
                    "id": "6002",
                    "body": "Line 1\nLine 2",
                    "created_at": "2024-01-15T10:00:00Z",
                }
            }
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            update = await connector.create_update(item_id=1001, body="Line 1\nLine 2")
            assert update.id == 6002
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "\\n" in payload["query"]

    @pytest.mark.asyncio
    async def test_create_update_sets_item_id(self, connector):
        mock_response = make_graphql_response(
            {
                "create_update": {
                    "id": "6003",
                    "body": "A comment",
                    "created_at": "2024-01-15T10:00:00Z",
                }
            }
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            update = await connector.create_update(item_id=1234, body="A comment")
            assert update.item_id == 1234


# =============================================================================
# search_items Extended Tests
# =============================================================================


class TestSearchItemsExtended:
    """Test search_items edge cases."""

    @pytest.mark.asyncio
    async def test_search_items_without_board_ids(self, connector):
        mock_response = make_graphql_response(
            {
                "items_page_by_column_values": {
                    "items": [
                        {"id": "1", "name": "Match", "state": "active", "column_values": []},
                    ]
                }
            }
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            items = await connector.search_items(query_text="Match")
            assert len(items) == 1
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "board_ids" not in payload["query"]

    @pytest.mark.asyncio
    async def test_search_items_with_board_ids(self, connector):
        mock_response = make_graphql_response({"items_page_by_column_values": {"items": []}})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            await connector.search_items(query_text="test", board_ids=[100, 200])
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "board_ids: [100, 200]" in payload["query"]

    @pytest.mark.asyncio
    async def test_search_items_failure_returns_empty(self, connector):
        """search_items catches exceptions and returns empty list."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=ValueError("API error"))

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            items = await connector.search_items(query_text="broken")
            assert items == []

    @pytest.mark.asyncio
    async def test_search_items_empty_result(self, connector):
        mock_response = make_graphql_response({"items_page_by_column_values": {"items": []}})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            items = await connector.search_items(query_text="nothing")
            assert items == []

    @pytest.mark.asyncio
    async def test_search_items_custom_limit(self, connector):
        mock_response = make_graphql_response({"items_page_by_column_values": {"items": []}})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            await connector.search_items(query_text="test", limit=5)
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "limit: 5" in payload["query"]

    @pytest.mark.asyncio
    async def test_search_items_os_error_returns_empty(self, connector):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=OSError("Network error"))

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            items = await connector.search_items(query_text="network fail")
            assert items == []

    @pytest.mark.asyncio
    async def test_search_items_key_error_returns_empty(self, connector):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=KeyError("missing key"))

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            items = await connector.search_items(query_text="key fail")
            assert items == []


# =============================================================================
# sync_items Tests
# =============================================================================


class TestSyncItems:
    """Test sync_items async iterator."""

    @pytest.mark.asyncio
    async def test_sync_items_with_board_ids(self, connector):
        """Should iterate over items from configured board_ids."""
        items_data = {
            "boards": [
                {
                    "items_page": {
                        "cursor": None,
                        "items": [
                            {
                                "id": "1",
                                "name": "Task 1",
                                "state": "active",
                                "board": {"id": "100", "name": "Board A"},
                                "group": {"id": "g1", "title": "G1"},
                                "column_values": [],
                            },
                        ],
                    }
                }
            ]
        }
        mock_response = make_graphql_response(items_data)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            state = SyncState(connector_id="monday")
            sync_items = []
            async for item in connector.sync_items(state):
                sync_items.append(item)

            # connector.board_ids = [100, 200], each returns 1 item
            assert len(sync_items) == 2  # 1 item per board x 2 boards

    @pytest.mark.asyncio
    async def test_sync_items_without_board_ids(self, bare_connector):
        """Should list boards first when no board_ids configured."""
        boards_data = {
            "boards": [
                {"id": "10", "name": "B1", "state": "active", "items_count": 1},
                {"id": "20", "name": "B2", "state": "active", "items_count": 1},
            ]
        }
        items_data = {
            "boards": [
                {
                    "items_page": {
                        "cursor": None,
                        "items": [
                            {
                                "id": "1",
                                "name": "Task",
                                "state": "active",
                                "board": {"id": "10", "name": "B1"},
                                "column_values": [],
                            },
                        ],
                    }
                }
            ]
        }

        mock_response_boards = make_graphql_response(boards_data)
        mock_response_items = make_graphql_response(items_data)
        mock_client = AsyncMock()
        # First call returns boards, subsequent calls return items
        mock_client.post = AsyncMock(
            side_effect=[mock_response_boards, mock_response_items, mock_response_items]
        )

        with (
            patch.object(bare_connector, "_get_client", return_value=mock_client),
            patch.object(bare_connector, "_get_token", return_value="tok"),
        ):
            state = SyncState(connector_id="monday")
            sync_items = []
            async for item in bare_connector.sync_items(state):
                sync_items.append(item)

            assert len(sync_items) == 2

    @pytest.mark.asyncio
    async def test_sync_items_pagination(self, connector):
        """Should follow cursor-based pagination."""
        first_page = make_graphql_response(
            {
                "boards": [
                    {
                        "items_page": {
                            "cursor": "page2_cursor",
                            "items": [
                                {
                                    "id": "1",
                                    "name": "Task 1",
                                    "state": "active",
                                    "board": {"id": "100", "name": "Board"},
                                    "column_values": [],
                                },
                            ],
                        }
                    }
                ]
            }
        )
        second_page = make_graphql_response(
            {
                "boards": [
                    {
                        "items_page": {
                            "cursor": None,
                            "items": [
                                {
                                    "id": "2",
                                    "name": "Task 2",
                                    "state": "active",
                                    "board": {"id": "100", "name": "Board"},
                                    "column_values": [],
                                },
                            ],
                        }
                    }
                ]
            }
        )

        # Board 100: 2 pages, Board 200: empty
        empty_page = make_graphql_response(
            {"boards": [{"items_page": {"cursor": None, "items": []}}]}
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[first_page, second_page, empty_page])

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            state = SyncState(connector_id="monday")
            sync_items = []
            async for item in connector.sync_items(state):
                sync_items.append(item)

            # Board 100 gives 2 items, Board 200 gives 0
            assert len(sync_items) == 2


# =============================================================================
# _item_to_sync_item Tests
# =============================================================================


class TestItemToSyncItem:
    """Test _item_to_sync_item conversion."""

    def test_basic_conversion(self, connector):
        item = MondayItem(
            id=1001,
            name="Test Task",
            board_id=100,
            board_name="Sprint Board",
            group_id="grp1",
            group_title="In Progress",
            state="active",
            creator_id=42,
            url="https://monday.com/items/1001",
            created_at=datetime(2024, 1, 15, 10, 0, 0),
            updated_at=datetime(2024, 1, 15, 12, 0, 0),
            column_values=[
                MondayColumnValue(id="status", column_type="status", text="Working"),
                MondayColumnValue(id="priority", column_type="dropdown", text="High"),
            ],
        )
        sync_item = connector._item_to_sync_item(item)
        assert sync_item.id == "monday-1001"
        assert sync_item.source_id == "monday/100/1001"
        assert sync_item.title == "Test Task"
        assert sync_item.url == "https://monday.com/items/1001"
        assert sync_item.author == "42"
        assert sync_item.domain == "enterprise/monday"
        assert sync_item.confidence == 0.9
        assert sync_item.source_type == "task"
        assert "Task: Test Task" in sync_item.content
        assert "Board: Sprint Board" in sync_item.content
        assert "Group: In Progress" in sync_item.content
        assert "status: Working" in sync_item.content
        assert "priority: High" in sync_item.content

    def test_conversion_without_creator(self, connector):
        item = MondayItem(id=1002, name="No Creator", board_id=100)
        sync_item = connector._item_to_sync_item(item)
        assert sync_item.author == ""

    def test_conversion_empty_column_values(self, connector):
        item = MondayItem(
            id=1003,
            name="Simple",
            board_id=100,
            column_values=[
                MondayColumnValue(id="empty_col", column_type="text", text=""),
            ],
        )
        sync_item = connector._item_to_sync_item(item)
        # Empty text should not appear in content
        assert "empty_col" not in sync_item.content

    def test_metadata_content(self, connector):
        item = MondayItem(
            id=1004,
            name="Meta Task",
            board_id=100,
            board_name="Board",
            group_id="g1",
            group_title="Group 1",
            state="archived",
            column_values=[
                MondayColumnValue(id="status", column_type="status", text="Done"),
                MondayColumnValue(id="empty", column_type="text", text=""),
            ],
        )
        sync_item = connector._item_to_sync_item(item)
        meta = sync_item.metadata
        assert meta["item_id"] == 1004
        assert meta["board_id"] == 100
        assert meta["board_name"] == "Board"
        assert meta["group_id"] == "g1"
        assert meta["group_title"] == "Group 1"
        assert meta["state"] == "archived"
        assert meta["column_values"] == {"status": "Done"}
        assert "empty" not in meta["column_values"]


# =============================================================================
# close() Tests
# =============================================================================


class TestClose:
    """Test connector close/cleanup."""

    @pytest.mark.asyncio
    async def test_close_with_open_client(self, connector):
        mock_client = AsyncMock()
        mock_client.is_closed = False
        connector._client = mock_client
        await connector.close()
        mock_client.aclose.assert_called_once()
        assert connector._client is None

    @pytest.mark.asyncio
    async def test_close_with_already_closed_client(self, connector):
        mock_client = AsyncMock()
        mock_client.is_closed = True
        connector._client = mock_client
        await connector.close()
        mock_client.aclose.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_with_no_client(self, connector):
        connector._client = None
        await connector.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_close_sets_client_to_none(self, connector):
        mock_client = AsyncMock()
        mock_client.is_closed = False
        connector._client = mock_client
        await connector.close()
        assert connector._client is None


# =============================================================================
# Error Handling Extended Tests
# =============================================================================


class TestErrorHandlingExtended:
    """Extended error handling tests."""

    @pytest.mark.asyncio
    async def test_http_403_forbidden(self, connector):
        mock_response = make_http_error_response(403)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            with pytest.raises(httpx.HTTPStatusError):
                await connector._graphql_request("query { boards { id } }")

    @pytest.mark.asyncio
    async def test_http_429_rate_limit(self, connector):
        mock_response = make_http_error_response(429)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            with pytest.raises(httpx.HTTPStatusError):
                await connector._graphql_request("query { boards { id } }")

    @pytest.mark.asyncio
    async def test_http_500_server_error(self, connector):
        mock_response = make_http_error_response(500)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            with pytest.raises(httpx.HTTPStatusError):
                await connector._graphql_request("query { boards { id } }")

    @pytest.mark.asyncio
    async def test_connection_error(self, connector):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            with pytest.raises(httpx.ConnectError):
                await connector._graphql_request("query { me { id } }")

    @pytest.mark.asyncio
    async def test_timeout_error(self, connector):
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            with pytest.raises(httpx.TimeoutException):
                await connector._graphql_request("query { boards { id } }")

    @pytest.mark.asyncio
    async def test_graphql_error_with_no_message(self, connector):
        """Error dict without 'message' key."""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"errors": [{"code": "UNKNOWN"}]}
        response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            with pytest.raises(ValueError, match="GraphQL error"):
                await connector._graphql_request("bad query")


# =============================================================================
# Workspace list_workspaces Extended Tests
# =============================================================================


class TestListWorkspacesExtended:
    """Test list_workspaces edge cases."""

    @pytest.mark.asyncio
    async def test_list_workspaces_empty(self, connector):
        mock_response = make_graphql_response({"workspaces": []})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            workspaces = await connector.list_workspaces()
            assert workspaces == []

    @pytest.mark.asyncio
    async def test_list_workspaces_query_structure(self, connector):
        mock_response = make_graphql_response({"workspaces": []})
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch.object(connector, "_get_client", return_value=mock_client),
            patch.object(connector, "_get_token", return_value="tok"),
        ):
            await connector.list_workspaces()
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "workspaces" in payload["query"]
            assert "id" in payload["query"]
            assert "name" in payload["query"]
            assert "kind" in payload["query"]
            assert "description" in payload["query"]


# =============================================================================
# Integration-style Tests (abstract method implementations)
# =============================================================================


class TestConcreteImplementation:
    """Test the concrete implementation search/fetch methods."""

    @pytest.mark.asyncio
    async def test_search_delegates_to_search_items(self, connector):
        mock_items = [
            MondayItem(id=1, name="Match", board_id=100, group_title="Group A"),
            MondayItem(id=2, name="Also Match", board_id=200, group_title="Group B"),
        ]
        with patch.object(connector, "search_items", return_value=mock_items):
            results = await connector.search("Match")
            assert len(results) == 2
            assert results[0].id == "monday-item-1"
            assert "Match" in results[0].content

    @pytest.mark.asyncio
    async def test_fetch_returns_evidence(self, connector):
        mock_item = MondayItem(id=42, name="Found Item", board_id=100, group_title="G")
        with patch.object(connector, "get_item", return_value=mock_item):
            evidence = await connector.fetch("monday-item-42")
            assert evidence is not None
            assert evidence.id == "monday-item-42"
            assert "Found Item" in evidence.content

    @pytest.mark.asyncio
    async def test_fetch_returns_none_for_missing(self, connector):
        with patch.object(connector, "get_item", return_value=None):
            evidence = await connector.fetch("monday-item-99999")
            assert evidence is None

    @pytest.mark.asyncio
    async def test_fetch_returns_none_for_non_monday_id(self, connector):
        evidence = await connector.fetch("other-system-42")
        assert evidence is None


# =============================================================================
# Connector Initialization Edge Cases
# =============================================================================


class TestConnectorInitEdgeCases:
    """Test connector initialization edge cases."""

    def test_connector_id(self):
        c = ConcreteMondayConnector()
        assert c.connector_id == "monday"

    def test_initial_state(self):
        c = ConcreteMondayConnector()
        assert c._api_token is None
        assert c._credentials is None
        assert c._client is None

    def test_custom_max_results(self):
        c = ConcreteMondayConnector(max_results=500)
        assert c.max_results == 500

    def test_workspace_ids_none(self):
        c = ConcreteMondayConnector(workspace_ids=None)
        assert c.workspace_ids is None

    def test_board_ids_empty_list(self):
        c = ConcreteMondayConnector(board_ids=[])
        assert c.board_ids == []
