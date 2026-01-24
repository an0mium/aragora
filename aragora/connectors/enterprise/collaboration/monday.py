"""
Monday.com Enterprise Connector.

Provides full integration with Monday.com for project and task management:
- Board and workspace traversal
- Item (task) CRUD operations
- Columns and column values
- Groups (sections)
- Subitems
- Updates (comments)
- Webhook support for real-time updates

Requires Monday.com API token or OAuth.
Monday.com uses GraphQL API v2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from aragora.connectors.enterprise.base import (
    EnterpriseConnector,
    SyncItem,
    SyncState,
)
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)


# Monday.com API constants
MONDAY_API_URL = "https://api.monday.com/v2"
MONDAY_AUTH_URL = "https://auth.monday.com/oauth2/authorize"
MONDAY_TOKEN_URL = "https://auth.monday.com/oauth2/token"


class ColumnType(str, Enum):
    """Monday.com column types."""

    TEXT = "text"
    LONG_TEXT = "long-text"
    NUMBERS = "numbers"
    STATUS = "status"
    DATE = "date"
    TIMELINE = "timeline"
    PERSON = "people"
    CHECKBOX = "checkbox"
    LINK = "link"
    EMAIL = "email"
    PHONE = "phone"
    DROPDOWN = "dropdown"
    RATING = "rating"
    HOUR = "hour"
    FILE = "file"
    COLOR_PICKER = "color-picker"
    TAGS = "tags"


class BoardKind(str, Enum):
    """Monday.com board kinds."""

    PUBLIC = "public"
    PRIVATE = "private"
    SHARE = "share"


@dataclass
class MondayCredentials:
    """Monday.com API credentials."""

    api_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_at: Optional[datetime] = None


@dataclass
class MondayWorkspace:
    """A Monday.com workspace."""

    id: int
    name: str
    kind: str = "open"  # open, closed
    description: str = ""

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> MondayWorkspace:
        """Create from API response."""
        return cls(
            id=int(data["id"]),
            name=data["name"],
            kind=data.get("kind", "open"),
            description=data.get("description", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
        }


@dataclass
class MondayBoard:
    """A Monday.com board."""

    id: int
    name: str
    workspace_id: Optional[int] = None
    workspace_name: str = ""
    description: str = ""
    board_kind: str = "public"  # public, private, share
    state: str = "active"  # active, archived, deleted, all
    item_count: int = 0
    permissions: str = "everyone"
    owner_id: Optional[int] = None
    url: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> MondayBoard:
        """Create from API response."""
        workspace = data.get("workspace") or {}
        owner = data.get("owner") or {}

        return cls(
            id=int(data["id"]),
            name=data["name"],
            workspace_id=int(workspace["id"]) if workspace.get("id") else None,
            workspace_name=workspace.get("name", ""),
            description=data.get("description", ""),
            board_kind=data.get("board_kind", "public"),
            state=data.get("state", "active"),
            item_count=data.get("items_count", 0),
            permissions=data.get("permissions", "everyone"),
            owner_id=int(owner["id"]) if owner.get("id") else None,
            url=data.get("url", ""),
            created_at=_parse_datetime(data.get("created_at")),
            updated_at=_parse_datetime(data.get("updated_at")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "workspace_id": self.workspace_id,
            "workspace_name": self.workspace_name,
            "description": self.description,
            "board_kind": self.board_kind,
            "state": self.state,
            "item_count": self.item_count,
            "permissions": self.permissions,
            "owner_id": self.owner_id,
            "url": self.url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass
class MondayColumn:
    """A Monday.com board column."""

    id: str
    title: str
    column_type: str
    settings_str: str = ""
    archived: bool = False
    width: Optional[int] = None

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> MondayColumn:
        """Create from API response."""
        return cls(
            id=data["id"],
            title=data["title"],
            column_type=data.get("type", "text"),
            settings_str=data.get("settings_str", ""),
            archived=data.get("archived", False),
            width=data.get("width"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "column_type": self.column_type,
            "settings_str": self.settings_str,
            "archived": self.archived,
            "width": self.width,
        }


@dataclass
class MondayGroup:
    """A Monday.com board group (section)."""

    id: str
    title: str
    color: str = ""
    archived: bool = False
    deleted: bool = False
    position: str = ""

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> MondayGroup:
        """Create from API response."""
        return cls(
            id=data["id"],
            title=data["title"],
            color=data.get("color", ""),
            archived=data.get("archived", False),
            deleted=data.get("deleted", False),
            position=data.get("position", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "color": self.color,
            "archived": self.archived,
            "deleted": self.deleted,
            "position": self.position,
        }


@dataclass
class MondayColumnValue:
    """A column value for an item."""

    id: str
    column_type: str
    text: str = ""
    value: Optional[str] = None  # JSON string

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> MondayColumnValue:
        """Create from API response."""
        return cls(
            id=data["id"],
            column_type=data.get("type", "text"),
            text=data.get("text", ""),
            value=data.get("value"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "column_type": self.column_type,
            "text": self.text,
            "value": self.value,
        }


@dataclass
class MondayItem:
    """A Monday.com item (task)."""

    id: int
    name: str
    board_id: int
    board_name: str = ""
    group_id: str = ""
    group_title: str = ""
    state: str = "active"  # active, archived, deleted
    creator_id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    url: str = ""
    column_values: List[MondayColumnValue] = field(default_factory=list)
    subitems: List[MondayItem] = field(default_factory=list)
    parent_item_id: Optional[int] = None

    @classmethod
    def from_api(cls, data: Dict[str, Any], board_id: int = 0) -> MondayItem:
        """Create from API response."""
        board = data.get("board") or {}
        group = data.get("group") or {}
        creator = data.get("creator") or {}
        parent = data.get("parent_item") or {}

        column_values = [MondayColumnValue.from_api(cv) for cv in data.get("column_values", [])]

        subitems = [cls.from_api(si, board_id=board_id) for si in data.get("subitems", [])]

        return cls(
            id=int(data["id"]),
            name=data["name"],
            board_id=int(board.get("id", board_id)),
            board_name=board.get("name", ""),
            group_id=group.get("id", ""),
            group_title=group.get("title", ""),
            state=data.get("state", "active"),
            creator_id=int(creator["id"]) if creator.get("id") else None,
            created_at=_parse_datetime(data.get("created_at")),
            updated_at=_parse_datetime(data.get("updated_at")),
            url=data.get("url", ""),
            column_values=column_values,
            subitems=subitems,
            parent_item_id=int(parent["id"]) if parent.get("id") else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "board_id": self.board_id,
            "board_name": self.board_name,
            "group_id": self.group_id,
            "group_title": self.group_title,
            "state": self.state,
            "creator_id": self.creator_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "url": self.url,
            "column_values": [cv.to_dict() for cv in self.column_values],
            "subitems": [si.to_dict() for si in self.subitems],
            "parent_item_id": self.parent_item_id,
        }

    def get_column_value(self, column_id: str) -> Optional[str]:
        """Get text value for a column."""
        for cv in self.column_values:
            if cv.id == column_id:
                return cv.text
        return None


@dataclass
class MondayUpdate:
    """A Monday.com update (comment)."""

    id: int
    item_id: int
    body: str
    text_body: str = ""
    creator_id: Optional[int] = None
    creator_name: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> MondayUpdate:
        """Create from API response."""
        creator = data.get("creator") or {}
        return cls(
            id=int(data["id"]),
            item_id=int(data.get("item_id", 0)),
            body=data.get("body", ""),
            text_body=data.get("text_body", ""),
            creator_id=int(creator["id"]) if creator.get("id") else None,
            creator_name=creator.get("name", ""),
            created_at=_parse_datetime(data.get("created_at")),
            updated_at=_parse_datetime(data.get("updated_at")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "item_id": self.item_id,
            "body": self.body,
            "text_body": self.text_body,
            "creator_id": self.creator_id,
            "creator_name": self.creator_name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


def _parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """Parse Monday.com datetime string."""
    if not dt_str:
        return None
    try:
        # Monday.com uses ISO format
        if dt_str.endswith("Z"):
            dt_str = dt_str.replace("Z", "+00:00")
        return datetime.fromisoformat(dt_str)
    except (ValueError, TypeError):
        return None


class MondayConnector(EnterpriseConnector):
    """
    Enterprise connector for Monday.com.

    Features:
    - GraphQL API integration
    - Board and item management
    - Column value updates
    - Groups (sections)
    - Subitems
    - Updates (comments)
    - Search across boards

    Authentication:
    - API Token (Personal or OAuth)

    Usage:
        connector = MondayConnector()

        # Authenticate with API token
        await connector.authenticate(api_token="your_token")

        # List boards
        boards = await connector.list_boards()

        # Get items from a board
        items = await connector.list_items(board_id=123456)

        # Create an item
        item = await connector.create_item(
            board_id=123456,
            group_id="new_group",
            item_name="New Task",
            column_values={"status": "Working on it"}
        )
    """

    def __init__(
        self,
        workspace_ids: Optional[List[int]] = None,
        board_ids: Optional[List[int]] = None,
        max_results: int = 100,
        **kwargs,
    ):
        """
        Initialize Monday.com connector.

        Args:
            workspace_ids: Filter to specific workspaces
            board_ids: Filter to specific boards
            max_results: Max items per request
        """
        super().__init__(connector_id="monday", **kwargs)

        self.workspace_ids = workspace_ids
        self.board_ids = board_ids
        self.max_results = max_results

        # Credentials
        self._api_token: Optional[str] = None
        self._credentials: Optional[MondayCredentials] = None

        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def source_type(self) -> SourceType:
        return SourceType.DOCUMENT

    @property
    def name(self) -> str:
        return "Monday.com"

    @property
    def is_configured(self) -> bool:
        """Check if connector has required configuration."""
        import os

        return bool(
            os.environ.get("MONDAY_API_TOKEN")
            or os.environ.get("MONDAY_ACCESS_TOKEN")
            or self._api_token
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=60)
        return self._client

    async def _get_token(self) -> str:
        """Get API token."""
        import os

        if self._api_token:
            return self._api_token

        token = os.environ.get("MONDAY_API_TOKEN") or os.environ.get("MONDAY_ACCESS_TOKEN") or ""

        if not token:
            raise ValueError("Monday.com API token not configured")

        return token

    async def authenticate(
        self,
        api_token: Optional[str] = None,
        oauth_token: Optional[str] = None,
    ) -> bool:
        """
        Authenticate with Monday.com.

        Args:
            api_token: Personal API token
            oauth_token: OAuth access token

        Returns:
            True if authentication successful
        """
        token = api_token or oauth_token

        if not token:
            # Try environment variable
            import os

            token = os.environ.get("MONDAY_API_TOKEN") or os.environ.get("MONDAY_ACCESS_TOKEN")

        if not token:
            logger.error("No Monday.com API token provided")
            return False

        self._api_token = token

        # Verify token works
        try:
            result = await self._graphql_request("query { me { id name } }")
            if result.get("me"):
                logger.info(f"Monday.com authenticated as: {result['me'].get('name')}")
                return True
        except Exception as e:
            logger.error(f"Monday.com authentication failed: {e}")

        return False

    async def _graphql_request(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a GraphQL request."""
        token = await self._get_token()
        client = await self._get_client()

        headers = {
            "Authorization": token,
            "Content-Type": "application/json",
            "API-Version": "2024-01",
        }

        payload: Dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables

        response = await client.post(
            MONDAY_API_URL,
            json=payload,
            headers=headers,
        )

        response.raise_for_status()
        result = response.json()

        if "errors" in result:
            errors = result["errors"]
            error_msg = "; ".join(e.get("message", str(e)) for e in errors)
            raise ValueError(f"Monday.com GraphQL error: {error_msg}")

        return result.get("data", {})

    # =========================================================================
    # Workspace Operations
    # =========================================================================

    async def list_workspaces(self) -> List[MondayWorkspace]:
        """List all accessible workspaces."""
        query = """
        query {
            workspaces {
                id
                name
                kind
                description
            }
        }
        """

        result = await self._graphql_request(query)
        workspaces = [MondayWorkspace.from_api(ws) for ws in result.get("workspaces", [])]

        return workspaces

    # =========================================================================
    # Board Operations
    # =========================================================================

    async def list_boards(
        self,
        workspace_id: Optional[int] = None,
        limit: Optional[int] = None,
        state: str = "active",
    ) -> List[MondayBoard]:
        """
        List boards.

        Args:
            workspace_id: Filter by workspace
            limit: Maximum boards to return
            state: Board state filter (active, archived, deleted, all)
        """
        args = []
        if limit:
            args.append(f"limit: {limit}")
        if state != "all":
            args.append(f"state: {state}")
        if workspace_id:
            args.append(f"workspace_ids: [{workspace_id}]")

        args_str = f"({', '.join(args)})" if args else ""

        query = f"""
        query {{
            boards{args_str} {{
                id
                name
                description
                board_kind
                state
                items_count
                permissions
                url
                created_at
                updated_at
                workspace {{
                    id
                    name
                }}
                owner {{
                    id
                }}
            }}
        }}
        """

        result = await self._graphql_request(query)
        boards = [MondayBoard.from_api(b) for b in result.get("boards", [])]

        return boards

    async def get_board(self, board_id: int) -> Optional[MondayBoard]:
        """Get a single board by ID."""
        query = f"""
        query {{
            boards(ids: [{board_id}]) {{
                id
                name
                description
                board_kind
                state
                items_count
                permissions
                url
                created_at
                updated_at
                workspace {{
                    id
                    name
                }}
                owner {{
                    id
                }}
            }}
        }}
        """

        result = await self._graphql_request(query)
        boards = result.get("boards", [])

        if boards:
            return MondayBoard.from_api(boards[0])
        return None

    async def get_board_columns(self, board_id: int) -> List[MondayColumn]:
        """Get columns for a board."""
        query = f"""
        query {{
            boards(ids: [{board_id}]) {{
                columns {{
                    id
                    title
                    type
                    settings_str
                    archived
                    width
                }}
            }}
        }}
        """

        result = await self._graphql_request(query)
        boards = result.get("boards", [])

        if not boards:
            return []

        columns = [MondayColumn.from_api(col) for col in boards[0].get("columns", [])]

        return columns

    async def get_board_groups(self, board_id: int) -> List[MondayGroup]:
        """Get groups (sections) for a board."""
        query = f"""
        query {{
            boards(ids: [{board_id}]) {{
                groups {{
                    id
                    title
                    color
                    archived
                    deleted
                    position
                }}
            }}
        }}
        """

        result = await self._graphql_request(query)
        boards = result.get("boards", [])

        if not boards:
            return []

        groups = [MondayGroup.from_api(grp) for grp in boards[0].get("groups", [])]

        return groups

    # =========================================================================
    # Item Operations
    # =========================================================================

    async def list_items(
        self,
        board_id: int,
        group_id: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> tuple[List[MondayItem], Optional[str]]:
        """
        List items from a board.

        Returns:
            Tuple of (items, next_cursor)
        """
        limit = limit or self.max_results

        query = f"""
        query {{
            boards(ids: [{board_id}]) {{
                items_page(limit: {limit}{f', cursor: "{cursor}"' if cursor else ''}) {{
                    cursor
                    items {{
                        id
                        name
                        state
                        url
                        created_at
                        updated_at
                        board {{
                            id
                            name
                        }}
                        group {{
                            id
                            title
                        }}
                        creator {{
                            id
                        }}
                        column_values {{
                            id
                            type
                            text
                            value
                        }}
                        subitems {{
                            id
                            name
                            state
                            created_at
                            updated_at
                            column_values {{
                                id
                                type
                                text
                                value
                            }}
                        }}
                    }}
                }}
            }}
        }}
        """

        result = await self._graphql_request(query)
        boards = result.get("boards", [])

        if not boards:
            return [], None

        items_page = boards[0].get("items_page", {})
        items = [
            MondayItem.from_api(item, board_id=board_id) for item in items_page.get("items", [])
        ]

        # Filter by group if specified
        if group_id:
            items = [item for item in items if item.group_id == group_id]

        next_cursor = items_page.get("cursor")

        return items, next_cursor

    async def get_item(self, item_id: int) -> Optional[MondayItem]:
        """Get a single item by ID."""
        query = f"""
        query {{
            items(ids: [{item_id}]) {{
                id
                name
                state
                url
                created_at
                updated_at
                board {{
                    id
                    name
                }}
                group {{
                    id
                    title
                }}
                creator {{
                    id
                }}
                parent_item {{
                    id
                }}
                column_values {{
                    id
                    type
                    text
                    value
                }}
                subitems {{
                    id
                    name
                    state
                    created_at
                    updated_at
                    column_values {{
                        id
                        type
                        text
                        value
                    }}
                }}
            }}
        }}
        """

        result = await self._graphql_request(query)
        items = result.get("items", [])

        if items:
            return MondayItem.from_api(items[0])
        return None

    async def create_item(
        self,
        board_id: int,
        item_name: str,
        group_id: Optional[str] = None,
        column_values: Optional[Dict[str, Any]] = None,
    ) -> MondayItem:
        """
        Create a new item.

        Args:
            board_id: Board to create item in
            item_name: Name of the item
            group_id: Group to add item to (optional)
            column_values: Initial column values as dict
        """
        import json

        column_values_str = ""
        if column_values:
            column_values_str = f", column_values: {json.dumps(json.dumps(column_values))}"

        group_str = f', group_id: "{group_id}"' if group_id else ""

        query = f"""
        mutation {{
            create_item(
                board_id: {board_id},
                item_name: "{item_name}"{group_str}{column_values_str}
            ) {{
                id
                name
                state
                url
                created_at
                updated_at
                board {{
                    id
                    name
                }}
                group {{
                    id
                    title
                }}
                column_values {{
                    id
                    type
                    text
                    value
                }}
            }}
        }}
        """

        result = await self._graphql_request(query)
        item_data = result.get("create_item", {})

        return MondayItem.from_api(item_data, board_id=board_id)

    async def update_item(
        self,
        item_id: int,
        column_values: Dict[str, Any],
        board_id: Optional[int] = None,
    ) -> MondayItem:
        """
        Update an item's column values.

        Args:
            item_id: Item to update
            column_values: Column values to update
            board_id: Board ID (required for update)
        """
        import json

        if not board_id:
            # Fetch item to get board_id
            item = await self.get_item(item_id)
            if not item:
                raise ValueError(f"Item {item_id} not found")
            board_id = item.board_id

        column_values_str = json.dumps(json.dumps(column_values))

        query = f"""
        mutation {{
            change_multiple_column_values(
                item_id: {item_id},
                board_id: {board_id},
                column_values: {column_values_str}
            ) {{
                id
                name
                state
                url
                updated_at
                board {{
                    id
                    name
                }}
                group {{
                    id
                    title
                }}
                column_values {{
                    id
                    type
                    text
                    value
                }}
            }}
        }}
        """

        result = await self._graphql_request(query)
        item_data = result.get("change_multiple_column_values", {})

        return MondayItem.from_api(item_data, board_id=board_id)

    async def move_item_to_group(
        self,
        item_id: int,
        group_id: str,
    ) -> MondayItem:
        """Move an item to a different group."""
        query = f"""
        mutation {{
            move_item_to_group(
                item_id: {item_id},
                group_id: "{group_id}"
            ) {{
                id
                name
                state
                group {{
                    id
                    title
                }}
            }}
        }}
        """

        result = await self._graphql_request(query)
        item_data = result.get("move_item_to_group", {})

        return MondayItem.from_api(item_data)

    async def archive_item(self, item_id: int) -> bool:
        """Archive an item."""
        query = f"""
        mutation {{
            archive_item(item_id: {item_id}) {{
                id
            }}
        }}
        """

        result = await self._graphql_request(query)
        return "archive_item" in result

    async def delete_item(self, item_id: int) -> bool:
        """Delete an item."""
        query = f"""
        mutation {{
            delete_item(item_id: {item_id}) {{
                id
            }}
        }}
        """

        result = await self._graphql_request(query)
        return "delete_item" in result

    # =========================================================================
    # Subitem Operations
    # =========================================================================

    async def create_subitem(
        self,
        parent_item_id: int,
        subitem_name: str,
        column_values: Optional[Dict[str, Any]] = None,
    ) -> MondayItem:
        """Create a subitem under a parent item."""
        import json

        column_values_str = ""
        if column_values:
            column_values_str = f", column_values: {json.dumps(json.dumps(column_values))}"

        query = f"""
        mutation {{
            create_subitem(
                parent_item_id: {parent_item_id},
                item_name: "{subitem_name}"{column_values_str}
            ) {{
                id
                name
                state
                created_at
                updated_at
                board {{
                    id
                    name
                }}
                parent_item {{
                    id
                }}
                column_values {{
                    id
                    type
                    text
                    value
                }}
            }}
        }}
        """

        result = await self._graphql_request(query)
        subitem_data = result.get("create_subitem", {})

        return MondayItem.from_api(subitem_data)

    # =========================================================================
    # Update (Comment) Operations
    # =========================================================================

    async def list_updates(
        self,
        item_id: int,
        limit: int = 25,
    ) -> List[MondayUpdate]:
        """Get updates (comments) for an item."""
        query = f"""
        query {{
            items(ids: [{item_id}]) {{
                updates(limit: {limit}) {{
                    id
                    body
                    text_body
                    created_at
                    updated_at
                    creator {{
                        id
                        name
                    }}
                }}
            }}
        }}
        """

        result = await self._graphql_request(query)
        items = result.get("items", [])

        if not items:
            return []

        updates = [
            MondayUpdate.from_api({**upd, "item_id": item_id})
            for upd in items[0].get("updates", [])
        ]

        return updates

    async def create_update(
        self,
        item_id: int,
        body: str,
    ) -> MondayUpdate:
        """Create an update (comment) on an item."""
        # Escape quotes in body
        escaped_body = body.replace('"', '\\"').replace("\n", "\\n")

        query = f"""
        mutation {{
            create_update(
                item_id: {item_id},
                body: "{escaped_body}"
            ) {{
                id
                body
                text_body
                created_at
                updated_at
                creator {{
                    id
                    name
                }}
            }}
        }}
        """

        result = await self._graphql_request(query)
        update_data = result.get("create_update", {})
        update_data["item_id"] = item_id

        return MondayUpdate.from_api(update_data)

    # =========================================================================
    # Search
    # =========================================================================

    async def search_items(
        self,
        query_text: str,
        board_ids: Optional[List[int]] = None,
        limit: int = 25,
    ) -> List[MondayItem]:
        """
        Search for items across boards.

        Args:
            query_text: Search query
            board_ids: Boards to search in (optional)
        """
        board_filter = ""
        if board_ids:
            board_filter = f", board_ids: [{', '.join(map(str, board_ids))}]"

        query = f"""
        query {{
            items_page_by_column_values(
                limit: {limit}{board_filter},
                columns: [{{column_id: "name", column_values: ["{query_text}"]}}]
            ) {{
                items {{
                    id
                    name
                    state
                    url
                    created_at
                    board {{
                        id
                        name
                    }}
                    group {{
                        id
                        title
                    }}
                    column_values {{
                        id
                        type
                        text
                        value
                    }}
                }}
            }}
        }}
        """

        try:
            result = await self._graphql_request(query)
            items_page = result.get("items_page_by_column_values", {})
            items = [MondayItem.from_api(item) for item in items_page.get("items", [])]
            return items
        except Exception as e:
            logger.warning(f"Monday.com search failed: {e}")
            return []

    # =========================================================================
    # Sync Implementation
    # =========================================================================

    async def sync_items(
        self,
        state: SyncState,
        batch_size: int = 100,
    ) -> AsyncIterator[SyncItem]:
        """Yield Monday.com items for syncing."""
        board_ids = self.board_ids or []

        # If no specific boards, get all boards
        if not board_ids:
            boards = await self.list_boards()
            board_ids = [b.id for b in boards]

        for board_id in board_ids:
            cursor: Optional[str] = None

            while True:
                items, cursor = await self.list_items(
                    board_id=board_id,
                    limit=batch_size,
                    cursor=cursor,
                )

                for item in items:
                    yield self._item_to_sync_item(item)

                if not cursor:
                    break

    def _item_to_sync_item(self, item: MondayItem) -> SyncItem:
        """Convert MondayItem to SyncItem."""
        # Build content from item details
        content_parts = [
            f"Task: {item.name}",
            f"Board: {item.board_name}",
            f"Group: {item.group_title}",
        ]

        # Add column values
        for cv in item.column_values:
            if cv.text:
                content_parts.append(f"{cv.id}: {cv.text}")

        return SyncItem(
            id=f"monday-{item.id}",
            content="\n".join(content_parts),
            source_type="task",
            source_id=f"monday/{item.board_id}/{item.id}",
            title=item.name,
            url=item.url,
            author=str(item.creator_id) if item.creator_id else "",
            created_at=item.created_at,
            updated_at=item.updated_at,
            domain="enterprise/monday",
            confidence=0.9,
            metadata={
                "item_id": item.id,
                "board_id": item.board_id,
                "board_name": item.board_name,
                "group_id": item.group_id,
                "group_title": item.group_title,
                "state": item.state,
                "column_values": {cv.id: cv.text for cv in item.column_values if cv.text},
            },
        )

    async def close(self) -> None:
        """Close the connector."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


__all__ = [
    "MondayConnector",
    "MondayBoard",
    "MondayItem",
    "MondayColumn",
    "MondayGroup",
    "MondayUpdate",
    "MondayWorkspace",
    "MondayCredentials",
    "MondayColumnValue",
    "ColumnType",
    "BoardKind",
]
