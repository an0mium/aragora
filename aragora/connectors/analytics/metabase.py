"""
Metabase Connector.

Integration with Metabase BI platform:
- Questions (saved queries)
- Dashboards
- Collections
- Cards (visualizations)
- Databases and tables
- Query execution

Requires Metabase session token or API key.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class DisplayType(str, Enum):
    """Visualization display types."""

    TABLE = "table"
    SCALAR = "scalar"
    SMARTSCALAR = "smartscalar"
    PROGRESS = "progress"
    GAUGE = "gauge"
    LINE = "line"
    AREA = "area"
    BAR = "bar"
    COMBO = "combo"
    ROW = "row"
    SCATTER = "scatter"
    WATERFALL = "waterfall"
    PIE = "pie"
    FUNNEL = "funnel"
    MAP = "map"
    PIVOT = "pivot"


class CollectionType(str, Enum):
    """Collection namespace types."""

    ROOT = "root"
    PERSONAL = "personal"
    REGULAR = "regular"


@dataclass
class MetabaseCredentials:
    """Metabase API credentials."""

    base_url: str  # e.g., "https://metabase.example.com"
    session_token: str | None = None
    api_key: str | None = None
    username: str | None = None
    password: str | None = None


@dataclass
class Database:
    """Metabase database connection."""

    id: int
    name: str
    engine: str
    description: str | None = None
    is_sample: bool = False
    is_full_sync: bool = True
    tables: list[Table] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Database:
        """Create from API response."""
        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            engine=data.get("engine", ""),
            description=data.get("description"),
            is_sample=data.get("is_sample", False),
            is_full_sync=data.get("is_full_sync", True),
            tables=[Table.from_api(t) for t in data.get("tables", [])],
            created_at=_parse_datetime(data.get("created_at")),
            updated_at=_parse_datetime(data.get("updated_at")),
        )


@dataclass
class Table:
    """Database table."""

    id: int
    name: str
    display_name: str
    schema: str | None = None
    db_id: int | None = None
    description: str | None = None
    entity_type: str | None = None
    visibility_type: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Table:
        """Create from API response."""
        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            display_name=data.get("display_name", data.get("name", "")),
            schema=data.get("schema"),
            db_id=data.get("db_id"),
            description=data.get("description"),
            entity_type=data.get("entity_type"),
            visibility_type=data.get("visibility_type"),
            created_at=_parse_datetime(data.get("created_at")),
            updated_at=_parse_datetime(data.get("updated_at")),
        )


@dataclass
class Collection:
    """Metabase collection (folder)."""

    id: int | str
    name: str
    description: str | None = None
    slug: str | None = None
    color: str | None = None
    location: str | None = None
    namespace: str | None = None
    personal_owner_id: int | None = None
    archived: bool = False

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Collection:
        """Create from API response."""
        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            description=data.get("description"),
            slug=data.get("slug"),
            color=data.get("color"),
            location=data.get("location"),
            namespace=data.get("namespace"),
            personal_owner_id=data.get("personal_owner_id"),
            archived=data.get("archived", False),
        )


@dataclass
class Card:
    """Metabase card (saved question)."""

    id: int
    name: str
    description: str | None = None
    display: DisplayType = DisplayType.TABLE
    database_id: int | None = None
    table_id: int | None = None
    collection_id: int | None = None
    query_type: str = "query"
    dataset_query: dict[str, Any] = field(default_factory=dict)
    visualization_settings: dict[str, Any] = field(default_factory=dict)
    result_metadata: list[dict[str, Any]] = field(default_factory=list)
    archived: bool = False
    enable_embedding: bool = False
    creator_id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Card:
        """Create from API response."""
        display = data.get("display", "table")
        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            description=data.get("description"),
            display=DisplayType(display)
            if display in DisplayType.__members__.values()
            else DisplayType.TABLE,
            database_id=data.get("database_id"),
            table_id=data.get("table_id"),
            collection_id=data.get("collection_id"),
            query_type=data.get("query_type", "query"),
            dataset_query=data.get("dataset_query", {}),
            visualization_settings=data.get("visualization_settings", {}),
            result_metadata=data.get("result_metadata", []),
            archived=data.get("archived", False),
            enable_embedding=data.get("enable_embedding", False),
            creator_id=data.get("creator_id"),
            created_at=_parse_datetime(data.get("created_at")),
            updated_at=_parse_datetime(data.get("updated_at")),
        )


@dataclass
class Dashboard:
    """Metabase dashboard."""

    id: int
    name: str
    description: str | None = None
    collection_id: int | None = None
    parameters: list[dict[str, Any]] = field(default_factory=list)
    dashcards: list[DashCard] = field(default_factory=list)
    archived: bool = False
    enable_embedding: bool = False
    creator_id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> Dashboard:
        """Create from API response."""
        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            description=data.get("description"),
            collection_id=data.get("collection_id"),
            parameters=data.get("parameters", []),
            dashcards=[
                DashCard.from_api(dc) for dc in data.get("dashcards", data.get("ordered_cards", []))
            ],
            archived=data.get("archived", False),
            enable_embedding=data.get("enable_embedding", False),
            creator_id=data.get("creator_id"),
            created_at=_parse_datetime(data.get("created_at")),
            updated_at=_parse_datetime(data.get("updated_at")),
        )


@dataclass
class DashCard:
    """Dashboard card (positioned card on dashboard)."""

    id: int
    card_id: int | None = None
    card: Card | None = None
    row: int = 0
    col: int = 0
    size_x: int = 4
    size_y: int = 4
    parameter_mappings: list[dict[str, Any]] = field(default_factory=list)
    visualization_settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> DashCard:
        """Create from API response."""
        card_data = data.get("card")
        return cls(
            id=data.get("id", 0),
            card_id=data.get("card_id"),
            card=Card.from_api(card_data) if card_data else None,
            row=data.get("row", 0),
            col=data.get("col", 0),
            size_x=data.get("size_x", data.get("sizeX", 4)),
            size_y=data.get("size_y", data.get("sizeY", 4)),
            parameter_mappings=data.get("parameter_mappings", []),
            visualization_settings=data.get("visualization_settings", {}),
        )


@dataclass
class QueryResult:
    """Query execution result."""

    data: dict[str, Any]
    database_id: int | None = None
    row_count: int = 0
    running_time: int = 0  # milliseconds
    status: str = "completed"

    @property
    def columns(self) -> list[str]:
        """Get column names."""
        return [col.get("name", "") for col in self.data.get("cols", [])]

    @property
    def rows(self) -> list[list[Any]]:
        """Get data rows."""
        return self.data.get("rows", [])

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> QueryResult:
        """Create from API response."""
        return cls(
            data=data.get("data", {}),
            database_id=data.get("database_id"),
            row_count=data.get("row_count", len(data.get("data", {}).get("rows", []))),
            running_time=data.get("running_time", 0),
            status=data.get("status", "completed"),
        )


class MetabaseError(Exception):
    """Metabase API error."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class MetabaseConnector:
    """
    Metabase API connector.

    Provides integration with Metabase for:
    - Query execution
    - Dashboard and card management
    - Collection organization
    - Database metadata
    """

    def __init__(self, credentials: MetabaseCredentials):
        self.credentials = credentials
        self._client: httpx.AsyncClient | None = None
        self._session_token: str | None = credentials.session_token

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers: dict[str, str] = {"Content-Type": "application/json"}

            if self.credentials.api_key:
                headers["X-Api-Key"] = self.credentials.api_key
            elif self._session_token:
                headers["X-Metabase-Session"] = self._session_token

            self._client = httpx.AsyncClient(
                base_url=f"{self.credentials.base_url}/api",
                headers=headers,
                timeout=60.0,
            )
        return self._client

    async def authenticate(self) -> str:
        """Authenticate and get session token."""
        if not self.credentials.username or not self.credentials.password:
            raise MetabaseError("Username and password required for authentication")

        client = await self._get_client()
        response = await client.post(
            "/session",
            json={
                "username": self.credentials.username,
                "password": self.credentials.password,
            },
        )

        if response.status_code >= 400:
            raise MetabaseError(f"Authentication failed: {response.text}", response.status_code)

        data = response.json()
        self._session_token = data.get("id", "")

        # Update client headers
        if self._client:
            self._client.headers["X-Metabase-Session"] = self._session_token

        return self._session_token

    async def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        json_data: dict | None = None,
    ) -> dict[str, Any] | list[Any]:
        """Make API request."""
        client = await self._get_client()
        response = await client.request(method, path, params=params, json=json_data)

        if response.status_code >= 400:
            raise MetabaseError(
                f"HTTP {response.status_code}: {response.text}", response.status_code
            )

        if response.status_code == 204:
            return {}
        return response.json()

    # =========================================================================
    # Databases
    # =========================================================================

    async def get_databases(self) -> list[Database]:
        """Get all databases."""
        data = await self._request("GET", "/database")
        dbs = data if isinstance(data, list) else data.get("data", [])
        return [Database.from_api(db) for db in dbs]

    async def get_database(self, database_id: int, include_tables: bool = False) -> Database:
        """Get a single database."""
        params = {"include": "tables"} if include_tables else {}
        data = await self._request("GET", f"/database/{database_id}", params=params)
        return Database.from_api(data)  # type: ignore

    async def get_database_metadata(self, database_id: int) -> dict[str, Any]:
        """Get database metadata including tables and fields."""
        data = await self._request("GET", f"/database/{database_id}/metadata")
        return data  # type: ignore

    # =========================================================================
    # Cards (Questions)
    # =========================================================================

    async def get_cards(self, collection_id: int | None = None) -> list[Card]:
        """Get all cards, optionally filtered by collection."""
        params: dict[str, Any] = {}
        if collection_id is not None:
            params["collection_id"] = collection_id

        data = await self._request("GET", "/card", params=params)
        cards = data if isinstance(data, list) else data.get("data", [])
        return [Card.from_api(c) for c in cards]

    async def get_card(self, card_id: int) -> Card:
        """Get a single card."""
        data = await self._request("GET", f"/card/{card_id}")
        return Card.from_api(data)  # type: ignore

    async def create_card(
        self,
        name: str,
        database_id: int,
        query: dict[str, Any],
        display: DisplayType = DisplayType.TABLE,
        collection_id: int | None = None,
        description: str | None = None,
        visualization_settings: dict[str, Any] | None = None,
    ) -> Card:
        """Create a new card (saved question)."""
        card_data: dict[str, Any] = {
            "name": name,
            "database_id": database_id,
            "dataset_query": {
                "database": database_id,
                "type": "query",
                "query": query,
            },
            "display": display.value,
            "visualization_settings": visualization_settings or {},
        }

        if collection_id is not None:
            card_data["collection_id"] = collection_id
        if description:
            card_data["description"] = description

        data = await self._request("POST", "/card", json_data=card_data)
        return Card.from_api(data)  # type: ignore

    async def update_card(
        self,
        card_id: int,
        name: str | None = None,
        description: str | None = None,
        display: DisplayType | None = None,
        visualization_settings: dict[str, Any] | None = None,
    ) -> Card:
        """Update a card."""
        card_data: dict[str, Any] = {}
        if name:
            card_data["name"] = name
        if description is not None:
            card_data["description"] = description
        if display:
            card_data["display"] = display.value
        if visualization_settings:
            card_data["visualization_settings"] = visualization_settings

        data = await self._request("PUT", f"/card/{card_id}", json_data=card_data)
        return Card.from_api(data)  # type: ignore

    async def execute_card(
        self,
        card_id: int,
        parameters: dict[str, Any] | None = None,
    ) -> QueryResult:
        """Execute a card query and get results."""
        json_data = {"parameters": parameters} if parameters else {}
        data = await self._request("POST", f"/card/{card_id}/query", json_data=json_data)
        return QueryResult.from_api(data)  # type: ignore

    async def archive_card(self, card_id: int) -> bool:
        """Archive a card."""
        await self._request("PUT", f"/card/{card_id}", json_data={"archived": True})
        return True

    # =========================================================================
    # Dashboards
    # =========================================================================

    async def get_dashboards(self) -> list[Dashboard]:
        """Get all dashboards."""
        data = await self._request("GET", "/dashboard")
        dbs = data if isinstance(data, list) else data.get("data", [])
        return [Dashboard.from_api(d) for d in dbs]

    async def get_dashboard(self, dashboard_id: int) -> Dashboard:
        """Get a single dashboard with all cards."""
        data = await self._request("GET", f"/dashboard/{dashboard_id}")
        return Dashboard.from_api(data)  # type: ignore

    async def create_dashboard(
        self,
        name: str,
        collection_id: int | None = None,
        description: str | None = None,
        parameters: list[dict[str, Any]] | None = None,
    ) -> Dashboard:
        """Create a new dashboard."""
        dashboard_data: dict[str, Any] = {"name": name}
        if collection_id is not None:
            dashboard_data["collection_id"] = collection_id
        if description:
            dashboard_data["description"] = description
        if parameters:
            dashboard_data["parameters"] = parameters

        data = await self._request("POST", "/dashboard", json_data=dashboard_data)
        return Dashboard.from_api(data)  # type: ignore

    async def add_card_to_dashboard(
        self,
        dashboard_id: int,
        card_id: int,
        row: int = 0,
        col: int = 0,
        size_x: int = 4,
        size_y: int = 4,
    ) -> DashCard:
        """Add a card to a dashboard."""
        data = await self._request(
            "POST",
            f"/dashboard/{dashboard_id}/cards",
            json_data={
                "cardId": card_id,
                "row": row,
                "col": col,
                "size_x": size_x,
                "size_y": size_y,
            },
        )
        return DashCard.from_api(data)  # type: ignore

    # =========================================================================
    # Collections
    # =========================================================================

    async def get_collections(self, namespace: str | None = None) -> list[Collection]:
        """Get all collections."""
        params = {"namespace": namespace} if namespace else {}
        data = await self._request("GET", "/collection", params=params)
        colls = data if isinstance(data, list) else data.get("data", [])
        return [Collection.from_api(c) for c in colls]

    async def get_collection(self, collection_id: int | str) -> Collection:
        """Get a single collection."""
        data = await self._request("GET", f"/collection/{collection_id}")
        return Collection.from_api(data)  # type: ignore

    async def create_collection(
        self,
        name: str,
        parent_id: int | None = None,
        description: str | None = None,
        color: str | None = None,
    ) -> Collection:
        """Create a new collection."""
        coll_data: dict[str, Any] = {"name": name}
        if parent_id is not None:
            coll_data["parent_id"] = parent_id
        if description:
            coll_data["description"] = description
        if color:
            coll_data["color"] = color

        data = await self._request("POST", "/collection", json_data=coll_data)
        return Collection.from_api(data)  # type: ignore

    async def get_collection_items(
        self,
        collection_id: int | str,
        models: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get items in a collection."""
        params: dict[str, Any] = {}
        if models:
            params["models"] = models

        data = await self._request("GET", f"/collection/{collection_id}/items", params=params)
        return data.get("data", []) if isinstance(data, dict) else []

    # =========================================================================
    # Native Query
    # =========================================================================

    async def execute_native_query(
        self,
        database_id: int,
        query: str,
        template_tags: dict[str, Any] | None = None,
    ) -> QueryResult:
        """Execute a native SQL query."""
        dataset_query: dict[str, Any] = {
            "database": database_id,
            "type": "native",
            "native": {"query": query},
        }

        if template_tags:
            dataset_query["native"]["template-tags"] = template_tags

        data = await self._request(
            "POST",
            "/dataset",
            json_data={"database": database_id, "query": dataset_query},
        )
        return QueryResult.from_api(data)  # type: ignore

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> MetabaseConnector:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


def _parse_datetime(value: str | None) -> datetime | None:
    """Parse ISO datetime string."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def get_mock_card() -> Card:
    """Get a mock card for testing."""
    return Card(
        id=123,
        name="Sales by Region",
        description="Total sales broken down by region",
        display=DisplayType.BAR,
        database_id=1,
    )


def get_mock_dashboard() -> Dashboard:
    """Get a mock dashboard for testing."""
    return Dashboard(
        id=456,
        name="Executive Dashboard",
        description="Key metrics overview",
    )
