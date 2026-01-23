"""
Knack Connector.

Integration with Knack API:
- Objects (tables)
- Records (CRUD)
- Views
- Fields
- Files and images

Requires Knack application ID and API key.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class FieldType(str, Enum):
    """Knack field types."""

    SHORT_TEXT = "short_text"
    PARAGRAPH_TEXT = "paragraph_text"
    RICH_TEXT = "rich_text"
    NUMBER = "number"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    YES_NO = "boolean"
    MULTIPLE_CHOICE = "multiple_choice"
    DATE_TIME = "date_time"
    EMAIL = "email"
    PHONE = "phone"
    LINK = "link"
    ADDRESS = "address"
    NAME = "name"
    IMAGE = "image"
    FILE = "file"
    CONNECTION = "connection"
    AUTO_INCREMENT = "auto_increment"
    EQUATION = "equation"
    CONCATENATION = "concatenation"
    USER = "user"
    USER_ROLES = "user_roles"
    RATING = "rating"
    TIMER = "timer"
    SIGNATURE = "signature"


@dataclass
class KnackCredentials:
    """Knack API credentials."""

    application_id: str
    api_key: str
    base_url: str = "https://api.knack.com/v1"


@dataclass
class KnackField:
    """Knack object field."""

    key: str
    name: str
    type: FieldType
    required: bool = False
    unique: bool = False

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> KnackField:
        """Create from API response."""
        return cls(
            key=data.get("key", ""),
            name=data.get("name", ""),
            type=FieldType(data.get("type", "short_text")),
            required=data.get("required", False),
            unique=data.get("unique", False),
        )


@dataclass
class KnackObject:
    """Knack object (table)."""

    key: str
    name: str
    identifier: str | None = None
    fields: list[KnackField] = field(default_factory=list)
    connections: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> KnackObject:
        """Create from API response."""
        return cls(
            key=data.get("key", ""),
            name=data.get("name", ""),
            identifier=data.get("identifier"),
            fields=[KnackField.from_api(f) for f in data.get("fields", [])],
            connections=data.get("connections", {}),
        )


@dataclass
class KnackRecord:
    """Knack record."""

    id: str
    fields: dict[str, Any]

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> KnackRecord:
        """Create from API response."""
        record_id = data.get("id", "")
        # Remove 'id' from fields to avoid duplication
        fields = {k: v for k, v in data.items() if k != "id"}
        return cls(id=record_id, fields=fields)

    def get_field(self, field_key: str, default: Any = None) -> Any:
        """Get a field value by key."""
        return self.fields.get(field_key, default)

    def get_raw_field(self, field_key: str) -> Any:
        """Get the raw field value (for fields that have _raw versions)."""
        return self.fields.get(f"{field_key}_raw", self.fields.get(field_key))


@dataclass
class KnackView:
    """Knack view."""

    key: str
    name: str
    type: str
    source_object: str | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> KnackView:
        """Create from API response."""
        return cls(
            key=data.get("key", ""),
            name=data.get("name", ""),
            type=data.get("type", ""),
            source_object=data.get("source", {}).get("object"),
        )


@dataclass
class KnackScene:
    """Knack scene (page)."""

    key: str
    name: str
    slug: str
    views: list[KnackView] = field(default_factory=list)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> KnackScene:
        """Create from API response."""
        return cls(
            key=data.get("key", ""),
            name=data.get("name", ""),
            slug=data.get("slug", ""),
            views=[KnackView.from_api(v) for v in data.get("views", [])],
        )


class KnackError(Exception):
    """Knack API error."""

    def __init__(self, message: str, status_code: int | None = None, errors: list | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.errors = errors or []


class KnackConnector:
    """
    Knack API connector.

    Provides integration with Knack for:
    - Object (table) management
    - Record CRUD operations
    - View-based queries
    - Schema introspection
    """

    def __init__(self, credentials: KnackCredentials):
        self.credentials = credentials
        self._client: httpx.AsyncClient | None = None
        self._schema: dict[str, Any] | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.credentials.base_url,
                headers={
                    "X-Knack-Application-Id": self.credentials.application_id,
                    "X-Knack-REST-API-Key": self.credentials.api_key,
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        json_data: dict | None = None,
    ) -> dict[str, Any]:
        """Make API request."""
        client = await self._get_client()
        response = await client.request(method, path, params=params, json=json_data)

        if response.status_code >= 400:
            try:
                error_data = response.json()
                raise KnackError(
                    message=error_data.get("message", response.text),
                    status_code=response.status_code,
                    errors=error_data.get("errors", []),
                )
            except ValueError:
                raise KnackError(
                    f"HTTP {response.status_code}: {response.text}",
                    status_code=response.status_code,
                )

        return response.json()

    # =========================================================================
    # Schema
    # =========================================================================

    async def get_application_schema(self, force_refresh: bool = False) -> dict[str, Any]:
        """Get the full application schema (objects, scenes, etc.)."""
        if self._schema is None or force_refresh:
            data = await self._request(
                "GET",
                f"/applications/{self.credentials.application_id}",
            )
            self._schema = data.get("application", data)
        return self._schema

    async def get_objects(self) -> list[KnackObject]:
        """Get all objects in the application."""
        schema = await self.get_application_schema()
        return [KnackObject.from_api(obj) for obj in schema.get("objects", [])]

    async def get_object(self, object_key: str) -> KnackObject:
        """Get a specific object by key."""
        objects = await self.get_objects()
        for obj in objects:
            if obj.key == object_key:
                return obj
        raise KnackError(f"Object {object_key} not found", status_code=404)

    async def get_scenes(self) -> list[KnackScene]:
        """Get all scenes (pages) in the application."""
        schema = await self.get_application_schema()
        return [KnackScene.from_api(scene) for scene in schema.get("scenes", [])]

    # =========================================================================
    # Records - Object-based
    # =========================================================================

    async def get_records(
        self,
        object_key: str,
        page: int = 1,
        rows_per_page: int = 25,
        sort_field: str | None = None,
        sort_order: str = "asc",
        filters: list[dict[str, Any]] | None = None,
    ) -> tuple[list[KnackRecord], int, int]:
        """
        Get records from an object.

        filters format:
        [
            {"field": "field_1", "operator": "is", "value": "test"},
            {"field": "field_2", "operator": "contains", "value": "search"}
        ]

        Returns (records, total_pages, total_records).
        """
        params: dict[str, Any] = {
            "page": page,
            "rows_per_page": min(rows_per_page, 1000),
        }

        if sort_field:
            params["sort_field"] = sort_field
            params["sort_order"] = sort_order

        if filters:
            params["filters"] = _encode_filters(filters)

        data = await self._request("GET", f"/objects/{object_key}/records", params=params)

        records = [KnackRecord.from_api(r) for r in data.get("records", [])]
        total_pages = data.get("total_pages", 1)
        total_records = data.get("total_records", len(records))

        return records, total_pages, total_records

    async def get_record(self, object_key: str, record_id: str) -> KnackRecord:
        """Get a single record by ID."""
        data = await self._request("GET", f"/objects/{object_key}/records/{record_id}")
        return KnackRecord.from_api(data)

    async def create_record(
        self,
        object_key: str,
        fields: dict[str, Any],
    ) -> KnackRecord:
        """Create a new record."""
        data = await self._request(
            "POST",
            f"/objects/{object_key}/records",
            json_data=fields,
        )
        return KnackRecord.from_api(data)

    async def update_record(
        self,
        object_key: str,
        record_id: str,
        fields: dict[str, Any],
    ) -> KnackRecord:
        """Update an existing record."""
        data = await self._request(
            "PUT",
            f"/objects/{object_key}/records/{record_id}",
            json_data=fields,
        )
        return KnackRecord.from_api(data)

    async def delete_record(
        self,
        object_key: str,
        record_id: str,
    ) -> bool:
        """Delete a record."""
        data = await self._request("DELETE", f"/objects/{object_key}/records/{record_id}")
        return data.get("delete", False)

    # =========================================================================
    # Records - View-based
    # =========================================================================

    async def get_view_records(
        self,
        scene_key: str,
        view_key: str,
        page: int = 1,
        rows_per_page: int = 25,
        sort_field: str | None = None,
        sort_order: str = "asc",
        filters: list[dict[str, Any]] | None = None,
    ) -> tuple[list[KnackRecord], int, int]:
        """
        Get records from a view.

        Returns (records, total_pages, total_records).
        """
        params: dict[str, Any] = {
            "page": page,
            "rows_per_page": min(rows_per_page, 1000),
        }

        if sort_field:
            params["sort_field"] = sort_field
            params["sort_order"] = sort_order

        if filters:
            params["filters"] = _encode_filters(filters)

        data = await self._request(
            "GET",
            f"/pages/{scene_key}/views/{view_key}/records",
            params=params,
        )

        records = [KnackRecord.from_api(r) for r in data.get("records", [])]
        total_pages = data.get("total_pages", 1)
        total_records = data.get("total_records", len(records))

        return records, total_pages, total_records

    async def create_view_record(
        self,
        scene_key: str,
        view_key: str,
        fields: dict[str, Any],
    ) -> KnackRecord:
        """Create a record through a view."""
        data = await self._request(
            "POST",
            f"/pages/{scene_key}/views/{view_key}/records",
            json_data=fields,
        )
        return KnackRecord.from_api(data)

    async def update_view_record(
        self,
        scene_key: str,
        view_key: str,
        record_id: str,
        fields: dict[str, Any],
    ) -> KnackRecord:
        """Update a record through a view."""
        data = await self._request(
            "PUT",
            f"/pages/{scene_key}/views/{view_key}/records/{record_id}",
            json_data=fields,
        )
        return KnackRecord.from_api(data)

    async def delete_view_record(
        self,
        scene_key: str,
        view_key: str,
        record_id: str,
    ) -> bool:
        """Delete a record through a view."""
        data = await self._request(
            "DELETE",
            f"/pages/{scene_key}/views/{view_key}/records/{record_id}",
        )
        return data.get("delete", False)

    # =========================================================================
    # Convenience methods
    # =========================================================================

    async def find_records(
        self,
        object_key: str,
        field_key: str,
        value: Any,
        operator: str = "is",
    ) -> list[KnackRecord]:
        """Find records where a field matches a value."""
        filters = [{"field": field_key, "operator": operator, "value": value}]
        records, _, _ = await self.get_records(object_key, filters=filters)
        return records

    async def get_all_records(
        self,
        object_key: str,
        sort_field: str | None = None,
        sort_order: str = "asc",
        filters: list[dict[str, Any]] | None = None,
    ) -> list[KnackRecord]:
        """Get all records (handles pagination automatically)."""
        all_records: list[KnackRecord] = []
        page = 1

        while True:
            records, total_pages, _ = await self.get_records(
                object_key,
                page=page,
                rows_per_page=1000,
                sort_field=sort_field,
                sort_order=sort_order,
                filters=filters,
            )
            all_records.extend(records)

            if page >= total_pages:
                break
            page += 1

        return all_records

    async def upsert_record(
        self,
        object_key: str,
        lookup_field: str,
        lookup_value: Any,
        fields: dict[str, Any],
    ) -> KnackRecord:
        """
        Update a record if it exists, create if not.

        Uses lookup_field and lookup_value to find existing record.
        """
        existing = await self.find_records(object_key, lookup_field, lookup_value)

        if existing:
            return await self.update_record(object_key, existing[0].id, fields)
        else:
            fields[lookup_field] = lookup_value
            return await self.create_record(object_key, fields)

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> KnackConnector:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


def _encode_filters(filters: list[dict[str, Any]]) -> str:
    """Encode filters for Knack API."""
    import json

    return json.dumps(filters)


def get_mock_record() -> KnackRecord:
    """Get a mock record for testing."""
    return KnackRecord(
        id="5f1234567890abcdef123456",
        fields={
            "field_1": "Test Value",
            "field_2": 42,
            "field_3": True,
            "field_4": "option1",
        },
    )


def get_mock_object() -> KnackObject:
    """Get a mock object for testing."""
    return KnackObject(
        key="object_1",
        name="Customers",
        identifier="field_1",
        fields=[
            KnackField(key="field_1", name="Name", type=FieldType.SHORT_TEXT, required=True),
            KnackField(key="field_2", name="Email", type=FieldType.EMAIL),
            KnackField(key="field_3", name="Active", type=FieldType.YES_NO),
        ],
    )
