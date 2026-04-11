"""Template registry endpoint definitions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS

__all__ = [
    "TEMPLATE_REGISTRY_ENDPOINTS",
    "get_registry_listing_schema",
    "get_registry_listing_required_fields",
    "get_registry_listing_property_names",
    "build_sample_listing",
]

_REGISTRY_LISTING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "description": {"type": "string"},
        "category": {"type": "string"},
        "author_id": {"type": "string"},
        "version": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "status": {"type": "string"},
        "is_verified": {"type": "boolean"},
        "is_builtin": {"type": "boolean"},
        "install_count": {"type": "integer"},
        "rating_average": {"type": "number"},
        "rating_count": {"type": "integer"},
        "created_at": {"type": "string", "format": "date-time"},
        "updated_at": {"type": "string", "format": "date-time"},
        "approved_by": {"type": ["string", "null"]},
    },
    "required": [
        "id",
        "name",
        "description",
        "category",
        "author_id",
        "version",
        "tags",
        "status",
        "is_verified",
        "is_builtin",
        "install_count",
        "rating_average",
        "rating_count",
        "created_at",
        "updated_at",
        "approved_by",
    ],
}


def get_registry_listing_schema() -> dict[str, Any]:
    """Return a copy of the registry listing schema for validation and testing."""
    return deepcopy(_REGISTRY_LISTING_SCHEMA)


def get_registry_listing_required_fields() -> list[str]:
    """Return the list of required fields for a registry listing."""
    return list(_REGISTRY_LISTING_SCHEMA["required"])


def get_registry_listing_property_names() -> list[str]:
    """Return all property names defined in the registry listing schema."""
    return sorted(_REGISTRY_LISTING_SCHEMA["properties"].keys())


def build_sample_listing(**overrides: Any) -> dict[str, Any]:
    """Build a sample registry listing dict suitable for schema validation tests."""
    sample: dict[str, Any] = {
        "id": "tpl_sample_001",
        "name": "Sample Template",
        "description": "A sample template for testing.",
        "category": "general",
        "author_id": "user_test",
        "version": "1.0.0",
        "tags": ["test", "sample"],
        "status": "published",
        "is_verified": False,
        "is_builtin": False,
        "install_count": 0,
        "rating_average": 0.0,
        "rating_count": 0,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "approved_by": None,
    }
    sample.update(overrides)
    return sample


TEMPLATE_REGISTRY_ENDPOINTS: dict[str, Any] = {
    "/api/v1/templates/registry/{listing_id}": {
        "get": {
            "tags": ["Templates"],
            "summary": "Get template registry listing",
            "operationId": "getTemplateRegistryListing",
            "description": "Retrieve a single public template registry listing by its identifier.",
            "parameters": [
                {
                    "name": "listing_id",
                    "in": "path",
                    "required": True,
                    "description": "Unique template registry listing identifier.",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Template registry listing", _REGISTRY_LISTING_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
        }
    }
}
