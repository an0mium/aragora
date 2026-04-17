"""OpenAPI endpoint definitions for FastAPI v2 marketplace routes."""

from typing import Any, cast

from aragora.server.openapi.helpers import _ok_response, AUTH_REQUIREMENTS, STANDARD_ERRORS

SecurityRequirements = list[dict[str, list[str]]]

PUBLIC_SECURITY = cast(SecurityRequirements, AUTH_REQUIREMENTS["none"]["security"])
REQUIRED_SECURITY = cast(SecurityRequirements, AUTH_REQUIREMENTS["required"]["security"])

_TEMPLATE_ID_PARAM = {
    "name": "template_id",
    "in": "path",
    "required": True,
    "schema": {"type": "string"},
    "description": "Marketplace template ID.",
}


_LISTING_ID_PARAM = {
    "name": "listing_id",
    "in": "path",
    "required": True,
    "schema": {"type": "string", "maxLength": 128},
    "description": "Marketplace listing ID.",
}

_CREATE_TEMPLATE_BODY = {
    "required": True,
    "content": {
        "application/json": {
            "schema": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "id": {"type": "string", "maxLength": 128},
                    "name": {"type": "string", "minLength": 1, "maxLength": 200},
                    "description": {"type": "string", "maxLength": 2000},
                    "category": {"type": "string"},
                    "template_type": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "config": {"type": "object"},
                },
            }
        }
    },
}

_MARKETPLACE_LISTINGS_QUERY_PARAMETERS = [
    {
        "name": "type",
        "in": "query",
        "description": "Filter by listing type.",
        "schema": {"type": "string"},
    },
    {
        "name": "tag",
        "in": "query",
        "description": "Filter by tag.",
        "schema": {"type": "string"},
    },
    {
        "name": "category",
        "in": "query",
        "description": "Filter by category.",
        "schema": {"type": "string"},
    },
    {
        "name": "search",
        "in": "query",
        "description": "Search query.",
        "schema": {"type": "string", "maxLength": 500},
    },
    {
        "name": "q",
        "in": "query",
        "description": "Alias for the search query.",
        "schema": {"type": "string", "maxLength": 500},
    },
    {
        "name": "limit",
        "in": "query",
        "description": "Maximum number of listings to return.",
        "schema": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
    },
    {
        "name": "offset",
        "in": "query",
        "description": "Pagination offset.",
        "schema": {"type": "integer", "minimum": 0, "maximum": 10000, "default": 0},
    },
]


def _marketplace_data_schema(data_properties: dict[str, Any] | None = None) -> dict[str, Any]:
    """Wrap a marketplace response in the handler's ``{"data": ...}`` envelope."""
    data_schema: dict[str, Any] = {"type": "object", "additionalProperties": True}
    if data_properties:
        data_schema["properties"] = data_properties
    return {
        "type": "object",
        "properties": {
            "data": data_schema,
        },
    }


def _marketplace_listing_summary_schema() -> dict[str, Any]:
    """Schema for a marketplace listing summary object."""
    return {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "type": {"type": "string"},
            "category": {"type": "string"},
            "description": {"type": "string"},
            "featured": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
    }


def _marketplace_template_summary_schema() -> dict[str, Any]:
    """Schema for a marketplace template recommendation summary object."""
    return {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "description": {"type": "string"},
            "category": {"type": "string"},
            "pattern": {"type": "string"},
            "author_name": {"type": "string"},
            "version": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "rating": {"type": "number"},
            "rating_count": {"type": "integer"},
            "download_count": {"type": "integer"},
            "is_featured": {"type": "boolean"},
            "is_verified": {"type": "boolean"},
            "created_at": {"type": "number"},
        },
    }


def _marketplace_list_response_schema() -> dict[str, Any]:
    """Schema for listing and featured responses."""
    return _marketplace_data_schema(
        {
            "items": {
                "type": "array",
                "items": _marketplace_listing_summary_schema(),
            },
            "total": {"type": "integer"},
            "limit": {"type": "integer"},
            "offset": {"type": "integer"},
        }
    )


def _marketplace_recommendations_response_schema() -> dict[str, Any]:
    """Schema for marketplace recommendations payloads."""
    return {
        "type": "object",
        "properties": {
            "recommendations": {
                "type": "array",
                "items": _marketplace_template_summary_schema(),
            },
            "total": {"type": "integer"},
        },
    }


def _marketplace_stats_response_schema() -> dict[str, Any]:
    """Schema for marketplace stats payloads."""
    return _marketplace_data_schema(
        {
            "total_items": {"type": "integer"},
            "types": {
                "type": "object",
                "additionalProperties": {"type": "integer"},
            },
        }
    )


def _marketplace_listing_operation(
    *,
    operation_id: str,
    summary: str,
    description: str,
    response_schema: dict[str, Any],
    parameters: list[dict[str, Any]] | None = None,
    request_body: dict[str, Any] | None = None,
    security: SecurityRequirements | None = None,
    deprecated: bool = False,
    include_404: bool = False,
) -> dict[str, Any]:
    """Build a curated marketplace listing operation."""
    operation: dict[str, Any] = {
        "tags": ["Marketplace"],
        "summary": summary,
        "operationId": operation_id,
        "description": description,
        "responses": {
            "200": _ok_response("Marketplace listing response.", response_schema),
            "400": STANDARD_ERRORS["400"],
            "500": STANDARD_ERRORS["500"],
        },
        "security": security or PUBLIC_SECURITY,
    }
    if parameters:
        operation["parameters"] = parameters
    if request_body:
        operation["requestBody"] = request_body
    if include_404:
        operation["responses"]["404"] = STANDARD_ERRORS["404"]
    if security == REQUIRED_SECURITY:
        operation["responses"]["401"] = STANDARD_ERRORS["401"]
        operation["responses"]["403"] = STANDARD_ERRORS["403"]
    if deprecated:
        operation["deprecated"] = True
        operation["x-preserve-legacy-operation-id"] = True
    return operation


MARKETPLACE_ENDPOINTS = {
    "/api/v1/marketplace/listings": {
        "get": _marketplace_listing_operation(
            operation_id="marketplaceListListings",
            summary="List marketplace listings",
            description=(
                "Browse marketplace listings with optional filters and pagination. "
                "Requires `marketplace:read`."
            ),
            response_schema=_marketplace_list_response_schema(),
            parameters=_MARKETPLACE_LISTINGS_QUERY_PARAMETERS,
            security=REQUIRED_SECURITY,
        )
    },
    "/api/v1/marketplace/listings/featured": {
        "get": _marketplace_listing_operation(
            operation_id="marketplaceListFeaturedListings",
            summary="List featured marketplace listings",
            description="Return featured marketplace listings. Requires `marketplace:read`.",
            response_schema=_marketplace_list_response_schema(),
            parameters=[
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of featured listings to return.",
                    "schema": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                }
            ],
            security=REQUIRED_SECURITY,
        )
    },
    "/api/v1/marketplace/listings/stats": {
        "get": _marketplace_listing_operation(
            operation_id="marketplaceGetListingStats",
            summary="Get marketplace listing stats",
            description="Return marketplace listing counts grouped by type. Requires `marketplace:read`.",
            response_schema=_marketplace_stats_response_schema(),
            security=REQUIRED_SECURITY,
        )
    },
    "/api/v1/marketplace/recommendations": {
        "get": _marketplace_listing_operation(
            operation_id="marketplaceGetRecommendations",
            summary="Get marketplace recommendations",
            description=(
                "Return featured marketplace template recommendations ranked by marketplace "
                "signals. Requires `marketplace:read`."
            ),
            response_schema=_marketplace_recommendations_response_schema(),
            parameters=[
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of recommendations to return.",
                    "schema": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                }
            ],
            security=REQUIRED_SECURITY,
        )
    },
    "/api/v1/marketplace/listings/{listing_id}": {
        "get": _marketplace_listing_operation(
            operation_id="marketplaceGetListing",
            summary="Get marketplace listing",
            description=(
                "Return marketplace listing details for the given listing ID. "
                "Requires `marketplace:read`."
            ),
            response_schema=_marketplace_data_schema(
                {"item": _marketplace_listing_summary_schema()}
            ),
            parameters=[_LISTING_ID_PARAM],
            security=REQUIRED_SECURITY,
            include_404=True,
        )
    },
    "/api/v1/marketplace/listings/{listing_id}/install": {
        "post": _marketplace_listing_operation(
            operation_id="marketplaceInstallListing",
            summary="Install marketplace listing",
            description="Install a marketplace listing for the authenticated user.",
            response_schema=_marketplace_data_schema(),
            parameters=[_LISTING_ID_PARAM],
            security=REQUIRED_SECURITY,
            include_404=True,
        )
    },
    "/api/v1/marketplace/listings/{listing_id}/rate": {
        "post": _marketplace_listing_operation(
            operation_id="marketplaceRateListing",
            summary="Rate marketplace listing",
            description="Submit a rating and optional review for a marketplace listing.",
            response_schema=_marketplace_data_schema(),
            parameters=[_LISTING_ID_PARAM],
            request_body={
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["score"],
                            "properties": {
                                "score": {"type": "integer", "minimum": 1, "maximum": 5},
                                "review": {"type": "string", "maxLength": 2000},
                            },
                        }
                    }
                },
            },
            security=REQUIRED_SECURITY,
            include_404=True,
        )
    },
    "/api/v1/marketplace/listings/{listing_id}/launch-debate": {
        "post": _marketplace_listing_operation(
            operation_id="marketplaceLaunchDebateFromListing",
            summary="Launch debate from marketplace listing",
            description="Build a debate configuration from a marketplace listing for the authenticated user.",
            response_schema=_marketplace_data_schema(),
            parameters=[_LISTING_ID_PARAM],
            request_body={
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["question"],
                            "properties": {
                                "question": {"type": "string", "maxLength": 5000},
                                "rounds": {"type": "integer", "minimum": 1, "maximum": 20},
                            },
                        }
                    }
                },
            },
            security=REQUIRED_SECURITY,
            include_404=True,
        )
    },
    "/api/marketplace/listings": {
        "get": _marketplace_listing_operation(
            operation_id="marketplaceListListingsLegacy",
            summary="List marketplace listings",
            description="Legacy alias for listing marketplace listings. Requires `marketplace:read`.",
            response_schema=_marketplace_list_response_schema(),
            parameters=_MARKETPLACE_LISTINGS_QUERY_PARAMETERS,
            security=REQUIRED_SECURITY,
            deprecated=True,
        )
    },
    "/api/marketplace/listings/featured": {
        "get": _marketplace_listing_operation(
            operation_id="marketplaceListFeaturedListingsLegacy",
            summary="List featured marketplace listings",
            description="Legacy alias for featured marketplace listings. Requires `marketplace:read`.",
            response_schema=_marketplace_list_response_schema(),
            parameters=[
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of featured listings to return.",
                    "schema": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                }
            ],
            security=REQUIRED_SECURITY,
            deprecated=True,
        )
    },
    "/api/marketplace/listings/stats": {
        "get": _marketplace_listing_operation(
            operation_id="marketplaceGetListingStatsLegacy",
            summary="Get marketplace listing stats",
            description="Legacy alias for marketplace listing stats. Requires `marketplace:read`.",
            response_schema=_marketplace_stats_response_schema(),
            security=REQUIRED_SECURITY,
            deprecated=True,
        )
    },
    "/api/marketplace/listings/{listing_id}": {
        "get": _marketplace_listing_operation(
            operation_id="marketplaceGetListingLegacy",
            summary="Get marketplace listing",
            description="Legacy alias for marketplace listing details. Requires `marketplace:read`.",
            response_schema=_marketplace_data_schema(
                {"item": _marketplace_listing_summary_schema()}
            ),
            parameters=[_LISTING_ID_PARAM],
            security=REQUIRED_SECURITY,
            deprecated=True,
            include_404=True,
        )
    },
    "/api/marketplace/listings/{listing_id}/install": {
        "post": _marketplace_listing_operation(
            operation_id="marketplaceInstallListingLegacy",
            summary="Install marketplace listing",
            description="Legacy alias for installing a marketplace listing.",
            response_schema=_marketplace_data_schema(),
            parameters=[_LISTING_ID_PARAM],
            security=REQUIRED_SECURITY,
            deprecated=True,
            include_404=True,
        )
    },
    "/api/marketplace/listings/{listing_id}/rate": {
        "post": _marketplace_listing_operation(
            operation_id="marketplaceRateListingLegacy",
            summary="Rate marketplace listing",
            description="Legacy alias for rating a marketplace listing.",
            response_schema=_marketplace_data_schema(),
            parameters=[_LISTING_ID_PARAM],
            request_body={
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["score"],
                            "properties": {
                                "score": {"type": "integer", "minimum": 1, "maximum": 5},
                                "review": {"type": "string", "maxLength": 2000},
                            },
                        }
                    }
                },
            },
            security=REQUIRED_SECURITY,
            deprecated=True,
            include_404=True,
        )
    },
    "/api/marketplace/listings/{listing_id}/launch-debate": {
        "post": _marketplace_listing_operation(
            operation_id="marketplaceLaunchDebateFromListingLegacy",
            summary="Launch debate from marketplace listing",
            description="Legacy alias for launching a debate from a marketplace listing.",
            response_schema=_marketplace_data_schema(),
            parameters=[_LISTING_ID_PARAM],
            request_body={
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["question"],
                            "properties": {
                                "question": {"type": "string", "maxLength": 5000},
                                "rounds": {"type": "integer", "minimum": 1, "maximum": 20},
                            },
                        }
                    }
                },
            },
            security=REQUIRED_SECURITY,
            deprecated=True,
            include_404=True,
        )
    },
    "/api/v2/marketplace/templates": {
        "get": {
            "tags": ["Marketplace"],
            "summary": "List marketplace templates",
            "operationId": "listMarketplaceTemplatesV2",
            "description": (
                "List/search marketplace templates with optional filters "
                "(query, category, type, tags, pagination)."
            ),
            "security": PUBLIC_SECURITY,
            "parameters": [
                {
                    "name": "q",
                    "in": "query",
                    "description": "Search query.",
                    "schema": {"type": "string", "maxLength": 500},
                },
                {
                    "name": "category",
                    "in": "query",
                    "description": "Filter by category.",
                    "schema": {"type": "string"},
                },
                {
                    "name": "type",
                    "in": "query",
                    "description": "Filter by template type.",
                    "schema": {"type": "string"},
                },
                {
                    "name": "tags",
                    "in": "query",
                    "description": "Comma-separated tag list.",
                    "schema": {"type": "string", "maxLength": 1000},
                },
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of records.",
                    "schema": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
                },
                {
                    "name": "offset",
                    "in": "query",
                    "description": "Pagination offset.",
                    "schema": {"type": "integer", "minimum": 0, "maximum": 10000, "default": 0},
                },
            ],
            "responses": {
                "200": _ok_response(
                    "Marketplace templates.",
                    {
                        "templates": {"type": "array", "items": {"type": "object"}},
                        "count": {"type": "integer"},
                        "limit": {"type": "integer"},
                        "offset": {"type": "integer"},
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "post": {
            "tags": ["Marketplace"],
            "summary": "Create marketplace template",
            "operationId": "createMarketplaceTemplateV2",
            "description": "Create/import a marketplace template. Requires `marketplace:write`.",
            "security": REQUIRED_SECURITY,
            "requestBody": _CREATE_TEMPLATE_BODY,
            "responses": {
                "201": _ok_response(
                    "Template created.",
                    {
                        "id": {"type": "string"},
                        "success": {"type": "boolean"},
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v2/marketplace/categories": {
        "get": {
            "tags": ["Marketplace"],
            "summary": "List marketplace categories",
            "operationId": "listMarketplaceCategoriesV2",
            "description": "List available marketplace categories.",
            "security": PUBLIC_SECURITY,
            "responses": {
                "200": _ok_response(
                    "Marketplace categories.",
                    {"categories": {"type": "array", "items": {"type": "string"}}},
                ),
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v2/marketplace/status": {
        "get": {
            "tags": ["Marketplace"],
            "summary": "Get marketplace status",
            "operationId": "getMarketplaceStatusV2",
            "description": "Get marketplace health/circuit-breaker status.",
            "security": PUBLIC_SECURITY,
            "responses": {
                "200": _ok_response(
                    "Marketplace status.",
                    {
                        "status": {"type": "string"},
                        "circuit_breaker": {"type": "object"},
                    },
                ),
            },
        }
    },
    "/api/v2/marketplace/templates/import": {
        "post": {
            "tags": ["Marketplace"],
            "summary": "Import marketplace template",
            "operationId": "importMarketplaceTemplateV2",
            "description": "Import a marketplace template. Requires `marketplace:write`.",
            "security": REQUIRED_SECURITY,
            "requestBody": _CREATE_TEMPLATE_BODY,
            "responses": {
                "201": _ok_response(
                    "Template imported.",
                    {
                        "id": {"type": "string"},
                        "success": {"type": "boolean"},
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v2/marketplace/templates/{template_id}/ratings": {
        "get": {
            "tags": ["Marketplace"],
            "summary": "Get template ratings",
            "operationId": "getMarketplaceTemplateRatingsV2",
            "description": "Get ratings and average score for a marketplace template.",
            "security": PUBLIC_SECURITY,
            "parameters": [_TEMPLATE_ID_PARAM],
            "responses": {
                "200": _ok_response(
                    "Template ratings.",
                    {
                        "ratings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "user_id": {"type": "string"},
                                    "score": {"type": "integer"},
                                    "review": {"type": "string"},
                                    "created_at": {"type": "string", "format": "date-time"},
                                },
                            },
                        },
                        "average": {"type": "number"},
                        "count": {"type": "integer"},
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "post": {
            "tags": ["Marketplace"],
            "summary": "Rate template",
            "operationId": "rateMarketplaceTemplateV2",
            "description": "Add a template rating. Requires `marketplace:write`.",
            "security": REQUIRED_SECURITY,
            "parameters": [_TEMPLATE_ID_PARAM],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["score"],
                            "properties": {
                                "score": {"type": "integer", "minimum": 1, "maximum": 5},
                                "review": {"type": "string", "maxLength": 2000},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response(
                    "Template rating saved.",
                    {"success": {"type": "boolean"}, "average_rating": {"type": "number"}},
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v2/marketplace/templates/{template_id}/export": {
        "get": {
            "tags": ["Marketplace"],
            "summary": "Export template",
            "operationId": "exportMarketplaceTemplateV2",
            "description": "Export a marketplace template as JSON.",
            "security": PUBLIC_SECURITY,
            "parameters": [_TEMPLATE_ID_PARAM],
            "responses": {
                "200": {
                    "description": "Template JSON export.",
                    "content": {"application/json": {"schema": {"type": "object"}}},
                },
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v2/marketplace/templates/{template_id}": {
        "get": {
            "tags": ["Marketplace"],
            "summary": "Get template",
            "operationId": "getMarketplaceTemplateV2",
            "description": "Get template details by ID.",
            "security": PUBLIC_SECURITY,
            "parameters": [_TEMPLATE_ID_PARAM],
            "responses": {
                "200": _ok_response(
                    "Template details.",
                    {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "category": {"type": "string"},
                        "template_type": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "downloads": {"type": "integer"},
                        "stars": {"type": "integer"},
                        "average_rating": {"type": "number"},
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "delete": {
            "tags": ["Marketplace"],
            "summary": "Delete template",
            "operationId": "deleteMarketplaceTemplateV2",
            "description": "Delete a marketplace template. Requires `marketplace:delete`.",
            "security": REQUIRED_SECURITY,
            "parameters": [_TEMPLATE_ID_PARAM],
            "responses": {
                "200": _ok_response(
                    "Template deleted.",
                    {
                        "success": {"type": "boolean"},
                        "deleted": {"type": "string"},
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/v2/marketplace/templates/{template_id}/star": {
        "post": {
            "tags": ["Marketplace"],
            "summary": "Star template",
            "operationId": "starMarketplaceTemplateV2",
            "description": "Star a marketplace template. Requires `marketplace:write`.",
            "security": REQUIRED_SECURITY,
            "parameters": [_TEMPLATE_ID_PARAM],
            "responses": {
                "200": _ok_response(
                    "Template starred.",
                    {"success": {"type": "boolean"}, "stars": {"type": "integer"}},
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
}
