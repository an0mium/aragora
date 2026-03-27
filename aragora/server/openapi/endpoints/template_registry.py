"""Template registry endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS

_REGISTRY_LISTING_SCHEMA = {
    "type": "object",
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
}


TEMPLATE_REGISTRY_ENDPOINTS = {
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
