"""Status page and public readiness endpoint definitions."""

from aragora.server.openapi.helpers import AUTH_REQUIREMENTS, _ok_response

_str = {"type": "string"}
_bool = {"type": "boolean"}

_PUBLIC_SURFACE_SCHEMA = {
    "type": "object",
    "required": [
        "id",
        "name",
        "readiness",
        "paths",
        "message",
        "backend_conditional",
        "placeholder_backed",
        "details",
    ],
    "properties": {
        "id": _str,
        "name": _str,
        "readiness": {
            "type": "string",
            "enum": ["live", "partial"],
            "description": "Readiness state for this exposed public surface.",
        },
        "paths": {
            "type": "array",
            "items": _str,
            "description": "Public paths that belong to this surface.",
        },
        "message": _str,
        "backend_conditional": _bool,
        "placeholder_backed": _bool,
        "details": {
            "type": "object",
            "additionalProperties": True,
            "description": "Optional deployment-specific readiness details.",
        },
    },
}

_PUBLIC_SURFACES_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["data"],
    "properties": {
        "data": {
            "type": "object",
            "required": ["surfaces", "summary"],
            "properties": {
                "surfaces": {
                    "type": "array",
                    "items": _PUBLIC_SURFACE_SCHEMA,
                },
                "summary": {
                    "type": "object",
                    "required": ["total", "live", "partial"],
                    "properties": {
                        "total": {"type": "integer"},
                        "live": {"type": "integer"},
                        "partial": {"type": "integer"},
                    },
                },
            },
        }
    },
}


STATUS_ENDPOINTS: dict = {
    "/api/v1/public/surfaces": {
        "get": {
            "tags": ["Status", "Public", "Readiness"],
            "summary": "List public surface readiness",
            "description": (
                "Return the public status, onboarding, spectate, OpenAPI, and memory "
                "surfaces exposed by this deployment, including readiness and "
                "backend-conditional flags for each surface."
            ),
            "operationId": "getPublicSurfaces",
            "security": AUTH_REQUIREMENTS["none"]["security"],
            "responses": {
                "200": _ok_response(
                    "Public surface readiness inventory",
                    _PUBLIC_SURFACES_RESPONSE_SCHEMA,
                )
            },
        }
    },
}
