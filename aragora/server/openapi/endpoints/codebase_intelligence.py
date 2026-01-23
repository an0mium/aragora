"""Code intelligence endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS, AUTH_REQUIREMENTS


def _codebase_params():
    return [
        {"name": "repo", "in": "path", "required": True, "schema": {"type": "string"}},
    ]


CODEBASE_INTELLIGENCE_ENDPOINTS = {
    "/api/v1/codebase/{repo}/analyze": {
        "post": {
            "tags": ["Codebase"],
            "summary": "Analyze codebase",
            "description": "Analyze codebase structure and symbols.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": _codebase_params(),
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Analysis", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/{repo}/symbols": {
        "get": {
            "tags": ["Codebase"],
            "summary": "List symbols",
            "description": "List symbols in a codebase.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": _codebase_params(),
            "responses": {
                "200": _ok_response("Symbols", "StandardSuccessResponse"),
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/{repo}/callgraph": {
        "get": {
            "tags": ["Codebase"],
            "summary": "Call graph",
            "description": "Get call graph for the codebase.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": _codebase_params(),
            "responses": {
                "200": _ok_response("Call graph", "StandardSuccessResponse"),
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/{repo}/deadcode": {
        "get": {
            "tags": ["Codebase"],
            "summary": "Dead code",
            "description": "Find dead or unreachable code.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": _codebase_params(),
            "responses": {
                "200": _ok_response("Dead code", "StandardSuccessResponse"),
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/{repo}/impact": {
        "post": {
            "tags": ["Codebase"],
            "summary": "Impact analysis",
            "description": "Analyze impact of changes.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": _codebase_params(),
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Impact", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/{repo}/understand": {
        "post": {
            "tags": ["Codebase"],
            "summary": "Understand code",
            "description": "Answer questions about the codebase.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": _codebase_params(),
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Answer", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/{repo}/audit": {
        "post": {
            "tags": ["Codebase"],
            "summary": "Run audit",
            "description": "Run comprehensive code audit.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": _codebase_params(),
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Audit started", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/{repo}/audit/{audit_id}": {
        "get": {
            "tags": ["Codebase"],
            "summary": "Audit status",
            "description": "Get status/result of a code audit.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "repo", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "audit_id", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Audit status", "StandardSuccessResponse"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/codebase/{repo}/analyze": {
        "post": {
            "tags": ["Codebase"],
            "summary": "Analyze codebase (alias)",
            "description": "Alias for codebase analysis.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": _codebase_params(),
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Analysis", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/codebase/{repo}/symbols": {
        "get": {
            "tags": ["Codebase"],
            "summary": "List symbols (alias)",
            "description": "Alias for codebase symbols.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": _codebase_params(),
            "responses": {
                "200": _ok_response("Symbols", "StandardSuccessResponse"),
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/codebase/{repo}/callgraph": {
        "get": {
            "tags": ["Codebase"],
            "summary": "Call graph (alias)",
            "description": "Alias for call graph.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": _codebase_params(),
            "responses": {
                "200": _ok_response("Call graph", "StandardSuccessResponse"),
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/codebase/{repo}/deadcode": {
        "get": {
            "tags": ["Codebase"],
            "summary": "Dead code (alias)",
            "description": "Alias for dead code.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": _codebase_params(),
            "responses": {
                "200": _ok_response("Dead code", "StandardSuccessResponse"),
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/codebase/{repo}/impact": {
        "post": {
            "tags": ["Codebase"],
            "summary": "Impact analysis (alias)",
            "description": "Alias for impact analysis.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": _codebase_params(),
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Impact", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/codebase/{repo}/understand": {
        "post": {
            "tags": ["Codebase"],
            "summary": "Understand code (alias)",
            "description": "Alias for code understanding.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": _codebase_params(),
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Answer", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/codebase/{repo}/audit": {
        "post": {
            "tags": ["Codebase"],
            "summary": "Run audit (alias)",
            "description": "Alias for code audit.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": _codebase_params(),
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Audit started", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/codebase/{repo}/audit/{audit_id}": {
        "get": {
            "tags": ["Codebase"],
            "summary": "Audit status (alias)",
            "description": "Alias for audit status.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "repo", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "audit_id", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Audit status", "StandardSuccessResponse"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
}


__all__ = ["CODEBASE_INTELLIGENCE_ENDPOINTS"]
