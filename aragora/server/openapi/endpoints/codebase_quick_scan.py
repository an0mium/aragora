"""Codebase quick scan endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS, AUTH_REQUIREMENTS

CODEBASE_QUICK_SCAN_ENDPOINTS = {
    "/api/codebase/quick-scan": {
        "post": {
            "tags": ["Codebase"],
            "summary": "Quick scan",
            "operationId": "createCodebaseQuickScan",
            "description": "Run a quick security scan.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "repository_url": {
                                    "type": "string",
                                    "description": "Repository URL to scan",
                                },
                                "branch": {"type": "string", "description": "Branch to scan"},
                                "paths": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Specific paths to scan",
                                },
                                "scan_types": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Types of scans to run",
                                },
                                "depth": {
                                    "type": "string",
                                    "enum": ["shallow", "deep"],
                                    "description": "Scan depth",
                                },
                            },
                            "required": ["repository_url"],
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Scan started", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/codebase/quick-scan/{scan_id}": {
        "get": {
            "tags": ["Codebase"],
            "summary": "Quick scan result",
            "operationId": "getCodebaseQuickScan",
            "description": "Get a quick scan result.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "scan_id", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Scan result", "StandardSuccessResponse"),
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/codebase/quick-scans": {
        "get": {
            "tags": ["Codebase"],
            "summary": "List quick scans",
            "operationId": "listCodebaseQuickScans",
            "description": "List recent quick scans.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "responses": {
                "200": _ok_response("Scan list", "StandardSuccessResponse"),
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
}


__all__ = ["CODEBASE_QUICK_SCAN_ENDPOINTS"]
