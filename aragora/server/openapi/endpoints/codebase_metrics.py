"""Codebase metrics endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS, AUTH_REQUIREMENTS

CODEBASE_METRICS_ENDPOINTS = {
    "/api/v1/codebase/{repo}/metrics/analyze": {
        "post": {
            "tags": ["Codebase"],
            "summary": "Run metrics analysis",
            "operationId": "createCodebaseMetricsAnalyze",
            "description": "Analyze code complexity and duplication for a repository.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "repo",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["repo_path"],
                            "properties": {
                                "repo_path": {"type": "string"},
                                "include_patterns": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "exclude_patterns": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "complexity_warning": {"type": "integer"},
                                "complexity_error": {"type": "integer"},
                                "duplication_threshold": {"type": "integer"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Metrics analysis started", "CodebaseMetricsStartResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/{repo}/metrics": {
        "get": {
            "tags": ["Codebase"],
            "summary": "Get latest metrics",
            "operationId": "getCodebaseMetricsLatest",
            "description": "Fetch the latest metrics report.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "repo",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Metrics report", "CodebaseMetricsReportResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/{repo}/metrics/{analysis_id}": {
        "get": {
            "tags": ["Codebase"],
            "summary": "Get metrics by ID",
            "operationId": "getCodebaseMetricById",
            "description": "Fetch a metrics report by analysis ID.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "repo",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "analysis_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("Metrics report", "CodebaseMetricsReportResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/{repo}/metrics/history": {
        "get": {
            "tags": ["Codebase"],
            "summary": "Metrics history",
            "operationId": "getCodebaseMetricsHistory",
            "description": "List metrics analyses for a repository.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "repo",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "default": 20},
                },
                {
                    "name": "offset",
                    "in": "query",
                    "schema": {"type": "integer", "default": 0},
                },
            ],
            "responses": {
                "200": _ok_response("Metrics history", "CodebaseMetricsHistoryResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/{repo}/hotspots": {
        "get": {
            "tags": ["Codebase"],
            "summary": "Complexity hotspots",
            "operationId": "getCodebaseHotspot",
            "description": "List complexity hotspots from the latest analysis.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "repo",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "min_complexity",
                    "in": "query",
                    "schema": {"type": "integer", "default": 5},
                },
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "default": 20},
                },
            ],
            "responses": {
                "200": _ok_response("Hotspots", "CodebaseHotspotListResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/{repo}/duplicates": {
        "get": {
            "tags": ["Codebase"],
            "summary": "Code duplicates",
            "operationId": "getCodebaseDuplicate",
            "description": "List duplicate code blocks from the latest analysis.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "repo",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "min_lines",
                    "in": "query",
                    "schema": {"type": "integer", "default": 6},
                },
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "default": 20},
                },
            ],
            "responses": {
                "200": _ok_response("Duplicates", "CodebaseDuplicateListResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/{repo}/metrics/file/{file_path}": {
        "get": {
            "tags": ["Codebase"],
            "summary": "File metrics",
            "operationId": "getCodebaseMetricsFile",
            "description": "Get metrics for a specific file (URL-encoded path).",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "repo",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "file_path",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("File metrics", "CodebaseFileMetricsResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
}

__all__ = ["CODEBASE_METRICS_ENDPOINTS"]
