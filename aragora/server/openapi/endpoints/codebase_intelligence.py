"""Code intelligence endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS, AUTH_REQUIREMENTS


def _codebase_params() -> list[dict[str, str | bool | dict[str, str]]]:
    return [
        {"name": "repo", "in": "path", "required": True, "schema": {"type": "string"}},
    ]


CODEBASE_INTELLIGENCE_ENDPOINTS = {
    "/api/v1/codebase/{repo}/analyze": {
        "post": {
            "tags": ["Codebase"],
            "summary": "Analyze codebase",
            "operationId": "createCodebaseAnalyze",
            "description": "Analyze codebase structure and symbols.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": _codebase_params(),
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "paths": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Paths to analyze (empty for full repo)",
                                },
                                "depth": {
                                    "type": "integer",
                                    "description": "Analysis depth level",
                                },
                                "include_tests": {
                                    "type": "boolean",
                                    "description": "Include test files in analysis",
                                },
                            },
                        }
                    }
                },
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
            "operationId": "getCodebaseSymbol",
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
            "operationId": "getCodebaseCallgraph",
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
            "operationId": "getCodebaseDeadcode",
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
            "operationId": "createCodebaseImpact",
            "description": "Analyze impact of changes.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": _codebase_params(),
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "files_changed": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of changed file paths",
                                },
                                "commit_sha": {
                                    "type": "string",
                                    "description": "Git commit SHA to analyze",
                                },
                                "include_indirect": {
                                    "type": "boolean",
                                    "description": "Include indirect dependencies",
                                },
                            },
                            "required": ["files_changed"],
                        }
                    }
                },
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
            "operationId": "createCodebaseUnderstand",
            "description": "Answer questions about the codebase.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": _codebase_params(),
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": "Question about the codebase",
                                },
                                "context": {
                                    "type": "string",
                                    "description": "Additional context for the question",
                                },
                                "file_hints": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Hint files to focus on",
                                },
                            },
                            "required": ["question"],
                        }
                    }
                },
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
            "operationId": "createCodebaseAudit",
            "description": "Run comprehensive code audit.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": _codebase_params(),
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "audit_type": {
                                    "type": "string",
                                    "enum": ["security", "quality", "performance", "full"],
                                    "description": "Type of audit to run",
                                },
                                "severity_threshold": {
                                    "type": "string",
                                    "enum": ["low", "medium", "high", "critical"],
                                    "description": "Minimum severity to report",
                                },
                                "paths": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Paths to audit (empty for full repo)",
                                },
                            },
                        }
                    }
                },
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
            "operationId": "getCodebaseAudit",
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
            "operationId": "createCodebaseAnalyzeLegacy",
            "description": "Alias for codebase analysis.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": _codebase_params(),
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "paths": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Paths to analyze (empty for full repo)",
                                },
                                "depth": {
                                    "type": "integer",
                                    "description": "Analysis depth level",
                                },
                                "include_tests": {
                                    "type": "boolean",
                                    "description": "Include test files in analysis",
                                },
                            },
                        }
                    }
                },
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
            "operationId": "getCodebaseSymbolLegacy",
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
            "operationId": "getCodebaseCallgraphLegacy",
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
            "operationId": "getCodebaseDeadcodeLegacy",
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
            "operationId": "createCodebaseImpactLegacy",
            "description": "Alias for impact analysis.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": _codebase_params(),
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "files_changed": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of changed file paths",
                                },
                                "commit_sha": {
                                    "type": "string",
                                    "description": "Git commit SHA to analyze",
                                },
                                "include_indirect": {
                                    "type": "boolean",
                                    "description": "Include indirect dependencies",
                                },
                            },
                            "required": ["files_changed"],
                        }
                    }
                },
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
            "operationId": "createCodebaseUnderstandLegacy",
            "description": "Alias for code understanding.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": _codebase_params(),
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": "Question about the codebase",
                                },
                                "context": {
                                    "type": "string",
                                    "description": "Additional context for the question",
                                },
                                "file_hints": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Hint files to focus on",
                                },
                            },
                            "required": ["question"],
                        }
                    }
                },
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
            "operationId": "createCodebaseAuditLegacy",
            "description": "Alias for code audit.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": _codebase_params(),
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "audit_type": {
                                    "type": "string",
                                    "enum": ["security", "quality", "performance", "full"],
                                    "description": "Type of audit to run",
                                },
                                "severity_threshold": {
                                    "type": "string",
                                    "enum": ["low", "medium", "high", "critical"],
                                    "description": "Minimum severity to report",
                                },
                                "paths": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Paths to audit (empty for full repo)",
                                },
                            },
                        }
                    }
                },
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
            "operationId": "getCodebaseAuditLegacy",
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
