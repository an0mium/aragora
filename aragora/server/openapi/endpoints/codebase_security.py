"""Codebase security endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS, AUTH_REQUIREMENTS


CODEBASE_SECURITY_ENDPOINTS = {
    "/api/v1/codebase/{repo}/scan": {
        "post": {
            "tags": ["Codebase"],
            "summary": "Run dependency vulnerability scan",
            "description": "Trigger a dependency vulnerability scan for a repository.",
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
                                "branch": {"type": "string"},
                                "commit_sha": {"type": "string"},
                                "workspace_id": {"type": "string"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Scan started", "CodebaseScanStartResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/{repo}/scan/latest": {
        "get": {
            "tags": ["Codebase"],
            "summary": "Get latest scan",
            "description": "Fetch the most recent dependency scan result.",
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
                "200": _ok_response("Latest scan", "CodebaseScanResultResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/{repo}/scan/{scan_id}": {
        "get": {
            "tags": ["Codebase"],
            "summary": "Get scan by ID",
            "description": "Fetch a specific scan result by scan ID.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "repo",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "scan_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("Scan result", "CodebaseScanResultResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/{repo}/scans": {
        "get": {
            "tags": ["Codebase"],
            "summary": "List scans",
            "description": "List scan history for a repository.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "repo",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "status",
                    "in": "query",
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
                "200": _ok_response("Scan list", "CodebaseScanListResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/{repo}/vulnerabilities": {
        "get": {
            "tags": ["Codebase"],
            "summary": "List vulnerabilities",
            "description": "List vulnerabilities from the latest scan.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "repo",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "severity",
                    "in": "query",
                    "schema": {"type": "string"},
                },
                {
                    "name": "package",
                    "in": "query",
                    "schema": {"type": "string"},
                },
                {
                    "name": "ecosystem",
                    "in": "query",
                    "schema": {"type": "string"},
                },
                {
                    "name": "limit",
                    "in": "query",
                    "schema": {"type": "integer", "default": 100},
                },
                {
                    "name": "offset",
                    "in": "query",
                    "schema": {"type": "integer", "default": 0},
                },
            ],
            "responses": {
                "200": _ok_response("Vulnerability list", "CodebaseVulnerabilityListResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/package/{ecosystem}/{package}/vulnerabilities": {
        "get": {
            "tags": ["Codebase"],
            "summary": "Query package vulnerabilities",
            "description": "Query advisories for a package and ecosystem.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "ecosystem",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "package",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "version",
                    "in": "query",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response(
                    "Package vulnerabilities", "CodebasePackageVulnerabilityResponse"
                ),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/cve/{cve_id}": {
        "get": {
            "tags": ["Codebase"],
            "summary": "Get CVE details",
            "description": "Fetch CVE details from vulnerability databases.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "cve_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("CVE details", "CodebaseCVEResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
}

__all__ = ["CODEBASE_SECURITY_ENDPOINTS"]
