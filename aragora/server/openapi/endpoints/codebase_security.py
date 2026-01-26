"""Codebase security endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS, AUTH_REQUIREMENTS


CODEBASE_SECURITY_ENDPOINTS = {
    "/api/v1/codebase/{repo}/scan": {
        "post": {
            "tags": ["Codebase"],
            "summary": "Run dependency vulnerability scan",
            "operationId": "createCodebaseScan",
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
            "operationId": "getCodebaseScanLatest",
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
            "operationId": "getCodebaseScan",
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
            "operationId": "listCodebaseScans",
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
            "operationId": "getCodebaseVulnerabilitie",
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
            "operationId": "getCodebasePackageVulnerabilitie",
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
            "operationId": "getCve",
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
    "/api/v1/codebase/analyze-dependencies": {
        "post": {
            "tags": ["Codebase"],
            "summary": "Analyze dependencies",
            "operationId": "createCodebaseAnalyzeDependencies",
            "description": "Analyze dependency graph and inventory.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/CodebaseDependencyAnalysisRequest"}
                    }
                },
            },
            "responses": {
                "200": _ok_response("Dependency analysis", "CodebaseDependencyAnalysisResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/scan-vulnerabilities": {
        "post": {
            "tags": ["Codebase"],
            "summary": "Scan vulnerabilities",
            "operationId": "createCodebaseScanVulnerabilities",
            "description": "Scan a repository for CVEs.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/CodebaseDependencyScanRequest"}
                    }
                },
            },
            "responses": {
                "200": _ok_response("Vulnerability scan", "CodebaseDependencyScanResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/check-licenses": {
        "post": {
            "tags": ["Codebase"],
            "summary": "Check licenses",
            "operationId": "createCodebaseCheckLicenses",
            "description": "Check license compatibility.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/CodebaseLicenseCheckRequest"}
                    }
                },
            },
            "responses": {
                "200": _ok_response("License check", "CodebaseLicenseCheckResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/sbom": {
        "post": {
            "tags": ["Codebase"],
            "summary": "Generate SBOM",
            "operationId": "createCodebaseSbom",
            "description": "Generate SBOM for a repository.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/CodebaseSBOMRequest"}
                    }
                },
            },
            "responses": {
                "200": _ok_response("SBOM generated", "CodebaseSBOMResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/clear-cache": {
        "post": {
            "tags": ["Codebase"],
            "summary": "Clear dependency cache",
            "operationId": "createCodebaseClearCache",
            "description": "Clear cached dependency analysis results.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "responses": {
                "200": _ok_response("Cache cleared", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/{repo}/scan/secrets": {
        "post": {
            "tags": ["Codebase"],
            "summary": "Trigger secrets scan",
            "operationId": "createCodebaseScanSecret",
            "description": "Trigger secrets scan for a repository.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "repo", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/CodebaseSecretsScanRequest"}
                    }
                },
            },
            "responses": {
                "200": _ok_response("Secrets scan started", "CodebaseSecretsScanStartResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/{repo}/scan/secrets/latest": {
        "get": {
            "tags": ["Codebase"],
            "summary": "Get latest secrets scan",
            "operationId": "getCodebaseScanSecretsLatest",
            "description": "Fetch the latest secrets scan.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "repo", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Secrets scan", "CodebaseSecretsScanResultResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/{repo}/scan/secrets/{scan_id}": {
        "get": {
            "tags": ["Codebase"],
            "summary": "Get secrets scan by ID",
            "operationId": "getCodebaseScanSecret",
            "description": "Fetch a secrets scan result by ID.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "repo", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "scan_id", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Secrets scan", "CodebaseSecretsScanResultResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/{repo}/secrets": {
        "get": {
            "tags": ["Codebase"],
            "summary": "List secrets",
            "operationId": "getCodebaseSecret",
            "description": "List secrets from the latest scan.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "repo", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Secrets list", "CodebaseSecretsListResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/{repo}/scans/secrets": {
        "get": {
            "tags": ["Codebase"],
            "summary": "List secrets scans",
            "operationId": "getCodebaseScansSecret",
            "description": "List secrets scan history.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "repo", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Secrets scan list", "CodebaseSecretsScanListResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/{repo}/scan/sast": {
        "post": {
            "tags": ["Codebase"],
            "summary": "Trigger SAST scan",
            "operationId": "createCodebaseScanSast",
            "description": "Trigger a SAST scan for a repository.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "repo", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "repo_path": {"type": "string"},
                                "rule_sets": {"type": "array", "items": {"type": "string"}},
                                "workspace_id": {"type": "string"},
                            },
                            "required": ["repo_path"],
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("SAST scan started", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/{repo}/scan/sast/{scan_id}": {
        "get": {
            "tags": ["Codebase"],
            "summary": "Get SAST scan status",
            "operationId": "getCodebaseScanSast",
            "description": "Fetch SAST scan status by scan ID.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "repo", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "scan_id", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("SAST scan status", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/{repo}/sast/findings": {
        "get": {
            "tags": ["Codebase"],
            "summary": "List SAST findings",
            "operationId": "getCodebaseSastFinding",
            "description": "List SAST findings for the latest scan.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "repo", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "severity", "in": "query", "schema": {"type": "string"}},
                {"name": "owasp_category", "in": "query", "schema": {"type": "string"}},
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 100}},
                {"name": "offset", "in": "query", "schema": {"type": "integer", "default": 0}},
            ],
            "responses": {
                "200": _ok_response("SAST findings", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/codebase/{repo}/sast/owasp-summary": {
        "get": {
            "tags": ["Codebase"],
            "summary": "OWASP summary",
            "operationId": "getCodebaseSastOwaspSummary",
            "description": "Summarize SAST findings by OWASP category.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {"name": "repo", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("OWASP summary", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
}

__all__ = ["CODEBASE_SECURITY_ENDPOINTS"]
