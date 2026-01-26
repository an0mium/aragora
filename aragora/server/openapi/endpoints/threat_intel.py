"""Threat intelligence endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS, AUTH_REQUIREMENTS

THREAT_INTEL_ENDPOINTS = {
    "/api/v1/threat/url": {
        "post": {
            "tags": ["Threat Intel"],
            "summary": "Scan URL",
            "operationId": "createThreatUrl",
            "description": "Check a URL against threat intel feeds.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Threat result", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/threat/urls": {
        "post": {
            "tags": ["Threat Intel"],
            "summary": "Batch scan URLs",
            "operationId": "createThreatUrls",
            "description": "Check multiple URLs for threats.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Threat results", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/threat/ip/{ip_address}": {
        "get": {
            "tags": ["Threat Intel"],
            "summary": "IP reputation",
            "operationId": "getThreatIp",
            "description": "Check IP reputation.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {
                    "name": "ip_address",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("IP result", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/threat/ips": {
        "post": {
            "tags": ["Threat Intel"],
            "summary": "Batch IP reputation",
            "operationId": "createThreatIps",
            "description": "Check multiple IPs for reputation.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("IP results", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/threat/hash/{hash_value}": {
        "get": {
            "tags": ["Threat Intel"],
            "summary": "File hash lookup",
            "operationId": "getThreatHash",
            "description": "Check a file hash for malware.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "parameters": [
                {
                    "name": "hash_value",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("Hash result", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/threat/hashes": {
        "post": {
            "tags": ["Threat Intel"],
            "summary": "Batch hash lookup",
            "operationId": "createThreatHashes",
            "description": "Check multiple hashes for malware.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Hash results", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/threat/email": {
        "post": {
            "tags": ["Threat Intel"],
            "summary": "Scan email content",
            "operationId": "createThreatEmail",
            "description": "Scan email body for threats.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Email scan", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/threat/status": {
        "get": {
            "tags": ["Threat Intel"],
            "summary": "Service status",
            "operationId": "listThreatStatus",
            "description": "Get threat intel service status.",
            "security": AUTH_REQUIREMENTS["required"]["security"],
            "responses": {
                "200": _ok_response("Status", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
}


__all__ = ["THREAT_INTEL_ENDPOINTS"]
