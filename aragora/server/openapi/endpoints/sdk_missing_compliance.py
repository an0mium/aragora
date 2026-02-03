"""SDK missing endpoints for Compliance, Policies, Audit, and Privacy.

This module contains endpoint definitions for compliance management, policy
enforcement, audit trails, and privacy controls.
"""

from aragora.server.openapi.endpoints.sdk_missing_core import (
    _ok_response,
    STANDARD_ERRORS,
)

SDK_MISSING_COMPLIANCE_ENDPOINTS: dict = {
    # Compliance endpoints
    "/api/compliance/summary": {
        "get": {
            "tags": ["Compliance"],
            "summary": "GET summary",
            "operationId": "getComplianceSummary",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    # Policies endpoints
    "/api/policies": {
        "get": {
            "tags": ["Policies"],
            "summary": "GET policies",
            "operationId": "getPolicies",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "post": {
            "tags": ["Policies"],
            "summary": "POST policies",
            "operationId": "postPolicies",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/policies/validate": {
        "post": {
            "tags": ["Policies"],
            "summary": "POST validate",
            "operationId": "postPoliciesValidate",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/policies/violations": {
        "get": {
            "tags": ["Policies"],
            "summary": "GET violations",
            "operationId": "getPoliciesViolations",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/policies/violations/{id}/resolve": {
        "post": {
            "tags": ["Policies"],
            "summary": "POST resolve",
            "operationId": "postViolationsResolve",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/policies/{id}": {
        "delete": {
            "tags": ["Policies"],
            "summary": "DELETE {id}",
            "operationId": "deletePolicies",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "get": {
            "tags": ["Policies"],
            "summary": "GET {id}",
            "operationId": "getPolicies",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "patch": {
            "tags": ["Policies"],
            "summary": "PATCH {id}",
            "operationId": "patchPolicies",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/policies/{id}/disable": {
        "post": {
            "tags": ["Policies"],
            "summary": "POST disable",
            "operationId": "postPoliciesDisable",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/policies/{id}/enable": {
        "post": {
            "tags": ["Policies"],
            "summary": "POST enable",
            "operationId": "postPoliciesEnable",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/policies/{id}/toggle": {
        "post": {
            "tags": ["Policies"],
            "summary": "POST toggle",
            "operationId": "postPoliciesToggle",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/policies/{id}/violations": {
        "get": {
            "tags": ["Policies"],
            "summary": "GET violations",
            "operationId": "getPoliciesViolations",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    # Audit endpoints
    "/api/v1/audit/denied": {
        "delete": {
            "tags": ["Audit"],
            "summary": "DELETE denied",
            "operationId": "deleteAuditDenied",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
        "post": {
            "tags": ["Audit"],
            "summary": "POST denied",
            "operationId": "postAuditDenied",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
        "put": {
            "tags": ["Audit"],
            "summary": "PUT denied",
            "operationId": "putAuditDenied",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    "/api/v1/audit/resource/{id}/{id}/history": {
        "get": {
            "tags": ["Audit"],
            "summary": "GET history",
            "operationId": "getResourceHistory",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "id2", "in": "path", "required": True, "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
    # Privacy endpoints
    "/api/v1/privacy/account": {
        "delete": {
            "tags": ["Privacy"],
            "summary": "DELETE account",
            "operationId": "deletePrivacyAccount",
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/v1/privacy/preferences": {
        "post": {
            "tags": ["Privacy"],
            "summary": "POST preferences",
            "operationId": "postPrivacyPreferences",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object"}}}},
            "responses": {
                "200": _ok_response("Success", {"success": {"type": "boolean"}}),
            },
        },
    },
}

__all__ = ["SDK_MISSING_COMPLIANCE_ENDPOINTS"]
