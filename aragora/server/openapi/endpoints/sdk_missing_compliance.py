"""SDK missing endpoints for Compliance, Policies, Audit, and Privacy.

This module contains endpoint definitions for compliance management, policy
enforcement, audit trails, and privacy controls.
"""

from aragora.server.openapi.endpoints.sdk_missing_core import (
    _ok_response,
    STANDARD_ERRORS,
)

# =============================================================================
# Response Schemas
# =============================================================================

_COMPLIANCE_SUMMARY_SCHEMA = {
    "overall_score": {"type": "number", "description": "Overall compliance score (0-100)"},
    "soc2_status": {
        "type": "string",
        "enum": ["compliant", "non_compliant", "partial", "not_applicable"],
    },
    "gdpr_status": {
        "type": "string",
        "enum": ["compliant", "non_compliant", "partial", "not_applicable"],
    },
    "hipaa_status": {
        "type": "string",
        "enum": ["compliant", "non_compliant", "partial", "not_applicable"],
    },
    "active_violations": {"type": "integer", "description": "Count of active policy violations"},
    "pending_reviews": {"type": "integer", "description": "Count of pending compliance reviews"},
    "last_audit": {"type": "string", "format": "date-time"},
    "next_audit_due": {"type": "string", "format": "date-time"},
}

_POLICY_SCHEMA = {
    "id": {"type": "string", "description": "Unique policy identifier"},
    "name": {"type": "string", "description": "Policy name"},
    "description": {"type": "string"},
    "type": {
        "type": "string",
        "enum": ["access_control", "data_retention", "encryption", "audit", "rate_limit", "custom"],
    },
    "enabled": {"type": "boolean"},
    "priority": {"type": "integer", "description": "Policy evaluation priority"},
    "conditions": {"type": "object", "description": "Policy conditions as JSON"},
    "actions": {"type": "array", "items": {"type": "string"}},
    "created_at": {"type": "string", "format": "date-time"},
    "updated_at": {"type": "string", "format": "date-time"},
}

_POLICY_LIST_SCHEMA = {
    "policies": {"type": "array", "items": {"type": "object"}},
    "total": {"type": "integer"},
    "page": {"type": "integer"},
    "per_page": {"type": "integer"},
}

_VALIDATION_RESULT_SCHEMA = {
    "valid": {"type": "boolean", "description": "Whether the policy is valid"},
    "errors": {"type": "array", "items": {"type": "string"}, "description": "Validation errors"},
    "warnings": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Validation warnings",
    },
    "conflicts": {
        "type": "array",
        "items": {"type": "object"},
        "description": "Conflicting policies",
    },
}

_VIOLATION_SCHEMA = {
    "id": {"type": "string", "description": "Violation identifier"},
    "policy_id": {"type": "string", "description": "Associated policy ID"},
    "policy_name": {"type": "string"},
    "severity": {"type": "string", "enum": ["critical", "high", "medium", "low", "info"]},
    "status": {"type": "string", "enum": ["open", "acknowledged", "resolved", "ignored"]},
    "resource_type": {"type": "string"},
    "resource_id": {"type": "string"},
    "description": {"type": "string"},
    "detected_at": {"type": "string", "format": "date-time"},
    "resolved_at": {"type": "string", "format": "date-time"},
}

_VIOLATION_LIST_SCHEMA = {
    "violations": {"type": "array", "items": {"type": "object"}},
    "total": {"type": "integer"},
    "by_severity": {"type": "object", "description": "Count by severity level"},
}

_AUDIT_ENTRY_SCHEMA = {
    "id": {"type": "string"},
    "action": {"type": "string", "description": "Action performed"},
    "resource_type": {"type": "string"},
    "resource_id": {"type": "string"},
    "actor_id": {"type": "string"},
    "actor_type": {"type": "string", "enum": ["user", "service", "system"]},
    "ip_address": {"type": "string"},
    "user_agent": {"type": "string"},
    "status": {"type": "string", "enum": ["allowed", "denied"]},
    "metadata": {"type": "object"},
    "timestamp": {"type": "string", "format": "date-time"},
}

_AUDIT_HISTORY_SCHEMA = {
    "entries": {"type": "array", "items": {"type": "object"}},
    "total": {"type": "integer"},
    "resource_type": {"type": "string"},
    "resource_id": {"type": "string"},
}

_PRIVACY_PREFERENCES_SCHEMA = {
    "user_id": {"type": "string"},
    "data_retention_days": {"type": "integer"},
    "marketing_consent": {"type": "boolean"},
    "analytics_consent": {"type": "boolean"},
    "third_party_sharing": {"type": "boolean"},
    "data_export_format": {"type": "string", "enum": ["json", "csv", "xml"]},
    "updated_at": {"type": "string", "format": "date-time"},
}

_DELETE_RESULT_SCHEMA = {
    "deleted": {"type": "boolean"},
    "resource_id": {"type": "string"},
    "deleted_at": {"type": "string", "format": "date-time"},
}

_TOGGLE_RESULT_SCHEMA = {
    "policy_id": {"type": "string"},
    "enabled": {"type": "boolean"},
    "updated_at": {"type": "string", "format": "date-time"},
}

# =============================================================================
# Endpoints
# =============================================================================

SDK_MISSING_COMPLIANCE_ENDPOINTS: dict = {
    # Compliance endpoints
    "/api/compliance/summary": {
        "get": {
            "tags": ["Compliance"],
            "summary": "Get compliance summary",
            "description": "Retrieve overall compliance status including SOC2, GDPR, and HIPAA compliance scores.",
            "operationId": "getComplianceSummary",
            "responses": {
                "200": _ok_response("Compliance summary", _COMPLIANCE_SUMMARY_SCHEMA),
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    # Policies endpoints
    "/api/policies": {
        "get": {
            "tags": ["Policies"],
            "summary": "List all policies",
            "description": "Retrieve a paginated list of all configured policies.",
            "operationId": "getPolicies",
            "parameters": [
                {"name": "page", "in": "query", "schema": {"type": "integer", "default": 1}},
                {"name": "per_page", "in": "query", "schema": {"type": "integer", "default": 20}},
                {"name": "type", "in": "query", "schema": {"type": "string"}},
                {"name": "enabled", "in": "query", "schema": {"type": "boolean"}},
            ],
            "responses": {
                "200": _ok_response("Policy list", _POLICY_LIST_SCHEMA),
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
        "post": {
            "tags": ["Policies"],
            "summary": "Create policy",
            "description": "Create a new policy with specified conditions and actions.",
            "operationId": "postPolicies",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["name", "type"],
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "access_control",
                                        "data_retention",
                                        "encryption",
                                        "audit",
                                        "rate_limit",
                                        "custom",
                                    ],
                                },
                                "enabled": {"type": "boolean", "default": True},
                                "priority": {"type": "integer", "default": 100},
                                "conditions": {"type": "object"},
                                "actions": {"type": "array", "items": {"type": "string"}},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Created policy", _POLICY_SCHEMA),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
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
            "summary": "GET violations for policy",
            "operationId": "getPolicyViolationsById",
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
