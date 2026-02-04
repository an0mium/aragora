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
    "/api/compliance/summary": {
        "get": {
            "tags": ["Compliance"],
            "summary": "Get compliance summary",
            "description": "Retrieve overall compliance status including SOC2, GDPR, and HIPAA scores.",
            "operationId": "getComplianceSummary",
            "responses": {
                "200": _ok_response("Compliance summary", _COMPLIANCE_SUMMARY_SCHEMA),
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
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
                                "type": {"type": "string"},
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
            "summary": "Validate policy configuration",
            "description": "Validate a policy configuration without creating it.",
            "operationId": "postPoliciesValidate",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["name", "type"],
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "conditions": {"type": "object"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Validation result", _VALIDATION_RESULT_SCHEMA),
                "400": STANDARD_ERRORS["400"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/policies/violations": {
        "get": {
            "tags": ["Policies"],
            "summary": "List all policy violations",
            "description": "Retrieve all active policy violations across the system.",
            "operationId": "getPoliciesViolations",
            "parameters": [
                {"name": "severity", "in": "query", "schema": {"type": "string"}},
                {"name": "status", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Violation list", _VIOLATION_LIST_SCHEMA),
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/policies/violations/{id}/resolve": {
        "post": {
            "tags": ["Policies"],
            "summary": "Resolve policy violation",
            "description": "Mark a policy violation as resolved.",
            "operationId": "postViolationsResolve",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "resolution_notes": {"type": "string"},
                                "resolution_type": {"type": "string"},
                            },
                        }
                    }
                }
            },
            "responses": {
                "200": _ok_response("Resolved violation", _VIOLATION_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/policies/{id}": {
        "delete": {
            "tags": ["Policies"],
            "summary": "Delete policy",
            "description": "Delete a policy by ID.",
            "operationId": "deletePolicies",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Policy deleted", _DELETE_RESULT_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
        "get": {
            "tags": ["Policies"],
            "summary": "Get policy by ID",
            "description": "Retrieve a specific policy by its identifier.",
            "operationId": "getPolicyById",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Policy details", _POLICY_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
        "patch": {
            "tags": ["Policies"],
            "summary": "Update policy",
            "description": "Partially update a policy configuration.",
            "operationId": "patchPolicies",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "enabled": {"type": "boolean"},
                            },
                        }
                    }
                }
            },
            "responses": {
                "200": _ok_response("Updated policy", _POLICY_SCHEMA),
                "400": STANDARD_ERRORS["400"],
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/policies/{id}/disable": {
        "post": {
            "tags": ["Policies"],
            "summary": "Disable policy",
            "description": "Disable a policy without deleting it.",
            "operationId": "postPoliciesDisable",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Policy disabled", _TOGGLE_RESULT_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/policies/{id}/enable": {
        "post": {
            "tags": ["Policies"],
            "summary": "Enable policy",
            "description": "Enable a previously disabled policy.",
            "operationId": "postPoliciesEnable",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Policy enabled", _TOGGLE_RESULT_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/policies/{id}/toggle": {
        "post": {
            "tags": ["Policies"],
            "summary": "Toggle policy enabled state",
            "description": "Toggle a policy between enabled and disabled states.",
            "operationId": "postPoliciesToggle",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            "responses": {
                "200": _ok_response("Policy toggled", _TOGGLE_RESULT_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/policies/{id}/violations": {
        "get": {
            "tags": ["Policies"],
            "summary": "Get violations for policy",
            "description": "Retrieve all violations associated with a specific policy.",
            "operationId": "getPolicyViolationsById",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
                {"name": "status", "in": "query", "schema": {"type": "string"}},
            ],
            "responses": {
                "200": _ok_response("Violations for policy", _VIOLATION_LIST_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/audit/denied": {
        "delete": {
            "tags": ["Audit"],
            "summary": "Clear denied audit logs",
            "description": "Clear audit logs of denied access attempts.",
            "operationId": "deleteAuditDenied",
            "parameters": [
                {
                    "name": "before",
                    "in": "query",
                    "schema": {"type": "string", "format": "date-time"},
                },
            ],
            "responses": {
                "200": _ok_response("Cleared audit logs", {"deleted_count": {"type": "integer"}}),
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
        "post": {
            "tags": ["Audit"],
            "summary": "Record denied access",
            "description": "Record a denied access attempt in the audit log.",
            "operationId": "postAuditDenied",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["action", "resource_type", "resource_id"],
                            "properties": {
                                "action": {"type": "string"},
                                "resource_type": {"type": "string"},
                                "resource_id": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Audit entry created", _AUDIT_ENTRY_SCHEMA),
                "400": STANDARD_ERRORS["400"],
            },
            "security": [{"bearerAuth": []}],
        },
        "put": {
            "tags": ["Audit"],
            "summary": "Update denied audit configuration",
            "description": "Update configuration for denied access audit logging.",
            "operationId": "putAuditDenied",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "retention_days": {"type": "integer"},
                                "alert_threshold": {"type": "integer"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Configuration updated", {"success": {"type": "boolean"}}),
                "400": STANDARD_ERRORS["400"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/audit/resource/{resource_type}/{resource_id}/history": {
        "get": {
            "tags": ["Audit"],
            "summary": "Get resource audit history",
            "description": "Retrieve complete audit history for a specific resource.",
            "operationId": "getResourceHistory",
            "parameters": [
                {
                    "name": "resource_type",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {
                    "name": "resource_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                },
                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 100}},
            ],
            "responses": {
                "200": _ok_response("Audit history", _AUDIT_HISTORY_SCHEMA),
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/privacy/account": {
        "delete": {
            "tags": ["Privacy"],
            "summary": "Delete account data (GDPR)",
            "description": "Delete all personal data associated with the user's account.",
            "operationId": "deletePrivacyAccount",
            "parameters": [
                {
                    "name": "confirmation",
                    "in": "query",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response(
                    "Account data deleted",
                    {
                        "deleted": {"type": "boolean"},
                        "completion_date": {"type": "string", "format": "date-time"},
                    },
                ),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/privacy/preferences": {
        "post": {
            "tags": ["Privacy"],
            "summary": "Update privacy preferences",
            "description": "Update user's privacy preferences including consent settings.",
            "operationId": "postPrivacyPreferences",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "marketing_consent": {"type": "boolean"},
                                "analytics_consent": {"type": "boolean"},
                                "third_party_sharing": {"type": "boolean"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": _ok_response("Preferences updated", _PRIVACY_PREFERENCES_SCHEMA),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
}

__all__ = ["SDK_MISSING_COMPLIANCE_ENDPOINTS"]
