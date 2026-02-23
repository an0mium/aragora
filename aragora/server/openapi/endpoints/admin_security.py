"""
OpenAPI endpoint definitions for Admin Security.

Security administration endpoints for encryption key management,
health checks, and user impersonation.
"""

from aragora.server.openapi.helpers import (
    STANDARD_ERRORS,
)

ADMIN_SECURITY_ENDPOINTS = {
    "/api/v1/admin/security/status": {
        "get": {
            "tags": ["Admin", "Security"],
            "summary": "Get encryption status",
            "description": """Get encryption and key status information.

**Requires:** `admin.security.status` permission

**Response includes:**
- Crypto library availability
- Active key ID and version
- Key age and rotation recommendations
- Total key count

**Rotation thresholds:**
- `rotation_recommended`: Key older than 60 days
- `rotation_required`: Key older than 90 days""",
            "operationId": "getSecurityStatus",
            "responses": {
                "200": {
                    "description": "Encryption status",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": ["object", "null"],
                                "properties": {
                                    "crypto_available": {
                                        "type": "boolean",
                                        "description": "Whether cryptography library is installed",
                                    },
                                    "active_key_id": {
                                        "type": ["string", "null"],
                                        "description": "ID of the active encryption key",
                                    },
                                    "key_version": {
                                        "type": "integer",
                                        "description": "Version number of active key",
                                    },
                                    "key_age_days": {
                                        "type": "integer",
                                        "description": "Age of active key in days",
                                    },
                                    "key_created_at": {
                                        "type": ["string", "null"],
                                        "format": "date-time",
                                        "description": "Timestamp when key was created",
                                    },
                                    "rotation_recommended": {
                                        "type": "boolean",
                                        "description": "Whether key rotation is recommended (>60 days)",
                                    },
                                    "rotation_required": {
                                        "type": "boolean",
                                        "description": "Whether key rotation is required (>90 days)",
                                    },
                                    "total_keys": {
                                        "type": "integer",
                                        "description": "Total number of encryption keys",
                                    },
                                },
                            }
                        }
                    },
                },
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/admin/security/health": {
        "get": {
            "tags": ["Admin", "Security"],
            "summary": "Check encryption health",
            "description": """Perform comprehensive encryption health checks.

**Requires:** `admin.security.health` permission

**Health checks performed:**
1. Cryptography library availability
2. Encryption service initialization
3. Active key presence and age
4. Encrypt/decrypt round-trip validation

**Status values:**
- `healthy`: All checks passed
- `degraded`: Warnings present (e.g., key aging)
- `unhealthy`: Critical issues found""",
            "operationId": "getSecurityHealth",
            "responses": {
                "200": {
                    "description": "Health check results",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": ["object", "null"],
                                "properties": {
                                    "status": {
                                        "type": ["string", "null"],
                                        "enum": ["healthy", "degraded", "unhealthy"],
                                        "description": "Overall health status",
                                    },
                                    "checks": {
                                        "type": "object",
                                        "properties": {
                                            "crypto_available": {"type": "boolean"},
                                            "service_initialized": {"type": "boolean"},
                                            "active_key": {"type": "boolean"},
                                            "key_age_days": {"type": "integer"},
                                            "key_version": {"type": "integer"},
                                            "round_trip": {"type": "boolean"},
                                        },
                                        "description": "Individual check results",
                                    },
                                    "issues": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Critical issues found",
                                    },
                                    "warnings": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Non-critical warnings",
                                    },
                                },
                            }
                        }
                    },
                },
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/admin/security/keys": {
        "get": {
            "tags": ["Admin", "Security"],
            "summary": "List encryption keys",
            "description": """List all encryption keys (without sensitive key material).

**Requires:** `admin.security.keys` permission

**Audit:** This action is logged for security audit trails.

**Response includes:**
- Key ID, version, and age
- Active key indicator
- Creation timestamp""",
            "operationId": "listSecurityKeys",
            "responses": {
                "200": {
                    "description": "List of encryption keys",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "keys": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "key_id": {"type": "string"},
                                                "version": {"type": "integer"},
                                                "is_active": {"type": "boolean"},
                                                "created_at": {
                                                    "type": ["string", "null"],
                                                    "format": "date-time",
                                                },
                                                "age_days": {"type": "integer"},
                                            },
                                        },
                                    },
                                    "active_key_id": {"type": "string"},
                                    "total_keys": {"type": "integer"},
                                },
                            }
                        }
                    },
                },
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v1/admin/security/rotate-key": {
        "post": {
            "tags": ["Admin", "Security"],
            "summary": "Rotate encryption key",
            "description": """Rotate the encryption key and re-encrypt stored data.

**Requires:** `admin.security.rotate` permission

**Audit:** This action is logged for security audit trails.

**Key rotation process:**
1. Generate new encryption key
2. Re-encrypt data in specified stores
3. Mark old key as inactive

**Safety features:**
- Keys younger than 30 days require `force: true`
- Dry run mode available for previewing changes
- Failed records are tracked for retry""",
            "operationId": "rotateSecurityKey",
            "requestBody": {
                "required": False,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "dry_run": {
                                    "type": "boolean",
                                    "default": False,
                                    "description": "Preview changes without executing",
                                },
                                "stores": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Specific stores to re-encrypt (default: all)",
                                },
                                "force": {
                                    "type": "boolean",
                                    "default": False,
                                    "description": "Force rotation even if key is recent (<30 days)",
                                },
                            },
                        },
                        "example": {
                            "dry_run": True,
                            "force": False,
                        },
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Rotation result",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "success": {"type": "boolean"},
                                    "dry_run": {"type": "boolean"},
                                    "old_key_version": {"type": "integer"},
                                    "new_key_version": {"type": "integer"},
                                    "stores_processed": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "records_reencrypted": {"type": "integer"},
                                    "failed_records": {"type": "integer"},
                                    "duration_seconds": {"type": "number"},
                                    "errors": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            }
                        }
                    },
                },
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    # =========================================================================
    # Compliance Violations
    # =========================================================================
    "/api/v1/compliance/violations/{violation_id}": {
        "get": {
            "tags": ["Admin", "Compliance"],
            "summary": "Get compliance violation",
            "description": (
                "Retrieve details of a specific compliance violation by ID. "
                "Includes violation type, severity, affected resources, and remediation status."
            ),
            "operationId": "getComplianceViolation",
            "parameters": [
                {
                    "name": "violation_id",
                    "in": "path",
                    "required": True,
                    "description": "Compliance violation ID",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": {
                    "description": "Compliance violation details",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "type": {
                                        "type": ["string", "null"],
                                        "description": "Violation type",
                                        "enum": [
                                            "data_retention",
                                            "access_control",
                                            "encryption",
                                            "audit_logging",
                                            "data_residency",
                                            "consent",
                                            "other",
                                        ],
                                    },
                                    "severity": {
                                        "type": ["string", "null"],
                                        "enum": ["low", "medium", "high", "critical"],
                                    },
                                    "status": {
                                        "type": ["string", "null"],
                                        "enum": [
                                            "open",
                                            "acknowledged",
                                            "in_progress",
                                            "resolved",
                                            "dismissed",
                                        ],
                                    },
                                    "description": {"type": "string"},
                                    "affected_resource": {
                                        "type": "object",
                                        "properties": {
                                            "type": {"type": "string"},
                                            "id": {"type": "string"},
                                        },
                                    },
                                    "framework": {
                                        "type": ["string", "null"],
                                        "description": "Compliance framework (e.g. SOC2, GDPR, HIPAA)",
                                    },
                                    "control_id": {
                                        "type": "string",
                                        "description": "Specific control reference",
                                    },
                                    "remediation": {
                                        "type": "object",
                                        "properties": {
                                            "suggested_action": {"type": "string"},
                                            "assigned_to": {
                                                "type": ["string", "null"],
                                            },
                                            "due_date": {
                                                "type": "string",
                                                "format": "date-time",
                                            },
                                        },
                                    },
                                    "detected_at": {
                                        "type": "string",
                                        "format": "date-time",
                                    },
                                    "resolved_at": {
                                        "type": "string",
                                        "format": "date-time",
                                    },
                                },
                            }
                        }
                    },
                },
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
        "put": {
            "tags": ["Admin", "Compliance"],
            "summary": "Update compliance violation",
            "description": (
                "Update a compliance violation's status, assignment, or remediation details. "
                "Used to acknowledge, assign, or resolve violations."
            ),
            "operationId": "updateComplianceViolation",
            "parameters": [
                {
                    "name": "violation_id",
                    "in": "path",
                    "required": True,
                    "description": "Compliance violation ID",
                    "schema": {"type": "string"},
                }
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "status": {
                                    "type": "string",
                                    "enum": [
                                        "acknowledged",
                                        "in_progress",
                                        "resolved",
                                        "dismissed",
                                    ],
                                    "description": "New violation status",
                                },
                                "assigned_to": {
                                    "type": "string",
                                    "description": "User ID to assign remediation to",
                                },
                                "remediation_notes": {
                                    "type": "string",
                                    "description": "Notes on remediation steps taken",
                                },
                                "due_date": {
                                    "type": "string",
                                    "format": "date-time",
                                    "description": "Remediation due date",
                                },
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Violation updated",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "status": {"type": "string"},
                                    "updated_at": {
                                        "type": "string",
                                        "format": "date-time",
                                    },
                                },
                            }
                        }
                    },
                },
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    # =========================================================================
    # V2 Backups
    # =========================================================================
    "/api/v2/backups": {
        "get": {
            "tags": ["Admin", "Backups"],
            "summary": "List backups",
            "description": (
                "List all backups with optional filtering by status and type. "
                "Returns backup metadata including size, duration, and retention policy."
            ),
            "operationId": "listBackups",
            "parameters": [
                {
                    "name": "status",
                    "in": "query",
                    "description": "Filter by backup status",
                    "schema": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "failed"],
                    },
                },
                {
                    "name": "type",
                    "in": "query",
                    "description": "Filter by backup type",
                    "schema": {
                        "type": "string",
                        "enum": ["full", "incremental", "differential"],
                    },
                },
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of results",
                    "schema": {"type": "integer", "default": 50, "maximum": 200},
                },
                {
                    "name": "offset",
                    "in": "query",
                    "description": "Pagination offset",
                    "schema": {"type": "integer", "default": 0},
                },
            ],
            "responses": {
                "200": {
                    "description": "List of backups",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "backups": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "type": {
                                                    "type": "string",
                                                    "enum": [
                                                        "full",
                                                        "incremental",
                                                        "differential",
                                                    ],
                                                },
                                                "status": {"type": "string"},
                                                "size_bytes": {"type": "integer"},
                                                "created_at": {
                                                    "type": "string",
                                                    "format": "date-time",
                                                },
                                                "completed_at": {
                                                    "type": "string",
                                                    "format": "date-time",
                                                },
                                                "retention_days": {"type": "integer"},
                                            },
                                        },
                                    },
                                    "total": {"type": "integer"},
                                },
                            }
                        }
                    },
                },
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
        "post": {
            "tags": ["Admin", "Backups"],
            "summary": "Create backup",
            "description": (
                "Initiate a new backup. Supports full, incremental, and differential backup types. "
                "Returns immediately with a backup ID; use GET to poll for completion."
            ),
            "operationId": "createBackup",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["full", "incremental", "differential"],
                                    "default": "incremental",
                                    "description": "Backup type",
                                },
                                "label": {
                                    "type": "string",
                                    "description": "Optional human-readable label",
                                },
                                "retention_days": {
                                    "type": "integer",
                                    "default": 30,
                                    "description": "Number of days to retain the backup",
                                },
                                "include_knowledge_mound": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Include Knowledge Mound data",
                                },
                            },
                        }
                    }
                },
            },
            "responses": {
                "202": {
                    "description": "Backup initiated",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "backup_id": {"type": "string"},
                                    "type": {"type": "string"},
                                    "status": {"type": "string"},
                                    "started_at": {
                                        "type": "string",
                                        "format": "date-time",
                                    },
                                    "message": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "409": {
                    "description": "Another backup is already in progress",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Error"},
                        },
                    },
                },
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v2/backups/{backup_id}": {
        "get": {
            "tags": ["Admin", "Backups"],
            "summary": "Get backup details",
            "description": (
                "Retrieve detailed information about a specific backup including "
                "status, size, duration, and component breakdown."
            ),
            "operationId": "getBackup",
            "parameters": [
                {
                    "name": "backup_id",
                    "in": "path",
                    "required": True,
                    "description": "Backup ID",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": {
                    "description": "Backup details",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "type": {"type": "string"},
                                    "status": {"type": "string"},
                                    "label": {
                                        "type": ["string", "null"],
                                    },
                                    "size_bytes": {"type": "integer"},
                                    "duration_seconds": {
                                        "type": ["number", "null"],
                                    },
                                    "components": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "status": {"type": "string"},
                                                "size_bytes": {"type": "integer"},
                                            },
                                        },
                                    },
                                    "created_at": {
                                        "type": "string",
                                        "format": "date-time",
                                    },
                                    "completed_at": {
                                        "type": "string",
                                        "format": "date-time",
                                    },
                                    "retention_days": {"type": "integer"},
                                    "expires_at": {
                                        "type": "string",
                                        "format": "date-time",
                                    },
                                },
                            }
                        }
                    },
                },
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
        "delete": {
            "tags": ["Admin", "Backups"],
            "summary": "Delete backup",
            "description": "Delete a specific backup by ID. In-progress backups cannot be deleted.",
            "operationId": "deleteBackup",
            "parameters": [
                {
                    "name": "backup_id",
                    "in": "path",
                    "required": True,
                    "description": "Backup ID",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": {
                    "description": "Backup deleted",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "deleted": {"type": "boolean"},
                                    "backup_id": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "409": {
                    "description": "Cannot delete an in-progress backup",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Error"},
                        },
                    },
                },
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    # =========================================================================
    # V2 Compliance
    # =========================================================================
    "/api/v2/compliance": {
        "get": {
            "tags": ["Admin", "Compliance"],
            "summary": "Get compliance status",
            "description": (
                "Get overall compliance status across all configured frameworks. "
                "Returns a summary of compliance posture including pass/fail counts "
                "and risk score per framework."
            ),
            "operationId": "getComplianceStatus",
            "parameters": [
                {
                    "name": "framework",
                    "in": "query",
                    "description": "Filter by compliance framework",
                    "schema": {
                        "type": "string",
                        "enum": ["soc2", "gdpr", "hipaa", "iso27001", "pci_dss"],
                    },
                },
            ],
            "responses": {
                "200": {
                    "description": "Compliance status",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "overall_status": {
                                        "type": "string",
                                        "enum": ["compliant", "non_compliant", "partial"],
                                    },
                                    "risk_score": {
                                        "type": ["number", "null"],
                                        "description": "Aggregate risk score (0-100)",
                                    },
                                    "frameworks": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "status": {"type": "string"},
                                                "controls_passed": {"type": "integer"},
                                                "controls_failed": {"type": "integer"},
                                                "controls_total": {"type": "integer"},
                                                "last_checked": {
                                                    "type": "string",
                                                    "format": "date-time",
                                                },
                                            },
                                        },
                                    },
                                    "open_violations": {"type": "integer"},
                                    "checked_at": {
                                        "type": "string",
                                        "format": "date-time",
                                    },
                                },
                            }
                        }
                    },
                },
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v2/compliance/{compliance_id}": {
        "get": {
            "tags": ["Admin", "Compliance"],
            "summary": "Get specific compliance check",
            "description": (
                "Retrieve details of a specific compliance check by ID, including "
                "individual control results, evidence collected, and timestamps."
            ),
            "operationId": "getComplianceCheck",
            "parameters": [
                {
                    "name": "compliance_id",
                    "in": "path",
                    "required": True,
                    "description": "Compliance check ID",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": {
                    "description": "Compliance check details",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "framework": {"type": "string"},
                                    "status": {
                                        "type": "string",
                                        "enum": ["passed", "failed", "warning", "skipped"],
                                    },
                                    "control_id": {"type": "string"},
                                    "control_name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "evidence": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "type": {"type": "string"},
                                                "source": {"type": "string"},
                                                "collected_at": {
                                                    "type": "string",
                                                    "format": "date-time",
                                                },
                                            },
                                        },
                                    },
                                    "checked_at": {
                                        "type": "string",
                                        "format": "date-time",
                                    },
                                    "next_check": {
                                        "type": "string",
                                        "format": "date-time",
                                    },
                                },
                            }
                        }
                    },
                },
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    # =========================================================================
    # V2 Disaster Recovery
    # =========================================================================
    "/api/v2/dr": {
        "get": {
            "tags": ["Admin", "Disaster Recovery"],
            "summary": "Get disaster recovery status",
            "description": (
                "Get the current disaster recovery readiness status including "
                "backup health, recovery point objective (RPO), recovery time objective (RTO), "
                "and last drill results."
            ),
            "operationId": "getDRStatus",
            "responses": {
                "200": {
                    "description": "Disaster recovery status",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "status": {
                                        "type": "string",
                                        "enum": ["ready", "degraded", "not_ready"],
                                    },
                                    "rpo_hours": {
                                        "type": "number",
                                        "description": "Recovery Point Objective in hours",
                                    },
                                    "rto_hours": {
                                        "type": "number",
                                        "description": "Recovery Time Objective in hours",
                                    },
                                    "last_backup": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "completed_at": {
                                                "type": "string",
                                                "format": "date-time",
                                            },
                                            "status": {"type": "string"},
                                        },
                                    },
                                    "last_drill": {
                                        "type": ["object", "null"],
                                        "properties": {
                                            "id": {"type": "string"},
                                            "completed_at": {
                                                "type": "string",
                                                "format": "date-time",
                                            },
                                            "result": {"type": "string"},
                                            "recovery_time_seconds": {"type": "number"},
                                        },
                                    },
                                    "plans": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "name": {"type": "string"},
                                                "status": {"type": "string"},
                                            },
                                        },
                                    },
                                    "checked_at": {
                                        "type": "string",
                                        "format": "date-time",
                                    },
                                },
                            }
                        }
                    },
                },
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v2/dr/{plan_id}": {
        "get": {
            "tags": ["Admin", "Disaster Recovery"],
            "summary": "Get DR plan",
            "description": (
                "Retrieve a specific disaster recovery plan including its configuration, "
                "schedule, component coverage, and execution history."
            ),
            "operationId": "getDRPlan",
            "parameters": [
                {
                    "name": "plan_id",
                    "in": "path",
                    "required": True,
                    "description": "DR plan ID",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": {
                    "description": "DR plan details",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "status": {
                                        "type": "string",
                                        "enum": ["active", "inactive", "testing"],
                                    },
                                    "components": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "priority": {"type": "integer"},
                                                "recovery_strategy": {"type": "string"},
                                            },
                                        },
                                    },
                                    "schedule": {
                                        "type": "object",
                                        "properties": {
                                            "backup_frequency": {"type": "string"},
                                            "drill_frequency": {"type": "string"},
                                            "next_drill": {
                                                "type": "string",
                                                "format": "date-time",
                                            },
                                        },
                                    },
                                    "last_execution": {
                                        "type": ["object", "null"],
                                        "properties": {
                                            "executed_at": {
                                                "type": "string",
                                                "format": "date-time",
                                            },
                                            "result": {"type": "string"},
                                            "duration_seconds": {"type": "number"},
                                        },
                                    },
                                    "created_at": {
                                        "type": "string",
                                        "format": "date-time",
                                    },
                                    "updated_at": {
                                        "type": "string",
                                        "format": "date-time",
                                    },
                                },
                            }
                        }
                    },
                },
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
        "post": {
            "tags": ["Admin", "Disaster Recovery"],
            "summary": "Execute DR plan",
            "description": (
                "Execute a disaster recovery plan. This can be a drill (test) execution "
                "or an actual recovery operation. Drill mode is the default and recommended "
                "for regular testing."
            ),
            "operationId": "executeDRPlan",
            "parameters": [
                {
                    "name": "plan_id",
                    "in": "path",
                    "required": True,
                    "description": "DR plan ID",
                    "schema": {"type": "string"},
                }
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "mode": {
                                    "type": "string",
                                    "enum": ["drill", "recovery"],
                                    "default": "drill",
                                    "description": "Execution mode (drill for testing, recovery for actual restore)",
                                },
                                "components": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Specific components to recover (default: all)",
                                },
                                "backup_id": {
                                    "type": "string",
                                    "description": "Specific backup to restore from (default: latest)",
                                },
                                "notify": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Send notifications on completion",
                                },
                            },
                        }
                    }
                },
            },
            "responses": {
                "202": {
                    "description": "DR plan execution started",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "execution_id": {"type": "string"},
                                    "plan_id": {"type": "string"},
                                    "mode": {"type": "string"},
                                    "status": {"type": "string"},
                                    "started_at": {
                                        "type": "string",
                                        "format": "date-time",
                                    },
                                    "message": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "409": {
                    "description": "Another DR execution is already in progress",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Error"},
                        },
                    },
                },
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    # =========================================================================
    # Impersonation
    # =========================================================================
    "/api/v1/admin/impersonate/{user_id}": {
        "post": {
            "tags": ["Admin", "Security"],
            "summary": "Impersonate user",
            "description": """Create an impersonation token to act as another user.

**Requires:** `admin.users.impersonate` permission

**Audit:** This action is logged with full audit trail including:
- Admin performing the impersonation
- Target user being impersonated
- Timestamp and IP address

**Security notes:**
- Impersonation tokens have limited validity
- All actions during impersonation are tracked
- Cannot impersonate other admins without explicit permission""",
            "operationId": "impersonateUser",
            "parameters": [
                {
                    "name": "user_id",
                    "in": "path",
                    "required": True,
                    "description": "ID of the user to impersonate",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": {
                    "description": "Impersonation token",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "token": {
                                        "type": "string",
                                        "description": "Impersonation JWT token",
                                    },
                                    "expires_at": {
                                        "type": "string",
                                        "format": "date-time",
                                        "description": "Token expiration timestamp",
                                    },
                                    "target_user": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "email": {"type": "string"},
                                            "name": {"type": "string"},
                                        },
                                    },
                                },
                            }
                        }
                    },
                },
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
}
