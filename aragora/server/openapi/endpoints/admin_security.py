"""
OpenAPI endpoint definitions for Admin Security.

Security administration endpoints for encryption key management,
health checks, and user impersonation.
"""

from aragora.server.openapi.helpers import (
    STANDARD_ERRORS,
)

_BACKUP_RECORD_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "created_at": {"type": "string", "format": "date-time"},
        "backup_type": {
            "type": "string",
            "enum": ["full", "incremental", "differential"],
        },
        "status": {
            "type": "string",
            "enum": [
                "pending",
                "in_progress",
                "completed",
                "verified",
                "failed",
                "expired",
            ],
        },
        "source_path": {"type": "string"},
        "backup_path": {"type": "string"},
        "size_bytes": {"type": "integer"},
        "compressed_size_bytes": {"type": "integer"},
        "checksum": {"type": ["string", "null"]},
        "row_counts": {
            "type": "object",
            "additionalProperties": {"type": "integer"},
        },
        "tables": {
            "type": "array",
            "items": {"type": "string"},
        },
        "duration_seconds": {"type": ["number", "null"]},
        "verified": {"type": "boolean"},
        "verified_at": {"type": ["string", "null"], "format": "date-time"},
        "restore_tested": {"type": "boolean"},
        "error": {"type": ["string", "null"]},
        "storage_backend": {"type": "string"},
        "encryption_key_id": {"type": ["string", "null"]},
        "metadata": {
            "type": "object",
            "additionalProperties": True,
        },
        "schema_hash": {"type": "string"},
        "table_checksums": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "foreign_keys": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "indexes": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    },
}

_BACKUP_LIST_SCHEMA = {
    "type": "object",
    "properties": {
        "backups": {
            "type": "array",
            "items": _BACKUP_RECORD_SCHEMA,
        },
        "pagination": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer"},
                "offset": {"type": "integer"},
                "total": {"type": "integer"},
                "has_more": {"type": "boolean"},
            },
        },
    },
}

_BACKUP_STATS_SCHEMA = {
    "type": "object",
    "properties": {
        "stats": {
            "type": "object",
            "properties": {
                "total_backups": {"type": "integer"},
                "verified_backups": {"type": "integer"},
                "failed_backups": {"type": "integer"},
                "total_size_bytes": {"type": "integer"},
                "total_size_mb": {"type": "number"},
                "latest_backup": {
                    "type": ["object", "null"],
                    "properties": _BACKUP_RECORD_SCHEMA["properties"],
                },
                "retention_policy": {
                    "type": "object",
                    "properties": {
                        "keep_daily": {"type": "integer"},
                        "keep_weekly": {"type": "integer"},
                        "keep_monthly": {"type": "integer"},
                        "min_backups": {"type": "integer"},
                    },
                },
            },
        },
        "generated_at": {"type": "string", "format": "date-time"},
    },
}

_DR_STATUS_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["healthy", "warning", "critical"],
        },
        "readiness_score": {"type": "integer"},
        "backup_status": {
            "type": "object",
            "properties": {
                "total_backups": {"type": "integer"},
                "verified_backups": {"type": "integer"},
                "failed_backups": {"type": "integer"},
                "latest_backup": {
                    "type": ["object", "null"],
                    "properties": _BACKUP_RECORD_SCHEMA["properties"],
                },
                "hours_since_backup": {"type": ["number", "null"]},
            },
        },
        "rpo_status": {
            "type": "object",
            "properties": {
                "target_hours": {"type": "integer"},
                "compliant": {"type": "boolean"},
                "current_hours": {"type": ["number", "null"]},
            },
        },
        "issues": {
            "type": "array",
            "items": {"type": "string"},
        },
        "recommendations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "checked_at": {"type": "string", "format": "date-time"},
    },
}

_DR_OBJECTIVES_SCHEMA = {
    "type": "object",
    "properties": {
        "rpo": {
            "type": "object",
            "properties": {
                "target_hours": {"type": "integer"},
                "current_hours": {"type": ["number", "null"]},
                "compliant": {"type": "boolean"},
                "violations_last_7_days": {"type": "integer"},
            },
        },
        "rto": {
            "type": "object",
            "properties": {
                "target_minutes": {"type": "integer"},
                "estimated_minutes": {"type": ["number", "null"]},
                "compliant": {"type": "boolean"},
            },
        },
        "backup_coverage": {
            "type": "object",
            "properties": {
                "total_backups": {"type": "integer"},
                "backups_last_7_days": {"type": "integer"},
                "latest_backup": {
                    "type": ["object", "null"],
                    "properties": _BACKUP_RECORD_SCHEMA["properties"],
                },
            },
        },
        "generated_at": {"type": "string", "format": "date-time"},
    },
}

_DR_DRILL_SCHEMA = {
    "type": "object",
    "properties": {
        "drill_id": {"type": "string"},
        "drill_type": {
            "type": "string",
            "enum": ["restore_test", "full_recovery_sim", "failover_test"],
        },
        "backup_id": {"type": "string"},
        "started_at": {"type": "string", "format": "date-time"},
        "completed_at": {"type": "string", "format": "date-time"},
        "duration_seconds": {"type": "number"},
        "success": {"type": "boolean"},
        "error": {"type": ["string", "null"]},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step": {"type": "string"},
                    "status": {"type": "string"},
                    "details": {
                        "type": "object",
                        "additionalProperties": True,
                    },
                },
            },
        },
    },
}

_DR_VALIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "valid": {"type": "boolean"},
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "status": {"type": "string"},
                    "details": {"type": "string"},
                    "recommendation": {"type": ["string", "null"]},
                },
            },
        },
    },
}

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
                                    "duration_seconds": {"type": ["number", "null"]},
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
                                                "type": ["string", "null"],
                                                "format": "date-time",
                                            },
                                        },
                                    },
                                    "detected_at": {
                                        "type": "string",
                                        "format": "date-time",
                                    },
                                    "resolved_at": {
                                        "type": ["string", "null"],
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
                                    "type": ["string", "null"],
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
                "List backups for the Backup & DR admin surface with filtering and pagination "
                "that match the live handler contract."
            ),
            "operationId": "listBackups",
            "parameters": [
                {
                    "name": "status",
                    "in": "query",
                    "description": "Filter by backup status",
                    "schema": {
                        "type": "string",
                        "enum": [
                            "pending",
                            "in_progress",
                            "completed",
                            "verified",
                            "failed",
                            "expired",
                        ],
                    },
                },
                {
                    "name": "source",
                    "in": "query",
                    "description": "Filter by source database path",
                    "schema": {"type": "string"},
                },
                {
                    "name": "since",
                    "in": "query",
                    "description": "Filter by backups created since the given ISO timestamp",
                    "schema": {
                        "type": "string",
                        "format": "date-time",
                    },
                },
                {
                    "name": "backup_type",
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
                    "schema": {"type": "integer", "default": 20, "maximum": 100},
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
                            "schema": _BACKUP_LIST_SCHEMA,
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
        "post": {
            "tags": ["Admin", "Backups"],
            "summary": "Create backup",
            "description": (
                "Create a new backup using an optional source override and the canonical "
                "default source path when none is provided."
            ),
            "operationId": "createBackup",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "source_path": {
                                    "type": ["string", "null"],
                                    "description": (
                                        "Optional database path to back up. "
                                        "When omitted, the server uses its default source."
                                    ),
                                },
                                "backup_type": {
                                    "type": "string",
                                    "enum": ["full", "incremental", "differential"],
                                    "default": "full",
                                    "description": "Backup type",
                                },
                                "metadata": {
                                    "type": "object",
                                    "additionalProperties": True,
                                    "description": "Additional metadata stored with the backup",
                                },
                            },
                        }
                    }
                },
            },
            "responses": {
                "201": {
                    "description": "Backup created",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "backup": _BACKUP_RECORD_SCHEMA,
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
                "500": STANDARD_ERRORS["500"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/v2/backups/stats": {
        "get": {
            "tags": ["Admin", "Backups"],
            "summary": "Get backup statistics",
            "description": (
                "Return aggregate backup counts, retention policy settings, "
                "and the latest known backup record for the admin Backup & DR page."
            ),
            "operationId": "getBackupStats",
            "responses": {
                "200": {
                    "description": "Backup statistics",
                    "content": {
                        "application/json": {
                            "schema": _BACKUP_STATS_SCHEMA,
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
                            "schema": _BACKUP_RECORD_SCHEMA,
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
                                    "message": {"type": "string"},
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
                                        "type": ["string", "null"],
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
    "/api/v2/dr/status": {
        "get": {
            "tags": ["Admin", "Disaster Recovery"],
            "summary": "Get disaster recovery status",
            "description": (
                "Get the current disaster recovery readiness score and supporting "
                "backup health metrics for the admin Backup & DR dashboard."
            ),
            "operationId": "getDRStatus",
            "responses": {
                "200": {
                    "description": "Disaster recovery status",
                    "content": {
                        "application/json": {
                            "schema": _DR_STATUS_SCHEMA,
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
    "/api/v2/dr/objectives": {
        "get": {
            "tags": ["Admin", "Disaster Recovery"],
            "summary": "Get DR objectives",
            "description": (
                "Return current RPO/RTO compliance metrics and recent backup coverage "
                "for the Backup & DR dashboard."
            ),
            "operationId": "getDRObjectives",
            "responses": {
                "200": {
                    "description": "Current DR objectives and compliance status",
                    "content": {
                        "application/json": {
                            "schema": _DR_OBJECTIVES_SCHEMA,
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
    "/api/v2/dr/drill": {
        "post": {
            "tags": ["Admin", "Disaster Recovery"],
            "summary": "Run DR drill",
            "description": (
                "Run a simulated disaster-recovery drill using the latest verified backup "
                "or a caller-provided backup ID."
            ),
            "operationId": "runDRDrill",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "backup_id": {
                                    "type": ["string", "null"],
                                    "description": "Optional backup ID to use for the drill",
                                },
                                "drill_type": {
                                    "type": "string",
                                    "enum": [
                                        "restore_test",
                                        "full_recovery_sim",
                                        "failover_test",
                                    ],
                                    "default": "restore_test",
                                    "description": "Type of drill to run",
                                },
                                "target_path": {
                                    "type": ["string", "null"],
                                    "description": "Optional restore target path for dry-run drills",
                                },
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "DR drill completed",
                    "content": {
                        "application/json": {
                            "schema": _DR_DRILL_SCHEMA,
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
    "/api/v2/dr/validate": {
        "post": {
            "tags": ["Admin", "Disaster Recovery"],
            "summary": "Validate DR configuration",
            "description": (
                "Validate storage access, RBAC permissions, encryption settings, "
                "and recent backup coverage for the DR configuration."
            ),
            "operationId": "validateDRConfiguration",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "check_storage": {
                                    "type": "boolean",
                                    "default": True,
                                },
                                "check_permissions": {
                                    "type": "boolean",
                                    "default": True,
                                },
                                "check_encryption": {
                                    "type": "boolean",
                                    "default": True,
                                },
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "DR configuration validation results",
                    "content": {
                        "application/json": {
                            "schema": _DR_VALIDATE_SCHEMA,
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
