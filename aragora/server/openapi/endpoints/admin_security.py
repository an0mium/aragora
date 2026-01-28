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
                                "type": "object",
                                "properties": {
                                    "crypto_available": {
                                        "type": "boolean",
                                        "description": "Whether cryptography library is installed",
                                    },
                                    "active_key_id": {
                                        "type": "string",
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
                                        "type": "string",
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
                                "type": "object",
                                "properties": {
                                    "status": {
                                        "type": "string",
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
                                                    "type": "string",
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
