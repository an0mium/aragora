"""Authentication endpoint definitions for OpenAPI documentation.

Handles user registration, login, session management, MFA, API keys, and SCIM provisioning.
"""

from typing import Any

from aragora.server.openapi.helpers import STANDARD_ERRORS


def _user_schema() -> dict[str, Any]:
    """User object schema."""
    return {
        "type": "object",
        "properties": {
            "id": {"type": ["string", "null"], "description": "User ID"},
            "email": {"type": ["string", "null"], "format": "email"},
            "name": {"type": "string"},
            "role": {"type": ["string", "null"], "enum": ["user", "admin", "superadmin"]},
            "mfa_enabled": {"type": "boolean"},
            "created_at": {"type": "string", "format": "date-time"},
        },
    }


def _session_schema() -> dict[str, Any]:
    """Session object schema."""
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Session ID"},
            "user_agent": {"type": "string"},
            "ip_address": {"type": "string"},
            "created_at": {"type": "string", "format": "date-time"},
            "last_active": {"type": "string", "format": "date-time"},
            "current": {"type": "boolean", "description": "Is this the current session"},
        },
    }


def _api_key_metadata_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "prefix": {
                "type": "string",
                "description": "API key prefix used for display and revocation.",
            },
            "name": {"type": "string"},
            "created_at": {"type": ["string", "null"], "format": "date-time"},
            "expires_at": {"type": ["string", "null"], "format": "date-time"},
            "last_used": {"type": ["string", "null"], "format": "date-time"},
        },
    }


def _api_key_list_operation(operation_id: str, summary: str, description: str) -> dict[str, Any]:
    return {
        "tags": ["Authentication", "API Keys"],
        "summary": summary,
        "operationId": operation_id,
        "description": description,
        "responses": {
            "200": {
                "description": "List of API keys returned",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "keys": {
                                    "type": "array",
                                    "items": _api_key_metadata_schema(),
                                },
                                "count": {"type": "integer"},
                            },
                        }
                    }
                },
            },
            "401": STANDARD_ERRORS["401"],
            "404": STANDARD_ERRORS["404"],
            "503": {"description": "Service unavailable"},
        },
        "security": [{"bearerAuth": []}],
    }


def _api_key_create_operation(operation_id: str, summary: str, description: str) -> dict[str, Any]:
    return {
        "tags": ["Authentication", "API Keys"],
        "summary": summary,
        "operationId": operation_id,
        "description": description,
        "requestBody": {
            "required": False,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Optional display name for the generated key.",
                            }
                        },
                    }
                }
            },
        },
        "responses": {
            "200": {
                "description": "API key generated successfully",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "api_key": {"type": "string"},
                                "prefix": {"type": "string"},
                                "name": {"type": "string"},
                                "created_at": {
                                    "type": ["string", "null"],
                                    "format": "date-time",
                                },
                                "expires_at": {
                                    "type": ["string", "null"],
                                    "format": "date-time",
                                },
                                "message": {"type": "string"},
                            },
                        }
                    }
                },
            },
            "401": STANDARD_ERRORS["401"],
            "403": STANDARD_ERRORS["403"],
            "404": STANDARD_ERRORS["404"],
            "503": {"description": "Service unavailable"},
        },
        "security": [{"bearerAuth": []}],
    }


def _api_key_revoke_operation(operation_id: str, summary: str, description: str) -> dict[str, Any]:
    return {
        "tags": ["Authentication", "API Keys"],
        "summary": summary,
        "operationId": operation_id,
        "description": description,
        "responses": {
            "200": {
                "description": "API key revoked successfully",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {"message": {"type": "string"}},
                        }
                    }
                },
            },
            "401": STANDARD_ERRORS["401"],
            "404": STANDARD_ERRORS["404"],
            "503": {"description": "Service unavailable"},
        },
        "security": [{"bearerAuth": []}],
    }


def _api_key_revoke_prefix_operation(
    operation_id: str, summary: str, description: str
) -> dict[str, Any]:
    operation = _api_key_revoke_operation(operation_id, summary, description)
    operation["parameters"] = [
        {
            "name": "prefix",
            "in": "path",
            "required": True,
            "description": "API key prefix to revoke.",
            "schema": {"type": "string"},
        }
    ]
    operation["responses"]["404"] = {"description": "API key not found"}
    return operation


AUTH_ENDPOINTS = {
    # =========================================================================
    # Registration and Login
    # =========================================================================
    "/api/auth/register": {
        "post": {
            "tags": ["Authentication"],
            "summary": "Register new user",
            "operationId": "registerUser",
            "description": "Create a new user account. Returns user info and session tokens.",
            "security": [],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["email", "password"],
                            "properties": {
                                "email": {"type": "string", "format": "email"},
                                "password": {"type": "string", "minLength": 8},
                                "name": {"type": "string"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "201": {
                    "description": "User created successfully",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "user": _user_schema(),
                                    "access_token": {"type": "string"},
                                    "refresh_token": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "400": {"description": "Invalid input or email already exists"},
            },
        }
    },
    "/api/auth/login": {
        "post": {
            "tags": ["Authentication"],
            "summary": "Login user",
            "operationId": "loginUser",
            "description": "Authenticate user with email and password. Returns session tokens. If MFA is enabled, returns mfa_required flag.",
            "security": [],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["email", "password"],
                            "properties": {
                                "email": {"type": "string", "format": "email"},
                                "password": {"type": "string"},
                                "mfa_code": {
                                    "type": "string",
                                    "description": "TOTP code if MFA enabled",
                                },
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Login successful",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "user": _user_schema(),
                                    "access_token": {"type": "string"},
                                    "refresh_token": {"type": "string"},
                                    "mfa_required": {"type": "boolean"},
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Invalid credentials"},
                "403": {"description": "Account locked or disabled"},
            },
        }
    },
    # =========================================================================
    # Session Management
    # =========================================================================
    "/api/auth/logout": {
        "post": {
            "tags": ["Authentication"],
            "summary": "Logout current session",
            "operationId": "logoutUser",
            "description": "Invalidate the current session token.",
            "responses": {
                "200": {
                    "description": "Logged out successfully",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "logged_out": {"type": "boolean"},
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Not authenticated"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/auth/logout-all": {
        "post": {
            "tags": ["Authentication"],
            "summary": "Logout all sessions",
            "operationId": "logoutAllSessions",
            "description": "Invalidate all sessions for the current user across all devices.",
            "responses": {
                "200": {
                    "description": "All sessions invalidated",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "sessions_revoked": {"type": "integer"},
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Not authenticated"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/auth/refresh": {
        "post": {
            "tags": ["Authentication"],
            "summary": "Refresh access token",
            "operationId": "refreshToken",
            "description": "Exchange a refresh token for a new access token.",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["refresh_token"],
                            "properties": {
                                "refresh_token": {"type": "string"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "New tokens issued",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "access_token": {"type": "string"},
                                    "refresh_token": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Invalid or expired refresh token"},
            },
        }
    },
    "/api/auth/revoke": {
        "post": {
            "tags": ["Authentication"],
            "summary": "Revoke a token",
            "operationId": "revokeToken",
            "description": "Revoke a specific access or refresh token.",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["token"],
                            "properties": {
                                "token": {"type": "string"},
                                "token_type": {
                                    "type": "string",
                                    "enum": ["access", "refresh"],
                                    "default": "access",
                                },
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Token revoked",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "revoked": {"type": "boolean"},
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Not authenticated"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/auth/me": {
        "get": {
            "tags": ["Authentication"],
            "summary": "Get current user",
            "operationId": "getCurrentUser",
            "description": "Returns the currently authenticated user's profile.",
            "responses": {
                "200": {
                    "description": "Current user info",
                    "content": {"application/json": {"schema": _user_schema()}},
                },
                "401": {"description": "Not authenticated"},
            },
            "security": [{"bearerAuth": []}],
        },
        "put": {
            "tags": ["Authentication"],
            "summary": "Update current user",
            "operationId": "updateCurrentUser",
            "description": "Update the current user's profile.",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "email": {"type": "string", "format": "email"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "User updated",
                    "content": {"application/json": {"schema": _user_schema()}},
                },
                "401": {"description": "Not authenticated"},
            },
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/auth/password": {
        "post": {
            "tags": ["Authentication"],
            "summary": "Change password",
            "operationId": "changePassword",
            "description": "Change the current user's password. Requires current password.",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["current_password", "new_password"],
                            "properties": {
                                "current_password": {"type": "string"},
                                "new_password": {"type": "string", "minLength": 8},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Password changed successfully",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "changed": {"type": "boolean"},
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Invalid current password"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    # =========================================================================
    # API Keys
    # =========================================================================
    "/api/auth/api-key": {
        "get": {
            "tags": ["Authentication"],
            "summary": "List API keys",
            "operationId": "listApiKeys",
            "description": "List all API keys for the current user.",
            "responses": {
                "200": {
                    "description": "List of API keys",
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
                                                "id": {"type": "string"},
                                                "name": {"type": "string"},
                                                "prefix": {"type": "string"},
                                                "created_at": {
                                                    "type": "string",
                                                    "format": "date-time",
                                                },
                                                "last_used": {
                                                    "type": ["string", "null"],
                                                    "format": "date-time",
                                                },
                                            },
                                        },
                                    },
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Not authenticated"},
            },
            "security": [{"bearerAuth": []}],
        },
        "post": {
            "tags": ["Authentication"],
            "summary": "Create API key",
            "operationId": "createApiKey",
            "description": "Create a new API key. The full key is only shown once.",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["name"],
                            "properties": {
                                "name": {"type": "string"},
                                "expires_at": {
                                    "type": ["string", "null"],
                                    "format": "date-time",
                                    "description": "Optional expiration date",
                                },
                            },
                        }
                    }
                },
            },
            "responses": {
                "201": {
                    "description": "API key created",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "name": {"type": "string"},
                                    "key": {
                                        "type": "string",
                                        "description": "Full API key (shown only once)",
                                    },
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Not authenticated"},
            },
            "security": [{"bearerAuth": []}],
        },
        "delete": {
            "tags": ["Authentication"],
            "summary": "Delete API key",
            "operationId": "deleteApiKey",
            "description": "Delete an API key by ID.",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["key_id"],
                            "properties": {
                                "key_id": {"type": "string"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "API key deleted",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "deleted": {"type": "boolean"},
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Not authenticated"},
                "404": {"description": "API key not found"},
            },
            "security": [{"bearerAuth": []}],
        },
    },
    # =========================================================================
    # MFA (Multi-Factor Authentication)
    # =========================================================================
    "/api/auth/mfa": {
        "post": {
            "tags": ["Authentication", "MFA"],
            "summary": "Run a compatibility MFA action",
            "operationId": "runMfaCompatibilityAction",
            "description": (
                "Compatibility endpoint that dispatches MFA setup, enable, disable, "
                "verify, and backup-code regeneration by action."
            ),
            "security": [{}, {"bearerAuth": []}],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["action"],
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": [
                                        "setup",
                                        "enable",
                                        "disable",
                                        "verify",
                                        "backup-codes",
                                    ],
                                },
                                "code": {"type": "string"},
                                "password": {"type": "string"},
                                "pending_token": {"type": "string"},
                                "method": {"type": "string"},
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Compatibility MFA action completed",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["status"],
                                "properties": {
                                    "status": {"type": "string"},
                                    "message": {"type": ["string", "null"]},
                                    "secret": {"type": ["string", "null"]},
                                    "provisioning_uri": {"type": ["string", "null"]},
                                    "qr_code_uri": {"type": ["string", "null"]},
                                    "backup_codes": {
                                        "type": ["array", "null"],
                                        "items": {"type": "string"},
                                    },
                                    "sessions_invalidated": {"type": ["boolean", "null"]},
                                    "disabled": {"type": ["boolean", "null"]},
                                    "tokens": {
                                        "type": ["object", "null"],
                                        "additionalProperties": True,
                                    },
                                    "access_token": {"type": ["string", "null"]},
                                    "refresh_token": {"type": ["string", "null"]},
                                    "token_type": {"type": ["string", "null"]},
                                    "expires_in": {"type": ["integer", "null"]},
                                    "user": {
                                        "type": ["object", "null"],
                                        "additionalProperties": True,
                                    },
                                    "backup_codes_remaining": {"type": ["integer", "null"]},
                                    "warning": {"type": ["string", "null"]},
                                },
                                "additionalProperties": True,
                            }
                        }
                    },
                },
                **STANDARD_ERRORS,
            },
        }
    },
    "/api/auth/mfa/setup": {
        "post": {
            "tags": ["Authentication", "MFA"],
            "summary": "Setup MFA",
            "operationId": "setupMfa",
            "description": "Initialize MFA setup. Returns a TOTP secret and QR code URL.",
            "responses": {
                "200": {
                    "description": "MFA setup initialized",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "secret": {
                                        "type": "string",
                                        "description": "TOTP secret for manual entry",
                                    },
                                    "qr_code": {
                                        "type": "string",
                                        "description": "Base64 encoded QR code image",
                                    },
                                    "otpauth_url": {
                                        "type": "string",
                                        "description": "OTPAuth URL for authenticator apps",
                                    },
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Not authenticated"},
                "409": {"description": "MFA already enabled"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/auth/mfa/enable": {
        "post": {
            "tags": ["Authentication", "MFA"],
            "summary": "Enable MFA",
            "operationId": "enableMfa",
            "description": "Enable MFA by verifying a TOTP code. Must call /mfa/setup first.",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["code"],
                            "properties": {
                                "code": {
                                    "type": "string",
                                    "description": "6-digit TOTP code",
                                },
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "MFA enabled",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "backup_codes": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "One-time backup codes",
                                    },
                                },
                            }
                        }
                    },
                },
                "400": {"description": "Invalid TOTP code"},
                "401": {"description": "Not authenticated"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/auth/mfa/disable": {
        "post": {
            "tags": ["Authentication", "MFA"],
            "summary": "Disable MFA",
            "operationId": "disableMfa",
            "description": "Disable MFA. Requires current TOTP code or backup code.",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["code"],
                            "properties": {
                                "code": {"type": "string"},
                                "password": {
                                    "type": "string",
                                    "description": "Current password for verification",
                                },
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "MFA disabled",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "disabled": {"type": "boolean"},
                                },
                            }
                        }
                    },
                },
                "400": {"description": "Invalid code"},
                "401": {"description": "Not authenticated"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/auth/mfa/verify": {
        "post": {
            "tags": ["Authentication", "MFA"],
            "summary": "Verify MFA code",
            "operationId": "verifyMfaCode",
            "description": "Verify a TOTP code during login when MFA is required.",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["code"],
                            "properties": {
                                "code": {"type": "string"},
                                "session_token": {
                                    "type": "string",
                                    "description": "Temporary session token from login",
                                },
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "MFA verified, full session issued",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "access_token": {"type": "string"},
                                    "refresh_token": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "400": {"description": "Invalid TOTP code"},
                "401": {"description": "Invalid session token"},
            },
        }
    },
    "/api/auth/mfa/backup-codes": {
        "post": {
            "tags": ["Authentication", "MFA"],
            "summary": "Regenerate backup codes",
            "operationId": "regenerateBackupCodes",
            "description": "Generate new backup codes. Invalidates old backup codes.",
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["code"],
                            "properties": {
                                "code": {
                                    "type": "string",
                                    "description": "Current TOTP code for verification",
                                },
                            },
                        }
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "New backup codes generated",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "backup_codes": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            }
                        }
                    },
                },
                "400": {"description": "Invalid TOTP code"},
                "401": {"description": "Not authenticated"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    # =========================================================================
    # Session Management
    # =========================================================================
    "/api/auth/sessions": {
        "get": {
            "tags": ["Authentication"],
            "summary": "List active sessions",
            "operationId": "listSessions",
            "description": "List all active sessions for the current user.",
            "responses": {
                "200": {
                    "description": "List of sessions",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "sessions": {
                                        "type": "array",
                                        "items": _session_schema(),
                                    },
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Not authenticated"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    "/api/auth/sessions/{session_id}": {
        "delete": {
            "tags": ["Authentication"],
            "summary": "Revoke session",
            "operationId": "revokeSession",
            "description": "Revoke a specific session by ID.",
            "parameters": [
                {
                    "name": "session_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": {
                    "description": "Session revoked",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "revoked": {"type": "boolean"},
                                    "session_id": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "401": {"description": "Not authenticated"},
                "404": {"description": "Session not found"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
    # =========================================================================
    # SCIM v2 - Groups (RFC 7643/7644)
    # =========================================================================
    "/scim/v2/Groups/{group_id}": {
        "get": {
            "tags": ["SCIM"],
            "summary": "Get SCIM group",
            "operationId": "scimGetGroup",
            "description": "Retrieve a SCIM group resource by ID per RFC 7644 Section 3.4.1.",
            "parameters": [
                {
                    "name": "group_id",
                    "in": "path",
                    "required": True,
                    "description": "SCIM group resource ID",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": {
                    "description": "SCIM Group resource",
                    "content": {
                        "application/scim+json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "schemas": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "id": {"type": "string"},
                                    "displayName": {"type": "string"},
                                    "members": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "value": {"type": "string"},
                                                "display": {"type": "string"},
                                                "$ref": {"type": "string"},
                                            },
                                        },
                                    },
                                    "meta": {
                                        "type": "object",
                                        "properties": {
                                            "resourceType": {"type": "string"},
                                            "created": {"type": "string", "format": "date-time"},
                                            "lastModified": {
                                                "type": "string",
                                                "format": "date-time",
                                            },
                                            "location": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
        "put": {
            "tags": ["SCIM"],
            "summary": "Replace SCIM group",
            "operationId": "scimReplaceGroup",
            "description": "Replace a SCIM group resource per RFC 7644 Section 3.5.1. Full replacement of the group resource.",
            "parameters": [
                {
                    "name": "group_id",
                    "in": "path",
                    "required": True,
                    "description": "SCIM group resource ID",
                    "schema": {"type": "string"},
                },
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/scim+json": {
                        "schema": {
                            "type": "object",
                            "required": ["schemas", "displayName"],
                            "properties": {
                                "schemas": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "displayName": {"type": "string"},
                                "members": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "value": {"type": "string"},
                                            "display": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": {
                    "description": "SCIM Group resource replaced",
                    "content": {
                        "application/scim+json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "schemas": {"type": "array", "items": {"type": "string"}},
                                    "id": {"type": "string"},
                                    "displayName": {"type": "string"},
                                },
                            },
                        },
                    },
                },
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
        "patch": {
            "tags": ["SCIM"],
            "summary": "Patch SCIM group",
            "operationId": "scimPatchGroup",
            "description": "Partially update a SCIM group resource per RFC 7644 Section 3.5.2. Supports add, remove, and replace operations.",
            "parameters": [
                {
                    "name": "group_id",
                    "in": "path",
                    "required": True,
                    "description": "SCIM group resource ID",
                    "schema": {"type": "string"},
                },
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/scim+json": {
                        "schema": {
                            "type": "object",
                            "required": ["schemas", "Operations"],
                            "properties": {
                                "schemas": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "Operations": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "required": ["op"],
                                        "properties": {
                                            "op": {
                                                "type": "string",
                                                "enum": ["add", "remove", "replace"],
                                            },
                                            "path": {"type": "string"},
                                            "value": {},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": {
                    "description": "SCIM Group resource patched",
                    "content": {
                        "application/scim+json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "schemas": {"type": "array", "items": {"type": "string"}},
                                    "id": {"type": "string"},
                                    "displayName": {"type": "string"},
                                },
                            },
                        },
                    },
                },
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
        "delete": {
            "tags": ["SCIM"],
            "summary": "Delete SCIM group",
            "operationId": "scimDeleteGroup",
            "description": "Delete a SCIM group resource per RFC 7644 Section 3.6.",
            "parameters": [
                {
                    "name": "group_id",
                    "in": "path",
                    "required": True,
                    "description": "SCIM group resource ID",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "204": {"description": "Group deleted successfully"},
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
    # =========================================================================
    # SCIM v2 - Users (RFC 7643/7644)
    # =========================================================================
    "/scim/v2/Users/{user_id}": {
        "get": {
            "tags": ["SCIM"],
            "summary": "Get SCIM user",
            "operationId": "scimGetUser",
            "description": "Retrieve a SCIM user resource by ID per RFC 7644 Section 3.4.1.",
            "parameters": [
                {
                    "name": "user_id",
                    "in": "path",
                    "required": True,
                    "description": "SCIM user resource ID",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": {
                    "description": "SCIM User resource",
                    "content": {
                        "application/scim+json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "schemas": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "id": {"type": "string"},
                                    "userName": {"type": "string"},
                                    "name": {
                                        "type": "object",
                                        "properties": {
                                            "givenName": {"type": "string"},
                                            "familyName": {"type": "string"},
                                            "formatted": {"type": "string"},
                                        },
                                    },
                                    "emails": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "value": {"type": "string", "format": "email"},
                                                "type": {"type": "string"},
                                                "primary": {"type": "boolean"},
                                            },
                                        },
                                    },
                                    "active": {"type": "boolean"},
                                    "groups": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "value": {"type": "string"},
                                                "display": {"type": "string"},
                                                "$ref": {"type": "string"},
                                            },
                                        },
                                    },
                                    "meta": {
                                        "type": "object",
                                        "properties": {
                                            "resourceType": {"type": "string"},
                                            "created": {"type": "string", "format": "date-time"},
                                            "lastModified": {
                                                "type": "string",
                                                "format": "date-time",
                                            },
                                            "location": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
        "put": {
            "tags": ["SCIM"],
            "summary": "Replace SCIM user",
            "operationId": "scimReplaceUser",
            "description": "Replace a SCIM user resource per RFC 7644 Section 3.5.1. Full replacement of the user resource.",
            "parameters": [
                {
                    "name": "user_id",
                    "in": "path",
                    "required": True,
                    "description": "SCIM user resource ID",
                    "schema": {"type": "string"},
                },
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/scim+json": {
                        "schema": {
                            "type": "object",
                            "required": ["schemas", "userName"],
                            "properties": {
                                "schemas": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "userName": {"type": "string"},
                                "name": {
                                    "type": "object",
                                    "properties": {
                                        "givenName": {"type": "string"},
                                        "familyName": {"type": "string"},
                                    },
                                },
                                "emails": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "value": {"type": "string", "format": "email"},
                                            "type": {"type": "string"},
                                            "primary": {"type": "boolean"},
                                        },
                                    },
                                },
                                "active": {"type": "boolean"},
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": {
                    "description": "SCIM User resource replaced",
                    "content": {
                        "application/scim+json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "schemas": {"type": "array", "items": {"type": "string"}},
                                    "id": {"type": "string"},
                                    "userName": {"type": "string"},
                                    "active": {"type": "boolean"},
                                },
                            },
                        },
                    },
                },
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
        "patch": {
            "tags": ["SCIM"],
            "summary": "Patch SCIM user",
            "operationId": "scimPatchUser",
            "description": "Partially update a SCIM user resource per RFC 7644 Section 3.5.2. Supports add, remove, and replace operations.",
            "parameters": [
                {
                    "name": "user_id",
                    "in": "path",
                    "required": True,
                    "description": "SCIM user resource ID",
                    "schema": {"type": "string"},
                },
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/scim+json": {
                        "schema": {
                            "type": "object",
                            "required": ["schemas", "Operations"],
                            "properties": {
                                "schemas": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "Operations": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "required": ["op"],
                                        "properties": {
                                            "op": {
                                                "type": "string",
                                                "enum": ["add", "remove", "replace"],
                                            },
                                            "path": {"type": "string"},
                                            "value": {},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": {
                    "description": "SCIM User resource patched",
                    "content": {
                        "application/scim+json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "schemas": {"type": "array", "items": {"type": "string"}},
                                    "id": {"type": "string"},
                                    "userName": {"type": "string"},
                                    "active": {"type": "boolean"},
                                },
                            },
                        },
                    },
                },
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
        "delete": {
            "tags": ["SCIM"],
            "summary": "Delete SCIM user",
            "operationId": "scimDeleteUser",
            "description": "Delete a SCIM user resource per RFC 7644 Section 3.6.",
            "parameters": [
                {
                    "name": "user_id",
                    "in": "path",
                    "required": True,
                    "description": "SCIM user resource ID",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "204": {"description": "User deleted successfully"},
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
            },
            "security": [{"bearerAuth": []}],
        },
    },
}

AUTH_ENDPOINTS.update(
    {
        "/api/auth/api-key": {
            "post": _api_key_create_operation(
                "createAuthApiKeyLegacy",
                "Create API key",
                "Create a new API key through the legacy singular auth route.",
            ),
            "delete": _api_key_revoke_operation(
                "revokeAuthApiKeyLegacy",
                "Revoke API key",
                "Revoke the current user's API key through the legacy singular auth route.",
            ),
        },
        "/api/v1/auth/api-key": {
            "post": _api_key_create_operation(
                "createAuthApiKeyLegacyV1",
                "Create API key",
                "Create a new API key through the versioned legacy singular auth route.",
            ),
            "delete": _api_key_revoke_operation(
                "revokeAuthApiKeyLegacyV1",
                "Revoke API key",
                "Revoke the current user's API key through the versioned legacy singular auth route.",
            ),
        },
        "/api/auth/api-keys": {
            "get": _api_key_list_operation(
                "listAuthApiKeys",
                "List API keys",
                "List API keys for the current user. Only prefix and metadata are shown.",
            ),
            "post": _api_key_create_operation(
                "createAuthApiKey",
                "Create API key",
                "Create a new API key for programmatic access. The full key is shown only once.",
            ),
            "delete": _api_key_revoke_operation(
                "revokeAuthApiKey",
                "Revoke API key",
                "Revoke the current user's active API key.",
            ),
        },
        "/api/v1/auth/api-keys": {
            "get": _api_key_list_operation(
                "listAuthApiKeysV1",
                "List API keys",
                "List API keys for the current user. Only prefix and metadata are shown.",
            ),
            "post": _api_key_create_operation(
                "createAuthApiKeyV1",
                "Create API key",
                "Create a new API key for programmatic access. The full key is shown only once.",
            ),
            "delete": _api_key_revoke_operation(
                "revokeAuthApiKeyV1",
                "Revoke API key",
                "Revoke the current user's active API key.",
            ),
        },
        "/api/auth/api-keys/{prefix}": {
            "delete": _api_key_revoke_prefix_operation(
                "revokeAuthApiKeyByPrefix",
                "Revoke API key by prefix",
                "Revoke the current user's API key by prefix.",
            )
        },
        "/api/v1/auth/api-keys/{prefix}": {
            "delete": _api_key_revoke_prefix_operation(
                "revokeAuthApiKeyByPrefixV1",
                "Revoke API key by prefix",
                "Revoke the current user's API key by prefix.",
            )
        },
        "/api/api-keys": {
            "get": _api_key_list_operation(
                "listSettingsApiKeys",
                "List settings API keys",
                "List API keys through the settings-panel alias.",
            ),
            "post": _api_key_create_operation(
                "createSettingsApiKey",
                "Create settings API key",
                "Create a new API key through the settings-panel alias.",
            ),
            "delete": _api_key_revoke_operation(
                "revokeSettingsApiKey",
                "Revoke settings API key",
                "Revoke the current user's active API key through the settings-panel alias.",
            ),
        },
        "/api/v1/api-keys": {
            "get": _api_key_list_operation(
                "listSettingsApiKeysV1",
                "List settings API keys",
                "List API keys through the versioned settings-panel alias.",
            ),
            "post": _api_key_create_operation(
                "createSettingsApiKeyV1",
                "Create settings API key",
                "Create a new API key through the versioned settings-panel alias.",
            ),
            "delete": _api_key_revoke_operation(
                "revokeSettingsApiKeyV1",
                "Revoke settings API key",
                "Revoke the current user's active API key through the versioned settings-panel alias.",
            ),
        },
        "/api/api-keys/{prefix}": {
            "delete": _api_key_revoke_prefix_operation(
                "revokeSettingsApiKeyByPrefix",
                "Revoke settings API key by prefix",
                "Revoke an API key by prefix through the settings-panel alias.",
            )
        },
        "/api/v1/api-keys/{prefix}": {
            "delete": _api_key_revoke_prefix_operation(
                "revokeSettingsApiKeyByPrefixV1",
                "Revoke settings API key by prefix",
                "Revoke an API key by prefix through the versioned settings-panel alias.",
            )
        },
    }
)

AUTH_ENDPOINTS.update(
    {
        "/api/auth/forgot-password": {
            "post": {
                "tags": ["Authentication", "Password"],
                "summary": "Request password reset (legacy route)",
                "operationId": "forgotPasswordLegacy",
                "description": "Compatibility alias for requesting a password reset email.",
                "security": [],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["email"],
                                "properties": {
                                    "email": {"type": "string", "format": "email"},
                                },
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Reset request accepted",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"sent": {"type": "boolean"}},
                                }
                            }
                        },
                    },
                    "400": STANDARD_ERRORS["400"],
                    "503": {"description": "Service unavailable"},
                },
            }
        },
        "/api/auth/reset-password": {
            "post": {
                "tags": ["Authentication", "Password"],
                "summary": "Reset password (legacy route)",
                "operationId": "resetPasswordLegacy",
                "description": "Compatibility alias for resetting a password with a reset token.",
                "security": [],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["token", "new_password"],
                                "properties": {
                                    "token": {"type": "string"},
                                    "new_password": {"type": "string", "minLength": 8},
                                },
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Password reset completed",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"reset": {"type": "boolean"}},
                                }
                            }
                        },
                    },
                    "400": STANDARD_ERRORS["400"],
                    "503": {"description": "Service unavailable"},
                },
            }
        },
        "/api/auth/verify-email": {
            "post": {
                "tags": ["Authentication"],
                "summary": "Verify email address (legacy route)",
                "operationId": "verifyEmailLegacy",
                "description": "Compatibility alias for verifying an email address with a token.",
                "security": [],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["token"],
                                "properties": {
                                    "token": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Email verified",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "message": {"type": "string"},
                                        "access_token": {"type": ["string", "null"]},
                                    },
                                }
                            }
                        },
                    },
                    "400": STANDARD_ERRORS["400"],
                },
            }
        },
        "/api/auth/verify-email/resend": {
            "post": {
                "tags": ["Authentication"],
                "summary": "Resend verification email (legacy route)",
                "operationId": "resendVerifyEmailLegacy",
                "description": "Compatibility alias for resending the verification email.",
                "security": [],
                "requestBody": {
                    "required": False,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "email": {"type": "string", "format": "email"},
                                },
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Verification email resent",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"sent": {"type": "boolean"}},
                                }
                            }
                        },
                    },
                    "400": STANDARD_ERRORS["400"],
                },
            }
        },
        "/api/auth/resend-verification": {
            "post": {
                "tags": ["Authentication"],
                "summary": "Resend verification email",
                "operationId": "resendVerificationLegacy",
                "description": "Legacy resend-verification compatibility route.",
                "security": [],
                "requestBody": {
                    "required": False,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "email": {"type": "string", "format": "email"},
                                },
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Verification resend accepted",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"sent": {"type": "boolean"}},
                                }
                            }
                        },
                    },
                    "400": STANDARD_ERRORS["400"],
                },
            }
        },
        "/api/auth/setup-organization": {
            "post": {
                "tags": ["Authentication", "Onboarding"],
                "summary": "Create organization after signup (legacy route)",
                "operationId": "setupOrganizationLegacy",
                "description": "Compatibility alias for the post-signup organization setup flow.",
                "security": [{}, {"bearerAuth": []}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["name"],
                                "properties": {
                                    "name": {"type": "string"},
                                    "slug": {"type": "string"},
                                    "plan": {"type": "string"},
                                    "billing_email": {"type": "string", "format": "email"},
                                },
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Organization created",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"organization": {"type": "object"}},
                                }
                            }
                        },
                    },
                    "400": STANDARD_ERRORS["400"],
                    "401": STANDARD_ERRORS["401"],
                },
            }
        },
        "/api/auth/invite": {
            "post": {
                "tags": ["Authentication", "Onboarding"],
                "summary": "Invite team member (legacy route)",
                "operationId": "inviteTeamMemberLegacy",
                "description": "Compatibility alias for sending a team invitation.",
                "security": [{"bearerAuth": []}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["email"],
                                "properties": {
                                    "email": {"type": "string", "format": "email"},
                                    "role": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Invitation created",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "invite_token": {"type": "string"},
                                        "invite_url": {"type": "string"},
                                    },
                                }
                            }
                        },
                    },
                    "400": STANDARD_ERRORS["400"],
                    "401": STANDARD_ERRORS["401"],
                    "403": STANDARD_ERRORS["403"],
                },
            }
        },
        "/api/auth/accept-invite": {
            "post": {
                "tags": ["Authentication", "Onboarding"],
                "summary": "Accept team invitation (legacy route)",
                "operationId": "acceptInviteLegacy",
                "description": "Compatibility alias for accepting a team invitation token.",
                "security": [],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["token"],
                                "properties": {
                                    "token": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Invitation accepted",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "organization_id": {"type": "string"},
                                        "role": {"type": "string"},
                                    },
                                }
                            }
                        },
                    },
                    "400": STANDARD_ERRORS["400"],
                },
            }
        },
    }
)
