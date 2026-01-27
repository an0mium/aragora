"""Authentication endpoint definitions for OpenAPI documentation.

Handles user registration, login, session management, MFA, and API keys.
"""

from typing import Any


def _user_schema() -> dict[str, Any]:
    """User object schema."""
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "User ID"},
            "email": {"type": "string", "format": "email"},
            "name": {"type": "string"},
            "role": {"type": "string", "enum": ["user", "admin", "superadmin"]},
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
                "200": {"description": "Logged out successfully"},
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
                "200": {"description": "Token revoked"},
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
                "200": {"description": "Password changed successfully"},
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
                                                    "type": "string",
                                                    "format": "date-time",
                                                    "nullable": True,
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
                                    "type": "string",
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
                "200": {"description": "API key deleted"},
                "401": {"description": "Not authenticated"},
                "404": {"description": "API key not found"},
            },
            "security": [{"bearerAuth": []}],
        },
    },
    # =========================================================================
    # MFA (Multi-Factor Authentication)
    # =========================================================================
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
                "200": {"description": "MFA disabled"},
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
                "200": {"description": "Session revoked"},
                "401": {"description": "Not authenticated"},
                "404": {"description": "Session not found"},
            },
            "security": [{"bearerAuth": []}],
        }
    },
}
