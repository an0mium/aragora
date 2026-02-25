"""
OpenClaw Credential Vault - Enum definitions.
"""

from __future__ import annotations

from enum import Enum


class CredentialType(str, Enum):
    """Types of credentials that can be stored."""

    API_KEY = "api_key"
    OAUTH_TOKEN = "oauth_token"  # noqa: S105 -- enum value
    OAUTH_SECRET = "oauth_secret"  # noqa: S105 -- enum value
    OAUTH_REFRESH_TOKEN = "oauth_refresh_token"  # noqa: S105 -- enum value
    SERVICE_ACCOUNT = "service_account"
    CERTIFICATE = "certificate"
    PASSWORD = "password"  # noqa: S105 -- enum value
    BEARER_TOKEN = "bearer_token"  # noqa: S105 -- enum value
    WEBHOOK_SECRET = "webhook_secret"  # noqa: S105 -- enum value
    ENCRYPTION_KEY = "encryption_key"


class CredentialFramework(str, Enum):
    """External frameworks that credentials may be used with."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    AWS = "aws"
    HUGGINGFACE = "huggingface"
    LANGCHAIN = "langchain"
    CREWAI = "crewai"
    AUTOGEN = "autogen"
    OPENCLAW = "openclaw"
    CUSTOM = "custom"


class CredentialAuditEvent(str, Enum):
    """Audit events for credential operations."""

    CREDENTIAL_CREATED = "credential_created"
    CREDENTIAL_ACCESSED = "credential_accessed"
    CREDENTIAL_UPDATED = "credential_updated"
    CREDENTIAL_ROTATED = "credential_rotated"
    CREDENTIAL_DELETED = "credential_deleted"
    CREDENTIAL_EXPIRED = "credential_expired"
    CREDENTIAL_ACCESS_DENIED = "credential_access_denied"
    CREDENTIAL_RATE_LIMITED = "credential_rate_limited"
    ROTATION_SCHEDULED = "rotation_scheduled"
    ROTATION_COMPLETED = "rotation_completed"
    ROTATION_FAILED = "rotation_failed"
    EXPIRY_ALERT_SENT = "expiry_alert_sent"
