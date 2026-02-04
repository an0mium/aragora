"""
OpenClaw Credential Vault - Enum definitions.
"""

from __future__ import annotations

from enum import Enum


class CredentialType(str, Enum):
    """Types of credentials that can be stored."""

    API_KEY = "api_key"
    OAUTH_TOKEN = "oauth_token"
    OAUTH_SECRET = "oauth_secret"
    OAUTH_REFRESH_TOKEN = "oauth_refresh_token"
    SERVICE_ACCOUNT = "service_account"
    CERTIFICATE = "certificate"
    PASSWORD = "password"
    BEARER_TOKEN = "bearer_token"
    WEBHOOK_SECRET = "webhook_secret"
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
