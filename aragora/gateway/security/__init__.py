"""
Gateway Security - Credential vault, output filtering, and audit bridge.

Provides security infrastructure for external agent execution:
- CredentialVault: Secure credential storage and runtime injection
- OutputFilter: PII/secret redaction from agent outputs
- AuditBridge: Comprehensive audit logging for compliance
"""

from aragora.gateway.security.credential_vault import (
    CredentialVault,
    CredentialScope,
    CredentialEntry,
)
from aragora.gateway.security.output_filter import (
    OutputFilter,
    RedactionResult,
    SensitivePattern,
)
from aragora.gateway.security.audit_bridge import (
    AuditBridge,
    AuditEvent,
    AuditEventType,
)

__all__ = [
    # Credential Vault
    "CredentialVault",
    "CredentialScope",
    "CredentialEntry",
    # Output Filter
    "OutputFilter",
    "RedactionResult",
    "SensitivePattern",
    # Audit Bridge
    "AuditBridge",
    "AuditEvent",
    "AuditEventType",
]
