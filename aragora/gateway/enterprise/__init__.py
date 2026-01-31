"""
Enterprise Gateway Components.

Provides enterprise-grade gateway features for compliance and security:
- AuditInterceptor: Request/response audit logging with cryptographic trails
- PII redaction and GDPR compliance
- SOC 2 audit evidence generation
- SIEM integration via webhooks

Usage:
    from aragora.gateway.enterprise import AuditInterceptor, AuditConfig

    interceptor = AuditInterceptor(config=AuditConfig(
        retention_days=365,
        emit_events=True,
    ))

    # Intercept a request/response
    record = await interceptor.intercept(
        request=request_data,
        response=response_data,
        correlation_id="req-123",
    )
"""

from aragora.gateway.enterprise.audit_interceptor import (
    AuditInterceptor,
    AuditRecord,
    AuditConfig,
    PIIRedactionRule,
    RedactionType,
    AuditEventType,
    AuditStorage,
    InMemoryAuditStorage,
    PostgresAuditStorage,
)

__all__ = [
    "AuditInterceptor",
    "AuditRecord",
    "AuditConfig",
    "PIIRedactionRule",
    "RedactionType",
    "AuditEventType",
    "AuditStorage",
    "InMemoryAuditStorage",
    "PostgresAuditStorage",
]
