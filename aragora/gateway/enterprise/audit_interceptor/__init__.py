"""
Audit Interceptor for Enterprise Gateway.

Provides comprehensive request/response audit logging with cryptographic
integrity verification, PII redaction, and compliance support.

Features:
- Request/response logging with timing metrics
- SHA-256 hashing with hash chain for tamper detection
- GDPR-compliant PII redaction
- SOC 2 Type II audit evidence generation
- Real-time event emission for SIEM integration
- Prometheus metrics for audit volume monitoring
- Configurable retention policies
- Webhook support for external integrations

Usage:
    from aragora.gateway.enterprise.audit_interceptor import (
        AuditInterceptor,
        AuditConfig,
        PIIRedactionRule,
        RedactionType,
    )

    # Configure with PII redaction
    config = AuditConfig(
        retention_days=365,
        pii_fields=["email", "phone", "ssn"],
        emit_events=True,
        webhook_url="https://siem.example.com/webhook",
        pii_rules=[
            PIIRedactionRule(
                field_pattern=r".*email.*",
                redaction_type=RedactionType.HASH,
            ),
            PIIRedactionRule(
                field_pattern=r".*password.*",
                redaction_type=RedactionType.REMOVE,
            ),
        ],
    )

    interceptor = AuditInterceptor(config=config)

    # Intercept request/response
    record = await interceptor.intercept(
        request={"method": "POST", "path": "/api/users", "body": {...}},
        response={"status": 200, "body": {...}},
        correlation_id="req-123",
        user_id="user-456",
    )

    # Verify chain integrity
    is_valid, errors = await interceptor.verify_chain(since=datetime.now() - timedelta(days=7))

    # Export for SOC 2 audit
    report = await interceptor.export_soc2_evidence(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31),
    )
"""

from .enums import (
    RedactionType,
    AuditEventType,
    get_interceptor_signing_key,
    set_interceptor_signing_key,
)
from .models import (
    PIIRedactionRule,
    AuditConfig,
    AuditRecord,
)
from .storage import (
    AuditStorage,
    InMemoryAuditStorage,
    PostgresAuditStorage,
)
from .interceptor import AuditInterceptor

__all__ = [
    # Core classes
    "AuditInterceptor",
    "AuditRecord",
    "AuditConfig",
    "PIIRedactionRule",
    # Enums
    "RedactionType",
    "AuditEventType",
    # Storage
    "AuditStorage",
    "InMemoryAuditStorage",
    "PostgresAuditStorage",
    # Key management
    "get_interceptor_signing_key",
    "set_interceptor_signing_key",
]
