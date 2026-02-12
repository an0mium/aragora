"""
Aragora Security Module.

Provides enterprise-grade security features including:
- Application-level encryption (AES-256-GCM)
- Key management and rotation
- Field-level encryption for sensitive data
- Cloud KMS integration (AWS, Azure, GCP)
- Threat intelligence enrichment for debate context

Usage:
    from aragora.security import EncryptionService, get_encryption_service

    # Get the service
    service = get_encryption_service()

    # Encrypt data
    encrypted = service.encrypt("sensitive data")
    decrypted = service.decrypt_string(encrypted)

    # Field-level encryption
    record = {"name": "John", "ssn": "123-45-6789"}
    encrypted_record = service.encrypt_fields(record, ["ssn"])

Cloud KMS Usage:
    from aragora.security import get_kms_provider

    # Auto-detect cloud provider
    kms = get_kms_provider()

    # Get encryption key from KMS
    key = await kms.get_encryption_key("aragora-master-key")

Threat Intel Enrichment Usage:
    from aragora.security import ThreatIntelEnrichment

    enrichment = ThreatIntelEnrichment()
    context = await enrichment.enrich_context(
        topic="How should we respond to CVE-2024-1234?"
    )
    if context:
        formatted = enrichment.format_for_debate(context)
"""

from .encryption import (
    EncryptionService,
    EncryptionConfig,
    EncryptionKey,
    EncryptedData,
    EncryptionAlgorithm,
    KeyDerivationFunction,
    get_encryption_service,
    init_encryption_service,
    CRYPTO_AVAILABLE,
)

from .kms_provider import (
    KmsProvider,
    KmsKeyMetadata,
    AwsKmsProvider,
    AzureKeyVaultProvider,
    GcpKmsProvider,
    HashiCorpVaultProvider,
    LocalKmsProvider,
    get_kms_provider,
    init_kms_provider,
    reset_kms_provider,
    detect_cloud_provider,
)

from .key_rotation import (
    KeyRotationScheduler,
    KeyRotationConfig,
    KeyRotationJob,
    KeyInfo,
    RotationStatus,
    get_key_rotation_scheduler,
    set_key_rotation_scheduler,
    start_key_rotation_scheduler,
    stop_key_rotation_scheduler,
)

from .threat_intel_enrichment import (
    ThreatIndicator,
    ThreatContext,
    ThreatIntelEnrichment,
    enrich_security_context,
    ENRICHMENT_ENABLED,
    MAX_INDICATORS,
)

from .ssrf_protection import (
    validate_url,
    is_url_safe,
    validate_webhook_url,
    validate_slack_url,
    validate_discord_url,
    validate_github_url,
    validate_microsoft_url,
    SSRFValidationResult,
    SSRFValidationError,
    SLACK_ALLOWED_DOMAINS,
    DISCORD_ALLOWED_DOMAINS,
    GITHUB_ALLOWED_DOMAINS,
    MICROSOFT_ALLOWED_DOMAINS,
    get_ssrf_config,
    ALLOWED_PROTOCOLS,
    BLOCKED_PROTOCOLS,
)

# Anomaly Detection
from .anomaly_detection import (
    AnomalyDetector,
    AnomalyDetectorConfig,
    AnomalyResult,
    AnomalyType,
    AnomalySeverity,
    get_anomaly_detector,
)

__all__ = [
    # Encryption
    "EncryptionService",
    "EncryptionConfig",
    "EncryptionKey",
    "EncryptedData",
    "EncryptionAlgorithm",
    "KeyDerivationFunction",
    "get_encryption_service",
    "init_encryption_service",
    "CRYPTO_AVAILABLE",
    # KMS Providers
    "KmsProvider",
    "KmsKeyMetadata",
    "AwsKmsProvider",
    "AzureKeyVaultProvider",
    "GcpKmsProvider",
    "HashiCorpVaultProvider",
    "LocalKmsProvider",
    "get_kms_provider",
    "init_kms_provider",
    "reset_kms_provider",
    "detect_cloud_provider",
    # Key Rotation
    "KeyRotationScheduler",
    "KeyRotationConfig",
    "KeyRotationJob",
    "KeyInfo",
    "RotationStatus",
    "get_key_rotation_scheduler",
    "set_key_rotation_scheduler",
    "start_key_rotation_scheduler",
    "stop_key_rotation_scheduler",
    # Threat Intelligence Enrichment
    "ThreatIndicator",
    "ThreatContext",
    "ThreatIntelEnrichment",
    "enrich_security_context",
    "ENRICHMENT_ENABLED",
    "MAX_INDICATORS",
    # SSRF Protection
    "validate_url",
    "is_url_safe",
    "validate_webhook_url",
    "validate_slack_url",
    "validate_discord_url",
    "validate_github_url",
    "validate_microsoft_url",
    "SSRFValidationResult",
    "SSRFValidationError",
    "SLACK_ALLOWED_DOMAINS",
    "DISCORD_ALLOWED_DOMAINS",
    "GITHUB_ALLOWED_DOMAINS",
    "MICROSOFT_ALLOWED_DOMAINS",
    "get_ssrf_config",
    "ALLOWED_PROTOCOLS",
    "BLOCKED_PROTOCOLS",
    # Anomaly Detection
    "AnomalyDetector",
    "AnomalyDetectorConfig",
    "AnomalyResult",
    "AnomalyType",
    "AnomalySeverity",
    "get_anomaly_detector",
]
