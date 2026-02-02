# Security Module

Enterprise-grade security infrastructure for encryption, key management, and threat protection.

## Overview

The security module provides comprehensive protection capabilities:

| Component | Purpose | Compliance |
|-----------|---------|------------|
| `EncryptionService` | AES-256-GCM encryption | SOC 2 CC6.1 |
| `KmsProvider` | Cloud key management (AWS/Azure/GCP/Vault) | SOC 2 CC6.7 |
| `KeyRotationScheduler` | Automated key rotation | SOC 2 CC6.7 |
| `SSRFValidationResult` | URL validation and SSRF protection | OWASP Top 10 |
| `ThreatIntelEnrichment` | CVE/threat intelligence for debates | Security Operations |

## Quick Start

```python
from aragora.security import (
    EncryptionService,
    get_encryption_service,
    get_kms_provider,
    validate_url,
    ThreatIntelEnrichment,
)

# Get encryption service
service = get_encryption_service()

# Encrypt data
encrypted = service.encrypt("sensitive data")
decrypted = service.decrypt_string(encrypted)

# Validate URLs for SSRF
if validate_url(user_url).is_safe:
    # Safe to make request
    pass
```

## Encryption

AES-256-GCM authenticated encryption with key versioning and rotation support.

### Basic Encryption

```python
from aragora.security.encryption import (
    EncryptionService,
    EncryptionConfig,
    get_encryption_service,
)

# Using global service (recommended)
service = get_encryption_service()

# Encrypt string or bytes
encrypted = service.encrypt("sensitive data")
print(f"Encrypted: {encrypted.to_base64()}")

# Decrypt
decrypted = service.decrypt_string(encrypted)
print(f"Decrypted: {decrypted}")
```

### With Associated Data (AAD)

Additional authenticated data provides integrity verification without encryption:

```python
# Encrypt with AAD (e.g., user ID for context binding)
encrypted = service.encrypt(
    "account balance: $10,000",
    associated_data="user_123",
)

# Must provide same AAD to decrypt
decrypted = service.decrypt_string(encrypted, associated_data="user_123")

# Wrong AAD fails authentication
try:
    service.decrypt_string(encrypted, associated_data="user_456")
except Exception:
    print("Authentication failed - data tampered or wrong context")
```

### Field-Level Encryption

Encrypt specific fields in records while leaving others in plaintext:

```python
record = {
    "user_id": "u_123",
    "name": "John Smith",
    "ssn": "123-45-6789",
    "email": "john@example.com",
}

# Encrypt only sensitive fields
encrypted_record = service.encrypt_fields(
    record,
    sensitive_fields=["ssn"],
    associated_data="u_123",  # Bind to user
)

print(encrypted_record)
# {
#     "user_id": "u_123",
#     "name": "John Smith",
#     "ssn": {"_encrypted": True, "_value": "base64..."},
#     "email": "john@example.com",
# }

# Decrypt fields
decrypted_record = service.decrypt_fields(
    encrypted_record,
    sensitive_fields=["ssn"],
    associated_data="u_123",
)
```

### Key Management

```python
# Generate a new key
key = service.generate_key(key_id="tenant_123", ttl_days=90)
print(f"Generated key: {key.key_id} v{key.version}")

# Derive key from password
key, salt = service.derive_key_from_password(
    password="user-provided-secret",
    key_id="derived_key",
)
# Store salt for future derivation

# Rotate key
new_key = service.rotate_key("tenant_123")
print(f"Rotated to v{new_key.version}")

# Re-encrypt data with new key
new_encrypted = service.re_encrypt(old_encrypted)
```

### Production Configuration

```bash
# Required in production (ARAGORA_ENV=production)
export ARAGORA_ENCRYPTION_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")

# Optional: enforce encryption (fails rather than falling back to plaintext)
export ARAGORA_ENCRYPTION_REQUIRED=true
```

## Cloud KMS Integration

Multi-cloud key management with auto-detection.

### Supported Providers

| Provider | Environment Variables |
|----------|----------------------|
| AWS KMS | `AWS_REGION`, `ARAGORA_AWS_KMS_KEY_ID` |
| Azure Key Vault | `AZURE_KEY_VAULT_URL`, `ARAGORA_AZURE_KEY_NAME` |
| GCP Cloud KMS | `GOOGLE_CLOUD_PROJECT`, `ARAGORA_GCP_KMS_KEYRING` |
| HashiCorp Vault | `VAULT_ADDR`, `VAULT_TOKEN` |
| Local (dev only) | `ARAGORA_ENCRYPTION_KEY` |

### Auto-Detection

```python
from aragora.security.kms_provider import (
    get_kms_provider,
    detect_cloud_provider,
)

# Auto-detect based on environment
provider = detect_cloud_provider()
print(f"Using provider: {provider}")  # "aws", "azure", "gcp", "vault", or "local"

# Get configured provider
kms = get_kms_provider()

# Generate data encryption key
key = await kms.get_encryption_key("aragora-master-key")

# Envelope encryption
plaintext_key = secrets.token_bytes(32)
encrypted_key = await kms.encrypt_data_key(plaintext_key, "master-key")
decrypted_key = await kms.decrypt_data_key(encrypted_key, "master-key")
```

### AWS KMS

```python
from aragora.security.kms_provider import AwsKmsProvider

kms = AwsKmsProvider(
    region="us-east-1",
    key_id="arn:aws:kms:us-east-1:123456789:key/abc123",
)

# Generate data key (for envelope encryption)
data_key = await kms.get_encryption_key("alias/aragora-master")

# Get key metadata
meta = await kms.get_key_metadata("alias/aragora-master")
print(f"Key ARN: {meta.key_arn}")
```

### Azure Key Vault

```python
from aragora.security.kms_provider import AzureKeyVaultProvider

kms = AzureKeyVaultProvider(
    vault_url="https://my-vault.vault.azure.net",
    key_name="aragora-master-key",
)

key = await kms.get_encryption_key("aragora-master-key")
```

### HashiCorp Vault

```python
from aragora.security.kms_provider import HashiCorpVaultProvider

kms = HashiCorpVaultProvider(
    addr="https://vault.company.com:8200",
    token="s.xxxxx",
    transit_path="transit",
    key_name="aragora-master-key",
)

# Generate data key via Transit engine
data_key = await kms.get_encryption_key("aragora-master-key")

# Rotate key in Vault
new_meta = await kms.rotate_key("aragora-master-key")
print(f"Rotated to version: {new_meta.version}")
```

## Key Rotation

Automated key rotation with re-encryption support.

### Configuration

```python
from aragora.security.key_rotation import (
    KeyRotationScheduler,
    KeyRotationConfig,
    start_key_rotation_scheduler,
)

config = KeyRotationConfig(
    rotation_interval_days=90,      # Rotate every 90 days
    check_interval_hours=6,         # Check every 6 hours
    auto_rotate_kms_keys=True,      # Rotate cloud KMS keys
    re_encrypt_on_rotation=True,    # Re-encrypt stored data
    key_overlap_days=7,             # Keep old key for 7 days
    notify_days_before=7,           # Alert 7 days before expiry
    stores_to_re_encrypt=[
        "integrations",
        "webhooks",
        "gmail_tokens",
    ],
)
```

### Starting the Scheduler

```python
# Start automated rotation
scheduler = await start_key_rotation_scheduler(config=config)

# Or manual control
scheduler = KeyRotationScheduler(config=config)
await scheduler.start()

# Track a key for rotation
scheduler.track_key(KeyInfo(
    key_id="tenant_123",
    provider="local",
    version=1,
    created_at=datetime.now(timezone.utc),
))

# Trigger immediate rotation
job = await scheduler.rotate_now(key_id="tenant_123")
print(f"Rotation status: {job.status}")
```

### Event Notifications

```python
def rotation_callback(event_type: str, data: dict):
    if event_type == "key_rotation_completed":
        print(f"Key {data['key_id']} rotated to v{data['new_version']}")
    elif event_type == "keys_expiring_soon":
        print(f"{data['count']} keys expiring soon")
    elif event_type == "key_rotation_failed":
        print(f"Rotation failed: {data['error']}")

scheduler = KeyRotationScheduler(
    config=config,
    event_callback=rotation_callback,
)
```

### Monitoring

```python
# Get scheduler statistics
stats = scheduler.get_stats()
print(f"Status: {stats.status}")
print(f"Total rotations: {stats.total_rotations}")
print(f"Failed rotations: {stats.failed_rotations}")
print(f"Keys expiring soon: {stats.keys_expiring_soon}")

# Get rotation history
history = scheduler.get_job_history(limit=10)
for job in history:
    print(f"{job.key_id}: {job.status} at {job.completed_at}")
```

## SSRF Protection

Prevents Server-Side Request Forgery attacks on outbound requests.

### Basic Validation

```python
from aragora.security.ssrf_protection import (
    validate_url,
    is_url_safe,
    SSRFValidationError,
)

# Simple check
if is_url_safe(user_provided_url):
    # Safe to make request
    pass

# Detailed validation
result = validate_url(user_provided_url)
if not result.is_safe:
    print(f"Blocked: {result.error}")
    raise SSRFValidationError(result.error)
```

### Domain Whitelisting

```python
# Only allow specific domains
result = validate_url(
    url,
    allowed_domains={"api.example.com", "cdn.example.com"},
)

# Block specific domains
result = validate_url(
    url,
    blocked_domains={"internal.company.com"},
)
```

### DNS Resolution Check

Prevents DNS rebinding attacks by verifying resolved IPs:

```python
result = validate_url(
    url,
    resolve_dns=True,  # Resolve and check IP
    dns_timeout=2.0,
)

if result.resolved_ip:
    print(f"Resolved to: {result.resolved_ip}")
```

### Service-Specific Validators

Pre-configured validators for common services:

```python
from aragora.security.ssrf_protection import (
    validate_slack_url,
    validate_discord_url,
    validate_github_url,
    validate_microsoft_url,
)

# Validates against Slack's allowed domains
result = validate_slack_url("https://hooks.slack.com/services/xxx")

# GitHub API URLs only
result = validate_github_url("https://api.github.com/repos/xxx")
```

### Protected IP Ranges

Automatically blocked:

- Private networks: `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`
- Loopback: `127.0.0.0/8`
- Link-local: `169.254.0.0/16` (includes cloud metadata)
- Cloud metadata: `169.254.169.254`

### Configuration

```bash
# Strict mode (default: true)
export ARAGORA_SSRF_STRICT=true

# DNS resolution for rebinding protection
export ARAGORA_SSRF_RESOLVE_DNS=false

# Allow localhost (NEVER in production)
export ARAGORA_SSRF_ALLOW_LOCALHOST=false  # Blocked in production even if true
```

## Threat Intelligence Enrichment

Enriches security-related debates with threat intelligence.

### Basic Usage

```python
from aragora.security.threat_intel_enrichment import (
    ThreatIntelEnrichment,
    enrich_security_context,
)

enrichment = ThreatIntelEnrichment()

# Check if topic is security-related
if enrichment.is_security_topic("How do we respond to CVE-2024-1234?"):
    context = await enrichment.enrich_context(
        topic="How do we respond to CVE-2024-1234?",
        existing_context="We run Python 3.11 with requests 2.28.0",
    )

    if context:
        formatted = enrichment.format_for_debate(context)
        # Add to debate prompt
```

### Quick Enrichment

```python
# Convenience function
context = await enrich_security_context(
    "How should we respond to CVE-2024-1234?"
)
if context:
    debate_prompt += context
```

### Detected Entities

Automatically extracts and looks up:

- **CVEs**: `CVE-2024-1234` format
- **IP Addresses**: Reputation lookup via AbuseIPDB
- **URLs**: Malicious URL detection
- **File Hashes**: MD5, SHA1, SHA256 malware analysis

### Security Keywords

Topics containing these keywords trigger enrichment:

| Category | Keywords |
|----------|----------|
| Threats | security, vulnerability, exploit, attack, malware, breach |
| Specific | ransomware, phishing, apt, injection, xss, csrf |
| Compliance | soc2, pci, gdpr, hipaa, audit, compliance |
| Technical | encryption, authentication, certificate, ssl, tls |

### Output Format

```markdown
## Threat Intelligence Context

### Relevant CVEs
- **CVE-2024-1234** [CRITICAL] (CVSS: 9.8): Remote code execution...
  - Fix: Upgrade to version 2.31.0

### Threat Indicators
- **IP**: `45.33.32.156` (high, 85% confidence, source: abuseipdb)
  - Abuse score: 85, Reports: 150

### Attack Patterns
- Remote Code Execution
- Privilege Escalation

### Recommended Mitigations
1. Upgrade to version 2.31.0 to fix CVE-2024-1234
2. Block 1 malicious IP(s) at firewall level

**Risk Summary**: Overall risk: CRITICAL. Found 1 CRITICAL CVE(s).
```

### Configuration

```bash
# Enable/disable enrichment
export ARAGORA_THREAT_INTEL_ENRICHMENT_ENABLED=true

# Maximum indicators to include
export ARAGORA_THREAT_INTEL_MAX_INDICATORS=10
```

## Environment Variables Reference

### Encryption

| Variable | Description | Default |
|----------|-------------|---------|
| `ARAGORA_ENCRYPTION_KEY` | 32-byte hex master key | Required in production |
| `ARAGORA_ENCRYPTION_REQUIRED` | Fail on encryption errors | `false` |
| `ARAGORA_ENV` | Environment (`production`/`staging`) | - |

### KMS Providers

| Variable | Description | Provider |
|----------|-------------|----------|
| `ARAGORA_KMS_PROVIDER` | Force provider selection | Auto-detect |
| `AWS_REGION` | AWS region | AWS |
| `ARAGORA_AWS_KMS_KEY_ID` | AWS KMS key ARN | AWS |
| `AZURE_KEY_VAULT_URL` | Azure vault URL | Azure |
| `ARAGORA_AZURE_KEY_NAME` | Azure key name | Azure |
| `GOOGLE_CLOUD_PROJECT` | GCP project ID | GCP |
| `ARAGORA_GCP_KMS_KEYRING` | GCP keyring name | GCP |
| `ARAGORA_GCP_KMS_KEY` | GCP key name | GCP |
| `VAULT_ADDR` | HashiCorp Vault address | Vault |
| `VAULT_TOKEN` | Vault authentication token | Vault |
| `ARAGORA_VAULT_TRANSIT_PATH` | Vault transit engine path | Vault |
| `ARAGORA_VAULT_KEY_NAME` | Vault key name | Vault |

### Key Rotation

| Variable | Description | Default |
|----------|-------------|---------|
| `ARAGORA_KEY_ROTATION_INTERVAL_DAYS` | Days between rotations | `90` |
| `ARAGORA_KEY_ROTATION_OVERLAP_DAYS` | Key overlap period | `7` |
| `ARAGORA_KEY_ROTATION_RE_ENCRYPT` | Re-encrypt on rotation | `true` |
| `ARAGORA_KEY_ROTATION_ALERT_DAYS` | Alert before expiry | `7` |

### SSRF Protection

| Variable | Description | Default |
|----------|-------------|---------|
| `ARAGORA_SSRF_STRICT` | Strict validation mode | `true` |
| `ARAGORA_SSRF_RESOLVE_DNS` | DNS rebinding protection | `false` |
| `ARAGORA_SSRF_DNS_TIMEOUT` | DNS resolution timeout | `2.0` |
| `ARAGORA_SSRF_ALLOW_LOCALHOST` | Allow localhost (never in prod) | `false` |

## Compliance Reference

| Requirement | Component |
|-------------|-----------|
| SOC 2 CC6.1 (Logical Access) | `EncryptionService`, Field-level encryption |
| SOC 2 CC6.7 (Key Management) | `KmsProvider`, `KeyRotationScheduler` |
| PCI DSS 3.5 (Protect Keys) | Cloud KMS integration, Key rotation |
| PCI DSS 3.6 (Key Management) | `KeyRotationConfig`, Auto-rotation |
| OWASP A10 (SSRF) | `validate_url`, Domain whitelisting |
| NIST 800-57 (Key Management) | 90-day rotation, Key versioning |

## API Reference

### Module Exports

```python
from aragora.security import (
    # Encryption
    EncryptionService,
    EncryptionConfig,
    EncryptionKey,
    EncryptedData,
    EncryptionAlgorithm,
    KeyDerivationFunction,
    get_encryption_service,
    init_encryption_service,
    CRYPTO_AVAILABLE,

    # KMS Providers
    KmsProvider,
    KmsKeyMetadata,
    AwsKmsProvider,
    AzureKeyVaultProvider,
    GcpKmsProvider,
    HashiCorpVaultProvider,
    LocalKmsProvider,
    get_kms_provider,
    init_kms_provider,
    detect_cloud_provider,

    # Key Rotation
    KeyRotationScheduler,
    KeyRotationConfig,
    KeyRotationJob,
    KeyInfo,
    RotationStatus,
    get_key_rotation_scheduler,
    start_key_rotation_scheduler,
    stop_key_rotation_scheduler,

    # SSRF Protection
    validate_url,
    is_url_safe,
    validate_webhook_url,
    validate_slack_url,
    validate_discord_url,
    validate_github_url,
    validate_microsoft_url,
    SSRFValidationResult,
    SSRFValidationError,
    ALLOWED_PROTOCOLS,
    BLOCKED_PROTOCOLS,

    # Threat Intelligence
    ThreatIndicator,
    ThreatContext,
    ThreatIntelEnrichment,
    enrich_security_context,
)
```

## See Also

- [Privacy Module](../privacy/README.md) - GDPR and data protection
- [RBAC Module](../rbac/README.md) - Role-based access control
- [Auth Module](../auth/README.md) - Authentication (OIDC, SAML, MFA)
- [Enterprise Features](../../docs/ENTERPRISE_FEATURES.md) - Full enterprise capabilities
