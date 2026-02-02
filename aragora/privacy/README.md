# Privacy Module

Enterprise-grade privacy infrastructure for GDPR, HIPAA, and SOC 2 compliance.

## Overview

The privacy module provides comprehensive data protection capabilities:

| Component | Purpose | Compliance |
|-----------|---------|------------|
| `GDPRDeletionScheduler` | Right to erasure (Article 17) | GDPR |
| `DataIsolationManager` | Workspace/tenant isolation | SOC 2 CC6.1 |
| `RetentionPolicyManager` | Data retention automation | GDPR Art. 5(1)(e), SOC 2 P4.1 |
| `PrivacyAuditLog` | Immutable audit trail | SOC 2 CC7.2 |
| `SensitivityClassifier` | Data classification | SOC 2 CC6.1 |
| `HIPAAAnonymizer` | Safe Harbor de-identification | HIPAA |

## Quick Start

```python
from aragora.privacy import (
    GDPRDeletionScheduler,
    DataIsolationManager,
    RetentionPolicyManager,
    SensitivityClassifier,
    PrivacyAuditLog,
)

# Initialize components
deletion_scheduler = GDPRDeletionScheduler()
isolation_manager = DataIsolationManager()
retention_manager = RetentionPolicyManager()
classifier = SensitivityClassifier()
audit_log = PrivacyAuditLog()
```

## GDPR Deletion

Implements Article 17 (Right to Erasure) with cascading deletions and cryptographic proof.

### Scheduling a Deletion Request

```python
from aragora.privacy.deletion import (
    GDPRDeletionScheduler,
    DeletionRequest,
    EntityType,
)

scheduler = GDPRDeletionScheduler(grace_period_days=30)

# Schedule user deletion
request = DeletionRequest(
    user_id="user_123",
    requester_id="user_123",
    reason="User requested account deletion",
    entity_types=[
        EntityType.USER,
        EntityType.CONSENT_RECORDS,
        EntityType.DEBATE_PARTICIPATION,
        EntityType.DOCUMENTS,
    ],
)

scheduled = await scheduler.schedule_deletion(request)
print(f"Deletion scheduled for: {scheduled.scheduled_for}")
```

### Entity Types

The module supports 15 entity types for granular deletion control:

| Entity Type | Description |
|-------------|-------------|
| `USER` | User account and profile |
| `CONSENT_RECORDS` | Consent history |
| `DEBATE_PARTICIPATION` | Debate contributions |
| `DOCUMENTS` | Uploaded documents |
| `QUERIES` | Search/query history |
| `VOTES` | Voting records |
| `SUGGESTIONS` | User suggestions |
| `MESSAGES` | Chat messages |
| `NOTIFICATIONS` | Notification history |
| `SESSIONS` | Session data |
| `API_KEYS` | API credentials |
| `AUDIT_LOGS` | User-specific audit entries |
| `ANALYTICS` | Analytics data |
| `PREFERENCES` | User preferences |
| `WORKSPACE_MEMBERSHIPS` | Workspace associations |

### Deletion Certificates

After deletion, cryptographic proof is generated:

```python
from aragora.privacy.deletion import DataErasureVerifier

verifier = DataErasureVerifier()
certificate = await verifier.verify_deletion(
    user_id="user_123",
    entity_types=[EntityType.USER, EntityType.DOCUMENTS],
)

print(f"Certificate ID: {certificate.certificate_id}")
print(f"Hash: {certificate.deletion_hash}")
print(f"Verified at: {certificate.verified_at}")
```

### Legal Holds

Pause deletions for litigation or regulatory requirements:

```python
from aragora.privacy.deletion import LegalHoldManager, LegalHold

hold_manager = LegalHoldManager()

# Create a legal hold
hold = LegalHold(
    hold_id="hold_litigation_2024",
    name="Smith v. Company Litigation",
    user_ids=["user_123", "user_456"],
    entity_types=[EntityType.DOCUMENTS, EntityType.MESSAGES],
    reason="Active litigation - preserve all relevant data",
)

await hold_manager.create_hold(hold)

# Check if user is under hold before deletion
if await hold_manager.is_under_hold("user_123"):
    print("User data is protected by legal hold")
```

## Data Isolation

Implements workspace-level data isolation with encryption and access control.

### Creating Isolated Workspaces

```python
from aragora.privacy.isolation import (
    DataIsolationManager,
    IsolationConfig,
    WorkspacePermission,
)

config = IsolationConfig(
    enable_encryption=True,
    encryption_key_rotation_days=90,
    strict_mode=True,
)

manager = DataIsolationManager(config)

# Create a workspace
workspace = await manager.create_workspace(
    workspace_id="ws_finance",
    name="Finance Department",
    owner_id="admin_001",
)

# Grant access
await manager.grant_permission(
    workspace_id="ws_finance",
    user_id="user_123",
    permission=WorkspacePermission.READ,
)
```

### Isolation Context

Thread isolation through async calls using context variables:

```python
from aragora.privacy.isolation import IsolationContext

async def process_user_request(workspace_id: str, user_id: str):
    # Set isolation context for this async call chain
    async with IsolationContext(workspace_id=workspace_id, user_id=user_id):
        # All operations within this context are isolated
        documents = await fetch_documents()  # Only returns workspace docs
        await save_result(documents)  # Saves to isolated storage
```

### Permission Levels

| Permission | Access Level |
|------------|--------------|
| `READ` | View workspace data |
| `WRITE` | Create and modify data |
| `DELETE` | Remove data |
| `ADMIN` | Manage workspace settings |
| `EXPORT` | Export data from workspace |

## Retention Policies

Automate data lifecycle management with configurable policies.

### Creating Policies

```python
from aragora.privacy.retention import (
    RetentionPolicyManager,
    RetentionAction,
)

manager = RetentionPolicyManager()

# Create a 90-day deletion policy
policy = manager.create_policy(
    name="Standard Data Retention",
    retention_days=90,
    action=RetentionAction.DELETE,
    applies_to=["documents", "findings", "sessions"],
    notify_before_days=14,
    notification_recipients=["admin@company.com"],
)

# Create a 7-year archival policy for audit logs
audit_policy = manager.create_policy(
    name="Audit Log Retention",
    retention_days=365 * 7,
    action=RetentionAction.ARCHIVE,
    applies_to=["audit_logs"],
)
```

### Retention Actions

| Action | Description |
|--------|-------------|
| `DELETE` | Permanently remove data |
| `ARCHIVE` | Move to cold storage |
| `ANONYMIZE` | Remove PII while keeping structure |
| `NOTIFY` | Alert administrators only |

### Executing Policies

```python
# Execute a specific policy
report = await manager.execute_policy("policy_abc123", dry_run=True)
print(f"Would delete: {report.items_deleted}")
print(f"Would archive: {report.items_archived}")

# Execute all enabled policies
reports = await manager.execute_all_policies(dry_run=False)

# Check what's expiring soon
expiring = await manager.check_expiring_soon(days=14)
for item in expiring:
    print(f"{item['resource_type']} {item['resource_id']} expires in {item['days_until_expiry']} days")
```

### Compliance Reporting

```python
from datetime import datetime, timedelta

report = await manager.get_compliance_report(
    workspace_id="ws_finance",
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now(),
)

print(f"Total deletions: {report['total_deletions']}")
print(f"Active policies: {report['active_policies']}")
```

## Privacy Audit Log

SOC 2-compliant immutable audit logging with cryptographic chain verification.

### Logging Operations

```python
from aragora.privacy.audit_log import (
    PrivacyAuditLog,
    AuditAction,
    AuditOutcome,
    Actor,
    Resource,
)

audit_log = PrivacyAuditLog()

# Log a data access
await audit_log.log(
    action=AuditAction.READ,
    actor=Actor(
        id="user_123",
        type="user",
        ip_address="192.168.1.100",
    ),
    resource=Resource(
        id="doc_456",
        type="document",
        workspace_id="ws_finance",
        sensitivity_level="confidential",
    ),
    outcome=AuditOutcome.SUCCESS,
    duration_ms=45,
)
```

### Audit Actions

| Category | Actions |
|----------|---------|
| Data Access | `READ`, `WRITE`, `DELETE`, `EXPORT` |
| Workspace | `CREATE_WORKSPACE`, `DELETE_WORKSPACE`, `ADD_MEMBER`, `REMOVE_MEMBER` |
| Documents | `UPLOAD_DOCUMENT`, `DELETE_DOCUMENT`, `CLASSIFY_DOCUMENT` |
| Queries | `QUERY`, `SEARCH` |
| Administrative | `MODIFY_POLICY`, `EXECUTE_RETENTION`, `GENERATE_REPORT` |
| Authentication | `LOGIN`, `LOGOUT`, `AUTH_FAILURE` |

### Querying Audit Logs

```python
# Find all actions by a user
entries = await audit_log.query(
    actor_id="user_123",
    start_date=datetime.now() - timedelta(days=7),
)

# Find denied access attempts
denied = await audit_log.get_denied_access_attempts(days=7)

# Get resource access history
history = await audit_log.get_resource_history("doc_456", days=30)
```

### Integrity Verification

```python
# Verify the audit log chain hasn't been tampered with
is_valid, errors = await audit_log.verify_integrity(
    start_date=datetime.now() - timedelta(days=30),
)

if not is_valid:
    print(f"Integrity errors: {errors}")
```

### Compliance Reports

```python
report = await audit_log.generate_compliance_report(
    start_date=datetime.now() - timedelta(days=30),
    workspace_id="ws_finance",
)

print(f"Total entries: {report['summary']['total_entries']}")
print(f"Denied attempts: {report['summary']['denied_count']}")
print(f"Integrity verified: {report['integrity']['verified']}")
```

## Sensitivity Classification

Automatic data classification using pattern matching and optional LLM enhancement.

### Classifying Content

```python
from aragora.privacy.classifier import (
    SensitivityClassifier,
    SensitivityLevel,
)

classifier = SensitivityClassifier()

result = await classifier.classify(
    content="John Smith's SSN is 123-45-6789 and his email is john@example.com",
    document_id="doc_001",
)

print(f"Level: {result.level}")  # SensitivityLevel.CONFIDENTIAL
print(f"Confidence: {result.confidence}")
print(f"PII detected: {result.pii_detected}")
print(f"Secrets detected: {result.secrets_detected}")
```

### Sensitivity Levels

| Level | Description | Recommended Policy |
|-------|-------------|-------------------|
| `PUBLIC` | Can be shared publicly | No encryption required |
| `INTERNAL` | Internal use only | Access logging |
| `CONFIDENTIAL` | Contains PII/financial data | Encryption required |
| `RESTRICTED` | Contains secrets/credentials | 30-day retention, approval required |
| `TOP_SECRET` | Maximum restriction | MFA required, 7-day retention |

### Built-in Detection Patterns

- **TOP SECRET**: National security markers, classification stamps
- **RESTRICTED**: API keys, private keys, database credentials
- **CONFIDENTIAL**: SSN, credit cards, medical records, financial data
- **INTERNAL**: Email addresses, internal-use markers, phone numbers

### Custom Indicators

```python
from aragora.privacy.classifier import SensitivityIndicator

classifier.add_indicator(
    SensitivityIndicator(
        name="customer_id",
        pattern=r"CUST-\d{8}",
        level=SensitivityLevel.CONFIDENTIAL,
        confidence=0.85,
        description="Customer identification numbers",
    )
)
```

### Level-Based Policies

```python
# Get recommended policy for a classification level
policy = classifier.get_level_policy(SensitivityLevel.CONFIDENTIAL)
print(policy)
# {
#     "encryption_required": True,
#     "access_logging": True,
#     "retention_days": 90,
#     "sharing_allowed": False,
#     "export_allowed": False,
# }
```

## HIPAA Anonymization

Safe Harbor de-identification with multiple anonymization methods.

### Basic Anonymization

```python
from aragora.privacy.anonymization import (
    HIPAAAnonymizer,
    AnonymizationMethod,
)

anonymizer = HIPAAAnonymizer()

# Redact PII
result = anonymizer.anonymize(
    "John Smith's SSN is 123-45-6789",
    method=AnonymizationMethod.REDACT,
)
print(result.anonymized_content)
# "[NAME]'s SSN is [SSN]"

# Hash identifiers (one-way)
result = anonymizer.anonymize(
    "Contact: john@example.com",
    method=AnonymizationMethod.HASH,
)
print(result.anonymized_content)
# "Contact: a1b2c3d4e5f6..."
```

### Anonymization Methods

| Method | Description | Reversible |
|--------|-------------|------------|
| `REDACT` | Replace with type markers `[NAME]`, `[SSN]` | No |
| `HASH` | One-way cryptographic hash | No |
| `GENERALIZE` | Reduce precision (age: 35 -> 30-40) | No |
| `SUPPRESS` | Remove entirely | No |
| `PSEUDONYMIZE` | Replace with consistent fake values | Yes (with mapping) |

### HIPAA Safe Harbor Verification

```python
# Check if content meets Safe Harbor requirements
verification = anonymizer.verify_safe_harbor(
    "Patient: John Smith, DOB: 01/15/1980, MRN: ABC123456"
)

print(f"Compliant: {verification.compliant}")
print(f"Identifiers found: {len(verification.identifiers_remaining)}")
```

### Supported Identifier Types

All 18 HIPAA Safe Harbor identifiers:

1. Names
2. Geographic data (address, city, state, zip)
3. Dates (except year)
4. Phone numbers
5. Fax numbers
6. Email addresses
7. Social Security numbers
8. Medical record numbers
9. Health plan beneficiary numbers
10. Account numbers
11. Certificate/license numbers
12. Vehicle identifiers (VIN, plates)
13. Device identifiers
14. Web URLs
15. IP addresses
16. Biometric identifiers
17. Full face photos
18. Any other unique identifier

### K-Anonymity for Datasets

```python
from aragora.privacy.anonymization import KAnonymizer

k_anon = KAnonymizer(k=5)

records = [
    {"name": "John", "age": 35, "zipcode": "12345"},
    {"name": "Jane", "age": 36, "zipcode": "12346"},
    # ... more records
]

anonymized = k_anon.anonymize_dataset(
    records,
    quasi_identifiers=["age", "zipcode"],
)

# Verify k-anonymity
is_k_anon, min_group = k_anon.check_k_anonymity(
    anonymized,
    quasi_identifiers=["age", "zipcode"],
)
print(f"K-anonymous (k=5): {is_k_anon}, min group size: {min_group}")
```

### Differential Privacy

```python
from aragora.privacy.anonymization import DifferentialPrivacy

dp = DifferentialPrivacy(epsilon=1.0)

# Privatize a count query
true_count = 1000
private_count = dp.privatize_count(true_count)

# Privatize a sum query
total_salary = 5_000_000
private_sum = dp.privatize_sum(total_salary, max_contribution=200_000)

# Privatize a mean query
ages = [25, 30, 35, 40, 45]
private_mean = dp.privatize_mean(ages, lower_bound=18, upper_bound=100)
```

## Integration Patterns

### Complete Privacy Workflow

```python
from aragora.privacy import (
    SensitivityClassifier,
    DataIsolationManager,
    PrivacyAuditLog,
    HIPAAAnonymizer,
)

async def process_document(document: dict, user_id: str, workspace_id: str):
    # 1. Classify sensitivity
    classification = await classifier.classify(document["content"])

    # 2. Check workspace access
    if not await isolation_manager.has_permission(
        workspace_id, user_id, WorkspacePermission.WRITE
    ):
        await audit_log.log(
            action=AuditAction.WRITE,
            actor=Actor(id=user_id, type="user"),
            resource=Resource(id=document["id"], type="document"),
            outcome=AuditOutcome.DENIED,
        )
        raise PermissionError("Access denied")

    # 3. Anonymize if contains PII
    if classification.pii_detected:
        result = anonymizer.anonymize(
            document["content"],
            method=AnonymizationMethod.REDACT,
        )
        document["content"] = result.anonymized_content

    # 4. Store with classification metadata
    document["sensitivity_level"] = classification.level.value

    # 5. Audit the operation
    await audit_log.log(
        action=AuditAction.WRITE,
        actor=Actor(id=user_id, type="user"),
        resource=Resource(
            id=document["id"],
            type="document",
            workspace_id=workspace_id,
            sensitivity_level=classification.level.value,
        ),
        outcome=AuditOutcome.SUCCESS,
    )

    return document
```

### GDPR Subject Access Request

```python
async def handle_subject_access_request(user_id: str):
    """Process GDPR Article 15 subject access request."""

    # 1. Gather all user data
    user_data = await gather_user_data(user_id)

    # 2. Audit the export
    await audit_log.log(
        action=AuditAction.EXPORT,
        actor=Actor(id=user_id, type="user"),
        resource=Resource(id=user_id, type="user_data"),
        outcome=AuditOutcome.SUCCESS,
    )

    # 3. Generate exportable format
    return format_for_export(user_data)

async def handle_deletion_request(user_id: str, requester_id: str):
    """Process GDPR Article 17 deletion request."""

    # 1. Check for legal holds
    if await hold_manager.is_under_hold(user_id):
        return {"status": "on_hold", "message": "Data protected by legal hold"}

    # 2. Schedule deletion with grace period
    request = DeletionRequest(
        user_id=user_id,
        requester_id=requester_id,
        reason="GDPR Article 17 request",
    )

    scheduled = await deletion_scheduler.schedule_deletion(request)

    # 3. Audit the request
    await audit_log.log(
        action=AuditAction.DELETE,
        actor=Actor(id=requester_id, type="user"),
        resource=Resource(id=user_id, type="user"),
        outcome=AuditOutcome.SUCCESS,
        details={"scheduled_for": scheduled.scheduled_for.isoformat()},
    )

    return {
        "status": "scheduled",
        "deletion_date": scheduled.scheduled_for,
        "grace_period_days": 30,
    }
```

## Configuration

### Environment Variables

```bash
# Audit log storage
PRIVACY_AUDIT_LOG_DIR=/var/log/aragora/audit

# Retention defaults
PRIVACY_DEFAULT_RETENTION_DAYS=90

# Encryption
PRIVACY_ENCRYPTION_ENABLED=true
PRIVACY_KEY_ROTATION_DAYS=90

# GDPR
GDPR_GRACE_PERIOD_DAYS=30
```

### Global Instances

```python
from aragora.privacy.deletion import get_deletion_scheduler
from aragora.privacy.retention import get_retention_manager
from aragora.privacy.audit_log import get_audit_log
from aragora.privacy.classifier import get_classifier

# Access global singleton instances
scheduler = get_deletion_scheduler()
retention = get_retention_manager()
audit = get_audit_log()
classifier = get_classifier()
```

## Compliance Reference

| Requirement | Module | Component |
|-------------|--------|-----------|
| GDPR Art. 17 (Right to erasure) | `deletion` | `GDPRDeletionScheduler` |
| GDPR Art. 5(1)(e) (Storage limitation) | `retention` | `RetentionPolicyManager` |
| GDPR Art. 15 (Right of access) | `audit_log` | `PrivacyAuditLog.query()` |
| GDPR Art. 20 (Data portability) | `isolation` | `DataIsolationManager.export()` |
| HIPAA Safe Harbor | `anonymization` | `HIPAAAnonymizer` |
| SOC 2 CC6.1 (Logical access) | `isolation` | `DataIsolationManager` |
| SOC 2 CC6.5 (Secure disposal) | `deletion` | `DataErasureVerifier` |
| SOC 2 CC7.2 (Monitoring) | `audit_log` | `PrivacyAuditLog` |
| SOC 2 P4.1 (Data retention) | `retention` | `RetentionPolicyManager` |

## API Reference

### Module Exports

```python
from aragora.privacy import (
    # Deletion
    GDPRDeletionScheduler,
    DeletionCascadeManager,
    DataErasureVerifier,
    LegalHoldManager,
    DeletionRequest,
    DeletionCertificate,
    LegalHold,
    EntityType,

    # Isolation
    DataIsolationManager,
    IsolationContext,
    IsolationConfig,
    Workspace,
    WorkspacePermission,

    # Retention
    RetentionPolicyManager,
    RetentionPolicy,
    RetentionAction,
    DeletionReport,

    # Audit
    PrivacyAuditLog,
    AuditEntry,
    AuditAction,
    AuditOutcome,
    Actor,
    Resource,

    # Classification
    SensitivityClassifier,
    SensitivityLevel,
    ClassificationResult,
    SensitivityIndicator,

    # Anonymization
    HIPAAAnonymizer,
    AnonymizationMethod,
    AnonymizationResult,
    IdentifierType,
    KAnonymizer,
    DifferentialPrivacy,
)
```

## See Also

- [Enterprise Features](../../docs/ENTERPRISE_FEATURES.md) - Full enterprise capabilities
- [RBAC Module](../rbac/README.md) - Role-based access control
- [Security Module](../security/README.md) - Encryption and authentication
- [Backup Module](../backup/README.md) - Disaster recovery
