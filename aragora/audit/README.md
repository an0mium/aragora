# Aragora Audit Module

Enterprise-grade audit system providing compliance logging (SOC 2, HIPAA, GDPR, SOX) and intelligent document auditing with multi-agent defect detection.

## Architecture

```
audit/
├── __init__.py              # Module exports
├── log.py                   # Compliance audit logging
├── document_auditor.py      # Document audit sessions
├── orchestrator.py          # Multi-auditor orchestration
├── registry.py              # Auditor registry
├── hive_mind.py             # Multi-agent coordination
├── base_auditor.py          # Base auditor class
├── codebase_auditor.py      # Code security auditing
├── security_scanner.py      # Security vulnerability scanning
├── dependency_analyzer.py   # Dependency analysis
├── bug_detector.py          # Bug pattern detection
├── consensus_adapter.py     # Debate consensus for findings
├── evidence_adapter.py      # Evidence collection
├── knowledge_adapter.py     # Knowledge Mound integration
├── audit_types/             # Audit type definitions
├── exploration/             # Exploratory audit tools
├── findings/                # Finding management
├── presets/                 # Pre-configured audit configs
└── reports/                 # Report generation
```

## Components

### 1. Compliance Audit Logging

For regulatory compliance (SOC 2, HIPAA, GDPR, SOX):

```python
from aragora.audit import AuditLog, AuditEvent, AuditCategory, AuditOutcome

audit = AuditLog()

# Log authentication event
audit.log(AuditEvent(
    category=AuditCategory.AUTH,
    action="login",
    actor_id="user_123",
    outcome=AuditOutcome.SUCCESS,
    metadata={"ip": "192.168.1.1", "method": "sso"}
))

# Convenience functions
audit_auth_login(user_id="user_123", success=True)
audit_data_access(user_id="user_123", resource="customers", action="read")
audit_admin_action(admin_id="admin_1", action="user_delete", target="user_456")
```

**Audit Categories:**
- `AUTH` - Authentication events
- `DATA_ACCESS` - Data read/write operations
- `ADMIN` - Administrative actions
- `SYSTEM` - System events
- `SECURITY` - Security-related events
- `COMPLIANCE` - Compliance checks

### 2. Document Auditing

Intelligent document defect detection with multi-agent debate:

```python
from aragora.audit import DocumentAuditor, AuditConfig, get_document_auditor

auditor = get_document_auditor()

# Create audit session
session = await auditor.create_session(
    document_ids=["doc1", "doc2"],
    config=AuditConfig(
        enable_security_scan=True,
        enable_dependency_check=True,
        severity_threshold="medium"
    )
)

# Run the audit
result = await auditor.run_audit(session.id)

# Access findings
for finding in result.findings:
    print(f"[{finding.severity}] {finding.title}: {finding.description}")
```

### 3. Audit Orchestrator

Coordinate multiple auditors for comprehensive audits:

```python
from aragora.audit import AuditOrchestrator, get_audit_orchestrator

orchestrator = get_audit_orchestrator()

# Run full audit pipeline
result = await orchestrator.run_audit(
    workspace_id="ws-123",
    audit_types=["security", "dependency", "codebase"],
    config={
        "parallel": True,
        "fail_fast": False
    }
)
```

### 4. Hive Mind (Multi-Agent Auditing)

Coordinate multiple AI agents for thorough audits:

```python
from aragora.audit import AuditHiveMind, HiveMindConfig

hive = AuditHiveMind(config=HiveMindConfig(
    agents=["claude", "gpt4", "gemini"],
    consensus_threshold=0.7,
    debate_rounds=2
))

# Multi-agent audit with consensus
findings = await hive.audit_document(
    document_id="doc-123",
    audit_types=["security", "compliance"]
)

# Findings include confidence scores from agent consensus
for finding in findings:
    print(f"{finding.title}: {finding.confidence}% confidence")
```

### 5. Custom Auditors

Extend with custom audit types:

```python
from aragora.audit import BaseAuditor, AuditContext, audit_registry

class CustomAuditor(BaseAuditor):
    @property
    def audit_type_id(self) -> str:
        return "custom_checks"

    @property
    def display_name(self) -> str:
        return "Custom Compliance Checks"

    async def audit(self, context: AuditContext) -> list[AuditFinding]:
        findings = []
        # Custom audit logic
        return findings

# Register the auditor
audit_registry.register(CustomAuditor())
```

## Built-in Auditors

| Auditor | ID | Description |
|---------|-----|-------------|
| CodebaseAuditor | `codebase` | Code quality and patterns |
| SecurityScanner | `security` | Vulnerability detection |
| DependencyAnalyzer | `dependency` | Dependency security |
| BugDetector | `bug` | Bug pattern detection |

## Finding Severity Levels

```python
from aragora.audit import FindingSeverity

FindingSeverity.CRITICAL   # Immediate action required
FindingSeverity.HIGH       # High priority fix
FindingSeverity.MEDIUM     # Should be addressed
FindingSeverity.LOW        # Minor issue
FindingSeverity.INFO       # Informational only
```

## Integration Points

### Knowledge Mound

Findings are automatically persisted to Knowledge Mound:

```python
from aragora.audit import KnowledgeAdapter

adapter = KnowledgeAdapter()
await adapter.persist_findings(session_id, findings)
```

### Evidence Collection

Link findings to supporting evidence:

```python
from aragora.audit import EvidenceAdapter

adapter = EvidenceAdapter()
await adapter.attach_evidence(
    finding_id="finding-123",
    evidence_urls=["https://cve.mitre.org/..."]
)
```

### Consensus Adapter

Use debate consensus for finding validation:

```python
from aragora.audit import ConsensusAdapter

adapter = ConsensusAdapter()
validated = await adapter.validate_finding(finding)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v2/audit/sessions` | POST | Create audit session |
| `/api/v2/audit/sessions/{id}` | GET | Get session status |
| `/api/v2/audit/sessions/{id}/run` | POST | Run audit |
| `/api/v2/audit/findings` | GET | List findings |
| `/api/v2/audit/findings/{id}` | PATCH | Update finding status |
| `/api/v2/audit/reports` | POST | Generate report |

## Presets

Pre-configured audit profiles:

```python
from aragora.audit.presets import (
    SOC2_PRESET,
    HIPAA_PRESET,
    GDPR_PRESET,
    SECURITY_PRESET,
)

session = await auditor.create_session(
    document_ids=["doc1"],
    preset=SOC2_PRESET
)
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AUDIT_LOG_LEVEL` | Audit log verbosity | `INFO` |
| `AUDIT_RETENTION_DAYS` | Log retention period | `365` |
| `AUDIT_ENCRYPTION_KEY` | Encryption for sensitive fields | Required in prod |
| `AUDIT_PARALLEL_WORKERS` | Parallel audit workers | `4` |

## Related Modules

- `aragora/analytics/` - Audit analytics and dashboards
- `aragora/compliance/` - Compliance frameworks
- `aragora/storage/audit_store.py` - Audit event persistence
- `aragora/server/handlers/auditing.py` - API handlers
