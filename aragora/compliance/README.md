# Compliance Module

Enterprise compliance framework for Aragora, providing multi-framework compliance checking, continuous monitoring, and regulatory violation detection for HIPAA, GDPR, SOX, PCI-DSS, and other standards.

## Overview

The compliance module enables organizations to:

- **Multi-Framework Compliance**: Check compliance against 8+ regulatory frameworks simultaneously
- **Pattern-Based Detection**: Identify violations using configurable regex rules and patterns
- **Continuous Monitoring**: Background scanning with drift detection and trend analysis
- **Alerting Integration**: Connect to observability module for compliance alerts
- **Audit Trail**: Generate compliance reports and violation histories

## Architecture

```
aragora/compliance/
├── __init__.py          # Module exports
├── framework.py         # Core compliance framework with rules and checks
└── monitor.py           # Continuous compliance monitoring
```

## Supported Frameworks

| Framework | Description | Focus Areas |
|-----------|-------------|-------------|
| **HIPAA** | Health Insurance Portability and Accountability Act | PHI protection, access controls, audit trails |
| **GDPR** | General Data Protection Regulation | Personal data, consent, data subject rights |
| **SOX** | Sarbanes-Oxley Act | Financial reporting, internal controls |
| **OWASP** | Open Web Application Security Project | Security vulnerabilities, injection attacks |
| **PCI-DSS** | Payment Card Industry Data Security Standard | Cardholder data, encryption, access control |
| **FDA 21 CFR** | FDA 21 CFR Part 11 | Electronic records, signatures, validation |
| **ISO 27001** | Information Security Management | Security controls, risk management |
| **FedRAMP** | Federal Risk and Authorization Management | Cloud security, government compliance |

## Key Classes

### Framework

- **`ComplianceFramework`**: Container for compliance rules and configuration
- **`ComplianceRule`**: Individual compliance rule with pattern matching
- **`ComplianceCheckResult`**: Result of a compliance check operation
- **`ComplianceIssue`**: Single compliance violation with details
- **`ComplianceSeverity`**: Severity levels (CRITICAL, HIGH, MEDIUM, LOW, INFO)

### Framework Manager

- **`ComplianceFrameworkManager`**: Manages multiple frameworks and orchestrates checks
  - Register and unregister frameworks
  - Run checks against one or all frameworks
  - Aggregate results across frameworks

### Monitor

- **`ComplianceMonitor`**: Continuous compliance monitoring
  - Background scanning at configurable intervals
  - Drift detection from baseline
  - Violation trend analysis
  - Alert integration

### Enums

- **`ComplianceHealth`**: Overall health status (HEALTHY, DEGRADED, CRITICAL)
- **`ViolationTrend`**: Trend direction (IMPROVING, STABLE, WORSENING)

## Usage Example

### Basic Compliance Checking

```python
from aragora.compliance import (
    ComplianceFrameworkManager,
    ComplianceFramework,
    ComplianceRule,
    ComplianceSeverity,
    check_compliance,
)

# Create a framework manager
manager = ComplianceFrameworkManager()

# Use pre-built frameworks (loaded automatically)
# Available: HIPAA, GDPR, SOX, OWASP, PCI_DSS, FDA_21_CFR, ISO_27001, FEDRAMP

# Check compliance for content
result = manager.check_compliance(
    content="Patient SSN: 123-45-6789",
    framework_id="hipaa",
)

print(f"Compliant: {result.compliant}")
print(f"Issues found: {len(result.issues)}")
for issue in result.issues:
    print(f"  - [{issue.severity}] {issue.rule_id}: {issue.description}")
    print(f"    Evidence: {issue.evidence}")
    print(f"    Recommendation: {issue.recommendation}")

# Quick check with convenience function
result = check_compliance(
    content="Credit card: 4111-1111-1111-1111",
    frameworks=["pci_dss", "gdpr"],
)
```

### Custom Compliance Rules

```python
from aragora.compliance import (
    ComplianceFramework,
    ComplianceRule,
    ComplianceSeverity,
    ComplianceFrameworkManager,
)

# Create a custom framework
internal_framework = ComplianceFramework(
    id="internal_security",
    name="Internal Security Policy",
    version="1.0",
    description="Company-specific security requirements",
)

# Add custom rules
internal_framework.add_rule(ComplianceRule(
    id="INT-001",
    name="No Internal IPs in Logs",
    description="Internal IP addresses must not appear in logs",
    severity=ComplianceSeverity.HIGH,
    pattern=r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}",
    recommendation="Redact internal IP addresses before logging",
    category="data_protection",
))

internal_framework.add_rule(ComplianceRule(
    id="INT-002",
    name="No Hardcoded Secrets",
    description="API keys and secrets must not be hardcoded",
    severity=ComplianceSeverity.CRITICAL,
    pattern=r"(api[_-]?key|secret|password)\s*[:=]\s*['\"][^'\"]+['\"]",
    recommendation="Use environment variables or secret management",
    category="security",
))

# Register with manager
manager = ComplianceFrameworkManager()
manager.register_framework(internal_framework)

# Check content
result = manager.check_compliance(
    content="config = {'api_key': 'sk-abc123', 'server': '192.168.1.100'}",
    framework_id="internal_security",
)
```

### Continuous Monitoring

```python
from aragora.compliance import ComplianceMonitor
import asyncio

# Create monitor with alerting integration
monitor = ComplianceMonitor(
    scan_interval_seconds=300,  # Scan every 5 minutes
    frameworks=["hipaa", "gdpr", "pci_dss"],
    alert_on_critical=True,
    alert_webhook="https://hooks.slack.com/...",
)

# Start monitoring
async def run_monitoring():
    await monitor.start()

    # Get current health status
    health = await monitor.get_health()
    print(f"Compliance health: {health.status}")
    print(f"Violation trend: {health.trend}")
    print(f"Critical issues: {health.critical_count}")

    # Get violation history
    history = await monitor.get_violation_history(days=30)
    for day, violations in history.items():
        print(f"{day}: {len(violations)} violations")

    # Check for drift from baseline
    drift = await monitor.check_drift()
    if drift.has_drift:
        print(f"Compliance drift detected: {drift.new_violations}")

    # Stop monitoring
    await monitor.stop()

asyncio.run(run_monitoring())
```

### Integration with Debate Engine

```python
from aragora import Arena, Environment, DebateProtocol
from aragora.compliance import ComplianceFrameworkManager

# Create compliance-aware debate
async def run_compliant_debate(task: str):
    manager = ComplianceFrameworkManager()

    # Check task for compliance before debate
    pre_check = manager.check_compliance(task, framework_id="gdpr")
    if not pre_check.compliant:
        raise ValueError(f"Task contains compliance violations: {pre_check.issues}")

    # Run debate
    arena = Arena(
        environment=Environment(task=task),
        protocol=DebateProtocol(rounds=3),
    )
    result = await arena.run()

    # Check result for compliance
    post_check = manager.check_compliance(
        result.final_answer,
        frameworks=["hipaa", "gdpr"],
    )

    if not post_check.compliant:
        # Log violations but allow result (with warnings)
        for issue in post_check.issues:
            logger.warning(
                "Compliance issue in debate result",
                rule=issue.rule_id,
                severity=issue.severity,
            )

    return result
```

## Integration Points

### With Observability
- Compliance alerts forwarded to alerting system
- Metrics for violation counts and trends
- Audit logs for compliance checks

### With RBAC
- Compliance frameworks can be restricted by role
- Audit trail includes user context

### With Knowledge Mound
- Compliance rules can reference knowledge base
- Violation evidence stored for analysis

### With Audit Logging
- All compliance checks logged immutably
- Violation history preserved for audits

## Pre-Built Rule Categories

### HIPAA Rules
- PHI (Protected Health Information) detection
- Medical record number patterns
- Health condition mentions
- Patient identifier exposure

### GDPR Rules
- Personal data patterns (email, phone, address)
- Consent requirement detection
- Data subject rights mentions
- Cross-border transfer indicators

### PCI-DSS Rules
- Credit card number patterns (Luhn validation)
- CVV/security code detection
- Cardholder name patterns
- Expiration date exposure

### OWASP Rules
- SQL injection patterns
- XSS attack vectors
- Command injection patterns
- Path traversal attempts

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `COMPLIANCE_ENABLED` | Enable compliance checking | `true` |
| `COMPLIANCE_FRAMEWORKS` | Comma-separated framework IDs | `hipaa,gdpr` |
| `COMPLIANCE_SCAN_INTERVAL` | Monitor scan interval (seconds) | `300` |
| `COMPLIANCE_ALERT_WEBHOOK` | Webhook URL for alerts | - |
| `COMPLIANCE_ALERT_SEVERITY` | Minimum severity to alert | `HIGH` |

## See Also

- `aragora/auth/` - Authentication and authorization
- `aragora/rbac/` - Role-based access control
- `aragora/observability/siem.py` - SIEM integration
- `docs/COMPLIANCE.md` - Full compliance guide
