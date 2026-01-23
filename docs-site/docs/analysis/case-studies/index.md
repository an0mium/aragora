---
title: Aragora Example Scenarios
description: Aragora Example Scenarios
---

# Aragora Example Scenarios

Hypothetical examples demonstrating how Aragora's Gauntlet can be used for adversarial validation.

> **Note:** These are illustrative scenarios showing potential use cases. The companies and specific findings described are fictional examples for demonstration purposes.

## Examples

| Scenario | Use Case | Key Capability Demonstrated |
|----------|----------|----------------------------|
| [Architecture Stress-Test](./architecture-stress-test) | Scaling & HIPAA validation | Multi-persona coverage, regulatory depth |
| [API Security Review](./security-api-review) | Pre-launch security validation | BOLA detection, heterogeneous validation |
| [GDPR Compliance Audit](./gdpr-compliance-audit) | EU market readiness | Regulatory specificity, Article citations |
| [Strategic Positioning](./epic-strategic-debate) | Internal decision-making | Adversarial critique for strategy |

## How to Use These Examples

These scenarios illustrate:
- What types of issues Aragora can identify
- How to configure Gauntlet for different use cases
- What output formats to expect (Decision Receipts, evidence chains)
- The value of heterogeneous multi-model validation

## Running Your Own Validation

```bash
# Security review
aragora gauntlet your-spec.md --persona security --profile thorough

# Compliance check
aragora gauntlet your-docs.md --persona gdpr --profile thorough

# Architecture review
aragora gauntlet architecture.md --persona security --profile thorough --focus infrastructure
```

See the [Gauntlet documentation](../../guides/gauntlet) for full configuration options.
