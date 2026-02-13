---
title: Aragora Case Studies & Example Scenarios
description: Aragora Case Studies & Example Scenarios
---

# Aragora Case Studies & Example Scenarios

Worked case studies and hypothetical examples demonstrating Aragora's multi-model review, Gauntlet adversarial validation, and decision receipt capabilities.

> **Note:** These are illustrative scenarios showing potential use cases. The companies and specific findings described are fictional examples for demonstration purposes.

## Case Studies

In-depth worked examples showing the full review pipeline from diff to Decision Receipt.

| Case Study | Use Case | Key Capability Demonstrated |
|------------|----------|----------------------------|
| [Multi-Model Security Review](./security-api-review) | OAuth2 token validation PR review | Model disagreement as signal, timing attack detection |

### Planned Case Studies

| Case Study | Use Case | Status |
|------------|----------|--------|
| Compliance Review | SOC 2 control validation across infrastructure-as-code | Planned |
| Architecture Decision | Microservices vs. monolith migration debate | Planned |
| Dependency Upgrade | Major framework upgrade risk assessment | Planned |

## Example Scenarios

Shorter examples demonstrating specific Gauntlet validation capabilities.

| Scenario | Use Case | Key Capability Demonstrated |
|----------|----------|----------------------------|
| [Architecture Stress-Test](./architecture-stress-test) | Scaling & HIPAA validation | Multi-persona coverage, regulatory depth |
| [API Security Review](./security-api-review) | Pre-launch security validation | BOLA detection, heterogeneous validation |
| [GDPR Compliance Audit](./gdpr-compliance-audit) | EU market readiness | Regulatory specificity, Article citations |
| [Strategic Positioning](./epic-strategic-debate) | Internal decision-making | Adversarial critique for strategy |

## How to Use These Examples

These scenarios illustrate:
- What types of issues Aragora can identify
- How multi-model disagreement surfaces subtle vulnerabilities that single-model review misses
- How to configure Gauntlet and `aragora review` for different use cases
- What output formats to expect (PR comments, Decision Receipts, SARIF, evidence chains)
- The value of heterogeneous multi-model validation

## Running Your Own Validation

```bash
# Multi-model code review (pipes diff to 3 models)
git diff main | aragora review --focus security

# Review a GitHub PR directly
aragora review https://github.com/owner/repo/pull/123

# Code review with Decision Receipt and SARIF export
aragora review https://github.com/owner/repo/pull/123 --gauntlet --sarif results.sarif

# Gauntlet adversarial validation on a spec
aragora gauntlet your-spec.md --persona security --profile thorough

# Compliance check
aragora gauntlet your-docs.md --persona gdpr --profile thorough

# Architecture review
aragora gauntlet architecture.md --persona security --profile thorough --focus infrastructure
```

See `aragora/cli/review.py` and [Gauntlet documentation](../../guides/gauntlet) for full configuration options.
