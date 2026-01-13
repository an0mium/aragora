# Gauntlet User Guide

> **Note**: For a quick overview, see [GAUNTLET.md](./GAUNTLET.md). This guide provides detailed usage examples and advanced patterns.

Gauntlet is Aragora's adversarial stress-testing system. It throws 12+ AI "attackers" at your specifications to find flaws before they become production failures.

## What is Gauntlet?

Gauntlet simulates:
- **Security Hackers** - Finding exploits and vulnerabilities
- **Regulatory Auditors** - Checking compliance (GDPR, HIPAA, SOC2)
- **Devil's Advocates** - Challenging core assumptions
- **Edge Case Hunters** - Finding boundary conditions that break
- **Scalability Critics** - Testing performance claims

**Output**: A Decision Receipt - an audit-ready document with findings, risk scores, and recommendations.

## Quick Start

### CLI

```bash
# Basic stress-test
aragora gauntlet spec.md

# Thorough analysis with HTML output
aragora gauntlet architecture.md --profile thorough --format html --output review.html

# Quick validation (faster, less comprehensive)
aragora gauntlet feature.md --profile quick

# Security-focused review
aragora gauntlet api_design.md --input-type architecture --verify
```

### API

```bash
# Start Gauntlet via API
curl -X POST http://localhost:8080/api/gauntlet/run \
  -H "Content-Type: application/json" \
  -d '{
    "input_content": "# My Spec\n\nThis is my specification...",
    "input_type": "spec",
    "profile": "thorough"
  }'

# Get results
curl http://localhost:8080/api/gauntlet/{id}

# Get Decision Receipt
curl http://localhost:8080/api/gauntlet/{id}/receipt?format=md
```

### Python

```python
import asyncio
from aragora.agents.base import create_agent
from aragora.gauntlet import GauntletRunner, GauntletConfig, AttackCategory, DecisionReceipt

def agent_factory(name: str):
    return create_agent(name, name=f"{name}_gauntlet", role="auditor")

config = GauntletConfig(
    agents=["anthropic-api", "openai-api"],
    attack_categories=[AttackCategory.SECURITY, AttackCategory.LOGIC],
)

runner = GauntletRunner(config, agent_factory=agent_factory)
result = asyncio.run(runner.run("Your specification content here..."))

print(f"Verdict: {result.verdict.value}")
print(f"Confidence: {result.confidence:.0%}")
print(f"Findings: {len(result.vulnerabilities)}")

receipt = DecisionReceipt.from_result(result)
print(receipt.to_markdown())
```

## Input Types

| Type | Use For | Example |
|------|---------|---------|
| `spec` | Feature specifications | PRD, RFC, design doc |
| `architecture` | System designs | Architecture diagrams, tech specs |
| `policy` | Business policies | Security policies, compliance docs |
| `code` | Source code | Python, JavaScript, etc. |
| `strategy` | Business decisions | Go-to-market, pricing strategies |
| `contract` | Agreements | API contracts, SLAs |

```bash
aragora gauntlet document.md --input-type architecture
```

## Profiles

### Quick (2-5 minutes)
- 2 attack rounds
- Basic red-team
- No deep audit
- Good for: Initial checks, drafts

```bash
aragora gauntlet spec.md --profile quick
```

### Default (5-15 minutes)
- 3 attack rounds
- Standard red-team + probing
- Light deep audit
- Good for: Most documents

```bash
aragora gauntlet spec.md  # Default profile
```

### Thorough (15-45 minutes)
- 5 attack rounds
- Full red-team + probing + deep audit
- Formal verification (if applicable)
- Good for: Production decisions, compliance

```bash
aragora gauntlet spec.md --profile thorough --verify
```

### Code Review
- Security-focused attacks
- Static analysis patterns
- Edge case probing
- Good for: Code, API designs

```bash
aragora gauntlet code.py --profile code --input-type code
```

## Understanding Results

### Verdicts

| Verdict | Meaning | Action |
|---------|---------|--------|
| **APPROVED** | No critical issues, low risk | Proceed with confidence |
| **APPROVED_WITH_CONDITIONS** | Issues found, mitigatable | Address findings before proceeding |
| **NEEDS_REVIEW** | Significant concerns | Human review required |
| **REJECTED** | Critical flaws found | Major revision needed |

### Confidence Scores

- **>80%**: High confidence in verdict
- **60-80%**: Moderate confidence, some uncertainty
- **<60%**: Low confidence, consider more analysis

### Risk Scores

- **0.0-0.3**: Low risk
- **0.3-0.6**: Medium risk
- **0.6-0.8**: High risk
- **0.8-1.0**: Critical risk

### Severity Levels

| Severity | Description |
|----------|-------------|
| **CRITICAL** | Showstopper - must fix before any proceed |
| **HIGH** | Significant - should fix before production |
| **MEDIUM** | Notable - plan to address |
| **LOW** | Minor - nice to fix |
| **INFO** | Observations - for awareness |

## Decision Receipts

The Decision Receipt is an audit-ready document containing:

### Header
- Receipt ID and timestamp
- Input hash (for integrity)
- Configuration used

### Summary
- Verdict with confidence
- Risk and robustness scores
- Finding counts by severity

### Findings
- Each finding with:
  - Severity level
  - Category (security, compliance, logic, etc.)
  - Description
  - Recommended mitigation

### Provenance Chain
- Who analyzed what
- When each phase completed
- Evidence hashes for verification

### Export Formats

```bash
# JSON (for programmatic use)
aragora gauntlet spec.md --format json --output receipt.json

# Markdown (for documentation)
aragora gauntlet spec.md --format md --output receipt.md

# HTML (for presentation)
aragora gauntlet spec.md --format html --output receipt.html
```

## Regulatory Personas

Gauntlet includes specialized compliance personas:

### GDPR
```bash
aragora gauntlet privacy_policy.md --input-type policy
```
Checks: Data subject rights, lawful basis, data minimization, retention policies

### HIPAA
```bash
aragora gauntlet health_app_spec.md
```
Checks: PHI handling, access controls, audit trails, encryption requirements

### SOC2
```bash
aragora gauntlet infrastructure.md --input-type architecture
```
Checks: Trust services criteria, availability, security, confidentiality

### EU AI Act
```bash
aragora gauntlet ml_model_spec.md
```
Checks: Risk classification, transparency requirements, human oversight

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/gauntlet.yml
name: Gauntlet Review

on:
  pull_request:
    paths:
      - 'docs/specs/**'
      - 'docs/architecture/**'

jobs:
  gauntlet:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Aragora
        run: pip install aragora

      - name: Run Gauntlet
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          for spec in docs/specs/*.md; do
            aragora gauntlet "$spec" --profile quick --format json --output "${spec%.md}_review.json"
          done

      - name: Check for Critical Issues
        run: |
          for review in docs/specs/*_review.json; do
            if jq -e '.severity_counts.critical > 0' "$review" > /dev/null; then
              echo "Critical issues found in $review"
              exit 1
            fi
          done

      - name: Upload Reviews
        uses: actions/upload-artifact@v4
        with:
          name: gauntlet-reviews
          path: docs/specs/*_review.json
```

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Find modified spec files
specs=$(git diff --cached --name-only | grep -E '\.md$' | grep -E 'spec|architecture')

for spec in $specs; do
  echo "Running Gauntlet on $spec..."
  if ! aragora gauntlet "$spec" --profile quick --format json | jq -e '.passed' > /dev/null; then
    echo "Gauntlet found issues in $spec. Run 'aragora gauntlet $spec' for details."
    exit 1
  fi
done
```

## Advanced Options

### Custom Agent Selection

```bash
aragora gauntlet spec.md --agents anthropic-api,openai-api,gemini,grok
```

### Disable Specific Phases

```bash
# Skip red-team (only probing + audit)
aragora gauntlet spec.md --no-redteam

# Skip capability probing
aragora gauntlet spec.md --no-probing

# Skip deep audit
aragora gauntlet spec.md --no-audit
```

### Timeout Control

```bash
# Set maximum duration (seconds)
aragora gauntlet spec.md --timeout 1800  # 30 minutes
```

### Formal Verification

```bash
# Enable Z3 SMT solver verification
aragora gauntlet spec.md --verify

# This adds:
# - Logical consistency checks
# - Constraint satisfaction
# - Proof generation for key claims
```

## Troubleshooting

### "No agents available"

Set API keys:
```bash
export ANTHROPIC_API_KEY=sk-ant-xxx
export OPENAI_API_KEY=sk-xxx
```

### "Gauntlet timeout"

Increase timeout or use quick profile:
```bash
aragora gauntlet spec.md --timeout 3600 --profile quick
```

### "Too many findings"

Review the most critical first:
```bash
aragora gauntlet spec.md --format json | jq '.findings | sort_by(.severity) | reverse | .[0:10]'
```

### "Verification failed"

Formal verification requires specific claim formats. Check `docs/FORMAL_VERIFICATION.md`.

## Best Practices

1. **Start Quick**: Use `--profile quick` for drafts
2. **Iterate**: Address critical findings, re-run
3. **Go Thorough**: Use `--profile thorough` before production
4. **Save Receipts**: Archive Decision Receipts for compliance
5. **CI Integration**: Automate for spec changes
6. **Review Dissent**: Dissenting views often reveal blind spots

---

*Stress-test before the market does.*
