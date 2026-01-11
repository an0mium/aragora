# Aragora Quickstart Guide

Get up and running with Aragora in 5 minutes.

Aragora is the AI red team for high-stakes decisions. Gauntlet is the fastest path to a decision stress-test.

## Installation

### Via pip (recommended)

```bash
pip install aragora
```

### Via Docker

```bash
docker pull aragora/aragora:latest
docker run -it aragora/aragora:latest
```

---

## Configuration

Set your API keys for the AI providers you want to use:

```bash
# Required: At least one provider
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"

# Optional: Additional providers
export GEMINI_API_KEY="your-key-here"
export XAI_API_KEY="your-key-here"
export OPENROUTER_API_KEY="your-key-here"  # Fallback provider
```

---

## Your First Gauntlet Run

### 1. Create a spec file

Create `spec.md` with the decision you want to stress-test:

```markdown
# User Authentication System

## Overview
Users can sign up with email/password or OAuth (Google, GitHub).
Sessions expire after 7 days of inactivity.

## Security Requirements
- Passwords must be hashed with bcrypt (cost 12)
- Rate limit: 5 failed attempts, then 15-minute lockout
- MFA optional but encouraged

## Data Storage
- User data stored in PostgreSQL
- Sessions stored in Redis
- Passwords never logged
```

### 2. Run Gauntlet

```bash
aragora gauntlet spec.md
```

### 3. Review results

```
GAUNTLET STRESS-TEST RESULT
============================

ID: gauntlet-20260111-abc123
Input Type: spec

VERDICT: APPROVED_WITH_CONDITIONS
Confidence: 78%

--- Scores ---
Risk Score: 35%
Robustness Score: 82%
Coverage Score: 91%

--- Findings ---
Critical: 0
High: 2
Medium: 4
Low: 3

HIGH ISSUES:
  - No mention of password reset token expiration
  - Redis session store lacks encryption at rest

Duration: 45.2s
Agents: claude, gpt-4, gemini
```

---

## Using Regulatory Personas

Run with a specific compliance persona:

```bash
# GDPR compliance check
aragora gauntlet spec.md --persona gdpr

# HIPAA compliance check
aragora gauntlet spec.md --persona hipaa

# Security-focused review
aragora gauntlet spec.md --persona security
```

---

## Programmatic Usage

```python
import asyncio
from aragora.gauntlet import GauntletRunner, GauntletConfig, AttackCategory, DecisionReceipt

async def main():
    config = GauntletConfig(
        agents=["anthropic-api", "openai-api", "gemini"],
        attack_categories=[
            AttackCategory.SECURITY,
            AttackCategory.LOGIC,
            AttackCategory.COMPLIANCE,
        ],
    )

    runner = GauntletRunner(config)

    spec = open("spec.md").read()
    result = await runner.run(spec)

    print(f"Verdict: {result.verdict.value}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Findings: {len(result.vulnerabilities)}")

    # Get decision receipt
    receipt = DecisionReceipt.from_result(result)
    print(receipt.to_markdown())

asyncio.run(main())
```

---

## Running the Server

Start the full Aragora server with API and WebSocket support:

```bash
aragora serve
```

Then use the REST API:

```bash
# Start a gauntlet run
curl -X POST http://localhost:8080/api/gauntlet/run \
  -H "Content-Type: application/json" \
  -d '{"input_content": "...", "input_type": "spec"}'

# Get status
curl http://localhost:8080/api/gauntlet/gauntlet-abc123

# Get decision receipt
curl http://localhost:8080/api/gauntlet/gauntlet-abc123/receipt
```

---

## Common Options

### Profiles

```bash
# Quick pass
aragora gauntlet spec.md --profile quick

# Deep pass
aragora gauntlet spec.md --profile thorough

# Code review
aragora gauntlet src/auth.py --profile code --input-type code
```

### Feature Toggles

```bash
# Disable red-team attacks
aragora gauntlet spec.md --no-redteam

# Disable probing
aragora gauntlet spec.md --no-probing

# Disable deep audit
aragora gauntlet spec.md --no-audit

# Enable formal verification
aragora gauntlet spec.md --verify
```

### Output Formats

```bash
# JSON output
aragora gauntlet spec.md --output receipt.json --format json

# Markdown output
aragora gauntlet spec.md --output receipt.md --format md

# HTML output (format inferred from extension)
aragora gauntlet spec.md --output receipt.html
```

---

## Next Steps

1. **Explore profiles/personas** - See options with `aragora gauntlet --help`
2. **Create custom personas** - Define your own compliance frameworks
3. **Set up CI/CD integration** - Run Gauntlet on every PR
4. **Join the community** - [GitHub Discussions](https://github.com/aragora/aragora/discussions)

---

## Getting Help

- **Documentation:** https://docs.aragora.ai
- **GitHub Issues:** https://github.com/aragora/aragora/issues
- **Discord:** https://discord.gg/aragora
- **Email:** support@aragora.ai
