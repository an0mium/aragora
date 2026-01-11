# Aragora Quickstart Guide

Get up and running with Aragora in 5 minutes.

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
aragora gauntlet run spec.md
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
aragora gauntlet run spec.md --persona gdpr

# HIPAA compliance check
aragora gauntlet run spec.md --persona hipaa

# Security-focused review
aragora gauntlet run spec.md --persona security
```

---

## Programmatic Usage

```python
import asyncio
from aragora import GauntletRunner, GauntletConfig

async def main():
    config = GauntletConfig(
        agents=["anthropic-api", "openai-api", "gemini"],
        attack_categories=["security", "logic", "compliance"],
    )

    runner = GauntletRunner(config)

    spec = open("spec.md").read()
    result = await runner.run(spec)

    print(f"Verdict: {result.verdict}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Findings: {len(result.vulnerabilities)}")

    # Get decision receipt
    receipt = result.to_receipt()
    print(receipt.to_markdown())

asyncio.run(main())
```

---

## Running the Server

Start the full Aragora server with API and WebSocket support:

```bash
aragora serve --port 8080
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

### Attack Categories

```bash
# Security-focused
aragora gauntlet run spec.md --attacks security,injection

# Compliance-focused
aragora gauntlet run spec.md --attacks compliance,gdpr

# Architecture-focused
aragora gauntlet run spec.md --attacks architecture,scalability
```

### Output Formats

```bash
# JSON (default)
aragora gauntlet run spec.md -o json

# Markdown
aragora gauntlet run spec.md -o markdown

# HTML report
aragora gauntlet run spec.md -o html --output-file report.html
```

### Verbosity

```bash
# Quiet mode (verdict only)
aragora gauntlet run spec.md -q

# Verbose mode (show agent reasoning)
aragora gauntlet run spec.md -v

# Debug mode (all internal details)
aragora gauntlet run spec.md --debug
```

---

## Next Steps

1. **Explore attack categories** - See all available attacks with `aragora attacks list`
2. **Create custom personas** - Define your own compliance frameworks
3. **Set up CI/CD integration** - Run Gauntlet on every PR
4. **Join the community** - [GitHub Discussions](https://github.com/aragora/aragora/discussions)

---

## Getting Help

- **Documentation:** https://docs.aragora.ai
- **GitHub Issues:** https://github.com/aragora/aragora/issues
- **Discord:** https://discord.gg/aragora
- **Email:** support@aragora.ai
