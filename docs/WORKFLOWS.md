# Aragora Workflows

Common workflows for using Aragora in your development process.

## Quick Reference

| Workflow | Command | Description |
|----------|---------|-------------|
| Run a debate | `aragora ask "question" --agents anthropic-api,openai-api` | Decision stress-test engine (exploratory) |
| Stress-test a spec | `aragora gauntlet spec.md --persona security` | Adversarial validation (primary workflow) |
| Check configuration | `aragora doctor` | Diagnose API keys and setup |
| List agents | `aragora agents` | Show available agent types |
| Start server | `aragora serve` | Run HTTP/WebSocket API |

---

## 1. Running a Multi-Agent Debate (Exploratory)

Use debates when you want multiple AI agents to critique and improve answers.
For high-stakes decisions, use Gauntlet to produce decision receipts and risk heatmaps.

### Basic Debate

```bash
# Quick debate with default agents (anthropic-api, openai-api)
aragora ask "Should we use microservices or a monolith for a new startup?"

# Specify agents explicitly
aragora ask "Design a rate limiter" --agents anthropic-api,openai-api,gemini

# More debate rounds for complex topics
aragora ask "Design a distributed cache" --agents anthropic-api,openai-api --rounds 5
```

### With Context

```bash
# Provide context from a file
aragora ask "What security vulnerabilities exist?" \
  --agents anthropic-api,openai-api \
  --context "$(cat architecture.md)"
```

### Consensus Modes

```bash
# Unanimous - all agents must agree
aragora ask "Is this approach safe?" --consensus unanimous

# Majority - more than half must agree (default)
aragora ask "Best database choice?" --consensus majority

# Supermajority - 2/3 must agree
aragora ask "Should we migrate?" --consensus supermajority
```

---

## 2. Stress-Testing with Gauntlet

Use gauntlet to adversarially validate specifications, architectures, or policies.

### Basic Usage

```bash
# Stress-test a specification
aragora gauntlet spec.md --input-type spec

# Stress-test architecture documentation
aragora gauntlet architecture.md --input-type architecture

# Stress-test a policy
aragora gauntlet privacy_policy.md --input-type policy
```

### Compliance Personas

Use personas for domain-specific validation:

```bash
# Security-focused analysis
aragora gauntlet spec.md --persona security

# GDPR compliance check
aragora gauntlet privacy_policy.md --persona gdpr

# HIPAA compliance check
aragora gauntlet patient_data.md --persona hipaa

# AI Act compliance
aragora gauntlet ml_system.md --persona ai_act

# Financial (SOX) compliance
aragora gauntlet financial_process.md --persona sox
```

### Analysis Depth

```bash
# Quick scan (5-10 minutes)
aragora gauntlet spec.md --profile quick

# Standard analysis (default)
aragora gauntlet spec.md --profile default

# Thorough deep-dive (30+ minutes)
aragora gauntlet spec.md --profile thorough
```

### Output Formats

```bash
# Generate HTML decision receipt
aragora gauntlet spec.md -o report.html

# Generate JSON for CI/CD integration
aragora gauntlet spec.md -o report.json --format json

# Generate Markdown
aragora gauntlet spec.md -o report.md --format md
```

### CI/CD Integration

Exit codes for automation:
- `0` - Approved
- `1` - Rejected
- `2` - Needs review

```bash
# In CI pipeline
aragora gauntlet spec.md --persona security || exit 1
```

---

## 3. Exporting Results

Export debate artifacts for documentation or sharing.

```bash
# Export as HTML (interactive viewer)
aragora export --debate-id abc123 --format html

# Export as JSON (for processing)
aragora export --debate-id abc123 --format json

# Export as Markdown (for documentation)
aragora export --debate-id abc123 --format md

# Demo export (no actual debate required)
aragora export --demo --format html -o ./exports/
```

---

## 4. Using the Python SDK

Integrate Aragora into your Python applications.

### Basic Debate

```python
from aragora.client import AragoraClient

# Initialize client
client = AragoraClient(base_url="http://localhost:8080")

# Run a debate (blocking)
debate = client.debates.run(
    task="Should we use Redis or Memcached?",
    agents=["anthropic-api", "openai-api"],
    rounds=3
)

print(f"Consensus: {debate.consensus.reached}")
print(f"Answer: {debate.consensus.final_answer}")
```

### Async Usage

```python
import asyncio
from aragora.client import AragoraClient

async def main():
    async with AragoraClient() as client:
        # Create debate (non-blocking)
        response = await client.debates.create_async(
            task="Design an authentication system",
            agents=["anthropic-api", "openai-api"],
        )

        # Poll for completion
        debate = await client.debates.get_async(response.debate_id)
        while debate.status.value == "running":
            await asyncio.sleep(2)
            debate = await client.debates.get_async(response.debate_id)

        print(debate.consensus.final_answer)

asyncio.run(main())
```

### Graph Debates

```python
# Graph debates allow for branching when agents diverge
result = client.graph_debates.create(
    task="Design a distributed system",
    agents=["anthropic-api", "openai-api"],
    max_rounds=5,
    branch_threshold=0.5,  # Create branch if divergence > 50%
)

# Get all branches
branches = client.graph_debates.get_branches(result.debate_id)
for branch in branches:
    print(f"Branch: {branch.name}, Nodes: {len(branch.nodes)}")
```

### Matrix Debates

```python
# Matrix debates explore the same question across scenarios
result = client.matrix_debates.create(
    task="Should we adopt microservices?",
    scenarios=[
        {"name": "small_team", "parameters": {"team_size": 5}},
        {"name": "large_team", "parameters": {"team_size": 50}},
        {"name": "startup", "parameters": {"budget": "low"}},
        {"name": "enterprise", "parameters": {"budget": "high"}},
    ]
)

# Get universal vs conditional conclusions
conclusions = client.matrix_debates.get_conclusions(result.matrix_id)
print("Universal (true in all scenarios):", conclusions.universal)
print("Conditional:", conclusions.conditional)
```

### Formal Verification

```python
# Verify a claim using formal methods
result = client.verification.verify(
    claim="All prime numbers greater than 2 are odd",
    backend="z3",  # or "lean", "coq"
    timeout=30
)

print(f"Status: {result.status}")
if result.status == "valid":
    print(f"Proof: {result.proof}")
elif result.status == "invalid":
    print(f"Counterexample: {result.counterexample}")
```

### Memory Analytics

```python
# Get memory tier analytics
analytics = client.memory.analytics(days=30)

print(f"Total entries: {analytics.total_entries}")
print(f"Learning velocity: {analytics.learning_velocity}")

for tier in analytics.tiers:
    print(f"{tier.tier_name}: {tier.entry_count} entries, {tier.hit_rate:.1%} hit rate")

for rec in analytics.recommendations:
    print(f"[{rec.impact}] {rec.description}")
```

---

## 5. Running the Server

Start the Aragora API server for HTTP/WebSocket access.

### Basic Server

```bash
# Start with defaults
aragora serve

# Custom ports
aragora serve --api-port 8080 --ws-port 8765

# Bind to all interfaces (for Docker/production)
aragora serve --host 0.0.0.0
```

### Health Check

```bash
# Check server status
curl http://localhost:8080/api/health

# Response:
# {"status": "healthy", "version": "1.0.0", "uptime_seconds": 123.4}
```

---

## 6. Batch Processing

Process multiple debates from a file.

### Input Format (JSONL)

```json
{"task": "Design a cache", "agents": ["anthropic-api", "openai-api"]}
{"task": "Design auth system", "agents": ["anthropic-api", "gemini"]}
{"task": "Design rate limiter", "agents": ["openai-api", "gemini"]}
```

### Running Batch

```bash
# Local processing
aragora batch debates.jsonl

# Submit to server
aragora batch debates.jsonl --server --url http://localhost:8080

# Wait for completion
aragora batch debates.jsonl --server --wait
```

---

## 7. Interactive Mode (REPL)

For exploratory sessions with multiple debates.

```bash
# Start interactive mode
aragora repl

# With specific agents
aragora repl --agents anthropic-api,openai-api,gemini
```

In REPL:
```
> Should we use GraphQL or REST?
[Agents debate...]
Final answer: ...

> /history
[Shows debate history]

> /export last
[Exports last debate]

> /quit
```

---

## 8. Configuration

### Environment Variables

```bash
# Required (at least one)
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."

# Optional (enables fallback)
export OPENROUTER_API_KEY="sk-or-..."

# Optional (additional providers)
export GEMINI_API_KEY="..."
export XAI_API_KEY="..."
export MISTRAL_API_KEY="..."
```

### Configuration File

```bash
# Show current config
aragora config show

# Set a value
aragora config set default_agents "anthropic-api,openai-api"

# Get a value
aragora config get default_agents

# Show config file path
aragora config path
```

---

## 9. Diagnostics

### Full System Check

```bash
# Run diagnostics
aragora doctor

# With API key validation (makes test calls)
aragora doctor --validate
```

### Agent Discovery

```bash
# List all available agents
aragora agents

# With detailed descriptions
aragora agents --verbose
```

### Environment Status

```bash
# Check environment health
aragora status

# Check specific server
aragora status --server http://localhost:8080
```

---

## 10. Common Patterns

### Pre-Commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Stress-test changed specs before commit
for file in $(git diff --cached --name-only | grep -E '\.(md|txt)$'); do
    if [[ "$file" == *spec* ]] || [[ "$file" == *architecture* ]]; then
        echo "Stress-testing: $file"
        aragora gauntlet "$file" --profile quick --persona security || exit 1
    fi
done
```

### GitHub Action

```yaml
name: Aragora Validation
on: [pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Aragora
        run: pip install aragora

      - name: Validate specs
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          for spec in docs/specs/*.md; do
            aragora gauntlet "$spec" --profile quick -o "reports/$(basename $spec).html"
          done

      - name: Upload reports
        uses: actions/upload-artifact@v4
        with:
          name: validation-reports
          path: reports/
```

### Docker Compose

```yaml
version: '3.8'
services:
  aragora:
    image: aragora/server:latest
    ports:
      - "8080:8080"
      - "8765:8765"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - aragora_data:/data

volumes:
  aragora_data:
```
