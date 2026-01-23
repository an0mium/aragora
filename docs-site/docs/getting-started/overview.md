---
title: Getting Started with Aragora
description: Getting Started with Aragora
---

# Getting Started with Aragora

> **Last Updated:** 2026-01-18


**Aragora** is the control plane for multi-agent vetted decisionmaking across organizational knowledge and channels. It orchestrates diverse AI models to debate your data and deliver defensible decisions wherever your team works. This is the canonical onboarding guide.

**Choose your path:**
- [Quick Start](#quick-start) - Get running in 5 minutes
- [CLI User Guide](#cli-user-guide) - Run debates from the terminal
- [API Integrator Guide](#api-integrator-guide) - Build on Aragora's API
- [Live Dashboard Dev](../contributing/frontend-development) - Build the Next.js UI
- [TypeScript SDK](../guides/sdk-typescript) - Use the JS client library
- [Gauntlet Guide](#gauntlet-guide) - Stress-test documents and policies
- [Troubleshooting](#troubleshooting) - Fix common issues

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/an0mium/aragora.git
cd aragora
pip install -e .
```

### 2. Configure API Key

Create a `.env` file with at least one AI provider key:

```bash
cp .env.starter .env
```

Edit `.env` (or use `.env.example` for the full template):
```bash
# Required: At least one of these
ANTHROPIC_API_KEY=sk-ant-xxx     # Claude (recommended)
OPENAI_API_KEY=sk-xxx            # GPT-4
GEMINI_API_KEY=AIzaSy...         # Gemini
XAI_API_KEY=xai-xxx              # Grok
MISTRAL_API_KEY=xxx              # Mistral (optional)
OPENROUTER_API_KEY=sk-or-xxx     # OpenRouter (optional)
```

Optional but recommended: keep runtime data out of the repo root.
```bash
ARAGORA_DATA_DIR=.nomic
```

### 3. Verify Setup

```bash
aragora doctor
```

Expected output:
```
Aragora Health Check
====================
API Keys:
  Anthropic: OK
  OpenAI: OK
  Gemini: Not configured
  Grok: Not configured

Environment:
  Python: 3.10.13
  aragora: 0.8.0

Status: Ready
```

### 4. Run Your First Stress-Test

```bash
aragora ask "Should we use microservices or monolith?" \
  --agents anthropic-api,openai-api
```

Expected output:
```
DEBATE: Should we use microservices or monolith?
Agents: anthropic-api, openai-api
Round 1/3...
  [anthropic-api] Proposing...
  [openai-api] Critiquing...
...
CONSENSUS REACHED (75% agreement)
Final Answer: [synthesized recommendation]
```

---

## CLI User Guide

### Basic Usage

```bash
aragora ask "<your question>" --agents <agent1>,<agent2>
```

Note: the CLI default is `codex,claude` (local CLI agents). If you only have API keys,
use `--agents anthropic-api,openai-api` or another API-backed set.

### Available Agents

| Agent | Provider | API Key Required |
|-------|----------|------------------|
| `anthropic-api` | Claude (Anthropic) | `ANTHROPIC_API_KEY` |
| `openai-api` | OpenAI | `OPENAI_API_KEY` |
| `gemini` | Google Gemini | `GEMINI_API_KEY` |
| `grok` | xAI Grok | `XAI_API_KEY` |
| `mistral-api` | Mistral (direct) | `MISTRAL_API_KEY` |
| `codestral` | Mistral (code) | `MISTRAL_API_KEY` |
| `deepseek` | OpenRouter | `OPENROUTER_API_KEY` |
| `qwen` / `qwen-max` | OpenRouter | `OPENROUTER_API_KEY` |
| `kimi` | Moonshot (Kimi) | `KIMI_API_KEY` |
| `ollama` | Local models | None (local) |

Note: OpenRouter agents require `OPENROUTER_API_KEY`. Full catalog is in [AGENTS.md](../core-concepts/agents).

### Common Options

```bash
# More rounds (deeper debate)
aragora ask "..." --rounds 5

# Consensus mode
aragora ask "..." --consensus majority    # 60% agreement (default)
aragora ask "..." --consensus unanimous   # All agents agree
aragora ask "..." --consensus judge       # One agent decides

# Add context
aragora ask "..." --context "Consider latency and cost"

# Verbose output
aragora ask "..." --verbose
```

### Example Debates

```bash
# Architecture decision
aragora ask "Design a caching strategy for 10M users" \
  --agents anthropic-api,openai-api,gemini --rounds 4

# Code review
aragora ask "Review this code for security issues: $(cat myfile.py)" \
  --agents anthropic-api,openai-api --consensus unanimous

# Framework comparison
aragora ask "React vs Vue vs Svelte for our new project" \
  --agents anthropic-api,openai-api,gemini,grok
```

### Code Review

Review pull requests with unanimous AI consensus:

```bash
# Review local changes
git diff main | aragora review

# Review GitHub PR
aragora review https://github.com/owner/repo/pull/123

# Demo mode (no API keys)
aragora review --demo
```

### Start Dashboard

```bash
aragora serve
# Open http://localhost:8080
```

---

## API Integrator Guide

### Start the Server

```bash
aragora serve --api-port 8080 --ws-port 8765
```

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/debates` | POST | Start a new debate |
| `/api/debates/\{id\}` | GET | Get debate status/results |
| `/api/debates` | GET | List recent debates |
| `/api/agents` | GET | List available agents |
| `/api/health` | GET | Health check |

### Start a Debate

```bash
curl -X POST http://localhost:8080/api/debates \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Should we use microservices or monolith?",
    "agents": ["anthropic-api", "openai-api"],
    "rounds": 3,
    "consensus": "majority"
  }'
```

Response:
```json
{
  "debate_id": "debate-20260111-abc123",
  "status": "running",
  "task": "Should we use microservices or monolith?"
}
```

### Get Debate Results

```bash
curl http://localhost:8080/api/debates/debate-20260111-abc123
```

Response:
```json
{
  "debate_id": "debate-20260111-abc123",
  "status": "completed",
  "task": "Should we use microservices or monolith?",
  "consensus": {
    "reached": true,
    "agreement": 0.75,
    "final_answer": "..."
  },
  "rounds": [...],
  "participants": ["anthropic-api", "openai-api"]
}
```

### WebSocket Streaming

For real-time debate updates:

```javascript
const ws = new WebSocket('ws://localhost:8765/ws');
const loopId = 'debate-20260111-abc123';

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (['connection_info', 'loop_list', 'sync'].includes(data.type)) return;

  const eventLoopId = data.loop_id || data.data?.debate_id || data.data?.loop_id;
  if (eventLoopId && eventLoopId !== loopId) return;

  console.log(data.type, data.data);
  // Types: debate_start, agent_message, critique, vote, consensus, debate_end
};
```

### Authentication

For protected endpoints, use Bearer token:

```bash
curl http://localhost:8080/api/protected \
  -H "Authorization: Bearer YOUR_TOKEN"
```

Generate a token:
```bash
curl -X POST http://localhost:8080/api/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'
```

### Full API Reference

See [API_REFERENCE.md](../api/reference) for the full endpoint catalog.

---

## Gauntlet Guide

The **Gauntlet** stress-tests documents, policies, and code using 12+ AI agents simulating hackers, regulators, and critics.

### Quick Demo

Run with simulated agents (no API keys needed):

```bash
python examples/gauntlet_demo.py
```

### Stress-Test Your Documents

```bash
# CLI usage
aragora gauntlet my_policy.md --persona gdpr

# With specific profile
aragora gauntlet my_spec.md --profile thorough

# Code security review
aragora gauntlet src/auth.py --profile code
```

### Gauntlet Profiles

| Profile | Duration | Best For |
|---------|----------|----------|
| `quick` | 2 min | Fast validation |
| `default` | 5 min | Balanced analysis |
| `thorough` | 15 min | Comprehensive review |
| `code` | 10 min | Security-focused code |
| `policy` | 10 min | Compliance-focused |

### Regulatory Personas

| Persona | Focus |
|---------|-------|
| `gdpr` | GDPR compliance (consent, data rights, transfers) |
| `hipaa` | HIPAA compliance (PHI, safeguards, breach) |
| `ai_act` | EU AI Act (risk levels, transparency, bias) |
| `security` | Security vulnerabilities (injection, auth, crypto) |
| `soc2` | SOC 2 controls (security, availability) |
| `pci_dss` | PCI DSS (cardholder data, encryption) |

### Gauntlet via API

```bash
# Start a gauntlet run
curl -X POST http://localhost:8080/api/gauntlet/run \
  -H "Content-Type: application/json" \
  -d '{
    "input_content": "Your policy content here...",
    "input_type": "policy",
    "persona": "gdpr",
    "profile": "default"
  }'

# Get status
curl http://localhost:8080/api/gauntlet/\{id\}

# Get Decision Receipt
curl http://localhost:8080/api/gauntlet/\{id\}/receipt?format=html

# Get Risk Heatmap
curl http://localhost:8080/api/gauntlet/\{id\}/heatmap?format=svg
```

### Understanding Results

The Decision Receipt contains:
- **Verdict**: APPROVED, APPROVED_WITH_CONDITIONS, NEEDS_REVIEW, or REJECTED
- **Risk Score**: 0-100% risk assessment
- **Findings**: Issues categorized by severity (Critical, High, Medium, Low)
- **Mitigations**: Recommended fixes for each finding
- **Audit Trail**: Full evidence chain for compliance

---

## Troubleshooting

### "No API key found"

**Error**: `No API key configured for anthropic-api`

**Solution**: Set at least one key in `.env` or environment:
```bash
export ANTHROPIC_API_KEY=your-key
# Or add to .env file
```

**Verify**: Run `aragora doctor` to check configuration.

### "Agent timed out"

**Error**: `Agent anthropic-api timed out after 60s`

**Cause**: API provider is slow or overloaded.

**Solutions**:
1. Increase timeout:
   ```bash
   export ARAGORA_DEBATE_TIMEOUT=1200  # seconds
   ```
2. Use fewer agents
3. Try a different provider

### "Rate limit exceeded"

**Error**: `Rate limit exceeded for openai-api`

**Solutions**:
1. Wait 60 seconds and retry
2. Use fewer agents
3. Use fallback providers:
   ```bash
   # OpenRouter provides fallback access
   export OPENROUTER_API_KEY=sk-or-xxx
   ```

### "Connection refused on port 8080"

**Error**: `Connection refused on localhost:8080`

**Cause**: Server not running or port in use.

**Solutions**:
1. Start the server:
   ```bash
   aragora serve
   ```
2. Use a different port:
   ```bash
   aragora serve --api-port 8081
   ```
3. Check what's using the port:
   ```bash
   lsof -i :8080
   ```

### "Invalid API key"

**Error**: `Invalid API key for anthropic-api`

**Cause**: API key is malformed or expired.

**Solutions**:
1. Verify the key format:
   - Anthropic: `sk-ant-api03-...`
   - OpenAI: `sk-...`
   - Gemini: `AIzaSy...`
2. Regenerate the key in your provider dashboard
3. Check for extra whitespace in `.env`

### "Module not found"

**Error**: `ModuleNotFoundError: No module named 'aragora'`

**Solution**: Install in development mode:
```bash
pip install -e .
```

### Still stuck?

1. Run diagnostics: `aragora doctor`
2. Check logs: `tail -f ~/.aragora/logs/aragora.log`
3. File an issue: https://github.com/an0mium/aragora/issues

---

## Next Steps

- **Deep Dive**: [Architecture Guide](../core-concepts/architecture)
- **All Options**: [Environment Variables](./environment)
- **Full API**: [API Reference](../api/reference)
- **Gauntlet Details**: [Gauntlet Guide](../guides/gauntlet)
- **Self-Improvement**: [Nomic Loop](../admin/nomic-loop)
- **Custom Agents**: [Custom Agents Guide](../guides/custom-agents)

---

## Quick Reference

### CLI Commands

| Command | Description |
|---------|-------------|
| `aragora ask "..."` | Run a debate |
| `aragora review` | Review code/PR |
| `aragora gauntlet` | Stress-test documents |
| `aragora serve` | Start dashboard |
| `aragora doctor` | Check health |
| `aragora status` | Show environment |
| `aragora config` | Manage settings |

### Environment Variables

**Quick Reference** (see [ENVIRONMENT.md](./environment) for complete reference):

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Claude API key | At least one |
| `OPENAI_API_KEY` | GPT-4 API key | At least one |
| `GEMINI_API_KEY` | Gemini API key | Optional |
| `XAI_API_KEY` | Grok API key | Optional |
| `MISTRAL_API_KEY` | Mistral API key | Optional |
| `OPENROUTER_API_KEY` | Fallback provider | Recommended |

**Configuration Templates:**
- `.env.starter` - Minimal config to get started
- `.env.example` - Full configuration reference

### Health Endpoints

| Endpoint | Description |
|----------|-------------|
| `/healthz` | Kubernetes liveness probe |
| `/readyz` | Kubernetes readiness probe |
| `/api/health` | Comprehensive health check |
| `/api/health/detailed` | Detailed component status |
