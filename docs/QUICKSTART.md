# Aragora Quick Start Guide

Get started with Aragora in under 5 minutes.

**Aragora** is an **Adversarial Validation Engine** that stress-tests high-stakes decisions through multi-agent debate. Use it to:
- Run debates between AI agents on any topic
- Stress-test policies, specs, and code with the **Gauntlet**
- Generate compliance-ready Decision Receipts

---

## 1. Install

```bash
git clone https://github.com/an0mium/aragora.git
cd aragora
pip install -e .
```

## 2. Set API Key

You need at least one AI provider key. Create a `.env` file:

```bash
cp .env.example .env
```

Add your key:

```bash
# Pick one (or more):
ANTHROPIC_API_KEY=sk-ant-xxx     # Claude
OPENAI_API_KEY=sk-xxx            # GPT-4
GEMINI_API_KEY=AIzaSy...         # Gemini
XAI_API_KEY=xai-xxx              # Grok
```

## 3. Run Your First Debate

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

## 4. Explore Results

### View in Terminal
Results are printed directly. For longer debates, pipe to a file:
```bash
aragora ask "..." --agents anthropic-api,openai-api > debate.log
```

### Start Live Dashboard
```bash
aragora serve
# Open http://localhost:8080
```

### View Recorded Replays
```bash
curl http://localhost:8080/api/replays
# Fetch a specific replay
curl http://localhost:8080/api/replays/<replay-id>
```

## Common Options

```bash
# More agents
--agents anthropic-api,openai-api,gemini,grok

# More rounds (deeper debate)
--rounds 5

# Different consensus
--consensus majority   # Default: 60% agreement
--consensus unanimous  # All agents agree
--consensus judge      # One agent decides

# Add context
--context "Include latency and cost constraints"

# Disable learning
--no-learn

# Verbose output (global)
--verbose
```

## Example Debates

```bash
# Technical architecture
aragora ask "Design a caching strategy for 10M users" \
  --agents anthropic-api,openai-api,gemini --rounds 4

# Code review
aragora ask "Review this code for security issues: $(cat myfile.py)" \
  --agents anthropic-api,openai-api --consensus unanimous

# Decision making
aragora ask "React vs Vue vs Svelte for our new project" \
  --agents anthropic-api,openai-api,gemini,grok
```

## Next Steps

- **Run the Nomic Loop (experimental):** Self-improving debates - see `docs/NOMIC_LOOP.md`
- **API Integration:** Build on Aragora - see `docs/API_REFERENCE.md`
- **Configuration:** All options - see `docs/ENVIRONMENT.md`
- **Architecture:** How it works - see `docs/ARCHITECTURE.md`

## Troubleshooting

### "No API key found"
Set at least one key in `.env` or environment:
```bash
export ANTHROPIC_API_KEY=your-key
```

### "Agent timed out"
Increase the debate timeout via environment:
```bash
export ARAGORA_DEBATE_TIMEOUT=1200  # seconds
```

### "Rate limit exceeded"
Wait a moment or use fewer agents. API providers have rate limits.

### "Connection refused on port 8080"
Another service is using that port. Use a different port:
```bash
aragora serve --api-port 8081
```

---

## Gauntlet: Stress-Test Documents

The **Gauntlet** is Aragora's adversarial validation engine. It stress-tests policies, specs, and code using 12+ AI agents simulating hackers, regulators, and critics.

### Quick Demo

Run with simulated agents (no API keys needed):

```bash
python scripts/demo_gauntlet.py
```

This generates a Decision Receipt showing:
- **Verdict**: APPROVED, NEEDS_REVIEW, or REJECTED
- **Findings**: Issues categorized by severity
- **Risk Score**: Overall risk assessment

### Real Documents

Stress-test your own documents:

```bash
# Quick validation (~30 sec)
python scripts/demo_gauntlet.py my_policy.md

# Thorough analysis (~15 min, with real APIs)
python scripts/demo_gauntlet.py my_spec.md --profile thorough --real-apis

# Code security review
python scripts/demo_gauntlet.py src/auth.py --profile code --real-apis
```

### Gauntlet Profiles

| Profile | Time | Best For |
|---------|------|----------|
| `demo` | 30 sec | Quick demos |
| `quick` | 2 min | Fast validation |
| `thorough` | 15 min | Comprehensive analysis |
| `code` | 5 min | Security-focused code review |
| `policy` | 5 min | Compliance-focused policy review |

### Understanding Results

Results are saved as Decision Receipts in multiple formats:

```bash
# After running, open the HTML report:
open gauntlet_receipt_*.html
```

The receipt contains:
- Verdict and confidence score
- All findings with severity levels
- Recommended mitigations
- Full audit trail for compliance

### Gauntlet via API

```bash
# Start server
python -m aragora.server.unified_server --port 8080

# Run Gauntlet
curl -X POST http://localhost:8080/api/gauntlet/run \
  -H "Content-Type: application/json" \
  -d '{"input_text": "Your policy here...", "template": "quick"}'

# Get results
curl http://localhost:8080/api/gauntlet/{id}

# Get Decision Receipt
curl http://localhost:8080/api/gauntlet/{id}/receipt?format=html
```

See [STATUS.md](./STATUS.md) for complete Gauntlet documentation
