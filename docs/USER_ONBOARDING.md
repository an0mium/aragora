# Aragora User Onboarding Guide

Get started with Aragora in 5 minutes. This guide walks you through installation, your first debate, and understanding results.

## Who Is This For?

- **Developers** wanting AI-assisted code review and architecture decisions
- **Architects** stress-testing technical proposals
- **Product Managers** validating feature specifications
- **Security Teams** running adversarial analysis on systems

## Quick Install

### Option 1: pip (Recommended)

```bash
pip install aragora
```

### Option 2: From Source

```bash
git clone https://github.com/an0mium/aragora.git
cd aragora
pip install -e .
```

### Option 3: Docker

```bash
docker pull aragora/aragora:latest
docker run -p 8080:8080 -e ANTHROPIC_API_KEY=sk-ant-xxx aragora/aragora
```

## Configure API Keys

Aragora needs at least one AI provider. Create `.env`:

```bash
# Required: Pick at least one
ANTHROPIC_API_KEY=sk-ant-xxx     # Claude (recommended)
OPENAI_API_KEY=sk-xxx            # GPT-4

# Optional: More perspectives
GEMINI_API_KEY=AIzaSy...         # Google Gemini
XAI_API_KEY=xai-xxx              # Grok
MISTRAL_API_KEY=xxx              # Mistral
OPENROUTER_API_KEY=xxx           # Access to 50+ models
```

## Your First Debate (2 Minutes)

### Command Line

```bash
aragora ask "Should we use PostgreSQL or MongoDB for our user data?" \
  --agents anthropic-api,openai-api
```

**Output:**
```
DEBATE: Should we use PostgreSQL or MongoDB for our user data?
Agents: anthropic-api, openai-api
Round 1/3...
  [anthropic-api] PostgreSQL for ACID compliance and complex queries...
  [openai-api] Agreeing, but noting MongoDB's flexibility for evolving schemas...
Round 2/3...
  [anthropic-api] Counterpoint: PostgreSQL's JSONB handles schema flexibility...
  [openai-api] Valid point. PostgreSQL preferred for relational user data...
Round 3/3...
  [anthropic-api] Synthesizing: PostgreSQL with JSONB for metadata...
  [openai-api] Concurring. Clear winner for user data use case...

CONSENSUS REACHED (85% agreement)
Recommendation: Use PostgreSQL with JSONB columns for flexible metadata.
Key factors: ACID compliance, mature tooling, query flexibility.
```

### Web Dashboard

```bash
# Start the server
aragora serve

# Open in browser
open http://localhost:8080
```

The dashboard shows:
- Live debate progress
- Agent voting and positions
- Consensus visualization
- Historical debates

## Understanding Results

### Consensus Levels

| Level | Meaning |
|-------|---------|
| **Strong (>80%)** | Clear recommendation, high confidence |
| **Moderate (60-80%)** | Recommendation with caveats |
| **Weak (<60%)** | No clear winner, consider more research |

### Key Metrics

- **Agreement Ratio**: How aligned the agents are
- **Confidence**: Certainty in the recommendation
- **Dissent Count**: Number of significant disagreements
- **Rounds**: How many debate iterations occurred

### Reading Dissent

Dissenting views are valuable. They highlight:
- Edge cases the majority missed
- Alternative approaches worth considering
- Risks in the consensus position

## Common Use Cases

### 1. Architecture Decisions

```bash
aragora ask "Microservices vs monolith for our 50-person startup" \
  --agents anthropic-api,openai-api,gemini \
  --rounds 4 \
  --context "We expect 10x growth in 2 years"
```

### 2. Code Review

```bash
aragora ask "Review this code for security issues:" \
  --agents anthropic-api,openai-api \
  --file src/auth/login.py \
  --consensus unanimous
```

### 3. Spec Validation (Gauntlet)

```bash
aragora gauntlet spec.md \
  --profile thorough \
  --format html \
  --output review.html
```

### 4. Risk Assessment

```bash
aragora ask "What could go wrong with this deployment plan?" \
  --agents anthropic-api,openai-api,gemini,grok \
  --context "$(cat deployment.md)"
```

## Dashboard Tour

### Home Page
- Recent debates with outcomes
- Quick-start debate form
- Agent health status

### Debate View
- Real-time message stream
- Agent stance indicators
- Voting progress bar
- Final consensus card

### Analytics
- Agent performance over time
- Topic trends
- Consensus patterns

### Gauntlet
- Stress-test specifications
- Risk heatmaps
- Decision receipts

## Next Steps

| Goal | Guide |
|------|-------|
| Run adversarial stress-tests | [GAUNTLET.md](GAUNTLET.md) |
| Build custom agents | [CUSTOM_AGENTS.md](CUSTOM_AGENTS.md) |
| Troubleshoot issues | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| API integration | [API_REFERENCE.md](API_REFERENCE.md) |
| Self-improving debates | [NOMIC_LOOP.md](NOMIC_LOOP.md) |

## Getting Help

- **GitHub Issues**: [github.com/an0mium/aragora/issues](https://github.com/an0mium/aragora/issues)
- **Documentation**: [docs/](.)
- **Examples**: https://github.com/aragora/aragora/tree/main/examples

---

*Welcome to Aragora. Let the debate begin.*
