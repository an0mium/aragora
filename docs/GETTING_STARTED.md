# Getting Started with Aragora

Run your first adversarial AI debate in under 5 minutes.

---

## Prerequisites

- **Python 3.10+** (check with `python --version`)
- **One API key** from any supported provider: [Anthropic](https://console.anthropic.com/), [OpenAI](https://platform.openai.com/), [Mistral](https://console.mistral.ai/), [Google](https://aistudio.google.com/), or [xAI](https://console.x.ai/)

No API key yet? Skip to [Try without API keys](#try-without-api-keys) below.

---

## Step 1: Install

### Option A: Standalone debate engine (lightweight)

```bash
pip install aragora-debate
```

Add your preferred LLM provider:

```bash
pip install aragora-debate[anthropic]    # Claude
pip install aragora-debate[openai]       # GPT
pip install aragora-debate[mistral]      # Mistral
pip install aragora-debate[all]          # All providers
```

### Option B: Full platform (server + CLI + connectors + SDKs)

```bash
pip install aragora
```

Or from source:

```bash
git clone https://github.com/an0mium/aragora.git
cd aragora && pip install -e .
```

---

## Step 2: Run your first debate

Set your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

Create a file called `first_debate.py`:

```python
import asyncio
from aragora_debate import Debate, create_agent

async def main():
    # Define the question
    debate = Debate("Should we use microservices or a monolith?")

    # Add agents from different LLM providers
    debate.add_agent(create_agent("anthropic", name="analyst"))
    debate.add_agent(create_agent("openai", name="challenger"))

    # Run the debate
    result = await debate.run()

    # Print the decision receipt
    print(result.receipt.to_markdown())

asyncio.run(main())
```

Run it:

```bash
python first_debate.py
```

That is it. Two models just debated your question across multiple rounds of proposals, critiques, and votes, then produced an auditable decision receipt.

---

## Step 3: Read the decision receipt

Every debate produces a `DecisionReceipt` -- a structured record of what was decided, who agreed, who dissented, and why:

```
# Decision Receipt DR-20260212-b7e4a1

**Question:** Should we use microservices or a monolith?
**Verdict:** Approved With Conditions
**Confidence:** 82%
**Consensus:** Reached (supermajority, 78% agreement)
**Agents:** analyst (Claude), challenger (GPT-4o)

## Conditions
- Start with a modular monolith; extract services only at proven boundaries
- Require team size > 15 before splitting into independent services

## Dissenting Views

**challenger:**
- Network latency and operational complexity underestimated
  > "A modular monolith achieves 80% of the benefits with 20% of the
  > risk for teams under 20 engineers."
```

Export in multiple formats:

```python
from aragora_debate import ReceiptBuilder

# JSON (for programmatic access)
print(ReceiptBuilder.to_json(result.receipt))

# HTML (for dashboards or reports)
with open("receipt.html", "w") as f:
    f.write(ReceiptBuilder.to_html(result.receipt))

# Sign for tamper detection (audit compliance)
ReceiptBuilder.sign_hmac(result.receipt, key="your-signing-key")
```

---

## Try without API keys

Use mock agents to explore the debate flow immediately:

```python
import asyncio
from aragora_debate import Debate, create_agent

async def main():
    debate = Debate("Should we use microservices or a monolith?")
    debate.add_agent(create_agent("mock", name="analyst"))
    debate.add_agent(create_agent("mock", name="challenger"))
    result = await debate.run()
    print(result.receipt.to_markdown())

asyncio.run(main())
```

Mock agents return synthetic responses. They are useful for understanding the debate structure before connecting real models.

If you installed the full platform, you can also use demo mode from the CLI:

```bash
aragora review --demo
```

---

## How debates work

Each debate runs multiple rounds. Every round has three phases:

```
Round 1                Round 2                Round 3
+------------------+   +------------------+   +------------------+
| 1. PROPOSE       |   | 1. PROPOSE       |   | 1. PROPOSE       |
|    Each agent     |   |    Address prior  |   |    Final         |
|    responds       |   |    critiques      |   |    positions     |
+------------------+   +------------------+   +------------------+
| 2. CRITIQUE      |   | 2. CRITIQUE      |   | 2. CRITIQUE      |
|    Challenge each |   |    Deeper         |   |    Last          |
|    other's logic  |   |    analysis       |   |    challenges    |
+------------------+   +------------------+   +------------------+
| 3. VOTE          |   | 3. VOTE          |   | 3. VOTE          |
|    Weighted vote  |   |    May stop early |   |    Final tally   |
+------------------+   +------------------+   +------------------+
                                                       |
                                               +-------v--------+
                                               | DECISION       |
                                               | RECEIPT        |
                                               +----------------+
```

Early stopping kicks in when agents reach consensus before all rounds complete. Different models with different training data surface genuinely different failure modes -- when they converge after adversarial challenge, that convergence is meaningful.

---

## Customize the debate

```python
from aragora_debate import Debate, DebateConfig, create_agent

debate = Debate(
    "Should we adopt GraphQL?",
    config=DebateConfig(
        rounds=3,                          # Number of debate rounds
        consensus_method="supermajority",  # How consensus is determined
        early_stopping=True,               # Stop when consensus is reached early
        require_reasoning=True,            # Agents must explain their votes
        timeout_seconds=300,               # Overall timeout
    ),
)
debate.add_agent(create_agent("anthropic", name="analyst"))
debate.add_agent(create_agent("openai", name="critic"))
debate.add_agent(create_agent("mistral", name="synthesizer"))
result = await debate.run()
```

### Consensus methods

| Method | Threshold | Best for |
|--------|-----------|----------|
| `majority` | >50% | General decisions |
| `supermajority` | >66.7% | Important decisions |
| `unanimous` | 100% | Safety-critical decisions |
| `weighted` | Configurable | When agent reliability varies |
| `judge` | N/A | One agent decides after hearing the debate |

---

## Detect hollow consensus

Models sometimes agree without substantive evidence. Enable the Trickster to automatically challenge this:

```python
debate = Debate(
    "Should we adopt Kubernetes?",
    enable_trickster=True,          # Challenge hollow consensus
    enable_convergence=True,        # Track how proposals converge
    trickster_sensitivity=0.7,      # Higher = more interventions
)
debate.add_agent(create_agent("anthropic", name="analyst"))
debate.add_agent(create_agent("openai", name="critic"))
result = await debate.run()
print(f"Trickster interventions: {result.trickster_interventions}")
```

---

## Monitor debates in real time

```python
from aragora_debate import Debate, create_agent

def on_event(event):
    print(f"[{event.event_type.value}] round={event.round_num} {event.data}")

debate = Debate("Should we use Kafka or RabbitMQ?", on_event=on_event)
debate.add_agent(create_agent("mock", name="analyst"))
debate.add_agent(create_agent("mock", name="critic"))
# Events: debate_start, round_start, proposal, critique, vote,
#         consensus_check, convergence_detected, trickster_intervention,
#         round_end, debate_end
```

---

## Full platform: CLI and server

If you installed the full `aragora` package, you get a CLI and API server:

```bash
# AI code review (demo mode, no API keys needed)
git diff main | aragora review --demo

# Stress-test a specification with adversarial agents
aragora gauntlet spec.md --profile thorough --output receipt.html

# Run a debate from the command line
aragora ask "Should we adopt microservices?" --agents anthropic-api,openai-api --rounds 3

# Start the API server (2,000+ operations)
aragora serve --api-port 8080 --ws-port 8765
```

### Connect with the SDK

```python
import asyncio
from aragora_sdk import AragoraClient

async def main():
    client = AragoraClient("http://localhost:8080")
    debate = await client.debates.run(
        task="Should we use microservices?",
        agents=["anthropic-api", "openai-api"],
    )
    print(f"Consensus: {debate.consensus.conclusion}")

asyncio.run(main())
```

---

## Next steps

| What you want to do | Where to go |
|----------------------|-------------|
| Install-to-receipt quickstart (2 min) | [SDK_QUICKSTART.md](SDK_QUICKSTART.md) |
| Learn the full Python and TypeScript SDKs | [SDK_GUIDE.md](SDK_GUIDE.md) |
| Explore the REST API | [api/API_REFERENCE.md](api/API_REFERENCE.md) |
| Stress-test specs with the Gauntlet | [GAUNTLET.md](GAUNTLET.md) |
| Set up Slack, Teams, or Discord connectors | [INTEGRATIONS.md](INTEGRATIONS.md) |
| Deploy to production | [DEPLOYMENT.md](DEPLOYMENT.md) |
| Add SSO, RBAC, and multi-tenancy | [enterprise/ENTERPRISE_FEATURES.md](enterprise/ENTERPRISE_FEATURES.md) |
| Generate EU AI Act compliance artifacts | [compliance/EU_AI_ACT_GUIDE.md](compliance/EU_AI_ACT_GUIDE.md) |
| Understand pricing and plans | [PRICING.md](PRICING.md) |
| Learn why adversarial debate works | [WHY_ARAGORA.md](WHY_ARAGORA.md) |

---

## Troubleshooting

### "No ANTHROPIC_API_KEY configured"

Set at least one LLM provider API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or use mock agents (`create_agent("mock", ...)`) to run without API keys.

### "Connection refused" on localhost:8080

The Aragora server is not running. Start it with:

```bash
aragora serve --api-port 8080 --ws-port 8765
```

Or use `--offline` for a zero-dependency start with SQLite:

```bash
python -m aragora.server --http-port 8080 --ws-port 8765 --offline
```

### Import errors

Make sure you are using the correct package:

```python
# For standalone debates (aragora-debate package):
from aragora_debate import Debate, create_agent

# For the full platform (aragora package):
from aragora import Arena, Environment, DebateProtocol

# For the API client (aragora-sdk package):
from aragora_sdk import AragoraClient
```

---

## Getting help

- **GitHub Issues:** [github.com/an0mium/aragora/issues](https://github.com/an0mium/aragora/issues)
- **Documentation:** [github.com/an0mium/aragora/tree/main/docs](https://github.com/an0mium/aragora/tree/main/docs)
- **Sales:** sales@aragora.ai
- **Support:** support@aragora.ai
