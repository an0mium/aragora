# Quickstart: Run Your First Debate in 5 Minutes

This guide gets you from zero to a running multi-agent debate as fast as possible.

## Prerequisites

- Python 3.10+
- An API key from at least one AI provider (Anthropic recommended)

## 1. Install

```bash
pip install aragora
```

Or install from source:

```bash
git clone https://github.com/an0mium/aragora.git
cd aragora
pip install -e .
```

## 2. Set Your API Key

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Any one of these providers works: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`, `MISTRAL_API_KEY`.

## 3. Run a Debate (CLI)

The fastest way -- one command:

```bash
aragora ask "Should we use microservices or a monolith for our new SaaS product?"
```

You will see agents propose solutions, critique each other, and converge on a consensus.

## 4. Run a Debate (Python)

```python
import asyncio
from aragora import Arena, Environment, Agent
from aragora.debate.protocol import DebateProtocol

async def main():
    env = Environment(task="Should we use microservices or a monolith?")
    agents = [
        Agent(name="claude", model="anthropic"),
        Agent(name="gpt4", model="openai"),
    ]
    protocol = DebateProtocol(rounds=3, consensus="majority")
    arena = Arena(environment=env, agents=agents, protocol=protocol)
    result = await arena.run()

    print(f"Consensus: {result.consensus}")
    print(f"Final answer:\n{result.final_answer}")

asyncio.run(main())
```

**Expected output** (varies by run):

```
[Round 1] claude proposes: Microservices offer independent scaling...
[Round 1] gpt4 proposes: A monolith-first approach reduces complexity...
[Round 2] claude critiques gpt4: While monoliths are simpler initially...
[Round 2] gpt4 critiques claude: Microservices introduce network overhead...
[Round 3] Synthesis round...
Consensus: majority
Final answer:
Start with a modular monolith and extract services as scaling demands emerge.
Clear domain boundaries now make future extraction straightforward.
```

## 5. Run the Server (Optional)

For the REST API and live dashboard:

```bash
aragora serve --api-port 8080 --ws-port 8765
```

Then open `http://localhost:8080` for the web UI, or call the API:

```bash
curl -X POST http://localhost:8080/api/v1/debates \
  -H "Content-Type: application/json" \
  -d '{"task": "Should we adopt Kubernetes?", "rounds": 3}'
```

## Next Steps

| Goal | Guide |
|------|-------|
| Try different consensus modes | Change `consensus=` to `"judge"`, `"supermajority"`, `"weighted"`, or `"unanimous"` |
| Add more agents | Add Gemini, Grok, Mistral, DeepSeek -- see [Agents](../debate/AGENTS.md) |
| Stream events in real time | Connect via WebSocket -- see [WebSocket Events](../streaming/WEBSOCKET_EVENTS.md) |
| Stress-test a decision | Use `aragora gauntlet "your question"` -- see [Gauntlet](../debate/GAUNTLET.md) |
| Use the Python SDK | `pip install aragora-sdk` -- see [SDK Guide](../SDK_GUIDE.md) |
| Use the TypeScript SDK | `npm install @aragora/sdk` -- see [TypeScript SDK](SDK_TYPESCRIPT.md) |
| Full onboarding | [Getting Started](GETTING_STARTED.md) |
| API cookbook with 20 patterns | [API Cookbook](API_COOKBOOK.md) |

## Troubleshooting

**"No API keys configured"** -- Set at least one provider key. Run `aragora doctor` to verify.

**"Module not found"** -- Make sure you installed with `pip install -e .` from the repo root, or `pip install aragora` from PyPI.

**Agents timing out** -- Check your network connection and API key validity. Add `OPENROUTER_API_KEY` as a fallback provider.
