# Quickstart

Get from zero to a working adversarial debate in under a minute.

---

## 1. Install

```bash
pip install aragora-debate
```

## 2. Zero-Key Demo

No API keys required. The offline demo runs a complete adversarial debate with
mock agents:

```bash
python3 -m aragora_debate
```

You'll see three agents propose, critique each other, vote, reach consensus, and
produce an audit-ready decision receipt with a SHA-256 verdict hash.

(If you installed the full platform instead — `pip install -U 'aragora>=2.9.0'`
— the equivalent zero-key receipt demo is
`aragora demo --offline --receipt aragora-demo-receipt.json`, followed by
`aragora receipt verify aragora-demo-receipt.json`.)

## 3. Three-Line Debate (Python)

```python
import asyncio

from aragora_debate.arena import Arena
from aragora_debate.styled_mock import StyledMockAgent

agents = [
    StyledMockAgent("analyst", style="supportive"),
    StyledMockAgent("critic", style="critical"),
    StyledMockAgent("pm", style="balanced"),
]
arena = Arena(question="Should we adopt GraphQL?", agents=agents)
result = asyncio.run(arena.run())
print(result.receipt.to_markdown())
```

## 4. Add Real AI Models

Set both provider keys for the two-model example below. If you only have one
provider key, remove the other `create_agent(...)` entry.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # Claude
# or
export OPENAI_API_KEY="sk-..."          # GPT
```

Install the full `aragora` package before using the platform API imports below:

```bash
pip install -U 'aragora>=2.9.0'
```

Then run a real debate:

```python
import asyncio
from aragora import Arena, Environment, DebateProtocol
from aragora.agents import create_agent

env = Environment(task="Design a rate limiter for our API")
protocol = DebateProtocol(rounds=3, consensus="majority")

agents = [
    create_agent("anthropic-api", name="claude", role="proposer"),
    create_agent("openai-api", name="openai", role="critic"),
]
arena = Arena(env, agents=agents, protocol=protocol)
result = asyncio.run(arena.run())
print(result.summary())
```

## 5. TypeScript SDK

```bash
npm install @aragora/sdk
```

```typescript
import { AragoraClient } from "@aragora/sdk";

const client = new AragoraClient({ baseUrl: "http://localhost:8080" });
const result = await client.debates.create({
  task: "Should we use microservices or a monolith?",
  agents: ["claude", "gpt-4"],
  rounds: 3,
});
console.log(result.debate_id, result.status);
```

## 6. Self-Host the Full Platform

```bash
docker compose -f deploy/demo/docker-compose.yml up
```

Then visit:
- **Landing page:** http://localhost:3000
- **API docs (Swagger):** http://localhost:8080/api/v2/docs
- **API docs (Redoc):** http://localhost:8080/api/v2/redoc
- **Interactive playground:** http://localhost:3000/playground

## 7. CLI

Current PyPI package:

```bash
pip install -U 'aragora>=2.9.0'
aragora demo --offline --receipt aragora-demo-receipt.json
aragora receipt verify aragora-demo-receipt.json
aragora ask "Should we build or buy our auth system?"   # real debate (needs an API key)
aragora serve --api-port 8080 --ws-port 8765
```

Current source checkout:

```bash
python3 -m pip install -e .
aragora demo --offline --receipt aragora-demo-receipt.json
aragora receipt verify aragora-demo-receipt.json
```

Use `aragora>=2.9.0` for the explicit offline demo receipt round trip shown
above. Earlier PyPI releases do not support the `--offline` receipt flags. Use
the source checkout path when you need to audit this exact branch or unreleased
local changes.

## Next Steps

| Guide | What you'll learn |
|-------|-------------------|
| [Receipt Lineage Reconciliation](specs/RECEIPT_LINEAGE_RECONCILIATION.md) | What a Decision Receipt is: the native record vs. the portable ODR |
| [Independent Verifier Guide](specs/INDEPENDENT_VERIFIER_GUIDE.md) | Verify a receipt offline with `aragora-verify`, no Aragora install required |
| [GitHub Action Setup](GITHUB_ACTION_SETUP.md) | Add multi-model CI review + receipts to your pull requests |
| [CLI Reference](reference/CLI_REFERENCE.md) | All CLI commands and flags |
| [SDK Guide](SDK_GUIDE.md) | Python & TypeScript SDK reference |
| [API Reference](api/API_REFERENCE.md) | REST API endpoints |
| [Self-Hosting](deployment/DEPLOYMENT.md) | Production deployment |
| [Documentation Landing](README.md) | Deeper architectural overview |
