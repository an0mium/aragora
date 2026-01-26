# SDK Comparison: Python vs TypeScript

Feature comparison between the Aragora Python and TypeScript SDKs.

## Quick Comparison

| Feature | Python (`aragora-py`) | TypeScript (`@aragora/sdk`) |
|---------|----------------------|----------------------------|
| Package | `pip install aragora` | `npm install @aragora/sdk` |
| Async | Native async/await | Native async/await |
| Types | Type hints (mypy) | Full TypeScript types |
| Streaming | Yes | Yes (WebSocket) |
| Retry | Built-in | Built-in |
| Auto-complete | IDE support | Full IDE support |
| Bundle Size | ~50KB | ~30KB minified |

---

## Installation

### Python

```bash
pip install aragora
# or
pip install aragora[all]  # with all optional dependencies
```

### TypeScript

```bash
npm install @aragora/sdk
# or
yarn add @aragora/sdk
# or
pnpm add @aragora/sdk
```

---

## Client Initialization

### Python

```python
from aragora import AragoraClient

client = AragoraClient(
    base_url="http://localhost:8080",
    api_key="your-api-key",
    timeout=30,
    retry_config={
        "max_retries": 3,
        "initial_delay": 1.0,
        "max_delay": 30.0,
        "backoff_multiplier": 2.0,
    },
)
```

### TypeScript

```typescript
import { AragoraClient } from "@aragora/sdk";

const client = new AragoraClient({
  baseUrl: "http://localhost:8080",
  apiKey: "your-api-key",
  timeout: 30000,
  retry: {
    maxRetries: 3,
    initialDelay: 1000,
    maxDelay: 30000,
    backoffMultiplier: 2,
  },
});
```

---

## Basic Operations

### Create and Run Debate

**Python:**
```python
# Create and wait
result = await client.debates.run(
    task="Should we adopt microservices?",
    agents=["anthropic-api", "openai-api"],
    rounds=3,
)

# Or create separately
response = await client.debates.create(
    task="What's the best testing strategy?",
    agents=["anthropic-api", "openai-api"],
)
completed = await client.debates.wait_for_completion(response.debate_id)
```

**TypeScript:**
```typescript
// Create and wait
const result = await client.debates.run({
  task: "Should we adopt microservices?",
  agents: ["anthropic-api", "openai-api"],
  rounds: 3,
});

// Or create separately
const response = await client.debates.create({
  task: "What's the best testing strategy?",
  agents: ["anthropic-api", "openai-api"],
});
const completed = await client.debates.waitForCompletion(response.debate_id);
```

### Error Handling

**Python:**
```python
from aragora.exceptions import AragoraError, NotFoundError, RateLimitError

try:
    debate = await client.debates.get("nonexistent")
except NotFoundError:
    print("Debate not found")
except RateLimitError as e:
    print(f"Rate limited, retry after {e.retry_after}s")
except AragoraError as e:
    print(f"API error: {e.code} - {e.message}")
```

**TypeScript:**
```typescript
import { AragoraError } from "@aragora/sdk";

try {
  const debate = await client.debates.get("nonexistent");
} catch (error) {
  if (error instanceof AragoraError) {
    switch (error.code) {
      case "NOT_FOUND":
        console.log("Debate not found");
        break;
      case "RATE_LIMITED":
        console.log("Rate limited");
        break;
      default:
        console.log(`API error: ${error.code} - ${error.message}`);
    }
  }
}
```

---

## Feature Parity Matrix

| Endpoint | Python | TypeScript | Notes |
|----------|--------|------------|-------|
| **Debates** ||||
| create | ✓ | ✓ | |
| get | ✓ | ✓ | |
| list | ✓ | ✓ | |
| run | ✓ | ✓ | Create + wait |
| waitForCompletion | ✓ | ✓ | |
| messages | ✓ | ✓ | |
| summary | ✓ | ✓ | |
| citations | ✓ | ✓ | |
| evidence | ✓ | ✓ | |
| convergence | ✓ | ✓ | |
| impasse | ✓ | ✓ | |
| fork | ✓ | ✓ | |
| export | ✓ | ✓ | |
| followupSuggestions | ✓ | ✓ | |
| **Graph Debates** ||||
| create | ✓ | ✓ | |
| get | ✓ | ✓ | |
| branch | ✓ | ✓ | |
| run | ✓ | ✓ | |
| **Matrix Debates** ||||
| create | ✓ | ✓ | |
| get | ✓ | ✓ | |
| conclusions | ✓ | ✓ | |
| run | ✓ | ✓ | |
| **Batch Debates** ||||
| create | ✓ | ✓ | |
| status | ✓ | ✓ | |
| queueStatus | ✓ | ✓ | |
| **Agents** ||||
| profile | ✓ | ✓ | |
| leaderboard | ✓ | ✓ | |
| compare | ✓ | ✓ | |
| network | ✓ | ✓ | |
| consistency | ✓ | ✓ | |
| history | ✓ | ✓ | |
| **Memory** ||||
| analytics | ✓ | ✓ | |
| snapshot | ✓ | ✓ | |
| retrieve | ✓ | ✓ | |
| consolidate | ✓ | ✓ | |
| cleanup | ✓ | ✓ | |
| tierStats | ✓ | ✓ | |
| archiveStats | ✓ | ✓ | |
| pressure | ✓ | ✓ | |
| **Documents** ||||
| upload | ✓ | ✓ | |
| get | ✓ | ✓ | |
| list | ✓ | ✓ | |
| delete | ✓ | ✓ | |
| formats | ✓ | ✓ | |
| **Gauntlet** ||||
| run | ✓ | ✓ | |
| get | ✓ | ✓ | |
| list | ✓ | ✓ | |
| **Verification** ||||
| verify | ✓ | ✓ | |
| status | ✓ | ✓ | |
| **Auth** ||||
| register | ✓ | ✓ | |
| login | ✓ | ✓ | |
| logout | ✓ | ✓ | |
| me | ✓ | ✓ | |
| updateMe | ✓ | ✓ | |
| refresh | ✓ | ✓ | |
| mfaSetup | ✓ | ✓ | |
| mfaEnable | ✓ | ✓ | |
| mfaVerify | ✓ | ✓ | |
| **Billing** ||||
| plans | ✓ | ✓ | |
| usage | ✓ | ✓ | |
| subscription | ✓ | ✓ | |
| checkout | ✓ | ✓ | |
| portal | ✓ | ✓ | |
| invoices | ✓ | ✓ | |
| forecast | ✓ | ✓ | |
| **Health** ||||
| health | ✓ | ✓ | |
| healthDeep | ✓ | ✓ | |

---

## WebSocket Streaming

### Python

```python
import asyncio
import websockets
import json

async def stream_debate(debate_id: str):
    uri = "ws://localhost:8765/ws"
    async with websockets.connect(uri) as ws:
        async for message in ws:
            data = json.loads(message)

            if data.get("loop_id") != debate_id:
                continue

            if data["type"] == "agent_message":
                print(f"{data['agent']}: {data['data']['content']}")
            elif data["type"] == "consensus":
                print(f"Consensus: {data['data']['answer']}")
                break
```

### TypeScript

```typescript
const ws = new WebSocket("ws://localhost:8765/ws");

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.loop_id !== debateId) return;

  switch (data.type) {
    case "agent_message":
      console.log(`${data.agent}: ${data.data.content}`);
      break;
    case "consensus":
      console.log(`Consensus: ${data.data.answer}`);
      ws.close();
      break;
  }
};
```

---

## Async Patterns

### Python (asyncio)

```python
import asyncio

async def run_parallel_debates():
    tasks = [
        client.debates.run(task=f"Question {i}", agents=["anthropic-api"])
        for i in range(3)
    ]
    results = await asyncio.gather(*tasks)
    return results
```

### TypeScript (Promise.all)

```typescript
async function runParallelDebates() {
  const tasks = [1, 2, 3].map((i) =>
    client.debates.run({
      task: `Question ${i}`,
      agents: ["anthropic-api"],
    })
  );
  const results = await Promise.all(tasks);
  return results;
}
```

---

## Framework Integration

### Python Frameworks

```python
# FastAPI
from fastapi import FastAPI, Depends
from aragora import AragoraClient

app = FastAPI()

async def get_client():
    return AragoraClient(base_url=settings.aragora_url)

@app.post("/debates")
async def create_debate(task: str, client: AragoraClient = Depends(get_client)):
    return await client.debates.run(task=task, agents=["anthropic-api"])

# Django (async view)
from django.http import JsonResponse
from aragora import AragoraClient

async def debate_view(request):
    client = AragoraClient(base_url=settings.ARAGORA_URL)
    result = await client.debates.run(task=request.POST["task"], agents=["anthropic-api"])
    return JsonResponse(result.dict())
```

### TypeScript Frameworks

```typescript
// Next.js API Route
import { AragoraClient } from "@aragora/sdk";
import { NextRequest, NextResponse } from "next/server";

const client = new AragoraClient({ baseUrl: process.env.ARAGORA_URL! });

export async function POST(request: NextRequest) {
  const { task } = await request.json();
  const result = await client.debates.run({
    task,
    agents: ["anthropic-api"],
  });
  return NextResponse.json(result);
}

// Express.js
import express from "express";
import { AragoraClient } from "@aragora/sdk";

const app = express();
const client = new AragoraClient({ baseUrl: process.env.ARAGORA_URL! });

app.post("/debates", async (req, res) => {
  const result = await client.debates.run({
    task: req.body.task,
    agents: ["anthropic-api"],
  });
  res.json(result);
});
```

---

## When to Use Which

### Use Python SDK When:

- Building backend services with FastAPI/Django
- Data science and ML workflows
- Jupyter notebook analysis
- CLI tools and scripts
- Existing Python codebase

### Use TypeScript SDK When:

- Building React/Vue/Angular frontends
- Next.js or Nuxt.js applications
- Node.js backend services
- Full-stack TypeScript projects
- Browser-based applications

---

## See Also

- [Python SDK Guide](SDK_GUIDE.md)
- [TypeScript SDK Reference](SDK_TYPESCRIPT.md)
- [API Reference](API_REFERENCE.md)
- [TypeScript Examples](../examples/typescript/)
