# TypeScript SDK Quickstart

Get started with Aragora in under 5 minutes.

## Installation

```bash
npm install @aragora/sdk
# or
yarn add @aragora/sdk
# or
pnpm add @aragora/sdk
```

## Basic Usage

```typescript
import { createClient } from '@aragora/sdk';

const client = createClient({
  baseUrl: 'http://localhost:8080',
  apiKey: 'your-api-key'
});

// Run a debate
const result = await client.createDebate({
  task: 'Should we use TypeScript or JavaScript?',
  agents: ['claude', 'gpt-4'],
  rounds: 3
});

console.log('Conclusion:', result.final_answer);
```

## Full Example

```typescript
import { createClient, AragoraClientSync } from '@aragora/sdk';

async function main() {
  // Option 1: Low-level client
  const client = createClient({
    baseUrl: 'http://localhost:8080',
    apiKey: process.env.ARAGORA_API_KEY
  });

  // Create a debate
  const debate = await client.createDebate({
    task: 'Design a caching strategy for our API',
    agents: ['anthropic-api', 'openai-api', 'gemini-api'],
    rounds: 3,
    consensus: 'majority'
  });

  console.log('Debate ID:', debate.debate_id);

  // Get results
  const result = await client.getDebate(debate.debate_id);
  console.log('Status:', result.status);
  console.log('Final answer:', result.final_answer);

  // Option 2: Sync-style wrapper (simpler API)
  const syncClient = new AragoraClientSync({
    baseUrl: 'http://localhost:8080',
    apiKey: process.env.ARAGORA_API_KEY
  });

  const debates = await syncClient.listDebates();
  console.log('Total debates:', debates.length);
}

main();
```

## Real-time Streaming

```typescript
import { createClient } from '@aragora/sdk';

const client = createClient({ baseUrl: 'http://localhost:8080' });

// Stream debate events
const stream = client.streamDebate(debateId);

for await (const event of stream) {
  switch (event.type) {
    case 'agent_message':
      console.log(`${event.agent}: ${event.content}`);
      break;
    case 'consensus':
      console.log('Consensus reached:', event.conclusion);
      break;
  }
}
```

## Key APIs

| Method | Description |
|--------|-------------|
| `client.createDebate()` | Create and run a debate |
| `client.getDebate(id)` | Get debate details |
| `client.listDebates()` | List all debates |
| `client.streamDebate(id)` | Stream real-time events |
| `client.getExplanation(id)` | Get decision explanation |
| `client.health()` | Check API health |

## SME Workflows (Small Business)

```typescript
// Quick invoice generation
const invoice = await client.sme.quickInvoice({
  customer: 'Acme Corp',
  items: [{ description: 'Consulting', amount: 1500 }]
});

// Quick inventory check
const status = await client.sme.quickInventoryCheck('SKU-123');

// List available SME workflows
const workflows = await client.sme.listWorkflows();
```

## Environment Variables

```bash
export ARAGORA_API_URL="http://localhost:8080"
export ARAGORA_API_KEY="your-api-key"
```

```typescript
const client = createClient({
  baseUrl: process.env.ARAGORA_API_URL,
  apiKey: process.env.ARAGORA_API_KEY
});
```

## Next Steps

- [Full API Reference](https://docs.aragora.ai/sdk/typescript)
- [WebSocket Streaming Guide](https://docs.aragora.ai/features/streaming)
- [Graph Debates](https://docs.aragora.ai/features/graph-debates)
- [Workflows](https://docs.aragora.ai/features/workflows)
