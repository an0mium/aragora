# TypeScript SDK Quickstart

Get started with the Aragora TypeScript/JavaScript SDK in 5 minutes.

## Installation

```bash
npm install aragora-js
# or
yarn add aragora-js
# or
pnpm add aragora-js
```

## Prerequisites

Start the Aragora server:

```bash
# Terminal 1: Start the server
python -m aragora.server.unified_server --port 8080
```

Set API keys for at least one provider:

```bash
export ANTHROPIC_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."
```

## Basic Usage

### 1. Create a Client

```typescript
import { AragoraClient } from 'aragora-js';

// Connect to local server
const client = new AragoraClient({
  baseUrl: 'http://localhost:8080',
  // token: process.env.ARAGORA_API_TOKEN, // Optional auth
});

// Check server health
const health = await client.health.check();
console.log(`Server status: ${health.status}`);
```

### 2. Run a Debate

```typescript
// Run a debate and wait for completion
const result = await client.debates.run({
  task: 'Should we use microservices or a monolith for our new project?',
  agents: ['anthropic-api', 'openai-api'],
  rounds: 3,
});

console.log(`Consensus reached: ${result.consensus?.reached}`);
console.log(`Confidence: ${(result.consensus?.confidence ?? 0) * 100}%`);
console.log(`Final answer: ${result.consensus?.conclusion?.slice(0, 500)}...`);
```

### 3. Create and Poll a Debate

For more control, create a debate and poll for status:

```typescript
// Create debate (returns immediately)
const debate = await client.debates.create({
  topic: "What's the best database for a real-time analytics platform?",
  agents: ['anthropic-api', 'openai-api', 'gemini'],
  rounds: 2,
  consensus: 'majority',
});

console.log(`Debate ID: ${debate.id}`);
console.log(`Status: ${debate.status}`);

// Poll for completion
const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));

while (true) {
  const status = await client.debates.get(debate.id);
  if (status.status === 'completed') {
    console.log(`Completed! Consensus: ${status.consensus?.reached}`);
    break;
  }
  await sleep(2000);
}
```

## Real-time Streaming

Stream debate events in real-time using WebSockets:

```typescript
import { streamDebate } from 'aragora-js';

// Stream events as they happen
const stream = await streamDebate({
  baseUrl: 'http://localhost:8080',
  task: 'Design a caching strategy',
  agents: ['anthropic-api', 'openai-api'],
});

stream.on('agent_message', (event) => {
  console.log(`[${event.agent}]: ${event.content.slice(0, 100)}...`);
});

stream.on('consensus', (event) => {
  console.log(`Consensus reached: ${JSON.stringify(event.data)}`);
});

stream.on('error', (error) => {
  console.error('Stream error:', error);
});

// Wait for completion
await stream.wait();
```

## React Integration

Use with React hooks:

```tsx
import { useState, useEffect } from 'react';
import { AragoraClient, Debate } from 'aragora-js';

const client = new AragoraClient({ baseUrl: 'http://localhost:8080' });

function useDebate(debateId: string) {
  const [debate, setDebate] = useState<Debate | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const fetchDebate = async () => {
      try {
        const result = await client.debates.get(debateId);
        setDebate(result);
      } catch (e) {
        setError(e as Error);
      } finally {
        setLoading(false);
      }
    };

    fetchDebate();
    const interval = setInterval(fetchDebate, 2000);
    return () => clearInterval(interval);
  }, [debateId]);

  return { debate, loading, error };
}

// Usage in component
function DebateViewer({ id }: { id: string }) {
  const { debate, loading, error } = useDebate(id);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  if (!debate) return <div>Not found</div>;

  return (
    <div>
      <h2>{debate.topic}</h2>
      <p>Status: {debate.status}</p>
      {debate.consensus && (
        <p>Consensus: {debate.consensus.conclusion}</p>
      )}
    </div>
  );
}
```

## Gauntlet: Adversarial Validation

Stress-test decisions with adversarial AI personas:

```typescript
import { readFileSync } from 'fs';

// Validate a policy document
const content = readFileSync('policy.md', 'utf-8');

const receipt = await client.gauntlet.runAndWait({
  inputContent: content,
  inputType: 'policy',
  persona: 'gdpr',
  profile: 'thorough',
});

console.log(`Verdict: ${receipt.verdict}`);
console.log(`Risk Score: ${receipt.riskScore}`);

for (const finding of receipt.findings) {
  console.log(`  [${finding.severity}] ${finding.title}`);
}
```

## Agent Rankings

Query agent performance:

```typescript
// Get agent leaderboard
const rankings = await client.agents.rankings();

rankings.slice(0, 10).forEach((agent, i) => {
  console.log(`${i + 1}. ${agent.name}: ${agent.elo?.toFixed(0)} ELO`);
});

// Get specific agent profile
const agent = await client.agents.get('anthropic-api');
console.log(`Agent: ${agent.name}`);
console.log(`Rating: ${agent.elo}`);
console.log(`Win rate: ${(agent.winRate ?? 0) * 100}%`);
```

## Advanced Features

### Graph Debates (Branching)

Explore multiple solution paths:

```typescript
// Create a graph debate
const graph = await client.graphDebates.create({
  rootTopic: 'Design a notification system',
  branchDepth: 3,
  agents: ['anthropic-api', 'openai-api'],
});

// Explore branches
for (const branch of graph.branches) {
  console.log(`Branch: ${branch.topic}`);
  console.log(`  Path: ${branch.path.join(' -> ')}`);
}
```

### Matrix Debates (Parallel Scenarios)

Test multiple scenarios in parallel:

```typescript
// Create a matrix debate
const matrix = await client.matrixDebates.create({
  baseTopic: 'Evaluate authentication approaches',
  scenarios: [
    { name: 'high_traffic', context: '10M daily users' },
    { name: 'regulated', context: 'HIPAA compliance required' },
    { name: 'startup', context: 'Minimum viable product' },
  ],
  agents: ['anthropic-api', 'openai-api'],
});

// Get results for each scenario
for (const scenario of matrix.scenarios) {
  console.log(`${scenario.name}: ${scenario.recommendation}`);
}
```

### Control Plane

For enterprise orchestration:

```typescript
// List registered agents
const agents = await client.controlPlane.listAgents();
console.log(`Registered agents: ${agents.length}`);

// Submit a task
const taskId = await client.controlPlane.submitTask({
  type: 'debate',
  payload: { topic: 'System design review' },
  priority: 'high',
});

// Wait for task completion
const task = await client.controlPlane.waitForTask(taskId, {
  timeout: 60000,
  pollInterval: 2000,
});

console.log(`Task result: ${JSON.stringify(task.result)}`);
```

## Error Handling

```typescript
import { AragoraError } from 'aragora-js';

try {
  const result = await client.debates.get('invalid-id');
} catch (error) {
  if (error instanceof AragoraError) {
    switch (error.statusCode) {
      case 404:
        console.log('Debate not found');
        break;
      case 429:
        console.log(`Rate limited. Retry after ${error.retryAfter}s`);
        break;
      case 401:
        console.log('Invalid API token');
        break;
      default:
        console.log(`API error: ${error.message}`);
    }
  } else {
    throw error;
  }
}
```

## Configuration

```typescript
const client = new AragoraClient({
  baseUrl: 'http://localhost:8080',
  token: 'your-api-token', // Optional auth
  timeout: 60000, // 60 second timeout
  retries: 3, // Retry failed requests
  headers: {
    'X-Custom-Header': 'value',
  },
});
```

## Next.js API Route Example

```typescript
// pages/api/debate.ts
import type { NextApiRequest, NextApiResponse } from 'next';
import { AragoraClient } from 'aragora-js';

const client = new AragoraClient({
  baseUrl: process.env.ARAGORA_API_URL || 'http://localhost:8080',
});

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { topic, agents } = req.body;

    const debate = await client.debates.create({
      topic,
      agents: agents || ['anthropic-api', 'openai-api'],
      rounds: 2,
    });

    return res.status(200).json({ debateId: debate.id });
  } catch (error) {
    console.error('Debate creation failed:', error);
    return res.status(500).json({ error: 'Failed to create debate' });
  }
}
```

## Next Steps

- [Python SDK Quickstart](./python-quickstart.md)
- [API Reference](../API_REFERENCE.md)
- [Examples](../../examples/README.md)
- [WebSocket Events](../WEBSOCKET_EVENTS.md)
