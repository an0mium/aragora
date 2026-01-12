# Aragora SDK Examples

This directory contains example code demonstrating how to use the Aragora JavaScript/TypeScript SDK.

## Setup

```bash
# Install dependencies
npm install

# Set environment variables
export ARAGORA_API_URL=http://localhost:8080
export ARAGORA_API_KEY=your-api-key
```

## Examples

### Basic Debate

Run a simple adversarial debate between two agents:

```bash
npx ts-node examples/basic-debate.ts
```

### Graph Debate

Create a branching debate that explores multiple perspectives:

```bash
npx ts-node examples/graph-debate.ts
```

### Gauntlet Security Audit

Run a comprehensive security audit using the Gauntlet:

```bash
npx ts-node examples/gauntlet-audit.ts
```

## Quick Start

```typescript
import { AragoraClient } from 'aragora-js';

const client = new AragoraClient({
  baseUrl: 'http://localhost:8080',
  apiKey: 'your-api-key',
});

// Run a debate
const debate = await client.debates.run({
  task: 'Should we use microservices?',
  agents: ['claude-sonnet', 'gpt-4'],
  rounds: 3,
});

console.log(debate.consensus);
```

## API Reference

### Debates

```typescript
// Create a debate
const response = await client.debates.create({
  task: 'Your topic here',
  agents: ['agent1', 'agent2'],
});

// Get debate by ID
const debate = await client.debates.get(debateId);

// List debates
const debates = await client.debates.list({ limit: 10 });

// Run and wait for completion
const result = await client.debates.run({ ... });
```

### Graph Debates

```typescript
// Create branching debate
const response = await client.graphDebates.create({
  root_claim: 'Your claim',
  agents: ['agent1', 'agent2', 'agent3'],
});

// Get branches
const branches = await client.graphDebates.getBranches(debateId);
```

### Matrix Debates

```typescript
// Run parallel scenario debates
const response = await client.matrixDebates.create({
  question: 'How should we handle X?',
  scenarios: ['Scenario A', 'Scenario B', 'Scenario C'],
  agents: ['agent1', 'agent2'],
});

// Get conclusions across all scenarios
const conclusions = await client.matrixDebates.getConclusions(matrixId);
```

### Gauntlet (Red Team Testing)

```typescript
// Run security gauntlet
const receipt = await client.gauntlet.runAndWait({
  target: 'System description',
  playbook: 'security-red-team',
  config: {
    intensity: 'thorough',
    categories: ['prompt-injection', 'jailbreak'],
  },
});

console.log(receipt.findings);
console.log(receipt.score);
```

### Verification

```typescript
// Verify a claim
const result = await client.verification.verify({
  claim: 'All primes > 2 are odd',
});

// Batch verification
const results = await client.verification.verifyBatch([
  'Claim 1',
  'Claim 2',
]);
```

### Memory & Analytics

```typescript
// Get memory analytics
const analytics = await client.memory.analytics(30); // last 30 days

// Get tier statistics
const tierStats = await client.memory.tierStats('fast');
```

### Agents & Leaderboard

```typescript
// List agents
const agents = await client.agents.list();

// Get agent profile
const profile = await client.agents.get('claude-sonnet');

// Get leaderboard
const leaders = await client.leaderboard.get({ limit: 10 });
```

## WebSocket Streaming

For real-time debate updates, use the WebSocket client:

```typescript
import { AragoraWebSocket } from 'aragora-js';

const ws = new AragoraWebSocket({
  baseUrl: 'ws://localhost:8080',
  apiKey: 'your-api-key',
});

ws.on('debate_start', (data) => console.log('Debate started:', data));
ws.on('agent_message', (data) => console.log('Message:', data));
ws.on('consensus', (data) => console.log('Consensus:', data));
ws.on('debate_end', (data) => console.log('Debate ended:', data));

await ws.connect();
await ws.subscribe(debateId);
```

## Error Handling

```typescript
import { AragoraError } from 'aragora-js';

try {
  await client.debates.get('invalid-id');
} catch (error) {
  if (error instanceof AragoraError) {
    console.log('API Error:', error.message);
    console.log('Code:', error.code);
    console.log('Status:', error.status);
  }
}
```
