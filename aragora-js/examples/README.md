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

### Matrix Debate

Run parallel debates across multiple scenarios:

```bash
npx ts-node examples/matrix-debate.ts
```

### Consensus Voting

Use debates for group decision-making:

```bash
npx ts-node examples/consensus-voting.ts
```

### Evidence Research

Collect and analyze evidence for research tasks:

```bash
npx ts-node examples/evidence-research.ts
```

### Batch Analysis

Process multiple topics in parallel:

```bash
npx ts-node examples/batch-analysis.ts
```

### Real-Time Streaming

Stream debate events via WebSocket:

```bash
npx ts-node examples/streaming-example.ts
```

### Error Handling

Comprehensive error handling patterns:

```bash
npx ts-node examples/error-handling.ts
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

### Evidence Collection & Research

```typescript
// Collect evidence on a topic
const evidence = await client.evidence.collect({
  task: 'Climate change impacts on agriculture',
  connectors: ['duckduckgo', 'wikipedia'],
});

// Search existing evidence
const results = await client.evidence.search({
  query: 'renewable energy',
  limit: 20,
  min_reliability: 0.5,
});

// Get evidence for a debate
const debateEvidence = await client.evidence.forDebate(debateId);

// Associate evidence with a debate
await client.evidence.associateWithDebate(debateId, evidenceIds, {
  relevance_score: 0.8,
});

// Get evidence statistics
const stats = await client.evidence.statistics();
```

### Batch Debates

```typescript
// Create batch of debates
const batch = await client.batchDebates.create({
  debates: [
    { task: 'Topic 1', agents: ['claude', 'gpt-4'] },
    { task: 'Topic 2', agents: ['claude', 'gpt-4'] },
  ],
});

// Check batch status
const status = await client.batchDebates.getStatus(batch.batch_id);

// Get all results
const results = await client.batchDebates.get(batch.batch_id);
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
import { DebateStream } from '@aragora/sdk';

const stream = new DebateStream('ws://localhost:8765/ws', debateId);

stream.on('debate_start', (event) => console.log('Debate started:', event.data));
stream.on('agent_message', (event) => console.log('Message:', event.data));
stream.on('consensus', (event) => console.log('Consensus:', event.data));
stream.on('debate_end', (event) => console.log('Debate ended:', event.data));

await stream.connect();
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
