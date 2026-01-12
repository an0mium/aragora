# @aragora/sdk

TypeScript/JavaScript SDK for the Aragora multi-agent debate framework.

## Installation

```bash
npm install @aragora/sdk
# or
yarn add @aragora/sdk
# or
pnpm add @aragora/sdk
```

## Quick Start

```typescript
import { AragoraClient } from '@aragora/sdk';

const client = new AragoraClient({
  baseUrl: 'http://localhost:8080',
  apiKey: 'your-api-key', // optional
});

// Run a debate
const debate = await client.debates.run({
  task: 'Should we use microservices?',
  agents: ['anthropic-api', 'openai-api'],
});

console.log('Consensus:', debate.consensus?.conclusion);
```

## API Reference

### Client Initialization

```typescript
const client = new AragoraClient({
  baseUrl: 'http://localhost:8080',
  apiKey: 'your-api-key',      // optional
  timeout: 30000,              // optional, default 30s
  headers: {                   // optional
    'X-Custom-Header': 'value',
  },
});
```

### Debates API

```typescript
// Create a debate
const response = await client.debates.create({
  task: 'Design a rate limiter',
  agents: ['anthropic-api', 'openai-api'],
  max_rounds: 5,
  consensus_threshold: 0.8,
});

// Get debate details
const debate = await client.debates.get('debate-123');

// List debates
const debates = await client.debates.list({ limit: 10 });

// Run debate and wait for completion
const result = await client.debates.run({
  task: 'Should we use TypeScript?',
});
```

### Graph Debates API

Graph debates support automatic branching when agents identify fundamentally different approaches.

```typescript
// Create graph debate
const response = await client.graphDebates.create({
  task: 'Design a distributed system',
  agents: ['anthropic-api', 'openai-api'],
  max_rounds: 5,
  branch_threshold: 0.5,  // Divergence threshold for branching
  max_branches: 10,
});

// Get debate with branches
const debate = await client.graphDebates.get('debate-123');

// Get branches
const branches = await client.graphDebates.getBranches('debate-123');
```

### Matrix Debates API

Matrix debates run the same question across different scenarios to identify universal vs conditional conclusions.

```typescript
// Create matrix debate
const response = await client.matrixDebates.create({
  task: 'Should we adopt microservices?',
  scenarios: [
    { name: 'small_team', parameters: { team_size: 5 } },
    { name: 'large_team', parameters: { team_size: 50 } },
    { name: 'high_traffic', parameters: { rps: 100000 }, is_baseline: true },
  ],
  max_rounds: 3,
});

// Get conclusions
const conclusions = await client.matrixDebates.getConclusions('matrix-123');
console.log('Universal:', conclusions.universal);
console.log('Conditional:', conclusions.conditional);
```

### Verification API

Formal verification of claims using Z3 or Lean 4.

```typescript
// Verify a claim
const result = await client.verification.verify({
  claim: 'All primes > 2 are odd',
  backend: 'z3',  // 'z3' | 'lean'
  timeout: 30,
});

if (result.status === 'valid') {
  console.log('Claim is valid!');
  console.log('Formal translation:', result.formal_translation);
}

// Check backend status
const status = await client.verification.status();
console.log('Z3 available:', status.backends.find(b => b.name === 'z3')?.available);
```

### Memory API

Analytics for the multi-tier memory system.

```typescript
// Get analytics
const analytics = await client.memory.analytics(30); // last 30 days
console.log('Total entries:', analytics.total_entries);
console.log('Learning velocity:', analytics.learning_velocity);

// Get tier-specific stats
const fastTier = await client.memory.tierStats('fast');

// Take manual snapshot
const snapshot = await client.memory.snapshot();
```

### Agents API

```typescript
// List available agents
const agents = await client.agents.list();

// Get agent profile
const agent = await client.agents.get('anthropic-api');
console.log('ELO rating:', agent.elo_rating);

// Get match history
const history = await client.agents.history('anthropic-api', 20);

// Get rivals and allies
const rivals = await client.agents.rivals('anthropic-api');
const allies = await client.agents.allies('anthropic-api');
```

### Gauntlet API

Adversarial validation of specifications.

```typescript
// Run gauntlet
const response = await client.gauntlet.run({
  input_content: 'Your spec content here...',
  input_type: 'spec',
  persona: 'security',  // security, performance, usability, etc.
});

// Get receipt
const receipt = await client.gauntlet.getReceipt(response.gauntlet_id);
console.log('Score:', receipt.score);
console.log('Findings:', receipt.findings);

// Run and wait for completion
const result = await client.gauntlet.runAndWait({
  input_content: specContent,
  persona: 'devil_advocate',
});
```

### Replay API

```typescript
// List replays
const replays = await client.replays.list({ limit: 10 });

// Get replay
const replay = await client.replays.get('replay-123');

// Export replay
const data = await client.replays.export('replay-123', 'json');

// Delete replay
await client.replays.delete('replay-123');
```

### Health Check

```typescript
const health = await client.health();
console.log('Status:', health.status);
console.log('Version:', health.version);
```

## WebSocket Streaming

Stream debate events in real-time.

### Class-based API

```typescript
import { DebateStream } from '@aragora/sdk';

const stream = new DebateStream('http://localhost:8080', 'debate-123');

stream
  .on('agent_message', (event) => {
    console.log('Agent message:', event.data);
  })
  .on('consensus', (event) => {
    console.log('Consensus reached!', event.data);
  })
  .on('debate_end', (event) => {
    console.log('Debate ended');
    stream.disconnect();
  })
  .onError((error) => {
    console.error('Error:', error);
  });

await stream.connect();
```

### Async Iterator API

```typescript
import { streamDebate } from '@aragora/sdk';

const stream = streamDebate('http://localhost:8080', 'debate-123');

for await (const event of stream) {
  console.log(event.type, event.data);

  if (event.type === 'debate_end') {
    break;
  }
}
```

### WebSocket Options

```typescript
const stream = new DebateStream('http://localhost:8080', 'debate-123', {
  reconnect: true,           // Auto-reconnect on disconnect
  reconnectInterval: 1000,   // Base reconnect delay (ms)
  maxReconnectAttempts: 5,   // Max reconnect attempts
  heartbeatInterval: 30000,  // Heartbeat ping interval (ms)
});
```

## Error Handling

```typescript
import { AragoraError } from '@aragora/sdk';

try {
  await client.debates.get('nonexistent-123');
} catch (error) {
  if (error instanceof AragoraError) {
    console.log('Code:', error.code);      // 'NOT_FOUND'
    console.log('Status:', error.status);  // 404
    console.log('Message:', error.message);
    console.log('Details:', error.details);
  }
}
```

## TypeScript Types

All types are exported for use in your application:

```typescript
import type {
  Debate,
  DebateStatus,
  ConsensusResult,
  GraphDebate,
  GraphDebateBranch,
  MatrixDebate,
  MatrixConclusion,
  VerifyClaimResponse,
  VerificationStatus,
  AgentProfile,
  GauntletReceipt,
  DebateEvent,
} from '@aragora/sdk';
```

## Browser Support

The SDK uses the standard `fetch` API and works in:
- Modern browsers (Chrome, Firefox, Safari, Edge)
- Node.js 18+
- Deno
- Bun

For WebSocket streaming in Node.js, install the `ws` package:

```bash
npm install ws
```

## License

MIT
