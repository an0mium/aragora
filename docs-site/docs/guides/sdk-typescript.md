---
title: Aragora TypeScript SDK
description: Aragora TypeScript SDK
---

# Aragora TypeScript SDK

The Aragora TypeScript SDK (`@aragora/sdk`) provides a type-safe client for the Aragora API with built-in retry logic, WebSocket streaming, and comprehensive type definitions.

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
  apiKey: 'your-api-key',
});

// Create and run a debate
const debate = await client.debates.run({
  task: 'Should we adopt microservices?',
  agents: ['anthropic-api', 'openai-api'],
  rounds: 3,
});

console.log('Consensus:', debate.consensus);
```

## Configuration

```typescript
import { AragoraClient } from '@aragora/sdk';

const client = new AragoraClient({
  baseUrl: 'http://localhost:8080',  // Required: API server URL
  apiKey: 'your-api-key',            // Optional: Bearer token auth
  timeout: 30000,                     // Optional: Request timeout (ms)
  headers: {                          // Optional: Custom headers
    'X-Custom-Header': 'value',
  },
  retry: {                            // Optional: Retry configuration
    maxRetries: 3,                    // Max retry attempts
    initialDelay: 1000,               // Initial delay (ms)
    maxDelay: 30000,                  // Max delay (ms)
    backoffMultiplier: 2,             // Exponential backoff multiplier
    jitter: true,                     // Add random jitter to delays
  },
});
```

## API Reference

### Debates

Standard debates with propose-critique-revise workflow.

```typescript
// Create a debate (returns immediately)
const response = await client.debates.create({
  task: 'Design a rate limiter',
  agents: ['anthropic-api', 'openai-api', 'gemini'],
  rounds: 3,
  consensus: 'majority',
});
console.log('Debate ID:', response.debate_id);

// Get debate status
const debate = await client.debates.get(response.debate_id);
console.log('Status:', debate.status);

// Wait for completion with custom timeout
const completed = await client.debates.waitForCompletion(response.debate_id, {
  timeout: 300000,    // 5 minutes
  pollInterval: 2000, // Poll every 2 seconds
});

// Create and wait for completion in one call
const result = await client.debates.run({
  task: 'Should we use GraphQL or REST?',
  agents: ['anthropic-api', 'openai-api'],
});

// List debates
const debates = await client.debates.list({
  limit: 20,
  offset: 0,
  status: 'completed',
});
```

#### Additional Debate Methods

```typescript
// Get impasse information
const impasse = await client.debates.impasse(debateId);

// Get convergence metrics
const convergence = await client.debates.convergence(debateId);

// Get citations used
const citations = await client.debates.citations(debateId);

// Get debate messages (paginated)
const messages = await client.debates.messages(debateId, {
  limit: 50,
  offset: 0,
});

// Get evidence collected
const evidence = await client.debates.evidence(debateId);

// Get AI-generated summary
const summary = await client.debates.summary(debateId);

// Get followup suggestions
const followups = await client.debates.followupSuggestions(debateId);

// Fork a debate
const forked = await client.debates.fork(debateId, {
  task: 'Modified question...',
});

// Export debate
const exported = await client.debates.export(debateId, {
  format: 'markdown',
});
```

### Graph Debates

Branching debate trees for exploring multiple paths.

```typescript
// Create a graph debate
const response = await client.graphDebates.create({
  task: 'Explore database options',
  agents: ['anthropic-api', 'openai-api'],
});

// Get graph debate
const graph = await client.graphDebates.get(response.debate_id);

// Create a branch
const branch = await client.graphDebates.branch(response.debate_id, {
  parent_node: 'node-123',
  task: 'What about NoSQL?',
});

// Wait for completion
const completed = await client.graphDebates.waitForCompletion(response.debate_id);

// Or use run() for create + wait
const result = await client.graphDebates.run({
  task: 'Explore caching strategies',
  agents: ['anthropic-api', 'openai-api'],
});
```

### Matrix Debates

Multi-scenario analysis with parameter variations.

```typescript
// Create a matrix debate
const response = await client.matrixDebates.create({
  task: 'Evaluate deployment strategies',
  scenarios: [
    { name: 'Low traffic', params: { users: 1000 } },
    { name: 'High traffic', params: { users: 1000000 } },
  ],
  agents: ['anthropic-api', 'openai-api'],
});

// Get matrix results
const matrix = await client.matrixDebates.get(response.debate_id);

// Get conclusions across scenarios
const conclusions = await client.matrixDebates.conclusions(response.debate_id);

// Wait for completion
const completed = await client.matrixDebates.waitForCompletion(response.debate_id);

// Or use run()
const result = await client.matrixDebates.run({
  task: 'Compare cloud providers',
  scenarios: [...],
});
```

### Batch Debates

Run multiple debates concurrently.

```typescript
// Create batch
const batch = await client.batchDebates.create({
  debates: [
    { task: 'Question 1', agents: ['anthropic-api'] },
    { task: 'Question 2', agents: ['openai-api'] },
  ],
});

// Check batch status
const status = await client.batchDebates.status(batch.batch_id);

// Get queue status
const queue = await client.batchDebates.queueStatus();
```

### Gauntlet (Adversarial Validation)

Stress-test specifications and architectures.

```typescript
// Start gauntlet analysis
const receipt = await client.gauntlet.run({
  input_type: 'specification',
  input_text: 'The system shall handle 10k requests/second...',
});

// Get gauntlet result
const result = await client.gauntlet.get(receipt.receipt_id);

// List previous runs
const runs = await client.gauntlet.list({ limit: 10 });
```

### Agents

Agent profiles and rankings.

```typescript
// Get agent profile
const profile = await client.agents.profile('anthropic-api');

// Get leaderboard
const leaderboard = await client.agents.leaderboard({
  limit: 10,
  sort: 'elo',
});

// Compare agents
const comparison = await client.agents.compare(['anthropic-api', 'openai-api']);

// Get agent network relationships
const network = await client.agents.network('anthropic-api');

// Get agent consistency metrics
const consistency = await client.agents.consistency('anthropic-api');

// Get agent history
const history = await client.agents.history('anthropic-api', {
  limit: 20,
});
```

### Memory

Continuum memory management.

```typescript
// Get memory analytics
const analytics = await client.memory.analytics();

// Get memory snapshot
const snapshot = await client.memory.snapshot();

// Retrieve memories
const memories = await client.memory.retrieve({
  query: 'rate limiting',
  limit: 10,
});

// Consolidate memories
await client.memory.consolidate();

// Cleanup old memories
await client.memory.cleanup({ older_than_days: 30 });

// Get tier statistics
const tiers = await client.memory.tierStats();

// Get archive statistics
const archive = await client.memory.archiveStats();

// Check memory pressure
const pressure = await client.memory.pressure();
```

### Documents

Document upload and management.

```typescript
// Upload document
const doc = await client.documents.upload({
  file: fileBlob,
  name: 'architecture.pdf',
});

// Get document
const document = await client.documents.get(doc.document_id);

// List documents
const documents = await client.documents.list({ limit: 20 });

// Delete document
await client.documents.delete(doc.document_id);

// Get supported formats
const formats = await client.documents.formats();
```

### Verification

Formal verification of claims.

```typescript
// Verify a claim
const job = await client.verification.verify({
  claim: 'The algorithm terminates in O(n log n)',
  context: 'Merge sort implementation',
});

// Check verification status
const status = await client.verification.status(job.job_id);
```

### Authentication

User authentication and management.

```typescript
// Register new user
const user = await client.auth.register({
  email: 'user@example.com',
  password: 'secure-password',
  name: 'John Doe',
});

// Login
const session = await client.auth.login({
  email: 'user@example.com',
  password: 'secure-password',
});

// Get current user
const me = await client.auth.me();

// Update profile
await client.auth.updateMe({ name: 'Jane Doe' });

// Refresh token
const newTokens = await client.auth.refresh({
  refresh_token: session.refresh_token,
});

// Logout
await client.auth.logout();

// MFA Setup
const mfaSetup = await client.auth.mfaSetup();
await client.auth.mfaEnable({ code: '123456' });
await client.auth.mfaVerify({ code: '123456' });
```

### Billing

Subscription and usage management.

```typescript
// Get available plans
const plans = await client.billing.plans();

// Get current usage
const usage = await client.billing.usage();

// Get subscription
const subscription = await client.billing.subscription();

// Create checkout session
const checkout = await client.billing.checkout({
  plan_id: 'pro',
  success_url: 'https://app.example.com/success',
  cancel_url: 'https://app.example.com/cancel',
});

// Get billing portal
const portal = await client.billing.portal();

// Get invoices
const invoices = await client.billing.invoices();

// Get usage forecast
const forecast = await client.billing.forecast();
```

### Health Checks

```typescript
// Basic health check
const health = await client.health();
console.log('Status:', health.status);

// Deep health check with dependency status
const deep = await client.healthDeep();
console.log('Healthy:', deep.healthy);
console.log('Checks:', deep.checks);
```

## Error Handling

The SDK throws `AragoraError` for all API errors:

```typescript
import { AragoraError } from '@aragora/sdk';

try {
  await client.debates.get('nonexistent');
} catch (error) {
  if (error instanceof AragoraError) {
    console.log('Code:', error.code);        // e.g., 'NOT_FOUND'
    console.log('Status:', error.status);    // e.g., 404
    console.log('Message:', error.message);  // Human-readable message
    console.log('Retryable:', error.retryable); // Whether retry might help

    // Get user-friendly message
    console.log(error.toUserMessage());
  }
}
```

### Common Error Codes

| Code | Status | Description |
|------|--------|-------------|
| `TIMEOUT` | 408 | Request timed out |
| `NETWORK_ERROR` | 0 | Network connectivity issue |
| `RATE_LIMITED` | 429 | Too many requests |
| `UNAUTHORIZED` | 401 | Invalid or missing auth |
| `FORBIDDEN` | 403 | Permission denied |
| `NOT_FOUND` | 404 | Resource not found |
| `VALIDATION_ERROR` | 400 | Invalid request data |

## Retry Behavior

The SDK automatically retries failed requests for transient errors:

- **Retryable status codes**: 408, 429, 500, 502, 503, 504
- **Retryable error codes**: TIMEOUT, NETWORK_ERROR, RATE_LIMITED, SERVICE_UNAVAILABLE
- **Exponential backoff** with configurable jitter

Disable retries for specific requests:

```typescript
await client.debates.get(debateId, { retry: false });

// Or with custom retry settings
await client.debates.create(request, {
  retry: {
    maxRetries: 5,
    initialDelay: 500,
  },
});
```

## WebSocket Streaming

For real-time debate updates, use the WebSocket connection:

```typescript
// Connect to WebSocket (default `aragora serve`: 8765; single-port server: 8080)
const ws = new WebSocket('ws://localhost:8765/ws');
const loopId = 'debate-123';

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  if (['connection_info', 'loop_list', 'sync'].includes(message.type)) return;

  const eventLoopId = message.loop_id || message.data?.debate_id || message.data?.loop_id;
  if (eventLoopId && eventLoopId !== loopId) return;

  switch (message.type) {
    case 'debate_start':
      console.log('Debate started:', message.data?.task);
      break;
    case 'agent_message':
      console.log(`${message.agent}: ${message.data?.content}`);
      break;
    case 'consensus':
      console.log('Consensus reached:', message.data?.answer);
      break;
    case 'debate_end':
      console.log('Debate ended');
      break;
  }
};
```

For the full event list and payload envelope, see [WEBSOCKET_EVENTS.md](./websocket-events).

## React Integration

```tsx
import { useState, useEffect } from 'react';
import { AragoraClient, Debate } from '@aragora/sdk';

const client = new AragoraClient({
  baseUrl: process.env.NEXT_PUBLIC_API_URL!,
});

function useDebate(debateId: string) {
  const [debate, setDebate] = useState<Debate | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    client.debates.get(debateId)
      .then(setDebate)
      .catch(setError)
      .finally(() => setLoading(false));
  }, [debateId]);

  return { debate, loading, error };
}

function DebateView({ debateId }: { debateId: string }) {
  const { debate, loading, error } = useDebate(debateId);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  if (!debate) return <div>Not found</div>;

  return (
    <div>
      <h1>{debate.task}</h1>
      <p>Status: {debate.status}</p>
      <p>Consensus: {debate.consensus}</p>
    </div>
  );
}
```

## TypeScript Types

All types are exported from the package:

```typescript
import {
  // Client types
  AragoraClientOptions,
  RequestOptions,
  RetryOptions,

  // Debate types
  Debate,
  DebateCreateRequest,
  DebateCreateResponse,

  // Graph debate types
  GraphDebate,
  GraphDebateCreateRequest,

  // Matrix debate types
  MatrixDebate,
  MatrixDebateCreateRequest,

  // Agent types
  AgentProfile,
  LeaderboardEntry,

  // Error type
  AragoraError,
} from '@aragora/sdk';
```

## Migration from v0.x

If upgrading from an earlier version:

1. **Retry is now enabled by default** - disable with `retry: { maxRetries: 0 }`
2. **AragoraError now includes `retryable`** - check before manual retries
3. **New `run()` methods** - combine create + waitForCompletion

## Support

- [GitHub Issues](https://github.com/an0mium/aragora/issues)
- [API Reference](../api/reference)
- [Python SDK Guide](./sdk)
