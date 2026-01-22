# @aragora/sdk

TypeScript/JavaScript SDK for the Aragora control plane for multi-agent deliberation across organizational knowledge and channels.

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

### Selection Plugins API

The selection API provides access to pluggable agent selection algorithms.

```typescript
// List available plugins (scorers, team selectors, role assigners)
const plugins = await client.selection.listPlugins();
console.log('Scorers:', plugins.scorers.map(s => s.name));
console.log('Team Selectors:', plugins.team_selectors.map(ts => ts.name));
console.log('Role Assigners:', plugins.role_assigners.map(ra => ra.name));

// Get default configuration
const defaults = await client.selection.getDefaults();
console.log('Default scorer:', defaults.scorer);

// Score agents for a task
const scores = await client.selection.scoreAgents({
  task_description: 'Design a distributed cache system',
  primary_domain: 'systems',
  scorer: 'elo_weighted',  // optional, uses default if not specified
});

for (const agent of scores.agents) {
  console.log(`${agent.name}: ${agent.score} (ELO: ${agent.elo_rating})`);
}

// Select an optimal team for a task
const team = await client.selection.selectTeam({
  task_description: 'Build a secure authentication system',
  min_agents: 3,
  max_agents: 5,
  diversity_preference: 0.7,  // Prefer diverse viewpoints
  quality_priority: 0.8,      // Prioritize quality over cost
  scorer: 'elo_weighted',
  team_selector: 'diverse',
  role_assigner: 'domain_based',
});

console.log('Team:', team.agents.map(a => `${a.name} (${a.role})`).join(', '));
console.log('Expected quality:', team.expected_quality);
console.log('Diversity score:', team.diversity_score);
console.log('Rationale:', team.rationale);
```

Available built-in plugins:
- **Scorers**: `elo_weighted` (default), `domain_specialist`, `balanced`
- **Team Selectors**: `diverse` (default), `greedy`, `random`
- **Role Assigners**: `domain_based` (default), `simple`

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

Use the WebSocket base URL (default `http://localhost:8765` for `aragora serve`).
If you run a single-port server, use that port instead.

### Class-based API

```typescript
import { DebateStream } from '@aragora/sdk';

const debateId = 'debate-123';
const stream = new DebateStream('http://localhost:8765', debateId);

const shouldHandle = (event: any) => {
  const eventLoopId = event.loop_id || event.data?.debate_id || event.data?.loop_id;
  return !eventLoopId || eventLoopId === debateId;
};

stream
  .on('agent_message', (event) => {
    if (!shouldHandle(event)) return;
    console.log('Agent message:', event.data);
  })
  .on('consensus', (event) => {
    if (!shouldHandle(event)) return;
    console.log('Consensus reached!', event.data);
  })
  .on('debate_end', (event) => {
    if (!shouldHandle(event)) return;
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

const debateId = 'debate-123';
const stream = streamDebate('http://localhost:8765', debateId);

for await (const event of stream) {
  const eventLoopId = event.loop_id || event.data?.debate_id || event.data?.loop_id;
  if (eventLoopId && eventLoopId !== debateId) continue;

  console.log(event.type, event.data);

  if (event.type === 'debate_end') {
    break;
  }
}
```

Events follow the server WebSocket envelope; see `docs/WEBSOCKET_EVENTS.md` for details.

### WebSocket Options

```typescript
const stream = new DebateStream('http://localhost:8765', 'debate-123', {
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
  // Debate types
  Debate,
  DebateStatus,
  ConsensusResult,
  GraphDebate,
  GraphDebateBranch,
  MatrixDebate,
  MatrixConclusion,
  // Verification types
  VerifyClaimResponse,
  VerificationStatus,
  // Agent types
  AgentProfile,
  GauntletReceipt,
  // Event types
  DebateEvent,
  // Selection plugin types
  SelectionPluginsResponse,
  ScoreAgentsRequest,
  ScoreAgentsResponse,
  SelectTeamRequest,
  SelectTeamResponse,
  ScorerInfo,
  TeamSelectorInfo,
  RoleAssignerInfo,
} from '@aragora/sdk';
```

## Framework Integration

### React Hook

Create a custom hook for debates:

```typescript
// hooks/useDebate.ts
import { useState, useCallback, useEffect, useRef } from 'react';
import { AragoraClient, DebateStream, Debate, DebateEvent } from '@aragora/sdk';

const client = new AragoraClient({
  baseUrl: process.env.NEXT_PUBLIC_ARAGORA_URL || 'http://localhost:8080',
});

export function useDebate(debateId?: string) {
  const [debate, setDebate] = useState<Debate | null>(null);
  const [events, setEvents] = useState<DebateEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const streamRef = useRef<DebateStream | null>(null);

  // Fetch debate details
  const fetchDebate = useCallback(async (id: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await client.debates.get(id);
      setDebate(data);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to fetch debate'));
    } finally {
      setLoading(false);
    }
  }, []);

  // Create a new debate
  const createDebate = useCallback(async (task: string, agents?: string[]) => {
    setLoading(true);
    setError(null);
    try {
      const result = await client.debates.run({ task, agents });
      setDebate(result);
      return result;
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to create debate'));
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  // Subscribe to real-time events
  const subscribe = useCallback((id: string) => {
    if (streamRef.current) {
      streamRef.current.disconnect();
    }

    const stream = new DebateStream(
      process.env.NEXT_PUBLIC_ARAGORA_URL || 'http://localhost:8080',
      id
    );

    stream
      .on('agent_message', (event) => {
        setEvents((prev) => [...prev, event]);
      })
      .on('consensus', (event) => {
        setEvents((prev) => [...prev, event]);
        fetchDebate(id); // Refresh full debate on consensus
      })
      .on('debate_end', () => {
        fetchDebate(id);
      })
      .onError((err) => {
        setError(new Error(err.message));
      });

    stream.connect();
    streamRef.current = stream;
  }, [fetchDebate]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      streamRef.current?.disconnect();
    };
  }, []);

  // Auto-fetch if debateId provided
  useEffect(() => {
    if (debateId) {
      fetchDebate(debateId);
      subscribe(debateId);
    }
  }, [debateId, fetchDebate, subscribe]);

  return {
    debate,
    events,
    loading,
    error,
    createDebate,
    subscribe,
    refetch: debateId ? () => fetchDebate(debateId) : undefined,
  };
}
```

Usage in a component:

```tsx
function DebateViewer({ debateId }: { debateId: string }) {
  const { debate, events, loading, error } = useDebate(debateId);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  if (!debate) return null;

  return (
    <div>
      <h1>{debate.task}</h1>
      <div>Status: {debate.status}</div>
      {events.map((event, i) => (
        <div key={i}>
          {event.type}: {JSON.stringify(event.data)}
        </div>
      ))}
      {debate.consensus && (
        <div>Consensus: {debate.consensus.conclusion}</div>
      )}
    </div>
  );
}
```

### Vue Composable

```typescript
// composables/useDebate.ts
import { ref, onMounted, onUnmounted, watch } from 'vue';
import { AragoraClient, DebateStream, Debate, DebateEvent } from '@aragora/sdk';

const client = new AragoraClient({
  baseUrl: import.meta.env.VITE_ARAGORA_URL || 'http://localhost:8080',
});

export function useDebate(debateId?: string) {
  const debate = ref<Debate | null>(null);
  const events = ref<DebateEvent[]>([]);
  const loading = ref(false);
  const error = ref<Error | null>(null);
  let stream: DebateStream | null = null;

  async function fetchDebate(id: string) {
    loading.value = true;
    error.value = null;
    try {
      debate.value = await client.debates.get(id);
    } catch (err) {
      error.value = err instanceof Error ? err : new Error('Failed to fetch');
    } finally {
      loading.value = false;
    }
  }

  async function createDebate(task: string, agents?: string[]) {
    loading.value = true;
    error.value = null;
    try {
      debate.value = await client.debates.run({ task, agents });
      return debate.value;
    } catch (err) {
      error.value = err instanceof Error ? err : new Error('Failed to create');
      throw err;
    } finally {
      loading.value = false;
    }
  }

  function subscribe(id: string) {
    if (stream) stream.disconnect();

    stream = new DebateStream(
      import.meta.env.VITE_ARAGORA_URL || 'http://localhost:8080',
      id
    );

    stream
      .on('agent_message', (event) => {
        events.value = [...events.value, event];
      })
      .on('consensus', (event) => {
        events.value = [...events.value, event];
        fetchDebate(id);
      })
      .on('debate_end', () => fetchDebate(id))
      .onError((err) => {
        error.value = new Error(err.message);
      });

    stream.connect();
  }

  onMounted(() => {
    if (debateId) {
      fetchDebate(debateId);
      subscribe(debateId);
    }
  });

  onUnmounted(() => {
    stream?.disconnect();
  });

  return {
    debate,
    events,
    loading,
    error,
    createDebate,
    subscribe,
    refetch: () => debateId && fetchDebate(debateId),
  };
}
```

## Advanced Patterns

### Retry with Exponential Backoff

```typescript
import { AragoraClient, AragoraError } from '@aragora/sdk';

async function withRetry<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  baseDelay: number = 1000
): Promise<T> {
  let lastError: Error;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      // Don't retry client errors (4xx)
      if (error instanceof AragoraError && error.status < 500) {
        throw error;
      }

      if (attempt < maxRetries - 1) {
        const delay = baseDelay * Math.pow(2, attempt);
        await new Promise((r) => setTimeout(r, delay));
      }
    }
  }

  throw lastError!;
}

// Usage
const debate = await withRetry(() =>
  client.debates.run({ task: 'Design a system' })
);
```

### Request Caching

```typescript
import { AragoraClient, Debate } from '@aragora/sdk';

class CachedClient {
  private client: AragoraClient;
  private cache = new Map<string, { data: unknown; expires: number }>();
  private ttl: number;

  constructor(baseUrl: string, ttlMs: number = 60000) {
    this.client = new AragoraClient({ baseUrl });
    this.ttl = ttlMs;
  }

  async getDebate(id: string): Promise<Debate> {
    const cacheKey = `debate:${id}`;
    const cached = this.cache.get(cacheKey);

    if (cached && cached.expires > Date.now()) {
      return cached.data as Debate;
    }

    const debate = await this.client.debates.get(id);
    this.cache.set(cacheKey, {
      data: debate,
      expires: Date.now() + this.ttl,
    });

    return debate;
  }

  invalidate(pattern?: string) {
    if (!pattern) {
      this.cache.clear();
      return;
    }
    for (const key of this.cache.keys()) {
      if (key.includes(pattern)) {
        this.cache.delete(key);
      }
    }
  }
}
```

### Batch Operations with Concurrency Control

```typescript
async function processBatch<T, R>(
  items: T[],
  processor: (item: T) => Promise<R>,
  concurrency: number = 3
): Promise<R[]> {
  const results: R[] = [];
  const queue = [...items];

  async function worker() {
    while (queue.length > 0) {
      const item = queue.shift()!;
      const result = await processor(item);
      results.push(result);
    }
  }

  await Promise.all(
    Array(Math.min(concurrency, items.length))
      .fill(null)
      .map(worker)
  );

  return results;
}

// Run multiple debates with controlled concurrency
const tasks = [
  'Design auth system',
  'Choose database',
  'API architecture',
];

const debates = await processBatch(
  tasks,
  (task) => client.debates.run({ task }),
  2 // Max 2 concurrent debates
);
```

### Event Aggregation

```typescript
import { DebateStream, DebateEvent } from '@aragora/sdk';

class DebateEventAggregator {
  private events: DebateEvent[] = [];
  private stream: DebateStream;

  constructor(baseUrl: string, debateId: string) {
    this.stream = new DebateStream(baseUrl, debateId);
  }

  async collect(): Promise<DebateEvent[]> {
    return new Promise((resolve, reject) => {
      this.stream
        .on('agent_message', (e) => this.events.push(e))
        .on('critique', (e) => this.events.push(e))
        .on('consensus', (e) => this.events.push(e))
        .on('debate_end', () => {
          this.stream.disconnect();
          resolve(this.events);
        })
        .onError((err) => {
          this.stream.disconnect();
          reject(new Error(err.message));
        });

      this.stream.connect();
    });
  }

  getMessagesByAgent(): Map<string, DebateEvent[]> {
    const byAgent = new Map<string, DebateEvent[]>();
    for (const event of this.events) {
      if (event.type === 'agent_message') {
        const agentId = (event.data as { agent_id?: string }).agent_id || 'unknown';
        const existing = byAgent.get(agentId) || [];
        byAgent.set(agentId, [...existing, event]);
      }
    }
    return byAgent;
  }
}
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
