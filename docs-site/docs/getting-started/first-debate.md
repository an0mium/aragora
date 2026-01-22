---
title: Your First Debate
description: Step-by-step tutorial for running your first multi-agent debate
sidebar_position: 4
---

# Your First Debate

This tutorial walks you through creating and running a multi-agent debate step by step.

## What You'll Build

By the end of this tutorial, you'll have:
- Created a debate with 3 AI agents
- Watched them deliberate through multiple rounds
- Obtained a consensus with confidence scores

## Prerequisites

- Aragora server running (see [Installation](/docs/getting-started/installation))
- At least one AI provider API key configured

## Step 1: Create a Debate

Let's start by creating a simple debate on a technical topic.

```bash
curl -X POST http://localhost:8080/api/debates \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "What is the best approach to implement caching in a microservices architecture?",
    "context": "We have 10 microservices with varying read/write patterns. Some services are read-heavy, others write-heavy.",
    "agents": ["claude", "gpt4", "gemini"],
    "rounds": 3,
    "protocol": {
      "phases": ["opening", "critique", "revision", "vote"],
      "consensus_threshold": 0.75
    }
  }'
```

Response:

```json
{
  "id": "debate_x7k9p2",
  "status": "running",
  "topic": "What is the best approach to implement caching...",
  "agents": ["claude", "gpt4", "gemini"],
  "created_at": "2024-01-15T10:30:00Z"
}
```

## Step 2: Monitor Progress

Watch the debate unfold in real-time using WebSocket:

```javascript
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'subscribe',
    debate_id: 'debate_x7k9p2'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`[${data.type}] ${data.agent}: ${data.content?.substring(0, 100)}...`);
};
```

Or poll the status:

```bash
curl http://localhost:8080/api/debates/debate_x7k9p2
```

## Step 3: Understanding the Phases

Each debate round goes through these phases:

### Opening Phase

Each agent presents their initial position:

```json
{
  "round": 1,
  "phase": "opening",
  "messages": [
    {
      "agent": "claude",
      "content": "For microservices caching, I recommend a hybrid approach combining..."
    },
    {
      "agent": "gpt4",
      "content": "Given the varying read/write patterns, I suggest implementing..."
    },
    {
      "agent": "gemini",
      "content": "Considering the 10 services, we should consider distributed caching..."
    }
  ]
}
```

### Critique Phase

Agents review and critique each other's proposals:

```json
{
  "round": 1,
  "phase": "critique",
  "messages": [
    {
      "agent": "claude",
      "critiques": {
        "gpt4": "The proposal lacks consideration for cache invalidation across services...",
        "gemini": "While distributed caching is valid, the complexity might be excessive..."
      }
    }
  ]
}
```

### Revision Phase

Agents incorporate feedback and refine their positions:

```json
{
  "round": 2,
  "phase": "revision",
  "messages": [
    {
      "agent": "claude",
      "content": "After considering the critiques, I've revised my approach to include..."
    }
  ]
}
```

### Vote Phase

Agents vote on the final consensus:

```json
{
  "round": 3,
  "phase": "vote",
  "votes": {
    "claude": { "position": "agree", "confidence": 0.85 },
    "gpt4": { "position": "agree", "confidence": 0.90 },
    "gemini": { "position": "agree_with_modifications", "confidence": 0.78 }
  }
}
```

## Step 4: Get the Consensus

Once the debate completes, retrieve the consensus:

```bash
curl http://localhost:8080/api/debates/debate_x7k9p2/consensus
```

Response:

```json
{
  "debate_id": "debate_x7k9p2",
  "status": "consensus_reached",
  "consensus": {
    "summary": "The agents reached consensus on implementing a hybrid caching strategy with: (1) Local in-memory cache (Redis/Memcached) for read-heavy services, (2) Distributed cache with write-through for write-heavy services, (3) Event-driven cache invalidation via message queue, and (4) Circuit breakers for cache failures.",
    "confidence": 0.84,
    "key_points": [
      "Use Redis for shared cache layer",
      "Implement cache-aside pattern for read-heavy services",
      "Use write-through for write-heavy services",
      "Add TTL-based and event-driven invalidation"
    ],
    "disagreements": [
      "Optimal TTL values (claude: 5min, gpt4: 10min)"
    ]
  },
  "voting_summary": {
    "agree": 2,
    "agree_with_modifications": 1,
    "disagree": 0
  }
}
```

## Step 5: Review the Full Transcript

Get the complete debate history:

```bash
curl http://localhost:8080/api/debates/debate_x7k9p2/transcript
```

## Tips for Better Debates

### Provide Good Context

```json
{
  "topic": "How should we handle rate limiting?",
  "context": "We have a REST API serving 1M requests/day. Current stack: Node.js, PostgreSQL, Redis available.",
  "constraints": ["Must not impact p99 latency by more than 10ms", "Must support tenant isolation"]
}
```

### Choose Complementary Agents

Different models have different strengths:

| Agent | Strengths |
|-------|-----------|
| Claude | Careful reasoning, safety considerations |
| GPT-4 | Broad knowledge, code generation |
| Gemini | Speed, multimodal understanding |
| Grok | Real-time information |

### Adjust Round Count

- **Simple topics**: 2 rounds
- **Complex topics**: 3-4 rounds
- **Contentious topics**: 4-5 rounds

## Next Steps

- [Core Concepts: Debates](/docs/core-concepts/debates) - Deep dive into debate mechanics
- [Custom Agents Guide](/docs/guides/custom-agents) - Create your own agents
- [API Reference](/docs/api-reference) - Full API documentation
