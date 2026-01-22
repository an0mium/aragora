# Aragora SDK Examples

This directory contains example code demonstrating various features of the Aragora SDK.

## Setup

```bash
# Install dependencies
npm install @aragora/sdk

# Set environment variables
export ARAGORA_API_KEY="your-api-key"
export ARAGORA_API_URL="https://api.aragora.ai"  # Optional, defaults to production
```

## Examples

### Basic Debate
Create and run a simple debate between AI agents.

```bash
npx ts-node examples/basic-debate.ts
```

### Streaming Debate
Real-time debate streaming using WebSockets.

```bash
npx ts-node examples/streaming-debate.ts
```

### Workflow Automation
Create automated workflows that chain debates with conditional logic.

```bash
npx ts-node examples/workflow-automation.ts
```

### Agent Selection
List agents, get recommendations, and select the best agents for a task.

```bash
npx ts-node examples/agent-selection.ts
```

## Quick Start

```typescript
import { createClient } from '@aragora/sdk';

// Create client
const client = createClient({
  apiKey: process.env.ARAGORA_API_KEY,
});

// Create a debate
const debate = await client.debates.create({
  task: 'What is the best approach for error handling in TypeScript?',
  agents: ['claude', 'gpt-4'],
  protocol: { rounds: 2 },
});

// Get results
const result = await client.debates.get(debate.debate_id);
console.log(result.consensus?.final_answer);
```

## API Reference

See the full [API documentation](https://docs.aragora.ai/guides/sdk) for detailed information.

## Support

- [Documentation](https://docs.aragora.ai)
- [GitHub Issues](https://github.com/aragora/aragora/issues)
- [Discord Community](https://discord.gg/aragora)
