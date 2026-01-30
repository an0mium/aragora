# Aragora SDK Cookbook

Practical, runnable examples for the Aragora SDK. Each example is self-contained
and demonstrates specific SDK features.

## Prerequisites

### Installation

```bash
# Install the Aragora SDK
pip install aragora-sdk

# Or install from source
pip install -e ./sdk
```

### Environment Setup

Create a `.env` file or export these environment variables:

```bash
# Required: At least one API key
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENAI_API_KEY="your-openai-key"

# Optional: Additional providers
export MISTRAL_API_KEY="your-mistral-key"
export GEMINI_API_KEY="your-gemini-key"

# Required for server examples
export ARAGORA_API_URL="http://localhost:8080"
export ARAGORA_API_TOKEN="your-api-token"
```

### Dry Run Mode

All examples support `--dry-run` flag for testing without making actual API calls:

```bash
python 01_simple_debate.py --dry-run
```

---

## Examples Index

| # | Example | Description | Key Concepts |
|---|---------|-------------|--------------|
| 01 | [Simple Debate](01_simple_debate.py) | Basic 3-agent debate with minimal setup | `ArenaClient`, `DebateConfig`, basic flow |
| 02 | [Streaming Debate](02_streaming_debate.py) | Real-time WebSocket streaming with reconnection | `StreamingClient`, event handlers, reconnect |
| 03 | [Knowledge Integration](03_knowledge_integration.py) | Query and inject knowledge into debates | `KnowledgeMound`, evidence collection |
| 04 | [Custom Agents](04_custom_agents.py) | Define custom agent personas and behavior | `AgentConfig`, personas, weighting |
| 05 | [Consensus Tracking](05_consensus_tracking.py) | Monitor consensus evolution and voting | `ConsensusTracker`, convergence detection |
| 06 | [Batch Debates](06_batch_debates.py) | Run multiple debates in parallel | `asyncio.gather`, aggregation, error isolation |
| 07 | [Auth Patterns](07_auth_patterns.py) | JWT, API key, and OAuth authentication | `AuthClient`, token refresh, OAuth flows |
| 08 | [Error Handling](08_error_handling.py) | Retry patterns, fallbacks, circuit breakers | `RetryPolicy`, `CircuitBreaker`, fallback agents |
| 09 | [TypeScript Quickstart](09_typescript_quickstart.ts) | TypeScript SDK equivalent of #1 | TypeScript types, async/await |
| 10 | [Advanced Workflow](10_advanced_workflow.py) | End-to-end workflow combining features | Workflow orchestration, combined patterns |

---

## Quick Start

```bash
# Run the simplest example
python 01_simple_debate.py --dry-run

# Run with actual API calls (requires API keys)
python 01_simple_debate.py

# Stream debate events in real-time
python 02_streaming_debate.py --topic "Should AI systems be open source?"
```

---

## Example Categories

### Getting Started
- **01_simple_debate.py** - Start here! Minimal setup to run your first debate.

### Real-Time Features
- **02_streaming_debate.py** - WebSocket streaming for live debate updates.
- **05_consensus_tracking.py** - Track how consensus evolves over rounds.

### Knowledge & Context
- **03_knowledge_integration.py** - Integrate your organization's knowledge.

### Customization
- **04_custom_agents.py** - Create specialized agent personas.

### Production Patterns
- **06_batch_debates.py** - Scale with parallel debate execution.
- **07_auth_patterns.py** - Secure authentication strategies.
- **08_error_handling.py** - Resilient error handling.

### Multi-Language
- **09_typescript_quickstart.ts** - TypeScript SDK usage.

### Advanced
- **10_advanced_workflow.py** - Complete end-to-end workflow.

---

## Running Tests

Each example can be tested without API calls:

```bash
# Test all examples
for f in *.py; do python "$f" --dry-run; done

# Test TypeScript example
npx ts-node 09_typescript_quickstart.ts --dry-run
```

---

## Additional Resources

- [SDK Reference Documentation](../docs/SDK_REFERENCE.md)
- [API Reference](../../docs/API_REFERENCE.md)
- [Enterprise Features](../../docs/ENTERPRISE_FEATURES.md)
- [Status & Roadmap](../../docs/STATUS.md)
