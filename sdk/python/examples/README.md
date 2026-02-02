# Aragora Python SDK Examples

Comprehensive examples demonstrating the Aragora Python SDK.

## Setup

```bash
# Install the SDK
pip install aragora

# Set environment variables
export ARAGORA_API_KEY="your-api-key"
export ARAGORA_API_URL="https://api.aragora.ai"
```

## Quick Start Examples

| File | Description |
|------|-------------|
| `basic_debate.py` | Create a debate, poll for results, print consensus |
| `streaming_debate.py` | Real-time WebSocket event streaming |
| `workflow_automation.py` | Create and execute workflow templates |
| `agent_selection.py` | List agents, compare, view leaderboard |
| `receipts_demo.py` | Decision receipt retrieval and verification |

## Foundational Examples

| File | Description |
|------|-------------|
| `error_handling.py` | All exception types and retry strategies |
| `async_vs_sync.py` | Comparison of sync vs async patterns |
| `configuration.py` | Environment variables, logging, client options |

## Common Use Cases

| File | Description |
|------|-------------|
| `knowledge_integration.py` | Knowledge-powered debates with Gauntlet validation |
| `workflow_chains.py` | Sequential, conditional, and parallel workflows |
| `explainability_deep_dive.py` | Decision factors, counterfactuals, narratives |
| `multi_agent_comparison.py` | Compare agent combinations and ELO tracking |
| `batch_operations.py` | Bulk debate submission with error handling |

## Advanced Patterns

| File | Description |
|------|-------------|
| `realtime_collaboration.py` | WebSocket streaming, voting, suggestions |
| `control_plane_example.py` | Enterprise agent registry, scheduling, policies |

## Running Examples

```bash
# Run any example (dry run - shows patterns without API calls)
python examples/basic_debate.py

# Run with actual API calls
RUN_EXAMPLES=true python examples/knowledge_integration.py

# Run with a custom API URL
ARAGORA_API_URL=http://localhost:8080 python examples/basic_debate.py
```

## Example Categories

### Error Handling (`error_handling.py`)

Learn to handle all SDK exception types:
- `AuthenticationError` - Invalid API key (401)
- `AuthorizationError` - Access denied (403)
- `NotFoundError` - Resource not found (404)
- `RateLimitError` - Rate limit exceeded (429)
- `ValidationError` - Invalid request (400)
- `ServerError` - Server errors (5xx)
- `TimeoutError` - Request timeout
- `ConnectionError` - Network issues

### Async vs Sync (`async_vs_sync.py`)

Understand when to use each pattern:
- **Sync**: Scripts, CLI tools, simple operations
- **Async**: Web apps, concurrent calls, streaming

### Knowledge Integration (`knowledge_integration.py`)

Full knowledge workflow:
1. Create facts in knowledge base
2. Query with semantic search
3. Run knowledge-powered debate
4. Validate with Gauntlet
5. Extract new learnings

### Explainability (`explainability_deep_dive.py`)

Decision transparency features:
- **Factors**: What influenced the decision
- **Counterfactuals**: What would change it
- **Narratives**: Plain language explanations
- **Provenance**: Reasoning chain tracing

### Control Plane (`control_plane_example.py`)

Enterprise orchestration:
- Agent registry and health monitoring
- Task scheduling (cron-based)
- Policy management (rate limits, filters)
- Resource quotas and alerts
- Audit trail export

## Best Practices

1. **Use environment variables** for API keys
2. **Prefer async client** for web applications
3. **Implement retry logic** for rate limits
4. **Use context managers** for proper cleanup
5. **Handle partial failures** in batch operations

## More Resources

- [SDK Documentation](https://docs.aragora.ai/sdk/python)
- [API Reference](https://docs.aragora.ai/api)
- [Tutorials](https://docs.aragora.ai/tutorials)
